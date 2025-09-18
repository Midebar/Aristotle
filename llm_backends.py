# llm_backends.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import os
import warnings
from collections.abc import Mapping

class HFBackend:
    def __init__(self, local_model_path: str = None, hf_model_id: str = None, quantize_4bit: bool = True):
        """
        local_model_path: local path to the model (preferred)
        hf_model_id: remote model id (unused when local_model_path provided)
        quantize_4bit: attempt 4-bit (requires bitsandbytes + compatible hardware)
        """
        self.model_source = local_model_path or hf_model_id
        if not self.model_source:
            raise ValueError("Either local_model_path or hf_model_id must be provided.")

        # ---- Tokenizer load (robust) ----
        tokenizer = None
        last_exc = None
        try:
            # Preferred: load tokenizer from the local folder, avoid use_fast=True
            tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        except Exception as e:
            last_exc = e
            warnings.warn(f"AutoTokenizer load failed: {e}. Tried default AutoTokenizer.")

        if tokenizer is None:
            # give a clear error if nothing worked
            raise RuntimeError(f"Failed to load tokenizer for {self.model_source}. Last error: {last_exc}")

        # Safety check: ensure tokenizer is sensible
        if isinstance(tokenizer, bool) or not callable(getattr(tokenizer, "__call__", None)):
            raise RuntimeError(f"Loaded tokenizer is not callable (type={type(tokenizer)}). Aborting.")

        # ensure pad token exists (use eos_token if needed)
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "eos_token_id", None) is not None:
                try:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                except Exception:
                    tokenizer.pad_token_id = 0

        self.tokenizer = tokenizer

        # ---- Model load (attempt quantized if requested) ----
        model = None
        if quantize_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4" if hasattr(BitsAndBytesConfig, "bnb_4bit_quant_type") else "fp4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype="float16",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_source,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=quant_config
                )
            except Exception as e:
                warnings.warn(f"4-bit quantized load failed: {e}. Falling back to non-quantized load.")
                model = None

        if model is None:
            # final fallback: non-quantized load
            model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                device_map="auto",
                trust_remote_code=True
            )

        # record device
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            # model has no parameters (unlikely) -> cpu fallback
            self.device = torch.device("cpu")
        self.model = model

        # ensure pad/eos token ids are present in model config (helpful for generate)
        if getattr(self.model.config, "pad_token_id", None) is None and getattr(self.model.config, "eos_token_id", None) is not None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> str:
        """
        Generate text from the local HF model.
        Returns: single string (decoded).
        """
        # safety: ensure tokenizer callable
        if self.tokenizer is None or not callable(getattr(self.tokenizer, "__call__", None)):
            raise RuntimeError("Tokenizer is not ready or not callable.")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # BatchEncoding (or others) might have .to(device) which moves contained tensors.
        # If so, use it (this is efficient). Otherwise move each tensor manually.
        try:
            if hasattr(inputs, "to") and callable(getattr(inputs, "to")):
                # BatchEncoding.to will move tensors inside it and return BatchEncoding
                inputs = inputs.to(self.device)
        except Exception:
            # ignore; fallback to manual move below
            pass

        # Convert Mapping-like objects (BatchEncoding) into plain dict of tensors
        if isinstance(inputs, Mapping):
            inputs_dict = {k: v for k, v in inputs.items()}
        elif hasattr(inputs, "items"):
            # duck-typed fallback
            inputs_dict = {k: v for k, v in inputs.items()}
        else:
            # Unexpected type, raise helpful error
            raise RuntimeError(f"Tokenizer returned unsupported type for inputs: {type(inputs)}. Expected Mapping or BatchEncoding.")

        # Move tensors to device if they are not already on it
        for k, v in list(inputs_dict.items()):
            if isinstance(v, torch.Tensor):
                inputs_dict[k] = v.to(self.device)

        # Prepare generate kwargs
        gen_kwargs = dict(
            input_ids = inputs_dict.get("input_ids"),
            attention_mask = inputs_dict.get("attention_mask"),
            max_new_tokens = int(max_new_tokens),
            do_sample = bool(do_sample),
            temperature = float(temperature),
            top_p = float(top_p),
            pad_token_id = getattr(self.tokenizer, "pad_token_id", getattr(self.model.config, "pad_token_id", None)),
            eos_token_id = getattr(self.tokenizer, "eos_token_id", getattr(self.model.config, "eos_token_id", None)),
        )

        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Call generate
        outputs = self.model.generate(**gen_kwargs)

        # outputs may be tensor or tuple/list; extract first generated ids
        if isinstance(outputs, torch.Tensor):
            out_ids = outputs[0]
        elif isinstance(outputs, (list, tuple)):
            out_ids = outputs[0]
        else:
            # Unknown return type
            raise RuntimeError(f"model.generate returned unexpected type: {type(outputs)}")

        # Decode
        decoded = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return decoded
