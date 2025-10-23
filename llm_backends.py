# llm_backends.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import warnings
from collections.abc import Mapping
import os
import traceback
from typing import Optional
import psutil

def build_max_memory_dict(reserve_per_gpu_gb: int = 2, cpu_reserve_gb: int = 8):
    """
    Build a max_memory mapping for device_map='auto'.
    reserve_per_gpu_gb: how many GiB to leave free per GPU to avoid collisions.
    cpu_reserve_gb: how many GiB to reserve on host for OS/other processes.
    """
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gib = int(props.total_memory // (1024 ** 3))
            usable = max(1, total_gib - reserve_per_gpu_gb)
            max_memory[i] = f"{usable}GiB"   # integer key for GPU 0,1,...
    cpu_avail_gb = int(psutil.virtual_memory().available // (1024 ** 3))
    cpu_for_model = max(4, cpu_avail_gb - cpu_reserve_gb)
    max_memory["cpu"] = f"{cpu_for_model}GiB"
    return max_memory


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

        tokenizer = None
        last_exc = None

        local_files_only = bool(local_model_path)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
        hub_kwargs = {"use_auth_token": hf_token} if hf_token else {}

        attempts = [
            {"use_fast": False, "local_files_only": local_files_only},
            {"use_fast": True,  "local_files_only": local_files_only},
            {"local_files_only": local_files_only},
            {"use_fast": False, "local_files_only": False},
            {"use_fast": True,  "local_files_only": False},
            {"local_files_only": False},
        ]

        errors = []
        for attempt_kwargs in attempts:
            try:
                load_kwargs = dict(attempt_kwargs)
                load_kwargs.update(hub_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(self.model_source, **load_kwargs)
                # sanity check: tokenizer should be callable
                if tokenizer is not None and callable(getattr(tokenizer, "__call__", None)):
                    break
                else:
                    raise RuntimeError(f"Loaded object is not a callable tokenizer (type={type(tokenizer)})")
            except Exception as e:
                last_exc = e
                errors.append((attempt_kwargs, traceback.format_exc()))
                warnings.warn(f"AutoTokenizer.from_pretrained failed with args {attempt_kwargs}: {e}")

        if tokenizer is None:
            extra = (
                "\nTried AutoTokenizer.from_pretrained with these attempts:\n"
                + "\n".join([f"  {a}: {err.splitlines()[-1] if err else ''}" for a, err in errors])
            )
            raise RuntimeError(f"Failed to load tokenizer for {self.model_source}. Last error: {last_exc}\n{extra}")

        # ensure pad token exists
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "eos_token_id", None) is not None:
                try:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                except Exception:
                    tokenizer.pad_token_id = 0

        self.tokenizer = tokenizer

        # ---- Model load (attempt quantized) ----
        model = None
        # Prepare a conservative max_memory mapping to avoid accidental OOMs on GPU
        max_memory = build_max_memory_dict(reserve_per_gpu_gb=2, cpu_reserve_gb=8)

        if quantize_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    # nf4 preferred when available
                    bnb_4bit_quant_type="nf4" if hasattr(BitsAndBytesConfig, "bnb_4bit_quant_type") else "fp4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype="float16",
                    # NEW: enable fp32 CPU offload so large modules can remain on CPU instead of failing the load
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                # Pass max_memory so device_map='auto' will respect GPU/CPU memory limits
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_source,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,  # reduce peak memory during load
                    **hub_kwargs
                )
            except Exception as e:
                warnings.warn(f"4-bit quantized load failed: {e}. Falling back to non-quantized load.")
                model = None

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                device_map="auto",
                trust_remote_code=True,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                **hub_kwargs
            )

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self.model = model

        # ensure pad/eos token ids are present in model config (keeps erroring when no pad token)
        if getattr(self.model.config, "pad_token_id", None) is None and getattr(self.model.config, "eos_token_id", None) is not None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False, max_time: Optional[float] = None) -> str:
        """
        Generate text from the local HF model.
        Returns: single string (decoded).
        """
        if self.tokenizer is None or not callable(getattr(self.tokenizer, "__call__", None)):
            raise RuntimeError("Tokenizer is not ready or not callable.")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        try:
            if hasattr(inputs, "to") and callable(getattr(inputs, "to")):
                inputs = inputs.to(self.device)
        except Exception:
            pass

        # Convert Mapping-like objects (BatchEncoding) into plain dict of tensors
        if isinstance(inputs, Mapping):
            inputs_dict = {k: v for k, v in inputs.items()}
        elif hasattr(inputs, "items"):
            # duck-typed fallback
            inputs_dict = {k: v for k, v in inputs.items()}
        else:
            raise RuntimeError(f"Tokenizer returned unsupported type for inputs: {type(inputs)}. Expected Mapping or BatchEncoding.")

        for k, v in list(inputs_dict.items()):
            if isinstance(v, torch.Tensor):
                inputs_dict[k] = v.to(self.device)

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

        if max_time is not None:
            gen_kwargs["max_time"] = float(max_time)

        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Do generation in inference mode (no grads)
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        try:
            out_ids = outputs[0] if isinstance(outputs, (list, tuple)) else outputs[0]
            decoded = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        finally:
            # free generated tensors (if on GPU) and clear small caches if needed
            try:
                del outputs, out_ids
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return decoded
