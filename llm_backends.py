# llm_backends.py
from pathlib import Path
import os
import torch

def _ensure_pad_token(tokenizer, model):
    try:
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None or isinstance(model.config.pad_token_id, bool):
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

class HFBackend:
    def __init__(self, local_model_path: str = None, hf_model_id: str = None, quantize_4bit: bool = True):
        self.local_model_path = local_model_path
        self.hf_model_id = hf_model_id
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.quantize_4bit = quantize_4bit
        self._load()

    def _resolve_source(self):
        if self.local_model_path and Path(self.local_model_path).exists():
            return str(Path(self.local_model_path))
        if self.hf_model_id:
            return self.hf_model_id
        raise RuntimeError("No model path or HF id provided for HFBackend")

    def _load(self):
        src = self._resolve_source()
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("Install transformers/bitsandbytes/safetensors first") from e

        # Tokenizer: try remote code then fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(src)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(src)

        # Model: attempt bitsandbytes 4-bit then fallback
        try:
            if self.quantize_4bit:
                try:
                    quant_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        src,
                        device_map="auto",
                        quantization_config=quant_config,
                        trust_remote_code=True,
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        src,
                        device_map="auto",
                        load_in_4bit=True,
                        trust_remote_code=True,
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(src, device_map="auto", trust_remote_code=True)
        except Exception as e:
            try:
                # final fallback to CPU non-quantized
                self.model = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, device_map=None)
                self.model.to(self.device)
            except Exception as e2:
                raise RuntimeError(f"Model load failed for {src}: {e2}") from e2

        _ensure_pad_token(self.tokenizer, self.model)

        # if generate is not callable, bind default HF generate
        try:
            if not callable(getattr(self.model, "generate", None)):
                from transformers.generation import GenerationMixin
                self.model.generate = GenerationMixin.generate.__get__(self.model, type(self.model))
        except Exception:
            pass

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception:
            # fallback pipeline
            from transformers import pipeline
            gen = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
            res = gen(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
            return res[0]["generated_text"]
