# llm_backends.py
import os
import json
import requests
from typing import List, Dict, Optional

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---- Ollama HTTP chat wrapper ----
def _call_ollama_chat(model: str, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    resp = r.json()
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp, indent=2)

# ---- OpenAI fallback ----
try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def _call_openai_chat(model: str, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed/configured")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp["choices"][0]["message"]["content"]

# ---- HF Transformers + bitsandbytes backend (lazy init) ----
_HF_STATE = {"model": None, "tokenizer": None, "device": None, "model_id": None}

def _init_hf_model(hf_model_id: str, load_in_4bit: bool = False, device_map: str = "auto", offload_folder: Optional[str] = None):
    if _HF_STATE["model"] is not None and _HF_STATE["model_id"] == hf_model_id:
        print("HF model already loaded:", hf_model_id)
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
    import torch  # type: ignore

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
    )
    _HF_STATE["model"] = model
    _HF_STATE["tokenizer"] = tokenizer
    _HF_STATE["device"] = next(model.parameters()).device
    _HF_STATE["model_id"] = hf_model_id

def _call_hf_chat(hf_model_id: str, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.0,
                  load_in_4bit: bool = False, offload_folder: Optional[str] = None, device_map: str = "auto"):
    if _HF_STATE["model"] is None or _HF_STATE["model_id"] != hf_model_id:
        _init_hf_model(hf_model_id, load_in_4bit=load_in_4bit, device_map=device_map, offload_folder=offload_folder)

    model = _HF_STATE["model"]
    tokenizer = _HF_STATE["tokenizer"]
    device = _HF_STATE["device"]

    # Build a concatenated chat-style prompt from messages
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        prompt_parts.append(f"[{role}]: {m.get('content','')}")
    prompt = "\n".join(prompt_parts) + "\n[assistant]:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    idx = out.find("[assistant]:")
    return out[idx + len("[assistant]:"):].strip() if idx != -1 else out[len(prompt):].strip()

# ---- public wrapper ----
def call_chat(model: str, messages: List[Dict], backend: str = "ollama", max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
    """
    model: HF model id (for backend='hf'), or Ollama model name (backend='ollama'),
           or OpenAI model id (backend='openai').
    backend: 'ollama' | 'hf' | 'openai'
    kwargs: load_in_4bit (bool), offload_folder (str), device_map (str)
    """
    backend = backend.lower()
    if backend == "ollama":
        return _call_ollama_chat(model, messages, max_tokens=max_tokens, temperature=temperature)
    elif backend == "hf":
        return _call_hf_chat(model, messages, max_tokens=max_tokens, temperature=temperature,
                             load_in_4bit=kwargs.get("load_in_4bit", False),
                             offload_folder=kwargs.get("offload_folder", None),
                             device_map=kwargs.get("device_map", "auto"))
    elif backend == "openai":
        return _call_openai_chat(model, messages, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")
