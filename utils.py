# utils.py
import os
import backoff
import asyncio
from typing import Any, Dict, List, Optional
from retrying import retry
from collections.abc import Mapping

# Prefer local llm_backends adapter
try:
    from llm_backends import HFBackend
except Exception:
    HFBackend = None

# Optional OpenAI import (only used if backend == 'openai' is requested)
try:
    import openai
except Exception:
    openai = None

# simple requests fallback for shim-based endpoints
try:
    import requests
except Exception:
    requests = None


# -------------------------
# call_chat: routing to backends
# -------------------------
_hf_backend_cache: Dict[str, "HFBackend"] = {}


def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of messages (OpenAI style: {'role':..,'content':..})
    into a single prompt string for local HF models.
    We preserve role markers to make outputs more predictable.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role.upper()}] {content}")
    return "\n\n".join(parts)


def call_chat(
    model: str,
    messages: List[Dict[str, str]],
    backend: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    load_in_4bit: bool = False,
    offload_folder: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Core router that returns a single text string (the generated text).
    - model: model identifier (or 'local-hf' / HF model id)
    - messages: list of dicts with keys 'role' and 'content'
    - backend: 'hf' (default), 'openai', or 'shim' (if base_url provided)
    - returns: plain string output
    """
    backend = backend or os.getenv("LLM_BACKEND", "hf")

    # If base_url is passed explicitly (or via env), prefer contacting shim/openai-compat server
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("SHIM_BASE_URL") or None

    # 1) HF backend (local model)
    if backend.lower() in ("hf", "local", "local-hf"):
        if HFBackend is None:
            raise RuntimeError("HFBackend adapter not available (llm_backends.HFBackend import failed).")

        # Prefer LOCAL_MODEL_PATH environment variable if present
        local_path = os.getenv("LOCAL_MODEL_PATH", None)

        # Cache backend by model string (model may be HF id or local path)
        cache_key = model or local_path or "hf-default"
        hf_backend = _hf_backend_cache.get(cache_key)
        if hf_backend is None:
            # Try to instantiate HFBackend using local model path first, then model id
            try:
                if local_path:
                    hf_backend = HFBackend(local_model_path=local_path, hf_model_id=model, quantize_4bit=load_in_4bit)
                else:
                    hf_backend = HFBackend(local_model_path=None, hf_model_id=model, quantize_4bit=load_in_4bit)
            except Exception as e:
                # If instantiation fails, give helpful message
                raise RuntimeError(f"Failed to initialize HFBackend for model '{model}' (local_path={local_path}): {e}")
            _hf_backend_cache[cache_key] = hf_backend

        prompt = format_messages_to_prompt(messages)
        # HFBackend.generate expects max_new_tokens (map max_tokens -> max_new_tokens)
        return hf_backend.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)

    # 2) OpenAI package direct call (if available and user explicitly asked for openai backend)
    if backend.lower() == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed or importable but backend set to 'openai'.")

        # If a base_url/shim is provided, configure openai to use it (useful for local shim)
        if base_url:
            openai.api_base = base_url
        # set key if provided via env
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

        # Build the ChatCompletion call
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Normalize the usual response shape
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI response parsing failed: {e} -- response: {resp}")

    # 3) Shim / HTTP local server (OpenAI-compatible shim)
    if base_url:
        if requests is None:
            raise RuntimeError("requests is not available but base_url shim requested.")
        url = base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Shim POST to {url} failed: {e}")

    # If none matched, error
    raise RuntimeError(f"Unsupported backend '{backend}'. Set LLM_BACKEND to 'hf' or 'openai' or provide SHIM_BASE_URL.")


# -------------------------
# Existing Backoff-wrapped helpers (unchanged interface)
# -------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def completions_with_backoff(**kwargs) -> Dict[str, Any]:
    """
    Emulate OpenAI Completion.create returning {'choices':[{'text': ...}]}
    Accepts: model, prompt, max_tokens, temperature, top_p, stop, suffix...
    """
    model = kwargs.get("model")
    prompt = kwargs.get("prompt", "")
    max_tokens = int(kwargs.get("max_tokens", 512))
    temperature = float(kwargs.get("temperature", 0.0))

    # convert single prompt string into messages list (user role)
    messages = [{"role": "user", "content": prompt}]

    text = call_chat(model=model,
                     messages=messages,
                     backend=os.getenv("LLM_BACKEND", "hf"),
                     max_tokens=max_tokens,
                     temperature=temperature,
                     load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
                     offload_folder=os.getenv("LLM_OFFLOAD_FOLDER", None),
                     base_url=os.getenv("SHIM_BASE_URL", None))
    return {"choices": [{"text": text}]}


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def chat_completions_with_backoff(**kwargs) -> Dict[str, Any]:
    """
    Emulate OpenAI ChatCompletion.create returning:
    {'choices':[{'message':{'content': ...}, 'finish_reason': 'stop'}]}
    """
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    max_tokens = int(kwargs.get("max_tokens", 512))
    temperature = float(kwargs.get("temperature", 0.0))

    text = call_chat(model=model,
                     messages=messages,
                     backend=os.getenv("LLM_BACKEND", "hf"),
                     max_tokens=max_tokens,
                     temperature=temperature,
                     load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
                     offload_folder=os.getenv("LLM_OFFLOAD_FOLDER", None),
                     base_url=os.getenv("SHIM_BASE_URL", None))
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


# -------------------------
# Async wrappers: run the sync backoff functions in threadpool so asyncio usage remains supported
# -------------------------
async def _async_chat_call_wrapper(messages, model, temperature, max_tokens, top_p, stop_words):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: chat_completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_words)
    )


async def _async_prompt_call_wrapper(prompt, model, temperature, max_tokens, top_p, stop_words):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: completions_with_backoff(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_words)
    )


async def dispatch_openai_chat_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[Dict[str, Any]]:
    tasks = [_async_chat_call_wrapper(x, model, temperature, max_tokens, top_p, stop_words) for x in messages_list]
    return await asyncio.gather(*tasks)


async def dispatch_openai_prompt_requests(
    prompt_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str]
) -> List[Dict[str, Any]]:
    tasks = [_async_prompt_call_wrapper(x, model, temperature, max_tokens, top_p, stop_words) for x in prompt_list]
    return await asyncio.gather(*tasks)


# -------------------------
# OpenAIModel compatibility wrapper (keeps your existing code shape)
# -------------------------
class OpenAIModel:
    """
    Compatibility wrapper for the original repo's OpenAIModel.
    Methods:
      - generate(prompt) -> (text, finish_reason) for chat-style,
      - prompt_generate(prompt) -> text for completion-style,
      - batch_generate(...) etc.
    """

    def __init__(self, API_KEY: str, model_name: str, stop_words: Any, max_new_tokens: int, base_url: Optional[str] = None) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 512
        self.stop_words = stop_words
        self.backend = os.getenv("LLM_BACKEND", "hf")
        self.load_in_4bit = os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")
        self.offload_folder = os.getenv("LLM_OFFLOAD_FOLDER", None)
        self.base_url = base_url or os.getenv("SHIM_BASE_URL", None)
        if self.base_url:
            # if base_url provided, prefer using the shim (treated as openai-compatible)
            self.backend = "openai"

        # configure openai if needed
        if self.backend == "openai" and openai is not None:
            if os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            if self.base_url:
                openai.api_base = self.base_url

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string: str, temperature: float = 0.0):
        messages = [
            {"role": "system", "content": "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
            {"role": "user", "content": input_string}
        ]
        resp = chat_completions_with_backoff(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            stop=self.stop_words
        )
        generated_text = resp["choices"][0]["message"]["content"].strip()
        finish_reason = resp["choices"][0].get("finish_reason", None)
        return generated_text, finish_reason

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def prompt_generate(self, input_string: str, temperature: float = 0.0):
        resp = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words
        )
        generated_text = resp["choices"][0]["text"].strip()
        return generated_text

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def generate(self, input_string: str, temperature: float = 0.0):
        """
        Keep decision behavior similar to original repo:
         - For chat-like models, return (text, finish_reason)
         - For completion-like, return text (string)
        We prefer to call chat_generate for robustness.
        """
        try:
            return self.chat_generate(input_string, temperature)
        except Exception:
            text = self.prompt_generate(input_string, temperature)
            return text, None

    def batch_chat_generate(self, messages_list, temperature: float = 0.0):
        system_prompt = "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": message}])

        predictions = asyncio.run(
            dispatch_openai_chat_requests(open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words)
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]

    def batch_prompt_generate(self, prompt_list, temperature: float = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words)
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature: float = 0.0):
        return self.batch_chat_generate(messages_list, temperature)

    def generate_insertion(self, input_string: str, suffix: str, temperature: float = 0.0):
        resp = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        generated_text = resp['choices'][0]['text'].strip()
        return generated_text
