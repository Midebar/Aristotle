import os
import backoff
import asyncio
from typing import Any, Dict, List, Optional
from retrying import retry
import re

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    name = name.replace(os.path.sep, '_')
    if os.path.altsep:
        name = name.replace(os.path.altsep, '_')
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name if name else 'model'

# --- Backend Loading ---

try:
    from llm_backends import HFBackend, ExLlamaBackend, VLLMBackend, LlamaCPPBackend
except Exception:
    HFBackend = None
    ExLlamaBackend = None

# Global Cache (Singleton)
_LOADED_BACKEND = {
    "type": None,
    "model_name": None,
    "instance": None
}

def get_backend_instance(backend_type: str, model_name: str, load_in_4bit: bool):
    global _LOADED_BACKEND
    
    if (_LOADED_BACKEND["type"] == backend_type and 
        _LOADED_BACKEND["model_name"] == model_name and 
        _LOADED_BACKEND["instance"] is not None):
        return _LOADED_BACKEND["instance"]

    print(f"--- Loading Model: {model_name} ({backend_type}) ---")
    
    instance = None
    local_path = os.getenv("LOCAL_MODEL_PATH", None)

    if backend_type == "exllama":
        if not local_path:
            raise ValueError("LOCAL_MODEL_PATH is required for ExLlama backend.")
        instance = ExLlamaBackend(local_model_path=local_path)
        
    elif backend_type == "hf":
        instance = HFBackend(local_model_path=local_path, hf_model_id=model_name, quantize_4bit=load_in_4bit)

    elif backend_type == "vllm":
        instance = VLLMBackend(local_model_path=local_path, hf_model_id=model_name)
        
    elif backend_type == "llamacpp":
        if not local_path:
            raise ValueError("LOCAL_MODEL_PATH is required for LlamaCPP backend.")
        instance = LlamaCPPBackend(local_model_path=local_path)
        
    else:
        raise ValueError(f"Unsupported backend: {backend_type}")

    _LOADED_BACKEND = {
        "type": backend_type,
        "model_name": model_name,
        "instance": instance
    }
    return instance

def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of messages (style: {'role':..,'content':..})
    into a single prompt string for local HF models.
    Preserve role markers to make outputs more predictable.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role.upper()}] {content}")
    return "\n\n".join(parts)

# --- Core Chat Function ---

def call_chat(
    model: str,
    messages: List[Dict[str, str]],
    backend: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    load_in_4bit: bool = False,
    max_time: int = 120,
) -> str:
    backend = backend or os.getenv("LLM_BACKEND", "hf")
    backend = backend.lower()

    if backend in ("hf", "local", "local-hf", "exllama", "vllm", "llamacpp"):
        
        backend_instance = get_backend_instance(
            backend_type=backend,
            model_name=model,
            load_in_4bit=load_in_4bit
        )

        prompt = format_messages_to_prompt(messages)
        
        return backend_instance.generate(
            prompt, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            max_time=max_time
        )

    raise RuntimeError(f"Unknown backend: {backend}")

# --- Backoff Wrappers ---

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def completions_with_backoff(**kwargs) -> Dict[str, Any]:
    model = kwargs.get("model")
    prompt = kwargs.get("prompt", "")
    max_tokens = int(kwargs.get("max_tokens", 512))
    temperature = float(kwargs.get("temperature", 0.0))

    messages = [{"role": "user", "content": prompt}]

    text = call_chat(
        model=model,
        messages=messages,
        backend=os.getenv("LLM_BACKEND", "hf"),
        max_tokens=max_tokens,
        temperature=temperature,
        load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
        max_time=int(os.getenv("LLM_WORKER_MAX_TIME", "120"))
    )
    return {"choices": [{"text": text}]}

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def chat_completions_with_backoff(**kwargs) -> Dict[str, Any]:
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    max_tokens = int(kwargs.get("max_tokens", 512))
    temperature = float(kwargs.get("temperature", 0.0))

    text = call_chat(
        model=model,
        messages=messages,
        backend=os.getenv("LLM_BACKEND", "hf"),
        max_tokens=max_tokens,
        temperature=temperature,
        load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
        max_time=int(os.getenv("LLM_WORKER_MAX_TIME", "120"))
    )
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

# --- Async Wrappers ---

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

# --- Model Wrapper Class ---

class ModelWrapper:
    def __init__(self, model_name: str, stop_words: Any, max_new_tokens: int) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 512
        self.stop_words = stop_words
        self.backend = os.getenv("LLM_BACKEND")
        self.load_in_4bit = os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string: str, temperature: float = 0.0, task: Optional[str] = None):
        if task == "translation":
            messages = [
                {"role": "user", "content": input_string}
            ]
        else:
            messages = [
                {"role": "system", "content": "Anda adalah asisten yang cerdas dan sangat membantu dan diakui sebagai salah satu ilmuwan AI, ahli logika, dan matematikawan terbaik serta selalu menjawab dalam Bahasa Indonesia yang baik dan benar. Sebelum mulai menyelesaikan masalah, pastikan Anda memahami secara cermat dan menyeluruh setiap detail kebutuhan pengguna."},
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
    def generate(self, input_string: str, temperature: float = 0.0, task: Optional[str] = None):
        try:
            return self.chat_generate(input_string, temperature, task)
        except Exception:
            text = self.prompt_generate(input_string, temperature)
            return text, None

    def batch_chat_generate(self, messages_list, temperature: float = 0.0):
        # The wrapper will auto-inject system prompt in format_messages_to_prompt
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append([{"role": "user", "content": message}])

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