import os
import backoff
import asyncio
from typing import Any, Dict, List, Optional
from retrying import retry
import re
from persistent_worker import PersistentWorkerManager

# Try to import local llm_backends adapter
try:
    from llm_backends import HFBackend
except Exception:
    HFBackend = None

def sanitize_filename(name: str) -> str:
    # replace path separators (both normal and alt) with underscore
    name = name.replace(os.path.sep, '_')
    if os.path.altsep:
        name = name.replace(os.path.altsep, '_')
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name:
        name = 'model'
    return name 

def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of messages (OpenAI style: {'role':..,'content':..})
    into a single prompt string for local HF models.
    Preserve role markers to make outputs more predictable.
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
) -> str:
    """
    Core router that returns a single text string (the generated text).
    - model: model identifier (or 'local-hf' / HF model id)
    - messages: list of dicts with keys 'role' and 'content'
    - backend: 'hf' (default)
    - returns: plain string output
    """
    backend = backend or os.getenv("LLM_BACKEND", "hf")

    # HF backend (local model)
    if backend.lower() in ("hf", "local", "local-hf"):
        if HFBackend is None:
            raise RuntimeError("HFBackend adapter not available (llm_backends.HFBackend import failed).")
        local_path = os.getenv("LOCAL_MODEL_PATH", None)
        try:
            if local_path:
                hf_backend = HFBackend(local_model_path=local_path, hf_model_id=model, quantize_4bit=load_in_4bit)
            else:
                hf_backend = HFBackend(local_model_path=None, hf_model_id=model, quantize_4bit=load_in_4bit)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HFBackend for model '{model}' (local_path={local_path}): {e}")

        prompt = format_messages_to_prompt(messages)
        # HFBackend.generate expects max_new_tokens (map max_tokens -> max_new_tokens)
        return hf_backend.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)


# -------------------------
# Backoff-wrapped helpers
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
                     load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),)
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
# Model wrapper class
# -------------------------
class ModelWrapper:
    """
    Compatibility wrapper for the original repo's OpenAIModel.
    Methods:
      - generate(prompt) -> (text, finish_reason) for chat-style,
      - prompt_generate(prompt) -> text for completion-style,
    """

    def __init__(self, model_name: str, stop_words: Any, max_new_tokens: int) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 512
        self.stop_words = stop_words
        self.backend = os.getenv("LLM_BACKEND")
        self.load_in_4bit = os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")
        # persistent worker manager (lazy init)
        self._use_persistent = (os.getenv("LLM_USE_PERSISTENT_WORKER", "0") in ("1", "true", "True")) and self.backend in ("hf", "local", "local-hf")
        self._pw_manager: Optional[PersistentWorkerManager] = None
        self._model_source = os.getenv("LOCAL_MODEL_PATH", None) or self.model_name

    def _ensure_manager(self):
        if not self._use_persistent:
            return
        if self._pw_manager is None or not getattr(self._pw_manager, "proc", None) or not self._pw_manager.proc.is_alive():
            # (re)start manager
            try:
                self._pw_manager = PersistentWorkerManager(self._model_source, quant_4bit=self.load_in_4bit)
            except Exception as e:
                # fallback: disable persistent and raise
                self._pw_manager = None
                raise

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string: str, temperature: float = 0.0):
        messages = [
            {"role": "system", "content": "Anda adalah asisten yang sangat membantu dan diakui sebagai salah satu ilmuwan AI, ahli logika, dan matematikawan terbaik. Sebelum mulai menyelesaikan masalah, pastikan Anda memahami secara cermat dan menyeluruh setiap detail kebutuhan pengguna."},
            {"role": "user", "content": input_string}
        ]
        # If using persistent worker, route via manager
        if self._use_persistent:
            self._ensure_manager()
            # format messages to prompt string
            prompt = format_messages_to_prompt(messages)
            client_wait = float(os.getenv("LLM_CLIENT_WAIT", "60"))  # how long client will wait for a response
            per_call_max_time = float(os.getenv("LLM_WORKER_MAX_TIME", str(self.max_new_tokens)))  # seconds for worker watchdog
            try:
                res = self._pw_manager.generate(prompt, timeout=client_wait, max_new_tokens=self.max_new_tokens, temperature=temperature, max_time=per_call_max_time)
            except Exception as e:
                raise RuntimeError(f"Persistent worker generation failed: {e}")
            if not res.get("ok"):
                return res.get("error", ""), "timeout"
            return res.get("text", ""), "stop"
        else:
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
        if self._use_persistent:
            self._ensure_manager()
            prompt = input_string
            client_wait = float(os.getenv("LLM_CLIENT_WAIT", "60"))
            per_call_max_time = float(os.getenv("LLM_WORKER_MAX_TIME", str(self.max_new_tokens)))
            res = self._pw_manager.generate(prompt, timeout=client_wait, max_new_tokens=self.max_new_tokens, temperature=temperature, max_time=per_call_max_time)
            if not res.get("ok"):
                return res.get("error", "")
            # response might be a list if batch used; for prompt_generate expect string
            text = res.get("text")
            if isinstance(text, list):
                return text[0] if text else ""
            return text
        else:
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
        system_prompt = "Anda adalah asisten yang sangat membantu dan diakui sebagai salah satu ilmuwan AI, ahli logika, dan matematikawan terbaik. Sebelum mulai menyelesaikan masalah, pastikan Anda memahami secara cermat dan menyeluruh setiap detail kebutuhan pengguna."
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
