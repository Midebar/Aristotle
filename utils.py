# utils.py  (patched / robust)
import os
import asyncio
import backoff  # for exponential backoff
import openai
from typing import Any, List, Tuple, Optional
from retrying import retry

# --- Resolve OpenAI exception classes robustly ---
# Some versions of the openai package expose errors under openai.error,
# some expose them at top-level, and in other contexts the package may be
# shadowed. We try several ways and fall back to Exception.
OPENAI_EXC_TYPES = None
try:
    # common shape: openai.error.RateLimitError, openai.error.APIConnectionError
    OPENAI_EXC_TYPES = (openai.error.RateLimitError, openai.error.APIConnectionError)
except Exception:
    # fallback: check attributes directly on openai module
    _tuple = []
    for _name in ("RateLimitError", "APIConnectionError", "OpenAIAPIError"):
        _ex = getattr(openai, _name, None)
        if _ex:
            _tuple.append(_ex)
    if _tuple:
        OPENAI_EXC_TYPES = tuple(_tuple)
    else:
        # Last resort: use Exception so backoff still functions (less precise)
        OPENAI_EXC_TYPES = (Exception,)


# Backoff-wrapped wrappers around openai client calls (if available)
@backoff.on_exception(backoff.expo, OPENAI_EXC_TYPES)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, OPENAI_EXC_TYPES)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# Async batches for OpenAI (same behaviour as original repo)
async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[Any]:
    """Dispatches requests to OpenAI API asynchronously (ChatCompletion.acreate)."""
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requests(
    messages_list: list[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[Any]:
    """Dispatch prompt-style requests to OpenAI API asynchronously (Completion.acreate)."""
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


# Try to import HFBackend (used when LOCAL_MODEL_PATH is set). Import failure is handled.
_HFBackendAvailable = False
try:
    from llm_backends import HFBackend
    _HFBackendAvailable = True
except Exception:
    _HFBackendAvailable = False


class OpenAIModel:
    """
    Backwards-compatible wrapper:
      - Uses OpenAI API (openai package) when LOCAL_MODEL_PATH not provided.
      - Uses local HFBackend (llm_backends.HFBackend) when LOCAL_MODEL_PATH env var or local_model_path arg is provided.
    """

    def __init__(
        self,
        API_KEY: Optional[str],
        model_name: str,
        stop_words: str,
        max_new_tokens: int,
        base_url: Optional[str] = None,
        local_model_path: Optional[str] = None,
        quantize_4bit: bool = True
    ) -> None:
        # Normalize stop words to list
        if isinstance(stop_words, str):
            self.stop_words = [stop_words]
        else:
            self.stop_words = stop_words or []

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Decide whether to use local HF backend
        env_local = os.environ.get("LOCAL_MODEL_PATH")
        requested_local_path = local_model_path or env_local

        if requested_local_path:
            # Use HF backend
            if not _HFBackendAvailable:
                raise RuntimeError("HFBackend not importable. Ensure llm_backends.HFBackend is available.")
            self.mode = "local"
            self.local_model_path = requested_local_path
            # instantiate HFBackend (should handle quantization/device)
            self.hf_backend = HFBackend(local_model_path=self.local_model_path, hf_model_id=None, quantize_4bit=quantize_4bit)
            # use friendly name for file outputs
            self.model_name = model_name or "local-hf"
        else:
            # OpenAI mode
            self.mode = "openai"
            openai.api_key = API_KEY
            if base_url:
                openai.api_base = base_url
            self.hf_backend = None

    # Local backend sync/async helpers
    def _local_generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> str:
        max_nt = max_new_tokens or self.max_new_tokens or 256
        return self.hf_backend.generate(prompt, max_new_tokens=max_nt, temperature=temperature, top_p=top_p, do_sample=do_sample)

    async def _local_generate_async(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> str:
        return await asyncio.to_thread(self._local_generate, prompt, max_new_tokens, temperature, top_p, do_sample)

    # OpenAI compatibility methods
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string: str, temperature: float = 0.0) -> Tuple[str, str]:
        if self.mode == "openai":
            response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                    {"role": "user", "content": input_string}
                ],
                temperature = temperature,
                top_p = 1,
                stop = self.stop_words
            )
            generated_text = response['choices'][0]['message']['content'].strip()
            finish_reason = response['choices'][0].get('finish_reason', '')
            return generated_text, finish_reason
        else:
            generated_text = self._local_generate(input_string, max_new_tokens=self.max_new_tokens, temperature=temperature, top_p=1.0, do_sample=(temperature>0))
            return generated_text, "stop"

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def prompt_generate(self, input_string: str, temperature: float = 0.0) -> str:
        if self.mode == "openai":
            response = completions_with_backoff(
                model = self.model_name,
                prompt = input_string,
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
                stop = self.stop_words
            )
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
        else:
            return self._local_generate(input_string, max_new_tokens=self.max_new_tokens, temperature=temperature, top_p=1.0, do_sample=(temperature>0))

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def generate(self, input_string: str, temperature: float = 0.0):
        if self.mode == "openai":
            chat_models = ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo-0613', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-turbo']
            if self.model_name in chat_models:
                return self.chat_generate(input_string, temperature)
            else:
                return self.prompt_generate(input_string, temperature)
        else:
            return self.chat_generate(input_string, temperature)

    # Batch helpers
    def batch_chat_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        if self.mode == "openai":
            open_ai_messages_list = []
            system_prompt = "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
            for message in messages_list:
                open_ai_messages_list.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": message}])
            predictions = asyncio.run(dispatch_openai_chat_requests(open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words))
            return [x['choices'][0]['message']['content'].strip() for x in predictions]
        else:
            async def _run_all():
                tasks = [self._local_generate_async(m, max_new_tokens=self.max_new_tokens, temperature=temperature, top_p=1.0, do_sample=(temperature>0)) for m in messages_list]
                return await asyncio.gather(*tasks)
            return asyncio.run(_run_all())

    def batch_prompt_generate(self, prompt_list: List[str], temperature: float = 0.0) -> List[str]:
        if self.mode == "openai":
            predictions = asyncio.run(dispatch_openai_prompt_requests(prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words))
            return [x['choices'][0]['text'].strip() for x in predictions]
        else:
            async def _run_all():
                tasks = [self._local_generate_async(p, max_new_tokens=self.max_new_tokens, temperature=temperature, top_p=1.0, do_sample=(temperature>0)) for p in prompt_list]
                return await asyncio.gather(*tasks)
            return asyncio.run(_run_all())

    def batch_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        if self.mode == "openai":
            chat_models = ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo-0613', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-turbo']
            if self.model_name in chat_models:
                return self.batch_chat_generate(messages_list, temperature)
            else:
                return self.batch_prompt_generate(messages_list, temperature)
        else:
            return self.batch_chat_generate(messages_list, temperature)

    def generate_insertion(self, input_string: str, suffix: str, temperature: float = 0.0):
        if self.mode == "openai":
            response = completions_with_backoff(
                model = self.model_name,
                prompt = input_string,
                suffix= suffix,
                temperature = temperature,
                max_tokens = self.max_new_tokens,
                top_p = 1.0,
                frequency_penalty = 0.0,
                presence_penalty = 0.0
            )
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
        else:
            combined = input_string + (suffix or "")
            return self._local_generate(combined, max_new_tokens=self.max_new_tokens, temperature=temperature, top_p=1.0, do_sample=(temperature>0))
