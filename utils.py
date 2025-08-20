# utils.py
import backoff
import os
import asyncio
from typing import Any, Dict, List
from retrying import retry

# We use the llm_backends adapter for all LLM calls
from llm_backends import call_chat

# Backoff-wrapped sync "completion" and "chat" functions that emulate OpenAI return shapes
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def completions_with_backoff(**kwargs) -> Dict[str, Any]:
    """
    Emulate OpenAI Completion.create returning {'choices':[{'text': ...}]}
    Accepts: model, prompt, max_tokens, temperature, top_p, stop, suffix...
    """
    model = kwargs.get("model")
    prompt = kwargs.get("prompt", "")
    max_tokens = kwargs.get("max_tokens", 512)
    temperature = kwargs.get("temperature", 0.0)
    text = call_chat(model=model,
                     messages=[{"role": "user", "content": prompt}],
                     backend=os.getenv("LLM_BACKEND", "hf"),
                     max_tokens=max_tokens,
                     temperature=temperature,
                     load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
                     offload_folder=os.getenv("LLM_OFFLOAD_FOLDER", None))
    return {"choices": [{"text": text}]}

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def chat_completions_with_backoff(**kwargs) -> Dict[str, Any]:
    """
    Emulate OpenAI ChatCompletion.create returning:
    {'choices':[{'message':{'content': ...}, 'finish_reason': 'stop'}]}
    """
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    max_tokens = kwargs.get("max_tokens", 512)
    temperature = kwargs.get("temperature", 0.0)
    text = call_chat(model=model,
                     messages=messages,
                     backend=os.getenv("LLM_BACKEND", "hf"),
                     max_tokens=max_tokens,
                     temperature=temperature,
                     load_in_4bit=(os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")),
                     offload_folder=os.getenv("LLM_OFFLOAD_FOLDER", None))
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

# Async wrappers: run the sync backoff functions in threadpool so asyncio usage remains supported
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


class OpenAIModel:
    """
    Compatibility wrapper for the original repo's OpenAIModel.
    Methods: generate(prompt) -> (text, finish_reason) for chat-style,
             prompt_generate(prompt) -> text for completion-style,
             batch_generate(...) etc.
    """

    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens, base_url=None) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 512
        self.stop_words = stop_words
        self.backend = os.getenv("LLM_BACKEND", "hf")
        self.load_in_4bit = os.getenv("LLM_LOAD_IN_4BIT", "0") in ("1", "true", "True")
        self.offload_folder = os.getenv("LLM_OFFLOAD_FOLDER", None)
        self.base_url = base_url
        if base_url:
            self.backend = "openai"

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string, temperature = 0.0):
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
    def prompt_generate(self, input_string, temperature = 0.0):
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
    def generate(self, input_string, temperature = 0.0):
        """
        Keep decision behavior similar to original repo:
         - For chat-like models, return (text, finish_reason)
         - For completion-like, return text (string)
        The original code sometimes expects tuple unpacking and sometimes indexing [0], so both behaviors are supported:
        - If returned value is tuple/list, callers can use [0] or unpack (text, finish_reason).
        - If returned value is a string, callers expecting tuple should adapt, but original repo uses both patterns carefully.
        """
        # treat everything as chat-style for robustness: always return (text, finish_reason)
        try:
            return self.chat_generate(input_string, temperature)
        except Exception:
            text = self.prompt_generate(input_string, temperature)
            return text, None

    def batch_chat_generate(self, messages_list, temperature = 0.0):
        # messages_list: list of strings (each will be wrapped with system prompt)
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": message}])
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        # predictions: list of dicts like {'choices':[{'message':{'content':...}}]}
        return [x['choices'][0]['message']['content'].strip() for x in predictions]

    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        # each item is {'choices':[{'text':...}]}
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0):
        return self.batch_chat_generate(messages_list, temperature)

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
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
