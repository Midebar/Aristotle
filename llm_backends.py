import torch
import os
import logging

# --- 1. Hugging Face Backend ---
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def build_max_memory_dict(reserve_per_gpu_gb: int = 2, cpu_reserve_gb: int = 8):
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gib = int(props.total_memory // (1024 ** 3))
            usable = max(1, total_gib - reserve_per_gpu_gb)
            max_memory[i] = f"{usable}GiB"
    max_memory["cpu"] = "16GiB" # Safe default
    return max_memory

class HFBackend:
    def __init__(self, local_model_path: str = None, hf_model_id: str = None, quantize_4bit: bool = True):
        self.model_source = local_model_path or hf_model_id
        if not self.model_source: raise ValueError("Model path required")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source, trust_remote_code=True)
        
        # Basic quantization config
        quant_config = None
        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            device_map="auto",
            quantization_config=quant_config,
            dtype=torch.float16,
            trust_remote_code=True,
            max_memory=build_max_memory_dict()
        )
        self.device = self.model.device

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temperature > 0: gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- 2. ExLlamaV2 Backend ---
try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
    EXLLAMA_AVAILABLE = True
except ImportError:
    EXLLAMA_AVAILABLE = False

class ExLlamaBackend:
    def __init__(self, local_model_path: str):
        if not EXLLAMA_AVAILABLE: raise ImportError("ExLlamaV2 not installed")
        
        config = ExLlamaV2Config()
        config.model_dir = local_model_path
        config.prepare()
        
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        if temperature == 0: settings.top_k = 1

        return self.generator.generate_simple(prompt, settings, num_tokens=max_new_tokens)

# --- 3. Llama.cpp Backend ---
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

class LlamaCPPBackend:
    def __init__(self, local_model_path: str):
        """
        local_model_path: Must point to a specific .gguf FILE, not just a directory.
        """
        if not LLAMACPP_AVAILABLE: raise ImportError("llama-cpp-python not installed. Run 'pip install llama-cpp-python'")
        
        # n_gpu_layers=-1 means offload ALL layers to GPU
        self.llm = Llama(
            model_path=local_model_path, 
            n_ctx=0,
            n_gpu_layers=-1, 
            verbose=False
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            stop=[],
            echo=True, # Return prompt + completion to match others
            temperature=temperature
        )
        return output['choices'][0]['text']

# --- 4. vLLM Backend  ---
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class VLLMBackend:
    def __init__(self, local_model_path: str, gpu_memory_utilization: float = 0.8, quantization: str = "fp8"):
        """
        local_model_path: HuggingFace repo ID (e.g. 'Sahabat-AI/...') or local path.
        """
        if not VLLM_AVAILABLE: raise ImportError("vllm not installed (Linux only).")

        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        logging.getLogger("vllm").setLevel(logging.ERROR)

        print(f"   [vLLM] Loading: {local_model_path}")
        print(f"   [vLLM] VRAM: {gpu_memory_utilization*100}%, Quantization: {quantization}")
        
        print(f"Initializing vLLM with model: {local_model_path}")
        # enforce_eager=True can help with stability on some setups, but default is usually faster
        self.llm = LLM(
            model=local_model_path,
            dtype="auto",
            quantization=quantization, # Pass "fp8", "awq", "gptq", or None
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            disable_log_stats=True
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
        # vLLM expects a list of prompts
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_new_tokens
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # vLLM returns ONLY the new tokens. 
        # To match others behavior, prepend prompt.
        return prompt + generated_text