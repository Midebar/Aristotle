import time
import torch
import os
import sys
import gc
from dataclasses import dataclass

# Ensure we can import from the local file
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from llm_backends import HFBackend, ExLlamaBackend, LlamaCPPBackend, VLLMBackend
except ImportError:
    print("CRITICAL ERROR: Could not import 'llm_backends.py'.")
    print("Please ensure 'llm_backends.py' is in the same directory as this notebook.")

# ---Configuration Class (Replaces argparse)---
@dataclass
class BenchmarkConfig:
    backend: str          # Options: 'hf', 'exllama', 'llamacpp', 'vllm'
    model_path: str
    max_new_tokens: int = 300
    quantize_4bit: bool = False
    quantization: str = None  # For vLLM: None, 'fp8', 'gptq', 'awq'

PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Anda adalah asisten AI yang cerdas dan membantu.<|eot_id|><|start_header_id|>user<|end_header_id|>

Jelaskan secara rinci perbedaan antara ExLlamaV2, Llama.cpp, dan vLLM dalam hal kecepatan inferensi.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def calculate_tokens(backend, backend_name, text):
    """
    Universal token counter that handles the specific tokenizer object 
    of each backend type.
    """
    try:
        if backend_name == 'hf':
            return backend.tokenizer.encode(text)
        
        elif backend_name == 'exllama':
            # ExLlama tokenizer returns a tensor [[ids]]
            return backend.tokenizer.encode(text)[0].tolist()
        
        elif backend_name == 'llamacpp':
            # LlamaCPP expects bytes
            return backend.llm.tokenize(text.encode("utf-8"))
        
        elif backend_name == 'vllm':
            # vLLM exposes the HF tokenizer via get_tokenizer()
            return backend.llm.get_tokenizer().encode(text)
            
    except Exception as e:
        print(f"Warning: Token counting failed ({e}). Returning approximation.")
        return [0] * (len(text) // 4) # Crude fallback

# ---Benchmark Runner---
def run_benchmark(args: BenchmarkConfig):
    print(f"\n=======================================================")
    print(f" Starting Benchmark: {args.backend.upper()}")
    print(f" Model: {args.model_path}")
    print(f"=======================================================")

    backend = None
    try:
        if args.backend == 'hf':
            backend = HFBackend(local_model_path=args.model_path, quantize_4bit=args.quantize_4bit)
        elif args.backend == 'exllama':
            backend = ExLlamaBackend(local_model_path=args.model_path)
        elif args.backend == 'llamacpp':
            backend = LlamaCPPBackend(local_model_path=args.model_path)
        elif args.backend == 'vllm':
            backend = VLLMBackend(local_model_path=args.model_path, quantization=args.quantization)
    except Exception as e:
        print(f"Failed to initialize backend: {e}")
        return

    print("Warming up...")
    try:
        # Short generation to prime the engine
        backend.generate(PROMPT, max_new_tokens=10, temperature=0.0)
    except Exception as e:
        print(f"Warmup failed: {e}")

    print(f"Benchmarking generation of {args.max_new_tokens} tokens...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    output_text = backend.generate(PROMPT, max_new_tokens=args.max_new_tokens, temperature=0.0)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Calculate tokens
    full_tokens = calculate_tokens(backend, args.backend, output_text)
    prompt_tokens = calculate_tokens(backend, args.backend, PROMPT)
    
    # Calculate delta (generated tokens only)
    if len(full_tokens) > len(prompt_tokens):
        gen_tokens = len(full_tokens) - len(prompt_tokens)
    else:
        gen_tokens = len(full_tokens)

    tps = gen_tokens / total_time if total_time > 0 else 0.0

    print("\nRESULTS")
    print(f"   Total Time:      {total_time:.2f} s")
    print(f"   Tokens Gen:      {gen_tokens}")
    print(f"   Speed (TPS):     {tps:.2f} tokens/sec")
    print(f"   Output:          {output_text}")

    # Cleanup
    del backend
    cleanup_vram()