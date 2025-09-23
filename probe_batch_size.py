# Probe batch size for remote exec
import os, time, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Configure ===
MODEL_PATH = "/workspace/LLM_MODELS/llama3-8b-cpt-sahabatai-v1-instruct"  # <--- change it based on model path
USE_BNB_4BIT = os.environ.get("LLM_LOAD_IN_4BIT", "0") == "1"  # or set True/False manually
MAX_NEW_TOKENS_PROBE = 128   # <--------- MANUALLLY CHANGE
PROBE_BATCHES = [1,2,4,8,16,32,64]  # tries

print("DEVICE:", "cuda" if torch.cuda.is_available() else "cpu")
print("Trying model path:", MODEL_PATH, "USE_BNB_4BIT=", USE_BNB_4BIT)

# set tokenizer parallelism false to avoid warning/locking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Tokenizer loaded. Special tokens:", tokenizer.pad_token, tokenizer.eos_token)

# model loader helper (choose quantized if bitsandbytes available)
def load_model(use_bnb):
    kwargs = dict(trust_remote_code=True)
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    try:
        if use_bnb:
            # try quantized load (requires bitsandbytes installed)
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False)
            print("Attempting 4-bit quantized load via BitsAndBytesConfig...")
            return AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, **kwargs)
    except Exception as e:
        print("4-bit load failed:", e)
    # fallback
    print("Loading non-quantized model (fp16/auto device_map)...")
    return AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)

# load once for probing (may be slower) - catch errors
try:
    model = load_model(USE_BNB_4BIT)
    model.eval()
    print("Model loaded. Device map:", getattr(model, "device_map", None))
except Exception as e:
    print("Model load failed:", e)
    traceback.print_exc()
    raise

# create tokenized small input (single short prompt)
sample_prompt = "Translate to Indonesian: If it rains, the ground will be wet."
tok = tokenizer(sample_prompt, return_tensors="pt")
input_ids = tok["input_ids"]

def try_batch(batch_size, max_new_tokens=MAX_NEW_TOKENS_PROBE):
    try:
        # repeat batch on CPU then move to model device
        batch_ids = input_ids.repeat(batch_size, 1).to(next(model.parameters()).device)
        torch.cuda.empty_cache()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(batch_ids, max_new_tokens=max_new_tokens)
        t = time.time()-t0
        print(f"OK batch={batch_size}  time={t:.2f}s")
        return True
    except RuntimeError as e:
        print(f"OOM/RuntimeError at batch={batch_size}: {e}")
        traceback.print_exc(limit=1)
        return False
    except Exception as e:
        print(f"Other error at batch={batch_size}: {e}")
        traceback.print_exc(limit=1)
        return False

# probe increasing sizes until failure
largest_ok = 0
for b in PROBE_BATCHES:
    torch.cuda.empty_cache()
    ok = try_batch(b)
    if ok:
        largest_ok = b
    else:
        break

print("Probe finished. Suggested batch_size (small max_new_tokens):", largest_ok)
