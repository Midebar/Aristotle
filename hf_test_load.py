# hf_test_load.py
import os
import sys
from pathlib import Path
import argparse

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

ROOT = Path(__file__).parent.resolve()

def load_env_file(env_path: Path):
    if env_path.exists():
        if DOTENV_AVAILABLE:
            load_dotenv(dotenv_path=str(env_path))
        else:
            # Minimalist loader
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# ---- Parse CLI args ----
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
args = parser.parse_args()
env_path = ROOT / args.env
load_env_file(env_path)

# ---- Model config ----
MODEL_ID = os.environ.get(
    "LLM_MODEL",
    "Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct"
)

print("Testing HF model load:", MODEL_ID)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("transformers imported")
except Exception as e:
    print("Install transformers/bitsandbytes/safetensors first:", e)
    sys.exit(1)

# ---- Load tokenizer ----
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded:", MODEL_ID)
    
except Exception as e:
    print("Tokenizer load failed:", e)
    sys.exit(1)

# ---- Load model ----
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,  # or replace with BitsAndBytesConfig for new API
        trust_remote_code=True
    )
    print("Model loaded (4-bit) — OK")
    device = next(model.parameters()).device
    print("Model device:", device)

    # ---- Test generation ----
    inputs = tokenizer("Hello from Aristotle", return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    print("Generated text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print("Model load/generation failed:", e)
    sys.exit(2)
