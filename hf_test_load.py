# hf_test_load.py
import os
from pathlib import Path
from llm_backends import HFBackend

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")  # e.g. /kaggle/input/sahabatai_model
HF_ID = os.environ.get("HF_MODEL_ID", "Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct")

print("HF test: local_path=", MODEL_PATH, "hf_id=", HF_ID)
backend = HFBackend(local_model_path=MODEL_PATH, hf_model_id=HF_ID, quantize_4bit=True)

prompt = "Translate to formal logic: If it rains, the ground will be wet."
print("Prompt:", prompt)
out = backend.generate(prompt, max_new_tokens=128, temperature=0.0)
print("Output:", out[:1000])
