# hf_test_load.py
import os
import traceback
from llm_backends import HFBackend

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")  # e.g. path to local model folder
HF_ID = os.environ.get("HF_MODEL_ID", "Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct")

print("HF test: local_path=", MODEL_PATH, "hf_id=", HF_ID)
try:
    backend = HFBackend(local_model_path=MODEL_PATH, hf_model_id=HF_ID, quantize_4bit=True)
    prompt = "Translate to formal logic: If it rains, the ground will be wet."
    print("Prompt:", prompt)
    out = backend.generate(prompt, max_new_tokens=128)
    print("Output (first 1000 chars):\n", out[:1000])
except Exception as e:
    print("Error during HF test:")
    traceback.print_exc()
    raise
