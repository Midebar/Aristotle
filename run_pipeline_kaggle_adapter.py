# run_pipeline_kaggle_adapter.py
import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

# paths - override with env vars if needed
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/kaggle/input/sahabatai_model")
DATA_JSON_PATH = os.environ.get("DATA_JSON_PATH", "/kaggle/input/aristotle-data/data/ProofWriter/dev.json")
PROMPTS_INPUT_PATH = os.environ.get("PROMPTS_PATH", "/kaggle/input/aristotle-data/prompts")
WORK_ROOT = Path.cwd()
WORK_DATA_ROOT = WORK_ROOT / "data"
WORK_PROMPTS_ROOT = WORK_ROOT / "prompts"
RESULTS_ROOT = WORK_ROOT / "results"
OUTPUT_DIR = Path(os.environ.get("OUT_DIR", str(RESULTS_ROOT)))

# script names (original files) — assumed present in same working dir
SCRIPT_TRANSLATE = "translate_decompose.py"
SCRIPT_NEGATE = "negate.py"
SCRIPT_SEARCH = "search_resolve.py"
SCRIPT_EVAL = "evaluate.py"

# OpenAI shim server port
SHIM_HOST = "127.0.0.1"
SHIM_PORT = int(os.environ.get("OPENAI_SHIM_PORT", "11434"))
SHIM_BASE_URL = f"http://{SHIM_HOST}:{SHIM_PORT}"

# which dataset/split to run
DATASET_NAME = os.environ.get("DATASET_NAME", "ProofWriter")
SPLIT = os.environ.get("SPLIT", "dev")
EXAMPLE_INDEX = os.environ.get("EXAMPLE_INDEX")  # if set, we will create single-example file

# Ensure working dirs
WORK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
WORK_PROMPTS_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Adapter configuration:")
print(" LOCAL_MODEL_PATH:", LOCAL_MODEL_PATH)
print(" DATA_JSON_PATH:", DATA_JSON_PATH)
print(" PROMPTS_INPUT_PATH:", PROMPTS_INPUT_PATH)
print(" WORK_DATA_ROOT:", WORK_DATA_ROOT)
print(" WORK_PROMPTS_ROOT:", WORK_PROMPTS_ROOT)
print(" RESULTS_ROOT:", RESULTS_ROOT)
print(" SHIM_BASE_URL:", SHIM_BASE_URL)

# 1) copy prompts into ./prompts/<dataset>
if PROMPTS_INPUT_PATH and Path(PROMPTS_INPUT_PATH).exists():
    src_prompts = Path(PROMPTS_INPUT_PATH)
    dst_prompts_for_ds = WORK_PROMPTS_ROOT / DATASET_NAME
    if dst_prompts_for_ds.exists():
        shutil.rmtree(dst_prompts_for_ds)
    shutil.copytree(src_prompts / DATASET_NAME, dst_prompts_for_ds)
    print("Copied prompts to:", dst_prompts_for_ds)
else:
    print("Prompts input path not found - continuing without copying prompts. Prompt-dependent steps may fail.")

# 2) prepare data: either full JSON (copy) or single-example file
data_source = Path(DATA_JSON_PATH)
dataset_folder = WORK_DATA_ROOT / DATASET_NAME
dataset_folder.mkdir(parents=True, exist_ok=True)

if EXAMPLE_INDEX is None:
    # copy the entire dev.json (or other split) into expected location: ./data/<dataset_name>/<split>.json
    dst_file = dataset_folder / f"{SPLIT}.json"
    shutil.copyfile(data_source, dst_file)
    print(f"Copied full dataset to {dst_file}")
else:
    # create single-example split file containing exactly one example
    with open(data_source, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = int(EXAMPLE_INDEX)
    example = data[idx]
    dst_file = dataset_folder / f"{SPLIT}.json"
    with open(dst_file, "w", encoding="utf-8") as outf:
        json.dump([example], outf, indent=2, ensure_ascii=False)
    print(f"Wrote single-example {idx} to {dst_file}")

# 3) start HF backend and openai shim
print("Starting HF backend and OpenAI shim server...")
# import backend and server locally
from llm_backends import HFBackend
import openai_compat_server as shim

backend = HFBackend(local_model_path=LOCAL_MODEL_PATH, hf_model_id=None, quantize_4bit=True)
shim_thread = shim.run_server(bind_host=SHIM_HOST, port=SHIM_PORT, backend=backend)
print("Shim started at", SHIM_BASE_URL)

# 4) run the original scripts with arguments that match original CLI signature
# translate_decompose.py args (matches original parse_args)
translate_cmd = [
    sys.executable, SCRIPT_TRANSLATE,
    "--data_path", str(WORK_DATA_ROOT),
    "--dataset_name", DATASET_NAME,
    "--split", SPLIT,
    "--save_path", str(RESULTS_ROOT),
    "--api_key", "",  # empty, not used; base_url is used by OpenAIModel in utils.py
    "--model_name", "local-hf",  # model name passed through; shim ignores actual model name
    "--stop_words", "------",
    "--mode", "",
    "--max_new_tokens", "512",
    "--batch_num", "1",
    "--base_url", SHIM_BASE_URL
]
print("Running translate_decompose (this may take a while)...")
subprocess.check_call(translate_cmd)
print("translate_decompose finished.")

# 5) run negate.py (original script expects dataset_name and model)
negate_cmd = [
    sys.executable, SCRIPT_NEGATE,
    "--dataset_name", DATASET_NAME,
    "--model", "local-hf"
]
print("Running negate.py ...")
subprocess.check_call(negate_cmd)
print("negate finished.")

# 6) run search_resolve.py
search_cmd = [
    sys.executable, SCRIPT_SEARCH,
    "--data_path", str(WORK_DATA_ROOT),
    "--dataset_name", DATASET_NAME,
    "--split", SPLIT,
    "--save_path", str(RESULTS_ROOT),
    "--api_key", "",
    "--model_name", "local-hf",
    "--stop_words", "------",
    "--mode", "",
    "--max_new_tokens", "256",
    "--base_url", SHIM_BASE_URL,
    "--batch_num", "1",
    "--search_round", "10",
    "--negation", "True"
]
print("Running search_resolve (negation True) ...")
subprocess.check_call(search_cmd)
print("search_resolve (negation True) finished.")

# Also run search_resolve for negation False (original evaluate compares both)
search_cmd_neg_false = [arg if arg!="--negation" else "--negation" for arg in search_cmd]  # copy
# replace negation value at last arg with False - simpler to build again:
search_cmd_false = search_cmd.copy()
# replace "--negation", "True" -> "--negation", "False"
for i, a in enumerate(search_cmd_false):
    if a == "--negation" and i+1 < len(search_cmd_false):
        search_cmd_false[i+1] = "False"
        break
print("Running search_resolve (negation False) ...")
subprocess.check_call(search_cmd_false)
print("search_resolve (negation False) finished.")

# 7) run evaluate.py (original evaluate expects dataset_name and model_name)
eval_cmd = [
    sys.executable, SCRIPT_EVAL,
    "--dataset_name", DATASET_NAME,
    "--model_name", "local-hf"
]
print("Running evaluate.py ...")
subprocess.check_call(eval_cmd)
print("evaluate finished.")

# 8) done - stop shim (Flask thread is daemon so process exit will stop it)
print("Pipeline completed. Results are in:", RESULTS_ROOT)
