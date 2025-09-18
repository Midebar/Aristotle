#!/usr/bin/env python3
"""
run_pipeline_kaggle_adapter.py

Usage:
  python run_pipeline_kaggle_adapter.py            # runs full pipeline based on .env (or Kaggle env)
  python run_pipeline_kaggle_adapter.py --pilot    # runs pipeline in pilot mode (PILOT_ONLY override)
  python run_pipeline_kaggle_adapter.py --kaggle   # enable Kaggle-adapter behavior (auto-detected too)
  python run_pipeline_kaggle_adapter.py --split-percent 10  # use only 10% of dataset

This script expects a .env file in the same directory or environment variables set.
When running on Kaggle, set environment variables (or let the adapter infer them):
  LOCAL_MODEL_PATH, DATA_JSON_PATH, PROMPTS_INPUT_PATH, WORK_DATA_ROOT,
  WORK_PROMPTS_ROOT, RESULTS_ROOT, SHIM_BASE_URL

It will copy prompts and a sampled dataset into /kaggle/working and run the pipeline there.
"""
import os
import sys
import subprocess
import argparse
import shutil
import json
import random
from pathlib import Path

# dependency: python-dotenv for reading .env files (optional)
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
            # minimalist loader
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

def check_ollama(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/api/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_hf_model_env() -> bool:
    # Lightweight check: attempt to import transformers
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False

def run_cmd(cmd_list, env=None):
    print(">>>", " ".join(map(str, cmd_list)))
    subprocess.run(cmd_list, check=True, env=env)

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def copy_prompts(src_prompts: Path, dst_prompts_root: Path, dataset_name: str):
    """Copy prompts for dataset into working prompts folder."""
    if not src_prompts.exists():
        print(f"Prompts input path {src_prompts} does not exist. Skipping copy.")
        return
    dst_for_dataset = dst_prompts_root / dataset_name
    if dst_for_dataset.exists():
        print("Prompts already copied to", dst_for_dataset)
        return
    print(f"Copying prompts from {src_prompts} -> {dst_for_dataset}")
    try:
        if src_prompts.is_dir():
            shutil.copytree(src_prompts, dst_for_dataset)
        else:
            # single file - create folder and copy
            dst_for_dataset.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_prompts, dst_for_dataset / src_prompts.name)
    except Exception as e:
        print("Error copying prompts:", e)

def prepare_dataset_input(original_json_path: Path, dst_data_dir: Path, dataset_name: str, split_name: str, split_percent: int):
    """
    Copy (and optionally sample) a dataset JSON into dst_data_dir/{dataset_name}/{split}.json
    Supports input path being either a JSON file or a folder containing <split>.json.
    """
    ensure_dir(dst_data_dir)
    dst_dataset_dir = dst_data_dir / dataset_name
    ensure_dir(dst_dataset_dir)

    # Find source file
    if original_json_path.is_file() and original_json_path.suffix == ".json":
        src_file = original_json_path
    else:
        # try to find default split file under original_json_path/<split>.json
        candidate = original_json_path / split_name
        if candidate.is_dir():
            # If folder like data/ProofWriter/dev/dev.json — try deeper
            maybe = candidate / f"{split_name}.json"
            if maybe.exists():
                src_file = maybe
            else:
                # fallback: search for first .json in directory
                jsons = list(candidate.glob("*.json"))
                src_file = jsons[0] if jsons else None
        else:
            maybe = original_json_path / f"{split_name}.json"
            src_file = maybe if maybe.exists() else None

    if src_file is None or not src_file.exists():
        raise FileNotFoundError(f"Could not find source dataset JSON. Checked {original_json_path}")

    print("Source dataset JSON:", src_file)
    # Load and sample if needed
    with src_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        # If dataset is object mapping (e.g., {'data': [...]}) try common keys
        if isinstance(data, dict):
            # try to detect list field
            for key in ("data", "examples", "items", "instances"):
                if key in data and isinstance(data[key], list):
                    data_list = data[key]
                    break
            else:
                # fallback: treat top-level dict as single example list
                data_list = [data]
        else:
            raise ValueError("Unsupported dataset format (expected list or dict containing list).")
    else:
        data_list = data

    n_total = len(data_list)
    if split_percent is None or split_percent >= 100:
        sampled = data_list
    else:
        k = max(1, int(n_total * (split_percent / 100.0)))
        random.seed(42)
        if k >= n_total:
            sampled = data_list
        else:
            sampled = random.sample(data_list, k)

    dst_file = dst_dataset_dir / f"{split_name}.json"
    print(f"Writing {len(sampled)} / {n_total} instances to {dst_file}")
    with dst_file.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    return dst_file

def build_base_kwargs(env):
    # build base args robustly (use --key=value for options whose values might start with '-')
    base_kwargs = [
        "--data_path", env["DATA_PATH"],
        "--dataset_name", env["DATASET_NAME"],
        "--split", env["SPLIT"],
        "--save_path", env["RESULTS_PATH"],
        "--api_key", env.get("OPENAI_API_KEY", ""),
        "--model_name", env["MODEL_NAME"],
        f"--stop_words={env.get('STOP_WORDS','------')}",
        "--mode", env.get("MODE", ""),
        "--max_new_tokens", str(env.get("MAX_NEW_TOKENS", "512")),
        "--batch_num", str(env.get("BATCH_NUM", "1")),
    ]
    if env.get("BASE_URL"):
        base_kwargs.append(f"--base_url={env['BASE_URL']}")
    return base_kwargs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--pilot", action="store_true", help="Run pilot only (overrides PILOT_ONLY)")
    parser.add_argument("--skip-negate", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle adapter behavior (copy inputs to /kaggle/working)")
    parser.add_argument("--split-percent", "-n", type=int, default=100, help="Use only N%% of the dataset (integer 1-100).")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    # Detect kaggle runtime if path exists
    running_on_kaggle = args.kaggle or Path("/kaggle").exists()
    if running_on_kaggle:
        print("Kaggle runtime detected or --kaggle provided.")

    # Gather environment variables (with reasonable defaults)
    BACKEND = os.environ.get("LLM_BACKEND", "ollama")
    MODEL = os.environ.get("LLM_MODEL", "")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    DATASET = os.environ.get("DATASET_NAME", "ProofWriter")
    SPLIT = os.environ.get("SPLIT", "dev")
    # Default working roots (these will be used if Kaggle adapter enabled)
    KAGGLE_WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))
    # Inputs (when running on Kaggle set these envs to the corresponding /kaggle/input/... paths)
    LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "")  # e.g., /kaggle/input/sahabatai/pytorch/...
    DATA_JSON_PATH = os.environ.get("DATA_JSON_PATH", "")  # e.g., /kaggle/input/aristotle-kaggle/data/ProofWriter/dev.json or folder
    PROMPTS_INPUT_PATH = os.environ.get("PROMPTS_INPUT_PATH", "")  # e.g., /kaggle/input/aristotle-kaggle/prompts
    # Working/target paths
    WORK_DATA_ROOT = Path(os.environ.get("WORK_DATA_ROOT", str(KAGGLE_WORK_ROOT / "data")))
    WORK_PROMPTS_ROOT = Path(os.environ.get("WORK_PROMPTS_ROOT", str(KAGGLE_WORK_ROOT / "prompts")))
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(KAGGLE_WORK_ROOT / "results")))

    # General runtime config
    DATA_PATH = os.environ.get("DATA_PATH", "./data")
    PROMPTS_PATH = os.environ.get("PROMPTS_PATH", "./prompts")
    RESULTS_PATH = os.environ.get("RESULTS_PATH", "./results")
    BATCH_NUM = os.environ.get("BATCH_NUM", "1")
    MAX_NEW_TOKENS = os.environ.get("MAX_NEW_TOKENS", "512")
    SEARCH_ROUND = os.environ.get("SEARCH_ROUND", "10")
    USE_MODEL_FOR_NEGATION = os.environ.get("USE_MODEL_FOR_NEGATION", "false").lower() in ("1","true","yes")
    PILOT_ONLY_ENV = os.environ.get("PILOT_ONLY", "true").lower() in ("1","true","yes")
    pilot_mode = args.pilot or PILOT_ONLY_ENV

    print(f"Backend={BACKEND} Model={MODEL} Dataset={DATASET} Pilot={pilot_mode}")

    # If Kaggle adapter: copy prompts and dataset to working dir; set DATA_PATH/PROMPTS_PATH/RESULTS_PATH to working copy
    if running_on_kaggle:
        # If user provided explicit paths via env, respect them; otherwise fallback to expected convention
        src_prompts = Path(PROMPTS_INPUT_PATH) if PROMPTS_INPUT_PATH else (ROOT / "prompts")
        if PROMPTS_INPUT_PATH and not Path(PROMPTS_INPUT_PATH).exists():
            print(f"Warning: PROMPTS_INPUT_PATH={PROMPTS_INPUT_PATH} not found on disk.")
        ensure_dir(WORK_PROMPTS_ROOT)
        copy_prompts(Path(src_prompts), WORK_PROMPTS_ROOT, DATASET)

        # Prepare dataset (copy and optionally sample)
        if DATA_JSON_PATH:
            src_data_path = Path(DATA_JSON_PATH)
        else:
            # Fall back to checking common kaggle input layout
            # try /kaggle/input/<something>/data/<dataset>/<split>.json
            src_data_path = Path("/kaggle/input")  # heuristic fallback; prepare_dataset_input will search inside
        ensure_dir(WORK_DATA_ROOT)
        try:
            prepared_json = prepare_dataset_input(src_data_path, WORK_DATA_ROOT, DATASET, SPLIT, int(args.split_percent))
        except Exception as e:
            print("Error preparing dataset for Kaggle adapter:", e)
            print("Falling back to copying entire data directory if exists.")
            # Try to copy folder if possible
            try:
                if Path(DATA_JSON_PATH).exists():
                    dst_dataset_dir = WORK_DATA_ROOT / DATASET
                    ensure_dir(dst_dataset_dir)
                    if Path(DATA_JSON_PATH).is_dir():
                        # copy whole dir
                        shutil.copytree(Path(DATA_JSON_PATH), dst_dataset_dir, dirs_exist_ok=True)
                    else:
                        shutil.copy2(Path(DATA_JSON_PATH), dst_dataset_dir / Path(DATA_JSON_PATH).name)
                    # set DATA_PATH to work root
                    prepared_json = dst_dataset_dir / f"{SPLIT}.json"
                else:
                    prepared_json = None
            except Exception as e2:
                print("Final fallback failed:", e2)
                prepared_json = None

        # set paths used by downstream scripts to working copies
        DATA_PATH = str(WORK_DATA_ROOT)
        PROMPTS_PATH = str(WORK_PROMPTS_ROOT)
        RESULTS_PATH = str(RESULTS_ROOT)
        # Update MODEL: if local model path is provided, we'll signal child processes via LOCAL_MODEL_PATH env
        if LOCAL_MODEL_PATH:
            print("Using LOCAL_MODEL_PATH:", LOCAL_MODEL_PATH)
            MODEL = "local-hf" if not MODEL else MODEL

    # Basic checks for backends (informational)
    if BACKEND == "ollama":
        print("Checking Ollama availability at", OLLAMA_URL)
        ok = check_ollama(OLLAMA_URL) if not running_on_kaggle else False
        print("Ollama reachable?" , ok)
        if not ok and not running_on_kaggle:
            print("Warning: Unable to reach Ollama at", OLLAMA_URL)
            print("If Ollama is running locally, set OLLAMA_URL correctly or start Ollama.")
    elif BACKEND == "hf":
        print("Checking HF dependencies")
        if not check_hf_model_env():
            print("Warning: Transformers not available — install 'transformers accelerate bitsandbytes safetensors' to use HF backend.")
    elif BACKEND == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set in env.")
    else:
        print("Unknown backend:", BACKEND)
        # but continue — user may want to proceed

    # Prepare environment dict for child processes
    child_env = os.environ.copy()
    child_env.update({
        "DATA_PATH": DATA_PATH,
        "PROMPTS_PATH": PROMPTS_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "DATASET_NAME": DATASET,
        "SPLIT": SPLIT,
        "MODEL_NAME": MODEL,
        "BATCH_NUM": str(BATCH_NUM),
        "MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
        "SEARCH_ROUND": str(SEARCH_ROUND),
        "BASE_URL": OLLAMA_URL,
        "STOP_WORDS": os.environ.get("STOP_WORDS", "------"),
    })
    if LOCAL_MODEL_PATH:
        child_env["LOCAL_MODEL_PATH"] = LOCAL_MODEL_PATH
    # give child processes the working root as well
    if running_on_kaggle:
        child_env.setdefault("WORK_ROOT", str(KAGGLE_WORK_ROOT))

    base_kwargs = build_base_kwargs({
        "DATA_PATH": child_env["DATA_PATH"],
        "DATASET_NAME": child_env["DATASET_NAME"],
        "SPLIT": child_env["SPLIT"],
        "RESULTS_PATH": child_env["RESULTS_PATH"],
        "OPENAI_API_KEY": child_env.get("OPENAI_API_KEY", ""),
        "MODEL_NAME": child_env["MODEL_NAME"],
        "STOP_WORDS": child_env.get("STOP_WORDS","------"),
        "MODE": child_env.get("MODE",""),
        "MAX_NEW_TOKENS": child_env.get("MAX_NEW_TOKENS","512"),
        "BATCH_NUM": child_env.get("BATCH_NUM","1"),
        "BASE_URL": child_env.get("BASE_URL",""),
    })

    # 1) translate_decompose.py
    print("\n==> Running translate_decompose")
    try:
        cmd = [sys.executable, str(ROOT / "translate_decompose.py")] + base_kwargs
        run_cmd(cmd, env=child_env)
    except subprocess.CalledProcessError as e:
        print("translate_decompose failed:", e)
        sys.exit(2)

    # 2) negate.py (optional)
    if not args.skip_negate:
        print("\n==> Running negate")
        try:
            negate_cmd = [sys.executable, str(ROOT / "negate.py"),
                          "--dataset_name", DATASET,
                          "--model", MODEL if MODEL else "model",
                          "--model_name", MODEL,
                          "--max_new_tokens", "256"]
            if USE_MODEL_FOR_NEGATION:
                negate_cmd.append("--use_model")
            run_cmd(negate_cmd, env=child_env)
        except subprocess.CalledProcessError as e:
            print("negate.py failed:", e)
            sys.exit(3)
    else:
        print("Skipping negate step (--skip-negate)")

    # 3) search_resolve.py for negation True and False
    if not args.skip_search:
        for neg in ("True","False"):
            print(f"\n==> Running search_resolve (negation={neg})")
            try:
                search_cmd = [sys.executable, str(ROOT / "search_resolve.py")] + base_kwargs + [
                    "--negation", neg,
                    "--search_round", str(SEARCH_ROUND)
                ]
                run_cmd(search_cmd, env=child_env)
            except subprocess.CalledProcessError as e:
                print("search_resolve.py failed:", e)
                sys.exit(4)
    else:
        print("Skipping search step (--skip-search)")

    # 4) evaluate.py
    if not args.skip_eval:
        print("\n==> Running evaluate")
        try:
            eval_cmd = [sys.executable, str(ROOT / "evaluate.py"),
                        "--dataset_name", DATASET,
                        "--model_name", MODEL]
            run_cmd(eval_cmd, env=child_env)
        except subprocess.CalledProcessError as e:
            print("evaluate.py failed:", e)
            sys.exit(5)
    else:
        print("Skipping evaluate (--skip-eval)")

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
