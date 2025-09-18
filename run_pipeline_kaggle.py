"""
run_pipeline_kaggle_adapter.py

Usage:
  python run_pipeline_kaggle_adapter.py            # runs full pipeline based on .env (or Kaggle env)
  python run_pipeline_kaggle_adapter.py --kaggle   # enable Kaggle adapter (copy inputs to /kaggle/working)

This script expects a .env file in the same directory or environment variables set.
When running on Kaggle, set environment variables (or let the adapter infer them):
  LOCAL_MODEL_PATH, DATA_JSON_PATH, PROMPTS_INPUT_PATH, WORK_DATA_ROOT,
  WORK_PROMPTS_ROOT, RESULTS_ROOT, SHIM_BASE_URL
"""
import os
import sys
import subprocess
import argparse
import shutil
import json
import random
from pathlib import Path

# optional dotenv loader
try:
    from dotenv import load_dotenv  # type: ignore
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

ROOT = Path(__file__).parent.resolve()

def load_env_file(env_path: Path):
    if env_path.exists():
        if DOTENV_AVAILABLE:
            load_dotenv(dotenv_path=str(env_path))
        else:
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

def check_ollama(url: str) -> bool:
    try:
        import requests  # local import
        r = requests.get(url + "/api/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_hf_model_env() -> bool:
    try:
        import transformers
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
            dst_for_dataset.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_prompts, dst_for_dataset / src_prompts.name)
    except Exception as e:
        print("Error copying prompts:", e)

def prepare_dataset_input(original_json_path: Path, dst_data_dir: Path, dataset_name: str, split_name: str, split_percent: int):
    ensure_dir(dst_data_dir)
    dst_dataset_dir = dst_data_dir / dataset_name
    ensure_dir(dst_dataset_dir)

    # locate source
    if original_json_path.is_file() and original_json_path.suffix == ".json":
        src_file = original_json_path
    else:
        candidate = original_json_path / split_name
        if candidate.is_dir():
            maybe = candidate / f"{split_name}.json"
            src_file = maybe if maybe.exists() else (list(candidate.glob("*.json"))[0] if list(candidate.glob("*.json")) else None)
        else:
            maybe = original_json_path / f"{split_name}.json"
            src_file = maybe if maybe.exists() else None

    if src_file is None or not src_file.exists():
        raise FileNotFoundError(f"Could not find source dataset JSON. Checked {original_json_path}")

    print("Source dataset JSON:", src_file)
    with src_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        data_list = data
    elif isinstance(data, dict):
        for key in ("data", "examples", "items", "instances"):
            if key in data and isinstance(data[key], list):
                data_list = data[key]
                break
        else:
            data_list = [data]
    else:
        raise ValueError("Unsupported dataset format (expected list or dict containing list).")

    n_total = len(data_list)
    if split_percent is None or split_percent >= 100:
        sampled = data_list
    else:
        k = max(1, int(n_total * (split_percent / 100.0)))
        random.seed(42)
        sampled = random.sample(data_list, k) if k < n_total else data_list

    dst_file = dst_dataset_dir / f"{split_name}.json"
    print(f"Writing {len(sampled)} / {n_total} instances to {dst_file}")
    with dst_file.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    return dst_file

def build_base_kwargs(env):
    base_kwargs = [
        "--data_path", env["DATA_PATH"],
        "--dataset_name", env["DATASET_NAME"],
        "--split", env["SPLIT"],
        "--split_percent", env.get("DATA_SPLIT_PERCENTAGE", "10"),
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
    parser.add_argument("--skip-negate", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle adapter (copy inputs to /kaggle/working)")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    running_on_kaggle = args.kaggle or Path("/kaggle").exists()
    if running_on_kaggle:
        print("Kaggle runtime detected or --kaggle provided.")

    # environment / defaults
    BACKEND = os.environ.get("LLM_BACKEND", "ollama")
    MODEL = os.environ.get("LLM_MODEL", "")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    DATASET = os.environ.get("DATASET_NAME", "ProofWriter")
    SPLIT = os.environ.get("SPLIT", "dev")
    DATA_SPLIT_PERCENTAGE = int(os.environ.get("DATA_SPLIT_PERCENTAGE", "10"))
    KAGGLE_WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))

    LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "")
    DATA_JSON_PATH = os.environ.get("DATA_JSON_PATH", "")
    PROMPTS_INPUT_PATH = os.environ.get("PROMPTS_INPUT_PATH", "")

    WORK_DATA_ROOT = Path(os.environ.get("WORK_DATA_ROOT", str(KAGGLE_WORK_ROOT / "data")))
    WORK_PROMPTS_ROOT = Path(os.environ.get("WORK_PROMPTS_ROOT", str(KAGGLE_WORK_ROOT / "prompts")))
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(KAGGLE_WORK_ROOT / "results")))

    # fallback runtime config (child process values)
    DATA_PATH = os.environ.get("DATA_PATH", "./data")
    PROMPTS_PATH = os.environ.get("PROMPTS_PATH", "./prompts")
    RESULTS_PATH = os.environ.get("RESULTS_PATH", "./results")
    BATCH_NUM = os.environ.get("BATCH_NUM", "1")
    MAX_NEW_TOKENS = os.environ.get("MAX_NEW_TOKENS", "512")
    SEARCH_ROUND = os.environ.get("SEARCH_ROUND", "10")

    print(f"Backend={BACKEND} Model={MODEL} Dataset={DATASET}")

    # Kaggle adapter: copy prompts and dataset, update working paths
    if running_on_kaggle:
        src_prompts = Path(PROMPTS_INPUT_PATH) if PROMPTS_INPUT_PATH else (ROOT / "prompts")
        if PROMPTS_INPUT_PATH and not Path(PROMPTS_INPUT_PATH).exists():
            print(f"Warning: PROMPTS_INPUT_PATH={PROMPTS_INPUT_PATH} not found on disk.")
        ensure_dir(WORK_PROMPTS_ROOT)
        copy_prompts(Path(src_prompts), WORK_PROMPTS_ROOT, DATASET)

        src_data_path = Path(DATA_JSON_PATH) if DATA_JSON_PATH else Path("/kaggle/input")
        ensure_dir(WORK_DATA_ROOT)
        try:
            prepared_json = prepare_dataset_input(src_data_path, WORK_DATA_ROOT, DATASET, SPLIT, DATA_SPLIT_PERCENTAGE)
        except Exception as e:
            print("Error preparing dataset for Kaggle adapter:", e)
            # try fallback copying
            try:
                if DATA_JSON_PATH and Path(DATA_JSON_PATH).exists():
                    dst_dataset_dir = WORK_DATA_ROOT / DATASET
                    ensure_dir(dst_dataset_dir)
                    if Path(DATA_JSON_PATH).is_dir():
                        shutil.copytree(Path(DATA_JSON_PATH), dst_dataset_dir, dirs_exist_ok=True)
                    else:
                        shutil.copy2(Path(DATA_JSON_PATH), dst_dataset_dir / Path(DATA_JSON_PATH).name)
                    prepared_json = dst_dataset_dir / f"{SPLIT}.json"
                else:
                    prepared_json = None
            except Exception as e2:
                print("Final fallback failed:", e2)
                prepared_json = None

        DATA_PATH = str(WORK_DATA_ROOT)
        PROMPTS_PATH = str(WORK_PROMPTS_ROOT)
        RESULTS_PATH = str(RESULTS_ROOT)
        if LOCAL_MODEL_PATH:
            print("Using LOCAL_MODEL_PATH:", LOCAL_MODEL_PATH)
            MODEL = "local-hf" if not MODEL else MODEL

    # backend checks (informational)
    if BACKEND == "ollama":
        print("Checking Ollama availability at", OLLAMA_URL)
        ok = check_ollama(OLLAMA_URL) if not running_on_kaggle else False
        print("Ollama reachable?" , ok)
        if not ok and not running_on_kaggle:
            print("Warning: Unable to reach Ollama at", OLLAMA_URL)
    elif BACKEND == "hf":
        print("Checking HF dependencies")
        if not check_hf_model_env():
            print("Warning: Transformers not available — install 'transformers accelerate bitsandbytes safetensors' to use HF backend.")
    elif BACKEND == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set in env.")
    else:
        print("Unknown backend:", BACKEND)

    # prepare child env
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
            if os.environ.get("USE_MODEL_FOR_NEGATION", "false").lower() in ("1","true","yes"):
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
