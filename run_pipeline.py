"""
run_pipeline.py

Usage:
  python run_pipeline.py            # runs full pipeline based on .env
  python run_pipeline.py --pilot    # runs pipeline in pilot mode (PILOT_ONLY override)
  python run_pipeline.py --skip-negate
  python run_pipeline.py --skip-search
  python run_pipeline.py --skip-eval

This script expects a .env file in the same directory or environment variables set.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# dependency: python-dotenv for reading .env files
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
            # minimalist loader: parse KEY=VALUE lines
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

def check_hf_model(hf_model: str) -> bool:
    # Lightweight check: attempt to import transformers and tokenizers without loading the model
    try:
        import transformers
        return True
    except Exception:
        return False

def run_cmd(cmd_list, env=None):
    print(">>>", " ".join(map(str, cmd_list)))
    subprocess.run(cmd_list, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--pilot", action="store_true", help="Run pilot only (overrides PILOT_ONLY)")
    parser.add_argument("--skip-negate", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    # pull configuration from environment
    BACKEND = os.environ.get("LLM_BACKEND", "ollama")
    MODEL = os.environ.get("LLM_MODEL", "")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    DATASET = os.environ.get("DATASET_NAME", "ProofWriter")
    SPLIT = os.environ.get("SPLIT", "dev")
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

    # Basic checks
    if BACKEND == "ollama":
        print("Checking Ollama availability at", OLLAMA_URL)
        ok = check_ollama(OLLAMA_URL)
        print("Ollama reachable?" , ok)
        if not ok:
            print("Warning: Unable to reach Ollama at", OLLAMA_URL)
            print("If Ollama is running locally, set OLLAMA_URL correctly or start Ollama.")
    elif BACKEND == "hf":
        print("Checking HF dependencies")
        if not check_hf_model(MODEL):
            print("Warning: Transformers not available — install 'transformers accelerate bitsandbytes safetensors' to use HF backend.")
    elif BACKEND == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set in env.")
    else:
        print("Unknown backend:", BACKEND)
        sys.exit(1)

    # build base args robustly (use --key=value for options whose values might start with '-')
    base_kwargs = [
        "--data_path", DATA_PATH,
        "--dataset_name", DATASET,
        "--split", SPLIT,
        "--save_path", RESULTS_PATH,
        "--api_key", os.environ.get("OPENAI_API_KEY", ""),
        "--model_name", MODEL,
        # use = form for stop_words to avoid argparse treating '------' as an option token
        f"--stop_words={os.environ.get('STOP_WORDS', '------')}",
        "--mode", os.environ.get("MODE", ""),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--batch_num", str(BATCH_NUM),
    ]
    # only include base_url if set (avoid passing empty string as separate token)
    if os.environ.get("OLLAMA_URL", ""):
        base_kwargs.append(f"--base_url={os.environ.get('OLLAMA_URL')}")

    # 1) translate_decompose.py
    print("\n==> Running translate_decompose")
    try:
        cmd = [sys.executable, str(ROOT / "translate_decompose.py")] + base_kwargs
        run_cmd(cmd)
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
            run_cmd(negate_cmd)
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
                run_cmd(search_cmd)
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
            run_cmd(eval_cmd)
        except subprocess.CalledProcessError as e:
            print("evaluate.py failed:", e)
            sys.exit(5)
    else:
        print("Skipping evaluate (--skip-eval)")

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
