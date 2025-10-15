"""
run_pipeline.py

Usage:
  python run_pipeline.py

"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import re

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

def run_cmd(cmd_list, env=None):
    print(">>>", " ".join(map(str, cmd_list)))
    subprocess.run(cmd_list, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--data_path", type=str, default="./manual_data_translated")
    parser.add_argument("--prompts_path", type=str, default="./manual_prompts_translated")
    parser.add_argument("--dataset_name", type=str, default="ProntoQA")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--save_path", type=str, default="./results_translated")
    parser.add_argument("--sample_pct", type=int, default=10, help="Percent of examples to run (0-100)")
    parser.add_argument("--model_name", type=str, help="model name passed to model wrapper")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--search_round", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--language", type=str, default="en", help="Language for pipeline options: (id, en)")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    # Configs
    MODEL = args.model_name if args.model_name else os.environ.get("LLM_MODEL", "")
    DATASET = args.dataset_name if args.dataset_name else os.environ.get("DATASET_NAME")
    SPLIT = args.split if args.split else os.environ.get("SPLIT")
    DATA_PATH = args.data_path if args.data_path else os.environ.get("DATA_PATH")
    SAMPLE_PCT = str(args.sample_pct) if args.sample_pct is not None else int(os.environ.get("SAMPLE_PCT"))
    PROMPTS_PATH = args.prompts_path if args.prompts_path else os.environ.get("PROMPTS_PATH")
    RESULTS_PATH = args.save_path if args.save_path else os.environ.get("RESULTS_PATH")
    BATCH_NUM = str(args.batch_size) if args.batch_size else int(os.environ.get("BATCH_NUM"))
    MAX_NEW_TOKENS = str(args.max_new_tokens) if args.max_new_tokens else int(os.environ.get("MAX_NEW_TOKENS"))
    SEARCH_ROUND = args.search_round if args.search_round else int(os.environ.get("SEARCH_ROUND"))

    print(f"Model={MODEL} Dataset={DATASET} Prompts={PROMPTS_PATH} Split={SPLIT} Results={RESULTS_PATH}, Sample%={SAMPLE_PCT}")

    # build base args (use --key=value for options whose values might start with '-')
    base_kwargs = [
        "--data_path", DATA_PATH,
        "--dataset_name", DATASET,
        "--prompts_folder", PROMPTS_PATH,
        "--split", SPLIT,
        "--save_path", RESULTS_PATH,
        "--model_name", MODEL,
        # use = form for stop_words to avoid argparse treating '------' as an option token
        f"--stop_words={os.environ.get('STOP_WORDS', '------')}",
        "--max_new_tokens", MAX_NEW_TOKENS,
        "--batch_num", BATCH_NUM,
    ]

    # 1) translate_decompose.py
    print("\n==> Running translate_decompose")
    try:
        cmd = [sys.executable, str(ROOT / "translate_decompose.py")] + base_kwargs +[
            "--sample_pct", SAMPLE_PCT,
        ]
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print("translate_decompose failed:", e)
        sys.exit(2)

    # 2) negate.py
    print("\n==> Running negate")
    try:
        negate_cmd = [sys.executable, str(ROOT / "negate.py"),
                      "--dataset_name", DATASET,
                      "--model", MODEL,
                      "--save_path", RESULTS_PATH]
        run_cmd(negate_cmd)
    except subprocess.CalledProcessError as e:
        print("negate.py failed:", e)
        sys.exit(3)

    # 3) search_resolve.py for negation True and False
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

    # 4) evaluate.py
    print("\n==> Running evaluate")
    try:
        eval_cmd = [sys.executable, str(ROOT / "evaluate.py"),
                    "--dataset_name", DATASET,
                    "--model_name", MODEL,
                    "--save_path", RESULTS_PATH]
        run_cmd(eval_cmd)
    except subprocess.CalledProcessError as e:
        print("evaluate.py failed:", e)
        sys.exit(5)

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
