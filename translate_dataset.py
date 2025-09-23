"""
translate_dataset.py

Translate a dataset (JSON) into Bahasa Indonesia using LLM backend calls (via utils.OpenAIModel).
Parameters:
--data_json_path: optional path to single JSON file (if not provided, will search common
                     locations under --data_path/dataset_name/split/)
--data_path: root data folder (default ./data)
--dataset_name: name of dataset (e.g. ProofWriter)
--split: dataset split (default dev)
--output_dir: root output folder (default ./translated)
--sample-pct: percent of examples to translate (1-100, default 100)
--model_name: model name passed to OpenAIModel wrapper (if not set, will use env LLM_MODEL or default)
--max_new_tokens: max tokens to generate (default 512)
--temperature: generation temperature (default 0.0)
--batch_size: number of examples to process in one batch (default 1)
--prompts_root: folder where dataset-specific prompt templates live (default ./prompts)
--env: path to .env file (default ./.env)

Usage examples:

# translate full 100% dev split using .env settings:
python translate_dataset.py --dataset_name ProntoQA --split dev --sample-pct 100

# translate only 10%:
python translate_dataset.py --dataset_name ProntoQA --split dev --sample-pct 10

# translate only 1 row of data: set --sample-pct 0

"""

import os
import json
import argparse
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Try to import the OpenAIModel wrapper from utils.py,
# which already delegates to llm_backends (hf/ollama/openai).
try:
    from utils import OpenAIModel
except Exception as e:
    print("Error importing OpenAIModel from utils.py:", e)
    print("Make sure utils.py is on PYTHONPATH and contains OpenAIModel.")
    raise

# optional dotenv loader
try:
    from dotenv import load_dotenv  # type: ignore
    DOTENV_AVAILABLE = True
    print(f"Using load_dotenv")
except Exception:
    DOTENV_AVAILABLE = False

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


ROOT = Path(__file__).parent.resolve()

def find_source_json(data_json_path: Optional[str], data_path: str, dataset_name: str, split: str) -> Path:
    """
    Locate the JSON we want to translate. Returns path.
    """
    if data_json_path:
        p = Path(data_json_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"DATA_JSON_PATH provided but file not found: {p}")

    root = Path(data_path)
    # common candidate locations
    candidates = [
        root / dataset_name / split / f"{split}.json",
        root / dataset_name / f"{split}.json",
        root / split / f"{split}.json",
        root / f"{split}.json",
        root / dataset_name,
        root
    ]
    for c in candidates:
        if c.is_file() and c.suffix == ".json":
            return c
        if c.is_dir():
            # pick a JSON inside
            list_json = sorted(list(c.glob("*.json")))
            if list_json:
                return list_json[0]
    raise FileNotFoundError(f"Could not find dataset JSON. Checked candidates under {root}")

def extract_examples_from_json(tree: Any) -> Tuple[List[Dict[str, Any]], str]:
    """
    Given parsed JSON content, attempt to find list of examples and return (examples_list, container_key).
    container_key is '' if the JSON itself is a list.
    """
    if isinstance(tree, list):
        return tree, ""
    if isinstance(tree, dict):
        for candidate_key in ("data", "examples", "items", "instances", "questions"):
            if candidate_key in tree and isinstance(tree[candidate_key], list):
                return tree[candidate_key], candidate_key
        # fallback: find first list value
        for k, v in tree.items():
            if isinstance(v, list):
                return v, k
    raise ValueError("Unable to extract a list of examples from this JSON structure.")

def sample_examples(examples: List[Dict[str, Any]], pct: int, seed: int = 42) -> List[Dict[str, Any]]:
    if pct >= 100:
        return examples
    random.seed(seed)
    n = max(1, int(len(examples) * pct / 100.0))
    if n >= len(examples):
        return examples
    return random.sample(examples, n)

def default_translation_prompt(example: Dict[str, Any]) -> str:
    """
    Build a strict translation prompt instructing the model to output **only**
    a single JSON object which is the translated example.

    IMPORTANT: This prompt is intentionally very explicit about NOT translating
    the JSON key names. The output MUST contain the exact keys:
      id, context, question, options, answer, explanation
    (and any other non-textual keys must remain unchanged).
    """
    # pretty-print the JSON to give model full context
    example_json = json.dumps(example, indent=2, ensure_ascii=False)

    prompt = f"""
        You are a professional translator. Translate only the *values* (natural-language text)
        inside the JSON example below into Indonesian (Bahasa Indonesia).  DO NOT change, rename,
        or translate any JSON key names. The output must use the exact key names listed below.

        REQUIREMENTS (follow exactly):
        1) Output ONLY one valid JSON object (nothing else) that is the translated example.
        2) DO NOT translate or rename any key names. The output JSON MUST contain the exact keys:
        id, context, question, options, answer, explanation
        (if any extra keys exist in the input, keep them with their original key names).
        3) Preserve the data types and structure: strings remain strings, lists remain lists, booleans remain booleans.
        4) Do NOT modify the value of the "id" field. Keep the same id string as in the input.
        5) For multiple-choice options keep the letter labels (A), (B), (C), etc. unchanged — translate only the option text after the label.
        6) Start the response with '{' and end with '}' — nothing before or after (no markdown, no comments).
        7) If you cannot produce valid JSON, return exactly the single token: ERROR

        Below is the example to translate (do NOT repeat the original text in the output; only return the translated JSON object):

        {example_json}

        Desired output format example (this is an illustration of keys only — DO NOT COPY OR PUT THESE VALUES IN THE OUTPUT!; translate the actual values from the input):
        {{
        "id": "ProntoQA_1",
        "context": "Contoh konteks (terjemahan bahasa Indonesia)...",
        "question": "Contoh pertanyaan (terjemahan bahasa Indonesia)?",
        "options": ["A) Pilihan pertama", "B) Pilihan kedua"],
        "answer": "B",
        "explanation": ["Langkah 1 ...", "Langkah 2 ..."]
        }}

        Now produce the translated JSON object (with the exact keys as specified) and nothing else.
        """
    return prompt.strip()



def load_prompt_template(prompts_root: Path, dataset_name: str) -> Optional[str]:
    """
    If a prompts/{dataset_name}/translation.txt exists, use it as template.
    The template may contain the token "{example_json}" where the example JSON will be substituted.
    """
    candidate = prompts_root / dataset_name / "translation_bahasa.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return None

def apply_template(template: str, example: Dict[str, Any]) -> str:
    # If template contains {example_json} replace, else append the JSON at the end.
    example_json = json.dumps(example, indent=2, ensure_ascii=False)
    if "{example_json}" in template:
        return template.replace("{example_json}", example_json)
    else:
        return template + "\n\nExample:\n" + example_json

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a single JSON object from `text`.
    Strategy:
     - Clean common leading role tags like '[USER]' or 'assistant:'.
     - Scan for balanced braces pairs from left to right, but try parses starting from later
       closing braces (prefer the last valid JSON block).
     - Return the first successfully parsed JSON object (prefer later ones).
    """
    if not text:
        return None

    # quick clean of common role prefixes
    text = re.sub(r'^\s*(\[USER\]|\[ASSISTANT\]|user:|assistant:)', '', text, flags=re.IGNORECASE).strip()

    # find all positions of '{' and '}' to attempt balanced extraction
    starts = [m.start() for m in re.finditer(r'\{', text)]
    ends = [m.start() for m in re.finditer(r'\}', text)]

    if not starts or not ends:
        # fallback to trying json.loads on whole text
        try:
            return json.loads(text)
        except Exception:
            return None

    # To be robust, try candidate substrings using stack matching:
    # iterate over possible start positions (prefer later starts), and find matching end via stack.
    candidates = []
    for s_idx in range(len(starts)-1, -1, -1):  # prefer later starts
        s = starts[s_idx]
        depth = 0
        for i in range(s, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[s:i+1]
                    candidates.append(candidate)
                    # once we found a balanced block starting at s, break to prefer the shortest valid block for this start
                    break

    # Also try balanced blocks found by scanning from the end (prefer later valid blocks)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            return parsed
        except Exception:
            continue

    # last resort: try to parse any substring between any start and any end (bounded small sizes first)
    for s in starts:
        for e in ends:
            if e <= s:
                continue
            candidate = text[s:e+1]
            try:
                parsed = json.loads(candidate)
                return parsed
            except Exception:
                continue

    # final fallback: try whole text
    try:
        return json.loads(text)
    except Exception:
        return None


def write_translated_output(output_path: Path, original_tree: Any, examples_translated: List[Dict[str, Any]], container_key: str):
    """
    Rebuild the JSON structure and write to file.
    """
    ensure_parent = output_path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)

    if container_key == "":
        # examples list was root
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(examples_translated, f, indent=2, ensure_ascii=False)
        return

    # otherwise load original tree and replace container
    if isinstance(original_tree, dict):
        new_tree = dict(original_tree)  # shallow copy
        new_tree[container_key] = examples_translated
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(new_tree, f, indent=2, ensure_ascii=False)
        return

    # fallback: just write the list
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(examples_translated, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json_path", type=str, default="", help="Optional single JSON file input")
    parser.add_argument("--data_path", type=str, default="./data", help="Data root or dataset folder")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, default="./data_translated")
    parser.add_argument("--sample-pct", type=int, default=100, help="Percent of examples to translate (1-100)")
    parser.add_argument("--model_name", type=str, help="model name passed to OpenAIModel wrapper")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompts_root", type=str, default="./prompts", help="folder where dataset-specific prompt templates live")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    # locate source JSON
    source_json = find_source_json(args.data_json_path or None, args.data_path, args.dataset_name, args.split)
    print("[translate_dataset] Source JSON found:", source_json)

    # Read source
    with source_json.open("r", encoding="utf-8") as f:
        original_tree = json.load(f)

    examples, container_key = extract_examples_from_json(original_tree)
    print(f"[translate_dataset] Found {len(examples)} examples (container_key='{container_key}')")

    # sample
    sample_pct = max(0, min(100, int(args.sample_pct)))
    examples_to_translate = sample_examples(examples, sample_pct)
    print(f"[translate_dataset] Translating {len(examples_to_translate)} examples ({sample_pct}%)")

    # load prompt template if available
    prompts_root = Path(args.prompts_root)
    template = load_prompt_template(prompts_root, args.dataset_name)
    if template:
        print(f"[translate_dataset] Using prompt template: {prompts_root / args.dataset_name / 'translation_bahasa.txt'}")
    else:
        print("[translate_dataset] Using default translation prompt template (built-in).")

    # instantiate model wrapper from utils.OpenAIModel
    api_key = os.environ.get("OPENAI_API_KEY", "")
    stop_words = os.environ.get("STOP_WORDS", "------")
    model_name_env = os.environ.get("LLM_MODEL", "")
    print(model_name_env)
    openai_model = OpenAIModel(API_KEY=api_key, model_name=model_name_env, stop_words=stop_words, max_new_tokens=args.max_new_tokens, base_url=os.getenv("BASE_URL", None))

    # build prompts list
    prompts = []
    for ex in examples_to_translate:
        if template:
            prompts.append(apply_template(template, ex))
        else:
            prompts.append(default_translation_prompt(ex))

    translated_examples = []
    failed_count = 0

    # translation loop in batches
    batch_size = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(prompts), batch_size), desc="Translating", unit="batch"):
        chunk_prompts = prompts[i:i+batch_size]
        try:
            # batch_prompt_generate returns list of translated strings (or raises)
            outputs = openai_model.batch_prompt_generate(chunk_prompts, temperature=args.temperature)
        except Exception as e:
            print(f"[translate_dataset] Error calling model on batch starting at {i}: {e}")
            # try one-by-one fallback
            outputs = []
            for p in chunk_prompts:
                try:
                    out = openai_model.prompt_generate(p, temperature=args.temperature)
                    outputs.append(out)
                except Exception as e2:
                    print(f"  single-call failed: {e2}")
                    outputs.append("")

        # parse outputs and append
        for out_text, original_ex in zip(outputs, examples_to_translate[i:i+batch_size]):
            if not out_text or out_text.strip() == "":
                # can't parse -> keep original as fallback
                translated_examples.append(original_ex)
                failed_count += 1
                continue

            parsed = safe_parse_json(out_text)
            if parsed is None:
                # if cannot parse, attempt to salvage: put original but add `"__translated_text__": "<raw>"` field
                fallback = dict(original_ex)
                fallback["__translated_text_raw__"] = out_text.strip()
                translated_examples.append(fallback)
                failed_count += 1
            else:
                translated_examples.append(parsed)

    print(f"[translate_dataset] Translation completed. Failed/parsing issues: {failed_count}")

    # write output
    out_path = Path(args.output_dir) / args.dataset_name / f"{args.split}.json"
    write_translated_output(out_path, original_tree, translated_examples, container_key)
    print("[translate_dataset] Written translated dataset to:", out_path)
    print("Done.")

if __name__ == "__main__":
    main()