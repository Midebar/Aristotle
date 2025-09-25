"""
translate_dataset.py

Translate a dataset (JSON) into Bahasa Indonesia using LLM backend calls (via utils.OpenAIModel).

This version DOES NOT use masking. Instead it:
 - explicitly instructs the model not to translate option values (True/False/Unknown/Yes/No/None)
 - validates the returned JSON and restores original `options` if the model changed them.
"""

import os
import json
import argparse
import random
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Try to import the OpenAIModel wrapper from utils.py
try:
    from utils import OpenAIModel
except Exception as e:
    print("Error importing OpenAIModel from utils.py:", e)
    raise

# optional dotenv loader
try:
    from dotenv import load_dotenv  # type: ignore
    DOTENV_AVAILABLE = True
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
    if data_json_path:
        p = Path(data_json_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"DATA_JSON_PATH provided but file not found: {p}")

    root = Path(data_path)
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
            list_json = sorted(list(c.glob("*.json")))
            if list_json:
                return list_json[0]
    raise FileNotFoundError(f"Could not find dataset JSON. Checked candidates under {root}")

def extract_examples_from_json(tree: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(tree, list):
        return tree, ""
    if isinstance(tree, dict):
        for candidate_key in ("data", "examples", "items", "instances", "questions"):
            if candidate_key in tree and isinstance(tree[candidate_key], list):
                return tree[candidate_key], candidate_key
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

# tokens considered option-values we want to preserve (case-sensitive)
OPTION_TOKENS = ["True", "False", "Unknown", "Yes", "No", "None"]

_OPTION_RE = re.compile(r'\b(' + '|'.join(re.escape(t) for t in OPTION_TOKENS) + r')\b')

def default_translation_prompt(example: Dict[str, Any]) -> str:
    """
    Prompt instructs the model strongly to NOT translate option values.
    """
    example_json = json.dumps(example, indent=2, ensure_ascii=False)
    prompt = f"""
        You are a professional translator. Translate only the *values* (natural-language text)
        inside the JSON example below into Indonesian (Bahasa Indonesia). DO NOT change, rename,
        or translate any JSON key names.

        CRITICAL REQUIREMENTS:
        1) Output ONLY one valid JSON object (nothing else) that is the translated example.
        2) The output JSON MUST contain the exact keys: id, context, question, options, answer, explanation
           (and any other keys present in the input must be preserved).
        3) Preserve data types and structure (strings remain strings, lists remain lists).
        4) DO NOT modify the 'id' value.
        5) IMPORTANT: Do NOT translate option-values. If any option contains tokens like
           True, False, Unknown, Yes, No, None, keep those tokens exactly as they appear in the input.
           Also preserve letter labels like "A)", "B)" unchanged. If you cannot follow these rules,
           return the single token: ERROR
        6) Start the response with '{{' and end with '}}' — nothing before or after.

        Below is the example to translate (translate only the language in values; keep option-values EXACT):
        {example_json}

        Now produce the translated JSON object (and nothing else).
    """
    return prompt.strip()

def load_prompt_template(prompts_root: Path, dataset_name: str) -> Optional[str]:
    candidate = prompts_root / dataset_name / "translation_bahasa.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return None

def apply_template(template: str, example: Dict[str, Any]) -> str:
    example_json = json.dumps(example, indent=2, ensure_ascii=False)
    if "{example_json}" in template:
        return template.replace("{example_json}", example_json)
    else:
        return template + "\n\nExample:\n" + example_json

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = re.sub(r'^\s*(\[USER\]|\[ASSISTANT\]|user:|assistant:)', '', text, flags=re.IGNORECASE).strip()
    starts = [m.start() for m in re.finditer(r'\{', text)]
    ends = [m.start() for m in re.finditer(r'\}', text)]
    if not starts or not ends:
        try:
            return json.loads(text)
        except Exception:
            return None
    candidates = []
    for s_idx in range(len(starts)-1, -1, -1):
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
                    break
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            return parsed
        except Exception:
            continue
    try:
        return json.loads(text)
    except Exception:
        return None

def write_translated_output(output_path: Path, original_tree: Any, examples_translated: List[Dict[str, Any]], container_key: str):
    ensure_parent = output_path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    if container_key == "":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(examples_translated, f, indent=2, ensure_ascii=False)
        return
    if isinstance(original_tree, dict):
        new_tree = dict(original_tree)
        new_tree[container_key] = examples_translated
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(new_tree, f, indent=2, ensure_ascii=False)
        return
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(examples_translated, f, indent=2, ensure_ascii=False)

def options_need_restoration(original_options: List[Any], parsed_options: List[Any]) -> bool:
    """
    Decide whether to restore original options.
    - If lengths differ -> restore.
    - If any original option contains an OPTION_TOKEN but the corresponding parsed option
      does not contain the same token -> restore.
    """
    if not isinstance(original_options, list):
        return False
    if not isinstance(parsed_options, list):
        return True
    if len(original_options) != len(parsed_options):
        return True

    for orig_opt, parsed_opt in zip(original_options, parsed_options):
        if not isinstance(orig_opt, str) or not isinstance(parsed_opt, str):
            # if not strings, skip strict checking
            continue
        # find tokens in original option
        orig_tokens = [t for t in OPTION_TOKENS if re.search(r'\b' + re.escape(t) + r'\b', orig_opt)]
        if orig_tokens:
            # require that at least one of those tokens appears verbatim in parsed_opt
            if not any(re.search(r'\b' + re.escape(t) + r'\b', parsed_opt) for t in orig_tokens):
                return True
    return False

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

    source_json = find_source_json(args.data_json_path or None, args.data_path, args.dataset_name, args.split)
    print("[translate_dataset] Source JSON found:", source_json)

    with source_json.open("r", encoding="utf-8") as f:
        original_tree = json.load(f)

    examples, container_key = extract_examples_from_json(original_tree)
    print(f"[translate_dataset] Found {len(examples)} examples (container_key='{container_key}')")

    sample_pct = max(0, min(100, int(args.sample_pct)))
    examples_to_translate = sample_examples(examples, sample_pct)
    print(f"[translate_dataset] Translating {len(examples_to_translate)} examples ({sample_pct}%)")

    prompts_root = Path(args.prompts_root)
    template = load_prompt_template(prompts_root, args.dataset_name)
    if template:
        print(f"[translate_dataset] Using prompt template: {prompts_root / args.dataset_name / 'translation_bahasa.txt'}")
    else:
        print("[translate_dataset] Using default translation prompt template (built-in).")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    stop_words = os.environ.get("STOP_WORDS", "------")
    model_name_env = os.environ.get("LLM_MODEL", "")
    print(model_name_env)
    openai_model = OpenAIModel(API_KEY=api_key, model_name=model_name_env, stop_words=stop_words, max_new_tokens=args.max_new_tokens, base_url=os.getenv("BASE_URL", None))

    prompts = []
    # we will keep original examples for option restoration
    for ex in examples_to_translate:
        if template:
            prompts.append(apply_template(template, ex))
        else:
            prompts.append(default_translation_prompt(ex))

    translated_examples = []
    failed_count = 0
    batch_size = max(1, int(args.batch_size))

    for i in tqdm(range(0, len(prompts), batch_size), desc="Translating", unit="batch"):
        chunk_prompts = prompts[i:i+batch_size]
        try:
            outputs = openai_model.batch_prompt_generate(chunk_prompts, temperature=args.temperature)
        except Exception as e:
            print(f"[translate_dataset] Error calling model on batch starting at {i}: {e}")
            outputs = []
            for p in chunk_prompts:
                try:
                    out = openai_model.prompt_generate(p, temperature=args.temperature)
                    outputs.append(out)
                except Exception as e2:
                    print(f"  single-call failed: {e2}")
                    outputs.append("")

        batch_examples = examples_to_translate[i:i+batch_size]
        for out_text, original_ex in zip(outputs, batch_examples):
            if not out_text or out_text.strip() == "":
                translated_examples.append(original_ex)
                failed_count += 1
                continue

            parsed = safe_parse_json(out_text)
            if parsed is None:
                fallback = dict(original_ex)
                fallback["__translated_text_raw__"] = out_text.strip()
                translated_examples.append(fallback)
                failed_count += 1
                continue

            # ensure options preserved: if the model changed option-values, restore original options
            try:
                orig_opts = original_ex.get("options")
                parsed_opts = parsed.get("options")
                if orig_opts is not None and options_need_restoration(orig_opts, parsed_opts):
                    # restore the original options list unchanged
                    parsed["options"] = copy.deepcopy(orig_opts)
                # done: append parsed (possibly with restored options)
                translated_examples.append(parsed)
            except Exception as e:
                print("[translate_dataset] Warning during options validation/restoration:", e)
                translated_examples.append(parsed)

    print(f"[translate_dataset] Translation completed. Failed/parsing issues: {failed_count}")

    out_path = Path(args.output_dir) / args.dataset_name / f"{args.split}.json"
    write_translated_output(out_path, original_tree, translated_examples, container_key)
    print("[translate_dataset] Written translated dataset to:", out_path)
    print("Done.")

if __name__ == "__main__":
    main()
