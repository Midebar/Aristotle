#!/usr/bin/env python3
"""
translate_prompts.py

Translate all prompt templates (text files) for a dataset into Bahasa Indonesia using OpenAIModel (utils.OpenAIModel).

Usage examples:

# Translate all prompt files for ProntoQA:
python translate_prompts.py --dataset_name ProntoQA --prompts_root ./prompts --output_dir ./prompts_translated

# Translate a single file:
python translate_prompts.py --file ./prompts/ProntoQA/and_or_decomposer.txt --output_dir ./prompts_translated
"""
import os
import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Dict

from tqdm import tqdm

# try to import OpenAIModel from utils.py (delegates to llm_backends)
try:
    from utils import OpenAIModel
except Exception as e:
    print("Error importing OpenAIModel from utils.py:", e)
    raise

# optional dotenv loader
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

ROOT = Path(__file__).parent.resolve()

START_MARKER = "<<<BEGIN_TRANSLATION>>>"
END_MARKER = "<<<END_TRANSLATION>>>"

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

def find_prompt_files(prompts_root: Path, dataset_name: Optional[str], single_file: Optional[Path] = None) -> List[Path]:
    """
    If single_file provided, return only it.
    Otherwise return all prompt files under prompts_root/dataset_name (txt).
    """
    if single_file:
        p = Path(single_file)
        if not p.exists():
            raise FileNotFoundError(f"Specified prompt file not found: {p}")
        return [p]

    if not dataset_name:
        raise ValueError("When --file is not provided, --dataset_name must be provided.")

    dataset_dir = prompts_root / dataset_name
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory for dataset not found: {dataset_dir}")
    files = sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix in (".txt",)])
    if not files:
        raise FileNotFoundError(f"No prompt files found in {dataset_dir}")
    return files

def default_prompt_template(prompt_text: str) -> str:
    """
    Instruction template to translate prompt files while preserving placeholders, code, and LaTeX.
    Requests the output to be placed between START_MARKER and END_MARKER.
    """
    # NOTE: keep START_MARKER and END_MARKER exact. We will extract between them.
    prompt = f"""
        You are a professional translator translating prompt templates for dataset-processing pipelines.
        Translate the following prompt text into Bahasa Indonesia (Bahasa Indonesia).

        IMPORTANT (follow exactly):
        - Produce ONLY the translated prompt text and NOTHING else.
        - Place the translated prompt EXACTLY between the two marker lines:
        {START_MARKER}
        <translated prompt text here>
        {END_MARKER}
        - Do NOT include the original prompt, commentary, numbering, or any extra text outside the markers.
        - Preserve ALL placeholders exactly, including tokens that look like [[PLACEHOLDER]], variable markers like $x, $y, $var, and patterns like {{var}} or <tag>.
        - Preserve LaTeX and math delimiters (\\(...\\), \\[...\\], $...$), code fences (``` ... ```), and any markup/markdown structure. Do NOT translate the contents inside code fences or backticks.
        - Preserve pipeline tokens like [[PREMISES]] exactly (we may later optionally translate their inner words).
        - Keep file structure, headings, punctuation and line breaks as in the original; translate only the human-readable instruction text.
        - Output only the translated prompt text between the markers.

        Original prompt file:
        {prompt_text}

        Now output ONLY the translated prompt text between the markers (the markers MUST appear exactly). Do not output anything else.
        """
    return prompt.strip()

def extract_translation(raw_text: str, start_marker: str = START_MARKER, end_marker: str = END_MARKER) -> str:
    """
    Extract the substring between start_marker and end_marker.
    If markers are missing, attempt a simple fallback heuristic:
      - If the raw output contains multiple blocks separated by blank lines, return the last block.
      - Otherwise return the whole raw_text.
    """
    pattern = re.compile(re.escape(start_marker) + r'(.*?)' + re.escape(end_marker), re.S)
    m = pattern.search(raw_text)
    if m:
        return m.group(1).strip()

    # fallback: try to return the last large block (commonly translation)
    parts = [p.strip() for p in re.split(r'\n{2,}', raw_text) if p.strip()]
    if len(parts) >= 2:
        return parts[-1].strip()

    # give up: return raw_text
    return raw_text.strip()

def load_placeholder_map(prompts_root: Path, dataset_name: Optional[str]) -> Dict[str, str]:
    """
    Load dataset-specific placeholder mapping from prompts/{dataset_name}/placeholders_map.json if present.
    Otherwise return a small built-in default mapping (upper-case keys).
    Mapping keys should be the inside of [[...]] (case-insensitive).
    Example JSON:
      { "PREMISES": "PREMIS", "CONJECTURE": "KONJEKTUR" }
    """
    default_map = {
        "PREMISES": "PREMIS",
        "CONJECTURE": "KONJEKTUR",
        "CONTEXT": "KONTEKS",
        "FACTS": "FAKTA",
        "RULES": "ATURAN",
        "EXAMPLE": "CONTOH",
        "FINAL FORM": "BENTUK_AKHIR",
        "FINAL_FORM": "BENTUK_AKHIR",
        "PREMISE": "PREMIS",
        "CONJECTURES": "KONJEKTUR",
        "SELECTED-CLAUSE": "KLAUSA_TERPILIH",
        "OPTIONS": "OPSI",
    }
    if not dataset_name:
        return default_map
    candidate = prompts_root / dataset_name / "placeholders_map.json"
    if candidate.exists():
        try:
            text = candidate.read_text(encoding="utf-8")
            loaded = json.loads(text)
            # normalize keys to uppercase
            return {k.upper(): v for k, v in loaded.items()}
        except Exception:
            # ignore and return default
            return default_map
    return default_map

def translate_bracketed_placeholders(translated_text: str, placeholder_map: Dict[str,str]) -> str:
    """
    Replace the INSIDE of [[...]] according to placeholder_map (case-insensitive).
    Example: [[PREMISES]] -> [[PREMIS]] if mapping has PREMISES->PREMIS.

    This function only replaces inner token, keeping double brackets.
    """
    def repl(m: re.Match):
        inner = m.group(1)  # original inner text
        key = inner.strip().upper()
        if key in placeholder_map:
            return f"[[{placeholder_map[key]}]]"
        # no mapping -> keep original exactly as-is
        return m.group(0)
    return re.sub(r'\[\[\s*([^\]]+?)\s*\]\]', repl, translated_text)

def validate_placeholders(original_text: str, translated_text: str, placeholder_map: Dict[str,str]) -> bool:
    """
    Basic checks to ensure placeholders/patterns are preserved or intentionally translated:
      - For each [[TOKEN]] in original: either [[TOKEN]] exists in translated OR [[mapped_value]] exists (if mapping).
      - If original had $ tokens, ensure translated has at least one $ or alternative LaTeX delimiters.
      - Ensure code fences and backticks are preserved if present.
    """
    # bracket placeholders
    orig_ph = re.findall(r'\[\[\s*([^\]]+?)\s*\]\]', original_text)
    for ph in set(orig_ph):
        ph_u = ph.strip().upper()
        raw_form = f"[[{ph.strip()}]]"
        mapped_good = False
        # Allowed: exact preserved
        if raw_form in translated_text:
            continue
        # Allowed: mapped value present
        if ph_u in placeholder_map:
            mapped = placeholder_map[ph_u]
            if f"[[{mapped}]]" in translated_text:
                continue
        # Not preserved
        # Unknown: maybe model translated the inner word differently; consider it a failure
        return False

    # $ variables presence check (if original had $ tokens, require at least some $ in translated)
    if "$" in original_text:
        if "$" not in translated_text and ("\\(" not in translated_text and "\\[" not in translated_text):
            return False

    # code fences/backticks
    if "```" in original_text and "```" not in translated_text:
        return False
    if "`" in original_text and "`" not in translated_text:
        # if original used single backtick occurrences, be forgiving: only fail if none at all
        return False

    return True

def write_translated_file(dst_dir: Path, original_file: Path, translated_text: str, overwrite: bool = True) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / original_file.name
    if dst_file.exists() and not overwrite:
        raise FileExistsError(f"Destination exists and overwrite disabled: {dst_file}")
    dst_file.write_text(translated_text, encoding="utf-8")
    return dst_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=False, help="Dataset prompt subfolder under prompts_root")
    parser.add_argument("--prompts_root", type=str, default="./prompts", help="Root folder for prompts (contains per-dataset subfolders)")
    parser.add_argument("--file", type=str, default="", help="Optional single prompt file to translate")
    parser.add_argument("--output_dir", type=str, default="./prompts_translated", help="Where to write translated prompts")
    parser.add_argument("--model_name", type=str, default=os.getenv("LLM_MODEL", ""), help="Model name passed to OpenAIModel wrapper")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing translated files")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args()

    env_path = ROOT / args.env
    load_env_file(env_path)

    prompts_root = Path(args.prompts_root)
    single_file = Path(args.file) if args.file else None

    if single_file is not None and args.dataset_name:
        print("[translate_prompts] Both --file and --dataset_name provided; using --file only.")

    if single_file is None and not args.dataset_name:
        parser.error("Either --dataset_name or --file must be provided.")

    try:
        files = find_prompt_files(prompts_root, args.dataset_name, single_file=single_file)
    except Exception as e:
        print("[translate_prompts] Error locating files:", e)
        sys.exit(2)

    print(f"[translate_prompts] Found {len(files)} prompt files. Translating all of them.")

    # optional dataset-specific template (not required)
    dataset_template = None
    if args.dataset_name:
        candidate = prompts_root / args.dataset_name / "translation_bahasa_prompts.txt"
        if candidate.exists():
            dataset_template = candidate.read_text(encoding="utf-8")
            print(f"[translate_prompts] Using dataset-specific template: {candidate}")

    # instantiate model wrapper
    model_name_env = args.model_name or os.getenv("LLM_MODEL", "")
    if not model_name_env:
        print("[translate_prompts] No model specified via --model_name or LLM_MODEL env. Exiting.")
        sys.exit(3)
    openai_model = OpenAIModel(API_KEY=os.getenv("OPENAI_API_KEY", ""), model_name=model_name_env,
                               stop_words=os.getenv("STOP_WORDS", "------"), max_new_tokens=args.max_new_tokens,
                               base_url=os.getenv("BASE_URL", None))

    dst_base = Path(args.output_dir) / (args.dataset_name or "single_file")
    failed = []
    placeholder_map = load_placeholder_map(Path(args.prompts_root), args.dataset_name)

    for p in tqdm(files, desc="Translating prompts"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[translate_prompts] Failed to read {p}: {e}")
            failed.append(p)
            continue

        # prepare model prompt (dataset template overrides default)
        if dataset_template:
            prompt_for_model = dataset_template.replace("{prompt_text}", text) if "{prompt_text}" in dataset_template else dataset_template + "\n\n### SOURCE PROMPT ###\n\n" + text
        else:
            prompt_for_model = default_prompt_template(text)

        # call LLM
        try:
            raw_out = openai_model.prompt_generate(prompt_for_model, temperature=args.temperature)
            if raw_out is None:
                raw_out = ""
        except Exception as e:
            print(f"[translate_prompts] Error calling model for {p}: {e}")
            raw_out = ""
            failed.append(p)

        # attempt extraction
        translated_extracted = extract_translation(raw_out)

        # If extraction failed or empty, fall back to using raw_out (model may not have used markers)
        if not translated_extracted:
            # try raw out (it might already only contain translation)
            translated_extracted = raw_out.strip()

        # Now, translate bracketed placeholders according to the map (this is intentional)
        translated_after_placeholders = translate_bracketed_placeholders(translated_extracted, placeholder_map)

        # Validate preservation of placeholders / tokens
        ok = validate_placeholders(text, translated_after_placeholders, placeholder_map)
        if not ok:
            print(f"[translate_prompts] Placeholder validation FAILED for {p}; saving original as fallback.")
            # Save original file as fallback (but still write to dst so pipeline can keep running)
            try:
                written = write_translated_file(dst_base, p, text, overwrite=args.overwrite)
            except Exception as e:
                print(f"[translate_prompts] Failed to write fallback original for {p}: {e}")
            failed.append(p)
            continue

        # write translated file
        try:
            written = write_translated_file(dst_base, p, translated_after_placeholders, overwrite=args.overwrite)
        except Exception as e:
            print(f"[translate_prompts] Failed to write translated file for {p}: {e}")
            failed.append(p)
            continue

    success = len(files) - len(failed)
    print(f"[translate_prompts] Done. Translated {success} files. Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(" -", f)

if __name__ == "__main__":
    main()
