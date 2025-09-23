"""
translate_prompts.py

Robust translation of prompt templates into Bahasa Indonesia using utils.OpenAIModel.

Improvements over previous version:
 - More robust extraction when model doesn't follow markers.
 - Tolerant placeholder matching (ignores whitespace inside [[...]]).
 - Candidate scoring: choose block that preserves the most placeholders.
 - Save raw outputs and chosen candidate with metadata for debugging.
 - If fallback write collides and overwrite==False, write .fallback.txt and keep raw logs.

Usage: same as before.
"""

import os
import sys
import argparse
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

try:
    from utils import OpenAIModel
except Exception as e:
    print("Error importing OpenAIModel from utils.py:", e)
    raise

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

ROOT = Path(__file__).parent.resolve()
START_MARKER = "<<<BEGIN_TRANSLATION>>>"
END_MARKER = "<<<END_TRANSLATION>>>"

# -------------------------
# env loader
# -------------------------
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

# -------------------------
# file helpers
# -------------------------
def find_prompt_files(prompts_root: Path, dataset_name: Optional[str], single_file: Optional[Path] = None) -> List[Path]:
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

# -------------------------
# placeholders normalization & masking
# -------------------------
PH_PATTERN = re.compile(r'\[\[\s*([^\]]+?)\s*\]\]')  # matches [[  TOKEN  ]] capturing inner token

def find_placeholders(text: str) -> List[str]:
    """Return normalized placeholders like '[[TOKEN]]' (no extra whitespace)."""
    res = []
    for m in PH_PATTERN.finditer(text):
        inner = m.group(1).strip()
        res.append(f"[[{inner}]]")
    return sorted(set(res))

def mask_placeholders(text: str) -> Tuple[str, Dict[str,str]]:
    """
    Replace each [[TOKEN]] with an internal mask __PH_i__ and return mapping.
    """
    mapping = {}
    idx = 0
    def repl(m):
        nonlocal idx
        inner = m.group(1).strip()
        mask = f"__PH_{idx}__"
        mapping[mask] = f"[[{inner}]]"
        idx += 1
        return mask
    masked = PH_PATTERN.sub(repl, text)
    return masked, mapping

def unmask_placeholders(text: str, mapping: Dict[str,str]) -> str:
    # replace masks with original placeholders (longer keys first)
    for k in sorted(mapping.keys(), key=lambda x: -len(x)):
        text = text.replace(k, mapping[k])
    return text

# -------------------------
# prompt template
# -------------------------
def default_prompt_template(prompt_text: str) -> str:
    prompt = f"""
        You are a professional translator translating prompt templates for dataset-processing pipelines.
        Translate the following prompt text into Bahasa Indonesia (Bahasa Indonesia).

        IMPORTANT (FOLLOW EXACTLY):
        - Produce ONLY the translated prompt text and NOTHING else.
        - Place the translated prompt EXACTLY between the two marker lines below (markers MUST appear):
        {START_MARKER}
        <translated prompt text here>
        {END_MARKER}
        - Preserve ALL double-square placeholders like [[PREMISES]] exactly (do not translate/change whitespace inside the brackets).
        - Preserve code fences (```...```), inline backticks `...`, LaTeX delimiters \\(...\\), \\[...\\], $...$, and variable tokens like $x, {{}}, <tag>.
        - Preserve boolean values like True/False, None/null, and numbers as-is.
        - Do NOT add commentary, headers, or extra text outside the markers.
        - If you cannot produce a safe translation that preserves placeholders exactly, output exactly the block:
        {START_MARKER}
        ERROR
        {END_MARKER}

        Original prompt file (masked placeholders):
        {prompt_text}

        Now output ONLY the translated prompt text between the markers (the markers MUST appear). Nothing else.
        """
    return prompt.strip()

# -------------------------
# extraction & scoring
# -------------------------
def extract_between_markers(raw: str) -> Optional[str]:
    """Return last block between markers if present, else None."""
    pattern = re.compile(re.escape(START_MARKER) + r'(.*?)' + re.escape(END_MARKER), re.S)
    matches = pattern.findall(raw)
    if not matches:
        return None
    # prefer the last matched block (model may include earlier ones)
    return matches[-1].strip()

def split_into_blocks(raw: str) -> List[str]:
    """
    Split by blank-line clusters into candidate blocks, plus individual lines fallback.
    Returns non-empty trimmed blocks.
    """
    parts = [p.strip() for p in re.split(r'\n{2,}', raw) if p.strip()]
    if not parts:
        # split by lines
        parts = [line.strip() for line in raw.splitlines() if line.strip()]
    return parts

def normalize_placeholder_token(t: str) -> str:
    """
    Normalize [[ something ]] -> [[something]] for comparisons.
    """
    return re.sub(r'\s+', ' ', t).replace('[[ ', '[[').replace(' ]]', ']]').strip()

def placeholder_score(original_placeholders: List[str], candidate: str) -> Tuple[int,int]:
    """
    Score candidate by how many placeholders it contains.
    Returns (count_matched, count_total_original).
    Matching is done on normalized placeholders (ignore internal whitespace).
    """
    cand_ph_raw = PH_PATTERN.findall(candidate)
    cand_ph = set([f"[[{p.strip()}]]" for p in cand_ph_raw])
    orig_set = set(original_placeholders)
    matched = sum(1 for ph in orig_set if ph in cand_ph)
    return matched, len(orig_set)

def candidate_quality(original_text: str, candidate: str) -> Dict:
    """
    Produce a quality dict: placeholder_matched, total_placeholders, has_code_fence, code_counts, length, raw_candidate
    """
    orig_ph = find_placeholders(original_text)
    matched, total = placeholder_score(orig_ph, candidate)
    has_code = "```" in original_text and "```" in candidate
    backticks_ok = ("`" in original_text and "`" in candidate) or ("`" not in original_text)
    latex_ok = (("\\(" in original_text or "\\[" in original_text) and ("\\(" in candidate or "\\[" in candidate)) or ("\\(" not in original_text and "\\[" not in original_text)
    return {
        "matched": matched,
        "total": total,
        "has_code_ok": has_code,
        "backticks_ok": backticks_ok,
        "latex_ok": latex_ok,
        "len": len(candidate),
        "candidate": candidate
    }

def choose_best_candidate(raw_out: str, original_text: str) -> Tuple[str, Dict]:
    """
    Strategy:
     - If markers exist: use last marker block.
     - Else: split into blocks and score each by placeholder match (highest matched first).
       Favor block that matches all placeholders if any; otherwise use block with highest matched.
     - Return chosen string and a dict with diagnostics.
    """
    # 1) marker
    from_markers = extract_between_markers(raw_out)
    if from_markers:
        chosen = from_markers.strip()
        diag = {"method": "markers", "raw_selected": chosen}
        diag.update(candidate_quality(original_text, chosen))
        return chosen, diag

    # 2) split and score
    candidates = split_into_blocks(raw_out)
    if not candidates:
        return raw_out.strip(), {"method": "whole_raw", "raw_selected": raw_out.strip()}

    scored = []
    for c in candidates:
        q = candidate_quality(original_text, c)
        scored.append((q["matched"], q["total"], q["has_code_ok"], q["backticks_ok"], q["latex_ok"], q["len"], c, q))

    # sort: by matched desc, then has_code_ok True, then backticks_ok, then latex_ok, then shorter length prefer (to avoid huge repeated prompt)
    scored.sort(key=lambda t: (t[0], t[2], t[3], t[4], -t[5]), reverse=True)
    # pick top
    top = scored[0]
    chosen = top[6]
    diag = {"method": "best-block", "score_tuple": top[:6]}
    diag.update(top[7])
    return chosen.strip(), diag

# -------------------------
# validation
# -------------------------
def validate_placeholders(original_text: str, translated_text: str) -> Tuple[bool, str]:
    """
    Validate that double-square placeholders are preserved (tolerant to whitespace).
    Also check code fences/backticks if present.
    Returns (ok, reason)
    """
    orig_ph = find_placeholders(original_text)
    cand_ph_raw = PH_PATTERN.findall(translated_text)
    cand_ph = set([f"[[{p.strip()}]]" for p in cand_ph_raw])

    missing = [ph for ph in orig_ph if ph not in cand_ph]
    if missing:
        return False, f"missing_placeholders: {missing}"

    # code fences/backticks
    if "```" in original_text and "```" not in translated_text:
        return False, "missing_code_fence"
    if "`" in original_text and "`" not in translated_text:
        return False, "missing_backtick"

    return True, "ok"

# -------------------------
# file writers & logging
# -------------------------
def write_translated_file(dst_dir: Path, original_file: Path, translated_text: str, overwrite: bool = True) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / original_file.name
    if dst_file.exists() and not overwrite:
        fallback = dst_dir / f"{original_file.stem}.fallback{original_file.suffix}"
        fallback.write_text(translated_text, encoding="utf-8")
        return fallback
    dst_file.write_text(translated_text, encoding="utf-8")
    return dst_file

def save_raw(raw_dir: Path, original_file: Path, raw_text: str, diag: Dict):
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "diagnostic": diag,
        "raw": raw_text
    }
    dst = raw_dir / f"{original_file.stem}.raw.json"
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst

# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=False, help="Dataset prompt subfolder under prompts_root")
    parser.add_argument("--prompts_root", type=str, default="./prompts", help="Root folder for prompts (contains per-dataset subfolders)")
    parser.add_argument("--file", type=str, default="", help="Optional single prompt file to translate")
    parser.add_argument("--output_dir", type=str, default="./prompts_translated", help="Where to write translated prompts")
    parser.add_argument("--model_name", type=str, default=os.getenv("LLM_MODEL", ""), help="Model name passed to OpenAIModel wrapper")
    parser.add_argument("--max_new_tokens", type=int, default=512)
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

    # optional dataset-specific template
    dataset_template = None
    if args.dataset_name:
        candidate = prompts_root / args.dataset_name / "translation_bahasa_prompts.txt"
        if candidate.exists():
            dataset_template = candidate.read_text(encoding="utf-8")
            print(f"[translate_prompts] Using dataset-specific template: {candidate}")

    model_name_env = args.model_name or os.getenv("LLM_MODEL", "")
    if not model_name_env:
        print("[translate_prompts] No model specified via --model_name or LLM_MODEL env. Exiting.")
        sys.exit(3)

    openai_model = OpenAIModel(API_KEY=os.getenv("OPENAI_API_KEY", ""),
                               model_name=model_name_env,
                               stop_words=os.getenv("STOP_WORDS", "------"),
                               max_new_tokens=args.max_new_tokens,
                               base_url=os.getenv("BASE_URL", None))

    dst_base = Path(args.output_dir) / (args.dataset_name or "single_file")
    raw_dir = ROOT / "logs" / "raw"
    failed = []

    for p in tqdm(files, desc="Translating prompts"):
        try:
            original_text = p.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[translate_prompts] Failed to read {p}: {e}")
            failed.append(p)
            continue

        masked, mapping = mask_placeholders(original_text)
        if dataset_template and "{prompt_text}" in dataset_template:
            prompt_for_model = dataset_template.replace("{prompt_text}", masked)
        elif dataset_template:
            prompt_for_model = dataset_template + "\n\n### SOURCE PROMPT ###\n\n" + masked
        else:
            prompt_for_model = default_prompt_template(masked)

        raw_out = ""
        try:
            raw_out = openai_model.prompt_generate(prompt_for_model, temperature=args.temperature)
            if raw_out is None:
                raw_out = ""
        except Exception as e:
            print(f"[translate_prompts] Error calling model for {p}: {e}")
            # single retry with tiny change
            try:
                raw_out = openai_model.prompt_generate(prompt_for_model, temperature=max(0.0, args.temperature - 0.1))
            except Exception as e2:
                print(f"[translate_prompts] Retry failed for {p}: {e2}")
                raw_out = ""
                failed.append(p)
                # still save raw attempt
                save_raw(raw_dir, p, raw_out, {"error": str(e2)})
                continue

        # choose candidate
        chosen, diag = choose_best_candidate(raw_out, masked)
        # unmask placeholders
        restored = unmask_placeholders(chosen, mapping)

        # validate
        ok, reason = validate_placeholders(original_text, restored)
        diag["validation_reason"] = reason
        # save raw+diag for debugging
        try:
            save_raw(raw_dir, p, raw_out, diag)
        except Exception:
            pass

        if not ok:
            print(f"[translate_prompts] Initial validation FAILED for {p}. reason={reason}")
            # Save original as fallback (use fallback naming if file exists and overwrite False)
            try:
                fallback_written = write_translated_file(Path(dst_base), p, original_text, overwrite=args.overwrite)
                print(f"[translate_prompts] Saved original as fallback: {fallback_written}")
            except Exception as e:
                print(f"[translate_prompts] Failed to write fallback original for {p}: {e}")
            failed.append(p)
            continue

        # write translated output
        try:
            written = write_translated_file(Path(dst_base), p, restored, overwrite=args.overwrite)
            print(f"[translate_prompts] Wrote translated: {written}")
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
