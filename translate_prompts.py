"""
translate_prompts.py (masking with backticks + repair/unmasking)

Usage examples:
python translate_prompts.py --file ./prompts/ProntoQA/and_or_decomposer.txt --overwrite
python translate_prompts.py --dataset_name ProntoQA --prompts_root ./prompts --overwrite

Notes:
- Placeholders like [[PREMISES]] are masked to backticked tokens like `__PH_0__`.
- The model is instructed to preserve backticked tokens exactly.
- If repair fails, we attempt contextual insertion and then append a placeholder block.
"""
import os
import sys
import argparse
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

# Import wrapper that delegates to your backends (hf/ollama/openai)
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

# Placeholder regex (tolerant to whitespace inside the brackets)
PH_PATTERN = re.compile(r'\[\[\s*([^\]]+?)\s*\]\]')

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

def find_placeholders(text: str) -> List[str]:
    out = []
    for m in PH_PATTERN.finditer(text):
        inner = m.group(1).strip()
        out.append(f"[[{inner}]]")
    return sorted(set(out))

# --- MASKING: produce backticked masks like `__PH_0__` ---
def mask_placeholders_with_backticks(text: str) -> Tuple[str, Dict[str,str]]:
    mapping: Dict[str,str] = {}
    idx = 0
    def repl(m):
        nonlocal idx
        inner = m.group(1).strip()
        mask = f"__PH_{idx}__"
        masked_token = f"`{mask}`"  # crucial: keep backticks
        mapping[mask] = f"[[{inner}]]"
        idx += 1
        return masked_token
    masked = PH_PATTERN.sub(repl, text)
    return masked, mapping

# Unmask: replace backticked masks `__PH_0__` -> original [[TOKEN]],
# but be tolerant: if the model left original [[TOKEN]] we keep them as-is.
def unmask_backticks_and_brackets(text: str, mapping: Dict[str,str]) -> str:
    # Replace backticked masks first (explicit)
    for mask, orig in mapping.items():
        # backticked forms: `__PH_0__` or __PH_0__ (if model removed backticks)
        text = text.replace(f"`{mask}`", orig)
        text = text.replace(mask, orig)
    return text

def default_prompt_template(prompt_text: str) -> str:
    # Instruction explicitly tells model to preserve backticked masks exactly.
    prompt = f"""
        You are a professional translator translating prompt templates for dataset-processing pipelines.
        Translate the following prompt text (natural-language text) into Bahasa Indonesia.

        CRITICAL REQUIREMENTS:
        - Produce ONLY the translated prompt text and NOTHING else.
        - Place the translated prompt EXACTLY between the two marker lines below (markers MUST appear):
        {START_MARKER}
        <translated prompt text here>
        {END_MARKER}
        - Do NOT translate boolean values like True, False, yes, no, etc. Leave them as-is.
        - We HAVE MASKED placeholder tokens as inline code/backticks (for example `__PH_0__`). You MUST preserve those backticked tokens exactly (do NOT translate, alter, or remove the backticks or the masks).
        - Do NOT output the original double-square placeholders like [[...]] unless they already appear in your input (we prefer the backticked masks).
        - Preserve code fences (```...```), inline backticks `...`, LaTeX delimiters \\(...\\), \\[...\\], and variable tokens like $x, {{var}}, <tag>.
        - Do NOT add commentary, headers, or any extra text outside the markers.
        - If you cannot produce a safe translation that preserves the backticked masks, output exactly:
        {START_MARKER}
        ERROR
        {END_MARKER}

        Original (masked) prompt content below:
        {prompt_text}

        Now output ONLY the translated prompt text between the markers. Nothing else.
        """.strip()
    return prompt

# Extraction helpers
def extract_between_markers(raw: str) -> Optional[str]:
    pattern = re.compile(re.escape(START_MARKER) + r'(.*?)' + re.escape(END_MARKER), re.S)
    matches = pattern.findall(raw)
    if not matches:
        return None
    return matches[-1].strip()

def split_into_blocks(raw: str) -> List[str]:
    parts = [p.strip() for p in re.split(r'\n{2,}', raw) if p.strip()]
    if not parts:
        parts = [l.strip() for l in raw.splitlines() if l.strip()]
    return parts

def candidate_quality(original_text: str, candidate: str) -> Dict:
    orig_ph = find_placeholders(original_text)
    # count matched placeholders (either original [[...]] or masked `__PH_i__` forms)
    cand_ph_brackets = set(find_placeholders(candidate))
    cand_ph_masks = set(re.findall(r'`(__PH_\d+?)`', candidate))
    # map masked tokens back to original placeholders if possible in original_text? we'll count both
    matched = 0
    for ph in orig_ph:
        if ph in cand_ph_brackets:
            matched += 1
    # Also, if candidate contains masks we expect (we can't directly map mask -> placeholder here).
    # We'll prefer candidates that contain any `__PH_` tokens (higher preservation signal).
    has_mask = bool(re.search(r'`__PH_\d+?`', candidate))
    return {
        "matched_brackets": len([ph for ph in orig_ph if ph in cand_ph_brackets]),
        "has_mask": has_mask,
        "len": len(candidate),
        "candidate": candidate
    }

def choose_best_candidate(raw_out: str, original_text: str) -> Tuple[str, Dict]:
    # 1) If model used markers -> take last marker block
    from_markers = extract_between_markers(raw_out)
    if from_markers:
        diag = {"method": "markers"}
        diag.update(candidate_quality(original_text, from_markers))
        return from_markers.strip(), diag

    # 2) Otherwise split into blocks and pick the one that either contains backticked masks or most bracket placeholders
    candidates = split_into_blocks(raw_out)
    scored = []
    for c in candidates:
        q = candidate_quality(original_text, c)
        score = (q["matched_brackets"], 1 if q["has_mask"] else 0, -q["len"])
        scored.append((score, c, q))
    if not scored:
        return raw_out.strip(), {"method": "whole_raw", "candidate": raw_out.strip()}
    scored.sort(reverse=True)
    best = scored[0]
    return best[1].strip(), {"method": "best_block", **best[2]}

# Validation: accept either masked `__PH_i__` (backticked) or original [[TOKEN]] existing in result
def validate_placeholders(original_text: str, translated_text: str, mapping: Dict[str,str]) -> Tuple[bool, str, List[str]]:
    orig_ph = find_placeholders(original_text)
    # check for either original or masked tokens
    found = set()
    # original bracket placeholders present?
    for ph in orig_ph:
        if ph in translated_text:
            found.add(ph)
    # masked tokens present? map mask -> original if mapping exists
    for mask, orig in mapping.items():
        if f"`{mask}`" in translated_text or mask in translated_text:
            found.add(orig)
    missing = [ph for ph in orig_ph if ph not in found]
    if missing:
        return False, f"missing_placeholders: {missing}", missing
    # code/backtick checks (if original had them)
    if "```" in original_text and "```" not in translated_text:
        return False, "missing_code_fence", []
    if "`" in original_text and "`" not in translated_text:
        # if original had inline backticks and translated removed them, warning
        return False, "missing_backtick", []
    return True, "ok", []

# Save & write helpers
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
    data = {"diag": diag, "raw": raw_text}
    dst = raw_dir / f"{original_file.stem}.raw.json"
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst

# Repair attempt: ask model to fix *masks* (not original [[...]])
def attempt_repair_once(openai_model: OpenAIModel, chosen_masked: str, missing_placeholders: List[str], mapping: Dict[str,str], temperature: float) -> Tuple[str, Dict]:
    # Convert missing original placeholders to the corresponding masks (if possible)
    mask_list = []
    for ph in missing_placeholders:
        # find mask corresponding to this original placeholder
        found = [m for m,o in mapping.items() if o == ph]
        if found:
            mask_list.append(f"`{found[0]}`")
        else:
            # fallback: include original bracketed placeholder (model may accept it)
            mask_list.append(ph)
    masks_str = ", ".join(mask_list)
    repair_instruction = f"""
        You produced a translation but omitted some required masked tokens. Re-insert the missing masked tokens (verbatim) into the translation in appropriate places.
        Important:
        - We masked placeholders with backticked tokens such as `__PH_0__`. You MUST re-insert those backticked tokens exactly.
        - Do NOT output any explanation. Output ONLY the corrected translated text between markers {START_MARKER} and {END_MARKER}.

        Translation to repair:
        {START_MARKER}
        {chosen_masked}
        {END_MARKER}

        Missing tokens to re-insert (verbatim): {masks_str}

        Now produce the corrected translation block and nothing else.
        """.strip()
    try:
        raw_repair = openai_model.prompt_generate(repair_instruction, temperature=temperature)
        if raw_repair is None:
            raw_repair = ""
    except Exception as e:
        return "", {"repair_error": str(e)}
    repaired_candidate, diag = choose_best_candidate(raw_repair, chosen_masked)
    return repaired_candidate, {"raw_repair": raw_repair, **diag}

def attempt_repair(openai_model: OpenAIModel, chosen_masked: str, missing_placeholders: List[str], mapping: Dict[str,str], n_attempts: int, temperature: float) -> Tuple[Optional[str], Dict]:
    combined = {"attempts": []}
    for i in range(n_attempts):
        repaired, diag = attempt_repair_once(openai_model, chosen_masked, missing_placeholders, mapping, temperature)
        combined["attempts"].append(diag)
        # check repaired contains masks or originals
        ok_masks = True
        for ph in missing_placeholders:
            # is original present in repaired or its mask present?
            mask_for_ph = [m for m,o in mapping.items() if o == ph]
            has = False
            if ph in repaired:
                has = True
            if mask_for_ph and f"`{mask_for_ph[0]}`" in repaired:
                has = True
            if not has:
                ok_masks = False
                break
        if ok_masks:
            combined["success_attempt"] = i + 1
            return repaired, combined
    return None, combined

# Contextual insertion (best-effort)
def insert_placeholders_by_context(original: str, restored: str, missing_placeholders: List[str]) -> Tuple[str, Dict]:
    new_text = restored
    diag = {"insertions": []}
    for ph in missing_placeholders:
        # find approximate location in original
        pos = original.find(ph)
        if pos == -1:
            # append if not found
            new_text = new_text + "\n\n" + ph
            diag["insertions"].append({"placeholder": ph, "method": "append"})
            continue
        # get left context words
        left = original[max(0, pos-60):pos]
        anchors = re.findall(r'\w{3,}', left)
        inserted = False
        if anchors:
            anchor = anchors[-1]
            m = re.search(r'\b' + re.escape(anchor) + r'\b', new_text)
            if m:
                insert_at = m.start()
                new_text = new_text[:insert_at] + ph + " " + new_text[insert_at:]
                diag["insertions"].append({"placeholder": ph, "method": "anchor_before", "anchor": anchor})
                inserted = True
        if not inserted:
            new_text = new_text + "\n\n" + ph
            diag["insertions"].append({"placeholder": ph, "method": "append_fallback"})
    return new_text, diag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--prompts_root", type=str, default="./prompts")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./prompts_translated")
    parser.add_argument("--model_name", type=str, default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--no-mask", action="store_true", help="Do not mask placeholders (not recommended)")
    parser.add_argument("--repair-attempts", type=int, default=3)
    args = parser.parse_args()

    load_env_file(Path(args.env))
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

    print(f"[translate_prompts] Found {len(files)} prompt files. Translating...")
    dataset_template = None
    if args.dataset_name:
        candidate = Path(args.prompts_root) / args.dataset_name / "translation_bahasa_prompts.txt"
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

        # mask placeholders (unless user requested no-mask)
        if args.no_mask:
            masked = original_text
            mapping = {}
        else:
            masked, mapping = mask_placeholders_with_backticks(original_text)

        # prepare prompt for model
        if dataset_template and "{prompt_text}" in dataset_template:
            prompt_for_model = dataset_template.replace("{prompt_text}", masked)
        elif dataset_template:
            prompt_for_model = dataset_template + "\n\n### SOURCE PROMPT ###\n\n" + masked
        else:
            prompt_for_model = default_prompt_template(masked)

        # call model (single retry)
        raw_out = ""
        try:
            raw_out = openai_model.prompt_generate(prompt_for_model, temperature=args.temperature)
            if raw_out is None:
                raw_out = ""
        except Exception as e:
            print(f"[translate_prompts] Error calling model for {p}: {e}")
            try:
                clarifier = "\n\n(Please output only the translated prompt text inside the markers and nothing else.)"
                raw_out = openai_model.prompt_generate(prompt_for_model + clarifier, temperature=args.temperature)
                if raw_out is None:
                    raw_out = ""
            except Exception as e2:
                print(f"[translate_prompts] Retry failed for {p}: {e2}")
                raw_out = ""
                failed.append(p)
                save_raw(raw_dir, p, raw_out, {"error": str(e2)})
                continue

        # choose candidate
        chosen_masked, diag = choose_best_candidate(raw_out, masked)
        diag["initial_raw"] = raw_out

        # unmask (prefer backticked masks -> original placeholders)
        restored = unmask_backticks_and_brackets(chosen_masked, mapping)

        # validate
        ok, reason, missing = validate_placeholders(original_text, restored, mapping)
        diag["validation_reason"] = reason
        try:
            save_raw(raw_dir, p, raw_out, diag)
        except Exception:
            pass

        if not ok and missing:
            print(f"[translate_prompts] Initial validation FAILED for {p}. reason={reason}. Attempting repairs...")
            repaired_masked, repair_diag = attempt_repair(openai_model, chosen_masked, missing, mapping, args.repair_attempts, args.temperature)
            diag["repair_diag"] = repair_diag
            if repaired_masked:
                repaired_unmasked = unmask_backticks_and_brackets(repaired_masked, mapping)
                ok2, reason2, missing2 = validate_placeholders(original_text, repaired_unmasked, mapping)
                diag["repair_validation_reason"] = reason2
                try:
                    save_raw(raw_dir, p, repaired_masked, {"after_repair": diag})
                except Exception:
                    pass
                if ok2:
                    try:
                        written = write_translated_file(dst_base, p, repaired_unmasked, overwrite=args.overwrite)
                        print(f"[translate_prompts] Repaired and wrote translated: {written}")
                    except Exception as e:
                        print(f"[translate_prompts] Failed writing repaired for {p}: {e}")
                        failed.append(p)
                    continue
                else:
                    print(f"[translate_prompts] Repair attempt did not fix validation: {reason2}")

            # contextual insertion attempt
            print(f"[translate_prompts] Repair failed; trying contextual insertion for missing placeholders.")
            inserted_text, insert_diag = insert_placeholders_by_context(original_text, restored, missing)
            ok3, reason3, missing3 = validate_placeholders(original_text, inserted_text, mapping)
            diag["contextual_insertion"] = insert_diag
            diag["post_insertion_validation"] = reason3
            try:
                save_raw(raw_dir, p, json.dumps({"post_insertion": diag}, ensure_ascii=False), diag)
            except Exception:
                pass
            if ok3:
                try:
                    written = write_translated_file(dst_base, p, inserted_text, overwrite=args.overwrite)
                    print(f"[translate_prompts] Context insertion succeeded; wrote translated: {written}")
                except Exception as e:
                    print(f"[translate_prompts] Failed to write insertion result for {p}: {e}")
                    failed.append(p)
                continue

            # final fallback: append clear placeholder block so nothing silently lost
            appended = restored + "\n\n/* MISSING_PLACEHOLDERS: " + ", ".join(missing) + " */\n" + "\n".join(missing)
            try:
                written = write_translated_file(dst_base, p, appended, overwrite=args.overwrite)
                print(f"[translate_prompts] Repair/insertion failed; wrote translation + appended placeholder block: {written}")
            except Exception as e:
                print(f"[translate_prompts] Failed to write appended fallback for {p}: {e}")
                failed.append(p)
            continue

        if not ok:
            # other validation issue (e.g., backticks/code fences)
            print(f"[translate_prompts] Validation FAILED for {p}. reason={reason}. Saving original as fallback.")
            try:
                fallback_written = write_translated_file(dst_base, p, original_text, overwrite=args.overwrite)
                print(f"[translate_prompts] Saved original as fallback: {fallback_written}")
            except Exception as e:
                print(f"[translate_prompts] Failed to write fallback original for {p}: {e}")
            failed.append(p)
            continue

        # success -> write restored output
        try:
            written = write_translated_file(dst_base, p, restored, overwrite=args.overwrite)
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
