"""
This script ONLY generates .pl and .json files.
It does NOT execute Prolog.
"""

import argparse
import json
import os
import re
from typing import List, Tuple
from utils import sanitize_filename

# Matches: Predicate(Arg, True|False)
PRED_RE = re.compile(r"([A-Za-z_]\w*)\s*\(\s*(\$?[A-Za-z_]\w*)\s*,\s*(True|False)\s*\)")

# -----------------------------
# Parsing helpers
# -----------------------------

def norm_atom(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s).lower()

def parse_predicates_from_text(text: str) -> List[Tuple[str, str, bool]]:
    out = []
    if not text:
        return out
    for m in PRED_RE.finditer(text):
        pred = norm_atom(m.group(1))
        arg = norm_atom(m.group(2))
        pol = m.group(3) == "True"
        out.append((pred, arg, pol))
    return out

def parse_rules(rule_block: str) -> List[Tuple[str, bool, str, bool]]:
    rules = []
    if not rule_block:
        return rules
    for line in rule_block.splitlines():
        line = line.strip()
        if not line or ">>>" not in line:
            continue
        left, right = line.split(">>>", 1)
        left_pars = parse_predicates_from_text(left)
        right_pars = parse_predicates_from_text(right)
        if len(left_pars) != 1 or len(right_pars) != 1:
            continue
        lp, _, lpol = left_pars[0]
        rp, _, rpol = right_pars[0]
        rules.append((lp, lpol, rp, rpol))
    return rules

def fact_to_prolog(pred: str, subj: str, pol: bool) -> str:
    return (
        f"{pred}({subj}).\n"
        if pol
        else f"neg_{pred}({subj}).\n"
    )

def rule_to_prolog(lp: str, lpol: bool, rp: str, rpol: bool) -> str:
    left = f"{lp}(X)" if lpol else f"neg_{lp}(X)"
    right = f"{rp}(X)" if rpol else f"neg_{rp}(X)"
    return f"{right} :- {left}.\n"

def build_prolog(facts, rules, conjecture):
    lines = []

    # 1. Necessary Headers
    lines.append(":- use_module(library(tabling)).\n")
    lines.append(":- style_check(-discontiguous).\n")
    lines.append(":- style_check(-singleton).\n\n")

    # 2. Extract every single predicate used
    all_preds = set()
    for p, s, pol in facts:
        all_preds.add(p)
    for lp, lpol, rp, rpol in rules:
        all_preds.add(lp)
        all_preds.add(rp)
    if conjecture:
        p, s, pol = conjecture
        all_preds.add(norm_atom(p))
    
    sorted_preds = sorted(list(all_preds))
    
    if sorted_preds:
        # FIXES TIMEOUT: Tells Prolog to fail silently if a predicate is missing
        dynamic_list = ", ".join([f"{p}/1, neg_{p}/1" for p in sorted_preds])
        lines.append(f":- dynamic {dynamic_list}.\n")
        
        # PREVENTS LOOPS: Handles circular logic (e.g., A :- B. B :- A.)
        table_list = ", ".join([f"{p}/1, neg_{p}/1" for p in sorted_preds])
        lines.append(f":- table {table_list}.\n")
    
    lines.append("\n")

    # 3. Facts
    for p, s, pol in facts:
        lines.append(fact_to_prolog(p, s, pol))
    
    lines.append("\n")

    # 4. Rules
    for lp, lpol, rp, rpol in rules:
        lines.append(rule_to_prolog(lp, lpol, rp, rpol))

    lines.append("\n")

    # 5. Main with Error Catching
    if conjecture is None:
        lines.append(":- initialization((format('PROLOG_RESULT|false|false|false~n'), halt)).\n")
        return "".join(lines)

    pred, subj, pol = conjecture
    pred, subj = norm_atom(pred), norm_atom(subj)

    pt_call = f"{pred}({subj})" if pol else f"neg_{pred}({subj})"
    pf_call = f"neg_{pred}({subj})" if pol else f"{pred}({subj})"

    main = f"""
:- initialization(main).

main :-
    % catch/3 ensures that even a runtime error won't hang the script
    catch((
        ( once({pt_call}) -> PT=true ; PT=false ),
        ( once({pf_call}) -> PF=true ; PF=false ),
        ( PT==true, PF==true -> Conflict=true ; Conflict=false ),
        format('PROLOG_RESULT|~w|~w|~w~n', [PT, PF, Conflict])
    ), _Error, (
        % If an error happens, we still report false and exit
        format('PROLOG_RESULT|false|false|false~n')
    )),
    halt.
"""
    lines.append(main)
    return "".join(lines)

def extract_fields(record: dict):
    tc = record.get("translated_context", {}) or {}

    facts = parse_predicates_from_text(
        tc.get("Translated_Facts", "")
        or record.get("Translated_Facts", "")
    )

    rules = parse_rules(
        tc.get("Translated_Rules", "")
        or record.get("Translated_Rules", "")
    )

    conj = parse_predicates_from_text(
        tc.get("Translated_Conjecture", "")
        or record.get("Translated_Conjecture", "")
    )

    conjecture = conj[0] if conj else None
    return facts, rules, conjecture

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--swipl_path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_json = os.path.join(
        args.data_path,
        args.dataset_name,
        f"{sanitize_filename(args.model_name)}_trans_only.json",
    )

    output_dir = os.path.join(
        args.save_path,
        args.dataset_name,
        sanitize_filename(args.model_name),
        "prolog_files",
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            d.pop("translation_process", None)

    for record in data:
        rid = sanitize_filename(str(record.get("id", "no_id")))
        out_pl = os.path.join(output_dir, f"{rid}.pl")
        out_json = os.path.join(output_dir, f"{rid}.json")

        if os.path.exists(out_pl) and not args.overwrite:
            continue

        facts, rules, conjecture = extract_fields(record)
        pl_text = build_prolog(facts, rules, conjecture)

        with open(out_pl, "w", encoding="utf-8") as g:
            g.write(pl_text)

        with open(out_json, "w", encoding="utf-8") as j:
            json.dump(record, j, ensure_ascii=False)

        print("Generated", rid)

if __name__ == "__main__":
    main()
