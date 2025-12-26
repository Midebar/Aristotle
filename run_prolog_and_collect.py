# run_prolog_and_collect.py
"""
Run SWI-Prolog over generated Prolog files and append augmented JSONL results.

Usage example:
python run_prolog_and_collect.py --save_path results_translated_fol_prolog \
  --dataset_name ProntoQA --model_name "$LLM_MODEL" --swipl_path "C:/Program Files/swipl/bin/swipl.exe" \
  --workers 8 --timeout 60
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import sanitize_filename

def parse_result(out: str):
    for line in out.splitlines():
        if line.startswith('PROLOG_RESULT|'):
            parts = line.split('|')
            if len(parts) >= 4:
                pt = parts[1].strip().lower() == 'true'
                pf = parts[2].strip().lower() == 'true'
                c  = parts[3].strip().lower() == 'true'
                return pt, pf, c
    return False, False, False

def apply_policy(pt, pf, conflict):
    if conflict:
        return 'Self-Contradictory', ''
    if pt:
        return True, 'A'
    if pf:
        return False, 'B'
    return 'Unknown', ''

def run_one(pl_file, swipl_path, timeout):
    try:
        proc = subprocess.run([swipl_path, '-q', '-s', pl_file], capture_output=True, text=True, timeout=timeout)
        stdout = proc.stdout or ''
        stderr = proc.stderr or ''
        pt, pf, c = parse_result(stdout)
        return pt, pf, c, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, False, False, '', f'TIMEOUT after {timeout}s'
    except FileNotFoundError:
        return False, False, False, '', f'ERROR: swipl executable not found: {swipl_path}'
    except PermissionError as e:
        return False, False, False, '', f'ERROR: Access denied when running swipl: {e}'
    except Exception as e:
        return False, False, False, '', f'ERROR: {e}'

def process_one(base_dir, filename, swipl_path, timeout):
    stem = filename[:-3]
    json_path = os.path.join(base_dir, f"{stem}.json")
    pl_path = os.path.join(base_dir, filename)
    if not os.path.exists(json_path):
        return None, f"Missing original json for {filename}"
    with open(json_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    pt, pf, c, out, err = run_one(pl_path, swipl_path, timeout)
    final_answer, final_choice = apply_policy(pt, pf, c)
    record.update({
        'prolog_true': bool(pt),
        'prolog_false': bool(pf),
        'conflict': bool(c),
        'final_answer': final_answer,
        'final_choice': final_choice,
        'prolog_raw_output': out,
        'prolog_error': err,
    })
    return record, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--swipl_path', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=60, help='Timeout per swipl call (seconds)')
    args = parser.parse_args()
    
    model_name = sanitize_filename(args.model_name)
    base_dir = os.path.join(args.save_path, args.dataset_name, model_name, 'prolog_files')
    out_file = os.path.join(args.save_path, args.dataset_name, f'{model_name}_naive_prompting.json')

    if not os.path.isdir(base_dir):
        raise SystemExit(f"Prolog files directory not found: {base_dir}")

    files = sorted([f for f in os.listdir(base_dir) if f.endswith('.pl')])
    if not files:
        raise SystemExit(f"No .pl files found in {base_dir}")

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    # do not remove existing output; append as before
    print(f"Running {len(files)} files -> appending to {out_file}")

    if args.workers <= 1:
        for fn in files:
            rec, err = process_one(base_dir, fn, args.swipl_path, args.timeout)
            if rec is None:
                print('Skipping', fn, 'Error:', err)
                continue
            with open(out_file, 'a', encoding='utf-8') as out:
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print('Processed', fn)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, base_dir, fn, args.swipl_path, args.timeout): fn for fn in files}
            for fut in as_completed(futures):
                fn = futures[fut]
                try:
                    rec, err = fut.result()
                except Exception as e:
                    print(f"Error processing {fn}: {e}")
                    rec, err = None, str(e)
                if rec is None:
                    print('Skipping', fn, 'Error:', err)
                    continue
                with open(out_file, 'a', encoding='utf-8') as out:
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                print('Processed', fn)

    print('Saved:', out_file)

if __name__ == '__main__':
    main()
