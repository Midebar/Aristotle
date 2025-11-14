#translate_dataset.py
import os
import json
import argparse
import re
import traceback
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import ModelWrapper, sanitize_filename

ROOT = Path(__file__).parent.resolve()


class Translator:
    EXPECTED_KEYS = ["id", "context", "question", "options", "answer", "explanation"]

    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sample_pct = args.sample_pct
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.batch_num = args.batch_num
        self.prompts_folder = args.prompts_folder
        self.prompts_file = args.prompts_file
        self.stop_words = args.stop_words
        self.max_new_tokens = args.max_new_tokens
        self.file_lock = threading.Lock()
        self.model_api = ModelWrapper(self.model_name, self.stop_words, self.max_new_tokens)

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.prompts_folder, exist_ok=True)

    def load_raw_dataset(self, split: str, sample_pct: int) -> List[Dict[str, Any]]:
        path = os.path.join(self.data_path, self.dataset_name, f'{split}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        if not isinstance(raw_dataset, list):
            raise ValueError("Expected dataset JSON to be a list of datapoints")
        n_keep = max(1, int(len(raw_dataset) * sample_pct / 100))
        return raw_dataset[:n_keep]

    def load_prompt_template(self) -> str:
        dataset_folder = os.path.join(self.prompts_folder, self.dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        file_path = os.path.join(dataset_folder, f"{self.prompts_file}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def construct_prompt(self, record: Dict[str, Any], template: str) -> str:
        explanations_str = "\n".join(record.get("explanation", []))
        t = template
        t = t.replace('[[CONTEXT]]', record.get("context", ""))
        t = t.replace('[[QUESTION]]', record.get("question", ""))
        t = t.replace('[[EXPLANATIONS]]', explanations_str)
        return t

    def _call_model_single(self, prompt: str) -> str:
        out = self.model_api.generate(prompt, task="translation")
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out if isinstance(out, str) else str(out)

    def _normalize_output(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """
        Extract 'context', 'question', and 'explanation(s)' from raw plain-text output.
        Returns dict: {"context": str, "question": str, "explanation": List[str]}
        """
        text = (raw_text or "").strip()
        out = {"context": "", "question": "", "explanation": []}

        marker_pattern = r'Now translate them!'
        marker_match = re.search(marker_pattern, text, flags=re.IGNORECASE)
        search_area = text[marker_match.end():].strip() if marker_match else text

        ctx_match = re.search(r'context\s*:\s*(.*?)(?=\n(?:question|explanation|explanations|context)\s*:|\Z)',
                            search_area, flags=re.IGNORECASE | re.DOTALL)
        q_match = re.search(r'question\s*:\s*(.*?)(?=\n(?:context|explanation|explanations|question)\s*:|\Z)',
                            search_area, flags=re.IGNORECASE | re.DOTALL)
        expl_match = re.search(r'(?:explanations?|explanation)\s*:\s*(.*?)(?=\n(?:context|question|explanation|explanations)\s*:|\Z)',
                            search_area, flags=re.IGNORECASE | re.DOTALL)

        if ctx_match:
            out["context"] = ctx_match.group(1).strip()
        if q_match:
            out["question"] = q_match.group(1).strip()

        if expl_match:
            expl_text = expl_match.group(1).strip()

            parts = []
            #Split by newlines first
            lines = [ln.strip() for ln in re.split(r'[\r\n]+', expl_text) if ln.strip()]

            if len(lines) == 0:
                # nothing found by newline splitting; try splitting by numbered patterns inline
                cand = expl_text
                lines = [s.strip() for s in re.split(r'(?:\d+\.\s+|\d+\)\s+|[•\-\u2022]\s+)', cand) if s and s.strip()]

            # Process each line: remove leading bullets/nums, then further split multi-sentence lines
            for line in lines:
                line = re.sub(r'^[\s]*[-\u2022\•\*]+\s*', '', line)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = line.strip()
                if not line:
                    continue

                # If the line contains multiple sentences, split on sentence boundaries
                subparts = re.split(r'(?<=[\.\?\!])\s+|\s*\|\s*', line)
                if len(subparts) == 1:
                    # also attempt splitting by ";", or " - " connecting clauses if it's long
                    if len(line) > 200:
                        subparts = re.split(r'[;]\s+|\s*-\s*', line)
                for sp in subparts:
                    sp = sp.strip()
                    if not sp:
                        continue
                    # strip any leftover leading numbering
                    sp = re.sub(r'^\d+[\.\)]\s*', '', sp).strip()
                    # ensure punctuation at end
                    if not re.search(r'[\.!\?]$', sp):
                        sp = sp + "."
                    parts.append(sp)

            # remove duplicates and empty strings while preserving order
            seen = set()
            cleaned = []
            for p in parts:
                p_norm = p.strip()
                if not p_norm:
                    continue
                if p_norm in seen:
                    continue
                seen.add(p_norm)
                cleaned.append(p_norm)

            out["explanation"] = cleaned

        return out
        
    def save_output(self, outputs: Any, file_suffix: Optional[str] = None):
        model_name = sanitize_filename(self.model_name)
        file_name = f'{model_name}_translation.json' if file_suffix is None else f'{model_name}_translation_{file_suffix}.json'
        file_path = os.path.join(self.save_path, self.dataset_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with self.file_lock:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        existing = json.loads(content) if content else []
                else:
                    existing = []

                if isinstance(outputs, list):
                    existing.extend(outputs)
                else:
                    existing.append(outputs)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving output to {file_path}: {e}")
                traceback.print_exc()

    def process_record(self, record: Dict[str, Any], template: str) -> Dict[str, Any]:
        try:
            prompt = self.construct_prompt(record, template)
            print(f"Processing record ID: {record.get('id')} with prompt:\n{prompt}\n")

            raw_model_out = self._call_model_single(prompt)
            print(f"Raw model output for record ID {record.get('id')}:\n{raw_model_out}\n")

            normalized = self._normalize_output(raw_model_out)
            print(f"Normalized output for record ID {record.get('id')}:\n{normalized}\n")

            result = {
                "id": record.get("id"),
                "context": normalized.get("context"),
                "question": normalized.get("question"),
                "options": record.get("options", []),
                "explanation": normalized.get("explanation", []),
                "answer": record.get("answer"),
                "process": raw_model_out,
            }
            return result
        except Exception as e:
            print(f"Error processing record {record.get('id')}: {e}")
            traceback.print_exc()

    def generate_translations(self):
        template = self.load_prompt_template()
        raw_dataset = self.load_raw_dataset(self.split, self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        results = []
        with ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            future_to_record = {executor.submit(self.process_record, record, template): record for record in raw_dataset}
            for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="Translating"):
                rec = future_to_record[future]
                try:
                    result = future.result()
                    print(f"Saving output for record: {result}")
                    self.save_output(result)
                    results.append(result)
                except Exception as e:
                    print(f"Exception for record {rec.get('id')}: {e}")
                    traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample_pct', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./results_bahasa_translation')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--prompts_folder', type=str, default='./manual_prompts_translated')
    parser.add_argument('--prompts_file', default='bahasa_translation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    translator = Translator(args)
    translator.generate_translations()
