# naive_prompting.py
import json
import os
from tqdm import tqdm
from utils import ModelWrapper, sanitize_filename
import argparse
import re
import concurrent.futures
import traceback
import threading
from typing import Dict, Any, Optional

class Ablation_Prompting:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.sample_pct = args.sample_pct
        self.start_index = args.start_index
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.batch_num = args.batch_num
        self.prompts_folder = args.prompts_folder
        self.prompts_file = args.prompts_file
        self.prompts_mode = args.prompts_mode
        self.file_lock = threading.Lock()
        self.model_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)

    def load_in_ablation_prompts(self, prompts_folder='./manual_prompts_transated', prompts_file='translations_only_prompting.txt'):
        file_path = os.path.join(prompts_folder, self.dataset_name, f"{prompts_file}.txt")
        print("Loading translation file: ", file_path)
        with open(file_path) as f:
            naive_prompts = f.read()
        return naive_prompts

    def load_raw_dataset(self, sample_pct):
        print(f"SAMPLE PCT: {sample_pct}")
        model_name = sanitize_filename(self.model_name)
        input_path = f'{model_name}_trans_only.json'
        input_path = os.path.join(self.data_path, self.dataset_name, input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
            raw_dataset = raw_dataset[:max(1, int(len(raw_dataset) * sample_pct / 100))]
        return raw_dataset
    
    def construct_prompt_fol(self, record, fol_prompts):
        full_prompt = fol_prompts
        context = record['translated_context']
        facts = context['Translated_Facts'].strip()
        rules = context['Translated_Rules'].strip()
        conjecture = context['Translated_Conjecture'].strip()
                
        full_prompt = full_prompt.replace('[[FACTS]]', facts)
        full_prompt = full_prompt.replace('[[RULES]]', rules)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', conjecture)

        return full_prompt
    
    def extract_answers(self, content: str) -> Dict[str, Any]:
        result = {"answer": None, "explanations": []}

        marker = r'Di bawah ini yang perlu Anda cari nilai kebenarannya:'
        m = re.search(marker, content, flags=re.IGNORECASE)
        area = content[m.end():] if m else content

        ans_label_re = re.compile(r'(?:\*{0,3}Jawaban\*{0,3}|Jawaban)\s*[:\-]?\s*', flags=re.IGNORECASE)
        exp_label_re = re.compile(r'(?:\*{0,3}Penjelasan\*{0,3}|Penjelasan)\s*[:\-]?\s*', flags=re.IGNORECASE)

        # Boundary pattern to detect next label or section end (###)
        boundary_re = re.compile(
            r'\r?\n(?:\*{0,3}Jawaban\*{0,3}|Jawaban|\*{0,3}Penjelasan\*{0,3}|Penjelasan|###)\b',
            flags=re.IGNORECASE
        )

        def extract_after_label(label_re):
            """Find first label match, return the substring after it up to next boundary or end, or None."""
            lab_match = label_re.search(area)
            if not lab_match:
                return None
            start = lab_match.end()
            bound = boundary_re.search(area, pos=start)
            end = bound.start() if bound else len(area)
            return area[start:end].strip()

        raw_answer = extract_after_label(ans_label_re)
        if raw_answer:
            # take first non-empty line
            for line in raw_answer.splitlines():
                s = line.strip()
                if s:
                    # normalize spaces and trailing punctuation/newlines
                    s = re.sub(r'\s{2,}', ' ', s)
                    result["answer"] = s
                    break

        raw_expl = extract_after_label(exp_label_re)
        if raw_expl:
            lines = []
            for raw in raw_expl.splitlines():
                line = raw.strip()
                if not line:
                    continue
                # remove common bullet prefixes (dash, *, numbered, arrows)
                line = re.sub(r'^[\-\*\u2022\>\s0-9\.\)]+', '', line).strip()
                # collapse excessive spaces
                line = re.sub(r'\s{2,}', ' ', line)
                if line:
                    lines.append(line)
            result["explanations"] = lines

        if raw_answer is None:
            result["answer"] = "No final answer found in the text."
        
        if not result["explanations"]:
            result["explanations"] = ["No explanations found in the text."]
            
        print("Extracted answer: ", result["answer"])
        print("Extracted explanations: ", result["explanations"])

        return result

    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        elif final_answer == "self-contradictory":
            final_answer = 'D'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer

    def save_output(self, outputs, file_suffix=None):
        model_name = self.model_name
        model_name = sanitize_filename(model_name)
        file_name = f'{model_name}_naive_prompting.json'
        file_path = os.path.join(self.save_path, self.dataset_name, file_name)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print("Saving result with thread lock in path: ", file_path)
        with self.file_lock:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        if file_content.strip():
                            existing_data = json.loads(file_content)
                        else:
                            print(f"File {file_path} is empty. Initializing with an empty list.")
                            existing_data = []
                else:
                    existing_data = []
                
                if isinstance(outputs, list):
                    existing_data.extend(outputs)
                else:
                    existing_data.append(outputs)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error in saving output: {e}")
    
    def process_record(self, record, naive_prompts):
        try:
            print("Running example id: ", record.get('id', None))
            prompt = self.construct_prompt_fol(record, naive_prompts)
            print(f"\nFOL prompting: {prompt}\n")
            response = self.model_api.generate(prompt)

            if isinstance(response, (list, tuple)):
                #print("Response is a list/tuple, taking the first element.") # ('content', 'stop')
                response = response[0]
            else:
                #print("Response is a single string.")
                response = response

            print(f"\nFOL prompting response: {response}\n")
            extracted = self.extract_answers(response)
            answer = extracted['answer']
            explanations = extracted['explanations']
            final_choice = self.final_process(answer)
            translated_context = record['translated_context']
            
            result = {
                'id': record['id'],
                'original_context': record['original_context'],
                'translated_facts': translated_context['Translated_Facts'],
                'translated_rules': translated_context['Translated_Rules'],
                'translated_conjecture': translated_context['Translated_Conjecture'],
                'answer': answer,
                'final_choice': final_choice,
                'explanations': explanations,
                'ground_truth': record['ground_truth'],
                'response': response,
            }
            return result
        except Exception as e:
            print(f"Error processing record {record['id']}: {e}")
            traceback.print_exc()
            return None
    
    def ablation_prompting_generation(self):
        naive_prompts = self.load_in_ablation_prompts(self.prompts_folder, self.prompts_file)
        raw_dataset = self.load_raw_dataset(self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples")
        print("Number of batch: ", self.batch_num)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            future_to_record = {executor.submit(self.process_record, record, naive_prompts): record for record in raw_dataset}
            for future in tqdm(concurrent.futures.as_completed(future_to_record), total=len(raw_dataset)):
                record = future_to_record[future]
                try:
                    result = future.result()
                    print(f"Saving output for record: {result}")
                    self.save_output(result)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f'{record["id"]} generated an exception: {e}')
                    traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./manual_data_translated')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--sample_pct', type=int, default=100)
    parser.add_argument('--start_index', type=int) # Fix MY SANITY
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--prompts_folder', type=str, default='./manual_prompts_translated')
    parser.add_argument('--prompts_file', default='naive_prompting.txt')
    parser.add_argument('--prompts_mode', default='filtered')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    ablation_prompting = Ablation_Prompting(args)
    ablation_prompting.ablation_prompting_generation()