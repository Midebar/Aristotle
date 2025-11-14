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
from typing import List

class Naive_Prompting:
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
        self.prompts_mode = args.prompts_mode
        self.file_lock = threading.Lock()
        self.model_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)

    def load_in_naive_prompts(self, prompts_folder='./manual_prompts_transated', prompts_file='naive_prompting.txt'):
        file_path = os.path.join(prompts_folder, self.dataset_name, f"{prompts_file}.txt")
        print("Loading translation file: ", file_path)
        with open(file_path) as f:
            naive_prompts = f.read()
        return naive_prompts

    def load_raw_dataset(self, split, sample_pct):
        print(f"SAMPLE PCT: {sample_pct}")
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
            raw_dataset = raw_dataset[:max(1, int(len(raw_dataset) * sample_pct / 100))]
        return raw_dataset
    
    def construct_prompt_naive(self, record, naive_prompts):
        full_prompt = naive_prompts
        if self.dataset_name == "LogicNLI":
            context = "\n".join(record['facts'] + record['rules'])
            question = record['conjecture']
        else:
            context = record['context']
            if self.prompts_mode == 'full':
                question = record['question'].strip()
            else:
                question = re.search(r'\?(.*)', record['question'].strip()).group(1).strip()

        full_prompt = full_prompt.replace('[[PREMISES]]', context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', question)

        return full_prompt
    
    def extract_answers(self, content):
        marker_pattern = r'Di bawah ini yang perlu Anda cari nilai kebenarannya:'
        marker_match = re.search(marker_pattern, content, flags=re.IGNORECASE)
        search_area = content[marker_match.end():] if marker_match else content

        #print(f"\n\nSEARCH AREA:\n\n{search_area}\n")

        answer_block_pattern = (r'\*{0,3}\s*(?:Nilai\s*Kebenaran|Jawaban)\s*\*{0,3}\s*:\s*(.*?)(?=\n|$)')

        answer_block = re.search(answer_block_pattern, search_area, flags=re.DOTALL | re.IGNORECASE)

        print(f"\n\nCHOSEN BLOCK:\n\n{answer_block}\n")
        print("END OF CHOSEN BLOCK\n\n")
        print(f"Group 0: {answer_block.group(0).strip()}")
        print(f"Group 1: {answer_block.group(1).strip()}")
        answer = answer_block.group(1).strip() if answer_block else "No answer found"

        return answer
    
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
                
                with open(file_path, 'w') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error in saving output: {e}")
    
    def process_record(self, record, naive_prompts):
        try:
            print("Running example id: ", record.get('id', None))
            prompt = self.construct_prompt_naive(record, naive_prompts)
            print(f"\nNaive prompting: {prompt}\n")
            response = self.model_api.generate(prompt)

            if isinstance(response, (list, tuple)):
                #print("Response is a list/tuple, taking the first element.") # ('content', 'stop')
                response = response[0]
            else:
                #print("Response is a single string.")
                response = response

            print(f"\nNaive prompting response: {response}\n")
            answer = self.extract_answers(response)
            final_choice = self.final_process(answer)
            
            result = {
                'id': record.get('id', None),
                'prompt': prompt,
                'response': response,
                'answer': answer,
                'final_choice': final_choice,
                'ground_truth': record['answer']
            }
            return result
        except Exception as e:
            print(f"Error processing record {record.get('id', None)}: {e}")
            traceback.print_exc()
            return None
    
    def naive_prompting_generation(self):
        naive_prompts = self.load_in_naive_prompts(self.prompts_folder, self.prompts_file)
        raw_dataset = self.load_raw_dataset(self.split, self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
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
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample_pct', type=int, default=100)
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
    naive_prompting = Naive_Prompting(args)
    naive_prompting.naive_prompting_generation()