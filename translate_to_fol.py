# translate_to_fol.py
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

class Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sample_pct = args.sample_pct
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.batch_num = args.batch_num
        self.prompts_folder = args.prompts_folder
        self.prompts_file = args.prompts_file
        self.file_lock = threading.Lock()
        self.model_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)
            
    def load_in_context_examples_trans(self, prompts_folder='./prompts', prompts_file='translation'):
        file_path = os.path.join(prompts_folder, self.dataset_name, f"{prompts_file}.txt")
        print("Loading translation file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
            
        return in_context_examples
    
    def load_in_context_and_or_decomposer(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'and_or_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split, sample_pct):
        print(f"SAMPLE PCT: {sample_pct}")
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
            raw_dataset = raw_dataset[:max(1, int(len(raw_dataset) * sample_pct / 100))]
        return raw_dataset
        
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        if self.dataset_name == "LogicNLI":
            context = "\n".join(record['facts'] + record['rules'])
            question = record['conjecture']
        else:
            context = record['context']
            question = re.search(r'\?(.*)', record['question'].strip()).group(1).strip()
        full_prompt = full_prompt.replace('[[PREMISES]]', context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', question)
        return full_prompt
    
    def clean_irrelevant_lines(self, content):
        lines = content.split("\n")
        cleaned_lines = [line.strip() for line in lines if "(" in line]
        return "\n".join(cleaned_lines)
    
    def remove_duplicates(self, context_lines):
        seen = set()
        unique_items = []
        
        for item in context_lines:
            if item not in seen:
                unique_items.append(item)
                seen.add(item) 
        return unique_items
    
    def split_cnf_clause(self, context):
        print("Splitting context: ", context)
        normalized_context_lines = context.split('\n')
        unique_context_lines = self.remove_duplicates(normalized_context_lines)
        normalized_context = [line.split('\\land') for line in unique_context_lines]
        flattened_normalized_context = [item for sublist in normalized_context for item in sublist]
        cleaned_normalized_context = [line.split(":::")[0].strip() for line in flattened_normalized_context]
        cleaned_normalized_context = [re.sub(r'^\d+\.\s*', '', line) for line in cleaned_normalized_context]
        renumbered_normalized_context = [f"{index + 1}. {line}" for index, line in enumerate(cleaned_normalized_context)]

        return renumbered_normalized_context
    
    def categorize_rule_lines(self, rule):
        either_or = []
        biconditional = []
        others = []

        for item in rule.split('\n'):
            if "(" in item:
                if '\u2295' in item:
                    either_or.append(item)
                elif '\u21d4' in item:
                    biconditional.append(item)
                else:
                    others.append(item)

        return either_or, biconditional, others
    
    def extract_facts_rules_conjecture(self, content, context_sentence_count=None):
        content = (content or "").replace('\u200b', '').replace('\ufeff', '')

        prompt_marker = re.compile(
            r'Di bawah ini(?:\s+adalah(?:\s+yang\s+perlu\s+Anda\s+terjemahkan)?)?:?',
            re.IGNORECASE
        )
        
        m_prompt = prompt_marker.search(content)
        search_start_pos = m_prompt.end() if m_prompt else 0

        block_header = re.compile(r'\*{0,3}Bentuk Akhir\*{0,3}', re.IGNORECASE)
        m_block = block_header.search(content, pos=search_start_pos)

        area = content[m_block.end():]

        # Define Section Headers
        fact_re = re.compile(r'(?:Fakta|Facts)\s*[:\-]?\s*', re.IGNORECASE)
        rule_re = re.compile(r'(?:Aturan|Rules)\s*[:\-]?\s*', re.IGNORECASE)
        conj_re = re.compile(r'(?:Konjektur|Conjecture)\s*[:\-]?\s*', re.IGNORECASE)

        # 5. Define Boundaries (Lookahead)
        # Stop extracting if a newline followed by:
        boundary_re = re.compile(
            r'\r?\n\s*(?:'
            r'(?:Fakta|Facts)\s*[:\-]?'         r'|' 
            r'(?:Aturan|Rules)\s*[:\-]?'        r'|' 
            r'(?:Konjektur|Conjecture)\s*[:\-]? 'r'|'
            r'(?:\*{0,3}\s*Akhir Blok\s*\*{0,3}|###|```|-{3,})'
            r')',
            flags=re.IGNORECASE
        )

        def extract_section(label_re):
            """Finds label in 'area', extracts text until the nearest boundary."""
            match = label_re.search(area)
            if not match:
                return ""
            
            start_pos = match.end()
            
            # Search for the nearest boundary starting from the end of the header
            bound = boundary_re.search(area, pos=start_pos)
            
            # If boundary found
            end_pos = bound.start() if bound else len(area)
            
            return area[start_pos:end_pos].strip()

        facts = extract_section(fact_re)
        rules = extract_section(rule_re)
        conjecture = extract_section(conj_re)

        return facts, rules, conjecture


    def clean_conjecture(self, conjecture):
        if isinstance(conjecture, dict):
            conjecture = "\n".join([f"{key}: {value}" for key, value in conjecture.items()])
        splitted_conjecture = conjecture.replace('\\n', '\n').split('\n')
        cleaned_conjecture = []
        for item in splitted_conjecture:
            if "(" in item:
                remove_list = ['Rules (others):', 'Rules (biconditional):', 'Rules (either_or):']
                if not any(remove_item in item for remove_item in remove_list):
                    splited_item = item.split(":::")[0]
                    cleaned_conjecture.append(splited_item)
                
        return '\n'.join(cleaned_conjecture)
    
    def save_output(self, outputs, file_suffix=None):
        model_name = self.model_name
        model_name = sanitize_filename(model_name)
        file_name = f'{model_name}_trans_only.json'
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
        
    def process_example(self, example, in_context_examples_trans):
        if self.dataset_name == 'LogicNLI':
            question = example['conjecture']
        else:
            question = example['question'].split('?')[1]
        print("Translating...")
        prompts_a = self.construct_prompt_a(example, in_context_examples_trans)
        print("Translation prompt_a: ", prompts_a)
        responses_a = self.model_api.generate(prompts_a)
        # responses_a might be (text, finish_reason) or a string; normalize to string
        if isinstance(responses_a, (list, tuple)):
            responses_a_text = responses_a[0]
        else:
            responses_a_text = responses_a
        print("Translation response: ", responses_a_text)
        
        # count sentences
        context_text = example.get('context', '') or ''
        raw_sentences = re.split(r'(?<=[.!?])\s+', context_text.strip())
        sentences = [s for s in raw_sentences if s.strip()]
        context_sentence_count = len(sentences)-1 # exclude facts at end of sentence

        translated_facts, translated_rules, translated_conjecture = self.extract_facts_rules_conjecture(responses_a_text, context_sentence_count)
        print("Translated Facts1: ", translated_facts)
        translated_facts = self.clean_irrelevant_lines(translated_facts)
        print(f"Translated Facts2: {translated_facts}")
        translated_conjecture = self.clean_conjecture(translated_conjecture)
        print(f"Translated Conjecture: {translated_conjecture}")

        either_or, biconditional, and_or = self.categorize_rule_lines(translated_rules)
        
        print(f"AND/OR: {and_or} \n EITHER/OR: {either_or} \n BICONDITIONAL: {biconditional}")

        if self.dataset_name == 'LogicNLI':
            original_context = "Facts: " + '\n'.join(example['facts']) + "Rules: " + '\n'.join(example['rules'])
        else:
            original_context = example['context']

        output = {
            'id': example['id'],
            'original_context': original_context,
            'question': question,  
            'and_or': and_or,
            'either_or': either_or,
            'biconditional': biconditional, 
            'translated_context': {"Translated_Facts": translated_facts, "Translated_Rules": translated_rules, "Translated_Conjecture": translated_conjecture},
            'translation_process': {key: value for key, value in {"responses_a": locals().get("responses_a"),}.items() if value is not None},
            'ground_truth': example['answer']
        }
        return output, None
        
    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.split, self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        in_context_examples_trans = self.load_in_context_examples_trans(self.prompts_folder, self.prompts_file)

        print("Number of batch: ", self.batch_num)
        counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            futures = {
                executor.submit(self.process_example, example, in_context_examples_trans): example 
                for example in raw_dataset
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(raw_dataset)):
                example = futures[future]
                try:
                    output = future.result()
                    if 'error' in output:
                        print(f"Error in generating example: {example['id']}")
                    else:
                        print(f"Saving output for example: {output}")
                        self.save_output(output[0])
                except Exception as exc:
                    print(f'{example["id"]} generated an exception: {exc}')
                    traceback.print_exc()
                counter += 1
            
    
    def update_answer(self, sample, translation, decomposed_process, translated_fact, normalized_context, normalized_conjecture, negated_label, sos_list):
        if isinstance(normalized_context, list):
            normalized_context = "\n".join(normalized_context)
        
        output = {'id': sample['id'], 
                'original_context': sample['context'] if self.dataset_name != "LogicNLI" else "\n".join(sample['facts'] + sample['rules']),
                'question': sample['question'] if self.dataset_name != "LogicNLI" else sample['conjecture'], 
                'translated_context': translation,
                'decomposed_process': decomposed_process,
                'normalized_context': "Facts: " + translated_fact + "\nRules: " + normalized_context,
                'normalized_conjecture': normalized_conjecture,
                'negated_label': negated_label,
                'sos_list': sos_list,
                'ground_truth': sample['answer'] if self.dataset_name != "LogicNLI" else sample['label']}
        
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample_pct', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--prompts_folder', type=str, default='./manual_prompts_translated')
    parser.add_argument('--prompts_file', default='translation_modified.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model_problem_reduction = Reasoning_Graph_Baseline(args)
    model_problem_reduction.reasoning_graph_generation()
