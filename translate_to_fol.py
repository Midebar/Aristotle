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
        if args.base_url:
            self.openai_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens, base_url=args.base_url)
        else:
            self.openai_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)
            
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
        def _clean_lead(s: str) -> str:
            return re.sub(r'^[\s:\-]*', '', (s or "").strip())

        def _rule_count_and_lines(txt: str):
            lines = [ln for ln in re.split(r'\r?\n', txt or "") if ln.strip()]
            return len(lines), lines

        def _search_complete_predicate(s: str) -> bool:
            if not s or not s.strip():
                return False
            s = s.strip()
            if s.count('(') != s.count(')'):
                return False
            if re.search(r'\b\w+\s*\([^()]*\b(True|False)\b[^()]*\)', s, re.IGNORECASE):
                return True
            if s.endswith(')'):
                return True
            if '(' in s and ')' in s:
                return True
            return False

        content = content or ""

        # Collect final-form blocks
        final_block_pattern = r'\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*\s*(.*?)(?=(\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*)|$)'
        final_blocks = re.findall(final_block_pattern, content, flags=re.DOTALL | re.IGNORECASE)
        final_blocks_text = [b[0] for b in final_blocks]  # list in occurrence order

        if not final_blocks_text:
            final_blocks_text = [content]

        # print("FINAL BLOCKS TEXT:", final_blocks_text)

        # Determine starting index based on prompts file name
        filename = os.path.basename(getattr(self, "prompts_file", "") or "")
        name, _ = os.path.splitext(filename)
        if "modified" in name.lower():
            final_block_index = 5
        else:
            final_block_index = 4

        # scan from first, skip to the element, use reversed if model is "instruct", for block in reversed(final_blocks_text)
        for block in (final_blocks_text[final_block_index:]):
            print(f"BLOCK:\n{block}\n---END BLOCK---")
            fact_iter = list(re.finditer(
                r'Fakta\s*[:\-]?\s*(.*?)(?=(Aturan\s*[:\-]?)|(Konjektur\s*[:\-]?)|$)',
                block, re.DOTALL | re.IGNORECASE))
            rule_iter = list(re.finditer(
                r'Aturan\s*[:\-]?\s*(.*?)(?=(Konjektur\s*[:\-]?)|(Fakta\s*[:\-]?)|$)',
                block, re.DOTALL | re.IGNORECASE))
            conj_iter = list(re.finditer(
                r'Konjektur\s*[:\-]?\s*(.*?)(?=(Fakta\s*[:\-]?)|(Aturan\s*[:\-]?)|$)',
                block, re.DOTALL | re.IGNORECASE))

            if not rule_iter:
                continue

            selected_rule = None
            # if isinstance(context_sentence_count, int) and context_sentence_count > 0:
            #     for m in reversed(rule_iter):
            #         rtxt = _clean_lead(m.group(1))
            #         cnt, _ = _rule_count_and_lines(rtxt)
            #         if cnt == context_sentence_count:
            #             selected_rule = m
            #             break

            # if selected_rule is None:
            #     for m in reversed(rule_iter):
            #         rtxt = _clean_lead(m.group(1))
            #         cnt, _ = _rule_count_and_lines(rtxt)
            #         if cnt > 0:
            #             selected_rule = m
            #             break

            # safety
            if selected_rule is None:
                selected_rule = rule_iter[-1]

            rule_text = _clean_lead(selected_rule.group(1)) if selected_rule else ""

            rule_start = selected_rule.start() if selected_rule else 0
            preceding_facts = [m for m in fact_iter if m.start() <= rule_start]
            if preceding_facts:
                fact_text = _clean_lead(preceding_facts[-1].group(1))
            else:
                fact_text = _clean_lead(fact_iter[-1].group(1)) if fact_iter else ""

            rule_end = selected_rule.end() if selected_rule else 0
            following_conjs = [m for m in conj_iter if m.start() >= rule_end]
            if following_conjs:
                conj_text = _clean_lead(following_conjs[0].group(1))
            else:
                conj_text = _clean_lead(conj_iter[-1].group(1)) if conj_iter else ""

            if conj_text:
                conj_text = re.split(r'\n-{3,}\n|-{5,}|-----|', conj_text, maxsplit=1)[0].strip()

            fact_ok = _search_complete_predicate(fact_text)
            conj_ok = _search_complete_predicate(conj_text)

            # If valid, return this triple
            if fact_ok and conj_ok:
                # normalize some arrow variants
                rule = re.sub(r'\s*(-)\s*', ' \n ', rule) # Sometimes they list '-' as proofs
                rule_text = re.sub(r'\s*(→|⇒|=>|->>|->|=>|—|-)\s*', ' >>> ', rule_text)
                rule_text = re.sub(r'\s*(\<\-\>|\<\=\>|\<\-\=\>)\s*', ' <-> ', rule_text)
                return fact_text, rule_text, conj_text

        # fallback
        # Use the last block in original order (final_blocks_text[-1])
        fallback_block = final_blocks_text[-1] if final_blocks_text else content

        fm = list(re.finditer(r'Fakta\s*[:\-]?\s*(.*?)(?=(Aturan\s*[:\-]?)|(Konjektur\s*[:\-]?)|$)',
                            fallback_block, re.DOTALL | re.IGNORECASE))
        rm = list(re.finditer(r'Aturan\s*[:\-]?\s*(.*?)(?=(Konjektur\s*[:\-]?)|(Fakta\s*[:\-]?)|$)',
                            fallback_block, re.DOTALL | re.IGNORECASE))
        cm = list(re.finditer(r'Konjektur\s*[:\-]?\s*(.*?)(?=(Fakta\s*[:\-]?)|(Aturan\s*[:\-]?)|$)',
                            fallback_block, re.DOTALL | re.IGNORECASE))

        fact = _clean_lead(fm[-1].group(1)) if fm else ""
        rule = _clean_lead(rm[-1].group(1)) if rm else ""
        conjecture = _clean_lead(cm[-1].group(1)) if cm else ""

        rule = re.sub(r'\s*(-)\s*', ' \n ', rule)
        rule = re.sub(r'\s*(→|⇒|=>|->>|->|=>|—|-)\s*', ' >>> ', rule)
        rule = re.sub(r'\s*(\<\-\>|\<\=\>|\<\-\=\>)\s*', ' <-> ', rule)

        return fact, rule, conjecture

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
        responses_a = self.openai_api.generate(prompts_a)
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
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--prompts_folder', type=str, default='./manual_prompts_translated')
    parser.add_argument('--prompts_file', default='translation_modified.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()
