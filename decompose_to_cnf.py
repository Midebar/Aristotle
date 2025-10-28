# decompose_to_cnf.py
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
        self.file_lock = threading.Lock()
        self.openai_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)
    
    def load_in_context_and_or_decomposer(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'and_or_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_either_or_decomposer(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'either_or_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_biconditional_decomposer(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'logical_biconditional_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_search_init(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'search_init.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_search_router(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'search_router.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, sample_pct):
        print(f"SAMPLE PCT: {sample_pct}")
        model_name = sanitize_filename(args.model_name)
        input_path = f'{model_name}_trans_only.json'
        input_path = os.path.join(args.save_path, args.dataset_name, input_path)
        with open(input_path, 'r') as f:
            raw_dataset = json.load(f)
            raw_dataset = raw_dataset[:max(1, int(len(raw_dataset) * sample_pct / 100))]
        return raw_dataset
        
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_b(self, responses_a, in_context_examples_decomposer):
        full_prompt = in_context_examples_decomposer
        if isinstance(responses_a, list):
            responses_a = '\n'.join(responses_a)
        full_prompt = full_prompt.replace('[[PREMISES]]', responses_a)
        return full_prompt

    def construct_prompt_c(self, responses_b, in_context_examples_search_init):
        full_prompt = in_context_examples_search_init
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_b)
        return full_prompt
    
    def construct_prompt_d(self, responses_b, negated_label, reasoning_step, sos_list, in_context_examples_search_router):
        full_prompt = in_context_examples_search_router
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_b)
        full_prompt = full_prompt.replace('[[NEGATION-INITIALIZATION]]', negated_label)
        full_prompt = full_prompt.replace('[[REASONING]]', reasoning_step)
        full_prompt = full_prompt.replace('[[SOS]]', sos_list)
        return full_prompt
        
    def post_process_b(self, response_b):
        parts = response_b.split("Final Form:") 
        if len(parts) > 1:
            context_text = parts[-1].strip()
        else:
            context_text = "Context not found."
        
        return context_text
    
    def negate_conjecture(self, conjecture):
        if re.search(r'True', conjecture):
            updated_conjecture = re.sub(r'True', 'False', conjecture)
        elif re.search(r'False', conjecture):
            updated_conjecture = re.sub(r'False', 'True', conjecture)
        else:
            updated_conjecture = conjecture
            print("No True/False found in the conjecture.")

        return updated_conjecture
        
    def post_process_c(self, response_c):
        sos_list = re.findall(r'\[(.*?)\]', response_c)
        negated_label = re.findall(r'\{(.*?)\}', response_c)
        str_sos_list = "".join(sos_list)
        str_negated_label = "".join(negated_label)
        return str_negated_label.lower(), str_sos_list.lower()
    
    def post_process_d(self, response_d):
        search_result_match = re.findall(r'\{(.*?)\}', response_d)
        search_result = search_result_match[0].strip() if search_result_match else None
        
        new_clause_match = re.search(r"New Clause:\s*```(.*?)```", response_d, re.DOTALL)
        new_clause = new_clause_match.group(1).strip() if new_clause_match else None
        
        sufficiency_check_match = re.search(r"Sufficiency check for final answer:\s*(.*)", response_d)
        sufficiency_check = sufficiency_check_match.group(1).strip() if sufficiency_check_match else None
        
        sufficiency_label_match = re.search(r"Sufficiency Label:\s*\[(.*?)\]", response_d)
        sufficiency_label = sufficiency_label_match.group(1).strip() if sufficiency_label_match else None
        
        final_answer = "***Bentuk Akhir: N/A***"
        if sufficiency_label and sufficiency_label.lower() == "true":
            final_answer_match = re.search(r'\*\*\*(.*?)\*\*\*', response_d)
            final_answer = final_answer_match.group(1).strip() if final_answer_match else "No final answer found"
    
        return {
            "Search Result": search_result,
            "New Clause": new_clause,
            "Sufficiency Check": sufficiency_check,
            "Sufficiency Label": sufficiency_label,
            "Final Answer": final_answer
        }
        
    def process_normalized_context(self, normalized_context):
        lines = normalized_context.split('\n')
        
        for i, line in enumerate(lines):
            or_positions = [m.start() for m in re.finditer(r'\∨|\\lor', line)]
            for pos in or_positions:
                left_part = line[:pos]
                match = re.search(r'(True|False)(?!.*?(True|False).*?$)', left_part)
                if match:
                    if match.group() == 'True':
                        new_left_part = left_part[:match.start()] + 'False' + left_part[match.end():]
                        lines[i] = new_left_part + line[pos:]
        return '\n'.join(lines)
    
    def post_process_final_answer(self, response_c):
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers =  match.group(1)
            return answers
        pattern_direct = r'\{(\w+)\}'
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    
    def list_to_indexed_string(self, item_list):
        indexed_list = [f"{i + 1}. {item}" for i, item in enumerate(item_list)]
        return "\n".join(indexed_list)
    
    def extract_facts_and_rules(self, content):
        fact_pattern = r'Fakta(.*?)Aturan'
        fact_match = re.search(fact_pattern, content, re.DOTALL)
        facts = fact_match.group(1).strip() if fact_match else None

        rules_pattern = r'Aturan(.*?)Konjektur'
        rules_match = re.search(rules_pattern, content, re.DOTALL)
        rules = rules_match.group(1).strip() if rules_match else None

        return facts, rules

    def extract_query(self, content):
        query_matches = list(re.finditer(r'Konjektur', content))
        
        if not query_matches:
            return None
        
        last_query_pos = query_matches[-1].start()
        
        last_query_content = content[last_query_pos:]
        
        patterns = [
            r'Konjektur\s*```(?:plaintext)?\n?(.+?)\n?```',
            r'Konjektur\s*(.+?)(?:\n|$)',
            r'Konjektur.*?\n(.*?)(?:\n|$)',
            r'Konjektur.*?\n.*?\n(.*?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, last_query_content, re.DOTALL)
            if match:
                query = match.group(1).strip()
                if '(' in query and ')' in query:
                    return query
        
        return None
    
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
    
    def post_process_decompose(self, content, rules_count=None):

        content = (content or "").replace('\u200b', '').replace('\ufeff', '')

        # marker after which we should look for the first final block
        marker_pattern = r'Di bawah ini adalah yang perlu Anda pecahkan|Di bawah ini adalah yang perlu Anda pecahkan\.\.|Di bawah ini adalah yang perlu Anda pecahkan:'
        marker_match = re.search(marker_pattern, content, flags=re.IGNORECASE)

        # final-form block pattern (first only)
        final_block_pattern = r'\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*\s*(.*?)(?=(\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*)|$)'

        search_area = content[marker_match.end():] if marker_match else content

        # find only the first final block in the search area; fallback to searching whole content if none found
        final_block_match = re.search(final_block_pattern, search_area, flags=re.DOTALL | re.IGNORECASE)
        if final_block_match:
            final_blocks_text = [final_block_match.group(1)]
        else:
            # fallback: try full content once
            final_block_match_full = re.search(final_block_pattern, content, flags=re.DOTALL | re.IGNORECASE)
            final_blocks_text = [final_block_match_full.group(1)] if final_block_match_full else [content]

        def _extract_section_raw(block: str, header: str, stop_headers: list) -> str:
            # allow common synonyms and optional punctuation after header
            pat = rf'(?i){re.escape(header)}\s*[:\-\)]?\s*(.*?)(?=(?:' + '|'.join([re.escape(h) + r'\s*[:\-\)]?' for h in stop_headers]) + r')|$)'
            m = re.search(pat, block, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def _nonempty_raw_lines(s: str):
            return [ln for ln in s.splitlines() if ln.strip()]

        def _balanced_parens(s: str) -> bool:
            # simple balance check for parentheses and inline latex \( \)
            if s.count('(') != s.count(')'):
                return False
            # count literal backslash-paren sequences
            if s.count(r'\(') != s.count(r'\)'):
                if s.count(r'\(') or s.count(r'\)'):
                    return False
            return True

        def _looks_truncated_line(s: str) -> bool:
            s_strip = s.rstrip()
            # trailing ellipsis or suspicious truncation marks
            if s_strip.endswith('...') or s_strip.endswith('…'):
                return True
            # incomplete ending tokens
            if s_strip.endswith('(') or s_strip.endswith(',') or s_strip.endswith('\\') or s_strip.endswith('\\left') or s_strip.endswith('\\right'):
                return True
            # if contains parentheses but doesn't end with ) or True/False) it's suspicious
            if '(' in s_strip or ')' in s_strip:
                if not re.search(r'\)\s*$|True\)\s*$|False\)\s*$', s_strip):
                    return True
            # unbalanced parentheses
            if not _balanced_parens(s_strip):
                return True
            return False

        # header synonym lists
        cnf_headers = ['Aturan dalam CNF', 'Aturan CNF', 'Aturan', 'Rules', 'Aturan (CNF)']
        skolem_headers = ['Skolemisasi', 'Skolem', 'Bentuk Akhir Setelah Skolemisasi', 'Skolemization']
        fakta_headers = ['Fakta', 'Facts', 'Fact']
        konj_headers = ['Konjektur', 'Konjecture', 'Conjecture', 'Konjektur:']

        # Process only the first final block (final_blocks_text contains exactly one element)
        block = final_blocks_text[0]

        cnf_raw = ""
        for h in cnf_headers:
            cnf_raw = _extract_section_raw(block, h, skolem_headers + konj_headers + fakta_headers + ['Pemecahan', 'Bentuk Akhir', 'Final Form'])
            if cnf_raw:
                break

        skolem_raw = ""
        for h in skolem_headers:
            skolem_raw = _extract_section_raw(block, h, cnf_headers + konj_headers + fakta_headers + ['Pemecahan', 'Bentuk Akhir', 'Final Form'])
            if skolem_raw:
                break

        # fallback: lines containing 'menjadi' anywhere in the block (mapping style)
        if not skolem_raw:
            mapping_lines = [ln for ln in block.splitlines() if 'menjadi' in ln]
            if mapping_lines:
                skolem_raw = "\n".join(mapping_lines)

        # extract fakta/konjektur if present
        fakta_raw = ""
        for h in fakta_headers:
            fakta_raw = _extract_section_raw(block, h, cnf_headers + skolem_headers + konj_headers + ['Pemecahan', 'Bentuk Akhir'])
            if fakta_raw:
                break

        konj_raw = ""
        for h in konj_headers:
            konj_raw = _extract_section_raw(block, h, cnf_headers + skolem_headers + fakta_headers + ['Pemecahan', 'Bentuk Akhir'])
            if konj_raw:
                break

        cnf_lines = _nonempty_raw_lines(cnf_raw) if cnf_raw else []
        skolem_lines = _nonempty_raw_lines(skolem_raw) if skolem_raw else []
        fakta_lines = _nonempty_raw_lines(fakta_raw) if fakta_raw else []
        konj_lines = _nonempty_raw_lines(konj_raw) if konj_raw else []

        # truncated heuristic applied to the lines we will expose
        lines_to_check = cnf_lines if cnf_lines else (skolem_lines if skolem_lines else (fakta_lines if fakta_lines else konj_lines))
        possibly_truncated = any(_looks_truncated_line(ln) for ln in lines_to_check) if lines_to_check else False
        actual_rule_count = len(cnf_lines) if cnf_lines else len(skolem_lines)

        # build output
        out_lines = []
        out_lines.append("**Bentuk Akhir:**")
        out_lines.append("")

        # CNF
        out_lines.append("Aturan dalam CNF:")
        if cnf_lines:
            for i, ln in enumerate(cnf_lines, start=1):
                out_lines.append(f"{i}. {ln}")
        else:
            out_lines.append("(tidak ada Aturan dalam CNF yang ditemukan)")

        out_lines.append("")

        # Fakta
        out_lines.append("Fakta:")
        if fakta_lines:
            for ln in fakta_lines:
                out_lines.append(ln)
        else:
            out_lines.append("(tidak ada blok Fakta yang ditemukan)")

        out_lines.append("")

        # Konjektur
        out_lines.append("Konjektur:")
        if konj_lines:
            for ln in konj_lines:
                out_lines.append(ln)
        else:
            out_lines.append("(tidak ada blok Konjektur yang ditemukan)")

        out_lines.append("")

        # Skolemisasi
        out_lines.append("**Skolemisasi:**")
        if skolem_lines:
            for ln in skolem_lines:
                out_lines.append(ln)
            out_lines.append("")
            out_lines.append("**Bentuk Akhir Setelah Skolemisasi:**")
            for ln in skolem_lines:
                out_lines.append(ln)
        else:
            out_lines.append("(tidak ada keluaran Skolemisasi eksplisit ditemukan — gunakan Aturan dalam CNF di atas)")

        out_lines.append("")

        # metadata
        computed_rule_count = len(cnf_lines) if cnf_lines else (len(skolem_lines) if skolem_lines else 0)
        out_lines.append(f"RULE_COUNT: {computed_rule_count}")
        out_lines.append(f"EXPECTED_RULE_COUNT: {str(rules_count) if rules_count is not None else 'None'}")
        out_lines.append(f"POSSIBLY_TRUNCATED: {str(possibly_truncated)}")
        out_lines.append("")

        return "\n".join(out_lines)

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
        file_name = f'{model_name}_trans_decompose_no_negation.json'
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
        
    def process_example(self, example, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer):
        ### File loads
        id= example['id']
        original_context= example['original_context']
        question= example['question']
        and_or= example['and_or']
        either_or= example['either_or']
        biconditional= example['biconditional']
        translated_facts= example['translated_facts']
        translated_rules= example['translated_rules']
        translated_conjecture= example['translated_conjecture']
        ground_truth= example['ground_truth']

        print("Decomposing rules...")
        if and_or:
            prompts_b = self.construct_prompt_b(and_or, icl_and_or_decomposer)
            print(f"Decomposition prompt_b with and_or len {len(and_or)} and with prompts_len{len(prompts_b)}: {prompts_b} ", )
            responses_and_or_process = self.openai_api.generate(prompts_b)
            responses_and_or_text = responses_and_or_process[0] if isinstance(responses_and_or_process, (list,tuple)) else responses_and_or_process
            print("Decomposition response: ", responses_and_or_text)
            responses_and_or = self.post_process_decompose(responses_and_or_text, len(prompts_b))
            responses_and_or = self.clean_irrelevant_lines(responses_and_or)
        if either_or:
            responses_either_or_process = self.openai_api.generate(self.construct_prompt_b(either_or, icl_either_or_decomposer))
            responses_either_or_text = responses_either_or_process[0] if isinstance(responses_either_or_process, (list,tuple)) else responses_either_or_process
            responses_either_or = self.post_process_decompose(responses_either_or_text)
            responses_either_or = self.clean_irrelevant_lines(responses_either_or)
        if biconditional:
            responses_biconditional_process = self.openai_api.generate(self.construct_prompt_b(biconditional, icl_biconditional_decomposer))
            responses_biconditional_text = responses_biconditional_process[0] if isinstance(responses_biconditional_process, (list,tuple)) else responses_biconditional_process
            responses_biconditional = self.post_process_decompose(responses_biconditional_text)
            responses_biconditional = self.clean_irrelevant_lines(responses_biconditional)
        
        responses_b = '\n'.join(filter(None, [responses_and_or, responses_either_or, responses_biconditional]))
        print("Decomposing response: ", responses_b)
        
        normalized_context = responses_b
        normalized_conjecture = self.clean_conjecture(translated_conjecture)
        print('Normalized context: ', normalized_context)
        print('Normalized conjecture: ', normalized_conjecture)
        
        negated_label = 'false'
        sos_list = normalized_conjecture

        print("Negated Label Initialization: ", negated_label)
        print("SOS List Initialization: ", sos_list)

        if self.dataset_name == "ProntoQA":
            normalized_context = self.process_normalized_context(normalized_context)

        if isinstance(normalized_context, list):
            normalized_context = "\n".join(self.split_cnf_clause(normalized_context))

        output = {
            'id': id, 
            'original_context': original_context,
            'question': question, 
            'translated_context': {"Translated_Facts": translated_facts, "Translated_Rules": translated_rules, "Translated_Conjecture": translated_conjecture},
            'decomposition_process': {key: value for key, value in {"and_or": locals().get("responses_and_or_process"), "either_or": locals().get("responses_either_or_process"), "biconditional": locals().get("responses_biconditional_process")}.items() if value is not None},
            'normalized_context': {"Fact": translated_facts, "and_or": responses_and_or, "either_or": responses_either_or, "biconditional": responses_biconditional},
            'normalized_conjecture': normalized_conjecture,
            'negated_label': negated_label,
            'sos_list': sos_list,
            'ground_truth': ground_truth
        }

        print(output)
        return output, None
        
    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.split, self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        if self.dataset_name == "ProntoQA":
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer(self.prompts_folder)
            icl_either_or_decomposer = None
            icl_biconditional_decomposer = None
        else:
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer(self.prompts_folder)
            icl_either_or_decomposer = self.load_in_context_either_or_decomposer(self.prompts_folder)
            icl_biconditional_decomposer = self.load_in_context_biconditional_decomposer(self.prompts_folder)

        print("Number of batch: ", self.batch_num)
        counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            futures = {
                executor.submit(self.process_example, example, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer): example 
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()
