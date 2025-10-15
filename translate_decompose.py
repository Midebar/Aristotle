# translate_decompose.py
import json
import os
from tqdm import tqdm
from utils import ModelWrapper, sanitize_filename
import argparse
import re
import concurrent.futures
import traceback
import threading

class GPT3_Reasoning_Graph_Baseline:
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
        if args.base_url:
            self.openai_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens, base_url=args.base_url)
        else:
            self.openai_api = ModelWrapper(args.model_name, args.stop_words, args.max_new_tokens)
            
    def load_in_context_examples_trans(self, prompts_folder='./prompts'):
        file_path = os.path.join(prompts_folder, self.dataset_name, 'translation.txt')
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
        normalized_context = [line.split('\land') for line in unique_context_lines]
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
        """
        Scan final-form blocks (***Bentuk Akhir*** / ***Final Form***) from the last to the first.
        """

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

        # scan from last to first
        for block in reversed(final_blocks_text):
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
            if isinstance(context_sentence_count, int) and context_sentence_count > 0:
                for m in reversed(rule_iter):
                    rtxt = _clean_lead(m.group(1))
                    cnt, _ = _rule_count_and_lines(rtxt)
                    if cnt == context_sentence_count:
                        selected_rule = m
                        break

            if selected_rule is None:
                for m in reversed(rule_iter):
                    rtxt = _clean_lead(m.group(1))
                    cnt, _ = _rule_count_and_lines(rtxt)
                    if cnt > 0:
                        selected_rule = m
                        break

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

            fact_ok = _search_complete_predicate(fact_text)
            conj_ok = _search_complete_predicate(conj_text)

            # If valid, return this triple
            if fact_ok and conj_ok:
                # normalize some arrow variants to a common form for downstream parsing
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

        rule = re.sub(r'\s*(→|⇒|=>|->>|->|=>|—|-)\s*', ' >>> ', rule)
        rule = re.sub(r'\s*(\<\-\>|\<\=\>|\<\-\=\>)\s*', ' <-> ', rule)

        return fact, rule, conjecture

    def post_process_decompose(self, content):
        # # normalize invisible chars
        # content = content.replace('\u200b', '').replace('\ufeff', '').strip()

        # capture final-form blocks
        final_block_pattern = r'\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*\s*(.*?)(?=(\*\*\*(?:Bentuk Akhir|Final Form)\*\*\*)|$)'
        final_blocks = re.findall(final_block_pattern, content, flags=re.DOTALL | re.IGNORECASE)
        final_blocks_text = [b[0] for b in final_blocks] if final_blocks else [content]

        def _balanced_parens(s: str) -> bool:
            if s.count('(') != s.count(')'):
                return False
            # latex inline \( \) balance check (if used)
            if s.count(r'\(') != s.count(r'\)'):
                # if one side present but not both, treat as unbalanced
                if s.count(r'\(') or s.count(r'\)'):
                    return False
            return True

        def _looks_predicate_like(s: str) -> bool:
            if not s or not s.strip():
                return False
            if re.search(r'\b\w+\s*\([^()]*\b(True|False)\b[^()]*\)', s, re.IGNORECASE):
                return True
            if re.search(r'\b[a-zA-Z_]\w*\s*\([^()]*\)', s):
                return True
            if r'\forall' in s:
                return True
            # fallback: ends with ')'
            if s.strip().endswith(')'):
                return True
            return False

        def _extract_section(block: str, header: str, stop_headers: list) -> str:
            pat = rf'{re.escape(header)}\s*[:\-]?\s*(.*?)(?=(?:' + '|'.join([re.escape(h) + r'\s*[:\-]?' for h in stop_headers]) + r')|$)'
            m = re.search(pat, block, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).rstrip() if m else ""

        def _nonempty_lines_preserve(s: str):
            return [ln for ln in (l.rstrip() for l in s.splitlines()) if ln.strip()]

        # scan from last to first and pick first valid block
        for block in reversed(final_blocks_text):
            fakta_raw = _extract_section(block, 'Fakta', ['Aturan', 'Konjektur'])
            aturan_raw = _extract_section(block, 'Aturan', ['Konjektur', 'Fakta', 'Aturan dalam CNF', 'Aturan dalam'])
            konj_raw = _extract_section(block, 'Konjektur', ['Fakta', 'Aturan'])

            fakta_lines = _nonempty_lines_preserve(fakta_raw)
            aturan_lines = _nonempty_lines_preserve(aturan_raw)
            konj_lines = _nonempty_lines_preserve(konj_raw)

            konj_line = konj_lines[0] if konj_lines else ""

            facts_ok = bool(fakta_lines) and all((_balanced_parens(l) and _looks_predicate_like(l)) for l in fakta_lines)
            conj_ok = bool(konj_line) and (_balanced_parens(konj_line) and _looks_predicate_like(konj_line))
            rules_ok = bool(aturan_lines)

            if facts_ok and conj_ok and rules_ok:
                out = []
                out.append("Pemecahan\n")
                out.append("***Bentuk Akhir***:")
                out.append("Fakta:")
                for ln in fakta_lines:
                    out.append(ln)
                out.append("")
                out.append("Aturan dalam CNF:")
                for ln in aturan_lines:
                    out.append(ln)
                out.append("")
                out.append("Konjektur:")
                out.append(konj_line)
                out.append("")  # trailing newline
                return "\n".join(out)

        # fallback: best-effort formatting of the last block (preserve lines)
        fallback = final_blocks_text[-1]
        fakta_raw = _extract_section(fallback, 'Fakta', ['Aturan', 'Konjektur'])
        aturan_raw = _extract_section(fallback, 'Aturan', ['Konjektur', 'Fakta'])
        konj_raw = _extract_section(fallback, 'Konjektur', ['Fakta', 'Aturan'])

        fakta_lines = _nonempty_lines_preserve(fakta_raw)
        aturan_lines = _nonempty_lines_preserve(aturan_raw)
        konj_lines = _nonempty_lines_preserve(konj_raw)
        konj_line = konj_lines[0] if konj_lines else ""

        out = []
        out.append("Pemecahan\n")
        out.append("***Bentuk Akhir***:")
        out.append("Fakta:")
        for ln in fakta_lines:
            out.append(ln)
        out.append("")
        out.append("Aturan dalam CNF:")
        for ln in aturan_lines:
            out.append(ln)
        out.append("")
        out.append("Konjektur:")
        out.append(konj_line)
        out.append("")
        return "\n".join(out)
            
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
        
    def process_example(self, example, in_context_examples_trans, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer):
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
        # Split on sentence end punctuation followed by whitespace (keeps punctuation on the chunk).
        raw_sentences = re.split(r'(?<=[.!?])\s+', context_text.strip())
        # Filter out any empty results (in case of trailing whitespace)
        sentences = [s for s in raw_sentences if s.strip()]
        context_sentence_count = len(sentences)-1 # exclude facts at end of sentence

        translated_facts, translated_rules, translated_conjecture = self.extract_facts_rules_conjecture(responses_a_text, context_sentence_count)
        print("Translated Facts1: ", translated_facts)
        translated_facts = self.clean_irrelevant_lines(translated_facts)
        print(f"Translated Facts2: {translated_facts}")

        either_or, biconditional, and_or = self.categorize_rule_lines(translated_rules)
        
        print(f"AND/OR: {and_or} \n EITHER/OR: {either_or} \n BICONDITIONAL: {biconditional}")

        responses_and_or = None
        responses_either_or = None
        responses_biconditional = None

        print("Decomposing rules...")
        if and_or:
            responses_and_or_process = self.openai_api.generate(self.construct_prompt_b(and_or, icl_and_or_decomposer))
            responses_and_or_text = responses_and_or_process[0] if isinstance(responses_and_or_process, (list,tuple)) else responses_and_or_process
            responses_and_or = self.post_process_decompose(responses_and_or_text)
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
            
        if self.dataset_name == 'LogicNLI':
            original_context = "Facts: " + '\n'.join(example['facts']) + "Rules: " + '\n'.join(example['rules'])
        else:
            original_context = example['context']

        output = {
            'id': example['id'], 
            'original_context': original_context,
            'question': question, 
            'translated_context': {"Translated_Facts": translated_facts, "Translated_Rules": translated_rules, "Translated_Conjecture": translated_conjecture},
            'decomposed_process': {key: value for key, value in {"and_or": locals().get("responses_and_or_process"), "either_or": locals().get("responses_either_or_process"), "biconditional": locals().get("responses_biconditional_process")}.items() if value is not None},
            'normalized_context': {"Fact": translated_facts, "and_or": responses_and_or, "either_or": responses_either_or, "biconditional": responses_biconditional},
            'normalized_conjecture': normalized_conjecture,
            'negated_label': negated_label,
            'sos_list': sos_list,
            'ground_truth': example['answer']
        }

        print(output)
        return output, None
        
    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.split, self.sample_pct)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        in_context_examples_trans = self.load_in_context_examples_trans(self.prompts_folder)
        
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
                executor.submit(self.process_example, example, in_context_examples_trans, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer): example 
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
    parser.add_argument('--api_key', type=str)
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
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()
