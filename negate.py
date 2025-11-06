import json
import argparse
import re
import os
from utils import sanitize_filename

def negate_conjecture(conjecture):
    if isinstance(conjecture, str):
        if re.search(r'True', conjecture):
            updated_conjecture = re.sub(r'True', 'False', conjecture)
        elif re.search(r'False', conjecture):
            updated_conjecture = re.sub(r'False', 'True', conjecture)
        else:
            updated_conjecture = "false"
    else:
        updated_conjecture = "Invalid input: conjecture must be a string"
        print(f"Invalid input {conjecture}: conjecture must be a string")

    return updated_conjecture

def main(args):
    model_name = sanitize_filename(args.model_name)
    input_path = f'{model_name}_trans_decompose_no_negation.json'
    input_path = os.path.join(args.save_path, args.dataset_name, input_path)
    with open(input_path, 'r', encoding="utf8") as f:
        data = json.load(f)
    
    for item in data:
        item['sos_list'] = negate_conjecture(item['sos_list'])
        item['negated_label'] = "True"
    
    save_path = f'{model_name}_trans_decompose_negated_data.json'
    save_path = os.path.join(args.save_path, args.dataset_name, save_path)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)