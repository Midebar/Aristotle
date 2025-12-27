import json
import argparse
import os
import csv
import matplotlib.pyplot as plt
from utils import sanitize_filename

def load_json_file(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r', encoding="utf8") as file:
        content = file.read().strip()
        try:
            # Try loading as a standard single JSON object/list
            return json.loads(content)
        except json.JSONDecodeError:
            # Try loading as JSON Lines (one object per line)
            file.seek(0)  # Reset file pointer to the beginning
            data = []
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
    
def normalize_answer(answer):
    """Normalize the final choice: if not 'A' or 'B', treat it as 'C'."""
    if answer == 'A' or answer == 'B' or answer == 'D':
        return answer
    return 'C'

def evaluate_instance(id_, instance1, instance2, ground_truth):
    """Evaluates two instances with the same 'id' according to the rules provided."""
    try:
        answer1 = instance1.get('final_choice', 'C' if 'No final answer found in the text.' in instance1.get('final_answer', '') else None)
    except:
        answer1 = None
        
    try:
        answer2 = instance2.get('final_choice', 'C' if 'No final answer found in the text.' in instance2.get('final_answer', '') else None)
    except:
        answer2 = None
        
    answer1 = normalize_answer(answer1)
    answer2 = normalize_answer(answer2)
                
    if ground_truth == 'A':
        if {answer1, answer2} in [{'A', 'C'}, {'A'}]:
            return True
    
    elif ground_truth == 'B':
        if {answer1, answer2} in [{'B', 'C'}, {'B'}]:
            return True
    
    elif ground_truth == 'C':
        if answer1 == 'C' and (answer2 == 'C' or answer2 is None):
            return True
        
    elif ground_truth == 'D':
        if (answer1 == 'A' and answer2 == 'B') or (answer1 == 'B' and answer2 == 'A'):
            return True
    
    return False

def save_evaluation_file(rows, output_dir, model_name, fmt='csv'):
    """Saves the evaluation rows to a specific format."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_results.{fmt}"
    file_path = os.path.join(output_dir, filename)
    
    try:
        if fmt == 'csv':
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'gt', 'ans1', 'ans2', 'correct'])
                writer.writeheader()
                writer.writerows(rows)
        
        elif fmt == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(rows, f, indent=4)
                
        elif fmt == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Evaluation Results for {model_name}\n")
                f.write("="*50 + "\n")
                for row in rows:
                    status = "CORRECT" if row['correct'] else "WRONG"
                    f.write(f"ID: {row['id']} | GT: {row['gt']} | Ans1: {row['ans1']} | Ans2: {row['ans2']} | -> {status}\n")
                    
        else:
            print(f"Format '{fmt}' not supported. Skipping text file save.")
            return None

        print(f"Saved {fmt} results to: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Failed to save {fmt} file: {e}")
        return None

def show_table(rows, out_image_path='evaluation_table.png'):
    """rows: list of dicts with keys: id, gt, ans1, ans2, correct"""
    if not rows:
        print("No rows to display.")
        return

    cols = ["ID", "Ground Truth", "Negated", "Non-negated"]
    cell_text = []
    cell_colours = []
    for r in rows:
        cell_text.append([r['id'], r['gt'], r['ans1'], r['ans2']])
        color = "#d7f4dd" if r['correct'] else "#ffd6d6"
        # replicate color across columns for that row
        cell_colours.append([color] * len(cols))

    # adjust figure size based on number of rows (cap height)
    n = len(rows)
    row_height = 0.35
    height = max(2.5, min(20, n * row_height))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.axis('off')

    table = ax.table(cellText=cell_text,
                     colLabels=cols,
                     cellColours=cell_colours,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.1)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_image_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_image_path


def evaluate_files(dataset_name, model_name):
    """Evaluates all matching instances from two JSON files."""
    
    input_dir = args.save_path or './results/' 
    output_dir = args.output_path or input_dir
    
    model_name_clean = sanitize_filename(model_name)
    
    if args.evaluation_method == 'naive_prompting':
        print("Evaluating using the naive prompting method.")
        file1_path = f'{input_dir}/{dataset_name}/{model_name_clean}_naive_prompting.json'
        file2_path = file1_path
    else:
        file1_path = f'{input_dir}/{dataset_name}/{model_name_clean}_search_negation_True.json'
        file2_path = f'{input_dir}/{dataset_name}/{model_name_clean}_search_negation_False.json'
    
    # 1 is negated, 2 is non-negated
    try:
        file1 = load_json_file(file1_path)
        file2 = load_json_file(file2_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    file1_map = {}
    file2_map = {}
    counter = 0
    for item in file1:
        try:
            file1_map[item['id']] = item
        except:
            print(f"Error: 'id' not found or invalid in file1 item: {item}")
            print(f"Error item: {item}")
    for item in file2:
        try:
            file2_map[item['id']] = item
        except:
            print(f"Error: 'id' not found or invalid in file2 item: {item}")
        counter += 1
    
    print(f"Counter: {counter}")
    
    total_instances = 0
    correct_instances = 0
    
    common_ids = set(file1_map.keys()).union(set(file2_map.keys()))
    valid_ids = [id_ for id_ in common_ids if file1_map.get(id_) is not None and file2_map.get(id_) is not None]

    rows = []
    total_instances = 0
    correct_instances = 0
    error_id = []
    correct_id = []

    for id_ in sorted(valid_ids):
        try:
            instance1 = file1_map.get(id_)
            instance2 = file2_map.get(id_)

            ground_truth = instance1.get('ground_truth')

            fa1 = instance1.get('final_answer')
            fc1 = instance1.get('final_choice', 'C' if fa1 and 'No final answer found in the text.' in fa1 else None)
            fa2 = instance2.get('final_answer')
            fc2 = instance2.get('final_choice', 'C' if fa2 and 'No final answer found in the text.' in fa2 else None)

            n1 = normalize_answer(fc1)
            n2 = normalize_answer(fc2)
            print(f"ID: {id_}, GT: {ground_truth}, Negated: {n1}, Non-negated: {n2}")

            total_instances += 1
            correct = evaluate_instance(id_, instance1, instance2, ground_truth)
            if correct:
                correct_instances += 1
                correct_id.append(id_)
            else:
                error_id.append(id_)

            rows.append({
                'id': id_,
                'gt': ground_truth,
                'ans1': n1,
                'ans2': n2,
                'correct': bool(correct)
            })
        except Exception as e:
            print(f"Error processing {id_}: {e}")
            error_id.append(id_)

    accuracy = (correct_instances / total_instances) if total_instances > 0 else 0
    print(f"Total instances: {total_instances}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Error id: ", error_id)
    print("Correct id: ", correct_id)

    final_out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(final_out_dir, exist_ok=True)

    # Save accuracy file (CSV/JSON/TXT)
    save_evaluation_file(rows, final_out_dir, model_name_clean, args.output_format)

    # Save table image (PNG)
    out_image_path = os.path.join(final_out_dir, f"{model_name_clean}_results_table.png")
    saved_path = show_table(rows, out_image_path)
    
    if saved_path:
        print(f"Saved table image to: {saved_path}")
    else:
        print("No image saved.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, help="Path to input data files") 
    parser.add_argument('--output_path', type=str, help="Path to save results (defaults to save_path if not set)")
    parser.add_argument('--output_format', type=str, default='csv', choices=['csv', 'json', 'txt'], help="Output format for accuracy: csv, json, or txt")
    parser.add_argument('--evaluation_method', type=str, default='framework')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    evaluate_files(args.dataset_name, args.model_name)