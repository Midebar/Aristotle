import pandas as pd
import random

# --- CONFIGURATION ---
OUTPUT_FILE = "misc/cal_data_logic"
NUM_SAMPLES = 150000

# ("llama3", "qwen", "gemma", "yi")
MODEL_TYPE = "qwen" 
# ---------------------

# --- DATA GENERATOR ---

subjects = ["Alex", "Rex", "Fae", "Max", "Sam", "Stella", "Wren", "Polly"]
kinds = ["Impus", "Wumpus", "Jompus", "Numpus", "Tumpus", "Zumpus", "Dumpus", "Rompus", "Vumpus"]
attrs = ["Besar", "Kecil", "Merah", "Biru", "Panas", "Dingin", "Bahagia", "Sedih", "Asam", "Manis", "Agresif", "Pemalu"]

def apply_template(system, user, assistant, model_type):
    reinforced_system = "Anda adalah asisten AI yang cerdas dan sangat membantu, ahli logika, dan **SELALU menjawab dalam Bahasa Indonesia yang baik dan benar**. Jangan pernah menggunakan bahasa lain selain Bahasa Indonesia."
    
    if model_type == "qwen" or model_type == "yi":
        # ChatML Format
        text = f"<|im_start|>system\n{reinforced_system}<|im_end|>\n"
        text += f"<|im_start|>user\n{user}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{assistant}<|im_end|>"
        return text
    elif model_type == "gemma":
        # Gemma Format (No system role usually, prepend to user)
        text = f"<start_of_turn>user\n{reinforced_system}\n\n{user}<end_of_turn>\n"
        text += f"<start_of_turn>model\n{assistant}<end_of_turn>"
        return text
    else: 
        # Default to Llama 3
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{reinforced_system}<|eot_id|>"
        text += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
        return text

# 1. Template Translation to FOL
def gen_fol_sample():
    sub = random.choice(subjects)
    k1 = random.choice(kinds)
    k2 = random.choice(kinds)
    attr = random.choice(attrs)
    
    user_text = f"""Deskripsi Tugas: Anda adalah asisten logika. Terjemahkan konteks berikut ke First-Order Logic (FOL). Pastikan output Anda dalam format blok.

------

Konteks:
Setiap {k1.lower()} adalah {k2.lower()}. {k2} adalah {attr.lower()}. {sub} adalah {k1.lower()}.

Konjektur:
{sub} {attr.lower()}."""

    assistant_text = f"""###
***Bentuk Akhir***

Fakta:
{sub} adalah {k1.lower()} ::: {k1}({sub}, True)

Aturan:
{k1}($x, True) >>> {k2}($x, True)
{k2}($x, True) >>> {attr}($x, True)

Konjektur:
{attr}({sub}, True)

***Akhir Blok***
###"""

    # System text is handled inside apply_template now
    return apply_template("", user_text, assistant_text, MODEL_TYPE)

# 2. Template CNF Decomposition
def gen_cnf_sample():
    k1 = random.choice(kinds)
    k2 = random.choice(kinds)
    attr = random.choice(attrs)
    
    user_text = f"""Deskripsi Tugas: Konversikan aturan FOL berikut ke Conjunctive Normal Form (CNF) dalam Bahasa Indonesia.

Konteks:

Aturan:
1. {k1}($x, True) >>> {k2}($x, True)
2. {k2}($x, True) >>> {attr}($x, False)

Hasil konversi:"""

    assistant_text = f"""###
***Bentuk Akhir***

Aturan dalam CNF:
1. - \\left({k1}(x, False) \\lor {k2}(x, True)\\right)
2. - \\left({k2}(x, False) \\lor {attr}(x, False)\\right)

***Akhir Blok***
###"""
    return apply_template("", user_text, assistant_text, MODEL_TYPE)

# 3. Template Logic Resolver
def gen_resolver_sample():
    sub = random.choice(subjects)
    pred = random.choice(kinds)
    pred2 = random.choice(attrs)
    
    # Randomly choose between contradiction or derivation
    if random.random() > 0.5:
        # Contradiction
        user_text = f"""Tugas: Diberikan dua clause, lakukan resolusi untuk mencari kontradiksi.

Clause 1: {pred}({sub}, True)
Clause 2: {pred}({sub}, False)

Uraikan proses resolusi dan berikan *Bentuk Akhir*."""
        
        assistant_text = f"""### Langkah 1: Cek Literal yang Saling Komplemen
Literal saling komplemen ditemukan: {pred}({sub}, True) vs {pred}({sub}, False).

### Langkah 2: Terapkan Resolusi pada Kedua Clause
Terjadi kontradiksi langsung karena kedua literal saling meniadakan.

### Langkah 3: Pemeriksaan Setelah Resolusi
Resolusi menghasilkan kontradiksi, sehingga label cukup tercapai.

***Bentuk Akhir***
Clause Baru: {{Kontradiksi}}
Label Cukup: [True]
***Akhir Blok***"""
    else:
        # Derivation
        user_text = f"""Tugas: Diberikan dua clause, lakukan resolusi dan berikan clause baru yang diturunkan.

Clause 1: {pred}({sub}, True)
Clause 2: \\({pred}(x, False) \\lor {pred2}(x, True)\\)

Uraikan proses resolusi dan berikan *Bentuk Akhir*."""
        
        assistant_text = f"""### Langkah 1: Cek Literal yang Saling Komplemen
Instansiasi x = {sub}. Literal komplemen adalah: {pred}({sub}, True) dan {pred}({sub}, False).

### Langkah 2: Terapkan Resolusi pada Kedua Clause
Kita eliminasi literal komplemen. Sisa literal adalah: {pred2}({sub}, True).

### Langkah 3: Pemeriksaan Setelah Resolusi
Tidak ditemukan kontradiksi.

***Bentuk Akhir***
Clause Baru: {{ {pred2}({sub}, True) }}
Label Cukup: [False]
***Akhir Blok***"""

    return apply_template("", user_text, assistant_text, MODEL_TYPE)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Generating Calibration Data for model type: {MODEL_TYPE}...")
    
    data = []

    # Generate samples for each task type
    for _ in range(NUM_SAMPLES):
        # Mix the tasks evenly
        rand = random.random()
        if rand < 0.33:
            data.append(gen_fol_sample())
        elif rand < 0.66:
            data.append(gen_cnf_sample())
        else:
            data.append(gen_resolver_sample())

    # Save to Parquet
    try:
        OUTPUT_FILE = f"{OUTPUT_FILE}_{MODEL_TYPE}.parquet"
        df = pd.DataFrame({'text': data})
        df.to_parquet(OUTPUT_FILE, engine='pyarrow')

        print(f"\nSUCCESS! Created {OUTPUT_FILE} with {len(df)} LogicNLI samples.")
        print(f"Model Format used: {MODEL_TYPE}")
        
    except Exception as e:
        print(f"\nError saving parquet file: {e}")
        print("Make sure installed dependencies: pip install pandas pyarrow")