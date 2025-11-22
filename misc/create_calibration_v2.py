"""
Build an Indonesian text corpus from multiple sources and save as parquet.
- Streams Hugging Face datasets (stable Wikipedia, stable News Corpus, QA context, dictionaries).
- Replaced all sources failing due to "Dataset scripts are no longer supported" (e.g., old wikipedia, mc4).
- Cleans, dedups, appends synthetic logic samples.
- Outputs JSONL (stream-safe) then converts to single parquet.
"""

import os
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm
import mwparserfromhell
import re
import html
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- Configuration ----------
OUT_PREFIX = "misc/ind_corpus"
TARGET_TOTAL = 100000   # total target documents
SAMPLE_RATIOS = {
    "google/LoraxBench:NLI": 3,
    "Lyon28/kamus-besar-bahasa-indonesia": 3,
    "izzulgod/indonesian-conversation": 1,
    "indonesian-nlp/wikipedia-id": 8,
    "izzulgod/indonesian-reasoning": 2,
}
MAX_PER_SOURCE = None    # None -> proportional to TARGET_TOTAL & SAMPLE_RATIOS
MIN_CHARS = 200       # filter out very short items
JSONL_OUT = f"{OUT_PREFIX}.jsonl"
PARQUET_OUT = f"{OUT_PREFIX}.parquet"
BATCH_TO_PARQUET = 10000  # convert JSONL -> parquet in batches to save memory
# ----------------------------------------

# ---------- Basic cleaners (Kept unchanged) ----------
def clean_wikitext(text: str) -> str:
  """Strip wiki markup and HTML and return plain text."""
  if not text:
    return ""
  # If it looks like HTML, unescape & strip tags
  text = html.unescape(text)
  # Remove common image/file references
  text = re.sub(r'\[\[File:[^\]]+\]\]', ' ', text, flags=re.I)
  text = re.sub(r'\[\[Image:[^\]]+\]\]', ' ', text, flags=re.I)
  # If it looks like wikitext, use mwparserfromhell
  try:
    parsed = mwparserfromhell.parse(text)
    text = parsed.strip_code(normalize=True, collapse=True)
  except Exception:
    # remove brackets and templates crudely
    text = re.sub(r'\{\{[^}]+\}\}', ' ', text)
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
  # remove multiple newlines and spaces
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def clean_html_or_plain(text: str) -> str:
  if not text:
    return ""
  # remove tags crudely
  text = re.sub(r'<[^>]+>', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def get_best_text_field(example: dict):
    """
    Extract the most useful human-readable text from a HF dataset example.

    Priority order (most specific -> generic):
      1. Chat-like messages: "messages" (list of {role, content}) or "dialog"/"conversations"
      2. QA style: "context" + "question" (+ "answers")
      3. Common single-string fields: text, content, paragraph, body, article, ...
      4. Concatenate title/heading/summary-like fields
      5. If a field is a list of strings (e.g., paragraphs), join them
      6. Any sufficiently long string-valued field (>100 chars)
    """
    if not isinstance(example, dict):
        return None

    # 1) Chat-like messages
    msgs = example.get("messages") or example.get("dialog") or example.get("conversations")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            # each message might be dict-like or simple string
            if isinstance(m, dict):
                role = str(m.get("role", "")).strip()
                content = m.get("content", "")
            else:
                # sometimes messages are stored as ["role", "content"] or plain strings
                role = ""
                content = m
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    continue
            role_label = f"{role}: " if role else ""
            parts.append(role_label + content.strip())
        combined = "\n".join(p for p in parts if p).strip()
        if combined:
            return combined

    # 2) QA-like examples
    context = None
    if "context" in example and isinstance(example["context"], str) and example["context"].strip():
        context = example["context"].strip()
    elif "paragraph" in example and isinstance(example["paragraph"], str) and example["paragraph"].strip():
        context = example["paragraph"].strip()
    elif "Premise" in example and isinstance(example["Premise"], str) and example["Premise"].strip():
        context = example["Premise"].strip()

    question = None
    if "question" in example and isinstance(example["question"], str) and example["question"].strip():
        question = example["question"].strip()
    elif "query" in example and isinstance(example["query"], str) and example["query"].strip():
        question = example["query"].strip()
    elif "Hypothesis" in example and isinstance(example["Hypothesis"], str):
        answer = example["Hypothesis"].strip()

    # answers may be dict or list
    answer = None
    if "answers" in example:
        a = example["answers"]
        if isinstance(a, dict):
            # huggingface answers usually { "answer_start": [...], "text": [...] }
            txts = a.get("text") or a.get("answer") or a.get("answers")
            if isinstance(txts, list) and txts:
                answer = txts[0]
            elif isinstance(txts, str):
                answer = txts
        elif isinstance(a, list) and a:
            # list of strings
            if isinstance(a[0], str):
                answer = a[0]
            elif isinstance(a[0], dict) and "text" in a[0]:
                answer = a[0]["text"]
    elif "answer" in example and isinstance(example["answer"], str):
        answer = example["answer"].strip()

    if context and question:
        out = f"context: {context}\nquestion: {question}"
        if answer:
            out += f"\nanswer: {answer}"
        return out

    # 3) Common single-string fields (prefer longer ones)
    for k in ("text", "content", "body", "article", "plaintext", "clean_text", "excerpt"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 4) If those fields are lists, join them (paragraphs, passages, sentences)
    for k in ("paragraphs", "passages", "sentences", "lines"):
        v = example.get(k)
        if isinstance(v, list) and v:
            strs = [s.strip() for s in v if isinstance(s, str) and s.strip()]
            if strs:
                joined = "\n\n".join(strs)
                if len(joined) > 50:
                    return joined

    # 5) Concatenate common subfields (title, heading, summary, opencc)
    text_parts = []
    for k in ("title", "heading", "summary", "opencc", "sentence"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            text_parts.append(v.strip())
    if text_parts:
        return " ".join(text_parts)

    # 6) Try nested dicts: some examples store {"data": {"text": "..."}}
    for k in ("data", "meta", "payload", "document"):
        v = example.get(k)
        if isinstance(v, dict):
            # look for a text-like field inside
            for kk in ("text", "content", "body", "article"):
                vv = v.get(kk)
                if isinstance(vv, str) and len(vv.strip()) > 50:
                    return vv.strip()

    # 7) Any sufficiently large string-valued entry
    for k, v in example.items():
        if isinstance(v, str) and len(v.strip()) > 100:
            return v.strip()

    # 8) If some fields are small strings, fallback to joining them (avoid garbage)
    small_parts = []
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            small_parts.append(v.strip())
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v[:3]):
            # in case there are small lists of strings, include them
            small_parts.extend([s.strip() for s in v if isinstance(s, str)])
        if len(small_parts) >= 3:
            joined = " ".join(small_parts)
            if len(joined) > 120:
                return joined

    return None

# For dedup detection
def sha1(text: str) -> str:
  return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---------- Synthetic logic generator (Kept unchanged) ---------
try:
  from create_calibration import gen_fol_sample, gen_cnf_sample, gen_resolver_sample
  def generate_logic_examples(n):
    samples = []
    import random
    for _ in range(n):
      r = random.random()
      if r < 0.33:
        samples.append(gen_fol_sample())
      elif r < 0.66:
        samples.append(gen_cnf_sample())
      else:
        samples.append(gen_resolver_sample())
    return samples
except ImportError:
  print("Warning: Logic generation script 'create_calibration.py' not found or import failed. Using placeholder logic.")
  def generate_logic_examples(n):
    out = []
    for i in range(n):
      out.append(f"### Logika Sintetik Sample {i}\nDeskripsi: Contoh logika diskret untuk pelatihan.\n...")
    return out

# ---------- Main streaming + write JSONL (Updated) ----------
def stream_and_write_jsonl(target_total, sample_ratios, jsonl_out):
  os.makedirs(os.path.dirname(jsonl_out) or ".", exist_ok=True)
  # normalize ratios
  total_ratio = sum(sample_ratios.values())
  normalized = {k: v/total_ratio for k, v in sample_ratios.items()}
  per_source_target = {}
  for k, frac in normalized.items():
    per_source_target[k] = int(frac * target_total)

  print("Per-source targets:", per_source_target)
  seen = set()
  total_written = 0

  with open(jsonl_out, "w", encoding="utf-8") as fout:
    for full_ds_id, limit in per_source_target.items():
      if limit <= 0:
        continue
      print(f"\nStreaming dataset: {full_ds_id} (target {limit})")
      
      # Parse ID for config name (e.g., "Muennighoff/wikipedia_2023_all:id" -> name=..., config="id")
      ds_id_parts = full_ds_id.split(":")
      ds_name = ds_id_parts[0]
      ds_config = ds_id_parts[1] if len(ds_id_parts) > 1 else None
      
      hf_ds = None
      try:
        # Use streaming and 'train' split by default
        hf_ds = load_dataset(ds_name, ds_config, split="train", streaming=True)
      except Exception as e:
        print(f"Could not load {full_ds_id}: {e}")
        try:
            hf_ds = load_dataset(ds_name, ds_config, split="test", streaming=True)
        except Exception as e2:
            print(f"Fallback load failed for {full_ds_id}: {e2}")
            try:
                hf_ds = load_dataset(ds_name, split="train", streaming=True)
            except Exception as e3:
                print(f"Fallback load failed for {full_ds_id}: {e3}")
                continue


      written = 0
      # Use tqdm for status updates
      for item in tqdm(hf_ds, desc=f"Processing {full_ds_id}"):
        if written >= limit:
          break
        text = get_best_text_field(item)
        if text is None:
          continue
        
        # Heuristic cleaning
        if "wikipedia" in full_ds_id.lower() or "wiki" in full_ds_id.lower():
          text = clean_wikitext(text)
        else:
          text = clean_html_or_plain(text)

        if len(text) < MIN_CHARS:
          continue
        
        # dedup (hash prefix)
        h = sha1(text[:3000]) 
        if h in seen:
          continue
        seen.add(h)

        record = {
          "source": full_ds_id,
          "text": text
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        total_written += 1
        if total_written % 10000 == 0:
          print(f"\nTotal written so far: {total_written} records.", end="", flush=True)

      print(f"\nFinished {full_ds_id}, written {written} records.")

    # Append synthetic logic samples to reach target_total
    remaining = target_total - total_written
    if remaining > 0:
      print(f"\nAppending {remaining} synthetic logic samples...")
      logic_samples = generate_logic_examples(remaining)
      for sample in tqdm(logic_samples, desc="Writing logic samples"):
        text = sample if isinstance(sample, str) else str(sample)
        h = sha1(text[:3000])
        if h in seen:
          continue
        seen.add(h)
        fout.write(json.dumps({"source": "synthetic_logic", "text": text}, ensure_ascii=False) + "\n")
        total_written += 1
        
  print(f"\nAll done. Total records written to JSONL: {total_written}")

def jsonl_to_parquet_single(jsonl_path, parquet_out, batch=10000, coerce_to_str=False):
  """
  Stream jsonl_path -> parquet_out in one Parquet file using ParquetWriter.
  - batch: number of jsonl rows per write call (keeps memory low).
  - coerce_to_str: if True, cast all dataframe columns to string to avoid schema mismatches.
  """
  writer = None
  rows = []
  total = 0

  with open(jsonl_path, "r", encoding="utf-8") as fin:
    for line in tqdm(fin, desc="Reading JSONL"):
      obj = json.loads(line)
      rows.append(obj)
      if len(rows) >= batch:
        df = pd.DataFrame(rows)
        if coerce_to_str:
          df = df.astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
          writer = pq.ParquetWriter(parquet_out, table.schema)
        writer.write_table(table)
        total += len(rows)
        rows = []

    # final remainder
    if rows:
      df = pd.DataFrame(rows)
      if coerce_to_str:
        df = df.astype(str)
      table = pa.Table.from_pandas(df, preserve_index=False)
      if writer is None:
        writer = pq.ParquetWriter(parquet_out, table.schema)
      writer.write_table(table)
      total += len(rows)

  if writer:
    writer.close()
    print(f"Finished writing {total} records -> {parquet_out}")
  else:
    print("No data found in JSONL. No Parquet file written.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", type=int, default=TARGET_TOTAL, help="total documents to produce")
  parser.add_argument("--jsonl", type=str, default=JSONL_OUT)
  parser.add_argument("--parquet", type=str, default=PARQUET_OUT)
  args = parser.parse_args()

  stream_and_write_jsonl(args.target, SAMPLE_RATIOS, args.jsonl)
  jsonl_to_parquet_single(args.jsonl, args.parquet)
  print("Done. Parquet saved to:", args.parquet)