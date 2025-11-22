import pandas as pd

df = pd.read_parquet("misc/data.parquet")
# to CSV (comma)
df.to_csv("misc/data.csv", index=False)
# to TSV (tab)
df.to_csv("misc/data.tsv", sep="\t", index=False)
print(df.to_string(max_rows=200))
# to JSON Lines (one JSON object per line)
df.to_json("misc/data.jsonl", orient="records", lines=True)
