#!/usr/bin/env python3
"""Convert ~/dataset/coqa.parquet to verl GRPO format.

Input columns: prompt (JSON str), data_source, extra_info (JSON str)
Output adds:  reward_model (dict), converts prompt/extra_info to objects
"""
import json
import pandas as pd
from pathlib import Path

SRC = Path.home() / "dataset" / "coqa.parquet"
DST = Path.home() / "dataset" / "coqa_grpo.parquet"

df = pd.read_parquet(SRC)
print(f"Loaded {len(df)} rows from {SRC}")

records = []
for _, row in df.iterrows():
    prompt = json.loads(row["prompt"])           # list[dict]
    extra  = json.loads(row["extra_info"])        # dict
    answer = extra.get("answer", "")
    records.append({
        "data_source":  row["data_source"],
        "prompt":       prompt,
        "extra_info":   extra,
        "reward_model": {"ground_truth": answer, "style": "rule"},
    })

out = pd.DataFrame(records)
out.to_parquet(DST, index=False)
print(f"Saved {len(out)} rows → {DST}")
print("Sample reward_model:", out.iloc[0]["reward_model"])