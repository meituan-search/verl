#!/usr/bin/env python3
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert ~/dataset/coqa.parquet to verl GRPO format.

Input columns: prompt (JSON str), data_source, extra_info (JSON str)
Output adds:  reward_model (dict), converts prompt/extra_info to objects
"""

import json
from pathlib import Path

import pandas as pd

SRC = Path.home() / "dataset" / "coqa.parquet"
DST = Path.home() / "dataset" / "coqa_grpo.parquet"

df = pd.read_parquet(SRC)
print(f"Loaded {len(df)} rows from {SRC}")

records = []
for _, row in df.iterrows():
    prompt = json.loads(row["prompt"])  # list[dict]
    extra = json.loads(row["extra_info"])  # dict
    answer = extra.get("answer", "")
    records.append(
        {
            "data_source": row["data_source"],
            "prompt": prompt,
            "extra_info": extra,
            "reward_model": {"ground_truth": answer, "style": "rule"},
        }
    )

out = pd.DataFrame(records)
out.to_parquet(DST, index=False)
print(f"Saved {len(out)} rows → {DST}")
print("Sample reward_model:", out.iloc[0]["reward_model"])
