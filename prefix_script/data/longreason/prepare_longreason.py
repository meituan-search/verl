"""Convert LongReason HuggingFace dataset splits to verl-compatible parquet files.

Usage:
    python3 prepare_longreason.py --src /tmp/claude/LongReason --out ./
    python3 prepare_longreason.py --src /tmp/claude/LongReason --out ./ --splits 8k 16k
"""

import argparse
import os

import pandas as pd
from datasets import load_from_disk


def convert_split(dataset, split_name: str, out_dir: str) -> str:
    rows = []
    for row in dataset:
        rows.append(
            {
                "data_source": "longreason",
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "extra_info": {
                    "answer": row["answer"],
                    "example_idx": row["example_idx"],
                    "split": split_name,
                },
                "reward_model": {"ground_truth": row["answer"], "style": "rule"},
            }
        )
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"longreason_{split_name}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  {split_name}: {len(df)} rows → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/tmp/claude/LongReason")
    parser.add_argument("--out", default=os.path.dirname(__file__))
    parser.add_argument("--splits", nargs="+", default=["8k", "16k", "32k", "64k", "128k"])
    args = parser.parse_args()

    ds = load_from_disk(args.src)
    os.makedirs(args.out, exist_ok=True)

    for split in args.splits:
        if split not in ds:
            print(f"  skip {split} (not found)")
            continue
        convert_split(ds[split], split, args.out)


if __name__ == "__main__":
    main()
