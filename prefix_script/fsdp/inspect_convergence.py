#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-step side-by-side inspection of two training runs.

Prints a column-aligned table with baseline / treatment / diff for the key
metrics (pg_loss, reward, kl, grad_norm) so you can see at which step the
two runs start to diverge.

Usage:
    python prefix_script/fsdp/inspect_convergence.py \\
        --baseline_log /tmp/.../R1_dense.log \\
        --treatment_log /tmp/.../R2_tree_noncp.log
"""

from __future__ import annotations

import argparse
import os
import sys

# Reuse the parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_training_metrics import parse_log  # noqa: E402

METRICS = [
    "actor/pg_loss",
    "critic/score/mean",
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/entropy",
    "actor/loss",
    "actor/ppo_kl",
    "response_length/mean",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_log", required=True)
    parser.add_argument("--treatment_log", required=True)
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to inspect (defaults to a curated set)",
    )
    args = parser.parse_args()

    base = parse_log(args.baseline_log)
    treat = parse_log(args.treatment_log)
    if not base or not treat:
        print(f"[INSPECT] one or both logs have zero parsed steps (base={len(base)}, treat={len(treat)})")
        return 1

    metric_list = args.metrics.split(",") if args.metrics else METRICS

    print(f"[INSPECT] baseline:  {args.baseline_log}  ({len(base)} steps)")
    print(f"[INSPECT] treatment: {args.treatment_log} ({len(treat)} steps)")
    print()

    shared = sorted(set(base.keys()) & set(treat.keys()))
    for metric in metric_list:
        any_step_has_it = any(metric in base[s] or metric in treat[s] for s in shared)
        if not any_step_has_it:
            continue

        print(f"=== {metric} ===")
        print(f"{'step':<6} {'baseline':>14} {'treatment':>14} {'abs_diff':>14} {'rel_diff':>10}")
        print("-" * 66)
        for step in shared:
            b_val = base[step].get(metric)
            t_val = treat[step].get(metric)
            if b_val is None and t_val is None:
                continue
            if b_val is None or t_val is None:
                only = "treatment" if b_val is None else "baseline"
                print(
                    f"{step:<6} {'-' if b_val is None else f'{b_val:14.6g}'}"
                    f" {'-' if t_val is None else f'{t_val:14.6g}'}"
                    f" {'(only ' + only + ')':>14} {'-':>10}"
                )
                continue
            abs_diff = t_val - b_val
            rel_denom = max(abs(b_val), 1e-9)
            rel_diff = abs(abs_diff) / rel_denom
            # Flag large diffs for eyeballing
            flag = ""
            if abs(abs_diff) > 0.05 and rel_diff > 0.1:
                flag = "  <-- DIVERGE"
            print(f"{step:<6} {b_val:>14.6g} {t_val:>14.6g} {abs_diff:>14.6g} {rel_diff:>10.4f}{flag}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
