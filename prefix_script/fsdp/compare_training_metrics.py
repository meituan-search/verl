#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Compare per-step training metrics between two verl training runs.

Parses verl's console-logger format (one line per step):

    step:N - key1:val1 - key2:val2 - ...

For each requested metric, computes per-step abs/rel diff vs baseline, then
applies thresholds (per-step max + overall Pearson correlation). Reports
PASS/FAIL per metric and overall. Also dumps a JSON of all numbers for
downstream inspection.

Usage
-----
    python compare_training_metrics.py \\
        --baseline_log /tmp/dense.log \\
        --treatment_log /tmp/tree.log \\
        --label "tree non-CP vs dense" \\
        --output /tmp/compare_noncp.json

Default metric set + thresholds match what we use in the convergence test
(see research/2026-06-01-tree-training-test-plan.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional

# Ray colours its per-worker prefix with ANSI escapes (e.g. \x1b[36m(...)\x1b[0m).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Ray prefixes step lines with "(TaskRunner pid=NNN) " when logged through Ray.
_RAY_PREFIX_RE = re.compile(r"^\([\w]+\s+pid=\d+\)\s+")
# step:NN - key:value - key:value - ...
_STEP_LINE_RE = re.compile(r"^step:(\d+)\s+-\s+(.*)$")
# verl wraps metric values in np.<type>(...) (e.g. np.float64(0.18), np.int32(2)).
_NP_WRAP_RE = re.compile(r"^np\.\w+\((.*)\)$")


def _parse_value(s: str) -> Optional[float]:
    s = s.strip().rstrip("-").strip()
    m = _NP_WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(path: str) -> dict[int, dict[str, float]]:
    """Parse verl console-logger lines. Returns {step: {metric_name: value}}."""
    out: dict[int, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            # Strip ANSI colour codes Ray injects around its (worker pid=NNN) prefix.
            line = _ANSI_RE.sub("", line).strip()
            # Strip optional "(TaskRunner pid=NNN) " Ray prefix
            ray_m = _RAY_PREFIX_RE.match(line)
            if ray_m:
                line = line[ray_m.end() :]
            m = _STEP_LINE_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            rest = m.group(2)
            step_metrics: dict[str, float] = {}
            for part in rest.split(" - "):
                if ":" not in part:
                    continue
                k, _, v = part.partition(":")
                k = k.strip()
                val = _parse_value(v)
                if val is None:
                    continue
                step_metrics[k] = val
            if step_metrics:
                out[step] = step_metrics
    return out


@dataclass
class MetricSpec:
    name: str
    max_per_step_abs_diff: Optional[float] = None  # |diff| < threshold each step
    max_per_step_rel_diff: Optional[float] = None  # |diff| / max(|b|, eps) < threshold each step
    min_correlation: Optional[float] = None  # Pearson corr across all aligned steps
    allow_missing: bool = False  # don't fail when metric absent from logs


DEFAULT_METRICS = [
    # Loss should be very close (FFA bf16 non-determinism is the main noise floor)
    MetricSpec("actor/pg_loss", max_per_step_rel_diff=0.10, min_correlation=0.95),
    # Reward is in [0, 1] for GSM8K — abs diff is interpretable; rollout sampling
    # adds variance so correlation threshold is more forgiving
    MetricSpec("critic/score/mean", max_per_step_abs_diff=0.05, min_correlation=0.85, allow_missing=True),
    MetricSpec("reward/mean", max_per_step_abs_diff=0.05, min_correlation=0.85, allow_missing=True),
    # KL between actor and ref should match very tightly across the two runs
    MetricSpec("actor/kl_loss", max_per_step_abs_diff=1e-3, allow_missing=True),
    # Grad norm has more bf16 noise — looser threshold
    MetricSpec("actor/grad_norm", max_per_step_rel_diff=0.20, min_correlation=0.85, allow_missing=True),
]


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = (dx2 * dy2) ** 0.5
    if denom == 0:
        return None
    return num / denom


@dataclass
class MetricResult:
    name: str
    n_steps: int
    max_abs_diff: float
    max_rel_diff: float
    correlation: Optional[float]
    has_nan: bool
    passed: bool
    reason: str = ""


def _is_nan_or_inf(x: float) -> bool:
    return x != x or x in (float("inf"), float("-inf"))


def compare_metric(
    spec: MetricSpec,
    base: dict[int, dict[str, float]],
    treat: dict[int, dict[str, float]],
) -> Optional[MetricResult]:
    """Returns None when the metric isn't found in either log and allow_missing=True."""
    shared_steps = sorted(set(base.keys()) & set(treat.keys()))
    if not shared_steps:
        return (
            None if spec.allow_missing else MetricResult(spec.name, 0, 0.0, 0.0, None, False, False, "no shared steps")
        )

    pairs: list[tuple[int, float, float]] = []
    for step in shared_steps:
        b = base[step].get(spec.name)
        t = treat[step].get(spec.name)
        if b is None or t is None:
            continue
        pairs.append((step, b, t))

    if not pairs:
        return (
            None
            if spec.allow_missing
            else MetricResult(spec.name, 0, 0.0, 0.0, None, False, False, "metric not present in logs")
        )

    bs = [p[1] for p in pairs]
    ts = [p[2] for p in pairs]
    has_nan = any(_is_nan_or_inf(b) or _is_nan_or_inf(t) for b, t in zip(bs, ts, strict=True))
    abs_diffs = [abs(t - b) for b, t in zip(bs, ts, strict=True)]
    rel_diffs = [abs(t - b) / max(abs(b), 1e-9) for b, t in zip(bs, ts, strict=True)]
    max_abs = max(abs_diffs)
    max_rel = max(rel_diffs)
    corr = _pearson(bs, ts)

    reasons = []
    if has_nan:
        reasons.append("NaN/Inf in metric values")
    if spec.max_per_step_abs_diff is not None and max_abs > spec.max_per_step_abs_diff:
        reasons.append(f"max_abs_diff {max_abs:.4g} > {spec.max_per_step_abs_diff:.4g}")
    if spec.max_per_step_rel_diff is not None and max_rel > spec.max_per_step_rel_diff:
        reasons.append(f"max_rel_diff {max_rel:.4g} > {spec.max_per_step_rel_diff:.4g}")
    if spec.min_correlation is not None and (corr is None or corr < spec.min_correlation):
        reasons.append(f"correlation {corr} < {spec.min_correlation}")

    passed = not reasons
    return MetricResult(
        name=spec.name,
        n_steps=len(pairs),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        correlation=corr,
        has_nan=has_nan,
        passed=passed,
        reason="; ".join(reasons),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_log", required=True, help="Path to baseline (e.g. dense) training log")
    parser.add_argument("--treatment_log", required=True, help="Path to treatment (e.g. tree) training log")
    parser.add_argument("--label", default="treatment vs baseline", help="Human-readable label for the report")
    parser.add_argument("--output", default=None, help="Optional JSON path for machine-readable results")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to override the default set (uses default thresholds)",
    )
    args = parser.parse_args()

    base = parse_log(args.baseline_log)
    treat = parse_log(args.treatment_log)
    print(f"[COMPARE] {args.label}")
    print(f"[COMPARE]   baseline:  {args.baseline_log}  ({len(base)} steps)")
    print(f"[COMPARE]   treatment: {args.treatment_log} ({len(treat)} steps)")

    if not base or not treat:
        print("[COMPARE] FAIL: one or both logs have zero parsed steps")
        return 1

    if args.metrics:
        wanted = set(args.metrics.split(","))
        specs = [m for m in DEFAULT_METRICS if m.name in wanted]
        if not specs:
            specs = [MetricSpec(name=n) for n in wanted]  # no thresholds, just report numbers
    else:
        specs = DEFAULT_METRICS

    results: list[MetricResult] = []
    print()
    print(f"{'Metric':<30} {'n':>5} {'max_abs':>10} {'max_rel':>10} {'corr':>8} {'NaN':>5}  status")
    print("-" * 90)
    for spec in specs:
        r = compare_metric(spec, base, treat)
        if r is None:
            print(f"{spec.name:<30} {'-':>5} {'-':>10} {'-':>10} {'-':>8} {'-':>5}  SKIP (not in log)")
            continue
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        corr_str = f"{r.correlation:.4f}" if r.correlation is not None else "n/a"
        print(
            f"{r.name:<30} {r.n_steps:>5} {r.max_abs_diff:>10.4g} {r.max_rel_diff:>10.4g} "
            f"{corr_str:>8} {str(r.has_nan):>5}  {status}" + (f"  ({r.reason})" if r.reason else "")
        )

    overall_pass = all(r.passed for r in results) and bool(results)
    print()
    if overall_pass:
        print(f"[COMPARE] {args.label}: ALL METRICS PASS")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"[COMPARE] {args.label}: FAIL — {len(failed)} metric(s) failed: {', '.join(failed)}")

    if args.output:
        payload = {
            "label": args.label,
            "baseline_log": args.baseline_log,
            "treatment_log": args.treatment_log,
            "baseline_steps": len(base),
            "treatment_steps": len(treat),
            "results": [asdict(r) for r in results],
            "overall_pass": overall_pass,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[COMPARE] JSON written to {args.output}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
