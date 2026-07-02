#!/usr/bin/env python3
# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
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
"""Verify bit-identical label refactor: new shift-based derivation == old per-sample roll.

Run:  python3 test_label_refactor.py
      ray job submit --working-dir . --runtime-env '{"env_vars":{"PYTHONPATH":"."}}' \
          -- python3 test_label_refactor.py
"""
import sys
import torch
from verl.utils.prefix_tree.dynamic import greedy_build_tries, subtrie_view
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


# ── helpers ──────────────────────────────────────────────────────────────────

def make_subtrie(seqs: list[list[int]]):
    total = sum(len(s) for s in seqs)
    tries, _ = greedy_build_tries(seqs, max_tokens_per_tree=total * 10)
    return subtrie_view(tries[0], set(range(len(seqs))))


def reconstruct_per_sample_labels(params, n_samples: int) -> list[torch.Tensor]:
    """Expand tree_packed_labels back to per-sample using ancestor chain."""
    flat = params.tree_packed_labels
    result = [None] * n_samples
    for leaf_idx, sample_idx in enumerate(params.leaf_to_sample):
        anc = params._leaf_ancestor_ranges[leaf_idx]
        ls, le = params.leaf_ranges[leaf_idx]
        parts = [flat[a:b] for a, b in anc] + [flat[ls:le]]
        result[sample_idx] = torch.cat(parts) if parts else torch.empty(0, dtype=flat.dtype)
    return result


def check(name: str, seqs: list[list[int]]) -> bool:
    samples = [torch.tensor(s, dtype=torch.long) for s in seqs]
    expected = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype)]) for s in samples]

    subtrie = make_subtrie(seqs)
    if subtrie is None:
        print(f"  SKIP {name}: no prefix sharing")
        return True

    params = build_layout_from_tree_node(samples, subtrie)
    got = reconstruct_per_sample_labels(params, len(samples))

    ok = True
    for i, (exp, actual) in enumerate(zip(expected, got)):
        if not torch.equal(exp, actual):
            print(f"  FAIL {name} sample {i}: got {actual.tolist()} want {exp.tolist()}")
            ok = False
    if ok:
        flat = params.tree_packed_tokens.numel()
        raw  = sum(len(s) for s in seqs)
        print(f"  PASS {name}: {len(seqs)} samples, flat={flat}, raw={raw}, "
              f"dedup={1-flat/raw:.3f}")
    return ok


# ── test cases ────────────────────────────────────────────────────────────────

TESTS: list[tuple[str, list[list[int]]]] = []

# 1. Single group — basic 2-level tree
TESTS.append(("single_group_2_leaves", [
    [1, 2, 3, 10, 11, 12],
    [1, 2, 3, 20, 21, 22],
]))

# 2. Single group — 8 leaves (typical GRPO batch)
_PREFIX = [100, 200, 300, 400, 500]
TESTS.append(("single_group_8_leaves", [
    _PREFIX + [1000 + r * 10 + j for j in range(3)]
    for r in range(8)
]))

# 3. Two independent prefix groups (multi-root)
TESTS.append(("two_groups_4_each", [
    [1, 2, 10 + r] for r in range(4)
] + [
    [9, 8, 50 + r] for r in range(4)
]))

# 4. 10 groups × 8 rollouts (the main integration case)
_G10 = []
for g in range(10):
    prefix = [g * 1000 + i for i in range(5)]
    for r in range(8):
        _G10.append(prefix + [g * 1000 + 500 + r * 10 + j for j in range(3)])
TESTS.append(("10_groups_x_8_rollouts", _G10))

# 5. Duplicate sequences (zero-length leaf)
TESTS.append(("duplicates", [
    [1, 2, 3, 10, 11],
    [1, 2, 3, 10, 11],  # exact duplicate
    [1, 2, 3, 20, 21],
]))

# 6. Longer prefix, shorter response
TESTS.append(("long_prefix_short_resp", [
    list(range(1, 21)) + [100 + r] for r in range(4)
]))

# 7. Single-token prefix (boundary shift edge case)
TESTS.append(("single_token_prefix", [
    [99, 10, 11, 12],
    [99, 20, 21, 22],
    [99, 30, 31, 32],
]))

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, seqs in TESTS:
        ok = check(name, seqs)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'ALL PASSED' if failed == 0 else 'SOME FAILED'} ({passed}/{passed+failed})")
    sys.exit(0 if failed == 0 else 1)
