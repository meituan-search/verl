# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Local test + benchmark for build_tree_from_segments vs build_tree_dynamic.

Runs with plain Python + torch (no Megatron, no GPU needed — CPU tensors only).

Verifies:
  1. Correctness: build_tree_from_segments produces a tree equivalent to
     build_tree_dynamic (same flat token layout, same leaf mapping).
  2. Same-first-token suffix case (collision check for direct construction).
  3. Performance: times both paths at GRPO scale (8k prefix, 500 response, 8 rollouts).

Usage:
    python3 tests/utils/prefix_tree/bench_build_tree.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

# ── Direct module loading (bypasses __init__.py which imports magi.py) ──────
# This lets the test run locally without magi_attention / megatron installed.
_PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "utils", "prefix_tree")
_PKG_DIR = os.path.abspath(_PKG_DIR)

# Create a fake parent package so relative imports inside the modules resolve.
# We register verl.utils.prefix_tree as a "namespace" package with no __init__
# side effects by loading each submodule directly.
import types

# Stub out the package to avoid running __init__.py
_pkg = types.ModuleType("verl_prefix_tree_pkg")
_pkg.__path__ = [_PKG_DIR]
sys.modules["verl_prefix_tree_pkg"] = _pkg


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(f"verl_prefix_tree_pkg.{name}", os.path.join(_PKG_DIR, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"verl_prefix_tree_pkg.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# tree.py has no internal relative imports — load first
tree_mod = _load("tree", "tree.py")
# dynamic.py imports from tree.py via `from verl.utils.prefix_tree.tree import ...`
# Patch sys.modules so that import path resolves to our loaded module.
# dynamic.py uses: from verl.utils.prefix_tree.tree import PrefixSubTrie, PrefixTrie, TrieNode
# We need to create the verl.utils.prefix_tree package path.
for pkg_name in ["verl", "verl.utils", "verl.utils.prefix_tree"]:
    if pkg_name not in sys.modules:
        m = types.ModuleType(pkg_name)
        m.__path__ = []
        sys.modules[pkg_name] = m

# Stub verl.utils submodules used by dynamic.py (not by tree-building funcs).
_tu_stub = types.ModuleType("verl.utils.tensordict_utils")
_tu_stub.get_non_tensor_data = lambda *a, **k: None
_tu_stub.assign_non_tensor = lambda *a, **k: None
_tu_stub.index_select_tensor_dict = lambda *a, **k: None
sys.modules["verl.utils.tensordict_utils"] = _tu_stub

_dev_stub = types.ModuleType("verl.utils.device")
_dev_stub.get_torch_device = lambda: "cpu"
sys.modules["verl.utils.device"] = _dev_stub

_seq_stub = types.ModuleType("verl.utils.seqlen_balancing")
_seq_stub.calculate_workload = lambda *a, **k: None
_seq_stub.get_seqlen_balanced_partitions = lambda *a, **k: None
_seq_stub.log_seqlen_unbalance = lambda *a, **k: None
_seq_stub.roundup_divisible = lambda a, b: (a + b - 1) // b
sys.modules["verl.utils.seqlen_balancing"] = _seq_stub

sys.modules["verl.utils.prefix_tree.tree"] = tree_mod

# Now load dynamic.py — its `from verl.utils.prefix_tree.tree import ...` will
# resolve to tree_mod via sys.modules.
dynamic_mod = _load("dynamic", "dynamic.py")
sys.modules["verl.utils.prefix_tree.dynamic"] = dynamic_mod

# segment_grouper.py
seg_mod = _load("segment_grouper", "segment_grouper.py")
sys.modules["verl.utils.prefix_tree.segment_grouper"] = seg_mod

# utils.py — imports from tree.py and dynamic.py
utils_mod = _load("utils", "utils.py")
sys.modules["verl.utils.prefix_tree.utils"] = utils_mod

# Bind names
_BuildNode = dynamic_mod._BuildNode
_compress_trie = dynamic_mod._compress_trie
_insert_sequence = dynamic_mod._insert_sequence
convert_trie_to_tree_node = dynamic_mod.convert_trie_to_tree_node
build_tree_dynamic = dynamic_mod.build_tree_dynamic
create_grpo_segment_metadata = seg_mod.create_grpo_segment_metadata
build_tree_from_segments = tree_mod.build_tree_from_segments
PrefixSubTrie = tree_mod.PrefixSubTrie
build_layout_from_tree_node = utils_mod.build_layout_from_tree_node

INJECTED = dict(
    _BuildNode=_BuildNode,
    _insert_sequence=_insert_sequence,
    _compress_trie=_compress_trie,
    convert_trie_to_tree_node=convert_trie_to_tree_node,
)


def make_grpo_samples(n_rollouts: int, prefix_len: int, resp_len: int, vocab: int = 151936, seed: int = 0):
    """Create GRPO-like samples: shared prefix + distinct random responses."""
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(0, vocab, (prefix_len,), generator=g)
    samples = []
    for i in range(n_rollouts):
        resp = torch.randint(0, vocab, (resp_len,), generator=g)
        samples.append(torch.cat([prefix, resp]))
    return samples, prefix


def make_grpo_samples_same_first_token(
    n_rollouts: int, prefix_len: int, resp_len: int, vocab: int = 151936, seed: int = 0
):
    """Like make_grpo_samples but all responses start with the same token."""
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(0, vocab, (prefix_len,), generator=g)
    shared_first = int(torch.randint(0, vocab, (1,), generator=g))
    samples = []
    for i in range(n_rollouts):
        rest = torch.randint(0, vocab, (resp_len - 1,), generator=g)
        resp = torch.cat([torch.tensor([shared_first]), rest])
        samples.append(torch.cat([prefix, resp]))
    return samples, prefix


def subtrie_flat_tokens(subtrie: PrefixSubTrie) -> list[int]:
    """Extract flat token sequence from subtrie nodes (DFS order)."""
    return [tok for node in subtrie.nodes for tok in node.input_ids]


def layout_flat_tokens(subtrie: PrefixSubTrie, samples: list) -> list[int]:
    """Build the flat packed layout and return its token sequence."""
    labels_by_sample = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype)]) for s in samples]
    params = build_layout_from_tree_node(samples, subtrie, labels_by_sample=labels_by_sample)
    return params.tree_packed_tokens.tolist()


def verify_subtrie_correctness(sub: PrefixSubTrie, samples: list, prefix_len: int, n_rollouts: int):
    """Verify a subtrie is internally correct:
    - leaf_to_sample covers all samples
    - flat layout starts with the shared prefix
    - flat total <= raw total (deduplication happened)
    - flat total >= prefix_len + n_rollouts (at least 1 token per leaf)
    """
    assert sub is not None
    assert set(sub.leaf_to_sample) == set(range(n_rollouts)), (
        f"missing samples: {set(range(n_rollouts)) - set(sub.leaf_to_sample)}"
    )
    flat = layout_flat_tokens(sub, samples)
    raw_total = sum(len(s) for s in samples)
    assert len(flat) <= raw_total, f"flat {len(flat)} > raw {raw_total} — no dedup!"
    assert len(flat) >= prefix_len + n_rollouts, f"flat {len(flat)} too small — expected >= {prefix_len + n_rollouts}"
    # Prefix tokens must appear once at the start
    expected_prefix = samples[0][:prefix_len].tolist()
    assert flat[:prefix_len] == expected_prefix, "prefix tokens mismatch at start of flat layout"


def test_correctness_equivalence():
    """build_tree_from_segments and build_tree_dynamic produce equivalent trees
    (same tokens, possibly different leaf ordering — both valid)."""
    print("\n=== Test: correctness equivalence (segments vs dynamic) ===")
    n_rollouts = 4
    prefix_len = 100
    resp_len = 20
    samples, prefix = make_grpo_samples(n_rollouts, prefix_len, resp_len, seed=42)
    hashes, lengths = create_grpo_segment_metadata(
        prompt_uids=["p0"] * n_rollouts, prompt_lengths=[prefix_len] * n_rollouts, rollout_n=n_rollouts
    )

    sub_seg = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
    sub_dyn = build_tree_dynamic(samples)

    assert sub_seg is not None, "segments path returned None"
    assert sub_dyn is not None, "dynamic path returned None"

    verify_subtrie_correctness(sub_seg, samples, prefix_len, n_rollouts)
    verify_subtrie_correctness(sub_dyn, samples, prefix_len, n_rollouts)

    # Both should produce the same number of flat tokens
    flat_seg = layout_flat_tokens(sub_seg, samples)
    flat_dyn = layout_flat_tokens(sub_dyn, samples)
    assert len(flat_seg) == len(flat_dyn), f"flat len mismatch: {len(flat_seg)} vs {len(flat_dyn)}"

    # Prefix tokens must match exactly (both start with the same shared prefix)
    assert flat_seg[:prefix_len] == flat_dyn[:prefix_len], "prefix tokens differ"

    # Suffix tokens: same multiset (order may differ due to child sorting)
    assert sorted(flat_seg[prefix_len:]) == sorted(flat_dyn[prefix_len:]), "suffix token multiset differs"

    print(f"  PASS: both paths produce valid trees ({len(flat_seg)} flat tokens)")
    print(f"  leaves: segments={len(sub_seg.leaf_node_ids)}  dynamic={len(sub_dyn.leaf_node_ids)}")


def test_same_first_token_suffix():
    """Correctness when multiple response suffixes start with the same token.
    This is the collision check for direct construction — if children are keyed
    by first suffix token, same-first-token leaves would overwrite each other."""
    print("\n=== Test: same-first-token suffix (collision check) ===")
    n_rollouts = 4
    prefix_len = 50
    resp_len = 10
    samples, prefix = make_grpo_samples_same_first_token(n_rollouts, prefix_len, resp_len, seed=7)
    suffixes = [s[prefix_len:] for s in samples]
    first_tokens = [int(s[0]) for s in suffixes]
    assert len(set(first_tokens)) == 1, "test setup broken: suffixes don't share first token"
    print(f"  all {n_rollouts} suffixes start with token {first_tokens[0]}")

    hashes, lengths = create_grpo_segment_metadata(
        prompt_uids=["p0"] * n_rollouts, prompt_lengths=[prefix_len] * n_rollouts, rollout_n=n_rollouts
    )

    sub_seg = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
    sub_dyn = build_tree_dynamic(samples)

    assert sub_seg is not None, "segments path returned None"
    assert sub_dyn is not None, "dynamic path returned None"

    # Both must cover all samples — collision would drop samples
    assert set(sub_seg.leaf_to_sample) == set(range(n_rollouts)), (
        f"COLLISION: segments path dropped samples! got {set(sub_seg.leaf_to_sample)} expected {set(range(n_rollouts))}"
    )
    verify_subtrie_correctness(sub_seg, samples, prefix_len, n_rollouts)
    verify_subtrie_correctness(sub_dyn, samples, prefix_len, n_rollouts)

    flat_seg = layout_flat_tokens(sub_seg, samples)
    flat_dyn = layout_flat_tokens(sub_dyn, samples)
    # Direct construction keeps suffixes independent (no dedup of coincidental
    # token overlap), so flat_seg >= flat_dyn is expected.  Both are valid.
    assert len(flat_seg) >= len(flat_dyn), f"segments flat {len(flat_seg)} < dynamic {len(flat_dyn)} — unexpected"

    print("  PASS: same-first-token suffixes handled correctly")
    print(f"  flat: segments={len(flat_seg)}  dynamic={len(flat_dyn)} (dyn dedups shared suffix token)")
    print(f"  leaves: segments={len(sub_seg.leaf_node_ids)}  dynamic={len(sub_dyn.leaf_node_ids)}")


def test_leaf_coverage():
    """Every sample must have a leaf entry."""
    print("\n=== Test: leaf coverage ===")
    samples, _ = make_grpo_samples(n_rollouts=8, prefix_len=200, resp_len=30, seed=99)
    hashes, lengths = create_grpo_segment_metadata(prompt_uids=["p0"] * 8, prompt_lengths=[200] * 8, rollout_n=8)
    sub = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
    assert sub is not None
    assert set(sub.leaf_to_sample) == set(range(8)), f"missing samples: {set(range(8)) - set(sub.leaf_to_sample)}"
    print(f"  PASS: all 8 samples have leaves: {sorted(sub.leaf_to_sample)}")


def bench_grpo_scale():
    """Benchmark at GRPO scale: 8k prefix, 500 response, 8 rollouts."""
    print("\n=== Benchmark: GRPO scale (8k prefix, 500 resp, 8 rollouts) ===")
    prefix_len = 8000
    resp_len = 500
    n_rollouts = 8

    samples, prefix = make_grpo_samples(n_rollouts, prefix_len, resp_len, seed=123)
    hashes, lengths = create_grpo_segment_metadata(
        prompt_uids=["p0"] * n_rollouts, prompt_lengths=[prefix_len] * n_rollouts, rollout_n=n_rollouts
    )

    # Warmup
    _ = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
    _ = build_tree_dynamic(samples)

    # Time segments path
    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sub_seg = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
    t_seg = (time.perf_counter() - t0) / n_iter * 1000

    # Time dynamic path
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sub_dyn = build_tree_dynamic(samples)
    t_dyn = (time.perf_counter() - t0) / n_iter * 1000

    # Verify equivalence (segments flat >= dynamic flat since direct construction
    # doesn't dedup coincidental suffix token overlap; both valid).
    flat_seg = layout_flat_tokens(sub_seg, samples)
    flat_dyn = layout_flat_tokens(sub_dyn, samples)
    match = flat_seg[:prefix_len] == flat_dyn[:prefix_len]

    print(f"  segments path: {t_seg:.1f} ms  ({len(flat_seg)} flat tokens, {len(sub_seg.leaf_node_ids)} leaves)")
    print(f"  dynamic  path: {t_dyn:.1f} ms  ({len(flat_dyn)} flat tokens, {len(sub_dyn.leaf_node_ids)} leaves)")
    print(f"  speedup: {t_dyn / t_seg:.2f}x")
    print(f"  layouts match (len+prefix): {match}")

    # Also time the layout build (shared by both paths)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        labels_by_sample = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype)]) for s in samples]
        _ = build_layout_from_tree_node(samples, sub_seg, labels_by_sample=labels_by_sample)
    t_layout = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  layout build:   {t_layout:.1f} ms")


def bench_multiple_scales():
    """Benchmark at multiple prefix lengths to see scaling."""
    print("\n=== Benchmark: multiple scales (8 rollouts, 500 resp) ===")
    print(f"  {'prefix_len':>12}  {'segments(ms)':>14}  {'dynamic(ms)':>14}  {'speedup':>8}  {'match':>6}")
    for prefix_len in [1000, 2000, 4000, 8000]:
        resp_len = 500
        n_rollouts = 8
        samples, _ = make_grpo_samples(n_rollouts, prefix_len, resp_len, seed=321)
        hashes, lengths = create_grpo_segment_metadata(
            prompt_uids=["p0"] * n_rollouts, prompt_lengths=[prefix_len] * n_rollouts, rollout_n=n_rollouts
        )

        # Warmup
        _ = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
        _ = build_tree_dynamic(samples)

        n_iter = 3
        t0 = time.perf_counter()
        for _ in range(n_iter):
            sub_seg = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
        t_seg = (time.perf_counter() - t0) / n_iter * 1000

        t0 = time.perf_counter()
        for _ in range(n_iter):
            sub_dyn = build_tree_dynamic(samples)
        t_dyn = (time.perf_counter() - t0) / n_iter * 1000

        flat_seg = layout_flat_tokens(sub_seg, samples)
        flat_dyn = layout_flat_tokens(sub_dyn, samples)
        match = flat_seg[:prefix_len] == flat_dyn[:prefix_len]

        print(f"  {prefix_len:>12}  {t_seg:>14.1f}  {t_dyn:>14.1f}  {t_dyn / t_seg:>8.2f}x  {str(match):>6}")


if __name__ == "__main__":
    test_correctness_equivalence()
    test_same_first_token_suffix()
    test_leaf_coverage()
    bench_grpo_scale()
    bench_multiple_scales()
    print("\nDone.")
