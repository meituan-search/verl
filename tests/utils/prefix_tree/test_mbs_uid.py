#!/usr/bin/env python3
"""Local test for mbs_groups_from_uid — verifies uid atomicity + balance.

Runs with plain Python + torch (no Megatron, no GPU needed).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import torch

# ── Direct module loading (bypasses __init__.py) ─────────────────────────────
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "utils", "prefix_tree"))
for pkg_name in ["verl", "verl.utils", "verl.utils.prefix_tree"]:
    if pkg_name not in sys.modules:
        m = types.ModuleType(pkg_name)
        m.__path__ = []
        sys.modules[pkg_name] = m

_tu = types.ModuleType("verl.utils.tensordict_utils")
_tu.get_non_tensor_data = lambda *a, **k: None
_tu.assign_non_tensor = lambda *a, **k: None
_tu.index_select_tensor_dict = lambda *a, **k: None
sys.modules["verl.utils.tensordict_utils"] = _tu

_dev = types.ModuleType("verl.utils.device")
_dev.get_torch_device = lambda: type("D", (), {"current_device": lambda s: "cpu"})()
sys.modules["verl.utils.device"] = _dev

_seq = types.ModuleType("verl.utils.seqlen_balancing")
_seq.calculate_workload = lambda x: 24576 * x + x ** 2
_seq.get_seqlen_balanced_partitions = lambda seqlen_list, k_partitions, equal_size=False: _kk(seqlen_list, k_partitions, equal_size)
_seq.log_seqlen_unbalance = lambda *a, **k: None
_seq.roundup_divisible = lambda a, b: (a + b - 1) // b

def _kk(seqlen_list, k_partitions, equal_size=False):
    """Simple Karmarkar-Karp stub: largest-differencing greedy."""
    items = sorted([(s, i) for i, s in enumerate(seqlen_list)], reverse=True)
    partitions = [[] for _ in range(k_partitions)]
    sums = [0] * k_partitions
    for s, i in items:
        min_idx = sums.index(min(sums))
        partitions[min_idx].append(i)
        sums[min_idx] += s
    return partitions

sys.modules["verl.utils.seqlen_balancing"] = _seq


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(f"verl.utils.prefix_tree.{name}", os.path.join(_PKG_DIR, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"verl.utils.prefix_tree.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


tree_mod = _load("tree", "tree.py")
sys.modules["verl.utils.prefix_tree.tree"] = tree_mod
dyn_mod = _load("dynamic", "dynamic.py")
sys.modules["verl.utils.prefix_tree.dynamic"] = dyn_mod

mbs_groups_from_uid = dyn_mod.mbs_groups_from_uid
trie_group_flat_tokens = dyn_mod.trie_group_flat_tokens
greedy_build_tries = dyn_mod.greedy_build_tries
_build_tree_dynamic = dyn_mod.build_tree_dynamic


def make_samples(n_prompts: int, rollout_n: int, prefix_len: int, resp_len: int, seed: int = 0):
    """Create GRPO-like samples: n_prompts × rollout_n, each prompt has shared prefix."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    uids = []
    for p in range(n_prompts):
        prefix = torch.randint(0, 151936, (prefix_len,), generator=g)
        uid = f"prompt_{p}"
        for r in range(rollout_n):
            resp = torch.randint(0, 151936, (resp_len,), generator=g)
            samples.append(torch.cat([prefix, resp]))
            uids.append(uid)
    return samples, uids


def test_uid_atomicity():
    """Same-uid samples must all land in the same micro-batch."""
    print("\n=== Test: uid atomicity ===")
    samples, uids = make_samples(n_prompts=4, rollout_n=4, prefix_len=100, resp_len=20, seed=42)
    trie, _ = greedy_build_tries([s.tolist() for s in samples], max_tokens_per_tree=sum(len(s) for s in samples) * 10)
    trie = trie[0]

    max_token_len = 500  # flat tokens per mb
    mbs = mbs_groups_from_uid(uids, trie, max_token_len)

    # Check atomicity: all samples of each uid in the same mb
    uid_to_mb = {}
    for mb_i, mb in enumerate(mbs):
        for idx in mb:
            uid = uids[idx]
            if uid in uid_to_mb:
                assert uid_to_mb[uid] == mb_i, f"uid {uid} split across mbs {uid_to_mb[uid]} and {mb_i}"
            else:
                uid_to_mb[uid] = mb_i

    # All samples covered
    all_indices = sorted(i for mb in mbs for i in mb)
    assert all_indices == list(range(len(samples))), f"missing samples: {set(range(len(samples))) - set(all_indices)}"

    print(f"  PASS: {len(mbs)} mbs, all uids atomic, all {len(samples)} samples covered")
    for i, mb in enumerate(mbs):
        flats = [trie_group_flat_tokens(mb, trie)]
        uids_in = set(uids[j] for j in mb)
        print(f"  mb {i}: {len(mb)} samples, {len(uids_in)} uids, flat={flats[0]}")


def test_balance():
    """Flat tokens should be more balanced than DFS first-fit."""
    print("\n=== Test: balance vs DFS ===")
    # Varying prefix lengths to create imbalance opportunity — DFS fills
    # sequentially so a big prompt followed by small ones puts the big one
    # alone in the first mb; KK balances across all mbs.
    g = torch.Generator().manual_seed(7)
    samples = []
    uids = []
    prompt_lens = [400, 100, 100, 100, 400, 100, 100, 100]
    rollout_n = 8
    resp_len = 30
    for p, plen in enumerate(prompt_lens):
        prefix = torch.randint(0, 151936, (plen,), generator=g)
        for r in range(rollout_n):
            resp = torch.randint(0, 151936, (resp_len,), generator=g)
            samples.append(torch.cat([prefix, resp]))
            uids.append(f"prompt_{p}")

    trie, _ = greedy_build_tries([s.tolist() for s in samples], max_tokens_per_tree=sum(len(s) for s in samples) * 10)
    trie = trie[0]

    max_token_len = 2000  # flat tokens per mb

    # Uid-based
    mbs_uid = mbs_groups_from_uid(uids, trie, max_token_len)
    flats_uid = [trie_group_flat_tokens(mb, trie) for mb in mbs_uid]

    # DFS first-fit (existing)
    mbs_dfs = dyn_mod.mbs_groups_from_trie(trie, max_token_len, use_n2_cost=False)
    flats_dfs = [trie_group_flat_tokens(mb, trie) for mb in mbs_dfs]

    imb_uid = (max(flats_uid) - min(flats_uid)) / max(flats_uid) if max(flats_uid) > 0 else 0
    imb_dfs = (max(flats_dfs) - min(flats_dfs)) / max(flats_dfs) if max(flats_dfs) > 0 else 0

    print(f"  uid:  n_mb={len(mbs_uid)} flats={flats_uid} imb={imb_uid:.3f}")
    print(f"  dfs:  n_mb={len(mbs_dfs)} flats={flats_dfs} imb={imb_dfs:.3f}")
    assert imb_uid <= imb_dfs + 0.01, f"uid should be more balanced: uid={imb_uid:.3f} dfs={imb_dfs:.3f}"
    print(f"  PASS: uid imbalance ({imb_uid:.3f}) <= dfs ({imb_dfs:.3f})")


def test_budget_respected():
    """Each mb's flat tokens should respect max_token_len (with possible oversize for big uids)."""
    print("\n=== Test: budget respected ===")
    samples, uids = make_samples(n_prompts=6, rollout_n=4, prefix_len=300, resp_len=40, seed=99)
    trie, _ = greedy_build_tries([s.tolist() for s in samples], max_tokens_per_tree=sum(len(s) for s in samples) * 10)
    trie = trie[0]

    max_token_len = 1500
    mbs = mbs_groups_from_uid(uids, trie, max_token_len)
    flats = [trie_group_flat_tokens(mb, trie) for mb in mbs]

    # Each mb should be <= max_token_len (unless a single uid-group exceeds it)
    uid_group_flats = []
    for p in range(6):
        group = [i for i, u in enumerate(uids) if u == f"prompt_{p}"]
        uid_group_flats.append(trie_group_flat_tokens(group, trie))

    for i, f in enumerate(flats):
        if f > max_token_len:
            # Only allowed if a single uid-group exceeds the budget
            assert max(uid_group_flats) > max_token_len, (
                f"mb {i} flat={f} > budget={max_token_len} but no single uid-group exceeds it"
            )
            print(f"  mb {i}: flat={f} (oversize due to large uid-group, acceptable)")
        else:
            print(f"  mb {i}: flat={f} <= {max_token_len} ✓")

    print(f"  PASS: all mbs respect budget (or oversize only from single large uid-group)")


if __name__ == "__main__":
    test_uid_atomicity()
    test_balance()
    test_budget_respected()
    print("\nDone.")
