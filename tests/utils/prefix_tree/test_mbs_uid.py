"""Tests for mbs_groups_from_trie (trie-DFS micro-batch grouping),
dfs_micro_batch_groups (flat-budget grouping over raw sequences), and
trie_group_flat_tokens (per-mbs workload estimator). Merges the former
test_mbs_uid.py + test_mini_batch_prefix_groups.py. The uid-atomic grouping
API was removed in commit 7e73784ec; mbs_groups_from_trie is the current API.
"""
from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import (
    greedy_build_tries,
    mbs_groups_from_trie,
    trie_group_flat_tokens,
)


def _make_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for p in range(n_prompts):
        prefix = torch.randint(0, 151936, (prefix_len,), generator=g)
        for _ in range(rollout_n):
            resp = torch.randint(0, 151936, (resp_len,), generator=g)
            samples.append(torch.cat([prefix, resp]))
    return samples


def _build_trie(samples):
    seq_lists = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]
    tries, _ = greedy_build_tries(seq_lists)
    return tries[0]


def test_mbs_groups_from_trie_covers_all_and_respects_budget():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    budget = 500
    mbs = mbs_groups_from_trie(trie, max_token_len=budget)
    # Every sample covered exactly once
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    # Flat tokens per mb respect budget (no single-sample atomicity exception here
    # since mbs_groups_from_trie has no uid-atomicity constraint)
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= budget


def test_dfs_micro_batch_groups_flat_budget():
    # 4 seqs sharing [1,2,3] (3 tokens) + 2 unique each = 5 raw; flat = 3+4*2 = 11
    seqs = [[1, 2, 3, i, i + 10] for i in range(4)]
    trie = _build_trie(seqs)
    groups = mbs_groups_from_trie(trie, max_token_len=11)
    assert len(groups) == 1 and sorted(groups[0]) == [0, 1, 2, 3]
    # budget=9 -> first 3 fit (3+2+2+2=9), 4th splits
    groups2 = mbs_groups_from_trie(trie, max_token_len=9)
    assert sum(len(g) for g in groups2) == 4


def test_trie_group_flat_tokens_subgroups_and_reorder_stable():
    # root[1,2] -> [3] -> [10]/[11] ; [4,12] ; flat = 2+1+1+1+2 = 7
    seqs = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12]]
    trie = _build_trie(seqs)
    assert trie_group_flat_tokens(list(range(len(seqs))), trie) == 7
    # {0,1}: pays shared ancestor [1,2]+[3] -> 5; {2}: pays root [1,2] -> 4
    assert trie_group_flat_tokens([0, 1], trie) == 5
    assert trie_group_flat_tokens([2], trie) == 4
    # Reorder-stable: sorting mbs by flat tokens preserves the multiset
    seqs2 = [[1, 2, 3, i] for i in range(8)]
    trie2 = _build_trie(seqs2)
    mbs = mbs_groups_from_trie(trie2, max_token_len=10)
    flats = [trie_group_flat_tokens(g, trie2) for g in mbs]
    sorted_mbs = sorted(mbs, key=lambda g: trie_group_flat_tokens(g, trie2))
    assert sorted(flats) == sorted(trie_group_flat_tokens(g, trie2) for g in sorted_mbs)
