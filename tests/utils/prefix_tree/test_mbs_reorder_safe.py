"""Unit tests for the reorder-safe micro-batch grouping API.

Covers ``mbs_groups_from_leaf_idx`` (the LIVE API that reads ``leaf_idx``,
which survives ``DataProto.reorder``) and the end-to-end
``prepare_prefix_tree_micro_batches`` path.  The sibling ``test_mbs_uid.py``
covers only the OLD ``mbs_groups_from_trie`` API whose ``sequence_ids`` go
stale after reorder.
"""
from __future__ import annotations

import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.utils.prefix_tree.dynamic import (
    greedy_build_tries,
    mbs_groups_from_leaf_idx,
    prepare_prefix_tree_micro_batches,
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


def _leaf_idx_from_trie(trie, n_samples):
    """Build canonical leaf_idx: sample i -> its leaf's node_idx."""
    leaf_idx = torch.full((n_samples,), -1, dtype=torch.long)
    for node in trie.nodes:
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node.node_idx
    assert int(leaf_idx.min().item()) >= 0, "trie has samples with no leaf"
    return leaf_idx


def test_mbs_groups_from_leaf_idx_covers_all_and_respects_budget():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=budget)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= budget


def test_mbs_groups_from_leaf_idx_reorder_safe():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx0 = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    perm = torch.randperm(len(samples), generator=torch.Generator().manual_seed(7)).tolist()
    leaf_idx1 = leaf_idx0[perm].clone()
    mbs = mbs_groups_from_leaf_idx(leaf_idx1, trie, max_token_len=budget)
    # (a) union covers all permuted-space positions exactly once
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    # (b) grouping is content-addressed: the trie + budget fully determine
    #     which leaves land together, so the set of leaves in each
    #     permuted-space group must equal the set of leaves in the
    #     corresponding canonical group (no perm remapping needed).
    canon = mbs_groups_from_leaf_idx(leaf_idx0, trie, max_token_len=budget)
    canon_leaves = sorted(
        sorted({int(leaf_idx0[i]) for i in mb}) for mb in canon
    )
    perm_leaves = sorted(
        sorted({int(leaf_idx1[i]) for i in mb}) for mb in mbs
    )
    assert canon_leaves == perm_leaves


def test_mbs_groups_from_leaf_idx_duplicates_stay_together():
    # Two IDENTICAL samples share a trie leaf -> their positions must land in
    # the SAME micro-batch (no singleton group, which would force DP padding).
    base = [1, 2, 3, 4, 5]
    seqs = [base, base, [1, 2, 9, 9, 9], [1, 2, 7, 7, 7]]
    trie = _build_trie(seqs)
    leaf_idx = _leaf_idx_from_trie(trie, len(seqs))
    assert int(leaf_idx[0]) == int(leaf_idx[1])  # identical samples share leaf
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=10_000)
    pos0 = next(i for i, mb in enumerate(mbs) if 0 in mb)
    pos1 = next(i for i, mb in enumerate(mbs) if 1 in mb)
    assert pos0 == pos1, f"duplicate samples split: mb0={pos0} mb1={pos1}"


def test_mbs_groups_from_leaf_idx_raises_on_orphan():
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_idx[1] = -1  # orphan: sample 1 has no leaf
    with pytest.raises(ValueError, match="no leaf assigned"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)


def test_mbs_groups_from_leaf_idx_raises_on_uncovered_trie_leaf():
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    # Remap one sample onto another's leaf -> one trie leaf uncovered.
    leaf_idx[0] = int(leaf_idx[1])
    with pytest.raises(ValueError, match="out of sync"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)


def test_prepare_prefix_tree_micro_batches_attaches_subtrie():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    n = len(samples)
    seq_len = max(len(s) for s in samples)
    input_ids = torch.zeros((n, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    for i, s in enumerate(samples):
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = 1
    budget = 500
    td = tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "leaf_idx": leaf_idx,
        },
        non_tensor_dict={
            "prefix_tree": trie,
            "use_dynamic_bsz": True,
            "use_prefix_tree": True,
            "sp_size": 1,
            "force_group_size": 1,
            "max_token_len_per_gpu": budget,
        },
    )
    micro_batches, batch_idx_list = prepare_prefix_tree_micro_batches(td, sp_size=1)
    assert len(micro_batches) == len(batch_idx_list)
    for mb, mb_idx in zip(micro_batches, batch_idx_list, strict=False):
        subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
        assert subtree is not None, "prefix_tree_subtree not attached"
        mb_leaves = sorted(int(x) for x in mb["leaf_idx"].tolist())
        sub_leaves = sorted(subtree.leaf_node_ids)
        assert sub_leaves == mb_leaves, f"{sub_leaves} != {mb_leaves}"
