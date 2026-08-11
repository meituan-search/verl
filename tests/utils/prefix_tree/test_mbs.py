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
    balance_prefix_tree_blocks,
    greedy_build_tries,
    mbs_groups_from_leaf_idx,
    mbs_groups_from_trie,
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
    trie, _ = greedy_build_tries(seq_lists)
    return trie


def _leaf_idx_from_trie(trie, n_samples):
    """Build canonical leaf_idx: sample i -> its leaf's node_idx."""
    leaf_idx = torch.full((n_samples,), -1, dtype=torch.long)
    for node in trie.nodes:
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node.node_idx
    assert int(leaf_idx.min().item()) >= 0, "trie has samples with no leaf"
    return leaf_idx


def test_balance_prefix_tree_blocks_keeps_trees_whole():
    """Tree-level balance: no tree is split across ranks, all samples covered."""
    samples = _make_samples(3, 4, prefix_len=20, resp_len=10, seed=3)
    trie = _build_trie(samples)
    n = len(samples)
    permutation, partitions, workloads = balance_prefix_tree_blocks(trie, dp_size=2)
    assert sorted(permutation) == list(range(n)), "permutation must cover every sample once"
    # Every tree's samples appear contiguously in the permutation.
    for child in trie.children.values():
        tree_samples = set()
        stack = [child]
        while stack:
            node = stack.pop()
            tree_samples.update(node.sequence_ids)
            stack.extend(node.children.values())
        positions = [permutation.index(s) for s in sorted(tree_samples)]
        assert positions == list(range(min(positions), max(positions) + 1)), (
            f"tree split: samples {sorted(tree_samples)} not contiguous"
        )
    assert len(workloads) == len(trie.children)


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
    canon_leaves = sorted(sorted({int(leaf_idx0[i]) for i in mb}) for mb in canon)
    perm_leaves = sorted(sorted({int(leaf_idx1[i]) for i in mb}) for mb in mbs)
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


def test_mbs_groups_from_leaf_idx_raises_on_non_leaf_ref():
    """leaf_idx pointing to a non-leaf trie node (has children) should raise."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    # Find an internal node (has children) and point leaf_idx at it.
    internal_node = None
    for node in trie.nodes:
        if node.children:
            internal_node = node
            break
    assert internal_node is not None, "trie needs at least one internal node for this test"
    leaf_idx[0] = internal_node.node_idx
    with pytest.raises(ValueError, match="non-leaf"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)


def test_mbs_groups_from_leaf_idx_skips_other_rank_leaves():
    samples = _make_samples(4, 2, prefix_len=20, resp_len=10, seed=7)
    trie = _build_trie(samples)
    full_leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    n = len(samples)
    rank0_leaf_idx = full_leaf_idx[: n // 2].clone()
    rank1_leaf_idx = full_leaf_idx[n // 2 :].clone()
    # Each rank processes its subset without error — 0-based within subset.
    mbs0 = mbs_groups_from_leaf_idx(rank0_leaf_idx, trie, max_token_len=10_000)
    mbs1 = mbs_groups_from_leaf_idx(rank1_leaf_idx, trie, max_token_len=10_000)
    assert sorted(i for mb in mbs0 for i in mb) == list(range(len(rank0_leaf_idx)))
    assert sorted(i for mb in mbs1 for i in mb) == list(range(len(rank1_leaf_idx)))


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
