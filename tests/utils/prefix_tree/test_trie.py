"""Trie/dynamic-builder unit tests: greedy_build_tries, build_tree_dynamic,
convert_trie_to_tree_node, trie_dfs_leaf_order, subtrie_view, and the
flat layout / attention-spec output from build_layout_from_tree_node.

Merges the former test_trie.py + test_dynamic.py + the dfs_leaf_order tests
from test_mini_batch_prefix_groups.py.
"""
from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import (
    TrieNode,
    build_tree_dynamic,
    convert_trie_to_tree_node,
    dfs_leaf_order,
    greedy_build_tries,
    subtrie_view,
)
from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _build_trie(sequences, max_tokens=None):
    tries, _ = greedy_build_tries(sequences)
    return tries[0]


def test_greedy_build_tries_and_dfs_leaf_order():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    tries, _ = greedy_build_tries(seqs)
    assert len(tries) == 1 and tries[0].is_root
    order = dfs_leaf_order(seqs, tries[0])
    assert set(order) == {0, 1, 2} and len(order) == 3
    # DFS places each prefix group adjacent: A1,B1,A2,B2 -> A's adjacent, B's adjacent
    raw = [[1, 2, 10], [5, 6, 20], [1, 2, 11], [5, 6, 21]]
    o2 = dfs_leaf_order(raw, _build_trie(raw))
    a = sorted([o2.index(0), o2.index(2)])
    b = sorted([o2.index(1), o2.index(3)])
    assert a[1] - a[0] == 1 and b[1] - b[0] == 1


def test_subtrie_view_all_subset_and_empty():
    trie = _build_trie([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]])
    sub_all = subtrie_view(trie, {0, 1, 2})
    assert isinstance(sub_all, PrefixSubTrie)
    assert len(sub_all.leaf_to_sample) == 3 and len(sub_all.nodes[0].input_ids) == 2

    sub_one = subtrie_view(trie, {0})
    assert sub_one.leaf_to_sample == [0] and len(sub_one.nodes) == 3  # [1,2]+[3]+[4]

    sub_two = subtrie_view(trie, {0, 2})
    assert set(sub_two.leaf_to_sample) == {0, 2}
    for lid in sub_two.leaf_node_ids:
        assert 0 <= lid < len(trie.nodes)

    assert subtrie_view(trie, set()) is None and subtrie_view(trie, {99}) is None


def test_build_tree_dynamic_and_none_cases():
    # depth-2: 3 samples share [10,11]
    s2 = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    r2 = build_tree_dynamic(s2)
    assert r2 is not None and len(r2.nodes[0].input_ids) == 2 and sorted(r2.leaf_to_sample) == [0, 1, 2]
    # None cases: no sharing (multiple roots) and empty.
    # Single sample and duplicates now return a trie (not None) — source changed.
    assert build_tree_dynamic([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]) is None
    assert build_tree_dynamic([]) is None


def test_convert_trie_to_tree_node_normal_and_multi_root():
    trie = _build_trie([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]])
    r = convert_trie_to_tree_node(trie)
    assert r is not None and len(r.nodes[0].input_ids) == 2 and len(r.leaf_to_sample) == 3
    multi = TrieNode()  # Multi-children root (no shared prefix) -> None
    multi.children[1] = TrieNode(input_ids=[1], sequence_ids=[0])
    multi.children[2] = TrieNode(input_ids=[2], sequence_ids=[1])
    assert convert_trie_to_tree_node(multi) is None


def test_layout_token_conservation_and_zero_length_leaf_skipped():
    # depth-2: 3 samples share [10,11]; donation repeats the last prefix token
    # per child, so packed = 3 + 3*(1 + 2) = ... use invariant, not exact count.
    s2 = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    p2 = build_layout_from_tree_node(s2, build_tree_dynamic(s2))
    assert p2.tree_packed_tokens.shape[0] >= 8  # at least the raw 8 tokens
    assert p2.prefix_range[0] == 0 and p2.prefix_range[1] >= 1
    assert len(p2.leaf_ranges) == 3
    # Nested A/AB/ABC prefixes -> packed contains all unique tokens in order
    nested = [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])]
    pn = build_layout_from_tree_node(nested, build_tree_dynamic(nested))
    # All 6 unique tokens appear; prefix [1,2] is first
    assert list(pn.tree_packed_tokens[:2].tolist()) == [1, 2]
    assert set(pn.tree_packed_tokens.tolist()) == {1, 2, 3, 4, 5, 6}
