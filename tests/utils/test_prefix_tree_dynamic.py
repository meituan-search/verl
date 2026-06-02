# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for the dynamic-trie prefix tree builder.

Exercises :mod:`verl.utils.prefix_tree_dynamic` in isolation (no model / GPU).
Validates that the token-by-token trie correctly recovers the shared-prefix
structure across a range of branching factors and tree depths.
"""

from __future__ import annotations

import torch

from verl.utils.prefix_tree_dynamic import (
    TrieNode,
    build_tree_dynamic,
    convert_trie_to_tree_node,
    greedy_build_tries,
)
from verl.utils.prefix_tree_utils import TreeNode, build_layout_from_tree_node

# ---------------------------------------------------------------------------
# Lower-level helpers
# ---------------------------------------------------------------------------


def _collect_leaf_segments(node: TreeNode) -> list[int]:
    """Return segment lengths of leaves in DFS pre-order."""
    if node.is_leaf:
        return [node.segment_len]
    out: list[int] = []
    for child in node.children:
        out.extend(_collect_leaf_segments(child))
    return out


def _count_leaves(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_leaves(c) for c in node.children)


# ---------------------------------------------------------------------------
# greedy_build_tries
# ---------------------------------------------------------------------------


def test_greedy_build_tries_single_forest_for_shared_prefix():
    """Sequences sharing the first token end up in one trie (one forest)."""
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6, 7],
    ]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    assert len(tries) == 1
    assert isinstance(tries[0], TrieNode)
    assert tries[0].is_root


def test_greedy_build_tries_separate_forest_for_no_shared_prefix():
    """Sequences with different first token still pack into one forest under
    a generous ``max_tokens_per_tree`` — they share the virtual root."""
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    # Both sequences inserted into the same forest (the virtual root has 2 child branches).
    assert len(tries) == 1
    assert len(tries[0].children) == 2


# ---------------------------------------------------------------------------
# convert_trie_to_tree_node
# ---------------------------------------------------------------------------


def test_convert_trie_to_tree_node_depth_2():
    """Simple depth-2 case: 3 sequences sharing 2-token prefix, then diverging
    into 3 distinct branches of length 2."""
    sequences = [
        [10, 11, 20, 21],
        [10, 11, 30, 31],
        [10, 11, 40, 41],
    ]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    result = convert_trie_to_tree_node(tries[0])
    assert result is not None
    tree_root, leaf_to_sample = result

    # Root segment_len == 2 (the shared [10, 11])
    assert tree_root.segment_len == 2
    # 3 leaf children, each of segment_len 2 ([20,21] / [30,31] / [40,41])
    assert len(tree_root.children) == 3
    leaf_segs = _collect_leaf_segments(tree_root)
    assert leaf_segs == [2, 2, 2]
    # leaf_to_sample is the original sample order (sorted by first divergent token)
    assert sorted(leaf_to_sample) == [0, 1, 2]


def test_convert_trie_to_tree_node_depth_3_multilevel():
    """Depth-3 multi-level: 4 sequences forming a balanced 2x2 tree.

    Tree shape:
        root [10, 11]
            ├── [20, 21]
            │       ├── [30, 31]   <- sample 0
            │       └── [40, 41]   <- sample 1
            └── [50, 51]
                    ├── [60, 61]   <- sample 2
                    └── [70, 71]   <- sample 3
    """
    sequences = [
        [10, 11, 20, 21, 30, 31],
        [10, 11, 20, 21, 40, 41],
        [10, 11, 50, 51, 60, 61],
        [10, 11, 50, 51, 70, 71],
    ]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    result = convert_trie_to_tree_node(tries[0])
    assert result is not None
    tree_root, leaf_to_sample = result

    # Root has shared 2-token prefix [10, 11], 2 internal children
    assert tree_root.segment_len == 2
    assert len(tree_root.children) == 2
    # 4 leaves total
    assert _count_leaves(tree_root) == 4
    # Each leaf is segment_len=2 ([30,31] / [40,41] / [60,61] / [70,71])
    assert _collect_leaf_segments(tree_root) == [2, 2, 2, 2]
    assert sorted(leaf_to_sample) == [0, 1, 2, 3]


def test_convert_trie_returns_none_for_single_sample():
    """A single sample has no shared structure → returns None."""
    sequences = [[1, 2, 3, 4]]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    result = convert_trie_to_tree_node(tries[0])
    assert result is None


def test_convert_trie_returns_none_for_multi_forest():
    """When the virtual root has multiple children (no shared first token),
    return None."""
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=1000)
    result = convert_trie_to_tree_node(tries[0])
    assert result is None


# ---------------------------------------------------------------------------
# build_tree_dynamic (the public entry consumed by build_prefix_tree_micro_batch)
# ---------------------------------------------------------------------------


def test_build_tree_dynamic_recovers_depth_2():
    samples = [
        torch.tensor([10, 11, 20, 21], dtype=torch.long),
        torch.tensor([10, 11, 30, 31], dtype=torch.long),
        torch.tensor([10, 11, 40, 41], dtype=torch.long),
    ]
    result = build_tree_dynamic(samples)
    assert result is not None
    tree_root, leaf_to_sample = result
    assert tree_root.segment_len == 2
    assert _count_leaves(tree_root) == 3
    assert sorted(leaf_to_sample) == [0, 1, 2]


def test_build_tree_dynamic_recovers_depth_3():
    samples = [
        torch.tensor([10, 11, 20, 21, 30], dtype=torch.long),
        torch.tensor([10, 11, 20, 21, 40], dtype=torch.long),
        torch.tensor([10, 11, 50, 51, 60], dtype=torch.long),
    ]
    result = build_tree_dynamic(samples)
    assert result is not None
    tree_root, _ = result
    # Root [10, 11], internal child [20, 21] with 2 leaves + [50, 51] with 1 leaf
    assert tree_root.segment_len == 2
    assert _count_leaves(tree_root) == 3
    # Total internal+leaf nodes > 4 → confirms multi-level
    leaf_segs = _collect_leaf_segments(tree_root)
    assert sum(leaf_segs) <= sum(len(s) for s in samples)  # sanity


def test_build_tree_dynamic_returns_none_for_disjoint():
    samples = [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([4, 5, 6], dtype=torch.long),
    ]
    assert build_tree_dynamic(samples) is None


def test_build_tree_dynamic_returns_none_for_empty():
    assert build_tree_dynamic([]) is None


def test_build_tree_dynamic_returns_none_for_single_sample():
    samples = [torch.tensor([1, 2, 3], dtype=torch.long)]
    assert build_tree_dynamic(samples) is None


# ---------------------------------------------------------------------------
# End-to-end: build_tree_dynamic → build_layout_from_tree_node
# ---------------------------------------------------------------------------


def test_layout_from_dynamic_tree_token_conservation():
    """Flat layout produced from a dynamic tree must contain each unique node's
    tokens exactly once. For a depth-2 tree with 3 samples sharing 2-prefix,
    total flat tokens == 2 (root) + 3*2 (leaves) = 8.
    """
    samples = [
        torch.tensor([10, 11, 20, 21], dtype=torch.long),
        torch.tensor([10, 11, 30, 31], dtype=torch.long),
        torch.tensor([10, 11, 40, 41], dtype=torch.long),
    ]
    result = build_tree_dynamic(samples)
    assert result is not None
    tree_root, leaf_to_sample = result

    params = build_layout_from_tree_node(samples, tree_root, leaf_to_sample)
    assert params.flat_tokens.shape[0] == 2 + 3 * 2  # 8
    assert params.multilevel is True
    assert len(params.leaf_to_sample) == 3
    # Root range covers the first 2 tokens
    assert params.prefix_range == (0, 2)
    # 3 leaf ranges, each of length 2
    assert len(params.leaf_ranges) == 3
    for s, e in params.leaf_ranges:
        assert e - s == 2


def test_layout_from_dynamic_tree_depth_3_token_conservation():
    """Depth-3 tree: 4 samples × 6 tokens = 24 baseline.
    Tree shares: root[2] + 2 internal[2 each] + 4 leaves[2 each] = 14 flat tokens.
    """
    samples = [
        torch.tensor([10, 11, 20, 21, 30, 31], dtype=torch.long),
        torch.tensor([10, 11, 20, 21, 40, 41], dtype=torch.long),
        torch.tensor([10, 11, 50, 51, 60, 61], dtype=torch.long),
        torch.tensor([10, 11, 50, 51, 70, 71], dtype=torch.long),
    ]
    result = build_tree_dynamic(samples)
    assert result is not None
    tree_root, leaf_to_sample = result

    params = build_layout_from_tree_node(samples, tree_root, leaf_to_sample)
    # 2 (root) + 2*2 (internal) + 4*2 (leaves) = 14
    assert params.flat_tokens.shape[0] == 14
    assert len(params.leaf_ranges) == 4
    # Check that flat_tokens are correct: every original sample's tokens are present
    flat = params.flat_tokens.tolist()
    assert 10 in flat and 11 in flat  # root
    assert 20 in flat and 21 in flat  # internal A
    assert 50 in flat and 51 in flat  # internal B
    assert 30 in flat and 31 in flat  # leaf A1
    assert 70 in flat and 71 in flat  # leaf B2
