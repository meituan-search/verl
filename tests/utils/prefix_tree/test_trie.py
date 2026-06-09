# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Trie-based prefix-tree unit tests."""

from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import (
    TrieNode,
    build_tree_dynamic,
    convert_trie_to_tree_node,
    greedy_build_tries,
    mbs_groups_from_trie,
    prune_trie,
    trie_dfs_leaf_order,
    trie_to_leaf_ids,
)
from verl.utils.prefix_tree.utils import build_layout_from_tree_node

# ---------------------------------------------------------------------------
# trie_dfs_leaf_order
# ---------------------------------------------------------------------------


class TestTrieDfsLeafOrder:
    def test_basic(self):
        seqs = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 2, 6, 7]),
        ]
        result = build_tree_dynamic(seqs)
        assert result is not None
        tree_root, _ = result
        # Convert TreeNode → trie is not trivial; use the original trie
        # from greedy_build_tries for testing trie_dfs_leaf_order directly.
        raw_seqs = [s.tolist() for s in seqs]
        tries, _ = greedy_build_tries(raw_seqs, max_tokens_per_tree=sum(len(s) for s in raw_seqs) * 10)
        trie = tries[0]
        order = trie_dfs_leaf_order(trie)
        assert len(order) == 3
        assert set(order) == {0, 1, 2}

    def test_single_sample(self):
        raw = [[1, 2, 3]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=100)
        trie = tries[0]
        order = trie_dfs_leaf_order(trie)
        assert order == [0]

    def test_different_prefix_sets_stay_together(self):
        """Two sets with different prefixes: DFS groups each set together."""
        # Set A: share [1, 2]; Set B: share [5, 6]
        # shuffled input: A1, B1, A2, B2
        raw = [
            [1, 2, 10],  # idx 0: A1
            [5, 6, 20],  # idx 1: B1
            [1, 2, 11],  # idx 2: A2
            [5, 6, 21],  # idx 3: B2
        ]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        assert len(tries) == 1, "single trie for single forest"
        trie = tries[0]
        order = trie_dfs_leaf_order(trie)

        # DFS should visit children sorted by first token: [1,...] before [5,...]
        # So A-set (0, 2) should be consecutive, then B-set (1, 3) consecutive
        assert order in (
            [0, 2, 1, 3],
            [2, 0, 1, 3],
            [1, 3, 0, 2],
            [3, 1, 0, 2],
        ), f"DFS failed to group prefix sets: {order}"

        # Within each set, indices should be adjacent
        a_positions = sorted([order.index(0), order.index(2)])
        b_positions = sorted([order.index(1), order.index(3)])
        a_adjacent = a_positions[1] - a_positions[0] == 1
        b_adjacent = b_positions[1] - b_positions[0] == 1
        assert a_adjacent, f"A-set not adjacent: {order}"
        assert b_adjacent, f"B-set not adjacent: {order}"


# ---------------------------------------------------------------------------
# trie_to_leaf_ids
# ---------------------------------------------------------------------------


class TestTrieToLeafIds:
    def test_basic(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        trie = tries[0]
        leaf_ids = trie_to_leaf_ids(trie)
        assert len(leaf_ids) == len(raw)
        assert set(leaf_ids) == {0, 1, 2}  # 3 unique leaf IDs

    def test_no_sharing(self):
        """Different first tokens produce multiple root children; no single trie."""
        raw = [[1, 2], [3, 4], [5, 6]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        # May produce multiple tries
        assert len(tries) >= 1


# ---------------------------------------------------------------------------
# mbs_groups_from_trie
# ---------------------------------------------------------------------------


class TestMbsGroupsFromTrie:
    def test_basic(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        trie = tries[0]
        groups = mbs_groups_from_trie(trie, max_token_len=10)
        # All 3 should fit in one group since flat token count is ~9
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_split_by_budget(self):
        raw = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 3, 8, 9]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        trie = tries[0]
        # Flat tokens: [1,2,3] + [4,5] + [6,7] + [8,9] = 9
        # Budget of 8: fits first two ([1,2,3]+[4,5]+[6,7]=7), third must split
        groups = mbs_groups_from_trie(trie, max_token_len=8)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1


# ---------------------------------------------------------------------------
# prune_trie
# ---------------------------------------------------------------------------


class TestPruneTrie:
    def _build_trie(self, raw):
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=sum(len(s) for s in raw) * 10)
        return tries[0]

    def test_prune_all_leaves(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = prune_trie(trie, {0, 1, 2})
        assert result is not None
        tree_root, leaf_to_sample = result
        assert tree_root.segment_len == 2  # [1, 2]
        assert len(leaf_to_sample) == 3

    def test_prune_single_leaf(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = prune_trie(trie, {0})
        assert result is not None
        tree_root, leaf_to_sample = result
        # Only leaf 0: [1,2,3] + [4]
        assert tree_root.segment_len == 3  # [1, 2, 3]
        assert leaf_to_sample == [0]

    def test_prune_subset(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        # Keep 0 and 1 (share [1,2,3]), drop 2
        result = prune_trie(trie, {0, 1})
        assert result is not None
        tree_root, leaf_to_sample = result
        assert tree_root.segment_len == 3  # [1, 2, 3]
        assert set(leaf_to_sample) == {0, 1}

    def test_prune_empty(self):
        raw = [[1, 2, 3]]
        trie = self._build_trie(raw)
        result = prune_trie(trie, set())
        assert result is None

    def test_prune_no_match(self):
        raw = [[1, 2, 3]]
        trie = self._build_trie(raw)
        result = prune_trie(trie, {99})
        assert result is None

    def test_prune_to_layout_roundtrip(self):
        """prune_trie + build_layout_from_tree_node produces valid params."""
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = prune_trie(trie, {0, 2})
        assert result is not None
        tree_root, leaf_to_sample = result

        samples = [torch.tensor(s) for s in raw]
        params = build_layout_from_tree_node(samples, tree_root, leaf_to_sample)
        assert params.total_seqlen_q > 0
        assert len(params.leaf_ranges) == 2


# ---------------------------------------------------------------------------
# build_tree_dynamic
# ---------------------------------------------------------------------------


class TestBuildTreeDynamic:
    def test_basic(self):
        samples = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 2, 6, 7]),
        ]
        result = build_tree_dynamic(samples)
        assert result is not None
        tree_root, leaf_to_sample = result
        assert tree_root.segment_len == 2  # [1, 2]
        assert len(leaf_to_sample) == 3

    def test_no_shared_prefix(self):
        samples = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9]),
        ]
        result = build_tree_dynamic(samples)
        assert result is None

    def test_single_sample(self):
        samples = [torch.tensor([1, 2, 3])]
        result = build_tree_dynamic(samples)
        assert result is None

    def test_multi_level_tree(self):
        """Sequences with intermediate shared prefixes."""
        samples = [
            torch.tensor([10, 20, 30, 41]),
            torch.tensor([10, 20, 30, 42]),
            torch.tensor([10, 20, 50, 60]),
        ]
        result = build_tree_dynamic(samples)
        assert result is not None
        tree_root, leaf_to_sample = result
        # Root: [10, 20] (shared by all 3)
        # Child A: [30] with two leaves [41], [42]
        # Child B: [50, 60] (leaf)
        assert tree_root.segment_len == 2
        assert len(tree_root.children) == 2
        assert len(leaf_to_sample) == 3

    def test_duplicate_sequences(self):
        """GRPO with n>1: identical sequences should gracefully fallback."""
        samples = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
        ]
        result = build_tree_dynamic(samples)
        assert result is None

    def test_empty(self):
        result = build_tree_dynamic([])
        assert result is None


# ---------------------------------------------------------------------------
# convert_trie_to_tree_node
# ---------------------------------------------------------------------------


class TestConvertTrieToTreeNode:
    def test_normal(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        result = convert_trie_to_tree_node(tries[0])
        assert result is not None
        tree_root, leaf_to_sample = result
        assert tree_root.segment_len == 2
        assert len(leaf_to_sample) == 3

    def test_no_children(self):
        trie = TrieNode(tree_id=0)
        result = convert_trie_to_tree_node(trie)
        assert result is None

    def test_multi_children(self):
        """Trie root with >1 children → no single shared prefix → None."""
        trie = TrieNode(tree_id=0)
        trie.children[1] = TrieNode(tree_id=0, tokens=[1], sequence_ids=[0])
        trie.children[2] = TrieNode(tree_id=0, tokens=[2], sequence_ids=[1])
        result = convert_trie_to_tree_node(trie)
        assert result is None


# ---------------------------------------------------------------------------
# Mask (attention spec) output test
# ---------------------------------------------------------------------------


class TestAttentionSpec:
    def test_two_samples_shared_prefix(self):
        """2 samples, prefix_len=100, unshared_len=100 each.

        Flat layout: [0..100) prefix, [100..200) leaf 0, [200..300) leaf 1.
        Expected attention rectangles:

        q                k            mask     meaning
        (0,100)    →   (0,100)      causal   prefix self
        (100,200)  →   (0,100)      full     leaf0 → prefix
        (100,200)  →   (100,200)    causal   leaf0 self
        (200,300)  →   (0,100)      full     leaf1 → prefix
        (200,300)  →   (200,300)    causal   leaf1 self
        """
        prefix = list(range(100))
        sample0 = torch.tensor(prefix + [1000 + i for i in range(100)])
        sample1 = torch.tensor(prefix + [2000 + i for i in range(100)])

        from verl.utils.prefix_tree.dynamic import build_tree_dynamic
        from verl.utils.prefix_tree.utils import build_prefix_tree_attention_spec

        result = build_tree_dynamic([sample0, sample1])
        assert result is not None
        tree_root, leaf_to_sample = result

        q_ranges, k_ranges, mask_types = build_prefix_tree_attention_spec(tree_root)

        # 5 rectangles
        assert len(q_ranges) == 5
        assert len(k_ranges) == 5
        assert len(mask_types) == 5

        rects = list(zip(q_ranges, k_ranges, mask_types, strict=False))

        # prefix self: causal
        assert rects[0] == ((0, 100), (0, 100), "causal")

        # leaf0 → prefix: full, then leaf0 self: causal
        assert ((100, 200), (0, 100), "full") in rects
        assert ((100, 200), (100, 200), "causal") in rects

        # leaf1 → prefix: full, then leaf1 self: causal
        assert ((200, 300), (0, 100), "full") in rects
        assert ((200, 300), (200, 300), "causal") in rects
