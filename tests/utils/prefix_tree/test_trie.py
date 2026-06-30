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
    subtrie_view,
    trie_dfs_leaf_order,
    trie_to_leaf_ids,
)
from verl.utils.prefix_tree.tree import PrefixSubTrie

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
# subtrie_view
# ---------------------------------------------------------------------------


class TestPruneTrie:
    def _build_trie(self, raw):
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=sum(len(s) for s in raw) * 10)
        return tries[0]

    def test_prune_all_leaves(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = subtrie_view(trie, {0, 1, 2})
        assert result is not None
        assert isinstance(result, PrefixSubTrie)
        assert len(result.leaf_to_sample) == 3
        assert len(result.nodes[0].input_ids) == 2  # [1, 2] shared prefix

    def test_prune_single_leaf(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = subtrie_view(trie, {0})
        assert result is not None
        assert isinstance(result, PrefixSubTrie)
        assert result.leaf_to_sample == [0]
        # nodes: [1,2] ancestor + [3] node + [4] leaf
        assert len(result.nodes) == 3

    def test_prune_subset(self):
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        # Keep 0 and 1 (share [1,2,3]), drop 2
        result = subtrie_view(trie, {0, 1})
        assert result is not None
        assert isinstance(result, PrefixSubTrie)
        assert set(result.leaf_to_sample) == {0, 1}

    def test_prune_empty(self):
        raw = [[1, 2, 3]]
        trie = self._build_trie(raw)
        result = subtrie_view(trie, set())
        assert result is None

    def test_prune_no_match(self):
        raw = [[1, 2, 3]]
        trie = self._build_trie(raw)
        result = subtrie_view(trie, {99})
        assert result is None

    def test_prune_leaf_node_ids(self):
        """leaf_node_ids reference correct flat_idx values in the source trie."""
        raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        trie = self._build_trie(raw)
        result = subtrie_view(trie, {0, 2})
        assert result is not None
        assert len(result.leaf_to_sample) == 2
        assert set(result.leaf_to_sample) == {0, 2}
        # Each leaf_node_id should be a valid flat_idx in the trie
        for lid in result.leaf_node_ids:
            assert 0 <= lid < len(trie.nodes)


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
        assert len(result.nodes[0].input_ids) == 2  # [1, 2]
        assert len(result.leaf_to_sample) == 3

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
        # Root: [10, 20] (shared by all 3)
        # Child A: [30] with two leaves [41], [42]
        # Child B: [50, 60] (leaf)
        assert len(result.leaf_to_sample) == 3

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
        assert len(result.nodes[0].input_ids) == 2
        assert len(result.leaf_to_sample) == 3

    def test_no_children(self):
        trie = TrieNode(tree_id=0)
        result = convert_trie_to_tree_node(trie)
        assert result is None

    def test_multi_children(self):
        """Trie root with >1 children → no single shared prefix → None."""
        trie = TrieNode(tree_id=0)
        trie.children[1] = TrieNode(tree_id=0, input_ids=[1], sequence_ids=[0])
        trie.children[2] = TrieNode(tree_id=0, input_ids=[2], sequence_ids=[1])
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
        from verl.utils.prefix_tree.utils import build_layout_from_tree_node

        result = build_tree_dynamic([sample0, sample1])
        assert result is not None

        params = build_layout_from_tree_node([sample0, sample1], result)
        q_ranges, k_ranges, mask_types = params.q_ranges, params.k_ranges, params.mask_types

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

    def test_3layer_tree_7_leaves(self):
        """1 root, 2 intermediates, 4 leaves = 7 leaves total.

        Tree: root(3) → leaf0(2) direct
                      → child_a(2) → leaf1(2), leaf2(2)
                      → child_b(2) → leaf3(2), leaf4(2), leaf5(2), leaf6(2)

        Flat layout: [root(3)][leaf0(2)][child_a(2)][leaf1(2)][leaf2(2)]
                     [child_b(2)][leaf3(2)][leaf4(2)][leaf5(2)][leaf6(2)]
        = 21 tokens, 25 attention rects.
        """
        from verl.utils.prefix_tree.dynamic import build_tree_dynamic
        from verl.utils.prefix_tree.utils import build_layout_from_tree_node

        root_tok = [1, 2, 3]
        samples = [
            torch.tensor(root_tok + [10, 11]),  # idx 0: leaf0
            torch.tensor(root_tok + [20, 21, 41, 42]),  # idx 1: child_a → leaf1
            torch.tensor(root_tok + [20, 21, 51, 52]),  # idx 2: child_a → leaf2
            torch.tensor(root_tok + [30, 31, 61, 62]),  # idx 3: child_b → leaf3
            torch.tensor(root_tok + [30, 31, 71, 72]),  # idx 4: child_b → leaf4
            torch.tensor(root_tok + [30, 31, 81, 82]),  # idx 5: child_b → leaf5
            torch.tensor(root_tok + [30, 31, 91, 92]),  # idx 6: child_b → leaf6
        ]

        result = build_tree_dynamic(samples)
        assert result is not None
        assert len(result.leaf_to_sample) == 7

        params = build_layout_from_tree_node(samples, result)
        q_ranges, k_ranges, mask_types = params.q_ranges, params.k_ranges, params.mask_types
        assert len(q_ranges) == 25
        rects = list(zip(q_ranges, k_ranges, mask_types, strict=False))

        # root causal: (0,3)→(0,3)
        assert rects[0] == ((0, 3), (0, 3), "causal")

        # Every leaf sees root (FULL rects)
        full_to_root = [(q, k, t) for q, k, t in rects if k == (0, 3) and t == "full"]
        assert len(full_to_root) == 9  # leaf0 + child_a + leaf1~2 + child_b + leaf3~6

        # leaf0 directly under root
        assert ((3, 5), (3, 5), "causal") in rects  # leaf0 self

        # child_a subtree
        assert ((5, 7), (5, 7), "causal") in rects  # child_a self
        assert ((7, 9), (5, 7), "full") in rects  # leaf1 → child_a
        assert ((9, 11), (5, 7), "full") in rects  # leaf2 → child_a

        # child_b subtree
        assert ((11, 13), (11, 13), "causal") in rects  # child_b self
        assert ((13, 15), (11, 13), "full") in rects  # leaf3 → child_b
        assert ((15, 17), (11, 13), "full") in rects  # leaf4 → child_b

    def test_prune_3layer_to_4_leaves(self):
        """Prune the 7-leaf tree from test_3layer_tree_7_leaves to 4 leaves.

        Keep samples {0, 1, 2, 3}: leaf0, leaf1+leaf2 (under child_a), leaf3 (under child_b).
        child_b prunes from 4 leaves to 1 leaf.
        """
        from verl.utils.prefix_tree.dynamic import greedy_build_tries, subtrie_view

        raw = [
            [1, 2, 3, 10, 11],
            [1, 2, 3, 20, 21, 41, 42],
            [1, 2, 3, 20, 21, 51, 52],
            [1, 2, 3, 30, 31, 61, 62],
            [1, 2, 3, 30, 31, 71, 72],
            [1, 2, 3, 30, 31, 81, 82],
            [1, 2, 3, 30, 31, 91, 92],
        ]
        tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1000)
        trie = tries[0]

        result = subtrie_view(trie, {0, 1, 2, 3})
        assert result is not None
        assert isinstance(result, PrefixSubTrie)
        assert len(result.leaf_to_sample) == 4
        assert set(result.leaf_to_sample) == {0, 1, 2, 3}
        # All 4 leaves + their ancestor chain (root + child_a + child_b = 3 ancestors + 4 leaves = 7)
        assert len(result.nodes) == 7

    def test_zero_length_leaf_nodes(self):
        """A/AB/ABC: nested prefixes produce zero-length leaf nodes.

        A:   [1,2]           → leaf at root (0 tokens of its own)
        AB:  [1,2,3,4]       → leaf under child (0 tokens of its own)
        ABC: [1,2,3,4,5,6]   → leaf [5,6] at deepest level

        Zero-length leaves are skipped by mask generation — their tokens
        are covered by ancestor rects.
        """
        from verl.utils.prefix_tree.dynamic import build_tree_dynamic
        from verl.utils.prefix_tree.utils import build_layout_from_tree_node

        samples = [
            torch.tensor([1, 2]),  # A
            torch.tensor([1, 2, 3, 4]),  # AB
            torch.tensor([1, 2, 3, 4, 5, 6]),  # ABC
        ]

        result = build_tree_dynamic(samples)
        assert result is not None

        params = build_layout_from_tree_node(samples, result)
        q_ranges, k_ranges, mask_types = params.q_ranges, params.k_ranges, params.mask_types
        assert len(q_ranges) == 6  # root causal + 2 full→root + child causal + 1 full→child + ABC causal
        rects = list(zip(q_ranges, k_ranges, mask_types, strict=False))

        # root [1,2] = flat positions (0,2)
        assert ((0, 2), (0, 2), "causal") in rects  # root self

        # child [3,4] = flat positions (2,4)
        assert ((2, 4), (0, 2), "full") in rects  # child → root
        assert ((2, 4), (2, 4), "causal") in rects  # child self

        # ABC leaf [5,6] = flat positions (4,6)
        assert ((4, 6), (0, 2), "full") in rects  # ABC → root
        assert ((4, 6), (2, 4), "full") in rects  # ABC → child
        assert ((4, 6), (4, 6), "causal") in rects  # ABC self

        # No rects from the zero-length leaf nodes (they don't appear)
        assert len(rects) == 6

        # Verify flat token layout: [1,2] + [3,4] + [5,6] = 6 tokens
        assert torch.equal(params.tree_packed_tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
        assert params.total_seqlen_q == 6
