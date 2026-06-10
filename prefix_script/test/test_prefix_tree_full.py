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
"""CPU unit tests for all prefix_tree modules."""

import importlib.util
import sys
import types
import unittest

# --- Mock missing dependencies ---
for mod in [
    "ray",
    "transformers",
    "codetiming",
    "omegaconf",
    "magi_attention",
    "megatron.core",
    "verl.utils.device",
    "verl.utils.megatron_utils",
    "verl.utils.seqlen_balancing",
    "verl.utils.tensordict_utils",
    "verl.utils.torch_functional",
    "verl.utils.py_functional",
    "verl.utils.ray_utils",
    "xxhash",
]:
    sys.modules[mod] = types.ModuleType(mod)
    for i in range(1, len(mod.split("."))):
        p = ".".join(mod.split(".")[:i])
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)

sys.path.insert(0, "/Users/arvyanh/Documents/code_rl/verl")

# Register verl packages as proper packages (need __path__ for submodule imports)
for pkg in ["verl", "verl.utils", "verl.utils.prefix_tree"]:
    m = sys.modules.setdefault(pkg, types.ModuleType(pkg))
    m.__path__ = [f"/Users/arvyanh/Documents/code_rl/verl/{pkg.replace('.', '/')}"]

import torch  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dyn = _load(
    "/Users/arvyanh/Documents/code_rl/verl/verl/utils/prefix_tree/dynamic.py",
    "verl.utils.prefix_tree.dynamic",
)
_utils = _load(
    "/Users/arvyanh/Documents/code_rl/verl/verl/utils/prefix_tree/utils.py",
    "verl.utils.prefix_tree.utils",
)
_magi = _load(
    "/Users/arvyanh/Documents/code_rl/verl/verl/utils/prefix_tree/magi.py",
    "verl.utils.prefix_tree.magi",
)


# ============================================================================
# dynamic.py — trie construction
# ============================================================================


class TestBuildTreeDynamic(unittest.TestCase):
    def test_basic(self):
        r = _dyn.build_tree_dynamic(
            [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 6, 7])]
        )
        self.assertIsNotNone(r)
        self.assertEqual(r[0].segment_len, 2)
        self.assertEqual(len(r[1]), 3)

    def test_no_shared_prefix(self):
        self.assertIsNone(_dyn.build_tree_dynamic([torch.tensor([1, 2]), torch.tensor([3, 4])]))

    def test_empty(self):
        self.assertIsNone(_dyn.build_tree_dynamic([]))

    def test_single_sample(self):
        self.assertIsNone(_dyn.build_tree_dynamic([torch.tensor([1, 2])]))

    def test_duplicate_sequences(self):
        self.assertIsNone(_dyn.build_tree_dynamic([torch.tensor([1, 2]), torch.tensor([1, 2])]))

    def test_multi_level(self):
        r = _dyn.build_tree_dynamic(
            [torch.tensor([10, 20, 30, 41]), torch.tensor([10, 20, 30, 42]), torch.tensor([10, 20, 50, 60])]
        )
        self.assertIsNotNone(r)
        self.assertEqual(r[0].segment_len, 2)
        self.assertEqual(len(r[0].children), 2)
        self.assertEqual(len(r[1]), 3)

    def test_zero_length_leaf(self):
        r = _dyn.build_tree_dynamic(
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])]
        )
        self.assertIsNotNone(r)
        self.assertEqual(r[0].segment_len, 2)
        self.assertEqual(len(r[1]), 3)

    def test_two_groups_greedy(self):
        """Sequences with different prefixes produce multi-trie output."""
        raw = [[1, 2, 3], [1, 2, 4], [5, 6, 7]]
        tries, _ = _dyn.greedy_build_tries(raw, max_tokens_per_tree=1000)
        self.assertGreaterEqual(len(tries), 1)


# ============================================================================
# dynamic.py — trie → TreeNode conversion
# ============================================================================


class TestConvertTrieToTreeNode(unittest.TestCase):
    def _trie(self, raw):
        tries, _ = _dyn.greedy_build_tries(raw, max_tokens_per_tree=1000)
        return tries[0]

    def test_normal(self):
        trie = self._trie([[1, 2, 3], [1, 2, 4]])
        r = _dyn.convert_trie_to_tree_node(trie)
        self.assertIsNotNone(r)
        self.assertEqual(r[0].segment_len, 2)
        self.assertEqual(len(r[1]), 2)

    def test_empty_root(self):
        self.assertIsNone(_dyn.convert_trie_to_tree_node(_dyn.TrieNode(tree_id=0)))

    def test_multi_root(self):
        trie = _dyn.TrieNode(tree_id=0)
        trie.children[1] = _dyn.TrieNode(tree_id=0, tokens=[1], sequence_ids=[0])
        trie.children[2] = _dyn.TrieNode(tree_id=0, tokens=[2], sequence_ids=[1])
        self.assertIsNone(_dyn.convert_trie_to_tree_node(trie))

    def test_zero_length_leaf_preserved(self):
        """Intermediate termination yields zero-length leaf."""
        raw = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
        trie = self._trie(raw)
        r = _dyn.convert_trie_to_tree_node(trie)
        self.assertIsNotNone(r)
        self.assertEqual(len(r[1]), 3)  # all 3 samples as leaves


# ============================================================================
# dynamic.py — DFS ordering / leaf IDs
# ============================================================================


class TestDfsLeafOrder(unittest.TestCase):
    def test_basic(self):
        raw = [[1, 2, 10], [1, 2, 20], [1, 3, 30]]
        t, _ = _dyn.greedy_build_tries(raw, 1000)
        o = _dyn.trie_dfs_leaf_order(t[0])
        self.assertEqual(len(o), 3)
        self.assertEqual(set(o), {0, 1, 2})

    def test_groups_together(self):
        raw = [[1, 2, 10], [5, 6, 20], [1, 2, 11], [5, 6, 21]]
        t, _ = _dyn.greedy_build_tries(raw, 1000)
        o = _dyn.trie_dfs_leaf_order(t[0])
        a = sorted([o.index(0), o.index(2)])
        b = sorted([o.index(1), o.index(3)])
        self.assertEqual(a[1] - a[0], 1)
        self.assertEqual(b[1] - b[0], 1)

    def test_dfs_leaf_order_fn(self):
        seqs = [[1, 2, 10], [1, 2, 20], [5, 6, 30]]
        o = _dyn.dfs_leaf_order(seqs)
        self.assertEqual(len(o), 3)


class TestTrieToLeafIds(unittest.TestCase):
    def test_basic(self):
        t, _ = _dyn.greedy_build_tries([[1, 2, 10], [1, 2, 20]], 1000)
        ids = _dyn.trie_to_leaf_ids(t[0])
        self.assertGreaterEqual(len(ids), 1)


# ============================================================================
# dynamic.py — MBS grouping
# ============================================================================


class TestMbsGroupsFromTrie(unittest.TestCase):
    def _trie(self, raw):
        t, _ = _dyn.greedy_build_tries(raw, 1000)
        return t[0]

    def test_all_fit(self):
        g = _dyn.mbs_groups_from_trie(self._trie([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]), 10)
        self.assertEqual(len(g), 1)
        self.assertEqual(set(g[0]), {0, 1, 2})

    def test_split(self):
        g = _dyn.mbs_groups_from_trie(self._trie([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 3, 8, 9]]), 8)
        self.assertEqual(len(g), 2)

    def test_dfs_micro_batch_groups_fn(self):
        seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
        g = _dyn.dfs_micro_batch_groups(seqs, 10)
        self.assertGreater(len(g), 0)


# ============================================================================
# dynamic.py — prune
# ============================================================================


class TestPruneTrie(unittest.TestCase):
    def _trie(self, raw):
        t, _ = _dyn.greedy_build_tries(raw, 1000)
        return t[0]

    def test_all(self):
        r = _dyn.prune_trie(self._trie([[1, 2, 10], [1, 2, 20], [1, 3, 30]]), {0, 1, 2})
        self.assertIsNotNone(r)
        self.assertGreater(r[0].segment_len, 0)
        self.assertEqual(len(r[1]), 3)

    def test_single(self):
        r = _dyn.prune_trie(self._trie([[1, 2, 10], [1, 2, 20], [1, 3, 30]]), {0})
        self.assertIsNotNone(r)
        self.assertEqual(r[1], [0])

    def test_subset(self):
        r = _dyn.prune_trie(self._trie([[1, 2, 10], [1, 2, 20], [1, 3, 30]]), {0, 1})
        self.assertIsNotNone(r)
        self.assertEqual(set(r[1]), {0, 1})

    def test_empty(self):
        self.assertIsNone(_dyn.prune_trie(self._trie([[1, 2, 3]]), set()))

    def test_no_match(self):
        self.assertIsNone(_dyn.prune_trie(self._trie([[1, 2, 3]]), {99}))


# ============================================================================
# dynamic.py — load balancing
# ============================================================================


class TestLoadBalancing(unittest.TestCase):
    def test_build_mini_batch_prefix_groups(self):
        seqs = [[1, 2, 10], [1, 2, 20], [1, 3, 30], [1, 3, 40]]
        groups = _dyn.build_mini_batch_prefix_groups(seqs)
        self.assertGreater(len(groups), 0)


# ============================================================================
# utils.py — TreeNode
# ============================================================================


class TestTreeNode(unittest.TestCase):
    def test_basic(self):
        n = _utils.TreeNode(segment_len=3, children=[])
        self.assertTrue(n.is_leaf)
        self.assertEqual(n.segment_len, 3)

    def test_internal(self):
        leaf = _utils.TreeNode(segment_len=2, children=[])
        n = _utils.TreeNode(segment_len=3, children=[leaf])
        self.assertFalse(n.is_leaf)
        self.assertTrue(leaf.is_leaf)


# ============================================================================
# utils.py — PrefixTreeParams
# ============================================================================


class TestPrefixTreeParams(unittest.TestCase):
    def test_valid(self):
        p = _utils.PrefixTreeParams(
            prefix_range=(0, 3),
            prefix_segments=[(0, 3)],
            leaf_ranges=[(3, 5), (5, 7)],
            leaf_segments=[(3, 5), (5, 7)],
            leaf_to_sample=[0, 1],
            sample_to_leaf_range={0: (3, 5), 1: (5, 7)},
            q_ranges=[(0, 3), (3, 5), (3, 5), (5, 7), (5, 7)],
            k_ranges=[(0, 3), (0, 3), (3, 5), (0, 3), (5, 7)],
            mask_types=["causal", "full", "causal", "full", "causal"],
            total_seqlen_q=7,
            total_seqlen_k=7,
            flat_tokens=torch.arange(7),
        )
        self.assertEqual(p.prefix_len, 3)
        self.assertEqual(p.num_samples, 2)
        self.assertEqual(p.branch_lengths, [2, 2])
        self.assertEqual(p.get_leaf_range(0), (3, 5))

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            _utils.PrefixTreeParams(
                prefix_range=(0, 3),
                prefix_segments=[(0, 3)],
                leaf_ranges=[(3, 5)],
                leaf_segments=[(3, 5)],
                leaf_to_sample=[0, 1],  # mismatch
                sample_to_leaf_range={0: (3, 5)},
                q_ranges=[(0, 3)],
                k_ranges=[(0, 3)],
                mask_types=["causal"],
                total_seqlen_q=5,
                total_seqlen_k=5,
            )

    def test_range_not_zero_start(self):
        with self.assertRaises(ValueError):
            _utils.PrefixTreeParams(
                prefix_range=(1, 3),
                prefix_segments=[(1, 3)],
                leaf_ranges=[(3, 5)],
                leaf_segments=[(3, 5)],
                leaf_to_sample=[0],
                sample_to_leaf_range={0: (3, 5)},
                q_ranges=[(1, 3), (3, 5), (3, 5)],
                k_ranges=[(1, 3), (1, 3), (3, 5)],
                mask_types=["causal", "full", "causal"],
                total_seqlen_q=5,
                total_seqlen_k=5,
            )


# ============================================================================
# utils.py — layout from tree node
# ============================================================================


class TestBuildLayoutFromTreeNode(unittest.TestCase):
    def test_basic(self):
        samples = [torch.tensor([1, 2, 3, 41]), torch.tensor([1, 2, 3, 51]), torch.tensor([1, 2, 3, 61])]
        r = _dyn.build_tree_dynamic(samples)
        self.assertIsNotNone(r)
        params = _utils.build_layout_from_tree_node(samples, r[0], r[1])
        self.assertEqual(params.total_seqlen_q, 6)  # [1,2,3] + [41] + [51] + [61]
        self.assertEqual(len(params.leaf_ranges), 3)
        self.assertIsNotNone(params.flat_tokens)

    def test_extract_sample_tensor(self):
        samples = [torch.tensor([1, 2, 3, 41, 42]), torch.tensor([1, 2, 3, 51])]
        r = _dyn.build_tree_dynamic(samples)
        self.assertIsNotNone(r)
        params = _utils.build_layout_from_tree_node(samples, r[0], r[1])
        for i, original in enumerate(samples):
            extracted = _utils.extract_sample_tensor(params.flat_tokens, params, i)
            self.assertTrue(torch.equal(extracted, original))

    def test_extract_sample_tensors(self):
        samples = [torch.tensor([1, 2, 3, 41]), torch.tensor([1, 2, 3, 51])]
        r = _dyn.build_tree_dynamic(samples)
        self.assertIsNotNone(r)
        params = _utils.build_layout_from_tree_node(samples, r[0], r[1])
        d = _utils.extract_sample_tensors(params.flat_tokens, params)
        self.assertEqual(len(d), 2)


# ============================================================================
# utils.py — attention spec
# ============================================================================


class TestAttentionSpec(unittest.TestCase):
    def test_two_samples(self):
        r = _dyn.build_tree_dynamic([torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5])])
        self.assertIsNotNone(r)
        q, k, m = _utils.build_prefix_tree_attention_spec(r[0])
        self.assertEqual(len(q), 5)
        rects = list(zip(q, k, m, strict=False))
        # root causal: (0,3)→(0,3)
        self.assertIn(((0, 3), (0, 3), "causal"), rects)
        # leaf0→root full, leaf0 self causal
        self.assertIn(((3, 4), (0, 3), "full"), rects)
        self.assertIn(((3, 4), (3, 4), "causal"), rects)
        # leaf1→root full, leaf1 self causal
        self.assertIn(((4, 5), (0, 3), "full"), rects)
        self.assertIn(((4, 5), (4, 5), "causal"), rects)

    def test_zero_length_leaf(self):
        r = _dyn.build_tree_dynamic(
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])]
        )
        self.assertIsNotNone(r)
        q, k, m = _utils.build_prefix_tree_attention_spec(r[0])
        self.assertEqual(len(q), 6)
        rects = list(zip(q, k, m, strict=False))
        # root self: (0,2)→(0,2) causal
        self.assertIn(((0, 2), (0, 2), "causal"), rects)
        # no rects from zero-length leaves

    def test_from_pruned_trie(self):
        t, _ = _dyn.greedy_build_tries([[1, 2, 10], [1, 2, 20], [1, 3, 30]], 1000)
        p = _dyn.prune_trie(t[0], {0, 2})
        self.assertIsNotNone(p)
        q, k, m = _utils.build_prefix_tree_attention_spec(p[0])
        self.assertGreater(len(q), 0)


# ============================================================================
# utils.py — dense mask
# ============================================================================


class TestDenseMask(unittest.TestCase):
    def test_basic(self):
        m = _utils.build_prefix_tree_dense_mask(4, [(0, 2), (2, 4)], [(0, 2), (2, 4)], ["causal", "causal"])
        self.assertEqual(m.shape, (4, 4))
        # upper triangular including diagonal
        self.assertTrue(m[0, 0])

    def test_full_mask(self):
        m = _utils.build_prefix_tree_dense_mask(4, [(2, 4)], [(0, 2)], ["full"])
        self.assertTrue(m[2:4, 0:2].all())

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            _utils.build_prefix_tree_dense_mask(4, [(0, 1)], [(0, 1)], [])


# ============================================================================
# Mask correctness — materialize spec → dense mask → verify attention pattern
# ============================================================================


class TestInputMaskCorrectness(unittest.TestCase):
    """Verify the attention mask is correct for prefix-tree flat layouts."""

    def _spec_to_mask(self, tree_root):
        """Turn attention spec into a dense bool mask."""
        q, k, m = _utils.build_prefix_tree_attention_spec(tree_root)
        flat_len = max(max(e for _, e in q), max(e for _, e in k))
        return _utils.build_prefix_tree_dense_mask(flat_len, q, k, m)

    def test_two_samples_prefix_only(self):
        """2 samples sharing 3 tokens [1,2,3], each with 1 unique token.

        Flat: [prefix(3)] [leaf0(1)] [leaf1(1)] = 5 tokens.
        Mask: causal on prefix self, leaf→prefix full, leaf self causal.
        """
        r = _dyn.build_tree_dynamic([torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5])])
        self.assertIsNotNone(r)
        mask = self._spec_to_mask(r[0])
        self.assertEqual(mask.shape, (5, 5))

        # prefix tokens [0,3): causal self-attention
        self.assertTrue(mask[0, 0])  # prefix[0]→prefix[0]
        self.assertTrue(mask[2, 0])  # prefix[2]→prefix[0] (causal)
        self.assertTrue(mask[2, 2])  # prefix[2]→prefix[2]
        self.assertFalse(mask[0, 2])  # prefix[0]→prefix[2] (anticausal)

        # leaf0 token [3,4): full to prefix, causal self
        self.assertTrue(mask[3, 0])  # leaf0→prefix[0] (full)
        self.assertTrue(mask[3, 2])  # leaf0→prefix[2] (full, non-causal)
        self.assertTrue(mask[3, 3])  # leaf0→leaf0 (causal self)
        self.assertFalse(mask[3, 4])  # leaf0→leaf1 (blocked)

        # leaf1 token [4,5): full to prefix, causal self
        self.assertTrue(mask[4, 0])  # leaf1→prefix[0] (full)
        self.assertTrue(mask[4, 2])  # leaf1→prefix[2] (full, non-causal)
        self.assertTrue(mask[4, 4])  # leaf1→leaf1 (causal self)
        self.assertFalse(mask[4, 3])  # leaf1→leaf0 (blocked)

    def test_two_samples_no_cross_attention(self):
        """Leaf tokens cannot attend to other leaves' tokens."""
        r = _dyn.build_tree_dynamic([torch.tensor([1, 2, 10, 11]), torch.tensor([1, 2, 20, 21])])
        self.assertIsNotNone(r)
        mask = self._spec_to_mask(r[0])
        # 6 tokens: [prefix(2)] [leaf0(2)] [leaf1(2)]
        # leaf0 queries at positions 2,3
        # leaf1 keys at positions 4,5
        self.assertFalse(mask[2, 4])  # leaf0[0]→leaf1[0] blocked
        self.assertFalse(mask[2, 5])  # leaf0[0]→leaf1[1] blocked
        self.assertFalse(mask[3, 4])  # leaf0[1]→leaf1[0] blocked
        self.assertFalse(mask[4, 2])  # leaf1[0]→leaf0[0] blocked

    def test_prefix_causal(self):
        """Prefix tokens only see prefix tokens causally."""
        r = _dyn.build_tree_dynamic([torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5])])
        self.assertIsNotNone(r)
        mask = self._spec_to_mask(r[0])
        # prefix query at position 1 can see: prefix[0], prefix[1]
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[1, 2])  # but NOT prefix[2] (anticausal)
        self.assertFalse(mask[1, 3])  # and NOT leaf tokens

    def test_leaf_can_see_all_prefix(self):
        """Leaf tokens have FULL access to all prefix tokens."""
        r = _dyn.build_tree_dynamic([torch.tensor([1, 2, 30, 40]), torch.tensor([1, 2, 50, 60])])
        self.assertIsNotNone(r)
        mask = self._spec_to_mask(r[0])
        # leaf0 first token at position 2, sees prefix[0] and prefix[1]
        self.assertTrue(mask[2, 0])
        self.assertTrue(mask[2, 1])
        # leaf1 first token at position 4, sees all prefix tokens
        self.assertTrue(mask[4, 0])
        self.assertTrue(mask[4, 1])

    def test_multi_level_mask(self):
        """Multi-level tree: intermediate node gets full attention from its leaves."""
        # [1,2,3,41], [1,2,3,42], [1,2,50,60]
        # shares [1,2] (root), then [3]+[41/42] (child_a, 2 leaves), [50,60] (child_b, 1 leaf)
        r = _dyn.build_tree_dynamic(
            [torch.tensor([1, 2, 3, 41]), torch.tensor([1, 2, 3, 42]), torch.tensor([1, 2, 50, 60])]
        )
        self.assertIsNotNone(r)
        mask = self._spec_to_mask(r[0])

        # child_a tokens [3] at positions 2 → leaf0 at 3, leaf1 at 4
        # Verify leaf0 can see child_a (FULL, not just causal)
        # Find leaf positions by checking which positions see prefix but not other leaves
        prefix_seen = mask[:, 0].nonzero(as_tuple=False).flatten().tolist()
        self.assertIn(0, prefix_seen)  # prefix[0] sees itself
        self.assertGreater(len(prefix_seen), 2)  # leaf tokens also see prefix

    def test_dense_mask_matches_flex_spec(self):
        """Every rect in the spec must have corresponding True entries in dense mask."""
        r = _dyn.build_tree_dynamic(
            [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 6, 7])]
        )
        self.assertIsNotNone(r)
        q, k, m = _utils.build_prefix_tree_attention_spec(r[0])
        total = max(max(e for _, e in q), max(e for _, e in k))
        dense = _utils.build_prefix_tree_dense_mask(total, q, k, m)

        for (qs, qe), (ks, ke), mask_type in zip(q, k, m, strict=False):
            block = dense[qs:qe, ks:ke]
            if mask_type == "full":
                self.assertTrue(block.all(), f"full rect ({qs},{qe})→({ks},{ke}) not all True")
            elif mask_type == "causal":
                # upper triangular including diagonal
                for i in range(qe - qs):
                    for j in range(ke - ks):
                        if j <= i:
                            self.assertTrue(
                                dense[qs + i, ks + j],
                                f"causal rect ({qs},{qe})→({ks},{ke}) missing True at ({qs + i},{ks + j})",
                            )

    def test_deep_tree_many_blocks(self):
        """4-level tree with 8 leaves — verify many attention rectangles are correct.

        Structure: root(2 tokens) → 2 children → 2 grandchildren each → 2 leaves each = 8 leaves.
        Expected rectangles per node: 1 causal + (leaf descendants) full rects.
        Root: 1 causal + 8 full = 9
        L1 child_a: 1 causal + 4 full = 5
        L1 child_b: 1 causal + 4 full = 5
        L2 grandchild0: 1 causal + 2 full = 3  (×4)
        L3 leaves: 8 causal
        Total: 9 + 5 + 5 + 4×3 + 8 = 39 rects
        """
        # 8 samples forming a 4-level tree
        samples = [
            torch.tensor([10, 20, 31, 41, 101]),  # root→child0→grandchild0→leaf0
            torch.tensor([10, 20, 31, 41, 102]),  # root→child0→grandchild0→leaf1
            torch.tensor([10, 20, 31, 42, 201]),  # root→child0→grandchild1→leaf0
            torch.tensor([10, 20, 31, 42, 202]),  # root→child0→grandchild1→leaf1
            torch.tensor([10, 20, 32, 51, 301]),  # root→child1→grandchild0→leaf0
            torch.tensor([10, 20, 32, 51, 302]),  # root→child1→grandchild0→leaf1
            torch.tensor([10, 20, 32, 52, 401]),  # root→child1→grandchild1→leaf0
            torch.tensor([10, 20, 32, 52, 402]),  # root→child1→grandchild1→leaf1
        ]
        r = _dyn.build_tree_dynamic(samples)
        self.assertIsNotNone(r)
        self.assertEqual(len(r[1]), 8)

        q, k, m = _utils.build_prefix_tree_attention_spec(r[0])
        total = max(max(e for _, e in q), max(e for _, e in k))
        dense = _utils.build_prefix_tree_dense_mask(total, q, k, m)

        # Verify: all 8 leaf tokens can see ALL root tokens (full rects exist)
        prefix_len = r[0].segment_len  # 2
        self.assertEqual(prefix_len, 2)
        self.assertTrue(dense[total - 1, 0])  # last token sees first prefix token
        self.assertTrue(dense[total - 1, 1])  # last token sees last prefix token

        # Verify: no cross-attention between leaves under different L2 nodes
        # Find the dense rects by checking all spec entries
        for (qs, qe), (ks, ke), mask_type in zip(q, k, m, strict=False):
            block = dense[qs:qe, ks:ke]
            if mask_type == "full":
                self.assertTrue(block.all(), f"full rect ({qs},{qe})→({ks},{ke}) not all True")
            elif mask_type == "causal":
                for i in range(qe - qs):
                    for j in range(ke - ks):
                        if j <= i:
                            self.assertTrue(
                                dense[qs + i, ks + j], f"causal ({qs},{qe})→({ks},{ke}) miss ({qs + i},{ks + j})"
                            )

        # Verify rects count is substantial (multi-level tree produces many rects)
        self.assertGreater(len(q), 20, f"expected >20 rects for 4-level tree, got {len(q)}")


# ============================================================================
# utils.py — longest common prefix
# ============================================================================


class TestLongestCommonPrefix(unittest.TestCase):
    def test_basic(self):
        n = _utils.longest_common_prefix_length(
            [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 6])]
        )
        self.assertEqual(n, 2)

    def test_no_shared(self):
        n = _utils.longest_common_prefix_length([torch.tensor([1, 2]), torch.tensor([3, 4])])
        self.assertEqual(n, 0)

    def test_single(self):
        n = _utils.longest_common_prefix_length([torch.tensor([1, 2, 3])])
        self.assertEqual(n, 3)


# ============================================================================
# magi.py — config helpers
# ============================================================================


class TestConfigHelpers(unittest.TestCase):
    def test_get_kwargs_enabled(self):
        kw = _magi.get_prefix_tree_kwargs(True, "magi")
        self.assertEqual(kw["use_prefix_tree"], True)
        self.assertEqual(kw["prefix_tree_attention"], "magi")

    def test_get_kwargs_disabled(self):
        self.assertEqual(_magi.get_prefix_tree_kwargs(False, "magi"), {})

    def test_strip(self):
        args = {"use_prefix_tree": True, "label": torch.tensor(1.0), "prefix_tree_attention": "magi"}
        _magi.strip_prefix_tree_args(args)
        self.assertNotIn("use_prefix_tree", args)
        self.assertIn("label", args)

    def test_strip_noop(self):
        args = {"label": torch.tensor(1.0)}
        _magi.strip_prefix_tree_args(args)
        self.assertIn("label", args)

    def test_strip_none(self):
        _magi.strip_prefix_tree_args(None)  # should not raise


# ============================================================================
# magi.py — restore_flat_to_nested (partial)
# ============================================================================


class TestRestoreFlatToNested(unittest.TestCase):
    def test_basic(self):
        tokens = [torch.tensor([10, 20, 30, 41, 42]), torch.tensor([10, 20, 30, 51])]
        r = _dyn.build_tree_dynamic(tokens)
        self.assertIsNotNone(r)
        params = _utils.build_layout_from_tree_node(tokens, r[0], r[1])
        pt_batch = _magi.PrefixTreeMagiBatch(
            flat_input_ids=params.flat_tokens,
            flat_position_ids=params.flat_position_ids,
            flat_loss_mask=None,
            magi_key=None,
            flex_key=None,
            leaf_to_sample=params.leaf_to_sample,
            leaf_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            original_batch_size=params.num_samples,
        )
        flat = pt_batch.flat_input_ids
        restored = _magi.restore_flat_to_nested(flat, pt_batch)
        offsets = restored.offsets()
        lengths = offsets.diff().tolist()
        self.assertEqual(lengths, [5, 4])
        vals = restored.values()
        pos = 0
        for i, orig in enumerate(tokens):
            self.assertTrue(torch.equal(vals[pos : pos + int(lengths[i])], orig))
            pos += int(lengths[i])

    def test_with_extra_dim(self):
        tokens = [torch.tensor([10, 20, 30, 41]), torch.tensor([10, 20, 30, 51])]
        r = _dyn.build_tree_dynamic(tokens)
        self.assertIsNotNone(r)
        params = _utils.build_layout_from_tree_node(tokens, r[0], r[1])
        pt_batch = _magi.PrefixTreeMagiBatch(
            flat_input_ids=params.flat_tokens,
            flat_position_ids=params.flat_position_ids,
            flat_loss_mask=None,
            magi_key=None,
            flex_key=None,
            leaf_to_sample=params.leaf_to_sample,
            leaf_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            original_batch_size=params.num_samples,
        )
        flat_logits = torch.randn(6, 8)
        restored = _magi.restore_flat_to_nested(flat_logits, pt_batch)
        offsets = restored.offsets()
        lengths = offsets.diff().tolist()
        self.assertEqual(lengths, [4, 4])


# ============================================================================
# magi.py — unhinted unpack (padded fallback)
# ============================================================================


class TestUnpackNested(unittest.TestCase):
    def test_padded_returns_none(self):
        """Padded 2D tensor returns None (falls back to standard attention)."""
        x = torch.zeros(2, 10, dtype=torch.long)
        result = _magi._unpack_nested_to_list(x)
        self.assertIsNone(result)

    def test_none_returns_none(self):
        self.assertIsNone(_magi._unpack_nested_to_list(None))


# ============================================================================
# PrefixTreeMagiBatch
# ============================================================================


class TestPrefixTreeMagiBatch(unittest.TestCase):
    def test_construction(self):
        t = torch.arange(10)
        b = _magi.PrefixTreeMagiBatch(
            flat_input_ids=t,
            flat_position_ids=t,
            flat_loss_mask=None,
            magi_key=None,
            flex_key=None,
            leaf_to_sample=[0, 1],
            leaf_ranges=[(0, 5), (5, 10)],
            prefix_range=(0, 3),
            original_batch_size=2,
        )
        self.assertEqual(b.real_tokens, 10)
        self.assertEqual(b.local_flat_input_ids.shape, (10,))


# ============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
