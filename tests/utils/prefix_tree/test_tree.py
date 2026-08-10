"""Tests for build_global_tree_from_segments (segment-hash-driven global trie
builder) plus segment_grouper primitives (create_segment_metadata,
create_grpo_segment_metadata, group_by_segment_hash). Merges the former
test_tree.py + test_segment_grouper.py. The old build_tree_from_segments
(injectable internal-builder variant) was removed in commit a4a15b849.
"""
from __future__ import annotations

import pytest
import torch

from verl.utils.prefix_tree.segment_grouper import (
    create_grpo_segment_metadata,
    create_segment_metadata,
    group_by_segment_hash,
)
from verl.utils.prefix_tree.tree import PrefixTrie, TrieNode, build_global_tree_from_segments


def test_create_segment_metadata_dtypes_and_int_hashes():
    hashes, lengths = create_segment_metadata([[("a", 3), ("b", 2)]])
    assert hashes.dtype == object and lengths.dtype == object
    assert hashes[0].tolist() == [hash("a") & 0xFFFFFFFF, hash("b") & 0xFFFFFFFF]
    assert lengths[0].tolist() == [3, 2]
    h2, l2 = create_segment_metadata([[(7, 3)]])  # accepts int hashes too
    assert h2[0].tolist() == [7] and l2[0].tolist() == [3]


def test_create_grpo_segment_metadata_validates_and_groups():
    with pytest.raises(ValueError):
        create_grpo_segment_metadata(["p0", "p0", "p1"], [3, 3, 3], rollout_n=2)
    hashes, lengths = create_grpo_segment_metadata(["p0", "p0", "p1"], [3, 3, 3], rollout_n=3)
    assert hashes[0] == hashes[1] and hashes[0] != hashes[2]
    assert lengths[0].tolist() == [3]


def test_group_by_segment_hash_level0_and_out_of_range():
    hashes, lengths = create_segment_metadata([[("p0", 3)], [("p0", 3)], [("p1", 5)]])
    groups = group_by_segment_hash(hashes, lengths, level=0)
    assert sorted(len(v) for v in groups.values()) == [1, 2]
    big = next(v for v in groups.values() if len(v) == 2)
    assert sorted(idx for idx, _ in big) == [0, 1]
    h1, l1 = create_segment_metadata([[("p0", 3)]])  # out-of-range level -> empty
    assert group_by_segment_hash(h1, l1, level=5) == {}


def test_shared_hash_creates_trie_with_prefix_tokens():
    shared = [10, 20, 30]
    samples = [torch.tensor(shared + [1, 2]), torch.tensor(shared + [3, 4])]
    hashes, lengths = create_segment_metadata([[("p", 3), ("a", 2)], [("p", 3), ("b", 2)]])
    trie = build_global_tree_from_segments(samples, hashes, lengths)
    assert isinstance(trie, PrefixTrie) and trie.is_root
    tokens = [t for n in trie.nodes for t in n.input_ids]
    assert all(tok in tokens for tok in shared)


def test_no_sharing_single_and_empty_return_none():
    samples = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
    hashes, lengths = create_segment_metadata([[("a", 3)], [("b", 3)], [("c", 3)]])
    # All-different first hashes -> each sample is its own leaf, but root has
    # children so this returns a trie, not None. None only for <2 samples.
    h1, l1 = create_segment_metadata([[("uid", 3)]])
    assert build_global_tree_from_segments([torch.tensor([1, 2, 3])], h1, l1) is None
    import numpy as np
    assert build_global_tree_from_segments(
        [], np.array([], dtype=object), np.array([], dtype=object)
    ) is None


def test_grpo_two_prompts_builds_leaf_per_sample():
    p0, p1 = list(range(10, 15)), list(range(20, 25))
    samples = [torch.tensor(p0 + [100, 101, 102]), torch.tensor(p0 + [200, 201, 202]),
               torch.tensor(p1 + [300, 301, 302]), torch.tensor(p1 + [400, 401, 402])]
    hashes, lengths = create_grpo_segment_metadata(["p0", "p0", "p1", "p1"], [5] * 4, rollout_n=2)
    trie = build_global_tree_from_segments(samples, hashes, lengths)
    assert isinstance(trie, PrefixTrie)
    # Each sample maps to a leaf via trie.leaves[sample_idx]
    assert len(trie.leaves) == 4 and all(l is not None for l in trie.leaves)


def test_leaf_coverage_and_varying_segment_lengths():
    prefix = [1, 2]
    samples = [torch.tensor(prefix + [i]) for i in range(4)]
    hashes, lengths = create_segment_metadata([[("uid", 2), (f"r{i}", 1)] for i in range(4)])
    trie = build_global_tree_from_segments(samples, hashes, lengths)
    assert len(trie.leaves) == 4 and all(l is not None for l in trie.leaves)
    # Varying lengths: short prefix + long tail (1 vs 10)
    short = [torch.tensor([1] + list(range(100, 110))), torch.tensor([1] + list(range(200, 210)))]
    h3, l3 = create_segment_metadata([[("u", 1), ("a", 10)], [("u", 1), ("b", 10)]])
    trie2 = build_global_tree_from_segments(short, h3, l3)
    assert isinstance(trie2, PrefixTrie) and len(trie2.leaves) == 2
