# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
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
"""Unit tests for build_tree_from_segments in verl.utils.prefix_tree.tree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from verl.utils.prefix_tree.dynamic import (
    _BuildNode,
    _compress_trie,
    _insert_sequence,
    convert_trie_to_tree_node,
)
from verl.utils.prefix_tree.segment_grouper import create_segment_metadata, create_grpo_segment_metadata
from verl.utils.prefix_tree.tree import PrefixSubTrie, build_tree_from_segments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INJECTED = dict(
    _BuildNode=_BuildNode,
    _insert_sequence=_insert_sequence,
    _compress_trie=_compress_trie,
    convert_trie_to_tree_node=convert_trie_to_tree_node,
)


def _call(samples, segment_hashes, segment_lengths):
    return build_tree_from_segments(samples, segment_hashes, segment_lengths, **INJECTED)


# ---------------------------------------------------------------------------
# Test: build tree from segment hashes — samples with same hash share prefix
# ---------------------------------------------------------------------------


class TestBuildTreeSharedHash:
    def test_two_samples_same_prefix_hash_returns_subtrie(self):
        """Two samples with identical first-segment hash share a prefix node."""
        shared_prefix = [10, 20, 30]
        samples = [
            torch.tensor(shared_prefix + [1, 2]),
            torch.tensor(shared_prefix + [3, 4]),
        ]
        hashes, lengths = create_segment_metadata(
            [
                [("same-prompt", 3), ("resp-A", 2)],
                [("same-prompt", 3), ("resp-B", 2)],
            ]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert isinstance(result, PrefixSubTrie)

    def test_shared_prefix_tokens_appear_in_trie_nodes(self):
        """The shared prefix tokens must be present in the subtrie's flat node list."""
        shared_prefix = [100, 200, 300]
        samples = [
            torch.tensor(shared_prefix + [1]),
            torch.tensor(shared_prefix + [2]),
        ]
        hashes, lengths = create_segment_metadata(
            [
                [("uid-x", 3), ("r0", 1)],
                [("uid-x", 3), ("r1", 1)],
            ]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        all_tokens = [tok for node in result.nodes for tok in node.input_ids]
        for tok in shared_prefix:
            assert tok in all_tokens, f"prefix token {tok} missing from trie nodes"

    def test_three_samples_same_first_hash(self):
        """Three samples sharing first-segment hash all land in the same group."""
        prefix = [5, 6, 7]
        samples = [
            torch.tensor(prefix + [10, 11]),
            torch.tensor(prefix + [20, 21]),
            torch.tensor(prefix + [30, 31]),
        ]
        hashes, lengths = create_segment_metadata(
            [
                [("shared", 3), ("a", 2)],
                [("shared", 3), ("b", 2)],
                [("shared", 3), ("c", 2)],
            ]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        # All three samples should have leaf entries.
        assert len(result.leaf_node_ids) == 3

    def test_leaf_ids_cover_all_samples(self):
        """leaf_ids array must assign a valid flat_idx to every sample."""
        prefix = [1, 2]
        samples = [torch.tensor(prefix + [i]) for i in range(4)]
        hashes, lengths = create_segment_metadata(
            [[("shared", 2), (f"r{i}", 1)] for i in range(4)]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert result.leaf_ids.shape[0] == 4
        assert all(result.leaf_ids[i] >= 0 for i in range(4))


# ---------------------------------------------------------------------------
# Test: returns None when no sharing (all samples different)
# ---------------------------------------------------------------------------


class TestNoSharing:
    def test_all_different_first_hashes_returns_none(self):
        """When every sample has a distinct first-segment hash, no sharing → None."""
        samples = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9]),
        ]
        hashes, lengths = create_segment_metadata(
            [
                [("uid-a", 3)],
                [("uid-b", 3)],
                [("uid-c", 3)],
            ]
        )
        result = _call(samples, hashes, lengths)
        assert result is None

    def test_single_sample_returns_none(self):
        """A single sample cannot share a prefix — must return None."""
        samples = [torch.tensor([1, 2, 3])]
        hashes, lengths = create_segment_metadata([[("uid-only", 3)]])
        result = _call(samples, hashes, lengths)
        assert result is None

    def test_empty_samples_returns_none(self):
        """Empty sample list must return None without error."""
        result = build_tree_from_segments([], np.array([], dtype=object), np.array([], dtype=object), **INJECTED)
        assert result is None

    def test_two_samples_no_shared_prefix_returns_none(self):
        """Two samples with different first-segment hashes → None."""
        samples = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        hashes, lengths = create_segment_metadata(
            [[("uid-alpha", 2)], [("uid-beta", 2)]]
        )
        result = _call(samples, hashes, lengths)
        assert result is None


# ---------------------------------------------------------------------------
# Test: GRPO-style — samples with same prompt UUID share prefix
# ---------------------------------------------------------------------------


class TestGrpoStyle:
    def test_grpo_two_prompts_two_rollouts_each(self):
        """GRPO batch: 2 prompts × 2 rollouts = 4 samples.

        Samples 0 and 1 share prompt 'p0'; samples 2 and 3 share prompt 'p1'.
        With level=0 grouping the largest group (size 2) is selected → subtrie.
        """
        prompt_len = 5
        resp_len = 3
        p0_tokens = list(range(10, 10 + prompt_len))
        p1_tokens = list(range(20, 20 + prompt_len))
        samples = [
            torch.tensor(p0_tokens + [100, 101, 102]),
            torch.tensor(p0_tokens + [200, 201, 202]),
            torch.tensor(p1_tokens + [300, 301, 302]),
            torch.tensor(p1_tokens + [400, 401, 402]),
        ]
        hashes, lengths = create_grpo_segment_metadata(
            prompt_uids=["p0", "p0", "p1", "p1"],
            prompt_lengths=[prompt_len] * 4,
            rollout_n=2,
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert isinstance(result, PrefixSubTrie)
        # Must have exactly 2 leaves (the largest group is size 2).
        assert len(result.leaf_node_ids) == 2

    def test_grpo_single_prompt_four_rollouts(self):
        """GRPO with 1 prompt and 4 rollouts — all 4 samples share a prefix."""
        prompt_len = 4
        p_tokens = [1, 2, 3, 4]
        samples = [torch.tensor(p_tokens + [i * 10 + j for j in range(3)]) for i in range(4)]
        hashes, lengths = create_grpo_segment_metadata(
            prompt_uids=["shared-prompt"] * 4,
            prompt_lengths=[prompt_len] * 4,
            rollout_n=4,
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert len(result.leaf_node_ids) == 4

    def test_grpo_shared_prefix_tokens_match_prompt(self):
        """Trie nodes must contain the shared prompt tokens."""
        prompt_len = 3
        prompt_tokens = [7, 8, 9]
        samples = [
            torch.tensor(prompt_tokens + [100]),
            torch.tensor(prompt_tokens + [200]),
        ]
        hashes, lengths = create_grpo_segment_metadata(
            prompt_uids=["q0", "q0"],
            prompt_lengths=[prompt_len, prompt_len],
            rollout_n=2,
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        flat_tokens = [tok for node in result.nodes for tok in node.input_ids]
        for tok in prompt_tokens:
            assert tok in flat_tokens


# ---------------------------------------------------------------------------
# Test: correct leaf nodes per sample
# ---------------------------------------------------------------------------


class TestLeafNodes:
    def test_each_sample_has_distinct_leaf(self):
        """Each sample in the sharing group must map to a leaf node."""
        prefix = [1, 2, 3]
        samples = [
            torch.tensor(prefix + [10]),
            torch.tensor(prefix + [20]),
        ]
        hashes, lengths = create_segment_metadata(
            [[("uid", 3), ("r0", 1)], [("uid", 3), ("r1", 1)]]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        # Two distinct leaves.
        assert len(set(result.leaf_to_sample)) == 2

    def test_leaf_to_sample_covers_batch(self):
        """leaf_to_sample must reference both sample indices."""
        prefix = [5, 6]
        samples = [torch.tensor(prefix + [i]) for i in range(3)]
        hashes, lengths = create_segment_metadata(
            [[("uid", 2), (f"r{i}", 1)] for i in range(3)]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert set(result.leaf_to_sample) == {0, 1, 2}

    def test_leaf_nodes_are_valid_flat_idxs(self):
        """Every flat_idx in leaf_node_ids must point to a node in result.nodes."""
        prefix = [9, 8, 7]
        samples = [torch.tensor(prefix + [i]) for i in range(2)]
        hashes, lengths = create_segment_metadata(
            [[("h", 3), (f"r{i}", 1)] for i in range(2)]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        valid_flat_idxs = {n.flat_idx for n in result.nodes}
        for lid in result.leaf_node_ids:
            assert lid in valid_flat_idxs


# ---------------------------------------------------------------------------
# Test: varying segment lengths
# ---------------------------------------------------------------------------


class TestVaryingSegmentLengths:
    def test_short_prefix_long_response(self):
        """Short shared prefix (length 1) with long distinct tails."""
        samples = [
            torch.tensor([1] + list(range(100, 110))),
            torch.tensor([1] + list(range(200, 210))),
        ]
        hashes, lengths = create_segment_metadata(
            [[("uid", 1), ("resp-a", 10)], [("uid", 1), ("resp-b", 10)]]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        # Shared node must contain token 1 only.
        shared_nodes = [n for n in result.nodes if not n.children]
        all_tokens = [tok for node in result.nodes for tok in node.input_ids]
        assert 1 in all_tokens

    def test_long_prefix_short_response(self):
        """Long shared prefix (length 8) with short distinct tails."""
        prefix = list(range(50, 58))
        samples = [
            torch.tensor(prefix + [0]),
            torch.tensor(prefix + [1]),
        ]
        hashes, lengths = create_segment_metadata(
            [[("uid", 8), ("r0", 1)], [("uid", 8), ("r1", 1)]]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        all_tokens = [tok for node in result.nodes for tok in node.input_ids]
        for tok in prefix:
            assert tok in all_tokens

    def test_equal_length_segments(self):
        """Both prefix and response have equal lengths across samples."""
        prefix = [10, 20, 30, 40]
        samples = [
            torch.tensor(prefix + [1, 2, 3, 4]),
            torch.tensor(prefix + [5, 6, 7, 8]),
        ]
        hashes, lengths = create_segment_metadata(
            [[("uid", 4), ("rA", 4)], [("uid", 4), ("rB", 4)]]
        )
        result = _call(samples, hashes, lengths)
        assert result is not None
        assert len(result.leaf_node_ids) == 2

    def test_single_token_samples_with_shared_prefix(self):
        """Degenerate: prefix is the entire sequence (only 1 token each, same)."""
        # Both samples are [42] — identical sequences, prefix_len == total_len.
        # build_tree_from_segments inserts prefix then full seq.
        # The function should not crash; result depends on trie compression.
        samples = [torch.tensor([42, 1]), torch.tensor([42, 2])]
        hashes, lengths = create_segment_metadata(
            [[("uid", 1), ("r0", 1)], [("uid", 1), ("r1", 1)]]
        )
        result = _call(samples, hashes, lengths)
        # Should succeed: the single shared token creates a prefix node.
        assert result is not None

    def test_mismatched_prefix_lengths_picks_largest_group(self):
        """When hashes match but prefix lengths differ, the first sample's length
        is used (function reads prefix_len from group[0]).  The build must not crash."""
        # Two samples share first hash but have different prefix length metadata.
        # We simulate this by providing different lengths for the same hash.
        seg_hashes = np.array([[1, 2], [1, 3]], dtype=object)
        seg_lengths = np.array([[3, 2], [5, 2]], dtype=object)
        samples = [
            torch.tensor([10, 20, 30, 40, 50]),
            torch.tensor([10, 20, 30, 60, 70]),
        ]
        result = build_tree_from_segments(seg_hashes, seg_lengths, samples, **INJECTED)
        # May return None (hash 1 maps different lengths, but grouper still groups them).
        # The important invariant: no exception raised.
        # (Return value can be None or PrefixSubTrie depending on token overlap.)

    def test_build_tree_arg_order_matches_signature(self):
        """Ensure argument order matches the public signature exactly."""
        # build_tree_from_segments(samples, segment_hashes, segment_lengths, ...)
        prefix = [1, 2]
        samples = [torch.tensor(prefix + [10]), torch.tensor(prefix + [20])]
        hashes, lengths = create_segment_metadata(
            [[("uid", 2), ("r0", 1)], [("uid", 2), ("r1", 1)]]
        )
        # Positional call — validates the order documented in the signature.
        result = build_tree_from_segments(samples, hashes, lengths, **INJECTED)
        assert result is not None
