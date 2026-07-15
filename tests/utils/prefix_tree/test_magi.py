# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU-only unit tests for verl/utils/prefix_tree/magi.py.

These tests exercise build_prefix_tree_micro_batch (without the MAGI key
construction, which requires GPU + distributed) and restore_flat_to_nested.
All prefix-tree utilities now live in verl.utils — no Megatron-LM fork needed.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nested(token_lists: list[list[int]], device="cpu") -> torch.Tensor:
    """Build a jagged NestedTensor from a list of token ID lists."""
    tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in token_lists]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


def _make_nested_float(value_lists: list[list[float]], device="cpu") -> torch.Tensor:
    tensors = [torch.tensor(v, dtype=torch.float32, device=device) for v in value_lists]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


# ---------------------------------------------------------------------------
# Tests for the core layout helpers (no MAGI key, no model)
# ---------------------------------------------------------------------------


class TestBuildPrefixTreeLayout:
    """Tests that verify the flat layout and PrefixTreeParams fields.

    Uses the trie-based path (build_tree_dynamic → build_layout_from_tree_node)
    to keep these tests CPU-only.
    """

    @staticmethod
    def _build_params_from_tokens(tokens):
        from verl.utils.prefix_tree.dynamic import build_tree_dynamic
        from verl.utils.prefix_tree.utils import build_layout_from_tree_node

        result = build_tree_dynamic(tokens)
        assert result is not None, "Expected shared prefix trie"
        return build_layout_from_tree_node(tokens, result)

    def test_basic_shared_prefix(self):
        from verl.utils.prefix_tree.utils import longest_common_prefix_length

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
            torch.tensor([10, 20, 30, 61, 62, 63]),
        ]
        prefix_len = longest_common_prefix_length(tokens)
        assert prefix_len == 3

        params = self._build_params_from_tokens(tokens)

        # flat layout: [10,20,30] + [41,42] + [51] + [61,62,63]
        assert torch.equal(params.tree_packed_tokens, torch.tensor([10, 20, 30, 41, 42, 51, 61, 62, 63]))
        assert params.prefix_range == (0, 3)
        assert params.leaf_ranges == [(3, 5), (5, 6), (6, 9)]
        assert params.total_seqlen_q == 9

    def test_no_shared_prefix_returns_zero(self):
        from verl.utils.prefix_tree.utils import longest_common_prefix_length

        tokens = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ]
        assert longest_common_prefix_length(tokens) == 0

    def test_flex_spec_structure(self):
        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
        params = self._build_params_from_tokens(tokens)

        # Expected rectangles:
        # (0,3)→(0,3) causal  — prefix self
        # (3,5)→(0,3) full    — leaf_0 attends to prefix
        # (3,5)→(3,5) causal  — leaf_0 self
        # (5,6)→(0,3) full    — leaf_1 attends to prefix
        # (5,6)→(5,6) causal  — leaf_1 self
        rects = set(zip(params.q_ranges, params.k_ranges, params.mask_types, strict=False))
        assert ((0, 3), (0, 3), "causal") in rects
        assert ((3, 5), (0, 3), "full") in rects
        assert ((3, 5), (3, 5), "causal") in rects
        assert ((5, 6), (0, 3), "full") in rects
        assert ((5, 6), (5, 6), "causal") in rects
        assert len(rects) == 5


# ---------------------------------------------------------------------------
# Tests for restore_flat_to_nested
# ---------------------------------------------------------------------------


class TestRestoreFlatToNested:
    """Tests that verify round-trip: build params → restore → original tensors."""

    @staticmethod
    def _build_pt_batch(tokens):
        """Build a PrefixTreeMagiBatch stub without the MAGI key."""
        from verl.utils.prefix_tree.dynamic import build_tree_dynamic
        from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch
        from verl.utils.prefix_tree.utils import build_layout_from_tree_node

        result = build_tree_dynamic(tokens)
        assert result is not None
        params = build_layout_from_tree_node(tokens, result)
        return PrefixTreeMagiBatch(
            tree_packed_input_ids=params.tree_packed_tokens,
            tree_packed_position_ids=params.tree_packed_position_ids,
            tree_packed_loss_mask=None,
            magi_key=None,
            flex_key=None,
            segment_to_sample=params.leaf_to_sample,
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            original_batch_size=params.num_samples,
        )

    def test_restore_token_ids(self):
        from verl.utils.prefix_tree.magi import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
            torch.tensor([10, 20, 30, 61, 62, 63]),
        ]
        pt_batch = self._build_pt_batch(tokens)
        flat = pt_batch.tree_packed_input_ids  # (9,)

        restored = restore_flat_to_nested(flat, pt_batch)

        # restored is a NestedTensor; check each sample
        offsets = restored.offsets()
        vals = restored.values()
        lengths = offsets.diff().tolist()

        assert lengths == [5, 4, 6]
        pos = 0
        for i, orig in enumerate(tokens):
            length = int(lengths[i])
            assert torch.equal(vals[pos : pos + length], orig), f"sample {i} mismatch"
            pos += length

    def test_restore_with_extra_dim(self):
        """restore_flat_to_nested should work for (total_tokens, D) tensors too."""
        from verl.utils.prefix_tree.magi import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
        pt_batch = self._build_pt_batch(tokens)

        # Simulate logit output: (total_tokens=6, vocab=8)
        flat_logits = torch.randn(6, 8)
        restored = restore_flat_to_nested(flat_logits, pt_batch)

        offsets = restored.offsets()
        lengths = offsets.diff().tolist()
        assert lengths == [5, 4]

    def test_restore_preserves_prefix(self):
        """Each restored sample must start with the shared prefix tokens."""
        from verl.utils.prefix_tree.magi import restore_flat_to_nested

        tokens = [
            torch.tensor([10, 20, 30, 41]),
            torch.tensor([10, 20, 30, 51, 52]),
        ]
        pt_batch = self._build_pt_batch(tokens)
        flat = pt_batch.tree_packed_input_ids

        restored = restore_flat_to_nested(flat, pt_batch)
        vals = restored.values()
        offsets = restored.offsets()
        lengths = offsets.diff().tolist()

        # Both samples should start with [10, 20, 30]
        pos = 0
        for length in lengths:
            sample = vals[pos : pos + int(length)]
            assert torch.equal(sample[:3], torch.tensor([10, 20, 30]))
            pos += int(length)


# ---------------------------------------------------------------------------
# Integration: NestedTensor input → flat layout (no MAGI key)
# ---------------------------------------------------------------------------


class TestNestedTensorUnpack:
    """Tests that build_prefix_tree_micro_batch correctly unpacks NestedTensors."""

    def test_nested_tensor_unpack_and_flat_layout(self, monkeypatch):
        """Monkeypatch _build_magi_key to skip GPU/dist; verify flat layout."""
        import verl.utils.prefix_tree.forward as ptf
        import verl.utils.prefix_tree.magi as ptm

        # Stub out _build_magi_key so we don't need GPU
        monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())

        # Stub model with config
        class FakeConfig:
            num_attention_heads = 8
            num_query_groups = 8
            kv_channels = 128
            fp8 = None

        class FakeModel:
            config = FakeConfig()
            pre_process = True
            post_process = True

        model = FakeModel()

        token_lists = [
            [10, 20, 30, 41, 42],
            [10, 20, 30, 51],
            [10, 20, 30, 61, 62, 63],
        ]
        input_ids = _make_nested(token_lists)

        result = ptm.build_prefix_tree_micro_batch(model, input_ids)

        assert result is not None
        assert result.original_batch_size == 3
        assert torch.equal(result.tree_packed_input_ids, torch.tensor([10, 20, 30, 41, 42, 51, 61, 62, 63]))
        assert result.prefix_range == (0, 3)
        assert result.segment_ranges == [(3, 5), (5, 6), (6, 9)]

    def test_no_shared_prefix_returns_none(self, monkeypatch):
        import verl.utils.prefix_tree.forward as ptf
        import verl.utils.prefix_tree.magi as ptm

        monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())

        class FakeModel:
            class config:
                num_attention_heads = 8
                num_query_groups = 8
                kv_channels = 128
                fp8 = None

            pre_process = True
            post_process = True

        token_lists = [[1, 2, 3], [4, 5, 6]]
        input_ids = _make_nested(token_lists)

        result = ptm.build_prefix_tree_micro_batch(FakeModel(), input_ids)
        assert result is None

    def test_loss_mask_flattened(self, monkeypatch):
        """loss_mask NestedTensor is correctly flattened into flat_loss_mask."""
        import verl.utils.prefix_tree.forward as ptf
        import verl.utils.prefix_tree.magi as ptm

        monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())

        class FakeModel:
            class config:
                num_attention_heads = 8
                num_query_groups = 8
                kv_channels = 128
                fp8 = None

            pre_process = True
            post_process = True

        token_lists = [[10, 20, 30, 41, 42], [10, 20, 30, 51]]
        loss_lists = [[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

        input_ids = _make_nested(token_lists)
        loss_mask = _make_nested_float(loss_lists)

        result = ptm.build_prefix_tree_micro_batch(FakeModel(), input_ids, loss_mask=loss_mask)
        assert result is not None
        assert result.tree_packed_loss_mask is not None
        # flat: [prefix_loss(3)] + [leaf_0_loss(2)] + [leaf_1_loss(1)]
        # prefix loss taken from sample 0: [0,0,1], leaf_0: [1,1], leaf_1: [1]
        assert torch.equal(
            result.tree_packed_loss_mask,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        )
