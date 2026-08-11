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
"""Prefix-tree + MAGI utilities: flat layout packing, MAGI/flex keys, rope/decoder overrides, and flat→nested restore."""

from __future__ import annotations

import contextlib
import functools
import logging as _log
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


@dataclass
class PackRestorationParam:
    """Per-micro-batch layout info for restoring flat tensors to per-sample.

    Computed by build_layout_from_tree_node, consumed by restore_flat_to_nested.
    Separate from PrefixTreeMagiBatch (forward-pass data) — this is only for
    unpacking model output back to per-sample tensors.
    """

    # Per-leaf flat token range in the packed layout.
    segment_ranges: list[tuple[int, int]]
    # Shared prefix range (start, end) in flat layout.
    prefix_range: tuple[int, int]
    # Per-leaf ancestor ranges: ancestor_segment_ranges[i] = [(start,end), ...]
    # None for single-level trees (use prefix_range directly).
    ancestor_segment_ranges: Optional[list[list[tuple[int, int]]]] = None
    # Boundary registry for LCE boundary-patch. None when no branching.
    boundary_registry: Optional[object] = None

    def segment_to_sample(self, subtrie) -> list[int]:
        """Fetch leaf-to-sample mapping from the subtrie (live, not stored)."""
        return subtrie.leaf_to_sample

    def original_batch_size(self, subtrie) -> int:
        """Number of unique samples (from subtrie)."""
        return len(subtrie.global_sample_ids)

    def real_tokens(self, pt_batch: PrefixTreeMagiBatch) -> int:
        """Real token count (from packed input_ids shape)."""
        return pt_batch.tree_packed_input_ids.shape[0]


@dataclass
class PrefixTreeMagiBatch:
    """Holds the tree-packed layout and MAGI key for one prefix-tree micro-batch."""

    # tree-packed input tensors ready to pass to model(...)
    tree_packed_input_ids: Tensor  # (total_tokens,)
    tree_packed_position_ids: Tensor  # (total_tokens,)

    # Attention keys: one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # Per-token labels derived from tree_packed_tokens via within-segment shift
    tree_packed_labels: Optional[Tensor] = None  # (total_tokens,)

    # Restoration params for unpacking model output to per-sample tensors
    restoration: Optional[PackRestorationParam] = None

    # Live subtrie reference for restoration lookups (leaf_to_sample, etc.)
    subtrie: Optional[object] = None


def build_prefix_tree_micro_batch(
    model,
    input_ids: NestedTensor,
    loss_mask: Optional[NestedTensor] = None,
    position_ids: Optional[NestedTensor] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    subtrie: Optional[PrefixSubTrie] = None,
) -> Optional[PrefixTreeMagiBatch]:
    """Build a PrefixTreeMagiBatch from a per-mb subtrie (built once per step in prepare_prefix_tree_micro_batches).

    The subtrie is reused across OLP + actor update forwards. Returns None when subtrie absent."""
    # Lazy import to avoid cycle with forward.py.
    from verl.utils.prefix_tree.forward import (
        _finalize_prefix_tree_batch,
        _unpack_nested_to_list,
    )

    samples = _unpack_nested_to_list(input_ids, mask=loss_mask)
    if not samples:
        _log.getLogger(__name__).warning("prefix_tree: build_prefix_tree_micro_batch got empty samples; returning None")
        return None
    loss_masks_by_sample = _unpack_nested_to_list(loss_mask)
    position_ids_by_sample = _unpack_nested_to_list(position_ids, mask=loss_mask)

    if subtrie is None:
        # The per-microbatch subtrie MUST be built once globally on the driver
        # (build_global_trie -> create_and_attach_subtrie_views) and transmitted
        # to workers via the batch's prefix_tree_subtree field. Rebuilding per
        # micro-batch here (the old build_tree_dynamic fallback) is wrong: it
        # sees only this mb's samples, so prefix-sharing detection is local
        # instead of global, AND it costs ~13s/step (5x greedy_build_tries on
        # the actor hot path, starving the GPU). Fail loudly instead of
        # silently degrading correctness + perf.
        raise RuntimeError(
            "build_prefix_tree_micro_batch: prefix_tree_subtree is None. The global "
            "trie was not built/transmitted to this worker (build_global_trie not called "
            "on the driver, or prefix_tree_subtree did not survive dispatch). Per-microbatch "
            "rebuild is disabled — fix the driver to attach the global trie."
        )

    params = build_layout_from_tree_node(
        samples,
        subtrie,
        loss_masks_by_sample=loss_masks_by_sample,
        position_ids_by_sample=position_ids_by_sample,
    )
    return _finalize_prefix_tree_batch(
        params,
        model=model,
        num_samples=len(samples),
        attention_type=attention_type,
        tp_size=tp_size,
        cp_size=cp_size,
        subtrie=subtrie,
    )


def _build_sample_tensors(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
    boundary_logps: Optional[dict[int, list[tuple[int, Tensor]]]] = None,
) -> list:
    """Build per-sample tensor list from flat tensor, applying per-leaf boundary log-prob patches via split-and-cat.

    boundary_logps maps sample_idx → [(boundary_flat_pos, per_leaf_logp), ...] to fix the
    dedup→one-slot→copy-to-all-leaves bug where non-owner leaves inherit wrong boundary log-probs."""
    prefix_start, prefix_end = pt_batch.restoration.prefix_range
    prefix_slice = flat_tensor[prefix_start:prefix_end]
    # n = local sample count. Use leaf_to_sample (picklable list), not subtrie.leaves (indexed by GLOBAL sequence_ids).
    n = len(pt_batch.subtrie.leaf_to_sample)
    sample_tensors: list[Optional[Tensor]] = [None] * n
    for leaf_idx, sample_idx in enumerate(pt_batch.subtrie.leaf_to_sample):
        s, e = pt_batch.restoration.segment_ranges[leaf_idx]
        leaf_slice = flat_tensor[s:e]
        if pt_batch.restoration.ancestor_segment_ranges is not None:
            ranges = pt_batch.restoration.ancestor_segment_ranges[leaf_idx]
            parts: list[Tensor] = []
            for a, b in ranges:
                part = flat_tensor[a:b]
                # If boundary_logps is active, patch any boundary falling inside
                # this ancestor range (a, b).  A boundary at flat position b_pos
                # splits the slice: cat([flat[a:b_pos], leaf_val, flat[b_pos+1:b]]).
                # At most one boundary per range (each ancestor has one last token).
                if boundary_logps is not None:
                    for b_pos, leaf_val in boundary_logps.get(sample_idx, []):
                        if a <= b_pos < b:
                            part = torch.cat(
                                [flat_tensor[a:b_pos], leaf_val.unsqueeze(0), flat_tensor[b_pos + 1 : b]],
                                dim=0,
                            )
                            break
                parts.append(part)
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            # Single-level: prefix_slice contains the boundary (if any).
            prefix_part = prefix_slice
            if boundary_logps is not None:
                for b_pos, leaf_val in boundary_logps.get(sample_idx, []):
                    if prefix_start <= b_pos < prefix_end:
                        prefix_part = torch.cat(
                            [
                                flat_tensor[prefix_start:b_pos],
                                leaf_val.unsqueeze(0),
                                flat_tensor[b_pos + 1 : prefix_end],
                            ],
                            dim=0,
                        )
                        break
            sample_tensors[sample_idx] = torch.cat([prefix_part, leaf_slice], dim=0)
    return sample_tensors


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
    apply_boundary_patch: bool = False,
) -> NestedTensor:
    """Restore flat tensor to per-sample NestedTensor. Set apply_boundary_patch=True for log_probs."""
    boundary_logps = None
    if apply_boundary_patch:
        boundary_logps = getattr(pt_batch, "_boundary_logps", None)
    sample_tensors = _build_sample_tensors(flat_tensor, pt_batch, boundary_logps=boundary_logps)
    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by segment_to_sample"
    )
    # as_nested_tensor (not nested_tensor) preserves grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


@contextlib.contextmanager
def prefix_tree_rope_context(model, position_ids: Optional[Tensor]):
    """Override rope_mod.forward to use per-token position_ids instead of CP-rank sequential slicing.

    Handles standard RotaryEmbedding (index full table by position_ids) and M-RoPE (broadcast 1D→3D text-only)."""
    rope_mod = getattr(model, "rotary_pos_emb", None)
    if rope_mod is None or position_ids is None:
        yield
        return

    pids = position_ids.reshape(-1)
    _real_rope_fwd = rope_mod.forward

    # M-RoPE modules don't have get_emb; their forward takes position_ids (3D)
    # instead of max_seq_len (int).  Qwen3.5 text-only uses this path.
    _is_mrope = not hasattr(rope_mod, "get_emb")

    if _is_mrope:
        _mrope_section = getattr(model, "mrope_section", None)

        def _rope_fwd_with_pids(*args, **kwargs):
            # Broadcast 1D per-token positions to [3, 1, T]; text-only: all
            # three M-RoPE dims (temporal/height/width) are identical.
            pids_3d = pids.view(1, 1, -1).expand(3, 1, -1).contiguous()
            return _real_rope_fwd(pids_3d, _mrope_section, cp_group=None)
    else:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # bypass lru_cache

        def _rope_fwd_with_pids(max_seq_len, offset=0, packed_seq=False, cp_group=None):
            actual_seq_len = int(pids.max().item()) + 1
            emb = _orig_rope_fn(rope_mod, actual_seq_len, offset=0, packed_seq=True, cp_group=None)
            # All PP stages use seq-first Q=(seq,1,H,D); freqs=(seq,1,1,dim)
            # broadcasts correctly: Q×freqs → (seq,1,H,D).
            indexed = emb[pids.to(emb.device)]
            return indexed

    rope_mod.forward = _rope_fwd_with_pids
    try:
        yield
    finally:
        rope_mod.forward = _real_rope_fwd


@contextlib.contextmanager
def prefix_tree_decoder_key_context(model, magi_attention_key=None, flex_attention_key=None):
    """Override model.decoder.forward to inject magi/flex attention key into kwargs for one call."""
    if magi_attention_key is None and flex_attention_key is None:
        yield
        return
    _real_decoder_forward = model.decoder.forward

    @functools.wraps(_real_decoder_forward)
    def _decoder_forward_with_key(*args, **kw):
        return _real_decoder_forward(
            *args,
            magi_attention_key=magi_attention_key,
            flex_attention_key=flex_attention_key,
            **kw,
        )

    model.decoder.forward = _decoder_forward_with_key
    try:
        yield
    finally:
        model.decoder.forward = _real_decoder_forward


# model-forward helpers: consumed by verl/models/mcore/model_forward.py


_PREFIX_TREE_KEYS = frozenset(
    {
        "use_prefix_tree",
        "prefix_tree_attention",
        "prefix_tree_subtree",
    }
)


def strip_prefix_tree_args(logits_processor_args: dict | None) -> None:
    """Remove prefix-tree keys from *logits_processor_args* (mutates dict).

    Called after the prefix-tree path has consumed them so they don't
    leak into the downstream logits processor.
    """
    if logits_processor_args is None:
        return
    for k in _PREFIX_TREE_KEYS:
        logits_processor_args.pop(k, None)


def read_prefix_tree_batch_config(batch, tu, use_remove_padding: bool = True) -> tuple[bool, str]:
    """Read and validate prefix-tree flags from a batch non-tensor dict.

    Returns (use_prefix_tree, prefix_tree_attention).
    """
    use_prefix_tree = tu.get_non_tensor_data(batch, key="use_prefix_tree", default=False)
    prefix_tree_attention = tu.get_non_tensor_data(batch, key="prefix_tree_attention", default="flex")
    if use_prefix_tree:
        assert use_remove_padding, (
            "use_prefix_tree=True requires use_remove_padding=True (THD format). "
            "Set model.use_remove_padding=True in your config."
        )
        assert prefix_tree_attention in ("flex", "magi"), (
            f"prefix_tree_attention must be 'flex' or 'magi', got {prefix_tree_attention!r}"
        )
    return use_prefix_tree, prefix_tree_attention


def get_prefix_tree_logits_args(batch, tu) -> dict:
    """Build the prefix-tree fragment for logits_processor_args from a batch.

    The per-mb subtrie (built once in prepare_prefix_tree_micro_batches as
    a pruned view of the global trie) is the only thing needed here.
    """
    use_prefix_tree = tu.get_non_tensor_data(batch, key="use_prefix_tree", default=False)
    if not use_prefix_tree:
        return {}
    return {
        "use_prefix_tree": True,
        "prefix_tree_attention": tu.get_non_tensor_data(batch, key="prefix_tree_attention", default="flex"),
        "prefix_tree_subtree": tu.get_non_tensor_data(batch, "prefix_tree_subtree", default=None),
    }
