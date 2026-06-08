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
"""Prefix-tree + MAGI utilities for verl SFT training.

Dispatches a micro-batch through either the hash-based
(:mod:`verl.utils.prefix_tree_hash_based`) or dynamic-trie
(:mod:`verl.utils.prefix_tree_dynamic`) detection path, materialises a flat
layout via :func:`verl.utils.prefix_tree_utils.build_layout_from_tree_node`,
and builds a MAGI / flex attention key for the result.

Usage (inside gptmodel_forward_model_engine):

    pt_batch = build_prefix_tree_micro_batch(model, input_ids, loss_mask, position_ids)
    if pt_batch is not None:
        output = model(
            input_ids=pt_batch.flat_input_ids,
            attention_mask=None,
            position_ids=pt_batch.flat_position_ids,
            packed_seq_params=None,
            magi_attention_key=pt_batch.magi_key,
        )
        output = restore_flat_to_nested(output, pt_batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree_hash_based import (
    _hash_prefix,  # noqa: F401  re-exported for backwards-compat callers
    build_prefix_segments_single_turn,  # noqa: F401  re-exported for backwards-compat callers
    build_tree_hash_based,
)


@dataclass
class PrefixTreeMagiBatch:
    """Holds the flat layout and MAGI key for one prefix-tree micro-batch."""

    # flat input tensors ready to pass to model(...)
    flat_input_ids: Tensor  # (total_tokens,)
    flat_position_ids: Tensor  # (total_tokens,)
    flat_loss_mask: Optional[Tensor]  # (total_tokens,) or None

    # Attention keys — one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # mapping needed for output restoration
    # leaf_to_sample[i] = original sample index for leaf i
    leaf_to_sample: list[int]
    # leaf_ranges[i] = (start, end) token offset in flat layout for leaf i
    leaf_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]

    # original batch size (= number of leaves for single-level tree)
    original_batch_size: int

    # number of real (non-padding) tokens; may be < flat_input_ids.shape[0]
    # when tp_size > 1 padding was added for sequence-parallel divisibility
    real_tokens: int = 0

    # leaf_ancestor_ranges[i] = list of (start,end) flat ranges that precede leaf i
    # For single-level: None (use prefix_range directly)
    # For multilevel: [(0, root_end), (turn2_start, turn2_end)] etc.
    leaf_ancestor_ranges: Optional[list[list[tuple[int, int]]]] = None

    # CP-local tensors: after magi dispatch, each CP rank only processes its assigned tokens.
    # When CP=1, these equal flat_input_ids/flat_position_ids/flat_loss_mask.
    # Shape: (local_tokens, ...) where local_tokens = total_tokens / cp_effective
    local_flat_input_ids: Optional[Tensor] = None
    local_flat_position_ids: Optional[Tensor] = None
    local_flat_loss_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.real_tokens == 0:
            self.real_tokens = int(self.flat_input_ids.shape[0])
        # Default local to full when not set (CP=1 or flex path)
        if self.local_flat_input_ids is None:
            self.local_flat_input_ids = self.flat_input_ids
        if self.local_flat_position_ids is None:
            self.local_flat_position_ids = self.flat_position_ids
        if self.local_flat_loss_mask is None:
            self.local_flat_loss_mask = self.flat_loss_mask


def _unpack_nested_to_list(x, pad_token_id=None) -> Optional[list[Tensor]]:
    """Unpack a NestedTensor or padded 2-D Tensor into a list of 1-D tensors.

    - NestedTensor (jagged): uses ``.offsets()``
    - Padded 2-D Tensor ``(B, T)``: trims trailing pad_token_id from each row.
      If pad_token_id is None, trims zero-valued tokens.
    - ``None``: returns ``None``
    """
    if x is None:
        return None
    if hasattr(x, "is_nested") and x.is_nested:
        offsets = x.offsets()
        lengths = offsets.diff().tolist()
        vals = x.values()
        out: list[Tensor] = []
        pos = 0
        for length in lengths:
            out.append(vals[pos : pos + int(length)])
            pos += int(length)
        return out
    if x.dim() == 2:
        # Padded 2-D tensor — cannot safely unpack without risk of pad/loss_mask misalignment.
        # Return None to fall back to standard attention.
        return None
    return None


def build_prefix_tree_micro_batch(
    model,
    input_ids: NestedTensor,
    loss_mask: Optional[NestedTensor] = None,
    position_ids: Optional[NestedTensor] = None,
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    dynamic_trie: bool = False,
) -> Optional[PrefixTreeMagiBatch]:
    """Build a PrefixTreeMagiBatch from a micro-batch of NestedTensor sequences.

    Two detection paths, selected by ``dynamic_trie``:

      - ``dynamic_trie=False`` (default): hash-based fast path. Uses
        ``prefix_segments_batch`` (per-sample turn-level hashes) when
        provided, otherwise falls back to a token-level LCP scan. Supports
        depth-1 and depth-2 tree shapes only.
      - ``dynamic_trie=True``: token-by-token trie insertion. Detects
        arbitrary-depth shared-prefix trees directly from the token
        sequences. ``prefix_segments_batch`` is ignored on this path.

    Returns None when there is no shared prefix, signalling the caller to
    fall back to the standard attention path.

    Args:
        model: Megatron model (used to read num_heads / head_dim from config).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        loss_mask: Optional NestedTensor matching input_ids shape.
        position_ids: Optional NestedTensor matching input_ids shape.
            When None, default RoPE-compatible position IDs are generated.
        prefix_segments_batch: Optional per-sample prior knowledge injected by
            the dataset or trainer (hash-path only). See
            :func:`verl.utils.prefix_tree_hash_based.build_tree_hash_based`
            for the expected format. Ignored when ``dynamic_trie=True``.
        attention_type: ``"flex"`` or ``"magi"``.
        tp_size / cp_size: Tensor / context parallel world sizes (for
            SP-divisibility padding).
        dynamic_trie: when True, dispatch to the trie path in
            :mod:`verl.utils.prefix_tree_dynamic`. Default False.

    Returns:
        PrefixTreeMagiBatch or None.
    """

    import os as _os_pt
    import torch as _torch_pt
    from verl.utils.prefix_tree_utils import build_layout_from_tree_node

    _magi_timing = _os_pt.environ.get("MAGI_TIMING") == "1"
    def _ev(): e = _torch_pt.cuda.Event(enable_timing=True); e.record(); return e

    samples = _unpack_nested_to_list(input_ids)
    if not samples:
        return None
    loss_masks_by_sample = _unpack_nested_to_list(loss_mask)
    position_ids_by_sample = _unpack_nested_to_list(position_ids)

    if _magi_timing: _torch_pt.cuda.nvtx.range_push("prefix_tree/build_tree")
    t_tree0 = _ev() if _magi_timing else None
    if dynamic_trie:
        from verl.utils.prefix_tree_dynamic import build_tree_dynamic
        result = build_tree_dynamic(samples)
    else:
        result = build_tree_hash_based(samples, prefix_segments_batch=prefix_segments_batch)
    t_tree1 = _ev() if _magi_timing else None
    if _magi_timing: _torch_pt.cuda.nvtx.range_pop()

    if result is None:
        return None
    tree_root, leaf_to_sample = result

    try:
        if _magi_timing: _torch_pt.cuda.nvtx.range_push("prefix_tree/build_layout")
        t_layout0 = _ev() if _magi_timing else None
        params = build_layout_from_tree_node(
            samples,
            tree_root,
            leaf_to_sample,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
        t_layout1 = _ev() if _magi_timing else None
        if _magi_timing: _torch_pt.cuda.nvtx.range_pop()

        if _magi_timing: _torch_pt.cuda.nvtx.range_push("prefix_tree/finalize")
        t_final0 = _ev() if _magi_timing else None
        ret = _finalize_prefix_tree_batch(
            params,
            model=model,
            num_samples=len(samples),
            attention_type=attention_type,
            tp_size=tp_size,
            cp_size=cp_size,
        )
        t_final1 = _ev() if _magi_timing else None
        if _magi_timing: _torch_pt.cuda.nvtx.range_pop()

        if _magi_timing:
            _torch_pt.cuda.synchronize()
            N = len(samples)
            print(
                f"[MAGI-TIMING] build_tree_mb n={N}"
                f" tree={t_tree0.elapsed_time(t_tree1):.2f}ms"
                f" layout={t_layout0.elapsed_time(t_layout1):.2f}ms"
                f" finalize={t_final0.elapsed_time(t_final1):.2f}ms"
                f" total={t_tree0.elapsed_time(t_final1):.2f}ms",
                flush=True,
            )
        return ret
    except (ValueError, AssertionError, RuntimeError):
        # Multilevel tree produced non-monotonic leaf ranges or inconsistent
        # token layout.  Fall back to standard attention for this micro-batch.
        return None


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
) -> NestedTensor:
    """Restore a flat (total_tokens, ...) tensor to a per-sample NestedTensor.

    Each sample's view is ``[prefix_tokens || ancestor_tokens... || leaf_tokens]``
    concatenated, matching the original per-sample sequence length.

    Args:
        flat_tensor: Tensor with first dimension == total_tokens.
        pt_batch: PrefixTreeMagiBatch from build_prefix_tree_micro_batch.

    Returns:
        NestedTensor of shape (batch_size, variable_seqlen, ...).
    """
    prefix_start, prefix_end = pt_batch.prefix_range
    prefix_slice = flat_tensor[prefix_start:prefix_end]

    n = pt_batch.original_batch_size
    sample_tensors: list[Optional[Tensor]] = [None] * n

    for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
        leaf_start, leaf_end = pt_batch.leaf_ranges[leaf_idx]
        leaf_slice = flat_tensor[leaf_start:leaf_end]
        if pt_batch.leaf_ancestor_ranges is not None:
            parts = [flat_tensor[s:e] for s, e in pt_batch.leaf_ancestor_ranges[leaf_idx]]
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            sample_tensors[sample_idx] = torch.cat([prefix_slice, leaf_slice], dim=0)

    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by leaf_to_sample"
    )

    # Use as_nested_tensor (not nested_tensor) to preserve grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finalize_prefix_tree_batch(
    params,
    model,
    num_samples: int,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
) -> PrefixTreeMagiBatch:
    """Common downstream step for both detection paths.

    Pads to TP/CP divisibility, builds the requested attention key, and wraps
    the result into a :class:`PrefixTreeMagiBatch`. Padding tokens are not
    added to the attention rectangles — they are stripped before loss, and
    MAGI assigns zero attention weight to out-of-range positions.
    """
    real_tokens = params.flat_tokens.shape[0]
    if tp_size > 1:
        align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
        pad_len = (align_size - real_tokens % align_size) % align_size
        if pad_len > 0:
            params.flat_tokens = torch.cat([params.flat_tokens, params.flat_tokens.new_zeros(pad_len)])
            params.flat_position_ids = torch.cat(
                [params.flat_position_ids, params.flat_position_ids.new_zeros(pad_len)]
            )
            if params.flat_loss_mask is not None:
                params.flat_loss_mask = torch.cat([params.flat_loss_mask, params.flat_loss_mask.new_zeros(pad_len)])
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len

    if attention_type == "magi":
        magi_key = _build_magi_key(model, params)
        flex_key = None
    else:
        flex_key = _build_flex_key(params, params.flat_tokens.device)
        magi_key = None

    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=magi_key,
        flex_key=flex_key,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=num_samples,
        real_tokens=real_tokens,
        leaf_ancestor_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_flat_input_ids=params.flat_tokens,
        local_flat_position_ids=params.flat_position_ids,
        local_flat_loss_mask=params.flat_loss_mask,
    )


def _build_flex_key(params, device):
    """Build a torch flex_attention block_mask from PrefixTreeParams.

    The mask encodes the prefix-tree attention pattern:
    - Prefix tokens: causal self-attention
    - Leaf tokens: full attention to prefix + causal self-attention within same leaf
    - Cross-leaf attention: blocked (leaf_i cannot see leaf_j)

    Returns a compiled block_mask usable with torch.nn.attention.flex_attention.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    total = params.total_seqlen_q
    prefix_end = params.prefix_range[1]  # == prefix_len

    leaf_id = torch.full((total,), -1, dtype=torch.int32)
    for i, (s, e) in enumerate(params.leaf_ranges):
        leaf_id[s:e] = i
    leaf_id = leaf_id.to(device)

    def prefix_tree_mask(b, h, q_idx, kv_idx):
        q_leaf = leaf_id[q_idx]
        k_leaf = leaf_id[kv_idx]
        in_prefix_k = kv_idx < prefix_end
        same_leaf = (q_leaf == k_leaf) & (q_leaf >= 0)
        causal = kv_idx <= q_idx
        return (in_prefix_k & causal) | (same_leaf & causal) | (in_prefix_k & (q_leaf >= 0))

    # _compile=False: avoid Triton JIT which takes minutes for new shapes.
    # Memory is handled at the call site via torch.utils.checkpoint.
    block_mask = create_block_mask(
        prefix_tree_mask, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device, _compile=False
    )
    block_mask._leaf_id = leaf_id  # keep closure alive
    return block_mask


def _build_magi_key(model, params):
    """Construct a magi_attn_flex_key from PrefixTreeParams and model config."""
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    from verl.utils.megatron_utils import unwrap_model

    cfg = unwrap_model(model).config
    num_heads_q = cfg.num_attention_heads
    # GQA: num_query_groups may be set; fall back to num_heads_q if not
    num_heads_kv = getattr(cfg, "num_query_groups", num_heads_q) or num_heads_q
    head_dim = cfg.kv_channels  # hidden_size // num_attention_heads

    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(params.q_ranges),
        k_ranges=AttnRanges.from_ranges(params.k_ranges),
        attn_mask_type=[AttnMaskType(m) for m in params.mask_types],
        total_seqlen_q=params.total_seqlen_q,
        total_seqlen_k=params.total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )


def _build_magi_key_sp_scaled(original_key, model, tp_size: int):
    """Rebuild a MAGI key scaled for the SP-scattered token domain (T/TP seqlen).

    With sequence_parallel=True, the embedding scatter gives T/TP tokens per TP rank.
    The MAGI key must use T/TP as total_seqlen so dispatch/undispatch in
    _magi_attn_forward operate on T/TP tokens → T/(TP*CP) per rank → back to T/TP.
    """
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    q_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.q_ranges]
    k_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.k_ranges]
    total_q_sp = original_key.total_seqlen_q // tp_size
    total_k_sp = original_key.total_seqlen_k // tp_size

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(q_ranges_sp),
        k_ranges=AttnRanges.from_ranges(k_ranges_sp),
        attn_mask_type=list(original_key.attn_mask_type),
        total_seqlen_q=total_q_sp,
        total_seqlen_k=total_k_sp,
        num_heads_q=original_key.num_heads_q,
        num_heads_kv=original_key.num_heads_kv,
        head_dim=original_key.head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )
