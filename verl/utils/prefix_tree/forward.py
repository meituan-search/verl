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
"""Prefix-tree forward-path implementations.

Split out of :mod:`verl.utils.prefix_tree.magi` to keep the data-structure /
config helpers separate from the actual forward-pass code.  Everything in this
module is consumed only by the prefix-tree forward path
(``verl.models.mcore.model_forward`` and ``model_forward_fused``) or by other
functions in this module.

Public entry points:

- :func:`unfuse_try_forward_prefix_tree`: unfused-path driver.
- :func:`fuse_try_forward_prefix_tree`: fused-path driver.
- :func:`fuse_forward_body`: fused-path body invoked by the patched
  ``_fused_GPTModel_forward``.
- :func:`dispatch_magi` (renamed from ``dispatch_pt_batch``): slices
  per-CP-rank local tensors via magi dispatch.
"""

from __future__ import annotations

import logging as _log
from typing import Optional

import torch
import torch.distributed as _dist
from magi_attention.api import (
    DistAttnConfig,
    OverlapConfig,
    get_position_ids,
    magi_attn_flex_key,
    undispatch,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.solver.dispatch_solver import DispatchConfig
from megatron.core import parallel_state as mpu
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

from verl.utils.megatron_utils import unwrap_model
from verl.utils.prefix_tree.magi import (
    PrefixTreeMagiBatch,
    build_prefix_tree_micro_batch,
    prefix_tree_decoder_key_context,
    prefix_tree_rope_context,
    restore_flat_to_nested,
    strip_prefix_tree_args,
)

# ---------------------------------------------------------------------------
# Shared helpers (extracted from the forward functions below)
# ---------------------------------------------------------------------------


def _prepare_attn_inputs(
    pb: PrefixTreeMagiBatch,
    prefix_tree_attention: str,
) -> tuple[Tensor, Tensor, dict]:
    """Build local input ids / position ids + attention kwargs for one forward.

    Shared by :func:`unfuse_forward_prefix_tree` and
    :func:`fuse_try_forward_prefix_tree`.  For the ``magi`` branch the returned
    tensors are CP-local slices obtained via :func:`dispatch_magi`; for the
    ``flex`` branch they are the full tree-packed tensors with a leading
    batch dim.  The caller is responsible for wrapping the magi branch in
    :func:`prefix_tree_rope_context` if needed.
    """
    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_magi(pb)
        attn_kwargs = {"magi_attention_key": pb.magi_key}
    else:
        local_input_ids = pb.local_tree_packed_input_ids.unsqueeze(0)
        local_position_ids = pb.local_tree_packed_position_ids.unsqueeze(0)
        attn_kwargs = {"flex_attention_key": pb.flex_key}
    return local_input_ids, local_position_ids, attn_kwargs


def _restore_to_nested_per_sample(
    flat_tensor: Tensor,
    pb: PrefixTreeMagiBatch,
    apply_boundary_patch: bool = False,
) -> Tensor:
    """Restore a flat dedup tensor to per-sample nested (jagged) format.

    Returns a NestedTensor matching non-tree model output: per-sample
    constituents are prefix + ancestors + leaf concatenated, with DP-padding
    tokens excluded. ``postprocess_batch_func`` and ``no_padding_2_padding``
    handle this identically to origin's nested output.

    Args:
        apply_boundary_patch: When True, apply per-leaf boundary log-prob
            patches (see ``_build_sample_tensors`` / ``restore_flat_to_nested``
            in magi.py for the WHY).  Only ``log_probs`` restore passes True;
            ``entropy`` and other tensors pass False.
    """
    return restore_flat_to_nested(flat_tensor, pb, apply_boundary_patch=apply_boundary_patch)


def _expand_temperature(t, pt_batch: PrefixTreeMagiBatch, total_flat: int, device) -> Tensor:
    """Expand a temperature spec to a ``(total_flat, 1)`` per-token tensor.

    Handles three cases:
      * NestedTensor (per-sample): fill every token (prefix, each leaf, and
        each leaf's ancestor chain) with the sample's temperature.
      * Scalar ``Tensor``: broadcast via ``torch.full``.
      * Plain scalar (``float`` / ``int``): same broadcast.

    When ``t`` is None a ones tensor is returned (prior default).
    """
    if t is None:
        return torch.ones(total_flat, 1, dtype=torch.float32, device=device)
    if isinstance(t, torch.Tensor) and t.is_nested:
        # Per-sample temperature: expand to match tree-packed structure.
        # The flat layout contains prefix root + internal ancestor nodes +
        # leaf nodes, so we must fill every token: covering prefix, each
        # leaf, and each leaf's ancestor chain (ancestor_segment_ranges).
        # Missing the internal ancestor tokens shrinks the cat below total_flat.
        temp_by_sample = t.values()  # (batch_size,)
        tree_packed_t = torch.ones(total_flat, 1, dtype=torch.float32, device=device)
        for leaf_idx, sample_idx in enumerate(pt_batch.segment_to_sample):
            t_val = temp_by_sample[sample_idx].item()
            if pt_batch.ancestor_segment_ranges is not None:
                for a, b in pt_batch.ancestor_segment_ranges[leaf_idx]:
                    if b > a:
                        tree_packed_t[a:b] = t_val
            s, e = pt_batch.segment_ranges[leaf_idx]
            if e > s:
                tree_packed_t[s:e] = t_val
        # Shared prefix keeps sample[0]'s temp (prior convention); refill
        # last so ancestor writes from other leaves don't override it.
        prefix_start, prefix_end = pt_batch.prefix_range
        if prefix_end > prefix_start:
            tree_packed_t[prefix_start:prefix_end] = temp_by_sample[0].item()
        return tree_packed_t
    if isinstance(t, torch.Tensor):
        scalar_t = t.flatten()[0].item()
        return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)
    scalar_t = float(t)
    return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Low-level builders
# ---------------------------------------------------------------------------


def _unpack_nested_to_list(x, pad_token_id=None, mask: Optional[Tensor] = None) -> Optional[list[Tensor]]:
    """Unpack a NestedTensor or padded 2-D Tensor into a list of 1-D tensors.

    - NestedTensor (jagged): uses ``.offsets()``
    - Padded 2-D Tensor ``(B, T)``:
      * If ``mask`` is provided: uses ``mask.sum(dim=-1).tolist()`` as
        sequence lengths
      * If ``mask`` is None: returns None (cannot safely unpack)
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
        if mask is not None:
            seqlens = mask.sum(dim=-1).tolist()
            return [x[i, : int(seqlens[i])] for i in range(x.shape[0])]
        return None
    return None


def _build_flex_key(params, device):
    """Build a torch flex_attention block_mask from PrefixTreeParams.

    The mask encodes the prefix-tree attention pattern:
    - Prefix tokens: causal self-attention
    - Leaf tokens: full attention to prefix + causal self-attention within same leaf
    - Cross-leaf attention: blocked (leaf_i cannot see leaf_j)

    Returns a compiled block_mask usable with torch.nn.attention.flex_attention.
    """
    total = params.total_seqlen_q
    prefix_end = params.prefix_range[1]  # == prefix_len

    leaf_id = torch.full((total,), -1, dtype=torch.int32)
    for i, (s, e) in enumerate(params.segment_ranges):
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

    cfg = unwrap_model(model).config
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # Per-rank head counts: ColumnParallelLinear (linear_qkv) shards heads
    # across TP ranks, so each rank's Q/KV tensors hold heads/tp heads.
    # The kernel reads head counts from q.size(1)/k.size(1), but the key's
    # num_heads_q must match for the flatten_head_groups path (enabled via
    # MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=1) which asserts equality.
    num_heads_q = cfg.num_attention_heads // tp_size
    # GQA: num_query_groups may be set; fall back to num_attention_heads (full) if not
    num_heads_kv = (getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads) // tp_size
    head_dim = cfg.kv_channels  # hidden_size // num_attention_heads

    try:
        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = _dist.group.WORLD

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
        dist_attn_config=DistAttnConfig(
            dispatch_config=DispatchConfig(uneven_shard=True),
            overlap_config=OverlapConfig(degree=2, min_chunk_size=512),
        ),
    )


def _finalize_prefix_tree_batch(
    params,
    model,
    num_samples: int,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    subtrie=None,
) -> PrefixTreeMagiBatch:
    """Common downstream step for both detection paths.

    Pads to TP/CP divisibility, builds the requested attention key, and wraps
    the result into a :class:`PrefixTreeMagiBatch`. Padding tokens are not
    added to the attention rectangles; they are stripped before loss, and
    MAGI assigns zero attention weight to out-of-range positions.
    """
    real_tokens = params.tree_packed_tokens.shape[0]
    if tp_size > 1:
        align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
        pad_len = (align_size - real_tokens % align_size) % align_size
        if pad_len > 0:
            params.tree_packed_tokens = torch.cat(
                [params.tree_packed_tokens, params.tree_packed_tokens.new_zeros(pad_len)]
            )
            params.tree_packed_position_ids = torch.cat(
                [params.tree_packed_position_ids, params.tree_packed_position_ids.new_zeros(pad_len)]
            )
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len

    if attention_type == "magi":
        # Cache the MAGI key on the subtrie: OLP and actor_update process the same
        # micro-batch (same sequences, same seqlen) so the key is valid for both passes.
        # TODO(dynamic-cp): if dynamic_context_parallel is enabled, dump this cache.
        if subtrie is not None and getattr(subtrie, "_cached_magi_key", None) is not None:
            magi_key = subtrie._cached_magi_key
        else:
            magi_key = _build_magi_key(model, params)
            if subtrie is not None:
                subtrie._cached_magi_key = magi_key
        flex_key = None
    else:
        flex_key = _build_flex_key(params, params.tree_packed_tokens.device)
        magi_key = None

    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=magi_key,
        flex_key=flex_key,
        segment_to_sample=params.leaf_to_sample,
        segment_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=num_samples,
        real_tokens=real_tokens,
        ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_tree_packed_input_ids=params.tree_packed_tokens,
        local_tree_packed_position_ids=params.tree_packed_position_ids,
        boundary_registry=getattr(params, "boundary_registry", None),
    )


def dispatch_magi(pt_batch: PrefixTreeMagiBatch) -> tuple[Tensor, Tensor]:
    """Slice local_input_ids / local_position_ids from tree-packed tensors via magi dispatch.

    Shared by both fused and unfused paths.  Each CP rank processes only its
    assigned token slice through embedding / FFN / layer norms; cross-rank
    attention is handled by ``calc_attn`` inside the patched attention layer.
    When CP=1, ``local_indices`` covers all tokens.

    Args:
        pt_batch: PrefixTreeMagiBatch with a non-None ``magi_key``.

    Returns:
        (local_input_ids (1, local_tokens), local_position_ids (1, local_tokens)).
    """
    local_indices = get_position_ids(pt_batch.magi_key)
    local_input_ids = pt_batch.tree_packed_input_ids[local_indices].unsqueeze(0)
    local_position_ids = pt_batch.tree_packed_position_ids[local_indices].unsqueeze(0)
    return local_input_ids, local_position_ids


# ---------------------------------------------------------------------------
# Forward-path drivers
# ---------------------------------------------------------------------------


def build_prefix_tree_batch(model, input_ids, logits_processor_args, vision_model, mtp_enable_train):
    """Build prefix-tree micro-batch from *logits_processor_args*.

    Returns :class:`PrefixTreeMagiBatch` or ``None`` when the per-mb subtrie
    is not available.  Caller must gate on use_prefix_tree and skip conditions.
    """
    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")
    loss_mask_nested = (logits_processor_args or {}).get("loss_mask", None)
    position_ids_nested = (logits_processor_args or {}).get("position_ids", None)
    # Per-mb subtrie built once in prepare_prefix_tree_micro_batches (global trie
    # pruned to this mb's samples) and attached to the mb's non-tensor data.
    subtrie = (logits_processor_args or {}).get("prefix_tree_subtree")

    return build_prefix_tree_micro_batch(
        model,
        input_ids,
        loss_mask_nested,
        position_ids=position_ids_nested,
        attention_type=prefix_tree_attention,
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        cp_size=mpu.get_context_parallel_world_size(),
        subtrie=subtrie,
    )


def unfuse_forward_prefix_tree(
    model, pt_batch, prefix_tree_attention, logits_processor, logits_processor_args, post_process, model_kwargs
):
    """Unfused-path: forward pass for prefix-tree batches using magi or flex attention."""
    tree_packed_input_ids = pt_batch.local_tree_packed_input_ids.unsqueeze(0)
    # Use the layout builder's per-sample position IDs (resets within each sample,
    # stays within max_position_embeddings).  torch.arange(flat_tokens) would produce
    # monotonic IDs up to 172437+ which OOB the RoPE embedding table on large batches.
    tree_packed_position_ids = pt_batch.local_tree_packed_position_ids.unsqueeze(0)

    strip_prefix_tree_args(logits_processor_args)

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pt_batch, prefix_tree_attention)
    if prefix_tree_attention == "magi":
        with prefix_tree_rope_context(model, local_position_ids):
            output_orig = model(
                input_ids=local_input_ids,
                attention_mask=None,
                position_ids=local_position_ids,
                packed_seq_params=None,
                **attn_kwargs,
                **model_kwargs,
            )
    else:
        output_orig = model(
            input_ids=tree_packed_input_ids,
            attention_mask=None,
            position_ids=tree_packed_position_ids,
            packed_seq_params=None,
            **attn_kwargs,
            **model_kwargs,
        )

    real_tokens = pt_batch.real_tokens
    if output_orig.shape[0] == 1:
        output_orig = output_orig[:, :real_tokens]
    else:
        output_orig = output_orig[:real_tokens].permute(1, 0, 2)

    if post_process and logits_processor is not None:
        logits_flat = output_orig.squeeze(0)  # (flat_tokens, vocab)
        tree_packed_ids = pt_batch.tree_packed_input_ids[:real_tokens]  # (flat_tokens,)

        # Labels are derived from tree_packed_tokens via within-segment shift; leaf-end positions are 0.
        tree_packed_label = pt_batch.tree_packed_labels[:real_tokens].unsqueeze(1)

        orig_args = logits_processor_args or {}
        total_flat = tree_packed_ids.shape[0]
        tree_packed_t = _expand_temperature(
            orig_args.get("temperature"), pt_batch, total_flat, tree_packed_label.device
        )
        flat_args = {
            k: v for k, v in orig_args.items() if k not in ("label", "temperature", "loss_mask", "use_prefix_tree")
        }

        # For MAGI: logits are CP-local (local_tokens, vocab). Slice label/temp to match.
        # For flex: logits are full flat (real_tokens, vocab). Use as-is.
        if prefix_tree_attention == "magi":
            local_indices = get_position_ids(pt_batch.magi_key)  # (local_tokens,)
            flat_padded = pt_batch.tree_packed_input_ids.shape[0]
            pad = flat_padded - real_tokens

            def _pad_to_full(x):
                return torch.cat([x, x.new_zeros((pad,) + x.shape[1:])]) if pad > 0 else x

            flat_args["label"] = _pad_to_full(tree_packed_label)[local_indices]
            flat_args["temperature"] = _pad_to_full(tree_packed_t)[local_indices]
            n_logits = local_indices.shape[0]
        else:
            flat_args["label"] = tree_packed_label
            flat_args["temperature"] = tree_packed_t
            n_logits = total_flat

        output_dict = logits_processor(logits_flat.clone().unsqueeze(1), **flat_args)

        # Boundary-patch fix: compute per-leaf boundary log-probs from the
        # materialised logits, store on pt_batch for restore to apply.
        # See post_processing_packed_lce for the full WHY.  Runs AFTER
        # logits_processor (unfused equivalent of LCE), BEFORE restore.
        # logits_flat is (flat_tokens, vocab) for flex, (local_tokens, vocab)
        # for magi — post_processing_packed_lce handles the magi mapping.
        post_processing_packed_lce(
            pt_batch,
            logits_flat=logits_flat,
            temperature=flat_args.get("temperature", 1.0),
            magi_key=pt_batch.magi_key if prefix_tree_attention == "magi" else None,
        )

        if isinstance(output_dict, dict):
            for key, val in output_dict.items():
                if isinstance(val, torch.Tensor):
                    val_1d = val.reshape(-1)
                    if val_1d.shape[0] == n_logits:
                        if prefix_tree_attention == "magi":
                            val_1d = undispatch(val_1d, pt_batch.magi_key)[:real_tokens]
                        # log_probs: apply_boundary_patch=True to fix per-leaf
                        #   boundary log-probs.  Other keys (entropy): no patch.
                        is_log_probs = key == "log_probs"
                        output_dict[key] = _restore_to_nested_per_sample(
                            val_1d, pt_batch, apply_boundary_patch=is_log_probs
                        )
        return output_dict
    else:
        # Intermediate PP stage (post_process=False) or no logits_processor.
        # output_orig is (1, flat_tokens, hidden_dim) after normalization above.
        # Stage 0 transposes BSH→SBHD internally (embedding → seq-first).
        # We must send the same seq-first format (seq, 1, hidden) so all downstream
        # stages also get seq-first Q; no per-stage conditional needed, PP=N safe.
        return output_orig.permute(1, 0, 2)  # (1,seq,hid) → (seq,1,hid)


def unfuse_try_forward_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    prefix_tree_attention,
    logits_processor,
    post_process,
    model_kwargs,
    vision_model=False,
    mtp_enable_train=False,
):
    """Unfused-path: try to build + forward a prefix-tree batch; returns output dict or None.

    Consolidates build/forward/strip into one call.  Returns None when no
    prefix sharing is detected, in which case prefix-tree keys are stripped
    from *logits_processor_args* so the caller can fall through to the
    standard THD path.
    """
    # Unfused path blocks VLM unconditionally (3D M-RoPE not wired here).
    # Fused path (fuse_try_forward_prefix_tree) only blocks VLM-with-images
    # so text-only VLM batches proceed; see vision_model+has_vision_data guard there.
    if vision_model or mtp_enable_train:
        _log.getLogger(__name__).warning(
            "prefix_tree: skipping prefix-tree path (vision_model=%s, mtp_enable_train=%s); "
            "falling back to standard THD",
            vision_model,
            mtp_enable_train,
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    pb = build_prefix_tree_batch(
        model,
        input_ids,
        logits_processor_args,
        vision_model,
        mtp_enable_train,
    )
    if pb is not None:
        return unfuse_forward_prefix_tree(
            model,
            pb,
            prefix_tree_attention,
            logits_processor,
            logits_processor_args,
            post_process,
            model_kwargs,
        )

    _log.getLogger(__name__).warning(
        "prefix_tree: build_prefix_tree_batch returned None; falling back to standard THD path "
        "(post_process=%s). If this appears for one PP stage but not the other, the hidden-state "
        "format will mismatch between stages.",
        post_process,
    )
    strip_prefix_tree_args(logits_processor_args)
    return None


# ---------------------------------------------------------------------------
# Fused-path
# ---------------------------------------------------------------------------


def _run_lce(
    hidden_states: Tensor,
    output_weight: Tensor,
    labels: Tensor,
    temperature: float,
    model,
    magi_key=None,
    pt_batch: Optional[PrefixTreeMagiBatch] = None,
) -> tuple[Tensor, Tensor]:
    """Fused LCE for both MAGI and flex branches.

    Shared part: gather from sequence-parallel region (if enabled), then call
    :func:`linear_cross_entropy`.  When ``magi_key`` is given (MAGI branch),
    labels are padded to flat-padded length and sliced by CP-local indices
    before the call, and the 1D outputs are undispatched back to full flat
    order and trimmed to ``real_tokens``.  Without ``magi_key`` (flex branch),
    labels pass through directly and no undispatch is needed.
    """
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy as _lce

    if model.config.sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    if magi_key is not None:
        local_indices = get_position_ids(magi_key)
        flat_padded = pt_batch.tree_packed_input_ids.shape[0]
        pad = flat_padded - labels.shape[0]
        labels_full = torch.cat([labels, labels.new_zeros(pad)]) if pad > 0 else labels
        lce_labels = labels_full[local_indices]
    else:
        lce_labels = labels

    logprobs, entropy = _lce(
        hidden_states,
        output_weight,
        lce_labels,
        temperature,
        "none",
        mpu.get_tensor_model_parallel_group(),
    )

    if magi_key is not None:
        logprobs = undispatch(logprobs.reshape(-1), magi_key)[: pt_batch.real_tokens]
        entropy = undispatch(entropy.reshape(-1), magi_key)[: pt_batch.real_tokens]
    return logprobs, entropy


def post_processing_packed_lce(
    pt_batch: PrefixTreeMagiBatch,
    hidden_flat: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    logits_flat: Optional[Tensor] = None,
    temperature=1.0,
    magi_key=None,
) -> None:
    """Compute per-leaf boundary log-probs and store on ``pt_batch._boundary_logps``.

    WHY THIS EXISTS — the dedup → one-slot → LCE-1:1 → copy-to-all-leaves bug:
    -------------------------------------------------------------------------

    In the deduplicated flat layout, a shared ancestor's last token (the
    *boundary predictor position*) occupies ONE flat slot.  LCE is strictly
    1:1 (one label per slot → one log-prob per slot), and the label there is
    the OWNER's next-token id (from ``rolled_samples[owner]``).  So the flat
    logp at the boundary is ``log p(owner's next token | shared hidden)``.
    ``restore_flat_to_nested`` then copies that single scalar into every leaf
    sharing the ancestor — non-owner leaves receive the OWNER's log-prob
    instead of their own (a ~19-nat error at every shared-segment junction).

    This function runs AFTER LCE / logits_processor and BEFORE
    ``restore_flat_to_nested``.  For each boundary (identified by its flat
    position in ``pt_batch.boundary_registry``, built by
    ``prepare_packed_label`` in utils.py):

      1. Materialise ONE vocab-row of logits at the boundary:
         - Fused path: ``logits_b = weight @ hidden_flat[b_pos]``
           (one matvec — vs donation's N duplicated hidden slots which made
           LCE recompute ``W·hidden[b]`` N times; N× fewer boundary matvecs).
         - Unfused path: ``logits_b = logits_flat[b_pos]`` (already materialised).
      2. ``logp_all = log_softmax(logits_b / temperature)`` — the full
         next-token distribution at the shared hidden state.
      3. For each leaf: ``leaf_logp = logp_all[leaf_next_token]`` — the leaf's
         OWN next-token log-prob.

    The results are stored as ``pt_batch._boundary_logps``:
    ``dict[sample_idx -> [(boundary_flat_pos, leaf_logp), ...]]``.
    ``restore_flat_to_nested`` (via ``_build_sample_tensors``) reads this and
    patches each sample's boundary position during the cat — split-and-cat,
    autograd-safe, no layout surgery.

    MAGI / CP handling:
    -------------------
    When ``magi_key`` is set, ``hidden_flat`` / ``logits_flat`` are in
    CP-local (magi-dispatched) order, NOT flat order.  ``b_pos`` is a flat
    position.  We map flat → local via ``local_indices = get_position_ids(magi_key)``
    (``local_indices[i] = flat_pos``), then index by the matching local index.
    For CP>1, a boundary not on this rank is skipped (no patch — the leaf's
    sample is on a different rank).  For CP=1, every boundary is local.

    Only ``log_probs`` should be patched (entropy at the boundary is
    distribution-level — same for all leaves — so no per-leaf patch needed).
    The caller passes ``apply_boundary_patch=True`` only for the ``log_probs``
    restore.

    Args:
        pt_batch: The prefix-tree micro-batch (carries ``boundary_registry``).
        hidden_flat: ``(tokens, hidden_dim)`` decoder output — fused path.
        weight: ``(vocab, hidden_dim)`` vocab-projection weight — fused path.
        logits_flat: ``(tokens, vocab)`` materialised logits — unfused path.
        temperature: Scalar float or ``(tokens,)`` tensor.  The fused path
            always uses scalar (LCE asserts this); the unfused path may use
            per-token temperature.
        magi_key: When set, ``hidden_flat`` / ``logits_flat`` are CP-local.
    """
    registry = getattr(pt_batch, "boundary_registry", None)
    if not registry:
        return

    # Normalise hidden_flat / logits_flat to 2-D (tokens, dim).  The decoder may
    # return (1, tokens, H), (tokens, 1, H), (tokens, H), etc. depending on the
    # data format (thd vs bshd) and PP/CP layout — squeeze ALL size-1 dims except
    # the last two, so we robustly get (tokens, dim).
    if hidden_flat is not None and hidden_flat.dim() != 2:
        hidden_flat = hidden_flat.reshape(-1, hidden_flat.shape[-1])
    if logits_flat is not None and logits_flat.dim() != 2:
        logits_flat = logits_flat.reshape(-1, logits_flat.shape[-1])

    # For MAGI: build flat_pos → local_idx lookup.
    local_indices = None
    if magi_key is not None:
        local_indices = get_position_ids(magi_key)

    boundary_logps: dict[int, list[tuple[int, Tensor]]] = {}

    # ---- Compute the per-leaf patch value for each registry entry whose
    # boundary hidden is on THIS CP rank, emitting a TAGGED triple
    # (b_pos, sample_idx, logp).  The tag is the safety check: every rank has
    # the identical global registry (built once on the driver, replicated), so
    # after the cross-CP all_gather each (b_pos, sample_idx) must arrive from
    # exactly ONE owner rank — we assert that, catching any registry/ownership
    # divergence loudly instead of silently producing wrong patches.
    #
    # WHY a comm is needed: the patch for leaf c at boundary a is
    # ``log p(c_token | hidden[a])``.  ``hidden[a]`` lives on the CP rank that
    # holds flat position a; leaf c is restored (post-undispatch, global) on
    # every rank.  So the patch is computed on a's rank but needed on c's
    # restore rank -> the value must cross CP.  We move ONLY the tagged scalars
    # (3 ints/float per owned entry) — far cheaper than gathering hidden.
    #
    # sample_idx (the leaf i) is sent explicitly even though every rank already
    # knows it from the registry: it lets the receiver ASSERT that the arriving
    # (b_pos, sample_idx) matches its own registry, confirming the registries
    # really are identical across ranks (no silent misrouting).
    device = (
        hidden_flat.device
        if hidden_flat is not None
        else (logits_flat.device if logits_flat is not None else torch.device("cpu"))
    )
    _loc_bpos: list[int] = []
    _loc_sid: list[int] = []
    _loc_logp: list[Tensor] = []

    for b_pos, leaves in registry:
        # Resolve the local index for this flat boundary position.
        if local_indices is not None:
            matches = (local_indices == b_pos).nonzero()
            if matches.shape[0] == 0:
                # Boundary hidden is on another CP rank — that rank emits this
                # entry's triple; we receive it via the all_gather below.
                continue
            local_idx = int(matches[0, 0].item())
        else:
            local_idx = b_pos

        # Materialise one vocab-row of logits at the boundary.
        # weight is the output_layer weight (vocab, hidden); use F.linear (= h @ weight.T)
        # for correct orientation, robust to h being (hidden,) or (1, hidden).
        if logits_flat is not None:
            logits_b = logits_flat[local_idx]
        elif hidden_flat is not None and weight is not None:
            h_b = hidden_flat[local_idx]  # (hidden,) or (1, hidden)
            if h_b.dim() == 1:
                h_b = h_b.unsqueeze(0)  # (1, hidden)
            logits_b = torch.nn.functional.linear(h_b, weight).squeeze(0)  # (vocab,)
        else:
            # Nothing to compute from — skip patching (no logits available).
            return

        # Temperature at the boundary: scalar or per-token.
        if torch.is_tensor(temperature):
            temp_b = temperature[local_idx]
        else:
            temp_b = temperature

        # Full next-token distribution at the shared hidden state.
        logp_all = torch.log_softmax(logits_b / temp_b, dim=-1)

        # Emit a tagged triple per leaf sharing this boundary.
        for sample_idx, next_token in leaves:
            _loc_bpos.append(b_pos)
            _loc_sid.append(sample_idx)
            _loc_logp.append(logp_all[next_token])

    # ---- Cross-CP all_gather of the tagged triples so every rank has the FULL
    # patch set (restore runs globally post-undispatch).  Fixed-size padded
    # tensors + per-rank counts; the b_pos/sample_idx tags route + assert.
    cp_world = mpu.get_context_parallel_world_size()
    if cp_world > 1 and magi_key is not None:
        cp_group = mpu.get_context_parallel_group()
        n_loc = len(_loc_logp)
        counts = torch.tensor([n_loc], dtype=torch.long, device=device)
        counts_all = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(cp_world)]
        _dist.all_gather(counts_all, counts, group=cp_group)
        counts_list = [int(c.item()) for c in counts_all]
        max_n = max(counts_list) if counts_list else 0
        local_bpos_t = torch.zeros(max_n, dtype=torch.long, device=device)
        local_sid_t = torch.zeros(max_n, dtype=torch.long, device=device)
        local_logp_t = torch.zeros(max_n, dtype=torch.float32, device=device)
        if n_loc > 0:
            local_bpos_t[:n_loc] = torch.tensor(_loc_bpos, dtype=torch.long, device=device)
            local_sid_t[:n_loc] = torch.tensor(_loc_sid, dtype=torch.long, device=device)
            local_logp_t[:n_loc] = torch.stack(_loc_logp).to(torch.float32)
        bpos_all = [torch.zeros_like(local_bpos_t) for _ in range(cp_world)]
        sid_all = [torch.zeros_like(local_sid_t) for _ in range(cp_world)]
        logp_all = [torch.zeros_like(local_logp_t) for _ in range(cp_world)]
        _dist.all_gather(bpos_all, local_bpos_t, group=cp_group)
        _dist.all_gather(sid_all, local_sid_t, group=cp_group)
        _dist.all_gather(logp_all, local_logp_t, group=cp_group)

        # SAFETY CHECK: each (b_pos, sample_idx) must arrive from exactly one
        # rank (the owner).  A duplicate or a tag not in our registry means the
        # registries diverged across ranks — fail loudly, do NOT silently patch.
        reg_keys = {(bp, sid) for bp, leaves in registry for sid, _ in leaves}
        seen: set[tuple[int, int]] = set()
        for r in range(cp_world):
            for i in range(counts_list[r]):
                bp = int(bpos_all[r][i].item())
                sid = int(sid_all[r][i].item())
                key = (bp, sid)
                if key not in reg_keys:
                    raise AssertionError(
                        f"post_processing_packed_lce: received boundary patch (b_pos={bp}, "
                        f"sample_idx={sid}) from rank {r} that is NOT in this rank's registry "
                        f"— registries diverged across CP ranks. Aborting to avoid silent "
                        f"wrong patches."
                    )
                if key in seen:
                    raise AssertionError(
                        f"post_processing_packed_lce: boundary patch (b_pos={bp}, "
                        f"sample_idx={sid}) arrived from MULTIPLE ranks — ownership collision. "
                        f"Aborting to avoid silent wrong patches."
                    )
                seen.add(key)
                boundary_logps.setdefault(sid, []).append((bp, logp_all[r][i]))
        # Every registry entry must have been filled by exactly one owner.
        if seen != reg_keys:
            missing = reg_keys - seen
            raise AssertionError(
                f"post_processing_packed_lce: {len(missing)} registry entries received NO patch "
                f"from any CP rank (e.g. {next(iter(missing))}) — a boundary's hidden was on no "
                f"rank. Aborting to avoid silent wrong patches."
            )
    else:
        # CP=1 (or non-magi): local triples ARE the full set, no comm/assert.
        for bp, sid, lp in zip(_loc_bpos, _loc_sid, _loc_logp, strict=False):
            boundary_logps.setdefault(sid, []).append((bp, lp))

    pt_batch._boundary_logps = boundary_logps


def fused_prefix_tree_forward(
    model,
    *,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Optional[Tensor],
    temperature: float,
    pt_batch,
    decoder_input: Optional[Tensor],
    packed_seq_params,
    extra_block_kwargs: Optional[dict],
    inference_context,
    kwargs: dict,
):
    """Fused-path prefix-tree forward used by the patched ``_fused_GPTModel_forward``.

    Pops ``magi_attention_key`` / ``flex_attention_key`` from ``kwargs`` and installs
    rope + decoder-key contexts before delegating to :func:`fuse_forward_body`.
    Returns ``None`` when both attention keys are absent so the caller falls back
    to the standard fused path.
    """
    _magi_key = kwargs.pop("magi_attention_key", None)
    _flex_key = kwargs.pop("flex_attention_key", None)
    if _magi_key is None and _flex_key is None:
        return None

    with (
        prefix_tree_rope_context(model, position_ids),
        prefix_tree_decoder_key_context(model, _magi_key, _flex_key),
    ):
        return fuse_forward_body(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            temperature=temperature,
            pt_batch=pt_batch,
            magi_key=_magi_key,
            flex_key=_flex_key,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )


def fuse_forward_body(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Optional[Tensor],
    labels: Tensor,
    temperature: float,
    pt_batch: PrefixTreeMagiBatch,
    magi_key=None,
    flex_key=None,
    **kwargs,
):
    """Fused-path forward body for prefix-tree: preprocess → decoder → LCE.

    Shared entry point invoked by the unified ``_gpt_forward`` patch when the
    fused prefix-tree path is selected (``pt_batch`` present + attention key).
    Mirrors ``_fused_GPTModel_forward`` but assumes rope override and decoder
    key injection are already active (installed by the caller via
    :func:`prefix_tree_rope_context` and :func:`prefix_tree_decoder_key_context`).

    Vocab projection stays fused via :func:`linear_cross_entropy`: no
    ``(flat_tokens, vocab)`` logits tensor is materialised.
    """
    from collections import OrderedDict as _OrderedDict

    from megatron.core.config_logger import has_config_logger_enabled as _has_cfg_log
    from megatron.core.config_logger import log_config_to_disk as _log_cfg
    from megatron.core.utils import deprecate_inference_params as _dep_inf

    from verl.utils.model import CausalLMOutputForPPO as _CLMOutput

    inference_context = kwargs.pop("inference_context", None)
    inference_params = kwargs.pop("inference_params", None)
    inference_context = _dep_inf(inference_context, inference_params)
    decoder_input = kwargs.pop("decoder_input", None)
    packed_seq_params = kwargs.pop("packed_seq_params", None)
    extra_block_kwargs = kwargs.pop("extra_block_kwargs", None)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )
    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
        **kwargs,
    )

    if not model.post_process:
        return hidden_states

    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        output_weight = model.embedding.word_embeddings.weight

    if magi_key is not None:
        logprobs, entropy = _run_lce(
            hidden_states, output_weight, labels, temperature, model, magi_key=magi_key, pt_batch=pt_batch
        )
    else:
        logprobs, entropy = _run_lce(hidden_states, output_weight, labels, temperature, model)

    # Boundary-patch fix: compute per-leaf boundary log-probs from the shared
    # hidden state, store on pt_batch for restore_flat_to_nested to apply.
    # See post_processing_packed_lce for the full WHY (dedup → one-slot →
    # LCE:1:1 → copy-to-all-leaves bug).  Runs AFTER LCE, BEFORE restore.
    post_processing_packed_lce(
        pt_batch,
        hidden_flat=hidden_states,
        weight=output_weight,
        temperature=temperature,
        magi_key=magi_key,
    )

    if _has_cfg_log(model.config):
        payload = _OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        _log_cfg(model.config, payload, prefix="input_and_logits")

    output = _CLMOutput(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )
    output.entropy = entropy
    output.log_probs = logprobs
    return output


def fuse_try_forward_prefix_tree(
    model,
    input_ids,
    labels,
    temperature: float,
    logits_processor_args: dict,
    calculate_entropy: bool,
    *,
    vision_model: bool = False,
    has_vision_data: bool = False,
):
    """Fused-path: try to build + forward a prefix-tree batch with fused vocab projection.

    Counterpart of :func:`unfuse_try_forward_prefix_tree` for the
    ``use_fused_kernels=True`` path.  The vocab projection + log-prob
    computation stays fused inside ``_fused_GPTModel_forward`` via
    :func:`linear_cross_entropy`: the unfused path materialises
    ``(flat_tokens, vocab)`` logits and runs ``logits_processor`` outside the
    model, but the fused path never materialises the full vocab tensor.

    Limitations vs unfused path:
      - **Scalar temperature only.**  ``linear_cross_entropy`` asserts
        ``isinstance(temperature, float)``.  Per-sample temperature must use
        the unfused path.
      - **PP support**: on non-last stages (``not post_process``), returns the
        raw hidden-state tensor (pipeline schedule sends it to the next stage).
        Last stage (``post_process=True``) returns the log_probs/entropy dict.

    Args:
        model: Megatron GPTModel (forward patched to ``_fused_GPTModel_forward``).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        labels: NestedTensor, used for per-sample offsets only; actual labels
            come from ``pt_batch.tree_packed_labels`` (pre-shifted per sample).
        temperature: scalar float.
        logits_processor_args: dict containing ``use_prefix_tree``,
            ``prefix_tree_attention``, ``segment_hashes``, ``segment_lengths``,
            ``prefix_tree_subtree``.  Prefix-tree keys are stripped on return.
        calculate_entropy: whether to return ``entropy`` alongside ``log_probs``.
        vision_model: whether the model is a VLM-config model (has vision_config).
        has_vision_data: whether ``pixel_values`` is present in multi_modal_inputs.

    Returns:
        ``{"log_probs": NestedTensor, "entropy": NestedTensor}`` (entropy only
        when ``calculate_entropy=True``), or ``None`` when no prefix sharing is
        detected; caller falls through to the standard fused path.
    """

    # VLM-with-images: 3D M-RoPE position handling not yet wired
    # (prefix_tree_rope_context assumes 1D). Text-only on ViT-config models
    # passes through to the standard fused path below.
    if vision_model and has_vision_data:
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(
        model,
        input_ids,
        logits_processor_args,
        vision_model=False,
        mtp_enable_train=False,
    )
    if pb is None:
        _log.getLogger(__name__).warning(
            "prefix_tree: build_prefix_tree_batch returned None; falling back to standard fused path"
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)

    strip_prefix_tree_args(logits_processor_args)

    post_process = unwrap_model(model).post_process

    # Only the last PP stage (post_process=True) needs labels for LCE.
    # Non-last stages pass labels=None; fuse_forward_body returns before LCE.
    real_tokens = pb.real_tokens
    if post_process:
        if pb.tree_packed_labels is None:
            _log.getLogger(__name__).warning(
                "prefix_tree[fused]: tree_packed_labels is None; falling back to standard fused path"
            )
            return None
        # Pass flat (deduped) labels; LCE runs on real_tokens, not total_expanded.
        labels_arg = pb.tree_packed_labels[:real_tokens]
    else:
        labels_arg = None

    output_orig = model(
        input_ids=local_input_ids,
        attention_mask=None,
        position_ids=local_position_ids,
        packed_seq_params=None,
        labels=labels_arg,
        temperature=temperature,
        pt_batch=pb,
        **attn_kwargs,
    )

    if not post_process:
        return output_orig

    # output_orig.log_probs / .entropy are (real_tokens,) flat; restore to per-sample nested.
    # log_probs: apply_boundary_patch=True to fix per-leaf boundary log-probs
    #   (pt_batch._boundary_logps was set by post_processing_packed_lce above).
    # entropy: no patch (entropy at the boundary is distribution-level, same
    #   for all leaves sharing the hidden state).
    output = {
        "log_probs": _restore_to_nested_per_sample(output_orig.log_probs.reshape(-1), pb, apply_boundary_patch=True)
    }
    if calculate_entropy:
        output["entropy"] = _restore_to_nested_per_sample(output_orig.entropy.reshape(-1), pb)
    return output
