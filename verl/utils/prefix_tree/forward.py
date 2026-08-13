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
"""Prefix-tree forward-path: tree build, MAGI dispatch, rope override,
    fused/unfused forward drivers, and LCE post-processing.

Public: prepare_prefix_tree, tree_post_processing, fuse_forward_body, dispatch_magi."""

from __future__ import annotations

import logging as _log
from collections import Counter, OrderedDict, namedtuple
from typing import Optional

import torch
import torch.distributed as _dist
from magi_attention.api import (
    DistAttnConfig,
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
    PackRestorationParam,
    PrefixTreeMagiBatch,
    _clear_rope_pids,
    _set_rope_pids,
    build_prefix_tree_micro_batch,
    prefix_tree_decoder_key_context,
    restore_flat_to_nested,
    strip_prefix_tree_args,
)

_logger = _log.getLogger(__name__)

TreeForwardCtx = namedtuple("TreeForwardCtx", ["pb", "input_ids", "position_ids", "attention", "model"])
"""Returned by :func:`prepare_prefix_tree`.  ``rope_exit`` is a
callable to deactivate the MAGI rope context (``None`` for flex / non-tree)."""

# Shared helpers


def _prepare_attn_inputs(
    pb: PrefixTreeMagiBatch,
    prefix_tree_attention: str,
) -> tuple[Tensor, Tensor, dict]:
    """Build local_input_ids, local_position_ids, and attention kwargs.

    MAGI path returns CP-local slices; flex returns full tree-packed."""
    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_magi(pb)
        attn_kwargs = {"magi_attention_key": pb.magi_key}
    else:
        local_input_ids = pb.tree_packed_input_ids.unsqueeze(0)
        local_position_ids = pb.tree_packed_position_ids.unsqueeze(0)
        attn_kwargs = {"flex_attention_key": pb.flex_key}
    return local_input_ids, local_position_ids, attn_kwargs


# ============================================================


def _expand_temperature(t, pt_batch: PrefixTreeMagiBatch, total_flat: int, device) -> Tensor:
    """Expand temperature to (total_flat, 1) tensor: handles NestedTensor (per-sample), scalar Tensor, and float."""
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
        for leaf_idx, sample_idx in enumerate(pt_batch.subtrie.leaf_to_sample):
            t_val = temp_by_sample[sample_idx].item()
            if pt_batch.restoration.ancestor_segment_ranges is not None:
                for a, b in pt_batch.restoration.ancestor_segment_ranges[leaf_idx]:
                    if b > a:
                        tree_packed_t[a:b] = t_val
            s, e = pt_batch.restoration.segment_ranges[leaf_idx]
            if e > s:
                tree_packed_t[s:e] = t_val
        # Shared prefix keeps sample[0]'s temp (prior convention); refill
        # last so ancestor writes from other leaves don't override it.
        prefix_start, prefix_end = pt_batch.restoration.prefix_range
        if prefix_end > prefix_start:
            tree_packed_t[prefix_start:prefix_end] = temp_by_sample[0].item()
        return tree_packed_t
    if isinstance(t, torch.Tensor):
        scalar_t = t.flatten()[0].item()
        return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)
    scalar_t = float(t)
    return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)


# Low-level builders


def _unpack_nested_to_list(x, mask: Optional[Tensor] = None) -> Optional[list[Tensor]]:
    """Unpack NestedTensor or padded 2-D Tensor into list of 1-D tensors. Returns None if cannot safely unpack."""
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
    if x.dim() == 2 and mask is not None:
        seqlens = mask.sum(dim=-1).tolist()
        return [x[i, : int(seqlens[i])] for i in range(x.shape[0])]
    return None


def _build_flex_key(params, device):
    """Build torch flex_attention block_mask from PrefixTreeParams
    (prefix causal + per-leaf causal + cross-leaf blocked)."""
    total = params.total_seqlen_q
    prefix_end = params.prefix_range[1]  # == prefix_len

    leaf_id = torch.full((total,), -1, dtype=torch.int32, device=device)
    for i, (s, e) in enumerate(params.leaf_ranges):
        leaf_id[s:e] = i

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
    """Build magi_attn_flex_key from PrefixTreeParams and model config."""
    # TP shards heads: each rank holds heads/tp_size. GQA falls back to num_attention_heads.
    cfg = unwrap_model(model).config
    tp_size = mpu.get_tensor_model_parallel_world_size()
    num_heads_q = cfg.num_attention_heads // tp_size
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
    """Pad to TP/CP divisibility, build attention key, and wrap into PrefixTreeMagiBatch."""
    real_tokens = params.tree_packed_tokens.shape[0]
    align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
    if align_size > 1:
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
    elif attention_type == "flex":
        flex_key = _build_flex_key(params, params.tree_packed_tokens.device)
        magi_key = None
    else:
        raise ValueError(f"Unknown attention_type: {attention_type!r} (expected 'magi' or 'flex')")

    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=magi_key,
        flex_key=flex_key,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
            boundary_registry=getattr(params, "boundary_registry", None),
        ),
        subtrie=subtrie,
        real_tokens=real_tokens,
    )


def dispatch_magi(pt_batch: PrefixTreeMagiBatch) -> tuple[Tensor, Tensor]:
    """Slice local_input_ids / local_position_ids via magi dispatch
    (get_position_ids). Each CP rank processes its assigned slice."""
    local_indices = get_position_ids(pt_batch.magi_key)
    local_input_ids = pt_batch.tree_packed_input_ids[local_indices].unsqueeze(0)
    local_position_ids = pt_batch.tree_packed_position_ids[local_indices].unsqueeze(0)
    return local_input_ids, local_position_ids


# Forward-path drivers


def build_prefix_tree_batch(model, input_ids, logits_processor_args):
    """Build prefix-tree micro-batch from logits_processor_args. Returns PrefixTreeMagiBatch or None."""
    args = logits_processor_args or {}
    prefix_tree_attention = args.get("prefix_tree_attention", "flex")
    loss_mask_nested = args.get("loss_mask")
    position_ids_nested = args.get("position_ids")
    # Per-mb subtrie from prepare_prefix_tree_micro_batches (global trie pruned to mb).
    subtrie = args.get("prefix_tree_subtree")

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


def prepare_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    model_kwargs,
    *,
    vision_model=False,
    mtp_enable_train=False,
):
    """Prepare prefix-tree forward context.

    Returns a :class:`TreeForwardCtx` or ``None`` when tree is not applicable.
    On success, merges attention kwargs into *model_kwargs* in-place and (for
    MAGI) activates the rope context via ``ctx.rope_exit``.  The caller must
    deactivate rope via ``ctx.rope_exit(None, None, None)`` after
    post-processing (or on the intermediate-PP path).
    """
    if vision_model or mtp_enable_train:
        _logger.warning(
            "prefix_tree: skipping prefix-tree path (vision_model=%s, mtp_enable_train=%s), not fully supported yet",
            vision_model,
            mtp_enable_train,
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(model, input_ids, logits_processor_args)
    if pb is None:
        _logger.warning("prefix_tree: build_prefix_tree_batch returned None; falling back to standard THD path")
        strip_prefix_tree_args(logits_processor_args)
        return None

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)
    strip_prefix_tree_args(logits_processor_args)
    model_kwargs.update(attn_kwargs)

    _set_rope_pids(model, local_position_ids)

    return TreeForwardCtx(pb, local_input_ids, local_position_ids, prefix_tree_attention, model)


def tree_post_processing(ctx, output_orig, logits_processor, logits_processor_args, post_process):
    if ctx is None:
        return output_orig
    pt_batch = ctx.pb
    real_tokens = pt_batch.real_tokens
    if output_orig.shape[0] == 1:
        output_orig = output_orig[:, :real_tokens]
    else:
        output_orig = output_orig[:real_tokens].permute(1, 0, 2)

    if not post_process or logits_processor is None:
        return output_orig.permute(1, 0, 2)

    prefix_tree_attention = ctx.attention
    try:
        logits_flat = output_orig.squeeze(0)
        tree_packed_ids = pt_batch.tree_packed_input_ids[:real_tokens]
        tree_packed_label = pt_batch.tree_packed_labels[:real_tokens].unsqueeze(1)

        orig_args = logits_processor_args or {}
        total_flat = tree_packed_ids.shape[0]
        tree_packed_t = _expand_temperature(
            orig_args.get("temperature"), pt_batch, total_flat, tree_packed_label.device
        )
        flat_args = {
            k: v for k, v in orig_args.items() if k not in ("label", "temperature", "loss_mask", "use_prefix_tree")
        }

        if prefix_tree_attention == "magi":
            local_indices = get_position_ids(pt_batch.magi_key)
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

        post_processing_packed_lce(
            pt_batch,
            logits_flat=logits_flat,
            temperature=flat_args.get("temperature", 1.0),
            magi_key=pt_batch.magi_key if prefix_tree_attention == "magi" else None,
        )

        output = {}
        for key, val in output_dict.items():
            if isinstance(val, torch.Tensor):
                val_1d = val.reshape(-1)
                if val_1d.shape[0] == n_logits:
                    if prefix_tree_attention == "magi":
                        val_1d = undispatch(val_1d, pt_batch.magi_key)[:real_tokens]
                    output[key] = restore_flat_to_nested(val_1d, pt_batch, apply_boundary_patch=(key == "log_probs"))
        return output
    finally:
        _clear_rope_pids(ctx.model)


# Fused-path


def _prepare_lce_inputs_with_boundary(
    hidden_states: Tensor,
    labels: Tensor,
    model,
    magi_key,
    pt_batch: Optional[PrefixTreeMagiBatch],
):
    """Preprocess LCE inputs: SP gather, label pad+dispatch, boundary-pair resolution.

    Returns (hidden_ext, labels_ext, n_local, boundary_tags) where boundary_tags
    are [(boundary_position, sample_idx)] aligned with the appended tail rows.
    """
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
        local_indices = None

    hidden_2d = hidden_states.view(-1, hidden_states.shape[-1])
    registry = pt_batch.restoration.boundary_registry if (pt_batch is not None and pt_batch.restoration) else None

    # Boundary extension: append the same hidden token once per leaf label so a
    # single LCE pass covers both main and boundary log-probs (no second LCE).
    boundary_pairs: list[tuple[int, int]] = []
    boundary_tags: list[tuple[int, int]] = []
    if magi_key is not None and registry:
        for boundary_position, leaves in registry:
            matches = (local_indices == boundary_position).nonzero()
            if matches.shape[0] == 0:
                continue  # boundary hidden lives on another CP rank
            local_idx = int(matches[0, 0].item())
            for sample_idx, next_token in leaves:
                boundary_pairs.append((local_idx, int(next_token)))
                boundary_tags.append((boundary_position, sample_idx))

    n_local = hidden_2d.shape[0]
    if boundary_pairs:
        idx, lbl = zip(*boundary_pairs, strict=False)
        idx_t = torch.tensor(idx, device=hidden_2d.device)
        hidden_ext = torch.cat([hidden_2d, hidden_2d[idx_t]], dim=0)
        labels_ext = torch.cat(
            [lce_labels.reshape(-1), torch.tensor(lbl, device=lce_labels.device, dtype=lce_labels.dtype)]
        )
    else:
        hidden_ext, labels_ext = hidden_2d, lce_labels

    return hidden_ext, labels_ext, n_local, boundary_tags


def _run_lce_postprocess(
    logprobs_ext: Tensor,
    entropy_ext: Tensor,
    n_local: int,
    boundary_tags: list[tuple[int, int]],
    magi_key,
    pt_batch: Optional[PrefixTreeMagiBatch],
) -> tuple[Tensor, Tensor]:
    """Split boundary tail, stash _boundary_local_vals, undispatch + trim main output."""
    if boundary_tags:
        logprobs = logprobs_ext[:n_local]
        entropy = entropy_ext[:n_local]
        pt_batch._boundary_local_vals = [
            (pos, sid, logprobs_ext[n_local + i]) for i, (pos, sid) in enumerate(boundary_tags)
        ]
    else:
        logprobs, entropy = logprobs_ext, entropy_ext

    if magi_key is not None:
        logprobs = undispatch(logprobs.reshape(-1), magi_key)[: pt_batch.real_tokens]
        entropy = undispatch(entropy.reshape(-1), magi_key)[: pt_batch.real_tokens]
    return logprobs, entropy


def _run_lce(
    hidden_states: Tensor,
    output_weight: Tensor,
    labels: Tensor,
    temperature: float,
    model,
    magi_key=None,
    pt_batch: Optional[PrefixTreeMagiBatch] = None,
) -> tuple[Tensor, Tensor]:
    """Fused LCE for MAGI/flex: prepare → linear_cross_entropy → postprocess."""
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    hidden_ext, labels_ext, n_local, boundary_tags = _prepare_lce_inputs_with_boundary(
        hidden_states, labels, model, magi_key, pt_batch
    )

    logprobs_ext, entropy_ext = linear_cross_entropy(
        hidden_ext,
        output_weight,
        labels_ext,
        temperature,
        "none",
        mpu.get_tensor_model_parallel_group(),
    )

    return _run_lce_postprocess(logprobs_ext, entropy_ext, n_local, boundary_tags, magi_key, pt_batch)


def post_processing_packed_lce(
    pt_batch: PrefixTreeMagiBatch,
    hidden_flat: Optional[Tensor] = None,
    model=None,
    logits_flat: Optional[Tensor] = None,
    temperature=1.0,
    magi_key=None,
    output_weight: Optional[Tensor] = None,
) -> None:
    """Compute per-leaf boundary log-probs and store on pt_batch._boundary_logps for restore_flat_to_nested.

    When output_weight given (fused): uses batched LCE for bit-identical values. Otherwise: per-boundary log_softmax.
    Cross-CP all_gather of tagged triples ensures every rank has full patch set.
    """
    registry = pt_batch.restoration.boundary_registry if pt_batch.restoration else None
    if not registry:
        return

    # SP gather for full token dimension (boundary positions valid).
    if hidden_flat is not None and model is not None and getattr(model.config, "sequence_parallel", False):
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        hidden_flat = gather_from_sequence_parallel_region(hidden_flat)

    # Normalise hidden_flat / logits_flat to 2-D (tokens, dim).
    if hidden_flat is not None and hidden_flat.dim() != 2:
        hidden_flat = hidden_flat.reshape(-1, hidden_flat.shape[-1])
    if logits_flat is not None and logits_flat.dim() != 2:
        logits_flat = logits_flat.reshape(-1, logits_flat.shape[-1])

    # For MAGI: build flat_pos → local_idx lookup.
    local_indices = None
    if magi_key is not None:
        local_indices = get_position_ids(magi_key)

    boundary_logps: dict[int, list[tuple[int, Tensor]]] = {}

    # Each CP rank computes boundary patch for its owned positions, then cross-CP all_gather of tagged triples.
    device = (
        hidden_flat.device
        if hidden_flat is not None
        else (logits_flat.device if logits_flat is not None else torch.device("cpu"))
    )
    local_boundary_positions: list[int] = []
    local_sample_indices: list[int] = []
    local_log_probs: list[Tensor] = []

    precomputed = getattr(pt_batch, "_boundary_local_vals", None)
    if precomputed is not None:
        # Fused path: _run_lce already computed boundary log-probs in its single
        # LCE pass (appended duplicates). Only the cross-CP gather remains.
        local_boundary_positions = [p for p, _, _ in precomputed]
        local_sample_indices = [s for _, s, _ in precomputed]
        local_log_probs = [v for _, _, v in precomputed]
        if local_log_probs:
            device = local_log_probs[0].device
    else:
        # Unfused path: per-boundary log_softmax from materialized logits.
        for boundary_position, leaves in registry:
            if local_indices is not None:
                matches = (local_indices == boundary_position).nonzero()
                if matches.shape[0] == 0:
                    # Boundary hidden is on another CP rank — that rank emits this
                    # entry's triple; we receive it via the all_gather below.
                    continue
                local_idx = int(matches[0, 0].item())
            else:
                local_idx = boundary_position
            for sample_idx, next_token in leaves:
                if logits_flat is not None:
                    logits_b = logits_flat[local_idx]
                elif hidden_flat is not None and model is not None:
                    h_b = hidden_flat[local_idx]
                    # output_layer expects (seq, batch, hidden) = (1, 1, hidden).
                    logits_b, _ = model.output_layer(h_b.reshape(1, 1, -1))
                    logits_b = logits_b.squeeze(0).squeeze(0)  # (vocab,)
                else:
                    return
                temp_b = temperature[local_idx] if torch.is_tensor(temperature) else temperature
                logp_all = torch.log_softmax(logits_b / temp_b, dim=-1)
                local_boundary_positions.append(boundary_position)
                local_sample_indices.append(sample_idx)
                local_log_probs.append(logp_all[int(next_token)])

    # Cross-CP all_gather: pack (boundary_position, sample_idx, log_prob) into (max_n, 3) float32 tensor.
    cp_world = mpu.get_context_parallel_world_size()
    if cp_world > 1 and magi_key is not None:
        cp_group = mpu.get_context_parallel_group()
        local_count = len(local_log_probs)

        # Get max entry count across ranks via all_reduce(MAX) — we only need
        # the pad size, not individual per-rank counts.
        count_tensor = torch.tensor([local_count], dtype=torch.long, device=device)
        _dist.all_reduce(count_tensor, op=_dist.ReduceOp.MAX, group=cp_group)
        max_n = int(count_tensor.item())

        # Pack (boundary_position, sample_idx, log_prob) into one tensor.
        local_packed = torch.zeros(max_n, 3, dtype=torch.float32, device=device)
        local_packed[:, 1] = -1  # sentinel for padding rows
        if local_count > 0:
            local_packed[:local_count, 0] = torch.tensor(local_boundary_positions, dtype=torch.float32, device=device)
            local_packed[:local_count, 1] = torch.tensor(local_sample_indices, dtype=torch.float32, device=device)
            local_packed[:local_count, 2] = torch.stack(local_log_probs).to(torch.float32)

        # Single all_gather instead of 3 separate ones.
        all_packed = [torch.zeros_like(local_packed) for _ in range(cp_world)]
        _dist.all_gather(all_packed, local_packed, group=cp_group)

        # Validate: each (boundary_position, sample_idx) must arrive from exactly one rank.
        registry_keys = {
            (boundary_position, sample_idx) for boundary_position, leaves in registry for sample_idx, _ in leaves
        }
        key_counts: Counter[tuple[int, int]] = Counter()

        for r in range(cp_world):
            for i in range(max_n):
                sample_idx = int(all_packed[r][i, 1].item())
                if sample_idx == -1:
                    continue  # padding
                boundary_position = int(all_packed[r][i, 0].item())
                log_prob = all_packed[r][i, 2]
                key = (boundary_position, sample_idx)
                key_counts[key] += 1
                boundary_logps.setdefault(sample_idx, []).append((boundary_position, log_prob))

        # Single validation pass: unexpected, duplicate, or missing entries.
        unexpected = set(key_counts) - registry_keys
        duplicates = {k for k, v in key_counts.items() if v > 1}
        missing = registry_keys - set(key_counts)
        if unexpected or duplicates or missing:
            parts = []
            if unexpected:
                parts.append(f"{len(unexpected)} unexpected (e.g. {next(iter(unexpected))})")
            if duplicates:
                parts.append(f"{len(duplicates)} duplicate (e.g. {next(iter(duplicates))})")
            if missing:
                parts.append(f"{len(missing)} missing (e.g. {next(iter(missing))})")
            raise AssertionError(
                f"post_processing_packed_lce: registry mismatch across CP ranks — "
                f"{', '.join(parts)}. Aborting to avoid silent wrong patches."
            )
    else:
        # CP=1 (or non-magi): local triples ARE the full set, no comm/assert.
        for boundary_position, sample_idx, log_prob in zip(
            local_boundary_positions, local_sample_indices, local_log_probs, strict=True
        ):
            boundary_logps.setdefault(sample_idx, []).append((boundary_position, log_prob))

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
    """Fused prefix-tree forward: pops attention key from kwargs, installs
    rope+decoder contexts, delegates to fuse_forward_body."""
    _magi_key = kwargs.pop("magi_attention_key", None)
    _flex_key = kwargs.pop("flex_attention_key", None)
    if _magi_key is None and _flex_key is None:
        return None

    _set_rope_pids(model, position_ids)
    try:
        with prefix_tree_decoder_key_context(model, _magi_key, _flex_key):
            result = fuse_forward_body(
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
    finally:
        _clear_rope_pids(model)
    return result


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
    """Fused-path: preprocess → decoder → LCE (no logits tensor). Assumes rope+decoder-key contexts active."""
    from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
    from megatron.core.utils import deprecate_inference_params

    from verl.utils.model import CausalLMOutputForPPO

    inference_context = kwargs.pop("inference_context", None)
    inference_params = kwargs.pop("inference_params", None)
    inference_context = deprecate_inference_params(inference_context, inference_params)
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

    # Boundary-patch: compute per-leaf boundary log-probs after LCE, before restore.
    post_processing_packed_lce(
        pt_batch,
        hidden_flat=hidden_states,
        model=model,
        temperature=temperature,
        magi_key=magi_key,
        output_weight=output_weight,
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )
    output.entropy = entropy
    output.log_probs = logprobs
    return output


def run_fused_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    labels,
    temperature,
    calculate_entropy,
    *,
    vision_model=False,
    has_vision_data=False,
):
    """Prepare and run the fused prefix-tree forward path.

    Returns output dict, hidden tensor, or ``None`` (fallback to standard fused path).
    """
    if vision_model and has_vision_data:
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(model, input_ids, logits_processor_args)
    if pb is None:
        _logger.warning("prefix_tree: build_prefix_tree_batch returned None; falling back to standard fused path")
        strip_prefix_tree_args(logits_processor_args)
        return None

    strip_prefix_tree_args(logits_processor_args)
    return _fused_core(model, pb, prefix_tree_attention, labels, temperature, calculate_entropy)


def _fused_core(model, pb, prefix_tree_attention, labels, temperature, calculate_entropy):
    """Fused-path core: forward pass with fused vocab projection (LCE)."""
    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)

    post_process = unwrap_model(model).post_process

    # Only the last PP stage (post_process=True) needs labels for LCE.
    # Non-last stages pass labels=None; fuse_forward_body returns before LCE.
    real_tokens = pb.real_tokens
    if post_process:
        if pb.tree_packed_labels is None:
            _logger.warning("prefix_tree[fused]: tree_packed_labels is None; falling back to standard fused path")
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
    output = {"log_probs": restore_flat_to_nested(output_orig.log_probs.reshape(-1), pb, apply_boundary_patch=True)}
    if calculate_entropy:
        output["entropy"] = restore_flat_to_nested(output_orig.entropy.reshape(-1), pb)
    return output
