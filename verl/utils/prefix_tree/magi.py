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
"""Prefix-tree + MAGI utilities for verl SFT training.

Dispatches a micro-batch through either the hash-based
(:mod:`verl.utils.prefix_tree.hash_based`) or dynamic-trie
(:mod:`verl.utils.prefix_tree.dynamic`) detection path, materialises a flat
layout via :func:`verl.utils.prefix_tree.utils.build_layout_from_tree_node`,
and builds a MAGI / flex attention key for the result.

Usage (inside gptmodel_forward_model_engine):

    pt_batch = build_prefix_tree_micro_batch(model, input_ids, loss_mask, position_ids)
    if pt_batch is not None:
        output = model(
            input_ids=pt_batch.tree_packed_input_ids,
            attention_mask=None,
            position_ids=pt_batch.tree_packed_position_ids,
            packed_seq_params=None,
            magi_attention_key=pt_batch.magi_key,
        )
        output = restore_flat_to_nested(output, pt_batch)
"""

from __future__ import annotations

import contextlib
import functools
import logging as _log
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as _dist
from magi_attention.api import DistAttnConfig, get_position_ids, magi_attn_flex_key, undispatch
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.solver.dispatch_solver import DispatchConfig
from megatron.core import parallel_state as mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor
from torch.nn.attention.flex_attention import create_block_mask

from verl.utils.megatron_utils import unwrap_model
from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


@dataclass
class PrefixTreeMagiBatch:
    """Holds the tree-packed layout and MAGI key for one prefix-tree micro-batch."""

    # tree-packed input tensors ready to pass to model(...)
    tree_packed_input_ids: Tensor  # (total_tokens,)
    tree_packed_position_ids: Tensor  # (total_tokens,)
    tree_packed_loss_mask: Optional[Tensor]  # (total_tokens,) or None

    # Attention keys — one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # mapping needed for output restoration
    # segment_to_sample[i] = original sample index for leaf i
    segment_to_sample: list[int]
    # segment_ranges[i] = (start, end) token offset in flat layout for leaf i
    segment_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]

    # original batch size (= number of leaves for single-level tree)
    original_batch_size: int

    # per-token labels derived from tree_packed_tokens via within-segment shift
    tree_packed_labels: Optional[Tensor] = None  # (total_tokens,)

    # number of real (non-padding) tokens; may be < tree_packed_input_ids.shape[0]
    # when tp_size > 1 padding was added for sequence-parallel divisibility
    real_tokens: int = 0

    # ancestor_segment_ranges[i] = list of (start,end) flat ranges that precede leaf i
    # For single-level: None (use prefix_range directly)
    # For multilevel: [(0, root_end), (turn2_start, turn2_end)] etc.
    ancestor_segment_ranges: Optional[list[list[tuple[int, int]]]] = None

    # CP-local tensors: after magi dispatch, each CP rank only processes its assigned tokens.
    # When CP=1, these equal tree_packed_input_ids/tree_packed_position_ids/tree_packed_loss_mask.
    # Shape: (local_tokens, ...) where local_tokens = total_tokens / cp_effective
    local_tree_packed_input_ids: Optional[Tensor] = None
    local_tree_packed_position_ids: Optional[Tensor] = None
    local_tree_packed_loss_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.real_tokens == 0:
            self.real_tokens = int(self.tree_packed_input_ids.shape[0])
        # Default local to full when not set (CP=1 or flex path)
        if self.local_tree_packed_input_ids is None:
            self.local_tree_packed_input_ids = self.tree_packed_input_ids
        if self.local_tree_packed_position_ids is None:
            self.local_tree_packed_position_ids = self.tree_packed_position_ids
        if self.local_tree_packed_loss_mask is None:
            self.local_tree_packed_loss_mask = self.tree_packed_loss_mask


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
    """Build a PrefixTreeMagiBatch from a micro-batch using a per-mb subtrie.

    The subtrie is produced once per training step in
    ``prepare_prefix_tree_micro_batches``:
      1. ``greedy_build_tries`` builds a global trie from ALL batch samples.
      2. ``subtrie_view`` prunes it to this mb's sample subset → the subtrie.

    The subtrie is then reused across all forward passes (OLP + actor update)
    for this mb without rebuilding.

    Returns None when the subtrie is not available (prefix sharing not
    detected or dynamic bsz disabled), signalling the caller to fall back
    to standard attention.

    Args:
        model: Megatron model (used to read num_heads / head_dim from config).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        loss_mask: Optional NestedTensor matching input_ids shape.
        position_ids: Optional NestedTensor matching input_ids shape.
        attention_type: ``"flex"`` or ``"magi"``.
        tp_size / cp_size: Tensor / context parallel world sizes.
        subtrie: Per-mb subtrie from ``prepare_prefix_tree_micro_batches``.

    Returns:
        PrefixTreeMagiBatch or None.
    """

    samples = _unpack_nested_to_list(input_ids, mask=loss_mask)
    if not samples:
        _log.getLogger(__name__).warning(
            "prefix_tree: build_prefix_tree_micro_batch got empty samples — returning None"
        )
        return None
    loss_masks_by_sample = _unpack_nested_to_list(loss_mask)
    position_ids_by_sample = _unpack_nested_to_list(position_ids, mask=loss_mask)

    if subtrie is None:
        # No pre-built subtrie (e.g. use_dynamic_bsz=False): build locally.
        # build_tree_dynamic does token-by-token trie detection on this mb's
        # samples — slower than the global path but correct.
        subtrie = build_tree_dynamic(samples)

    if subtrie is None:
        _log.getLogger(__name__).error(
            "build_prefix_tree_micro_batch: no prefix sharing found (n=%d); falling back to standard attention",
            len(samples),
        )
        return None

    try:
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
    except Exception as _e:
        _log.getLogger(__name__).exception(
            "build_prefix_tree_micro_batch: falling back to standard attention (%s: %s) "
            "subtrie_nodes=%d subtrie_leaves=%d",
            type(_e).__name__,
            _e,
            len(subtrie.nodes) if subtrie is not None else -1,
            len(subtrie.leaf_node_ids) if subtrie is not None else -1,
        )
        return None


def _build_sample_tensors(flat_tensor: Tensor, pt_batch: PrefixTreeMagiBatch) -> list:
    """Build a per-sample list of tensors from a flat deduplicated tensor.

    Shared logic for restore_flat_to_nested and expand_flat_to_per_sample.
    Returns sample_tensors[sample_idx] = cat(ancestor_slices..., leaf_slice).
    """
    prefix_start, prefix_end = pt_batch.prefix_range
    prefix_slice = flat_tensor[prefix_start:prefix_end]
    n = pt_batch.original_batch_size
    sample_tensors: list[Optional[Tensor]] = [None] * n
    for leaf_idx, sample_idx in enumerate(pt_batch.segment_to_sample):
        s, e = pt_batch.segment_ranges[leaf_idx]
        leaf_slice = flat_tensor[s:e]
        if pt_batch.ancestor_segment_ranges is not None:
            parts = [flat_tensor[a:b] for a, b in pt_batch.ancestor_segment_ranges[leaf_idx]]
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            sample_tensors[sample_idx] = torch.cat([prefix_slice, leaf_slice], dim=0)
    return sample_tensors


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
    sample_tensors = _build_sample_tensors(flat_tensor, pt_batch)
    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by segment_to_sample"
    )
    # as_nested_tensor (not nested_tensor) preserves grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


def expand_flat_to_per_sample(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
) -> Tensor:
    """Expand deduplicated flat tensor to per-sample flat tensor via torch.cat.

    Replicates shared prefix/anchor slices for each sample, returning a single
    flat tensor ordered by original sample index (matching restore_flat_to_nested).
    Uses torch.cat instead of nested tensors — safe for autograd (training).

    Args:
        flat_tensor: (total_flat_tokens, ...) deduplicated representation.
        pt_batch: PrefixTreeMagiBatch from build_prefix_tree_micro_batch.

    Returns:
        (total_expanded, ...) flat tensor with per-sample concatenation.
    """
    sample_tensors = _build_sample_tensors(flat_tensor, pt_batch)
    assert all(t is not None for t in sample_tensors), (
        "expand_flat_to_per_sample: some sample indices were not covered by segment_to_sample"
    )
    return torch.cat(sample_tensors, dim=0)


def dispatch_pt_batch(pt_batch: PrefixTreeMagiBatch) -> tuple[Tensor, Tensor]:
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


@contextlib.contextmanager
def prefix_tree_rope_context(model, position_ids: Optional[Tensor]):
    """Override ``rotary_pos_emb.forward`` to index by per-token *position_ids*.

    Shared by both fused and unfused prefix-tree paths.  Megatron's default
    RoPE slicing assumes each CP rank holds sequential positions
    ``[r·T/CP .. (r+1)·T/CP]``; after MAGI dispatch each rank holds
    non-sequential tokens whose ``position_ids`` are arbitrary.  This context
    builds the full RoPE table (``cp_group=None``) and indexes it directly by
    the actual local ``position_ids``.

    No-op when ``model`` has no ``rotary_pos_emb`` or ``position_ids`` is None.
    """
    rope_mod = getattr(model, "rotary_pos_emb", None)
    if rope_mod is None or position_ids is None:
        yield
        return

    pids = position_ids.reshape(-1)
    _real_rope_fwd = rope_mod.forward
    _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # bypass lru_cache

    def _rope_fwd_with_pids(max_seq_len, offset=0, packed_seq=False, cp_group=None):
        actual_seq_len = int(pids.max().item()) + 1
        emb = _orig_rope_fn(rope_mod, actual_seq_len, offset=0, packed_seq=True, cp_group=None)
        # All PP stages use seq-first Q=(seq,1,H,D) — freqs=(seq,1,1,dim)
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
    """Override ``model.decoder.forward`` to inject the attention key.

    Shared by both fused and unfused prefix-tree paths.  The decoder's forward
    signature doesn't accept ``magi_attention_key`` / ``flex_attention_key``;
    the patched TEDotProductAttention reads them from its module's forward
    kwargs.  This context wraps ``decoder.forward`` to inject the keys for the
    duration of one call.
    """
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

    Vocab projection stays fused via :func:`linear_cross_entropy` — no
    ``(flat_tokens, vocab)`` logits tensor is materialised.
    """
    from collections import OrderedDict as _OrderedDict

    from megatron.core import parallel_state as _ps
    from megatron.core.config_logger import has_config_logger_enabled as _has_cfg_log
    from megatron.core.config_logger import log_config_to_disk as _log_cfg
    from megatron.core.utils import deprecate_inference_params as _dep_inf

    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy as _lce
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
        # CP-local LCE: run on local hidden (local_tokens), undispatch 1D outputs (KB-scale).
        # Gather from sequence-parallel region first (TP>1 shards the sequence in SP mode).
        if model.config.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        local_indices = get_position_ids(magi_key)
        flat_padded = pt_batch.tree_packed_input_ids.shape[0]
        pad = flat_padded - labels.shape[0]
        labels_full = torch.cat([labels, labels.new_zeros(pad)]) if pad > 0 else labels
        logprobs, entropy = _lce(
            hidden_states,
            output_weight,
            labels_full[local_indices],
            temperature,
            "none",
            _ps.get_tensor_model_parallel_group(),
        )
        logprobs = undispatch(logprobs.reshape(-1), magi_key)[: pt_batch.real_tokens]
        entropy = undispatch(entropy.reshape(-1), magi_key)[: pt_batch.real_tokens]
    else:
        if model.config.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        logprobs, entropy = _lce(
            hidden_states,
            output_weight,
            labels,
            temperature,
            "none",
            _ps.get_tensor_model_parallel_group(),
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


def fuse_undispatch_and_expand_hidden(
    hidden: Tensor,
    magi_key,
    pt_batch: PrefixTreeMagiBatch,
) -> Tensor:
    """Fused-path: convert per-rank local hidden states → per-sample flat order.

    Used inside ``_fused_GPTModel_forward`` to prepare hidden states for
    ``linear_cross_entropy``.  The output is in per-sample flat order (prefix
    replicated per sample, leaves concatenated in sample-index order), matching
    ``labels_rmpad`` from ``preprocess_thd_engine(labels, need_roll=True)``.

    Args:
        hidden: (1, local_tokens, hidden_dim) local hidden states from decoder.
        magi_key: MAGI key for ``undispatch``.
        pt_batch: PrefixTreeMagiBatch.

    Returns:
        (1, total_expanded, hidden_dim) per-sample flat hidden states.
    """
    hidden = undispatch(hidden.squeeze(0), magi_key)  # (flat_padded, hidden)
    hidden = hidden[: pt_batch.real_tokens]  # (real_tokens, hidden)
    hidden = expand_flat_to_per_sample(hidden, pt_batch)  # (total_expanded, hidden)
    return hidden.unsqueeze(0)


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
    added to the attention rectangles — they are stripped before loss, and
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
            if params.tree_packed_loss_mask is not None:
                params.tree_packed_loss_mask = torch.cat(
                    [params.tree_packed_loss_mask, params.tree_packed_loss_mask.new_zeros(pad_len)]
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
        tree_packed_loss_mask=params.tree_packed_loss_mask,
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
        local_tree_packed_loss_mask=params.tree_packed_loss_mask,
    )


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
    num_heads_q = cfg.num_attention_heads
    # GQA: num_query_groups may be set; fall back to num_heads_q if not
    num_heads_kv = getattr(cfg, "num_query_groups", num_heads_q) or num_heads_q
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
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )


# model-forward helpers — consumed by verl/models/mcore/model_forward.py


_PREFIX_TREE_KEYS = frozenset(
    {
        "loss_mask",
        "use_prefix_tree",
        "prefix_tree_attention",
        "prefix_tree_subtree",
        "prefix_tree_no_expand_middle",
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


def get_prefix_tree_kwargs(
    use_prefix_tree: bool,
    prefix_tree_attention: str,
) -> dict:
    """Return prefix-tree keys for injection into *logits_processor_args*.

    When prefix-tree is off returns an empty dict so no prefix keys pollute
    the args dict.
    """
    if not use_prefix_tree:
        return {}
    return {
        "use_prefix_tree": use_prefix_tree,
        "prefix_tree_attention": prefix_tree_attention,
    }


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
    if vision_model or mtp_enable_train:
        _log.getLogger(__name__).warning(
            "prefix_tree: skipping prefix-tree path (vision_model=%s, mtp_enable_train=%s) — "
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
        "prefix_tree: build_prefix_tree_batch returned None — falling back to standard THD path "
        "(post_process=%s). If this appears for one PP stage but not the other, the hidden-state "
        "format will mismatch between stages.",
        post_process,
    )
    strip_prefix_tree_args(logits_processor_args)
    return None


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

    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_pt_batch(pt_batch)
        with prefix_tree_rope_context(model, local_position_ids):
            output_orig = model(
                input_ids=local_input_ids,
                attention_mask=None,
                position_ids=local_position_ids,
                packed_seq_params=None,
                magi_attention_key=pt_batch.magi_key,
                **model_kwargs,
            )
    else:
        output_orig = model(
            input_ids=tree_packed_input_ids,
            attention_mask=None,
            position_ids=tree_packed_position_ids,
            packed_seq_params=None,
            flex_attention_key=pt_batch.flex_key,
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
        if "temperature" in orig_args:
            t = orig_args["temperature"]
            if isinstance(t, torch.Tensor) and t.is_nested:
                # Per-sample temperature: expand to match tree-packed structure.
                # t has shape (batch_size,) as nested tensor; we need a per-token
                # tensor over the full flat layout (total_flat). The flat layout
                # contains prefix root + internal ancestor nodes + leaf nodes, so
                # we must fill every token — covering prefix, each leaf, and each
                # leaf's ancestor chain (ancestor_segment_ranges). Missing the
                # internal ancestor tokens shrinks the cat below total_flat.
                temp_by_sample = t.values()  # (batch_size,)
                tree_packed_t = torch.ones(total_flat, 1, dtype=torch.float32, device=tree_packed_label.device)
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
            elif isinstance(t, torch.Tensor):
                scalar_t = t.flatten()[0].item()
                tree_packed_t = torch.full(
                    (total_flat, 1), scalar_t, dtype=torch.float32, device=tree_packed_label.device
                )
            else:
                scalar_t = float(t)
                tree_packed_t = torch.full(
                    (total_flat, 1), scalar_t, dtype=torch.float32, device=tree_packed_label.device
                )
        else:
            tree_packed_t = torch.ones(total_flat, 1, dtype=torch.float32, device=tree_packed_label.device)
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

        if isinstance(output_dict, dict):
            nested_ids = restore_flat_to_nested(tree_packed_ids, pt_batch)
            cu_seqlen = nested_ids.offsets()

            for key, val in output_dict.items():
                if isinstance(val, torch.Tensor):
                    val_1d = val.reshape(-1)
                    if val_1d.shape[0] == n_logits:
                        if prefix_tree_attention == "magi":
                            val_1d = undispatch(val_1d, pt_batch.magi_key)[:real_tokens]
                        val_expanded = expand_flat_to_per_sample(val_1d, pt_batch)
                        tensors = [val_expanded[cu_seqlen[i] : cu_seqlen[i + 1]] for i in range(len(cu_seqlen) - 1)]
                        output_dict[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        return output_dict
    else:
        # Intermediate PP stage (post_process=False) or no logits_processor.
        # output_orig is (1, flat_tokens, hidden_dim) after normalization above.
        # Stage 0 transposes BSH→SBHD internally (embedding → seq-first).
        # We must send the same seq-first format (seq, 1, hidden) so all downstream
        # stages also get seq-first Q — no per-stage conditional needed, PP=N safe.
        return output_orig.permute(1, 0, 2)  # (1,seq,hid) → (seq,1,hid)


def fuse_try_forward_prefix_tree(
    model,
    input_ids,
    labels,
    temperature: float,
    logits_processor_args: dict,
    calculate_entropy: bool,
):
    """Fused-path: try to build + forward a prefix-tree batch with fused vocab projection.

    Counterpart of :func:`unfuse_try_forward_prefix_tree` for the
    ``use_fused_kernels=True`` path.  The vocab projection + log-prob
    computation stays fused inside ``_fused_GPTModel_forward`` via
    :func:`linear_cross_entropy` — the unfused path materialises
    ``(flat_tokens, vocab)`` logits and runs ``logits_processor`` outside the
    model, but the fused path never materialises the full vocab tensor.

    Limitations vs unfused path:
      - **Scalar temperature only.**  ``linear_cross_entropy`` asserts
        ``isinstance(temperature, float)``.  Per-sample temperature must use
        the unfused path.
      - **Scalar temperature only.**  Per-sample temperature requires unfused path.
      - **PP support**: on non-last stages (``not post_process``), returns the
        raw hidden-state tensor (pipeline schedule sends it to the next stage).
        Last stage (``post_process=True``) returns the log_probs/entropy dict.

    Args:
        model: Megatron GPTModel (forward patched to ``_fused_GPTModel_forward``).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        labels: NestedTensor — used for per-sample offsets only; actual labels
            come from ``pt_batch.tree_packed_labels`` (pre-shifted per sample).
        temperature: scalar float.
        logits_processor_args: dict containing ``use_prefix_tree``,
            ``prefix_tree_attention``, ``segment_hashes``, ``segment_lengths``,
            ``prefix_tree_subtree``.  Prefix-tree keys are stripped on return.
        calculate_entropy: whether to return ``entropy`` alongside ``log_probs``.

    Returns:
        ``{"log_probs": NestedTensor, "entropy": NestedTensor}`` (entropy only
        when ``calculate_entropy=True``), or ``None`` when no prefix sharing is
        detected — caller falls through to the standard fused path.
    """

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
            "prefix_tree: build_prefix_tree_batch returned None — falling back to standard fused path"
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_pt_batch(pb)
        attn_kwargs = {"magi_attention_key": pb.magi_key}
    else:
        local_input_ids = pb.local_tree_packed_input_ids.unsqueeze(0)
        local_position_ids = pb.local_tree_packed_position_ids.unsqueeze(0)
        attn_kwargs = {"flex_attention_key": pb.flex_key}

    strip_prefix_tree_args(logits_processor_args)

    post_process = unwrap_model(model).post_process

    # Only the last PP stage (post_process=True) needs labels for LCE.
    # Non-last stages pass labels=None — fuse_forward_body returns before LCE.
    real_tokens = pb.real_tokens
    if post_process:
        if pb.tree_packed_labels is None:
            _log.getLogger(__name__).warning(
                "prefix_tree[fused]: tree_packed_labels is None — falling back to standard fused path"
            )
            return None
        # Pass flat (deduped) labels — LCE runs on real_tokens, not total_expanded.
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

    # output_orig.log_probs / .entropy are (real_tokens,) flat — expand then split.
    cu_seqlens = input_ids.offsets()
    batch_size = input_ids.shape[0]

    log_probs = expand_flat_to_per_sample(output_orig.log_probs.reshape(-1), pb)
    log_probs_nested = torch.nested.as_nested_tensor(
        [log_probs[cu_seqlens[i] : cu_seqlens[i + 1]] for i in range(batch_size)], layout=torch.jagged
    )
    output = {"log_probs": log_probs_nested}
    if calculate_entropy:
        entropy = expand_flat_to_per_sample(output_orig.entropy.reshape(-1), pb)
        output["entropy"] = torch.nested.as_nested_tensor(
            [entropy[cu_seqlens[i] : cu_seqlens[i + 1]] for i in range(batch_size)], layout=torch.jagged
        )
    return output
