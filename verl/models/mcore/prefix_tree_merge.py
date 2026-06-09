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
"""Monkey-patches for upstream Megatron-LM to support MAGI flex-attention.

Upstream Megatron-LM does not have the ``magi_attention_key`` parameter in its
forward call chain.  This module patches the five classes that need it so that
verl can work with a stock (unmodified) Megatron installation.

Call ``apply_prefix_tree_patch()`` once before model construction (e.g. from
``verl/models/mcore/patch.py:apply_patch`` or from the engine initialiser).

Patch chain (each wrapper accepts ``magi_attention_key``/``flex_attention_key`` and threads them):
    GPTModel.forward
    → TransformerBlock.forward  (both the checkpointed and normal variants)
    → TransformerLayer.forward
    → SelfAttention.forward     (both checkpointed and normal core-attention calls)
    → TEDotProductAttention.forward  (early-return MAGI branch)

The ``magi_attn_forward`` helper (dispatch → calc_attn → undispatch) is copied
verbatim from the fork and lives here so no Megatron source modification is needed.
"""

from __future__ import annotations

import functools
import threading

import torch
from torch import Tensor

# Set to True during prefix-tree forward to bypass CP-rank-specific RoPE slicing.
_magi_rope_bypass: threading.local = threading.local()


# ---------------------------------------------------------------------------
# flex_attention helper for prefix-tree
# ---------------------------------------------------------------------------


def flex_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    flex_attention_key: object,
) -> Tensor:
    """Execute PyTorch flex_attention for prefix-tree batches.

    Input tensors are in thd layout: ``(total_tokens, 1, num_heads, head_dim)``.
    Returns ``(total_tokens, 1, num_heads*head_dim)``.

    Uses torch.utils.checkpoint to avoid storing the O(T²) attention score
    matrix for backward — recomputes the forward pass instead (O(T) memory).
    """
    from torch.nn.attention.flex_attention import flex_attention

    T, _, H, D = query.shape
    q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0)  # (1, H, T, D)
    k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    enable_gqa = q.shape[1] != k.shape[1]

    out = flex_attention(q, k, v, block_mask=flex_attention_key, enable_gqa=enable_gqa)
    out = out.squeeze(0).permute(1, 0, 2)  # (T, Hq, D)
    return out.reshape(T, 1, -1)  # (T, 1, Hq*D)


# ---------------------------------------------------------------------------
# MAGI attention kernel helper (verbatim from Megatron-LM-prefix-tree fork)
# ---------------------------------------------------------------------------


def magi_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    magi_attention_key: object,
) -> Tensor:
    """Execute MAGI calc_attn for prefix-tree batches.

    When ``needs_merge`` / ``needs_spread`` are set on ``magi_attention_key``
    they gate dispatch/undispatch.  Otherwise both default to True.

    Returns ``(total_tokens, 1, num_heads*head_dim)``.
    """
    from magi_attention.api import calc_attn, dispatch, undispatch

    q = query.squeeze(1)
    k = key.squeeze(1)
    v = value.squeeze(1)

    needs_merge = getattr(magi_attention_key, "needs_merge", True)
    needs_spread = getattr(magi_attention_key, "needs_spread", True)

    if needs_merge:
        dq = dispatch(q, magi_attention_key)
        dk = dispatch(k, magi_attention_key)
        dv = dispatch(v, magi_attention_key)
    else:
        dq, dk, dv = q, k, v

    out, _ = calc_attn(dq, dk, dv, magi_attention_key)

    if needs_spread:
        out = undispatch(out, magi_attention_key)

    return out.reshape(out.shape[0], 1, -1)


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def apply_prefix_tree_patch() -> None:
    """Monkey-patch upstream Megatron-LM classes to support prefix-tree attention (flex and MAGI).

    Safe to call multiple times — subsequent calls are no-ops (checks for the
    ``_magi_patched`` sentinel attribute).
    """
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if getattr(TEDotProductAttention, "_prefix_tree_patched", False):
        return  # already patched

    # ------------------------------------------------------------------
    # 1. TEDotProductAttention.forward — add early-return MAGI branch
    # ------------------------------------------------------------------
    _orig_te_forward = TEDotProductAttention.forward

    @functools.wraps(_orig_te_forward)
    def _te_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
        magi_attention_key=None,
        flex_attention_key=None,
        **kwargs,
    ):
        if magi_attention_key is not None:
            return magi_attn_forward(query, key, value, magi_attention_key)
        if flex_attention_key is not None:
            return flex_attn_forward(query, key, value, flex_attention_key)
        # FA3 path
        return _orig_te_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            num_splits=num_splits,
            **kwargs,
        )

    TEDotProductAttention.forward = _te_forward

    # ------------------------------------------------------------------
    # 2. SelfAttention._checkpointed_attention_forward — thread kwarg
    # ------------------------------------------------------------------
    _orig_sa_ckpt = SelfAttention._checkpointed_attention_forward

    @functools.wraps(_orig_sa_ckpt)
    def _sa_ckpt_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        magi_attention_key=None,
        flex_attention_key=None,
        **kwargs,
    ):
        from megatron.core.tensor_parallel.random import checkpoint as tensor_parallel_checkpoint
        from megatron.core.transformer.attention import AttnMaskType

        try:
            from megatron.core.transformer.utils import apply_module
        except ImportError:
            apply_module = lambda m: m  # noqa: E731

        def custom_forward(*inputs):
            q, k, v, amask = inputs[0], inputs[1], inputs[2], inputs[3]
            _attn_mask_type = AttnMaskType(inputs[5].item())
            # Keys are already injected via _ca_forward_with_key wrapper on self.core_attention
            # (set by _sa_forward before calling _orig_sa_forward). Don't pass them again.
            return apply_module(self.core_attention)(
                q,
                k,
                v,
                amask,
                attn_mask_type=_attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type_tensor = torch.tensor([attn_mask_type.value], dtype=torch.int)
        return tensor_parallel_checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type_tensor,
        )

    SelfAttention._checkpointed_attention_forward = _sa_ckpt_forward

    # ------------------------------------------------------------------
    # 3. SelfAttention.forward — accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_sa_forward = SelfAttention.forward

    @functools.wraps(_orig_sa_forward)
    def _sa_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        attn_key = magi_attention_key or flex_attention_key
        _real_ca_forward = self.core_attention.forward

        @functools.wraps(_real_ca_forward)
        def _ca_forward_with_key(q, k, v, *args, **kw):
            return _real_ca_forward(
                q,
                k,
                v,
                *args,
                magi_attention_key=magi_attention_key if attn_key else None,
                flex_attention_key=flex_attention_key if attn_key else None,
                **kw,
            )

        self.core_attention.forward = _ca_forward_with_key
        try:
            out = _orig_sa_forward(self, hidden_states, attention_mask, **kwargs)
        finally:
            self.core_attention.forward = _real_ca_forward
        return out

    SelfAttention.forward = _sa_forward

    # ------------------------------------------------------------------
    # 4. TransformerLayer.forward — accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_tl_forward = TransformerLayer.forward

    @functools.wraps(_orig_tl_forward)
    def _tl_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
        else:
            no_expand = getattr(attn_key, "no_expand", False)
            if no_expand:
                attn_key.needs_merge = self.layer_number == 0
                attn_key.needs_spread = self.layer_number == self.config.num_layers - 1
            else:
                attn_key.needs_merge = True
                attn_key.needs_spread = True
            _real_sa_forward = self.self_attention.forward

            @functools.wraps(_real_sa_forward)
            def _sa_forward_with_key(*args, **kw):
                return _real_sa_forward(
                    *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
                )

            self.self_attention.forward = _sa_forward_with_key
            try:
                out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
            finally:
                self.self_attention.forward = _real_sa_forward

        return out

    TransformerLayer.forward = _tl_forward

    # ------------------------------------------------------------------
    # 5. TransformerBlock.forward — accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_tb_forward = TransformerBlock.forward

    @functools.wraps(_orig_tb_forward)
    def _tb_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            return _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        originals = []
        for layer in self.layers:
            originals.append(layer.forward)
            _orig = layer.forward

            def _make_wrapper(orig):
                @functools.wraps(orig)
                def _w(*args, **kw):
                    return orig(
                        *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
                    )

                return _w

            layer.forward = _make_wrapper(_orig)
        try:
            out = _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        finally:
            for layer, orig_fwd in zip(self.layers, originals, strict=False):
                layer.forward = orig_fwd
        return out

    TransformerBlock.forward = _tb_forward

    # ------------------------------------------------------------------
    # 6. GPTModel.forward — accept and pass magi/flex attention key.
    #    Also sets _magi_rope_bypass.active=True so patch #7 below skips
    #    the CP-rank-specific RoPE slicing for the duration of the call.
    # ------------------------------------------------------------------
    _orig_gpt_forward = GPTModel.forward

    @functools.wraps(_orig_gpt_forward)
    def _gpt_forward(
        self, input_ids, position_ids, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs
    ):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            return _orig_gpt_forward(self, input_ids, position_ids, attention_mask, **kwargs)
        _real_decoder_forward = self.decoder.forward

        @functools.wraps(_real_decoder_forward)
        def _decoder_forward_with_key(*args, **kw):
            return _real_decoder_forward(
                *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
            )

        self.decoder.forward = _decoder_forward_with_key
        _prev_rope_bypass = getattr(_magi_rope_bypass, "active", False)
        _magi_rope_bypass.active = True  # all prefix-tree paths bypass CP RoPE slicing
        try:
            out = _orig_gpt_forward(self, input_ids, position_ids, attention_mask, **kwargs)
        finally:
            self.decoder.forward = _real_decoder_forward
            _magi_rope_bypass.active = _prev_rope_bypass
        return out

    GPTModel.forward = _gpt_forward

    # ------------------------------------------------------------------
    # 7. RotaryEmbedding.forward — bypass CP-rank RoPE slicing for prefix-tree.
    #
    # Root cause of CP>1 bug: with BSHD format (packed_seq=False), Megatron slices
    # rotary position frequencies per CP rank via get_pos_emb_on_this_cp_rank.
    # MAGI/flex prefix-tree passes the full T-token sequence to every CP rank;
    # different per-rank RoPE rotations produce different Q/K values → wrong attention.
    #
    # Fix: when prefix-tree is active (_magi_rope_bypass.active=True), skip CP slicing
    # and return the full-sequence frequencies.  Only positions [0..T-1] are applied
    # in BSHD attention, so correctness is maintained regardless of CP rank.
    # ------------------------------------------------------------------
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    _orig_rope_cached = RotaryEmbedding.forward  # lru_cache wrapper
    _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # actual function body

    def _rope_forward_pt(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        if getattr(_magi_rope_bypass, "active", False):
            # Bypass CP slice: return full-sequence freqs identical on all CP ranks.
            # get_rotary_seq_len multiplies by cp_size (yielding 2T for CP=2), so we
            # divide it back out here to get the true flat-sequence length T.
            _cp = cp_group or getattr(self, "cp_group", None)
            _cp_size = _cp.size() if _cp is not None else 1
            actual_seq_len = max_seq_len // _cp_size
            return _orig_rope_fn(self, actual_seq_len, offset=offset, packed_seq=True, cp_group=None)
        return _orig_rope_cached(self, max_seq_len, offset=offset, packed_seq=packed_seq, cp_group=cp_group)

    RotaryEmbedding.forward = _rope_forward_pt

    TEDotProductAttention._prefix_tree_patched = True


# Backward-compatibility alias
apply_magi_patch = apply_prefix_tree_patch
