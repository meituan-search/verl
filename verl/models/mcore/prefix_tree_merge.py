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

The ``_magi_attn_forward`` helper (dispatch → calc_attn → undispatch) is copied
verbatim from the fork and lives here so no Megatron source modification is needed.
"""

from __future__ import annotations

import functools
import threading

import torch
from torch import Tensor

from verl.utils.device import get_torch_device

# Module-level capture state for COMPARE_LAYERS=1 debugging
_layer_capture: dict = {}
_layer_counter: dict = {"magi": 0, "flex": 0, "fa": 0}  # per-attn-type layer counters
_current_layer_idx: list = [-1]  # set by _sa_forward, read by _magi_attn_forward
_current_attn_type: list = ["fa"]  # set by _sa_forward, read by _magi_attn_forward
# When COMPARE_LAYERS=2, save full hidden state tensors (not just stats)
_layer_tensors: dict = {}  # key -> Tensor on CPU

# Thread-local storage for active MAGI key during prefix-tree forward.
# Set in model_forward.py before calling model(), cleared after.
_active_magi_key: threading.local = threading.local()

# Set to True during prefix-tree forward to bypass CP-rank-specific RoPE slicing.
_magi_rope_bypass: threading.local = threading.local()


# ---------------------------------------------------------------------------
# flex_attention helper for prefix-tree
# ---------------------------------------------------------------------------


def _flex_attn_forward(
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

    with torch.cuda.nvtx.range("prefix_attn/flex"):
        out = flex_attention(q, k, v, block_mask=flex_attention_key, enable_gqa=enable_gqa)  # (1, Hq, T, D)
    out = out.squeeze(0).permute(1, 0, 2)  # (T, Hq, D)
    return out.reshape(T, 1, -1)  # (T, 1, Hq*D)


# ---------------------------------------------------------------------------
# MAGI attention kernel helper (verbatim from Megatron-LM-prefix-tree fork)
# ---------------------------------------------------------------------------


def _magi_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    magi_attention_key: object,
) -> Tensor:
    """Execute MAGI dispatch → calc_attn → undispatch for prefix-tree batches.

    Input tensors are in thd layout: ``(total_tokens, 1, num_heads, head_dim)``.
    Returns ``(total_tokens, 1, num_heads*head_dim)`` — matching the packed-seq
    thd output shape expected by ``linear_proj``.

    With SP+CP: Q/K/V arrive as (T/TP, 1, H, D) due to SP scatter. We gather
    across TP to get full T tokens, then dispatch/calc_attn/undispatch across
    CP ranks, then scatter back to T/TP for SP-consistent output.
    """
    import os as _os

    import torch as _torch
    import torch.distributed as _dist
    from magi_attention.api import calc_attn, dispatch, undispatch

    q = query.squeeze(1)  # (T, np, hn) — already full T (Megatron gathers SP before calling here)
    k = key.squeeze(1)
    v = value.squeeze(1)
    _gathered = False

    _save_cp = _os.environ.get("SAVE_CP_TENSORS") == "1"
    _cp_rank = _dist.get_rank() if _dist.is_initialized() else 0
    _save_dir = _os.environ.get("CP_TENSOR_DIR", "/tmp/claude/cp_tensors")
    # Layer index and attn type set by _sa_forward (which owns the counter increment)
    _layer_idx = _current_layer_idx[0]
    _atype = _current_attn_type[0]

    def _save(name, tensor):
        if not _save_cp:
            return
        _os.makedirs(_save_dir, exist_ok=True)
        path = f"{_save_dir}/{_atype}_{name}_rank{_cp_rank}_layer{_layer_idx}.pt"
        _torch.save(tensor.detach().cpu(), path)
        print(
            f"[CP_SAVE] {_atype}_{name} shape={tuple(tensor.shape)} rank={_cp_rank} layer={_layer_idx} → {path}",
            flush=True,
        )

    # MAGI_TIMING=1 → CUDA-event timing for every layer on rank 0
    _timing = _os.environ.get("MAGI_TIMING") == "1" and _cp_rank == 0

    def _evt():
        e = _torch.cuda.Event(enable_timing=True); e.record(); return e

    # NVTX and CUDA-event timing for nsys / MAGI_TIMING profiling.
    # Gated behind _timing to avoid per-layer overhead in production
    # (36 layers × n_micro_batches × 6 NVTX calls = significant overhead otherwise).
    _pfx = f"magi/L{_layer_idx}"
    if _timing:
        _torch.cuda.nvtx.range_push(f"{_pfx}/dispatch")
    t0 = _evt() if _timing else None
    dq = dispatch(q, magi_attention_key)
    t0b = _evt() if _timing else None
    dk = dispatch(k, magi_attention_key)
    t0c = _evt() if _timing else None
    dv = dispatch(v, magi_attention_key)
    t1 = _evt() if _timing else None
    if _timing:
        _torch.cuda.nvtx.range_pop()

    _save("after_dispatch_Q", dq)
    _save("after_dispatch_K", dk)
    _save("after_dispatch_V", dv)

    if _timing:
        _torch.cuda.nvtx.range_push(f"{_pfx}/calc_attn")
    out, _ = calc_attn(dq, dk, dv, magi_attention_key)
    t2 = _evt() if _timing else None
    if _timing:
        _torch.cuda.nvtx.range_pop()

    _save("after_calc_attn_out", out)

    if _timing:
        _torch.cuda.nvtx.range_push(f"{_pfx}/undispatch")
    out = undispatch(out, magi_attention_key)
    t3 = _evt() if _timing else None
    if _timing:
        _torch.cuda.nvtx.range_pop()

    if _timing:
        _torch.cuda.synchronize()
        T = q.shape[0]
        print(
            f"[MAGI-TIMING] layer={_layer_idx} tokens={T}"
            f" dq={t0.elapsed_time(t0b):.2f}ms"
            f" dk={t0b.elapsed_time(t0c):.2f}ms"
            f" dv={t0c.elapsed_time(t1):.2f}ms"
            f" dispatch={t0.elapsed_time(t1):.2f}ms"
            f" calc_attn={t1.elapsed_time(t2):.2f}ms"
            f" undispatch={t2.elapsed_time(t3):.2f}ms"
            f" total={t0.elapsed_time(t3):.2f}ms",
            flush=True,
        )
    # after_undispatch_out saved by _sa_forward wrapper

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
            return _magi_attn_forward(query, key, value, magi_attention_key)
        if flex_attention_key is not None:
            return _flex_attn_forward(query, key, value, flex_attention_key)
        # FA3 path
        with torch.cuda.nvtx.range("full_attn/fa3"):
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

        import os as _os

        import torch as _torch
        import torch.distributed as _dist

        _save_cp = _os.environ.get("SAVE_CP_TENSORS") == "1"
        _cp_rank = _dist.get_rank() if _dist.is_initialized() else 0
        _save_dir = _os.environ.get("CP_TENSOR_DIR", "/tmp/claude/cp_tensors")
        _atype = "magi" if magi_attention_key else ("flex" if flex_attention_key else "fa")
        _layer_idx = _layer_counter[_atype]
        _layer_counter[_atype] += 1
        _current_layer_idx[0] = _layer_idx  # share with _magi_attn_forward
        _current_attn_type[0] = _atype  # share with _magi_attn_forward

        def _save_sa(name, tensor):
            if not _save_cp:
                return
            _os.makedirs(_save_dir, exist_ok=True)
            path = f"{_save_dir}/{_atype}_{name}_rank{_cp_rank}_layer{_layer_idx}.pt"
            _torch.save(tensor.detach().cpu(), path)
            print(
                f"[CP_SAVE] {_atype}_{name} shape={tuple(tensor.shape)} rank={_cp_rank} layer={_layer_idx} → {path}",
                flush=True,
            )

        @functools.wraps(_real_ca_forward)
        def _ca_forward_with_key(q, k, v, *args, **kw):
            _save_sa("before_dispatch_Q", q.squeeze(1) if q.dim() == 4 else q)
            _save_sa("before_dispatch_K", k.squeeze(1) if k.dim() == 4 else k)
            _save_sa("before_dispatch_V", v.squeeze(1) if v.dim() == 4 else v)
            out = _real_ca_forward(
                q,
                k,
                v,
                *args,
                magi_attention_key=magi_attention_key if attn_key else None,
                flex_attention_key=flex_attention_key if attn_key else None,
                **kw,
            )
            _save_sa("after_undispatch_out", out.squeeze(1) if out.dim() == 3 else out)
            return out

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

        # COMPARE_LAYERS: capture forward hidden state + backward gradient per layer
        import os as _os

        _cl = _os.environ.get("COMPARE_LAYERS", "0")
        if _cl in ("1", "2"):
            import torch.distributed as _dist

            hs_out = out[0] if isinstance(out, tuple) else out
            idx = _layer_counter[0]
            tag = "flex" if flex_attention_key is not None else "magi" if magi_attention_key is not None else "fa3"
            key = f"{tag}_{idx}"
            # Only capture every 5 layers (0, 5, 10, ...) to see trend without huge files
            _layer_counter[0] += 1
            if idx % 5 != 0 or key in _layer_capture:
                return out

            if True:  # replaces the old `if key not in _layer_capture:` block
                # All-gather across TP group to reconstruct full (T, H) hidden state
                h_local = hs_out.detach().float()  # (T_local, 1, H) — post-reduce-scatter slice
                if _dist.is_initialized():
                    try:
                        from megatron.core import mpu

                        tp_group = mpu.get_tensor_model_parallel_group()
                        tp_size = mpu.get_tensor_model_parallel_world_size()
                    except Exception:
                        tp_group = None
                        tp_size = _dist.get_world_size()
                    gathered = [torch.zeros_like(h_local) for _ in range(tp_size)]
                    _dist.all_gather(gathered, h_local, group=tp_group)
                    h = torch.cat(gathered, dim=0)  # (T_full, 1, H)
                else:
                    h = h_local
                if not _dist.is_initialized() or _dist.get_rank() == 0:
                    _layer_capture[key] = {
                        "fwd_mean": h.mean().item(),
                        "fwd_std": h.std().item(),
                        "fwd_norm": h.norm().item(),
                        "shape": list(h.shape),
                    }
                    # COMPARE_LAYERS=2: also save full tensor to disk for per-token diff
                    if _cl == "2":
                        _layer_tensors[key] = h.cpu()
                    # Capture gradient via retain_grad + hook
                    _bwd_key = key

                    def _make_bwd_hook(bk, save_full=(_cl == "2")):
                        def _bwd(grad):
                            if bk in _layer_capture and "bwd_norm" not in _layer_capture[bk]:
                                g = grad.detach().float()
                                _layer_capture[bk]["bwd_mean"] = g.mean().item()
                                _layer_capture[bk]["bwd_std"] = g.std().item()
                                _layer_capture[bk]["bwd_norm"] = g.norm().item()
                                if save_full:
                                    _layer_tensors[bk + "_bwd"] = g.cpu()

                        return _bwd

                    if hs_out.requires_grad:
                        hs_out.retain_grad()
                        hs_out.register_hook(_make_bwd_hook(_bwd_key))
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
