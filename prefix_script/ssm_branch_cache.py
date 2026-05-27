"""SSM branch cache for tree-structured rollouts with Qwen3.5 (Gated Delta Net).

Design follows PrefixGrouper's lifecycle:
  - cache object is created per micro-batch (or per tree)
  - passed explicitly to the forward call
  - cleared immediately after the micro-batch / tree completes

For SSM/GDN models the prefix state is O(1) in sequence length.
Cache it at branch points → run each leaf suffix in O(S) instead of O(P+S).

Patch API (mirrors PrefixGrouper):
    apply_ssm_patch()               # once at startup, idempotent
    cache = SSMBranchCache(model)   # per micro-batch / per tree
    with cache:                     # clear() on exit, matching PrefixGrouper lifecycle
        outputs = ssm_tree_forward(model, tree, cache=cache)

Architecture (Qwen3.5-0.8B, 24 layers):
  - 18 linear_attention (GDN) layers, 6 full_attention every 4th layer
  - GDN state per layer:
      conv_states[i]       (B, conv_dim, kernel_size)         causal-conv context
      recurrent_states[i]  (B, num_heads, k_head_dim, v_head_dim)  delta-rule memory
  - Both O(1) in prefix length → cheap snapshot at any branch point
  - Full-attention KV cache is O(P) per layer, but only 6 layers
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Thread-local active cache (one per running micro-batch / tree)
# Set in `SSMBranchCache.__enter__`, cleared in `__exit__`.
# ---------------------------------------------------------------------------

_active_cache: threading.local = threading.local()


def _get_active() -> Optional["SSMBranchCache"]:
    return getattr(_active_cache, "cache", None)


# ---------------------------------------------------------------------------
# SSMBranchCache  – per micro-batch object, context-managed
# ---------------------------------------------------------------------------

class SSMBranchCache:
    """Holds branch-point snapshots for one micro-batch / tree traversal.

    Lifecycle (mirrors PrefixGrouper):
        cache = SSMBranchCache(model)
        with cache:
            outputs = ssm_tree_forward(model, tree, cache=cache)
        # cache is cleared on context exit
    """

    def __init__(self, model=None):
        self._snapshots: dict = {}      # node_id → _NodeSnapshot

    # ---- context manager: clear on exit, matching PrefixGrouper lifecycle ----

    def __enter__(self):
        _active_cache.cache = self
        return self

    def __exit__(self, *_):
        self.clear()
        _active_cache.cache = None

    def clear(self):
        """Explicitly release all cached tensors. Call at end of micro-batch."""
        for snap in self._snapshots.values():
            snap.clear()
        self._snapshots.clear()

    # ---- snapshot API ----

    def snapshot(self, node_id: int, dyn_cache) -> "_NodeSnapshot":
        snap = _NodeSnapshot.from_dyn_cache(dyn_cache)
        self._snapshots[node_id] = snap
        return snap

    def get_snapshot(self, node_id: int) -> Optional["_NodeSnapshot"]:
        return self._snapshots.get(node_id)


@dataclass
class _NodeSnapshot:
    """Per-tree-node snapshot: conv_states + recurrent_states + KV cache."""
    conv_states: list      # [layer_idx] → Tensor | None
    recurrent_states: list
    key_cache: list
    value_cache: list

    @classmethod
    def from_dyn_cache(cls, dyn_cache) -> "_NodeSnapshot":
        def _c(lst):
            return [t.clone() if t is not None else None for t in lst]
        return cls(
            conv_states=_c(dyn_cache.conv_states),
            recurrent_states=_c(dyn_cache.recurrent_states),
            key_cache=_c(dyn_cache.key_cache),
            value_cache=_c(dyn_cache.value_cache),
        )

    def restore_to(self, dyn_cache) -> None:
        def _cp(src, dst):
            for i, t in enumerate(src):
                dst[i] = t.clone() if t is not None else None
        _cp(self.conv_states, dyn_cache.conv_states)
        _cp(self.recurrent_states, dyn_cache.recurrent_states)
        _cp(self.key_cache, dyn_cache.key_cache)
        _cp(self.value_cache, dyn_cache.value_cache)

    def clear(self):
        for lst in (self.conv_states, self.recurrent_states, self.key_cache, self.value_cache):
            for i in range(len(lst)):
                lst[i] = None


# ---------------------------------------------------------------------------
# Monkey-patch: Qwen3_5GatedDeltaNet.forward with initial-state support
# ---------------------------------------------------------------------------

_patched = False


def apply_ssm_patch() -> None:
    """Patch Qwen3_5GatedDeltaNet.forward to support SSMBranchCache injection.

    The patch is transparent when no SSMBranchCache is active (normal path).
    Idempotent — safe to call multiple times.
    """
    global _patched
    if _patched:
        return

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            apply_mask_to_padding_states,
        )
    except ImportError:
        raise ImportError("transformers with qwen3_5 required")

    _orig_gdn_fwd = Qwen3_5GatedDeltaNet.forward

    def _patched_fwd(
        self,
        hidden_states: Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        # PrefixGrouper-style: caller may pass ssm_initial as kwarg;
        # also fall back to thread-local active cache for backward compat.
        ssm_initial: Optional["_NodeSnapshot"] = None,
        **kwargs,
    ):
        # Resolve initial state: explicit kwarg > thread-local active cache
        snap = ssm_initial
        if snap is None:
            active = _get_active()
            if active is not None:
                snap = getattr(active, "_current_snap", None)

        if snap is None:
            # No branch cache active — original forward
            return _orig_gdn_fwd(
                self, hidden_states, cache_params, cache_position, attention_mask
            )

        init_conv = snap.conv_states[self.layer_idx]
        init_rec  = snap.recurrent_states[self.layer_idx]
        if init_conv is None:
            return _orig_gdn_fwd(
                self, hidden_states, cache_params, cache_position, attention_mask
            )

        # ---- Branch-cache chunk forward ----
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        B, L, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, L)
        z = self.in_proj_z(hidden_states).reshape(B, L, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Prepend kernel_size-1 context from saved conv_state for full receptive field
        ctx_len = self.conv_kernel_size - 1
        ctx = init_conv[:, :, -ctx_len:].to(mixed_qkv)
        full_in = torch.cat([ctx, mixed_qkv], dim=-1)   # (B, conv_dim, ctx_len+L)

        if self.causal_conv1d_fn is not None:
            conv_out = self.causal_conv1d_fn(
                x=full_in,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
            mixed_qkv = conv_out[:, :, ctx_len:]
        else:
            conv_out = F.silu(self.conv1d(full_in))
            mixed_qkv = conv_out[:, :, ctx_len: ctx_len + L]

        # Save new conv_state to cache
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = F.pad(
                mixed_qkv, (self.conv_kernel_size - L, 0)
            )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, L, -1, self.head_k_dim)
        k = k.reshape(B, L, -1, self.head_k_dim)
        v = v.reshape(B, L, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            f = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(f, dim=2)
            k = k.repeat_interleave(f, dim=2)

        core_out, last_rec = self.chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=init_rec.to(q) if init_rec is not None else None,
            output_final_state=(cache_params is not None),
            use_qk_l2norm_in_kernel=True,
        )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_rec

        core_out = self.norm(
            core_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        return self.out_proj(core_out.reshape(B, L, -1))

    Qwen3_5GatedDeltaNet.forward = _patched_fwd
    _patched = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_model(model):
    """Return the text backbone sub-model (handles VLM wrapper if present)."""
    # VLM: Qwen3_5ForConditionalGeneration wraps text in .language_model
    # Text-only: Qwen3_5ForCausalLM wraps text backbone in .model
    if hasattr(model, "language_model"):
        return model.language_model.model
    return model.model   # Qwen3_5TextModel inside Qwen3_5ForCausalLM


def _make_dyn_cache(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    cfg = getattr(model, "config", None)
    return Qwen3_5DynamicCache(cfg)


def _run_segment(
    model,
    token_ids: Tensor,      # (1, L)
    dyn_cache,              # Qwen3_5DynamicCache, updated in-place
    snap: Optional[_NodeSnapshot] = None,  # initial states (multi-token chunk path only)
) -> Tensor:
    """Forward token_ids through the text backbone, returning last_hidden_state (1,L,H).

    Uses past_key_values (the Qwen3.5 cache API).
    cache_position is auto-computed from dyn_cache.get_seq_length().

    snap is used for the patched multi-token chunk path (approximate).
    For the exact single-token decode path, snap is not needed — the restored
    dyn_cache states are used automatically by the original GDN forward.
    """
    tm = _text_model(model)

    active = _get_active()
    if active is not None and snap is not None:
        active._current_snap = snap
    try:
        with torch.no_grad():
            out = tm(
                input_ids=token_ids,
                past_key_values=dyn_cache,
                use_cache=True,
                return_dict=True,
            )
    finally:
        if active is not None:
            active._current_snap = None

    return out.last_hidden_state  # (1, L, H)


def _run_tokens_one_by_one(
    model,
    token_ids: Tensor,   # (1, L) — L tokens to process sequentially
    dyn_cache,           # Qwen3_5DynamicCache, updated in-place
) -> Tensor:
    """Process each token individually using the exact decode path.

    For GDN (linear attention) layers, seq_len=1 + has_previous_state=True triggers
    use_precomputed_states=True, which uses the exact recurrent_gated_delta_rule
    instead of the approximate chunk_gated_delta_rule.

    This is the correct mode for branch-cache continuation after a snapshot.
    Returns last_hidden_state of the final token, shape (1, H).
    """
    last_hidden = None
    for t in range(token_ids.shape[1]):
        tok = token_ids[:, t: t + 1]   # (1, 1)
        last_hidden = _run_segment(model, tok, dyn_cache, snap=None)
    return last_hidden  # (1, 1, H)


# ---------------------------------------------------------------------------
# ssm_tree_forward  – main entry point
# ---------------------------------------------------------------------------

def ssm_tree_forward(
    model,
    tree,           # verl.utils.trajectory_tree.TrajectoryTree
    cache: Optional[SSMBranchCache] = None,
    device: str = "cuda",
) -> list[Tensor]:
    """Run tree rollout using SSM branch caching.

    Each internal node is processed once; its SSM state is snapshotted.
    Each leaf runs only its suffix tokens starting from the nearest ancestor.

    Args:
        model:  Qwen3.5 model (text or VLM).
        tree:   TrajectoryTree with token IDs at each node.
        cache:  SSMBranchCache for this tree.  If None, a temporary one is
                created and cleared on return (mirrors PrefixGrouper lifecycle).
        device: target device.

    Returns:
        List of (1, H) last-hidden-state tensors for each leaf, DFS order.
    """
    apply_ssm_patch()

    own_cache = cache is None
    if own_cache:
        cache = SSMBranchCache(model)

    leaf_outputs: list[Tensor] = []

    try:
        cache.__enter__()

        def _process(node, parent_snap: Optional[_NodeSnapshot], node_id: int):
            dyn = _make_dyn_cache(model)
            if parent_snap is not None:
                parent_snap.restore_to(dyn)

            tokens = node.tokens.unsqueeze(0).to(device)   # (1, L)

            if parent_snap is None:
                # Root: chunk/prefill mode — all tokens at once
                hidden = _run_segment(model, tokens, dyn, snap=None)
            else:
                # Non-root: one-token-at-a-time decode mode.
                # restored dyn has has_previous_state=True → seq_len=1 triggers
                # use_precomputed_states=True → exact recurrent_gated_delta_rule.
                hidden = _run_tokens_one_by_one(model, tokens, dyn)

            if node.is_leaf:
                last_h = hidden[:, -1, :] if hidden.dim() == 3 else hidden[:, 0, :]
                leaf_outputs.append(last_h)
            else:
                snap = cache.snapshot(node_id, dyn)
                for i, child in enumerate(node.children):
                    _process(child, snap, node_id * 10 + i + 1)
                # Release snapshot after all children are done.
                # Mirrors PrefixGrouper lifecycle: clear at end of micro-batch.
                del snap
                cache._snapshots.pop(node_id, None)

        _process(tree.root, parent_snap=None, node_id=0)

    finally:
        if own_cache:
            cache.__exit__(None, None, None)

    return leaf_outputs
