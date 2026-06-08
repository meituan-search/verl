"""
Megatron-LM GatedDeltaNet SSM cache patch.

Patches the two kernel calls in GatedDeltaNet.forward:
  - causal_conv1d (line ~427): initial_state=None → cached conv state
  - self.gated_delta_rule (line ~481): initial_state=None → cached recurrent state

Both output_final_state=False → True when capturing state.

Backward (diamond DAG):
  With n branches sharing prefix A, run:
    _, s_A = ... output_final_state=True   # A forward, s_A in graph
    s_A_leaf = s_A.detach().requires_grad_(True)
    for B, C in branches:
        out = model(initial_state=s_A_leaf)
    (loss_B + loss_C).backward()           # accumulates s_A_leaf.grad
    s_A.backward(s_A_leaf.grad)            # chains to A's params

Verified by test_ssm_split_fwd_bwd.py (fwd diff=0, bwd diff<1e-4).

Usage:
    apply_gdn_cache_patch()    # once at startup
    cache = GDNStateCache()

    with cache.capture():
        model(prefix_tokens)   # states saved

    with cache.inject():
        model(suffix_B_tokens) # states injected as initial_state
"""

import threading
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Thread-local active cache
# ---------------------------------------------------------------------------

_tl = threading.local()

def _get_active() -> Optional["GDNStateCache"]:
    return getattr(_tl, "cache", None)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class GDNLayerState:
    conv_state:      Tensor   # (B, conv_dim, kernel_size-1)  — causal_conv1d final state
    recurrent_state: Tensor   # (B, num_heads, k_dim, v_dim)  — delta-rule recurrent state


# ---------------------------------------------------------------------------
# Cache object — one per micro-batch, context-managed
# ---------------------------------------------------------------------------

class GDNStateCache:
    """SSM state cache for Megatron GDN layers.

    Modes:
      capture: run model forward, save final (conv, recurrent) state per GDN layer
      inject:  run model forward, restore saved states as initial_state

    Lifecycle: create per micro-batch; cleared on __exit__ (PrefixGrouper pattern).
    """

    def __init__(self):
        self.states: dict[int, GDNLayerState] = {}
        self._mode: Optional[str] = None

    def capture(self):   return _Ctx(self, "capture")
    def inject(self):    return _Ctx(self, "inject")

    def clear(self):
        for st in self.states.values():
            del st.conv_state, st.recurrent_state
        self.states.clear()
        self._mode = None

    def __enter__(self):
        _tl.cache = self
        return self

    def __exit__(self, *_):
        self.clear()
        if getattr(_tl, "cache", None) is self:
            _tl.cache = None


class _Ctx:
    def __init__(self, cache, mode): self._c, self._m = cache, mode
    def __enter__(self): _tl.cache = self._c; self._c._mode = self._m; return self._c
    def __exit__(self, *_): self._c._mode = None; _tl.cache = None


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

_patched = False


def apply_gdn_cache_patch():
    """Idempotent patch of Megatron GatedDeltaNet.forward."""
    global _patched
    if _patched:
        return

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    import megatron.core.ssm.gated_delta_net as _mod

    _orig = GatedDeltaNet.forward

    def _fwd(self, hidden_states, attention_mask,
             inference_context=None, packed_seq_params=None,
             sequence_len_offset=None, *, inference_params=None, **kw):

        cache = _get_active()
        if cache is None or cache._mode is None:
            return _orig(self, hidden_states, attention_mask,
                         inference_context=inference_context,
                         packed_seq_params=packed_seq_params,
                         sequence_len_offset=sequence_len_offset,
                         inference_params=inference_params, **kw)

        mode     = cache._mode
        idx      = self.layer_number
        do_cap   = (mode == "capture")
        st       = cache.states.get(idx)
        init_conv = st.conv_state      if st else None
        init_rec  = st.recurrent_state if st else None

        # --- patch causal_conv1d for this call ---
        orig_conv = _mod.causal_conv1d
        _conv_out = [None]

        def _conv_hook(x, weight, bias=None, residual=None,
                       initial_state=None, output_final_state=False,
                       activation=None, backend=None, cu_seqlens=None,
                       cu_seqlens_cpu=None, chunk_indices=None,
                       cp_context=None, **extra):
            result = orig_conv(x, weight, bias,
                               initial_state=init_conv,
                               output_final_state=do_cap,
                               activation=activation, backend=backend,
                               cu_seqlens=cu_seqlens, **extra)
            if do_cap and isinstance(result, tuple):
                _conv_out[0] = result[1]   # final conv state
                result = result[0]
            return result

        # --- patch self.gated_delta_rule for this call ---
        orig_gdr = self.gated_delta_rule
        _rec_out = [None]

        def _gdr_hook(*args, initial_state=None, output_final_state=False, **gkw):
            r = orig_gdr(*args, initial_state=init_rec,
                         output_final_state=do_cap, **gkw)
            if do_cap and isinstance(r, tuple):
                _rec_out[0] = r[1]   # final recurrent state
            return r

        _mod.causal_conv1d      = _conv_hook
        self.gated_delta_rule   = _gdr_hook
        try:
            out = _orig(self, hidden_states, attention_mask,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        inference_params=inference_params, **kw)
        finally:
            _mod.causal_conv1d    = orig_conv
            self.gated_delta_rule = orig_gdr

        if do_cap and (_conv_out[0] is not None or _rec_out[0] is not None):
            cache.states[idx] = GDNLayerState(
                conv_state=_conv_out[0],
                recurrent_state=_rec_out[0],
            )

        return out

    GatedDeltaNet.forward = _fwd
    _patched = True
    print("[GDNCache] Megatron GatedDeltaNet patched")
