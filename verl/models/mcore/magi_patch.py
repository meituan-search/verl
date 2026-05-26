"""Backward-compatibility shim — use prefix_tree_merge instead."""
from verl.models.mcore.prefix_tree_merge import (  # noqa: F401
    apply_prefix_tree_patch,
    apply_magi_patch,
    _magi_attn_forward,
    _flex_attn_forward,
    _layer_capture,
    _layer_counter,
)
