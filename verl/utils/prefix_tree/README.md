# Prefix-Tree (MAGI) Attention

The prefix-tree attention system (MAGI) enables prefix-deduplicated training for LLMs. It packs multiple sequences with shared prefixes into a flat layout where shared tokens are processed once.

### Key Components

**Data Structures** (`verl/utils/prefix_tree/tree.py`):
- `TrieNode`: compressed trie node with `flat_idx`, `input_ids`, `ancestor`, `children`
- `PrefixTrie`: full batch view with flat `nodes` list indexed by `flat_idx`
- `PrefixSubTrie`: per-micro-batch view, serializable via `__getstate__`/`__setstate__`

**Layout Building** (`verl/utils/prefix_tree/utils.py`):
- `build_layout_from_tree_node()`: builds flat token layout and attention ranges
- `PrefixTreeParams`: holds `q_ranges`, `k_ranges`, `mask_types`, packed tensors
- Pre-packed labels avoid `torch.roll` cross-boundary issues

**MAGI Integration** (`verl/utils/prefix_tree/magi.py`):
- Dispatch happens **once per forward** (not per-layer)
- `get_position_ids(magi_key)` returns local token indices for CP rank
- Model receives pre-dispatched `local_input_ids` / `local_position_ids`
- `undispatch()` gathers local logits back to full layout for loss computation

**Attention Patch** (`verl/utils/prefix_tree/prefix_tree_patch_impl.py`):
- Patches `TEDotProductAttention.forward` to add MAGI/flex branches
- `magi_attn_forward()` calls `calc_attn()` from `magi_attention` package
- Fallback to FA3 if neither MAGI nor flex key is provided

**Trainer Helpers** (`verl/utils/prefix_tree/trainer.py`):
- `pt_metrics()`: computes prefix-sharing metrics if `use_prefix_tree` is enabled
- `apply_engine_config()` / `add_meta_info()`: thread prefix-tree flags into engine config and meta-info

**Dynamic Trie** (`verl/utils/prefix_tree/dynamic.py`):
- `greedy_build_tries()`: builds compressed tries from sequences
- `mbs_groups_from_leaf_idx()` / `trie_dfs_leaf_order_from_leaf_idx()`: reorder-safe micro-batch grouping
- `prepare_prefix_tree_micro_batches()`: splits batch into prefix-aware micro-batches

**Segment Grouper** (`verl/utils/prefix_tree/segment_grouper.py`):
- `create_grpo_segment_metadata()`: fast hash-based segment metadata for GRPO
- `group_by_segment_hash()`: groups samples by shared prefix segment

**Forward Path** (`verl/utils/prefix_tree/forward.py`):
- `unfuse_forward_prefix_tree()` / `fuse_try_forward_prefix_tree()`: unfused and fused forward entry points
- `_build_magi_key()`: builds the MAGI attention key from model config and trie

### Configuration

```bash
# Enable prefix-tree with MAGI attention
actor_rollout_ref.model.use_prefix_tree=True
actor_rollout_ref.model.prefix_tree_attention=magi  # or "flex"
```

### Key Files

- `verl/utils/prefix_tree/magi.py`: main forward path
- `verl/utils/prefix_tree/tree.py`: data structures
- `verl/utils/prefix_tree/utils.py`: layout builder
- `verl/utils/prefix_tree/forward.py`: unfused/fused forward drivers
- `verl/utils/prefix_tree/dynamic.py`: trie build + micro-batch grouping
- `verl/utils/prefix_tree/segment_grouper.py`: fast segment metadata
- `verl/utils/prefix_tree/trainer.py`: trainer-facing helpers
- `verl/utils/prefix_tree/prefix_tree_patch_impl.py`: Megatron patches

### Reorder-safety: `leaf_idx` is the source of truth

The trie's `sequence_ids` and `leaves[]` are indexed by **original** sample position and go stale after `DataProto.reorder` (inside `_balance_batch`). `leaf_idx` (numpy array in `non_tensor_batch`) is fancy-indexed by reorder automatically and stays correct.

- `_build_global_trie` (ray_trainer.py) attaches both `meta_info["prefix_tree"]` (trie) and `non_tensor_batch["leaf_idx"]` (sample to leaf flat_idx).
- `prepare_prefix_tree_micro_batches` calls `mbs_groups_from_leaf_idx` / `trie_dfs_leaf_order_from_leaf_idx` (NOT the `*_from_trie` variants) when a trie is attached.
- `create_and_attach_subtrie_views` reads `leaf_idx` from each microbatch's `non_tensor_batch` to build the subtrie view.
- All `leaf_idx`-based paths raise `ValueError` on `-1` entries (sample without leaf), no silent skips.

**Sort order is tree then sort** (build trie, then `_balance_batch` reorders). The `leaf_idx`-driven grouping makes this safe.

### Metric aggregation: wrap floats in `Metric`

`engine_workers._postprocess_output` aggregates metrics via `allgather_dict_into_dict` (wraps each value as `[val]` per DP rank) + `chain.from_iterable` (flattens lists-of-lists). Raw floats wrapped as `[float]` crash `chain.from_iterable` because floats aren't iterable.

- Metrics added BEFORE allgather must be wrapped in `Metric(value, aggregation=...)` so `Metric.aggregate_dp` handles them.
- Metrics added AFTER allgather (`loss`, `grad_norm`, `lr`, `mfu`, `perf/*`) stay scalar and bypass the list branch.
- `prefix_tree/attn_fa3_fallback_ratio` uses `Metric(MEAN)`. Counter tracks only `fa3` and `total` (magi/flex distinction dropped; only FA3 fallback ratio matters).

### VLM-config models (Qwen3.5): `vision_model` gating

`vision_model=hasattr(hf_config, "vision_config")` at both call sites in `transformer_impl.py`. Do NOT add `and "pixel_values" in multi_modal_inputs`; that breaks the standard (non-prefix-tree) path for Qwen3.5 because `vision_model=True` is needed to trigger the VLM code path that handles M-RoPE's 3D `position_ids` internally.

The `has_vision_data` check lives only at the fused path's prefix-tree guard: `if use_prefix_tree and not (vision_model and has_vision_data):` (model_forward_fused.py).

### Tree-builder diagnostics

When `PrefixTreeParams.__post_init__` raises `last leaf range must end at total sequence length`, the cause is `_assign_offsets` (uses `len(node.input_ids)`) disagreeing with `_emit` (slices actual samples). The per-node trace log (added in `build_layout_from_tree_node` when `last_leaf_end != total_tokens`) prints each node's `flat_idx`, `iid_len`, `start`, `end`, `donated_in`, `donated_out`, `n_children`.

Invariant per node: `end - start == (1 if donated_in else 0) + iid_len - (1 if donated_out else 0)`. Any node violating this is the bug.

The OLP crash `no_padding_2_padding: token count mismatch` is a separate symptom: `sum(prompt_len + response_len) != actual_tokens_in_output`. Diagnostic log prints `prompt_lens`, `response_lens`, `sequence_lens` lists to identify which sample is short.
