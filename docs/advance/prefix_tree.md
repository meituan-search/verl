# Prefix-Tree (MAGI) Attention for Shared-Prefix Training

**Author:** `https://github.com/meituan-search`

Last updated: 07/17/2026.

This document introduces a prefix-deduplicated attention system (MAGI) for RL/SFT training of LLMs that share long prompt prefixes across rollouts (e.g. GRPO, multi-turn agents, long-context reasoning). It packs sequences with shared prefixes into a flat layout where shared tokens are processed once, reducing both compute and memory proportional to the shared-prefix length.

## 1. Background

In GRPO and multi-turn RL, the same prompt is rolled out `n` times (rollout.n), producing `n` responses that share an identical prompt prefix. Standard attention processes the prompt tokens `n` times — once per rollout — wasting compute proportional to `prompt_len * (n - 1)`. For long-context reasoning (8k–128k prompt tokens, `n=8`), this overhead dominates training cost.

The prefix-tree (MAGI) system builds a compressed trie over the batch's sequences, identifies shared prefixes, and packs them into a flat token layout where shared tokens appear once. A custom attention kernel ([magi_attention](https://github.com/magi-attention/magi-attention)) computes attention over this packed layout using a block-sparse mask derived from the trie structure, so each rollout attends to its own prompt tokens without duplicating the forward pass.

## 2. When to Use

Enable prefix-tree when:

- Training with GRPO (`rollout.n >= 2`) — each prompt produces multiple responses sharing the prompt prefix.
- Multi-turn agent training — accumulated tool-call context is shared across turns.
- Long-context reasoning — prompt lengths >= 8k where the prompt dominates the sequence.
- Models with large vocab + long shared system prompts.

Do **not** enable when:

- Each sample has a unique prompt (no sharing) — the trie overhead is pure cost.
- `rollout.n == 1` and no multi-turn accumulation — no prefix sharing to exploit.

## 3. Configuration

Prefix-tree is controlled by two fields under `actor_rollout_ref.model`:

| Field | Default | Values | Description |
|-------|---------|--------|-------------|
| `use_prefix_tree` | `false` | `true` / `false` | Enable prefix-tree trie build + packed layout |
| `prefix_tree_attention` | `magi` | `magi` / `flex` | Attention backend. `magi` uses the magi_attention kernel; `flex` uses `torch.nn.attention.flex_attention` with a block mask. |

Example:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.use_prefix_tree=True \
    actor_rollout_ref.model.prefix_tree_attention=magi \
    ... # other config
```

For SFT, the same fields are set under `actor.model` in the SFT trainer config.

### Backend selection

- **`magi`** (default): uses the `magi_attention` package's distributed attention kernel. Supports TP + CP + SP. Required for the highest sharing ratios (cross-CP-rank attention). Falls back to FA3 if the magi key is not available.
- **`flex`**: uses `torch.nn.attention.flex_attention` with a block-sparse mask derived from the trie. Simpler dependency, no custom kernel. Good for CPU testing or when `magi_attention` is not installed.

## 4. Architecture

### Data flow

```
Trainer (RayPPOTrainer / SFTTrainer)
  ├─ attach_segment_metadata()
  │    └─ segment_hashes + segment_lengths from prompt UIDs + prompt lengths (GRPO fast path)
  ├─ build_global_trie()
  │    ├─ build_global_tree_from_segments (fast path) OR greedy_build_tries (fallback)
  │    └─ attaches: batch.meta_info["prefix_tree"] (TrieNode root)
  │                 batch.non_tensor_batch["leaf_idx"] (sample -> leaf flat_idx)
  ├─ pt_metrics()  ─ global_shared_ratio, packed_tokens, raw_tokens
  ├─ _balance_batch()  ─ DFS reorder for prefix locality (leaf_idx stays correct)
  └─ dispatch to DP ranks
       └─ prepare_prefix_tree_micro_batches()  (engine/utils.py)
            ├─ mbs_groups_from_leaf_idx()  ─ prefix-aware micro-batch grouping
            └─ create_and_attach_subtrie_views()  ─ per-micro-batch PrefixSubTrie
                 └─ forward()  (forward.py)
                      ├─ build_prefix_tree_batch()  ─ PrefixTreeMagiBatch (flat tokens, attn ranges)
                      ├─ _build_magi_key()  ─ magi_attention_key (global shapes + CP coordination)
                      ├─ dispatch_magi()  ─ slice to per-CP-rank local tokens
                      ├─ model(...) with patched TEDotProductAttention
                      │    └─ magi_attn_forward() -> calc_attn()  (magi_attention kernel)
                      ├─ undispatch()  ─ gather local logits back to full layout
                      └─ loss  (linear cross-entropy on packed labels)
```

### Trie -> packed layout

```
Samples (shared prompt P, responses R0..R3):
  [P R0]  [P R1]  [P R2]  [P R3]

Compressed trie:
  root
   └─ P (shared prefix, 1 node)
       ├─ leaf: R0
       ├─ leaf: R1
       ├─ leaf: R2
       └─ leaf: R3

Flat packed layout (tokens processed once):
  ┌─────┬─────┬─────┬─────┬─────┐
  │  P  │ R0  │ R1  │ R2  │ R3  │
  └─────┴─────┴─────┴─────┴─────┘
   shared   each response attends to P + itself
   (1x)     via block-sparse mask from trie structure

  Without prefix-tree: P processed 4x (once per rollout)
  With prefix-tree:    P processed 1x (shared node)
```

### Key components

**Data structures** (`verl/utils/prefix_tree/tree.py`):
- `TrieNode`: compressed trie node with `flat_idx`, `input_ids`, `ancestor`, `children`.
- `PrefixTrie`: full batch view with flat `nodes` list indexed by `flat_idx`.
- `PrefixSubTrie`: per-micro-batch serializable view (`__getstate__`/`__setstate__` for pickle across PP ranks).

**Layout building** (`verl/utils/prefix_tree/utils.py`):
- `build_layout_from_tree_node()`: walks the trie, assigns flat offsets, emits attention ranges (`q_ranges`, `k_ranges`, `mask_types`).
- `PrefixTreeParams`: holds the packed layout — flat tokens, attention spec rectangles, leaf-to-sample mapping.
- Pre-packed labels avoid `torch.roll` cross-boundary bugs at group edges.

**MAGI integration** (`verl/utils/prefix_tree/magi.py`):
- Dispatch happens **once per forward** (not per-layer) via `dispatch_magi`.
- `get_position_ids(magi_key)` returns local token indices for the CP rank.
- The model receives pre-dispatched `local_input_ids` / `local_position_ids`.
- `undispatch()` gathers local logits back to the full layout for loss computation.

**Attention patch** (`verl/utils/prefix_tree/prefix_tree_patch_impl.py`):
- Patches `TEDotProductAttention.forward` to add MAGI/flex branches.
- `magi_attn_forward()` calls `calc_attn()` from the `magi_attention` package.
- Falls back to FA3 if neither MAGI nor flex key is provided.

**Forward drivers** (`verl/utils/prefix_tree/forward.py`):
- `unfuse_try_forward_prefix_tree` / `fuse_try_forward_prefix_tree`: entry points for unfused and fused forward paths.
- `_build_magi_key` / `_build_flex_key`: build the attention key from model config and the trie.
- `fuse_forward_body`: the fused-path body that wires RoPE + decoder-key contexts + the model call.

**Micro-batch grouping** (`verl/utils/prefix_tree/dynamic.py`):
- `mbs_groups_from_leaf_idx`: groups samples into prefix-aware micro-batches using the reorder-safe `leaf_idx` (not the stale `sequence_ids`).
- `prepare_prefix_tree_micro_batches`: splits the batch into micro-batches and attaches subtrie views.

**Trainer helpers** (`verl/utils/prefix_tree/trainer.py`):
- `attach_segment_metadata` + `build_global_trie`: build the global trie at the trainer level (before DP dispatch).
- `pt_metrics`: compute prefix-sharing metrics (`global_shared_ratio`, `micro_batch_shared_ratio`, `packed_tokens`, `raw_tokens`, `timing_s`).

## 5. Reorder-safety: `leaf_idx` is the source of truth

The trie's `sequence_ids` and `leaves[]` are indexed by **original** sample position and go stale after `DataProto.reorder` (inside `_balance_batch`). `leaf_idx` (a numpy array in `non_tensor_batch`) is fancy-indexed by reorder automatically and stays correct.

- `build_global_trie` attaches both `meta_info["prefix_tree"]` (trie) and `non_tensor_batch["leaf_idx"]` (sample → leaf flat_idx).
- `prepare_prefix_tree_micro_batches` calls `mbs_groups_from_leaf_idx` (NOT `mbs_groups_from_trie`) when a trie is attached.
- `create_and_attach_subtrie_views` reads `leaf_idx` from each micro-batch's `non_tensor_batch` to build the subtrie view.
- All `leaf_idx`-based paths raise `ValueError` on `-1` entries (sample without leaf) — no silent skips.

The sort order is **tree-then-sort**: build the trie, then `_balance_batch` reorders for DP balance. The `leaf_idx`-driven grouping makes this safe.

## 6. Metric aggregation

`engine_workers._postprocess_output` aggregates metrics via `allgather_dict_into_dict` (wraps each value as `[val]` per DP rank) + `chain.from_iterable` (flattens lists-of-lists). Raw floats wrapped as `[float]` crash `chain.from_iterable` because floats aren't iterable.

- Metrics added **before** allgather must be wrapped in `Metric(value, aggregation=...)` so `Metric.aggregate_dp` handles them.
- Metrics added **after** allgather (`loss`, `grad_norm`, `lr`, `mfu`, `perf/*`) stay scalar and bypass the list branch.
- `prefix_tree/attn_fa3_fallback_ratio` uses `Metric(MEAN)`. The counter tracks only `fa3` and `total` (the magi/flex distinction is dropped; only the FA3 fallback ratio matters).

## 7. VLM-config models (Qwen3.5)

For text-only prefix-tree on VLM-config models (e.g. Qwen3.5), `vision_model = hasattr(hf_config, "vision_config")` at both call sites in `transformer_impl.py`. Do **not** add `and "pixel_values" in multi_modal_inputs`; that breaks the standard (non-prefix-tree) path for Qwen3.5 because `vision_model=True` is needed to trigger the VLM code path that handles M-RoPE's 3D `position_ids` internally.

The `has_vision_data` check lives only at the fused path's prefix-tree guard: `if use_prefix_tree and not (vision_model and has_vision_data):` (in `model_forward_fused.py`).

## 8. Tree-builder diagnostics

When `PrefixTreeParams.__post_init__` raises `last leaf range must end at total sequence length`, the cause is `_assign_offsets` (uses `len(node.input_ids)`) disagreeing with `_emit` (slices actual samples). The per-node trace log (emitted in `build_layout_from_tree_node` when `last_leaf_end != total_tokens`) prints each node's `flat_idx`, `iid_len`, `start`, `end`, `donated_in`, `donated_out`, `n_children`.

Invariant per node: `end - start == (1 if donated_in else 0) + iid_len - (1 if donated_out else 0)`. Any node violating this is the bug.

The OLP crash `no_padding_2_padding: token count mismatch` is a separate symptom: `sum(prompt_len + response_len) != actual_tokens_in_output`. The diagnostic log prints `prompt_lens`, `response_lens`, `sequence_lens` lists to identify which sample is short.

## 9. Key files

| File | Role |
|------|------|
| `verl/utils/prefix_tree/tree.py` | Trie data structures (`TrieNode`, `PrefixTrie`, `PrefixSubTrie`) |
| `verl/utils/prefix_tree/dynamic.py` | Trie construction, micro-batch grouping, metrics, DP partitioning |
| `verl/utils/prefix_tree/utils.py` | Layout builder (`build_layout_from_tree_node`), `PrefixTreeParams` |
| `verl/utils/prefix_tree/magi.py` | `PrefixTreeMagiBatch`, dispatch, RoPE/key contexts |
| `verl/utils/prefix_tree/forward.py` | Unfused/fused forward drivers, magi/flex key builders |
| `verl/utils/prefix_tree/prefix_tree_patch_impl.py` | Megatron attention patches (`magi_attn_forward`, `flex_attn_forward`) |
| `verl/utils/prefix_tree/segment_grouper.py` | Fast hash-based segment metadata for GRPO |
| `verl/utils/prefix_tree/trainer.py` | Trainer-facing helpers (`pt_metrics`, `build_global_trie`, `attach_segment_metadata`) |

## 10. Dependencies

- `magi_attention` package (for the `magi` backend). Install from [magi-attention/magi-attention](https://github.com/magi-attention/magi-attention).
- Megatron-LM (for the attention patch). The patch targets `TEDotProductAttention` from `megatron.core.transformer.attention`.
- The `flex` backend has no external kernel dependency (uses `torch.nn.attention.flex_attention`).
