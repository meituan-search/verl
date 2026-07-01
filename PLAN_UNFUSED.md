# Plan: MAGI Prefix-Tree in Fused Kernel Path

## Background

The fused kernel path (`use_fused_kernels=True`) uses `linear_cross_entropy` to fuse
the vocab projection + log-prob computation inside `_fused_GPTModel_forward`.
Currently prefix-tree (MAGI) only works on the non-fused path.

## Key insight

After MAGI tree-forward + undispatch + `expand_flat_to_per_sample`, hidden states are
back in **per-sample flat order** — the same order as `labels_rmpad` from standard
`preprocess_thd_engine(labels, need_roll=True)`.
So `linear_cross_entropy` and postprocessing are unchanged.

Attention compute: O(tree_tokens²) → big win for shared-prefix batches.
linear_cross_entropy compute: O(total_raw_tokens) → same as non-prefix-tree.

## Data flow

```
Non-prefix-tree fused (current):
  input_ids (NestedTensor)
    → preprocess_thd_engine → input_ids_rmpad (flat, per-sample)
  labels (NestedTensor)
    → preprocess_thd_engine(need_roll=True) → labels_rmpad (flat, per-sample)
  model(input_ids_rmpad, labels=labels_rmpad)
    decoder → hidden_states (flat, per-sample)
    linear_cross_entropy(hidden_states, labels_rmpad) → logprobs
  postprocess_thd_engine(logprobs) → NestedTensor

MAGI prefix-tree fused (new):
  input_ids (NestedTensor)
    → build_prefix_tree_micro_batch → pt_batch (tree layout + magi_key)
    → get_position_ids(magi_key) → local_indices
    → pt_batch.tree_packed_input_ids[local_indices] → local_input_ids (dispatched)
    → pt_batch.tree_packed_position_ids[local_indices] → local_position_ids
  labels (NestedTensor)
    → preprocess_thd_engine(need_roll=True) → labels_rmpad (flat, per-sample) ← UNCHANGED
  packed_seq_params from preprocess_thd_engine(labels) ← used for postprocess
  model(local_input_ids, position_ids=local_position_ids, labels=labels_rmpad,
        magi_attention_key=pt_batch.magi_key, pt_batch=pt_batch)
    decoder (MAGI attention via prefix_tree_patch_impl patches)
      → hidden_states (local, per-rank)
    undispatch(hidden_states, magi_key) → hidden_states (full tree-packed)
    expand_flat_to_per_sample(hidden_states, pt_batch) → hidden_states (per-sample flat)
    linear_cross_entropy(hidden_states, labels_rmpad) → logprobs ← UNCHANGED
  postprocess_thd_engine(logprobs, packed_seq_params, labels, ...) ← UNCHANGED
```

## Files to change

### 1. `verl/models/mcore/model_forward_fused.py`

**`fused_forward_model_engine_inner`**: add prefix-tree detection and routing.

```python
# After checking use_prefix_tree flag:
if use_prefix_tree and prefix_tree_attention == "magi":
    from magi_attention.api import get_position_ids
    from verl.utils.prefix_tree.magi import build_prefix_tree_micro_batch

    pt_batch = build_prefix_tree_micro_batch(
        model, input_ids, attention_type="magi",
        tp_size=..., cp_size=...
    )
    if pt_batch is not None:
        local_indices = get_position_ids(pt_batch.magi_key)
        local_input_ids = pt_batch.tree_packed_input_ids[local_indices].unsqueeze(0)
        local_position_ids = pt_batch.tree_packed_position_ids[local_indices].unsqueeze(0)

        # labels: standard preprocess — per-sample flat order matches expand output
        labels_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            labels, pre_process=True, need_roll=True, ...
        )

        output_orig = model(
            input_ids=local_input_ids,
            attention_mask=None,
            position_ids=local_position_ids,
            packed_seq_params=None,          # MAGI handles packing
            labels=labels_rmpad,
            temperature=temperature,
            magi_attention_key=pt_batch.magi_key,
            pt_batch=pt_batch,
            **model_kwargs,
        )
        # postprocess same as non-prefix-tree (logprobs already per-sample flat)
        log_probs = postprocess_thd_engine(output_orig.log_probs, packed_seq_params, labels, ...)
        ...
        return output
```

**`_fused_GPTModel_forward`**: add `magi_attention_key=None, pt_batch=None` params,
add MAGI branch after decoder.

```python
def _fused_GPTModel_forward(model, input_ids, position_ids, ...,
                             magi_attention_key=None, pt_batch=None, ...):
    ...
    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        packed_seq_params=packed_seq_params,
        **extra_block_kwargs,
        **kwargs,
    )
    # MAGI: hidden_states is per-rank local → undispatch → tree-packed → per-sample
    if magi_attention_key is not None and post_process:
        from magi_attention.api import undispatch
        from verl.utils.prefix_tree.magi import expand_flat_to_per_sample
        hidden_states = undispatch(hidden_states.squeeze(0), magi_attention_key).unsqueeze(0)
        hidden_states = expand_flat_to_per_sample(hidden_states.squeeze(0), pt_batch).unsqueeze(0)
    # labels is per-sample flat (from preprocess_thd_engine) — shapes now match ✓
    logprobs, entropy = linear_cross_entropy(hidden_states, output_weight, labels, ...)
    ...
```

### 2. `verl/workers/engine/megatron/transformer_impl.py`

Pass `use_prefix_tree` and `prefix_tree_attention` into the fused forward call
(currently only `labels`, `temperature`, `calculate_entropy`, `pad_token_id` are passed).

```python
fused_forward_fn = get_mcore_forward_fused_model_engine_fn(...)
output = fused_forward_fn(
    model=model,
    input_ids=input_ids,
    labels=label,
    multi_modal_inputs=multi_modal_inputs,
    temperature=temperature_value,
    calculate_entropy=calculate_entropy,
    pad_token_id=...,
    use_prefix_tree=use_prefix_tree,              # ← new
    prefix_tree_attention=prefix_tree_attention,  # ← new
    prefix_tree_subtree=...,                      # ← new (cached trie)
)
```

### 3. `fused_forward_model_engine` signature

Update the inner function to accept the new prefix-tree params.

## Open questions

- **PP intermediate stages**: when `not post_process`, return raw local hidden states
  (undispatch + expand not needed, just pass through as intermediate PP tensor).
- **CP=1**: `get_position_ids` / dispatch is a no-op; tree_packed_input_ids == local.
- **Fallback**: if `build_prefix_tree_micro_batch` returns None (no shared prefix),
  fall through to standard fused path unchanged.
- **labels alignment**: `expand_flat_to_per_sample` outputs in original sample index order.
  `preprocess_thd_engine(labels)` also packs in original sample order. They match by construction.
