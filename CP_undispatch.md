# CP Reorder: Move CE before undispatch in tree path

## Goal

Move the CP `all_gather` from the 2D logits/hidden tensor (GB-scale) to the 1D log_probs
result (KB), matching the FA3 path's CP efficiency. This eliminates 4x redundant
CE compute (CP=4) and the large gather.

## Background

### Current unfused tree path (unfuse_forward_prefix_tree, magi.py)

```
1. model(local_input_ids, magi_key)          → CP-local logits (1, local_tokens, vocab)
2. undispatch(logits, magi_key)              → full logits (1, flat_tokens, vocab)  [CP GATHER 3.6GB]
3. logits_flat = output[:, :real_tokens]     → (flat_tokens, vocab)
4. logits_processor(logits_flat, labels, temp) → CE on FULL logits [4x redundant per rank]
5. expand_flat_to_per_sample(log_probs)      → per-sample flat (total_expanded,)
6. as_nested_tensor                          → NestedTensor output
```

### Current fused tree path (fuse_try_forward_prefix_tree + fuse_forward_body, magi.py)

After the fix in this session (flat LCE instead of expand-then-LCE):

```
1. model(local_input_ids, magi_key)          → CP-local hidden (1, local_tokens, hidden)
   [inside _fused_GPTModel_forward → fuse_forward_body]
2. undispatch(hidden, magi_key)              → flat hidden (1, real_tokens, hidden)  [CP GATHER ~160MB]
3. LCE(flat_hidden, output_weight, flat_labels) → flat log_probs (real_tokens,)
4. expand_flat_to_per_sample(log_probs)      → per-sample flat (total_expanded,)
5. as_nested_tensor                          → NestedTensor output
```

Before this session's fix, fuse_undispatch_and_expand_hidden did undispatch → expand to
`total_expanded` (prefix replicated per sample), then LCE on the expanded hidden. LCE was
operating on `~1.78x` more tokens than necessary. Now LCE runs on `real_tokens` (deduped flat),
and expansion happens cheaply on the 1D log_probs result afterward.

### FA3 path (model_forward.py)

```
1. model(input_ids_rmpad, ...)               → CP-local logits (1, local_tokens, vocab)
2. logits_processor(local_logits, ...)       → CE on CP-LOCAL logits [1x per rank]
3. postprocess_thd_engine(log_probs)         → CP all_gather on 1D log_probs [KB]
```

## Proposed unfused tree path (reordered — PENDING)

```
1. model(local_input_ids, magi_key)          → CP-local logits (1, local_tokens, vocab)
2. logits_processor(local_logits, local_labels, local_temp) → CE on CP-LOCAL [1x per rank]
3. undispatch(log_probs, magi_key)           → CP gather on 1D log_probs [KB]
4. undispatch(entropy, magi_key)             → CP gather on 1D entropy [KB]
5. expand_flat_to_per_sample(gathered)       → per-sample flat
6. as_nested_tensor                          → NestedTensor output
```

## Proposed fused tree path (further optimization — PENDING)

Similarly, the fused path can move LCE before undispatch:

```
1. model(local_input_ids, magi_key)          → CP-local hidden (1, local_tokens, hidden)
2. LCE(local_hidden, output_weight, local_labels, local_temp)  → local log_probs [1x per rank]
3. undispatch(local_log_probs, magi_key)     → CP gather on 1D [KB]
4. expand_flat_to_per_sample(gathered)       → per-sample flat
5. as_nested_tensor                          → NestedTensor output
```

For CP=4 and hidden=8192, the undispatch data shrinks from ~160MB (hidden gather) to ~64KB (log_probs gather).

## Correctness

Proven equivalent by test (`test_e2e_ce_to_loss.py`):
- CE is per-token (no cross-token dependency)
- `undispatch` is a dim-0 gather-scatter (works on any tensor shape, including 1D)
- `CE(undispatch(logits)) == undispatch(CE(logits))` (verified, 0.0 max diff)
- Both paths produce the same full NestedTensor before the GRPO loss
- GRPO loss is identical for the same logits (Path A == Path B in test)

## Changes required for unfused path

### 1. `verl/utils/prefix_tree/magi.py` — `unfuse_forward_prefix_tree`

**Current** (lines ~886-1089):
```python
if post_process:
    output_orig = undispatch(output_orig.squeeze(0), pt_batch.magi_key).unsqueeze(0)
    # ... CE on full logits ...
    output_dict = logits_processor(logits_flat.clone().unsqueeze(1), **flat_args)
    # ... expand ...
```

**Reordered**:
```python
if post_process and logits_processor is not None:
    # 1. Slice labels + temperature to CP-local (same local_indices as input_ids)
    local_indices = get_position_ids(pt_batch.magi_key)
    local_logits = output_orig.squeeze(0)  # already CP-local (local_tokens, vocab)
    local_labels = pt_batch.tree_packed_labels[local_indices].unsqueeze(1)
    local_temp = tree_packed_t[local_indices]  # pre-built full, slice to local
    # 2. CE on CP-local logits
    output_dict = logits_processor(local_logits.clone().unsqueeze(1),
                                   label=local_labels, temperature=local_temp, **other_args)
    # 3. Undispatch 1D outputs across CP
    for key, val in output_dict.items():
        if isinstance(val, torch.Tensor) and val.reshape(-1).shape[0] == local_tokens:
            val_full = undispatch(val.reshape(-1), pt_batch.magi_key)  # gather to full
            val_expanded = expand_flat_to_per_sample(val_full, pt_batch)
            output_dict[key] = as_nested_tensor_from_offsets(val_expanded, cu_seqlen)
```

### 2. CP-local labels and temperature

The labels (`tree_packed_labels`) and temperature (`tree_packed_t`) are currently
built for the full flat layout. They must be sliced to CP-local using the same
`local_indices = get_position_ids(magi_key)` used for `local_input_ids`.

- `local_labels = tree_packed_labels[local_indices]`
- `local_temp = tree_packed_t[local_indices]`

### 3. Undispatch on 1D tensors

`undispatch(x, magi_key)` works on any tensor whose dim-0 is the dispatched
seqlen (verified from source: `mgr.undispatch_qo(x)` + `unpad_at_dim(x, dim=0)`).
For 1D log_probs `(local_tokens,)`, undispatch returns `(flat_tokens,)`.

### 4. Expand + NestedTensor (unchanged)

After undispatch, the 1D log_probs are in tree-packed flat order. The existing
`expand_flat_to_per_sample` + `restore_flat_to_nested` (or the view-based
equivalent) run unchanged on the gathered full tensor.

## Status

| Path | Issue | Status |
|------|-------|--------|
| Unfused: expand-then-LCE | N/A (logits are flat by nature) | — |
| Unfused: CE before undispatch | 3.6 GB logits gather → 1D gather | **PENDING** |
| Fused: expand-before-LCE | 1.78x extra LCE tokens | **FIXED** (this session) |
| Fused: LCE before undispatch | ~160 MB hidden gather → 1D gather | **PENDING** |

## Risk / edge cases

1. **`undispatch` on 1D**: verified safe (dim-0 op). But test on cluster to confirm
   magi runtime handles 1D correctly (the source says it should).

2. **CP-local labels alignment**: `local_indices` must index `tree_packed_labels`
   correctly. Since `tree_packed_labels` is in tree-packed flat order and
   `local_indices` maps local positions → global tree-packed positions, this
   should be correct. Verify with the debug print.

3. **Loss scaling**: proven equivalent (Path A == Path B). The loss aggregation
   uses global `batch_num_tokens` in both cases.

4. **Backward**: `undispatch` with `is_partial_grad=False` (default) selects
   local chunks in backward. Each rank's gradient for its local tokens is
   correct. The reorder doesn't change backward semantics — CE runs locally,
   undispatch gathers forward output, backward scatters gradients back.

## Expected win

For CP=4, 8k prefix, 8 rollouts:

### Unfused path (CE reorder)
- **CE compute**: 4x reduction (each rank does 1/4 the tokens)
- **CP gather data**: 3.6 GB → ~24 KB (~100,000x less)
- **Combined with fused CE**: 42ms × 4 ranks → ~3ms × 4 ranks

### Fused path (LCE reorder, pending)
- **CP gather data**: ~160 MB (hidden) → ~64 KB (log_probs)
- **LCE compute**: 4x reduction (each rank does 1/4 the tokens)

## Implementation order

### Unfused (pending)
1. Add CP-local slicing for labels + temperature
2. Move CE call before undispatch (use local logits)
3. Move undispatch into the output_dict loop (per 1D tensor)
4. Keep expand + NestedTensor unchanged
5. Test on cluster with debug prints, verify loss matches FA3

### Fused (pending)
1. In `fuse_forward_body`: remove the flat-hidden undispatch, return CP-local hidden
2. In `fuse_try_forward_prefix_tree`: do LCE on CP-local hidden with `local_labels`
3. Undispatch 1D log_probs, then expand + split into NestedTensor
