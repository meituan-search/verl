# [RFC] Prefix-Tree Shared Attention for Multi-Turn RL Training

## Summary

This RFC proposes **Prefix-Tree Shared Attention**, a training efficiency optimization that eliminates redundant computation across samples sharing a common token prefix. It is designed for multi-turn RL rollouts where each prompt is sampled `n` times, producing a tree-structured trajectory space with a shared root (the prompt) and diverging leaves (independent responses).

The core idea — originally proposed by [Forge](https://arxiv.org/abs/xxxx) under the name *Prefix Tree Merging* — is to replace the standard "one padded batch per sample" layout with a single flattened `[prefix || leaf_0 || leaf_1 || ... || leaf_{n-1}]` layout, then run attention once using a flex-range or MAGI attention primitive that enforces causal isolation between leaves while allowing all leaves to attend fully to the shared prefix. Following the forward pass, the flat output is reconstructed into per-sample views for standard loss computation — guaranteeing strict mathematical equivalence to a naïve independent-forward baseline.

This implementation extends Forge's original concept to VERL's Megatron-LM backend, adds a PyTorch-native `flex_attention` path requiring no external dependencies, integrates the optimization into both SFT and multi-turn RL training pipelines, and supports **multi-level trees** for deeper prefix sharing across conversation turns.

---

## Motivation

### The Redundancy Problem

In multi-turn RL, the actor generates `n` independent response samples for each prompt. A typical rollout with `n=4` and a prompt of `P` tokens processes:

```
4 × (P + R) tokens per step   (P = prompt length, R = avg response length)
```

When `P >> R` — the common case for long-context agent tasks — the prompt tokens dominate total compute. All four samples compute identical key-value representations for the prefix, paying full O(P²) attention cost four times.

At a 60% prefix ratio (e.g., a 10,240-token sequence where 6,144 tokens are a shared system prompt), the redundant computation exceeds half the total FLOPs per step.

### Forge's Observation

Forge first characterized this redundancy in agent training scenarios and proposed *Prefix Tree Merging* as the solution:

> "By utilizing attention primitives (such as Magi Attention), we ensure that the logical execution remains consistent with a standard forward pass. Following the forward pass, the prefix tree is deconstructed based on metadata to compute the loss normally, ensuring zero impact on downstream logic. By eliminating redundant prefix prefilling, this solution achieves a 40x training speedup and significantly reduces memory overhead."

VERL adopts this principle and extends it across two attention backends and both training paradigms (SFT and RL), with a focus on correctness, TP/PP/CP compatibility, and zero changes to loss computation or metrics.

### Target Use Case: n-Copy Multi-Turn RL

The primary motivating scenario is multi-turn RL with repeated sampling:

- `n_samples = n` per prompt (default: `rollout.n` in VERL PPO config)
- `max_turns = T` per episode
- At each turn `t`, all `n` copies share the conversation history up to turn `t` as a common prefix

This produces a tree-structured trajectory space:

```
                    [shared prompt]
                   /       |       \
             turn_1a   turn_1b   turn_1c    (n=3 copies, turn 1)
            /    \       |       /    \
         t2_aa t2_ab   t2_b  t2_ca  t2_cb   (turn 2)
```

For `n=2` and `max_turns=2`, we have 4 final leaf trajectories sharing a root. With prefix-tree merging, the shared prefix is computed once per step rather than `n` times, reducing training FLOPs by up to `(n-1) × prefix_ratio / n`.

---

## Design Overview

### Architecture

```
VERL Training Loop
│
├── Trainer (SFT or PPO)
│     Reads: data.use_prefix_tree, data.prefix_tree_attention
│     Computes/propagates: prefix_segments (non-tensor batch field)
│
└── Model Forward (gptmodel_forward_model_engine)
      │
      ├── build_prefix_tree_micro_batch()
      │     → PrefixTreeMagiBatch (flat layout + attention key + CP-local tensors)
      │
      └── Megatron GPTModel (patched via prefix_tree_merge.py)
            ├── flex backend: torch.nn.attention.flex_attention
            └── magi backend: magi_attention dispatch/calc/undispatch
```

### Flat Layout

For a micro-batch of `B` samples each of length `S` with a shared prefix of length `P`:

```
Standard layout:  [sample_0: P+R_0] [sample_1: P+R_1] ... [sample_{B-1}: P+R_{B-1}]
                  Total tokens: B × (P + avg_R)

Prefix-tree layout: [prefix: P] [leaf_0: R_0] [leaf_1: R_1] ... [leaf_{B-1}: R_{B-1}]
                    Total tokens: P + sum(R_i)   ← savings = (B-1) × P
```

The attention pattern in flat space:

```
Prefix tokens:       causal self-attention within prefix
Leaf_i → prefix:     full attention (prefix is fully visible to all leaves)
Leaf_i → leaf_i:     causal self-attention within own leaf
Leaf_i → leaf_j:     BLOCKED (cross-leaf isolation)
```

This is mathematically identical to computing full causal attention independently for each sample `[prefix || leaf_i]`.

---

## Data Flow

### 1. Prefix Segment Computation

**SFT path (dataset-level, load time):**

`MultiTurnSFTDataset.__getitem__` computes per-turn prefix boundaries using cumulative hashing:

```python
# For each sub-turn in a multi-turn conversation:
cumulative_ids = torch.cat(all_tokens_so_far)
hash_val = xxhash.xxh128_intdigest(cumulative_ids.numpy().tobytes())
prefix_segments.append((hash_val, len(cumulative_ids)))
```

Result: `prefix_segments = [(h_0, L_0), (h_1, L_1), ...]` — one entry per sub-turn, stored in `non_tensor_batch`.

**RL path (rollout time, per batch):**

After `gen_batch.repeat(n, interleave=True)` creates `n` copies of each prompt, the trainer injects single-turn prefix segments:

```python
gen_batch_output.non_tensor_batch["prefix_segments"] = np.array([
    build_prefix_segments_single_turn(input_ids[i], attention_mask[i])
    for i in range(batch_size)
], dtype=object)
```

Each entry is `[(hash(prompt), prompt_len)]`. With `n` copies of the same prompt, all `n` entries share the same hash → the prefix detector finds a universal prefix of length `prompt_len`.

### 2. Configuration Propagation

`data.use_prefix_tree` and `data.prefix_tree_attention` are propagated from the SFT trainer's data config into the Megatron engine via `McoreEngineConfig.use_prefix_tree` and `prefix_tree_attention` fields (both are `_mutable_fields` to allow post-init injection). `engine_workers.py` injects them into each micro-batch's non-tensor data so they reach `gptmodel_forward_model_engine`.

### 3. Micro-Batch Forward

At model forward time, `gptmodel_forward_model_engine` (THD path):

```
1. Extract use_prefix_tree, prefix_tree_attention from logits_processor_args
2. Resolve prefix_segments (unwrap NonTensorStack if from SFTTensorCollator)
3. build_prefix_tree_micro_batch(model, input_ids, loss_mask, prefix_segments, attention_type, tp_size)
   → PrefixTreeMagiBatch (flat_input_ids, flat_position_ids, flat_loss_mask,
                           local_flat_*, attention key)
4. Forward model with local_flat_input_ids (CP-local slice) and magi/flex key
5. [magi only] undispatch model output to reassemble full flat sequence
6. Strip TP padding (total_tokens → real_tokens)
7. Transpose to THD: (real_tokens, 1, vocab)
8. Shift labels by -1 (next-token prediction on flat layout)
9. logits_processor → log_probs (real_tokens, 1)
10. restore_flat_to_nested(log_probs[:real_tokens]) → NestedTensor (B, var_seq)
11. sft_loss: standard cross-entropy over response tokens
```

### 4. Output Reconstruction

`restore_flat_to_nested` reconstructs per-sample outputs from the flat layout:

```python
# For each leaf i (= sample i):
sample_logprobs[i] = cat(prefix_logprobs, leaf_i_logprobs)
# Shape matches original input_ids[i] exactly
```

This uses `torch.nested.as_nested_tensor` (not `nested_tensor`) to preserve `grad_fn` through the concatenation, enabling correct gradient flow to the prefix representations.

For **multilevel trees**, `leaf_ancestor_ranges` stores all ancestor segments per leaf: `[(root_range, turn2_range, ...)]`. `restore_flat_to_nested` concatenates all ancestor slices + the leaf slice to reconstruct each sample.

---

## Key Data Structures

### PrefixTreeParams

Built by `build_prefix_tree_params` from a list of per-sample token sequences:

```python
@dataclass
class PrefixTreeParams:
    prefix_range: tuple[int, int]          # (0, prefix_len) in flat layout
    leaf_ranges: list[tuple[int, int]]     # [(prefix_len, prefix_len+R_0), ...]
    leaf_to_sample: list[int]              # leaf_i → original sample index
    q_ranges: list[tuple[int, int]]        # attention rectangle q-sides
    k_ranges: list[tuple[int, int]]        # attention rectangle k-sides
    mask_types: list[str]                  # 'causal' | 'full' per rectangle
    flat_tokens: Tensor                    # (total_tokens,) token ids
    flat_position_ids: Tensor              # (total_tokens,) position ids
    flat_loss_mask: Tensor                 # (total_tokens,) 0/1 mask
    total_seqlen_q: int
    total_seqlen_k: int
    multilevel: bool = False               # True for multi-level trees
```

### PrefixTreeMagiBatch

Returned by `build_prefix_tree_micro_batch`, consumed by model forward:

```python
@dataclass
class PrefixTreeMagiBatch:
    flat_input_ids: Tensor           # (total_tokens,) full flat layout
    flat_position_ids: Tensor
    flat_loss_mask: Optional[Tensor]
    magi_key: object                 # MAGI DistAttnRuntimeKey (or None)
    flex_key: object                 # PyTorch block_mask (or None)
    leaf_to_sample: list[int]
    leaf_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]
    original_batch_size: int
    real_tokens: int                 # tokens before TP padding
    leaf_ancestor_ranges: Optional[list[list[tuple[int,int]]]]  # multilevel
    # CP-local tensors (dispatched by MAGI for CP>1, equal to flat_* when CP=1):
    local_flat_input_ids: Tensor     # (local_tokens,)
    local_flat_position_ids: Tensor
    local_flat_loss_mask: Optional[Tensor]
```

---

## Mask Generation Metadata

The attention pattern for a prefix tree is encoded as a list of **rectangles**, each describing a block of query positions, a block of key positions, and a mask type (`causal` or `full`). Both MAGI and flex_attention consume this same rectangle representation.

### Single-Level Tree (1-to-n): `build_prefix_tree_flex_spec`

For the common RL case — one shared prefix, `n` independent leaf branches — the rectangle list is built by `build_prefix_tree_flex_spec(prefix_len, branch_lengths)`.

**Input:** a flat layout `[prefix: P tokens | leaf_0: R_0 tokens | leaf_1: R_1 tokens | ... | leaf_{n-1}: R_{n-1} tokens]`

**Algorithm:**

```
rect 0:  q=[0,P)       k=[0,P)            CAUSAL   ← prefix self-attention

for each leaf_i with range [start_i, end_i):
  rect 2i+1: q=[start_i, end_i)  k=[0, P)          FULL    ← leaf attends to all prefix
  rect 2i+2: q=[start_i, end_i)  k=[start_i, end_i) CAUSAL ← leaf causal self-attention
```

Total rectangles: `1 + 2n` for `n` leaves.

**Example: n=3, P=4, R=[3,3,2]**

```
Flat layout:  [0 1 2 3 | 4 5 6 | 7 8 9 | 10 11]
               prefix   leaf_0   leaf_1  leaf_2

q_ranges    = [(0,4), (4,7), (4,7), (7,10), (7,10), (10,12), (10,12)]
k_ranges    = [(0,4), (0,4), (4,7), (0, 4), (7,10), (0,  4), (10,12)]
mask_types  = [CAUSAL, FULL, CAUSAL, FULL, CAUSAL, FULL, CAUSAL]
```

Visualized as a dense attention matrix (✓ = can attend, · = blocked):

```
        0 1 2 3 | 4 5 6 | 7 8 9 | 10 11
      ┌─────────────────────────────────
   0  │ ✓ · · · | · · · | · · · | ·  ·
   1  │ ✓ ✓ · · | · · · | · · · | ·  ·    prefix: causal
   2  │ ✓ ✓ ✓ · | · · · | · · · | ·  ·
   3  │ ✓ ✓ ✓ ✓ | · · · | · · · | ·  ·
      ├─────────────────────────────────
   4  │ ✓ ✓ ✓ ✓ | ✓ · · | · · · | ·  ·
   5  │ ✓ ✓ ✓ ✓ | ✓ ✓ · | · · · | ·  ·   leaf_0: full to prefix + causal self
   6  │ ✓ ✓ ✓ ✓ | ✓ ✓ ✓ | · · · | ·  ·
      ├─────────────────────────────────
   7  │ ✓ ✓ ✓ ✓ | · · · | ✓ · · | ·  ·
   8  │ ✓ ✓ ✓ ✓ | · · · | ✓ ✓ · | ·  ·   leaf_1: full to prefix + causal self
   9  │ ✓ ✓ ✓ ✓ | · · · | ✓ ✓ ✓ | ·  ·   (BLOCKED from leaf_0)
      ├─────────────────────────────────
  10  │ ✓ ✓ ✓ ✓ | · · · | · · · | ✓  ·
  11  │ ✓ ✓ ✓ ✓ | · · · | · · · | ✓  ✓   leaf_2: full to prefix + causal self
```

This is **mathematically identical** to computing full causal attention independently for `[prefix || leaf_i]` for each `i`.

### Position IDs

Each leaf's position IDs continue from the prefix: `[0..P-1, P..P+R_i-1]`. Leaf_0 and leaf_1 both start at position `P`, reflecting that they are independent continuations of the same prefix — not successive turns.

### Multi-Level Tree: `build_multilevel_flex_spec`

For deeper trees (e.g., multi-turn SFT where each turn has a distinct system prompt shared across samples), `build_multilevel_flex_spec(root: TreeNode)` generalizes the single-level encoding.

**Tree layout (DFS pre-order):**

```
TreeNode structure:
  root (P tokens)         ← system + turn1, shared by all samples
  ├── child_A (T1_A tok)  ← turn2 branch A, shared by samples 0,1
  │   ├── leaf_AA         ← turn3 leaf A1 (sample 0 unique)
  │   └── leaf_AB         ← turn3 leaf A2 (sample 1 unique)
  └── child_B (T1_B tok)  ← turn2 branch B, shared by samples 2,3
      ├── leaf_BA         ← turn3 leaf B1 (sample 2 unique)
      └── leaf_BB         ← turn3 leaf B2 (sample 3 unique)

Flat layout: [root | child_A | leaf_AA | leaf_AB | child_B | leaf_BA | leaf_BB]
```

**Rectangle generation rule (per node):**

```
For each node N with range [s, e):
  1. Emit CAUSAL rect: q=[s,e), k=[s,e)          ← self-attention
  2. For every descendant D of N:
     Emit FULL rect: q=[D.start, D.end), k=[s,e) ← descendant attends to ancestor
```

**Detection:** `_resolve_multilevel_tree()` in `prefix_tree_magi.py` groups samples by their first post-root segment hash (using token scan, not segment hash, for correct chat-template alignment), then computes per-group turn2 shared length via `longest_common_prefix_length`. Returns `None` when no multi-level structure exists, falling back to single-level.

### Rectangle Count

| Tree structure | Rectangles |
|---------------|-----------|
| Single-level, n leaves | `1 + 2n` |
| Multi-level, depth-2, 2 branches × 2 leaves | 17 |
| Multi-level, depth d, branching factor b | `O(nodes × descendants)` |

---

## Attention Backends

### flex (default)

Uses `torch.nn.attention.flex_attention` with a block mask:

```python
def prefix_tree_mask(b, h, q_idx, kv_idx):
    q_leaf = leaf_id[q_idx]; k_leaf = leaf_id[kv_idx]
    in_prefix_k = kv_idx < prefix_end
    same_leaf = (q_leaf == k_leaf) & (q_leaf >= 0)
    causal = kv_idx <= q_idx
    return (in_prefix_k & causal) | (same_leaf & causal) | (in_prefix_k & (q_leaf >= 0))

block_mask = create_block_mask(prefix_tree_mask, ..., _compile=False)
```

**Note:** `_compile=False` is used to avoid slow Triton JIT compilation. Without `_compile=True`, `flex_attention` materializes large transient O(T²) tensors during the forward kernel (~17GB at T=14k), causing OOM at mbs=4. This is a known limitation — magi is the recommended backend for production use.

**Advantages:** Pure PyTorch, no external dependencies, works out of the box.

### magi

Uses [Magi Attention](https://github.com/SandAI-org/magi-attention)'s dispatch → calc_attn → undispatch API.

**CP-aware flow (CP≥1):**

```python
# 1. Dispatch flat tokens to CP-local slice BEFORE model forward:
local_flat_tokens = dispatch(flat_tokens.unsqueeze(-1), magi_key).squeeze(-1)

# 2. Model forward on local tokens (each CP rank processes its assigned slice):
local_output = model(local_flat_tokens, ...)

# 3. Inside _magi_attn_forward — Q/K/V already local, no re-dispatch needed:
out, _ = calc_attn(q, k, v, magi_key)   # distributed attention

# 4. Undispatch model output AFTER all layers:
full_output = undispatch(local_output, magi_key)
```

**Key:** Use `mpu.get_context_parallel_group()` rather than `dist.group.WORLD`. With TP>1 and no CP, this group is size-1 — MAGI performs no cross-rank dispatch.

**CP correctness note:** MAGI's `dispatch` performs workload-balanced redistribution across CP ranks based on the attention rectangle pattern — not the naive 2×CP interleaved split used by Megatron's standard CP. This correctly handles non-causal masks where different tokens have different compute costs.

**Advantages:** O(T) backward (only local Q/K/V/out/lse stored), CP-aware load-balanced dispatch, optimized for large-scale training.

---

## Patch Chain

`prefix_tree_merge.py` monkey-patches Megatron's forward call chain at init time to thread `magi_attention_key` / `flex_attention_key` through all layers:

```
GPTModel.forward(magi_attention_key=key, flex_attention_key=key)
  └─ TransformerBlock.forward(...)
       └─ TransformerLayer.forward(...)
            └─ SelfAttention.forward(...)
                 └─ TEDotProductAttention.forward(magi_attention_key, flex_attention_key)
                      ├─ if magi_key: _magi_attn_forward(q, k, v, magi_key)
                      ├─ elif flex_key: _flex_attn_forward(q, k, v, flex_key)
                      └─ else: standard TE flash attention (FA3)
```

`magi_patch.py` is retained as a backward-compatibility shim re-exporting from `prefix_tree_merge.py`.

---

## TP/PP/CP Compatibility

### Tensor Parallel (TP)

Megatron's sequence parallel scatters hidden states along the token dimension across TP ranks, requiring `total_tokens % tp_size == 0`. For the prefix-tree flat layout, padding to the next TP multiple is applied before building the attention key, with `real_tokens` tracking the unpadded count.

### Pipeline Parallel (PP)

Prefix-tree is applied only in the THD (remove-padding) data format path, independent of PP stage boundaries. Each PP stage receives the flat token layout and produces outputs in the same shape.

### Context Parallel (CP)

MAGI's `dispatch`/`undispatch` handles CP natively: tokens are redistributed across CP ranks based on attention workload (not naive sequential split). The key is built with `cp_group=mpu.get_context_parallel_group()` so MAGI knows the CP topology. CP=2 correctness is under active implementation.

---

## Prefix Detection

**Fast path (O(batch × turns)):**

When `prefix_segments` are available, uses hash-based detection:
```python
# Build {(hash, cum_len): count} across all samples
# Find max cum_len where count == batch_size → universal prefix length
```

**Multi-level detection (O(batch × seqlen)):**

`_resolve_multilevel_tree()` groups samples by their first post-root segment content hash, then uses `longest_common_prefix_length` within each group to find the turn2 shared length. Returns `None` if no multi-level structure exists.

**Fallback (O(batch × seqlen)):**

`longest_common_prefix_length(tokens_by_sample)` — byte-wise scan across all token sequences.

---

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.use_prefix_tree` | bool | `False` | Enable prefix-tree detection and flat layout |
| `data.prefix_tree_attention` | str | `"flex"` | Attention backend: `"flex"` or `"magi"` |
| `data.shuffle` | bool | `True` | Set `False` for tree-structured datasets to preserve sample order |
| `engine.context_parallel_size` | int | `1` | CP degree (magi only for CP>1) |
| `MAGI_ATTENTION_KERNEL_BACKEND` | env | `"sdpa"` | MAGI kernel: `"sdpa"` (safe) or `"ffa"` (Triton) |

Requires: `model.use_remove_padding=True`, `engine.use_mbridge=True`.

---

## Known Limitations

| Limitation | Status |
|-----------|--------|
| flex OOM at mbs=4, seq≥10k — `flex_attention` materializes ~17GB transient O(T²) tensors during forward kernel | Known; magi is the recommended backend |
| flex slowdown — 2.5× slower than magi at same mbs (both with recompute=[]); root cause: transient kernel memory | Pending investigation (job_3 task_2) |
| magi CP=2 correctness — dispatch/undispatch integration with Megatron CP in progress | Active fix (job_4) |
| magi with `sdpa` backend has O(T²) backward via SDPA internal matrices | Known; use `ffa` backend when verified |
| Vision models (multimodal) not supported | Deferred |
| DSA (DeepSeek V3.2) sparse attention — `core_attention` replaced, patch doesn't fire | Safe (no crash, no benefit) |
| Gemma3 sliding window attention | Deferred |

---

## Comparison with Forge

| Aspect | Forge | This Implementation |
|--------|-------|---------------------|
| Core idea | Prefix Tree Merging via MAGI attention | Same, extended to PyTorch flex_attention |
| Attention backend | MAGI only | flex (default, no deps) + magi |
| Training framework | Custom | VERL (Megatron-LM backend) |
| Tree depth | Single-level | Single-level + multi-level (arbitrary depth) |
| Prefix detection | Token scan | Hash-based O(batch×turns) fast path + token scan fallback |
| TP support | Not described | Padding + real_tokens tracking; CP group fix for MAGI |
| CP support | Not described | MAGI dispatch/undispatch (in progress) |
| SFT integration | Not described | Dataset-level prefix_segments + trainer meta_info |
| RL integration | Focus | DataProto.repeat() + build_prefix_segments_single_turn |
| Backward correctness | Claimed equivalent | Verified: per-layer diff analysis confirms fp accumulation only |

---

## Experiment Results

All results on MiMo-7B-RL, 8gpu (H20), Megatron backend, TP=4, seed=42, `recompute_modules=[]`, `dynamic_bsz=False`.

### Single-Level Prefix Tree (gbs=16, seq=10240, ~60% prefix)

Group A — no recompute:

| Backend | mbs | Prefix-tree | MFU* | Step time | Peak mem |
|---------|-----|-------------|------|-----------|----------|
| FA3 | 2 | ❌ | 0.812 | ~24s | 68 GB |
| FA3 | 4 | ❌ | 0.835 | ~23s | 103 GB |
| **magi** | 4 | ✅ 45% sharing | **1.38** | **~16s** | **80 GB** |
| flex | 4 | ✅ | OOM | OOM | >133 GB |

Group B — core_attn recompute:

| Backend | mbs | Prefix-tree | MFU* | Step time | Peak mem |
|---------|-----|-------------|------|-----------|----------|
| FA3 | 2 | ❌ | 0.744 | ~25s | 70 GB |
| FA3 | 4 | ❌ | 0.767 | ~24s | 107 GB |
| **magi** | 4 | ✅ 45% sharing | **1.14** | **~18s** | **80 GB** |
| flex | 4 | ✅ | 0.388 | ~40s | 117 GB |

*MFU denominator = full pre-dedup token count — inflated for prefix-tree, useful for relative comparison.

**magi: 30% faster, 22% less memory vs FA3 mbs=4 (no recompute).**  
**flex: 2.5× slower than magi** due to O(T²) transient memory in forward kernel (known limitation).

### Multi-Level Tree (gbs=4, 2-branch tree, seq~12.8k, ~50% prefix, no recompute)

| Backend | mbs | prefix_sharing | step_time | peak_mem |
|---------|-----|----------------|-----------|----------|
| FA3 | 2 | — | ~6.7s | 77 GB |
| FA3 | 4 | — | ~6.6s | 122 GB |
| **magi** | 4 | **50.0%** | **~3.9s** | **86 GB** |

**42% faster than FA3 mbs=2 at 12k tokens with 50% prefix sharing.**  
At long sequences (≥8k), attention dominates → token reduction directly → step time reduction.  
Full_tokens=51149 → flat_tokens=25572 (confirmed via `PT-TIME prefix_sharing=50.0%`).

### CP=2 (TP=4, CP=2, gbs=16, seq=10240) — in progress

| Backend | MFU | Step time | Status |
|---------|-----|-----------|--------|
| FA3 CP=2 | 0.785 | ~25s | ✅ correct |
| magi CP=2 | — | — | 🔧 fix in progress |

CP=2 + magi requires dispatch before model forward and undispatch after all layers. Active implementation in `verl/utils/prefix_tree_magi.py` and `verl/models/mcore/model_forward.py`.
