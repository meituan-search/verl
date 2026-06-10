# [model, trainer, engine] feat: prefix-tree MAGI attention for verl SFT and RL

## What does this PR do?

The RFC: https://github.com/verl-project/verl/issues/6401

Adds prefix-tree shared-prefix deduplication for verl SFT and GRPO training using [MAGI attention](https://github.com/SandAI-org/magi-attention). Token-by-token trie detection discovers shared prefixes across rollout samples without requiring rollout-side metadata. The flat token layout + attention rectangle spec is dispatched to MAGI `calc_attn`, which computes correct prefix-tree attention patterns while internally deduplicating shared KV tokens.

### Key features

- **Dynamic trie detection**: `build_tree_dynamic` builds a compressed trie from input tokens — no prior knowledge of turn boundaries needed.
- **Multi-level tree support**: handles multi-turn rollout sharing with zero-length leaf nodes for samples that terminate at intermediate levels.
- **MAGI integration**: patches Megatron-LM's `TEDotProductAttention` through the `SelfAttention → TransformerLayer → GPTModel` chain to inject MAGI/flex attention.
- **Dynamic micro-batch grouping**: `dfs_micro_batch_groups` + trie-based token budgeting for efficient DP load balancing with prefix locality.
- **CP-safe dispatch**: always dispatches/undispatches at every layer for correctness under context parallelism (CP>1).
- **Configurable old-log-prob backend**: `prefix_tree_olb_backend` selects which backend to use when the roll-out log-prob computation path diverges from the training path.

## Design

This PR implements the design described in RFC #6401. All `n` GRPO rollout samples are packed into a single flat `[prefix | leaf_0 | ... | leaf_{n-1}]` sequence, run through **one** transformer forward pass, with cross-leaf attention blocked. The result is mathematically equivalent to `n` independent forwards.

### Attention rectangle spec

The attention pattern is encoded as rectangle specs:

```
          k: prefix    k: leaf0    k: leaf1
q: prefix   causal       ✗           ✗
q: leaf0     full      causal         ✗
q: leaf1     full        ✗         causal
```

MAGI interprets these rectangles natively.

### Prefix detection (dynamic trie)

Prefix detection runs entirely at train time — no rollout-side metadata required. `build_tree_dynamic` performs token-by-token compressed trie insertion over the micro-batch's input sequences. The trie is converted to a `TreeNode` tree, then fed to `build_prefix_tree_attention_spec`, which emits `(q_ranges, k_ranges, mask_types)` for every node.

The tree generalizes to arbitrary-depth multi-level trees. For example, in multi-turn agent RL where responses branch into sub-groups sharing a turn-2 prefix:

```
S0: turn0 + turn1_A + turn2_B1
S1: turn0 + turn1_A + turn2_B2
S2: turn0 + turn1_C + turn2_D1
S3: turn0 + turn1_C + turn2_D2
```

This produces a depth-3 tree with `turn0` as root, `turn1_A`/`turn1_C` as intermediate nodes, and four leaves. Zero-length leaf nodes are inserted when a sample terminates at an intermediate node.

### Megatron integration

The patch chain injects `magi_attention_key` through upstream Megatron-LM:

```
GPTModel.forward → TransformerBlock → TransformerLayer
    → SelfAttention → TEDotProductAttention
        → magi_attn_forward: dispatch → calc_attn → undispatch
```

Every layer always dispatches/undispatches for CP safety. The MAGI key carries the attention spec and model config (`num_heads`, `head_dim`, CP group). A RoPE bypass ensures CP-rank-specific frequency slicing is disabled during the prefix-tree forward — since MAGI passes the full flat sequence to every CP rank, each rank needs the full `[0, T)` position frequencies.

### File summary

| Area | Path |
|---|---|
| Trie + load balancing | `verl/utils/prefix_tree/dynamic.py` |
| Layout + attention spec | `verl/utils/prefix_tree/utils.py` |
| MAGI key + batch | `verl/utils/prefix_tree/magi.py` |
| Trainer helpers | `verl/utils/prefix_tree/trainer.py` |
| Megatron patch | `verl/models/mcore/prefix_tree_merge.py` |

## Result

[TODO]

## Test

### CPU unit tests (59 tests, all pass)
```bash
python3 prefix_script/test/test_prefix_tree_full.py  # temp location
```

### GPU training
Validated on Megatron-LM training with GRPO on CoQA — prefix-tree MAGI attention runs correctly with CP=2, SP=1, PP=1.

## Checklist Before Starting

- [x] Search for similar PRs: [upstream verl prefix-tree PRs](https://github.com/verl-project/verl/pulls?q=tree)
- [x] Format the PR title as `[{modules}] {type}: {description}`
- [x] Read the [Contribute Guide](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md)
- [x] Apply pre-commit checks
