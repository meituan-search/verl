# CLAUDE.md — verl-prefix-tree

> Project-specific instructions for Claude Code. For upstream contribution rules see `AGENTS.md`.

## Related Repos

- **Megatron-LM fork** (prefix-tree benchmark scripts + reference):
  `/Users/arvyanh/Documents/coderead/Megatron-LM-prefix-tree`
  - `prefix_tree_utils.py` and `prefix_tree_params.py` have been **copied into verl** (`verl/utils/`) — the fork is no longer required at runtime
  - `_magi_attn_forward` and `magi_attention_key` threading have been **moved into** `verl/models/mcore/magi_patch.py` — no Megatron fork needed
  - TEST.md, PERFTEST.md, FIX-*.md, benchmark scripts also live there (copies here for reference)

- **MAGI Attention runtime**: `/Users/arvyanh/Documents/coderead/magi-attention`
- **Megatron MAGI reference fork**: `/Users/arvyanh/Documents/coderead/Megatron-LM-magi-attention`

## Prefix-Tree MAGI Attention Integration

This repo contains the verl-side integration of prefix-tree MAGI attention for SFT and RL training.

### Files changed in this repo

| File | Purpose |
|------|---------|
| `verl/utils/prefix_tree_params.py` | `PrefixTreeParams` dataclass (copied from Megatron fork) |
| `verl/utils/prefix_tree_utils.py` | `build_prefix_tree_params`, `longest_common_prefix_length`, etc. (copied from fork) |
| `verl/utils/prefix_tree_magi.py` | `PrefixTreeMagiBatch`, `build_prefix_tree_micro_batch`, `restore_flat_to_nested`, `_hash_prefix`, `_resolve_prefix_len_from_segments`, `build_prefix_segments_single_turn` |
| `verl/models/mcore/magi_patch.py` | `apply_magi_patch()` — monkey-patches upstream Megatron to thread `magi_attention_key`; `_magi_attn_forward` MAGI kernel helper |
| `verl/models/mcore/model_forward.py` | Prefix-tree branch in `gptmodel_forward_model_engine` |
| `verl/workers/engine/megatron/transformer_impl.py` | Reads `use_prefix_tree` + `prefix_segments` from batch; calls `apply_magi_patch()` on init |
| `verl/trainer/sft_trainer.py` | Propagates `use_prefix_tree` via `meta_info` |
| `verl/trainer/config/sft_trainer_engine.yaml` | `data.use_prefix_tree: false` default |
| `verl/utils/dataset/multiturn_sft_dataset.py` | Computes `prefix_segments` per sub-turn in `__getitem__` |
| `verl/trainer/ppo/ray_trainer.py` | Injects `prefix_segments` fallback after `batch.repeat(n)` for RL |

### Running SFT smoke test

**Target host: `8gpu`** (migrated from `bjgpu` 2026-04-29)

See `TEST.md` for full procedure. Quick command:
```bash
bash /tmp/claude/run_mimo_megatron_prefix_tree_sft.sh 1 /tmp/claude/mimo_pt_ckpt \
    trainer.total_training_steps=1 data.use_prefix_tree=true trainer.seed=42
```

Requires:
- `magi_attention` installed on `8gpu`
- `MIMO_PATH=/tmp/claude/MiMo-7B-RL`
- **No Megatron fork needed** — `magi_patch.py` patches upstream Megatron at runtime

### Prefix-segments prior knowledge (job_2, 2026-04-29)

`prefix_segments: list[tuple[int,int]]` — per-sample `(hash, cumulative_len)` list stored in `non_tensor_batch`.

- **SFT producer**: `multiturn_sft_dataset.__getitem__` — computes one entry per sub-turn using xxhash/md5 of full cumulative prefix
- **RL producer**: `ray_trainer.py` — injected after `batch.repeat(n)` via `build_prefix_segments_single_turn`; if dataset already has it, `DataProto.repeat()` propagates it automatically
- **Consumer**: `build_prefix_tree_micro_batch` — uses `_resolve_prefix_len_from_segments` (O(batch×turns)) instead of O(batch×seqlen) GPU token scan; falls back to scan when absent
- Shuffle-safe: stored in `non_tensor_batch` (numpy object array), handled by `select_idxs`/`reorder`

### Smoke test results (2026-04-23, MiMo-7B-RL, 1 step, bjgpu)

| Path | train/loss | val/loss |
|------|-----------|---------|
| MAGI prefix-tree | 3.0427 | 17.007 |
| FA3 baseline | 3.0427 | 16.987 |

Numerically equivalent ✓

### Smoke test results (2026-05-06, MiMo-7B-RL, 1 step, 8gpu, seed=42, GSM8K, 1 GPU)

| Path | train/loss | val/loss |
|------|-----------|---------|
| FA3 baseline | 1.4800372 | 0.9588060 |
| MAGI prefix-tree | 1.4800372 | 0.9583334 |

Numerically equivalent ✓  Host migration bjgpu→8gpu confirmed working.

**Verified results (2026-05-07, gsm8k_sft_prefix60, TP=1, seed=42):**

| Backend | train/loss | diff vs FA3 |
|---------|-----------|------------|
| FA3 baseline | 1.4117 | — |
| `prefix_tree_attention=flex` (default) | 1.4993 | +6% numerical precision |
| `prefix_tree_attention=magi` | 1.4568 | +3% numerical precision |

TP=2 flex verified: `0.0733` vs FA3 `0.0715` (+3%). Magi TP=2 in progress.

**CP=2 MAGI verified (2026-05-19, gsm8k_sft_10240, TP=4, CP=2, 8 GPUs, seed=42):**

| Backend | train/loss@1 | diff vs FA3 |
|---------|-------------|------------|
| FA3 baseline | 0.0638 | — |
| MAGI prefix-tree | 0.0653 | +2.3% ✅ |

Fix for CP>1: `_magi_rope_bypass` thread-local in `prefix_tree_merge.py` bypasses
`RotaryEmbedding`'s CP-rank-specific RoPE slicing during prefix-tree forward passes.

**New param `data.prefix_tree_attention`:** `flex` (default, no MAGI dep) or `magi`.
Implemented in `verl/models/mcore/prefix_tree_merge.py` (renamed from magi_patch.py).
`magi_patch.py` is now a backward-compat shim.

**Key bugs fixed (2026-05-06/07):** See `FIX-e2e-sft-integration.md`.  
Notable: label must be shifted `-1` (`torch.roll`); `as_nested_tensor` not `nested_tensor` for grad preservation; logits need THD transpose `(T,1,V)`.

**Sparse attention safety:** DSA (DeepSeek V3.2) — `core_attention` replaced, our patch doesn't fire (safe but no prefix-tree benefit). Qwen3-MoE — `use_sliding_window=False` (safe). Gemma3 sliding window — deferred.

**SFT scripts (8gpu):**
- Smoke test: `/tmp/claude/run_mimo_megatron_prefix_tree_sft.sh` (sets `MAGI_ATTENTION_KERNEL_BACKEND=sdpa`)
- Short prefix dataset (TP=1): `/tmp/claude/gsm8k_sft_prefix60/`
- Long prefix dataset (TP=2): `/tmp/claude/gsm8k_sft_10240/`
- Dataset builders: `/tmp/claude/prepare_gsm8k_sft_prefix.py`
- Layer diff tool: `/tmp/claude/debug_layer_diff.py` (use `COMPARE_LAYERS=1`)

**Note:** `engine.use_mbridge=True` required. `optim.lr_warmup_steps=0` for 1-step runs.

## Job Tracking

Jobs and task history: `.jobs/`

```bash
python3 ~/.claude/skills/code/scripts/job_manager.py resume
```

## Investigation Logs

- `FIX-prefix-tree-kernel-identity.md` — profiler kernel identity, MAGI-vs-FA3 distinction
- `FIX-ncu-single-step-capture.md` — NCU flags for single-step capture, root access, scp procedure
- `FIX-megatron-core-namespace.md` — megatron.core namespace collision fix
- `FIX-e2e-sft-integration.md` — end-to-end SFT integration bugs + all results table — **active 2026-05-07**
- `FIX-layer-diff-analysis.md` — per-layer hidden state diff FA3 vs MAGI; 3% gap is fp accumulation
- `FIX-timing-analysis.md` — TP=4 mbs=2 timing: FA3 MFU=0.813, flex/magi MFU=0.24 (-3×); memory -43% ✓

## Testing

See `TEST.md` for:
- Prefix-tree + MAGI CPU unit tests: `pytest tests/utils/test_prefix_tree_magi.py -v` (18 tests, no GPU, no Megatron fork needed)
- MAGI GPU correctness test (`.claude/run_magi_gpu_test.py` in Megatron repo)
- SFT smoke test procedure on bjgpu

## Performance

See `PERFTEST.md` for benchmark results:
- 9-case timing matrix: seq_len × {FA3, MAGI FFA prefix-tree, torch-sdpa baseline}
- NCU kernel identity analysis
