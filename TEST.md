# Testing

## Environment (8gpu)

**Target host: `8gpu`** (migrated from `bjgpu` 2026-04-29)

On the GPU host, NVIDIA/CUDA/Python env vars are set only in **login shells** via `/etc/profile.d/`:

| Variable | Set in login shell | Non-login shell |
|---|---|---|
| `PATH` | includes `/usr/local/miniconda3/bin` and `/usr/local/cuda-12.8/bin` | minimal system PATH only |
| `LD_LIBRARY_PATH` | all NVIDIA lib paths from miniconda site-packages | empty |
| `NVIDIA_HOME` | `/usr/local/miniconda3/lib/python3.12/site-packages/nvidia` | empty |
| `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH` | NVIDIA + CUTLASS headers | empty |

**Consequence:** bare `python` in TEST.md commands only works in a login shell (`ssh 8gpu` interactive, or `bash --login`).
For non-login SSH commands (e.g. `ssh 8gpu "python ..."`) use the full path:
`/usr/local/miniconda3/bin/python` and ensure `LD_LIBRARY_PATH` is set, or use `bash --login -c '...'`.

**Root access:** `ssh root@8gpu` works directly (same host, root user). Required for: writing to `/usr/local/miniconda3/...` (patching installed packages), running `ncu` (GPU perf counters need root).

`ncu_profile_all.sh` hard-codes `/usr/local/miniconda3/bin/python` and sets the required env vars explicitly — use it as the reference for non-login invocations.

## SFT smoke test (MiMo-7B-RL + Megatron + prefix-tree)

**Verified on 8gpu: 2026-05-06**

Prerequisites:
- MiMo-7B-RL checkpoint at `/tmp/claude/MiMo-7B-RL` (copy from dolphinfs; see memory for path)
- 60%-prefix dataset at `/tmp/claude/gsm8k_sft_prefix60/` (build with `/tmp/claude/prepare_gsm8k_sft_prefix.py`)
- `magi_attention` and `mbridge` installed
- Set `MAGI_ATTENTION_KERNEL_BACKEND=sdpa` (FFA kernel not verified on 8gpu)

```bash
# Prepare datasets (one-time)
python3 /tmp/claude/prepare_gsm8k_sft.py          # plain GSM8K (no shared prefix)
python3 /tmp/claude/prepare_gsm8k_sft_prefix.py   # 60% shared-prefix dataset

# FA3 baseline on 60%-prefix data (MAGI actually activates)
MIMO_PATH=/tmp/claude/MiMo-7B-RL \
DATASET_DIR=/tmp/claude/gsm8k_sft_prefix60 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /tmp/claude/run_mimo_megatron_prefix_tree_sft.sh 1 /tmp/claude/mimo_fa3_p60_ckpt \
    trainer.total_training_steps=1 data.use_prefix_tree=false trainer.seed=42 \
    data.max_length=1024 data.max_token_len_per_gpu=4096 \
    data.use_dynamic_bsz=false data.micro_batch_size_per_gpu=2

# Prefix-tree flex (default) on 60%-prefix data
MIMO_PATH=/tmp/claude/MiMo-7B-RL \
DATASET_DIR=/tmp/claude/gsm8k_sft_prefix60 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /tmp/claude/run_mimo_megatron_prefix_tree_sft.sh 1 /tmp/claude/mimo_pt_p60_ckpt \
    trainer.total_training_steps=1 data.use_prefix_tree=true \
    data.prefix_tree_attention=flex trainer.seed=42 \
    data.max_length=1024 data.max_token_len_per_gpu=4096 \
    data.use_dynamic_bsz=false data.micro_batch_size_per_gpu=2

# Prefix-tree magi on 60%-prefix data
MIMO_PATH=/tmp/claude/MiMo-7B-RL \
DATASET_DIR=/tmp/claude/gsm8k_sft_prefix60 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /tmp/claude/run_mimo_megatron_prefix_tree_sft.sh 1 /tmp/claude/mimo_pt_p60_ckpt \
    trainer.total_training_steps=1 data.use_prefix_tree=true \
    data.prefix_tree_attention=magi trainer.seed=42 \
    data.max_length=1024 data.max_token_len_per_gpu=4096 \
    data.use_dynamic_bsz=false data.micro_batch_size_per_gpu=2
```

Expected results (2026-05-07, 8gpu, seed=42, gsm8k_sft_prefix60, TP=1, dynamic_bsz=false, micro_batch=2):

| path | train/loss | val/loss | grad_norm | note |
|------|-----------|---------|----------|------|
| FA3 baseline | **1.4117** | 0.7993 | 35.4 | reference |
| `prefix_tree_attention=flex` (default) | **1.4993** | 0.9131 | 39.5 | +6% — flex vs FA3 kernel |
| `prefix_tree_attention=magi` | **1.4568** | 0.9079 | 39.9 | +3% — SDPA vs FA3 kernel |

Both diffs are **numerical precision** (different kernel accumulation), not semantic bugs.
Set `MAGI_ATTENTION_KERNEL_BACKEND=sdpa` when using magi (FFA kernel not verified on 8gpu).

**Key bugs fixed (2026-05-06/07):** See `FIX-e2e-sft-integration.md` for complete list.

**New param (2026-05-07):** `data.prefix_tree_attention: flex|magi` — default `flex` (no MAGI dep).
Implemented in `verl/models/mcore/prefix_tree_merge.py` (renamed from magi_patch.py).

## Prefix-tree standalone test

```bash
python -m pytest test_prefixtree/test_prefix_tree.py -v
```

Expected: all 4 tests pass.

## Prefix-tree MAGI flex demo

```bash
python test_prefixtree/demo_magi.py
```

Expected:
- The reference section reports three matching sequences.
- The MAGI API section either runs on CUDA with a local MAGI checkout, or prints a clear skip message when CUDA/MAGI runtime is unavailable.
- When MAGI runs on CUDA, the script also prints MAGI timing plus a dense FlashAttention batch-size-2 benchmark or a clear FlashAttention skip message.

## Prefix-tree efficiency benchmark

```bash
python test_prefixtree/benchmark_prefix_tree_attention.py
python test_prefixtree/benchmark_prefix_tree_attention.py --batch-sizes 2,4,8 --breakdown
python test_prefixtree/benchmark_prefix_tree_attention.py --seq-len 32768 --prefix-ratio 20%
```

Expected:
- Requires CUDA plus either an importable `magi_attention` runtime or a local MAGI checkout, or prints a clear skip message.
- Verifies PrefixTree output against baseline batch attention path.
- Runs 10 warmup iterations first by default, then reports baseline vs PrefixTree timing.
- Accepts single-case inputs like `--batch-size`, `--prefix-len`, `--suffix-len`, `--seq-len`, `--prefix-ratio`, `--chunk-size`, `--measure-iters`, and `--seed`.
- Accepts multi-case sweeps through `--batch-sizes 2,4,8`.
- With `--breakdown`, prints one-time key build plus PrefixTree dispatch / calc / undispatch timing to debug overhead.
- Defaults to batch size 2 so shared-prefix win is easy to inspect.

## Prefix-tree kernel identity 9-run matrix

Use this when checking whether a profiler/timing run is hitting the dense baseline path, upstream FA3, or MAGI FFA.

Timed rerun matrix:

```bash
for seqlen in 1024 8192 32768; do
  python test_prefixtree/benchmark_prefix_tree_attention.py \
    --mode bench \
    --batch-size 2 \
    --seq-len "$seqlen" \
    --prefix-ratio 20% \
    --warmup-iters 10 \
    --measure-iters 50 \
    --baseline-backend torch-sdpa \
    --target baseline

  python test_prefixtree/benchmark_prefix_tree_attention.py \
    --mode bench \
    --batch-size 2 \
    --seq-len "$seqlen" \
    --prefix-ratio 20% \
    --warmup-iters 10 \
    --measure-iters 50 \
    --baseline-backend fa3 \
    --target baseline

  MAGI_ATTENTION_KERNEL_BACKEND=ffa \
  python test_prefixtree/benchmark_prefix_tree_attention.py \
    --mode bench \
    --batch-size 2 \
    --seq-len "$seqlen" \
    --prefix-ratio 20% \
    --warmup-iters 10 \
    --measure-iters 50 \
    --baseline-backend fa3 \
    --target prefix-tree
 done
```

Record these fields from stdout:
- torch-sdpa baseline: `baseline_time_ms=...`
- FA3 dense kernel: `baseline_time_ms=...`
- MAGI PrefixTree kernel: `prefix_tree_time_ms=...`

Interpretation:
- `--target baseline` benchmarks only the dense path selected by `--baseline-backend`.
- `--target prefix-tree` benchmarks only the PrefixTree path, but still runs one dense correctness pass first outside the timed section.
- The profiler symbol `flash::DynamicPersistentTileScheduler<...>` is upstream FA3 dense kernel.
- MAGI FFA uses `flash::DynamicPersistentTileSchedulerFwd<...>` plus flex-range inputs (`q_ranges`, `k_ranges`, `attn_type_map`).

## Prefix-tree speedup table sweep

```bash
python test_prefixtree/benchmark_prefix_tree_attention.py --table-sweep
```

Expected:
- Prints markdown-style speedup tables for batch size, prefix ratio, and sequence length.
- Uses default non-swept values of batch size 2, prefix ratio 20%, and sequence length 32768 unless overridden with `--table-default-*` flags.
- Uses `--table-batch-sizes`, `--table-prefix-ratios`, and `--table-seq-lens` to override the sweep axes.
- Reports usage/procedure data only; do not treat the printed numbers as durable documentation.

## Prefix-tree Nsight Systems timeline run

Loop capture:

```bash
nsys profile --trace=cuda,nvtx,osrt -o prefix_tree_attention \
  python test_prefixtree/benchmark_prefix_tree_attention.py --mode nsys --nvtx
```

Single-forward capture with CUDA profiler API:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o prefix_tree_attention_single \
  python test_prefixtree/benchmark_prefix_tree_attention.py \
    --mode nsys \
    --nvtx \
    --cuda-profiler-range
```

Expected:
- Runs the same correctness check first.
- Performs an explicit 10-iteration warmup before profiling for each path.
- Default loop mode emits NVTX ranges for `baseline_warmup`, `baseline_profile`, `prefix_tree_warmup`, and `prefix_tree_profile`.
- `--cuda-profiler-range` instead captures only one forward pass per selected path after warmup using `cudaProfilerStart()` / `cudaProfilerStop()`.
- Produces an `.nsys-rep` timeline you can inspect in Nsight Systems.

## verl prefix-tree CPU unit tests (no GPU, no Megatron fork)

Tests live in `tests/utils/test_prefix_tree_magi.py`. No GPU, no Megatron fork, no special env needed.

```bash
cd /path/to/verl-prefix-tree
python -m pytest tests/utils/test_prefix_tree_magi.py -v
```

Expected: **18 tests pass**.

Test classes:
- `TestBuildPrefixTreeLayout` — flat layout, position IDs, flex-rect structure
- `TestRestoreFlatToNested` — round-trip flat → NestedTensor restore
- `TestNestedTensorUnpack` — `build_prefix_tree_micro_batch` with monkeypatched MAGI key
- `TestPrefixSegmentsPrior` — prior-knowledge hash path: scan equivalence, fallback, shuffle safety, single-turn helper

All imports are from `verl.utils.*` — no `MEGATRON_PATH` or sys.path injection required.

## MAGI attention kernel GPU unit tests

Tests live in `tests/unit_tests/transformer/test_attention.py` in the **Megatron-LM-prefix-tree** repo, class `TestMAGIAttentionKernel`.

They require CUDA + `magi_attention` installed; both tests are auto-skipped otherwise.

**Run on bjgpu (login shell):**
```bash
cd /path/to/Megatron-LM-prefix-tree
MAGI_ATTENTION_KERNEL_BACKEND=ffa \
python -m pytest tests/unit_tests/transformer/test_attention.py::TestMAGIAttentionKernel -v
```

Tests:
- `test_magi_dispatch_output_shape` — verifies output shape `(total_tokens, 1, np, hn)` when `magi_attention_key` is passed to `TEDotProductAttention.forward`.
- `test_magi_output_matches_dense_causal_reference` — verifies prefix-tree MAGI output matches per-sample dense causal attention reference (max diff < 5e-2 in bf16).

**Run all attention unit tests:**
```bash
python -m pytest tests/unit_tests/transformer/test_attention.py -v
```

## Prefix-tree MAGI Ray wrapper

```bash
python test_prefixtree/ray_demo_magi.py
```

Expected:
- Prints host / CUDA availability information.
- Runs the same reference + MAGI benchmark flow as `demo_magi.py`.
- Returns the same exit code as the underlying demo.

Example Ray submission:

```bash
RAY_ADDRESS='http://33.32.20.73:44390' ray job submit --working-dir . -- python test_prefixtree/ray_demo_magi.py
```

Expected:
- Ray returns a job ID.
- Logs show the wrapper banner and the underlying demo output.
- The MAGI API section either runs on CUDA with a local MAGI checkout, or prints a clear skip message when CUDA/MAGI runtime is unavailable.
- When MAGI runs on CUDA, the script also prints MAGI timing plus a dense FlashAttention batch-size-2 benchmark or a clear FlashAttention skip message.
