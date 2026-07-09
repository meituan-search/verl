#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Memory diagnostic: runs a short tree + dense training (default 8×H100,
# 2 steps each) with PROFILE_MEMORY_LOG=1 enabled in forward_step. Prints
# a comparison at the end so we can see whether the 5GB per-rank delta
# observed in production training shows up here, and if so, whether it's
# inside forward_step or outside.
#
# Usage:
#   bash prefix_script/fsdp/run_memlog_diag.sh                # 8 GPU, 2 step
#   N_GPUS=1 bash prefix_script/fsdp/run_memlog_diag.sh       # 1 GPU diagnostic
#   STEPS=3 bash prefix_script/fsdp/run_memlog_diag.sh        # more steps
#   ONLY=tree bash prefix_script/fsdp/run_memlog_diag.sh      # skip dense
#   ONLY=dense bash prefix_script/fsdp/run_memlog_diag.sh     # skip tree
#
set -eu

MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen2.5-3B-Instruct}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
LOG_ROOT=${LOG_ROOT:-/root/convergence_logs}
STEPS=${STEPS:-2}
N_GPUS=${N_GPUS:-8}
SEED=${SEED:-42}
ONLY=${ONLY:-both}   # tree | dense | both

# Batch size auto-scales with N_GPUS so workload per rank stays at 8 samples.
# With rollout.n=8, each rank gets one prompt × 8 rollouts = 8 samples per
# micro-batch, which is the configuration that exposes tree's prefix sharing.
BATCH=${BATCH:-$((N_GPUS * 8 / 8))}   # = N_GPUS prompts per step
[ "$BATCH" -lt 1 ] && BATCH=1

mkdir -p "$LOG_ROOT"

export NCCL_NVLS_ENABLE=0

_COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size=$BATCH
    data.max_prompt_length=512
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=8
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.project_name=mem_diag
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=$STEPS
    +trainer.seed=$SEED
)

# ----------------------------------------------------------------------------
# Run a single training (tree or dense) with mem logging on
# ----------------------------------------------------------------------------
_run_one() {
    local mode=$1
    local exp_name=$2
    local log_file=$3

    echo ""
    echo "=========================================="
    echo "[MEMLOG] $mode: $exp_name → $log_file"
    echo "[MEMLOG]   N_GPUS=$N_GPUS  BATCH=$BATCH  STEPS=$STEPS"
    echo "=========================================="

    local tree_args=()
    if [ "$mode" = "tree" ]; then
        tree_args=(
            actor_rollout_ref.actor.use_prefix_tree_dynamic=True
            actor_rollout_ref.actor.prefix_tree_attention=magi
            actor_rollout_ref.actor.context_parallel_size=1
        )
    else
        tree_args=(
            actor_rollout_ref.actor.use_prefix_tree_dynamic=False
        )
    fi

    PROFILE_MEMORY_LOG=1 python3 -m verl.trainer.main_ppo \
        "${_COMMON_ARGS[@]}" \
        "${tree_args[@]}" \
        trainer.experiment_name="$exp_name" \
        2>&1 | tee "$log_file"
}

# ----------------------------------------------------------------------------
# Extract per-run summary from log
# ----------------------------------------------------------------------------
_summarize_run() {
    local label=$1
    local log_file=$2

    echo ""
    echo "----- $label  ($log_file) -----"

    # 1) Top-level max_memory_allocated_gb across all training steps
    echo "[$label] actor/perf/max_memory_allocated_gb per step:"
    grep -oE "step:[0-9]+ -" "$log_file" | head -5
    grep -oE "actor/perf/max_memory_allocated_gb:np\.float64\([0-9.]+\)" "$log_file" \
        | sed 's/.*(\(.*\)).*/  \1 GB/'

    # 2) forward_step internal peaks (unique sorted)
    echo "[$label] forward_step internal peak GB (unique, sorted):"
    grep "\[MEM\]" "$log_file" | awk '{print $NF}' | sort -u | sed 's/peak=/  /'

    # 3) prefix_tree token ratio (tree only)
    if grep -q "prefix_tree/token_ratio" "$log_file" 2>/dev/null; then
        echo "[$label] prefix_tree/token_ratio per step:"
        grep -oE "prefix_tree/token_ratio:np\.float64\([0-9.]+\)" "$log_file" \
            | sed 's/.*(\(.*\)).*/  \1/'
    fi
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
TREE_LOG="$LOG_ROOT/R2_tree_memlog_${N_GPUS}gpu.log"
DENSE_LOG="$LOG_ROOT/R1_dense_memlog_${N_GPUS}gpu.log"

if [ "$ONLY" = "tree" ] || [ "$ONLY" = "both" ]; then
    _run_one tree "R2_tree_memlog_${N_GPUS}gpu" "$TREE_LOG"
fi

if [ "$ONLY" = "dense" ] || [ "$ONLY" = "both" ]; then
    _run_one dense "R1_dense_memlog_${N_GPUS}gpu" "$DENSE_LOG"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[MEMLOG] SUMMARY"
echo "=========================================="

[ "$ONLY" != "dense" ] && _summarize_run "TREE" "$TREE_LOG"
[ "$ONLY" != "tree" ]  && _summarize_run "DENSE" "$DENSE_LOG"

echo ""
echo "=========================================="
echo "[MEMLOG] DONE — full logs at:"
[ "$ONLY" != "dense" ] && echo "  $TREE_LOG"
[ "$ONLY" != "tree" ]  && echo "  $DENSE_LOG"
echo "=========================================="
