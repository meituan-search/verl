#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# E2E convergence test for dynamic prefix-tree + Magi on FSDP2.
#
# Runs 3 training jobs (same seed, same hyperparams) and compares step-by-step
# metrics:
#
#   R1 dense                  : use_prefix_tree_dynamic=False (ground truth)
#   R2 tree non-CP            : use_prefix_tree_dynamic=True, context_parallel_size=1
#   R3 tree CP=2              : use_prefix_tree_dynamic=True, context_parallel_size=2
#
# Then compares R2 vs R1 (non-CP convergence) and R3 vs R1 (CP=2 convergence)
# via prefix_script/fsdp/compare_training_metrics.py.
#
# Stack: Qwen2.5-3B-Instruct + GSM8K + GRPO + FSDP2 + Magi FFA + vLLM.
# Hardware: 8×H100 (single node), ~2.5-3 hr total for STEPS=150.
#
# Usage:
#   bash prefix_script/fsdp/run_e2e_convergence.sh
#   STEPS=10 bash prefix_script/fsdp/run_e2e_convergence.sh   # smoke
#   SKIP_R3=1 bash prefix_script/fsdp/run_e2e_convergence.sh  # skip CP run
#   USE_WANDB=1 bash prefix_script/fsdp/run_e2e_convergence.sh # also log to wandb
#       (requires `wandb login` or WANDB_API_KEY env; wandb project name
#        = $WANDB_PROJECT, default 'tree_training_convergence_e2e')
#
set -ex

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/convergence_e2e.XXXXXX)}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-tree_training_convergence_e2e}
if [ "$USE_WANDB" = "1" ]; then
    # console kept so compare_training_metrics.py can still parse step lines
    LOGGER_ARG=("trainer.logger=[console,wandb]")
else
    LOGGER_ARG=("trainer.logger=console")
fi

STEPS=${STEPS:-150}
ROLLOUT_N=${ROLLOUT_N:-8}
SEED=${SEED:-42}
N_GPUS=${N_GPUS:-8}
SKIP_R3=${SKIP_R3:-0}

echo "=========================================="
echo "[E2E] Convergence test: dynamic prefix-tree + Magi vs dense"
echo "[E2E]   model:   $MODEL_PATH"
echo "[E2E]   data:    $DATA_DIR"
echo "[E2E]   steps:   $STEPS    rollout_n: $ROLLOUT_N    seed: $SEED"
echo "[E2E]   N_GPUS:  $N_GPUS"
echo "[E2E]   logs:    $LOG_DIR"
echo "=========================================="

# ----------------------------------------------------------------------------
# Common config — IDENTICAL across all 3 runs so the only varying knob is
# (use_prefix_tree_dynamic, context_parallel_size). Critical points:
#   - use_dynamic_bsz=False: stable per-step token budget, so micro-batch shapes
#     match exactly across runs
#   - ulysses_sequence_parallel_size=1: Magi CP and Ulysses SP are mutually
#     exclusive (see FSDPEngine._init_device_mesh assertion). All 3 runs use
#     ulysses_sp=1 so the dense baseline matches the tree runs' parallel layout.
#   - strategy=fsdp2 (top-level actor.strategy): dynamic + Magi REQUIRES FSDP2;
#     ActorConfig.__post_init__ syncs top-level into engine.strategy.
#   - +trainer.seed=$SEED: enforces deterministic rollout sampling order.
# ----------------------------------------------------------------------------
COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size=32
    data.max_prompt_length=512
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.critic_warmup=0
    "${LOGGER_ARG[@]}"
    trainer.project_name=$WANDB_PROJECT
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=$STEPS
    +trainer.seed=$SEED
)

# ----------------------------------------------------------------------------
# Run order: tree runs FIRST so we fail fast on tree-side bugs without
# spending baseline GPU-time. Dense baseline runs last; without it we still
# know whether the tree path itself trained successfully.
# ----------------------------------------------------------------------------

# R2: Tree non-CP. use_prefix_tree_dynamic=True (cp_size=1).
echo ""
echo "=========================================="
echo "[E2E] R2: Tree non-CP (use_prefix_tree_dynamic=True, cp_size=1)"
echo "=========================================="
python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_prefix_tree_dynamic=True \
    actor_rollout_ref.actor.prefix_tree_attention=magi \
    actor_rollout_ref.actor.context_parallel_size=1 \
    trainer.experiment_name=R2_tree_noncp \
    2>&1 | tee "$LOG_DIR/R2_tree_noncp.log"

# R3: Tree CP=2. context_parallel_size=2 (8 GPU = 4 DP × 2 CP).
if [ "$SKIP_R3" != "1" ]; then
    echo ""
    echo "=========================================="
    echo "[E2E] R3: Tree CP=2 (use_prefix_tree_dynamic=True, cp_size=2)"
    echo "=========================================="
    python3 -m verl.trainer.main_ppo \
        "${COMMON_ARGS[@]}" \
        actor_rollout_ref.actor.use_prefix_tree_dynamic=True \
        actor_rollout_ref.actor.prefix_tree_attention=magi \
        actor_rollout_ref.actor.context_parallel_size=2 \
        trainer.experiment_name=R3_tree_cp2 \
        2>&1 | tee "$LOG_DIR/R3_tree_cp2.log"
fi

# R1: Dense baseline (no prefix tree). The "gold ground truth" trajectory,
# run last because the tree runs above are what we're actually validating.
echo ""
echo "=========================================="
echo "[E2E] R1: Dense baseline (no prefix tree)"
echo "=========================================="
python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_prefix_tree_dynamic=False \
    trainer.experiment_name=R1_dense \
    2>&1 | tee "$LOG_DIR/R1_dense.log"

# ----------------------------------------------------------------------------
# Compare: R2 vs R1 (non-CP convergence), R3 vs R1 (CP convergence)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[E2E] Comparing metrics: R2 (tree non-CP) vs R1 (dense)"
echo "=========================================="
python3 prefix_script/fsdp/compare_training_metrics.py \
    --baseline_log "$LOG_DIR/R1_dense.log" \
    --treatment_log "$LOG_DIR/R2_tree_noncp.log" \
    --label "R2 tree non-CP vs R1 dense" \
    --output "$LOG_DIR/compare_R2_vs_R1.json"
RC_R2=$?

RC_R3=0
if [ "$SKIP_R3" != "1" ]; then
    echo ""
    echo "=========================================="
    echo "[E2E] Comparing metrics: R3 (tree CP=2) vs R1 (dense)"
    echo "=========================================="
    python3 prefix_script/fsdp/compare_training_metrics.py \
        --baseline_log "$LOG_DIR/R1_dense.log" \
        --treatment_log "$LOG_DIR/R3_tree_cp2.log" \
        --label "R3 tree CP=2 vs R1 dense" \
        --output "$LOG_DIR/compare_R3_vs_R1.json"
    RC_R3=$?
fi

echo ""
echo "=========================================="
echo "[E2E] DONE — logs + JSON in $LOG_DIR"
echo "=========================================="
if [ $RC_R2 -eq 0 ] && [ $RC_R3 -eq 0 ]; then
    echo "[E2E] OVERALL: PASS"
    exit 0
else
    echo "[E2E] OVERALL: FAIL (RC_R2=$RC_R2 RC_R3=$RC_R3)"
    exit 1
fi
