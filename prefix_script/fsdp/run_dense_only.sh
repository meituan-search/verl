#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Single-run dense baseline (R1: use_prefix_tree_dynamic=False).
# Companion to run_tree_only.sh.
#
# Usage:
#   bash prefix_script/fsdp/run_dense_only.sh                  # default STEPS=100
#   STEPS=50 bash prefix_script/fsdp/run_dense_only.sh
#   USE_WANDB=0 bash prefix_script/fsdp/run_dense_only.sh
#   EXTRA_ARGS='actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8' \
#       bash prefix_script/fsdp/run_dense_only.sh              # extra hydra overrides
#
set -ex

MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen2.5-3B-Instruct}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
LOG_ROOT=${LOG_ROOT:-/root/convergence_logs}
STEPS=${STEPS:-100}
SEED=${SEED:-42}
N_GPUS=${N_GPUS:-8}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-tree_training_convergence_e2e}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-R1_dense_${STEPS}step}

mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXPERIMENT_NAME}.log"

if [ "$USE_WANDB" = "1" ]; then
    LOGGER_ARG=("trainer.logger=[console,wandb]")
else
    LOGGER_ARG=("trainer.logger=console")
fi

EXTRA_ARGS=${EXTRA_ARGS:-}
read -ra EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"

export NCCL_NVLS_ENABLE=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.use_prefix_tree_dynamic=False \
    trainer.critic_warmup=0 \
    "${LOGGER_ARG[@]}" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEPS \
    +trainer.seed=$SEED \
    "${EXTRA_ARGS_ARR[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "[DENSE] DONE — log: $LOG_FILE"
echo "[DENSE] wandb run: $WANDB_PROJECT / $EXPERIMENT_NAME"
echo "=========================================="
