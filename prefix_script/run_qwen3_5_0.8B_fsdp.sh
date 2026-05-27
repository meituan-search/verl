#!/usr/bin/env bash
# GRPO smoke test for Qwen3.5-0.8B on local 8×H20
# Tests: PrefixGrouper (full-attn) + SSM prefix cache flag (GDN layers)
#
# Usage:
#   bash run_qwen3_5_0.8B_fsdp.sh                      # baseline
#   USE_PG=true bash run_qwen3_5_0.8B_fsdp.sh           # + PrefixGrouper
#   USE_SSM_CACHE=true bash run_qwen3_5_0.8B_fsdp.sh    # + SSM prefix cache (GDN)
#   USE_PG=true USE_SSM_CACHE=true bash ...             # both

set -xeuo pipefail

export PATH=/usr/local/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
# vllm 0.19.1 + torch 2.10.0 (user install): use SDPA, disable broken system flash_attn
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

MODEL_BASE="${MODEL_BASE:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen}"
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/Qwen3.5-0.8B}"

DATA_DIR="/tmp/claude/grpo_bench"
VERL_DIR="${VERL_DIR:-/home/hadoop-djst-algoplat/prefix-tree/verl-prefix-tree}"

USE_PG="${USE_PG:-false}"
USE_SSM_CACHE="${USE_SSM_CACHE:-false}"

LOG_TAG="qwen3_5-0.8B_pg${USE_PG}_ssm${USE_SSM_CACHE}"
LOG_DIR="/home/hadoop-djst-algoplat/prefix-tree/magi_script/logs/${LOG_TAG}"
mkdir -p "$LOG_DIR"

cd "$VERL_DIR"

echo "================================================================"
echo "Qwen3.5-0.8B GRPO smoke test"
echo "  model          : $MODEL_PATH"
echo "  use_prefix_grouper: $USE_PG"
echo "  use_ssm_cache  : $USE_SSM_CACHE"
echo "  log            : $LOG_DIR/stdout.log"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=18000 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.prompt_key=prompt \
    data.return_raw_chat=False \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_prefix_grouper="${USE_PG}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.prompt_length=18000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    reward.custom_reward_function.path="/home/hadoop-djst-algoplat/prefix-tree/magi_script/grpo_bench_reward.py" \
    reward.num_workers=1 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=3 \
    trainer.logger='["console"]' \
    trainer.project_name=grpo_qwen35_bench \
    trainer.experiment_name="${LOG_TAG}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    2>&1 | tee "${LOG_DIR}/stdout.log"
