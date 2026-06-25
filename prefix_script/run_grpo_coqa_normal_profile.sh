#!/usr/bin/env bash
# GRPO CoQA — FA3 BASELINE (Megatron, no prefix tree)
# MBS=8, n=8 for parity with MAGI profiling config, 30 steps
#
# Usage:
#   bash run_grpo_coqa_normal_profile.sh

set -xeuo pipefail
export PATH=/usr/local/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export HYDRA_FULL_ERROR=1

MODEL_PATH="${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-4B-Base}"
# Dataset: coqa_grpo.parquet prepared from CoQA via prefix_script/data/coqa/prepare_coqa_grpo.py
TRAIN_FILES="${TRAIN_FILES:-/home/hadoop-djst-algoplat/prefix-tree/verl_prefix_tree/prefix_script/data/coqa/coqa_grpo.parquet}"
REWARD_FN="${REWARD_FN:-/home/hadoop-djst-algoplat/prefix-tree/verl_prefix_tree/prefix_script/data/coqa/coqa_reward.py}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-/tmp/verl_submit/profiles/fa3_baseline/${TS}}"
mkdir -p "$OUTDIR"


echo "================================================================"
echo "FA3 BASELINE — Megatron, no prefix tree"
echo "  model: $MODEL_PATH  steps: 30"
echo "  output: $OUTDIR"
echo "================================================================"

mkdir -p "$HOME/profiles/fa3_${TS}/tb"
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TRAIN_FILES" \
    data.val_max_samples=32 \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.num_workers=2 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=30 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=grpo_coqa_4b_profile \
    trainer.experiment_name="fa3_baseline_${TS}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.balance_batch=True \
    2>&1 | tee "$OUTDIR/run.log"

echo "================================================================"
echo "FA3 baseline done → $OUTDIR"
echo "================================================================"
