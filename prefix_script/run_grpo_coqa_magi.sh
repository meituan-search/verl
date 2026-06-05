#!/usr/bin/env bash
# GRPO CoQA — Megatron actor + MAGI prefix tree, Qwen3-0.6B, 50 steps
# Actor training deduplicates shared prompt prefix across rollout.n=4 responses
# Expected prefix_sharing_ratio ~0.6-0.7 (prompt ~700 tok shared / total ~800 tok)

set -xeuo pipefail
export PATH=/usr/local/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export HYDRA_FULL_ERROR=1

MODEL_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen"
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/Qwen3-4B-Base}"
# Dataset: coqa_grpo.parquet prepared from CoQA via prefix_script/data/coqa/prepare_coqa_grpo.py
TRAIN_FILES="${VERL_DIR:-/home/hadoop-djst-algoplat/prefix-tree/verl_prefix_tree}/prefix_script/data/coqa/coqa_grpo.parquet"
REWARD_FN="${VERL_DIR:-/home/hadoop-djst-algoplat/prefix-tree/verl_prefix_tree}/prefix_script/data/coqa/coqa_reward.py"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-/tmp/claude/grpo_coqa_4b/magi/${TS}}"
mkdir -p "$OUTDIR"

VERL_DIR="${VERL_DIR:-/home/hadoop-djst-algoplat/prefix-tree/verl_prefix_tree}"
cd "$VERL_DIR"

echo "================================================================"
echo "GRPO CoQA — Megatron actor + MAGI prefix tree"
echo "  model : $MODEL_PATH  steps: 50"
echo "  actor : Megatron TP=4, use_prefix_tree=True, attn=magi"
echo "  rollout: vllm TP=4, n=4"
echo "  Watch: train/prefix_sharing_ratio — expect >0.5"
echo "================================================================"

TENSORBOARD_DIR="$OUTDIR/tb" python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.use_prefix_tree=True \
    actor_rollout_ref.model.prefix_tree_attention=magi \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.use_prefix_tree=True \
    \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.num_workers=2 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=100 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=grpo_coqa_4b \
    trainer.experiment_name=coqa_magi_4b \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.balance_batch=True \
    2>&1 | tee "$OUTDIR/run.log"
