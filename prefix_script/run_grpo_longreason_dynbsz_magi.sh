#!/usr/bin/env bash
# GRPO LongReason — dynbsz + prefix-tree MAGI attention, Qwen3-1.7B-Base
# TP=1, CP=4, PP=2 (8 GPUs total)

set -xeuo pipefail
export PATH=/usr/local/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen"
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/Qwen3-1.7B}"
SPLIT="${SPLIT:-8k}"

# ── per-split token budgets ────────────────────────────────────────────────────
case "$SPLIT" in
  8k)
    MAX_PROMPT=8000
    MAX_RESPONSE=500
    MAX_TOKEN_LEN=12000
    MAX_MODEL_LEN=12000
    ;;
  16k)
    MAX_PROMPT=16384
    MAX_RESPONSE=512
    MAX_TOKEN_LEN=8192
    MAX_MODEL_LEN=17408
    ;;
  32k)
    MAX_PROMPT=32768
    MAX_RESPONSE=512
    MAX_TOKEN_LEN=16384
    MAX_MODEL_LEN=33792
    ;;
  64k)
    MAX_PROMPT=65536
    MAX_RESPONSE=512
    MAX_TOKEN_LEN=32768
    MAX_MODEL_LEN=67584
    ;;
  128k)
    MAX_PROMPT=131072
    MAX_RESPONSE=512
    MAX_TOKEN_LEN=65536
    MAX_MODEL_LEN=131584
    ;;
  *)
    echo "Unknown SPLIT=$SPLIT. Use 8k/16k/32k/64k/128k." >&2
    exit 1
    ;;
esac

TRAIN_FILES="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-djst-algoplat/yumingxuan/prefix_script/data/longreason/longreason_${SPLIT}.parquet"
REWARD_FN="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-djst-algoplat/yumingxuan/prefix_script/data/longreason/longreason_reward.py"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-/tmp/claude/grpo_longreason/magi_${SPLIT}_${TS}}"
mkdir -p "$OUTDIR"


echo "================================================================"
echo "GRPO LongReason — MAGI dynbsz  SPLIT=$SPLIT  TS=$TS"
echo "  model: $MODEL_PATH"
echo "  prompt_len=$MAX_PROMPT  response_len=$MAX_RESPONSE"
echo "  token_budget_per_gpu=$MAX_TOKEN_LEN  vllm_max_len=$MAX_MODEL_LEN"
echo "  output: $OUTDIR"
echo "================================================================"

TENSORBOARD_DIR="$OUTDIR/tb" python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=false \
    \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TRAIN_FILES" \
    data.val_max_samples=16 \
    data.train_batch_size=128 \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_prefix_tree=True \
    actor_rollout_ref.model.prefix_tree_attention=magi \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    '+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full' \
    '+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform' \
    '+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1' \
    \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.num_workers=2 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=50 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="grpo_longreason_1.7b_magi" \
    trainer.experiment_name="longreason_${SPLIT}_magi_${TS}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.balance_batch=True \
    2>&1 | tee "$OUTDIR/run.log"

    #actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
