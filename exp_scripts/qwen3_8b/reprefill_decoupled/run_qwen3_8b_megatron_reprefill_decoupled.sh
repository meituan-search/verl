#!/usr/bin/env bash
# GRPO | Qwen3-8B | Megatron training | reprefill_decoupled trainer
#
# V1 PPO colocate-async trainer where π_b (old_log_probs) is computed by
# re-prefilling consumed trajectories on the rollout engine at its current
# weight W_{k-1}, replacing the trainer-side old_log_prob forward pass.
# Optional P2 pipelined pre-dispatch overlaps re-prefill with remaining
# generation as samples finish during the replay-buffer poll loop.
#
# Enable via:
#   trainer.v1.trainer_mode=reprefill_decoupled
#   trainer.v1.reprefill_decoupled.num_warmup_batches=N  (default 1)
#   trainer.v1.reprefill_decoupled.enable_prefill_pipeline={true|false}  (default false — P2)
#
# Rollout-correction presets (e.g. decoupled token-IS, bypass) are passed via
# +algorithm.rollout_correction=... overrides on the command line; see README.md
# for the three acceptance arms.
#
# INFER_BACKEND controls rollout backend: vllm | sglang | trtllm.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH="/usr/local/miniconda3/bin:$PATH"

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-sglang}
DATASET=${DATASET:-dapo}   # gsm8k | dapo — picks per-dataset defaults below

MODEL_PATH=${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B}
GSM8K_DIR=${GSM8K_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/gsm8k}
DAPO_DIR=${DAPO_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Number of warmup batches before the training loop starts. Increasing this
# (e.g. 4-8x the default) deepens staleness in the replay buffer for Arm 1
# (negative control) stress testing.
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}

# P2 pipelined pre-dispatch: overlap re-prefill with remaining generation as
# samples finish during the replay-buffer poll loop. false = P1 (post-hoc
# re-prefill); true = P2 (pipelined pre-dispatch).
ENABLE_PREFILL_PIPELINE=${ENABLE_PREFILL_PIPELINE:-false}

# Per-dataset defaults. Any knob can still be overridden via its env var
# (TRAIN_FILES, TRAIN_BATCH_SIZE, ACTOR_LR, ...). DATASET just picks the base.
case "$DATASET" in
    gsm8k)
        train_files=${TRAIN_FILES:-"['$GSM8K_DIR/train.parquet']"}
        val_files=${VAL_FILES:-"['$GSM8K_DIR/test.parquet']"}
        train_batch_size=${TRAIN_BATCH_SIZE:-1024}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
        max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
        max_response_length=${MAX_RESPONSE_LENGTH:-2048}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
        rollout_n=${ROLLOUT_N:-5}
        ;;
    dapo)
        train_files=${TRAIN_FILES:-"['$DAPO_DIR/dapo-math-17k.parquet']"}
        val_files=${VAL_FILES:-"['$DAPO_DIR/aime-2024.parquet']"}
        train_batch_size=${TRAIN_BATCH_SIZE:-128}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
        max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
        max_response_length=${MAX_RESPONSE_LENGTH:-20480}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
        rollout_n=${ROLLOUT_N:-8}
        ;;
    *)
        echo "Unknown DATASET=$DATASET (expected: gsm8k | dapo)" >&2
        exit 1
        ;;
esac

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
project_name=${PROJECT_NAME:-verl_grpo_${DATASET}_math}
pipeline_tag=$([ "$ENABLE_PREFILL_PIPELINE" = "true" ] && echo "_p2" || echo "_p1")
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_${INFER_BACKEND}_megatron_${DATASET}_reprefill_decoupled${pipeline_tag}}

actor_tp=${ACTOR_TP:-4}
actor_pp=${ACTOR_PP:-2}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=16
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=44398 \
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.val_before_train=False
    trainer.total_epochs=${total_epochs}
    trainer.use_v1=true
    trainer.v1.trainer_mode=reprefill_decoupled
    trainer.v1.reprefill_decoupled.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.reprefill_decoupled.enable_prefill_pipeline=${ENABLE_PREFILL_PIPELINE}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
