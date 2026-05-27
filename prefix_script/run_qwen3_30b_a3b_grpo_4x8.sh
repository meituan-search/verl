#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B | vLLM rollout | FSDP2 | 4 nodes × 8 GPUs (32 total)
#
# Direct run:
#   bash run_qwen3_30b_a3b_grpo_4x8.sh
#   USE_PG=true bash run_qwen3_30b_a3b_grpo_4x8.sh
#   USE_MAGI=true bash run_qwen3_30b_a3b_grpo_4x8.sh
#
# ray job submit (uses system libs, no working_dir upload):
#   ray job submit \
#     --address http://HEAD:8265 \
#     --runtime-env magi_script/ray_runtime_env.yaml \
#     -- bash /abs/path/to/magi_script/run_qwen3_30b_a3b_grpo_4x8.sh

set -xeuo pipefail

export OMP_NUM_THREADS=1

########################### user-adjustable ###########################
MODEL_BASE="${MODEL_BASE:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen}"
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/Qwen3-30B-A3B}"

NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Batch sizing for 32 GPUs
train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MBS:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

# vLLM: TP=4 → 2 rollout workers per node × 4 nodes = 8 workers
rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

GSM8K_TRAIN="${GSM8K_TRAIN:-$HOME/data/gsm8k/train.parquet}"
GSM8K_TEST="${GSM8K_TEST:-$HOME/data/gsm8k/test.parquet}"

USE_PG="${USE_PG:-false}"
USE_MAGI="${USE_MAGI:-false}"

project_name=${PROJECT_NAME:-grpo_qwen3_30b_a3b}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_4x8_pg${USE_PG}_magi${USE_MAGI}}
########################### end user-adjustable ###########################

# For ray job submit with system libs, verl is on PYTHONPATH already.
# For direct run, cd into source to make it importable.
VERL_DIR="${VERL_DIR:-/home/hadoop-djst-algoplat/prefix-tree/verl_meituan}"
[ -d "$VERL_DIR" ] && cd "$VERL_DIR"

LOG_DIR="/home/hadoop-djst-algoplat/prefix-tree/magi_script/logs/${experiment_name}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "GRPO Qwen3-30B-A3B  4×8 nodes"
echo "  model      : $MODEL_PATH"
echo "  nodes      : $NNODES × $NGPUS_PER_NODE GPUs"
echo "  PrefixGrp  : $USE_PG"
echo "  Magi       : $USE_MAGI"
echo "  log        : $LOG_DIR/stdout.log"
echo "================================================================"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${GSM8K_TRAIN}"
    data.val_files="${GSM8K_TEST}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.use_prefix_grouper="${USE_PG}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.use_prefix_tree="${USE_MAGI}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=False
)

REWARD=(
    # GSM8K uses built-in reward via data_source column
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/stdout.log"
