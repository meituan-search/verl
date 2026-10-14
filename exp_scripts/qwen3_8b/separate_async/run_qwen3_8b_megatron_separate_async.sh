#!/usr/bin/env bash
# GRPO | Qwen3-8B | Megatron training | separate_async trainer
#
# V1 PPO separate-async trainer (PPOTrainerSeparateAsync): the trainer runs on
# the global pool (trainer.nnodes x trainer.n_gpus_per_node) while a set of
# STANDALONE rollout replicas runs on its own pool
# (actor_rollout_ref.rollout.nnodes x n_gpus_per_node) and never pauses for
# training. The hybrid engine on the trainer pool alternates between trainer
# and rollout mode (switch_to_trainer/switch_to_rollout); the standalone
# replicas carry generation continuously.
#
# Decoupled PPO: within one parameter_sync_step cycle, all ppo_mini_batch
# updates use the same π_old — at local_trigger_step 0 the current weights are
# saved to CPU and every subsequent mini-batch restores them before computing
# old_log_probs (save_model_to_cpu / restore_model_from_cpu via
# DetachActorWorker). Weight updates propagate to all rollout replicas via the
# checkpoint engine at each step end.
#
# Enable via:
#   trainer.v1.trainer_mode=separate_async
#   trainer.v1.separate_async.num_warmup_batches=N  (default 1)
#   trainer.v1.separate_async.parameter_sync_step=N  (default 4)
#
# Hard constraints enforced by the trainer (asserts):
#   train_batch_size == parameter_sync_step * ppo_mini_batch_size
#   actor_rollout_ref.rollout.nnodes > 0 and n_gpus_per_node > 0
#   actor_rollout_ref.rollout.checkpoint_engine.backend != naive
#     → use nccl (or nixl/mooncake); set via CHECKPOINT_ENGINE_BACKEND.
# The per-dataset defaults below satisfy the batch-size identity with
# PARAMETER_SYNC_STEP=4 (gsm8k: 1024 = 4*256, dapo: 128 = 4*32). If you
# override TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE, re-derive
# PARAMETER_SYNC_STEP accordingly.
#
# Total GPU footprint = NGPUS_PER_NODE*NNODES (trainer pool)
#                     + ROLLOUT_NGPUS_PER_NODE*ROLLOUT_NNODES (standalone pool).
# The Ray cluster must expose both pools.
#
# ⚖️ PARITY WITH colocate_async (for A/B comparison):
#   - Total GPUs: default 4 (trainer) + 4 (standalone) = 8 = colocate_async's
#     single pool. Actor TP*PP is correspondingly 2*2 (colocate_async uses
#     4*2 on its 8-GPU pool).
#   - Training data per weight sync: colocate_async consumes train_batch_size
#     per weight update; here one sync cycle = parameter_sync_step *
#     ppo_mini_batch_size = train_batch_size (the asserted identity) — equal
#     by construction with the same DATASET defaults.
#   - Everything else (lr, kl, rollout_n, lengths, warmup, epochs) inherits
#     the same per-dataset defaults as colocate_async.
#
# INFER_BACKEND controls rollout backend: vllm | sglang | trtllm.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH="/usr/local/miniconda3/bin:$PATH"

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-sglang}
DATASET=${DATASET:-dapo}   # gsm8k | dapo — picks per-dataset defaults below
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}   # replay-buffer warmup depth

MODEL_PATH=${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B}
GSM8K_DIR=${GSM8K_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/gsm8k}
DAPO_DIR=${DAPO_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# Standalone rollout pool — dedicated GPUs outside the trainer resource pool.
# GPU-parity with colocate_async: trainer pool + standalone pool must equal
# colocate_async's NGPUS_PER_NODE*NNODES (default 8) so the two setups are
# comparable on total hardware. Default: 4 (trainer) + 4 (standalone) = 8.
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-4}

# Decoupled PPO cycle length: ppo_mini_batches per weight sync. Must satisfy
# train_batch_size == PARAMETER_SYNC_STEP * ppo_mini_batch_size (see header).
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-4}

# Weight-sync channel from trainer to rollout replicas. naive is rejected by
# the trainer; nccl for same-cluster, nixl/mooncake for cross-node transports.
CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-nccl}

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
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_${INFER_BACKEND}_megatron_${DATASET}_separate_async${affinity_tag}}

# Trainer pool is halved vs colocate_async (GPU parity, see ROLLOUT_NGPUS_PER_NODE
# comment), so TP*PP must fit NGPUS_PER_NODE*NNODES = 4: colocate_async's 4*2=8
# would not fit. 2*2=4 keeps both parallelism degrees.
actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-2}

rollout_tp=${ROLLOUT_TP:-2}
# Applies to BOTH hybrid replicas (sharing the trainer pool) and standalone
# replicas (dedicated GPUs). Note: standalone_gpu_memory_utilization is NOT
# honored on this path (only the experimental fully_async rollouter reads it),
# so a per-mode split is not possible here — pick a value that fits the hybrid
# replicas' shared-GPU constraint.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

# Group affinity ablation switch (Q16). When False (default): agent_loop_tq
# does not stamp group_id, verl LB falls back to per-sibling request_id sticky
# routing, SGLang scheduler retract/admit affinity no-ops, and the P2 cohesion
# metric (offpolicy_token/group_version_set_jaccard_mean) is suppressed. When
# True: current full affinity path is preserved.
ENABLE_GROUP_AFFINITY=${ENABLE_GROUP_AFFINITY:-False}
affinity_tag=$([ "$ENABLE_GROUP_AFFINITY" = "true" ] && echo "_affinity" || echo "_no_affinity")
########################### end user-adjustable ###########################

if [ "$((train_batch_size))" -ne "$((PARAMETER_SYNC_STEP * ppo_mini_batch_size))" ]; then
    echo "train_batch_size (${train_batch_size}) != PARAMETER_SYNC_STEP (${PARAMETER_SYNC_STEP}) * ppo_mini_batch_size (${ppo_mini_batch_size})" >&2
    echo "The separate_async trainer asserts this identity — re-derive PARAMETER_SYNC_STEP." >&2
    exit 1
fi

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
    # Standalone rollout pool (dedicated GPUs, never paused for training).
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
    # Weight sync channel trainer → rollout; naive is rejected by the trainer.
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND}
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=44398 \
    # Q16 group affinity ablation switch.
    actor_rollout_ref.rollout.enable_group_affinity=${ENABLE_GROUP_AFFINITY}
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
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}
    transfer_queue.enable=True
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
