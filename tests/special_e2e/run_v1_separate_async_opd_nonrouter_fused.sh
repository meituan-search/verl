#!/usr/bin/env bash
set -xeuo pipefail

# Workaround for NVIDIA driver bug (r560-r575) causing SIGSEGV in ncclCuMemHostEnable()
# on PCIe machines without P2P access. See: https://github.com/NVIDIA/nccl/issues/1838
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

# ============================================================================
# V1 separate-async OPD — nonrouter multi-teacher FUSED mode
# ============================================================================
#
# Key difference from the old run_fully_async_opd_nonrouter.sh:
#   Teachers are now standalone TrainingWorker instances living INSIDE the
#   ActorRolloutRefWorker process (self.teacher dict in engine_workers.py).
#   Each teacher is a frozen Megatron engine that shares the student's
#   TP/PP/EP parallelism. They are NOT sglang rollout replicas.
#
#   Therefore the fused script does NOT need:
#     - inference.name / inference.tensor_model_parallel_size / inference.* configs
#     - num_replicas (teachers have no dedicated rollout processes)
#     - n_gpus_per_node / nnodes for teachers (they share the trainer pool)
#
#   The fused script DOES need:
#     - teacher_execution=trainer
#     - model_path + key per teacher (for weight loading into the Megatron engine)
#
#   V1 keeps student params CPU-resident outside actor eval/train contexts.
#   A pre-teacher offload also handles NCCL synchronization leftovers.
# ============================================================================

############################ Quick Config ############################

ROLLOUT_NAME="sglang"

# Keep MTP/speculative decoding disabled while validating fused multi-teacher OPD.
mtp_params=(
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=False
)

MODEL_35B="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/deepsearch_files_xtssd/LLMbasemodels/huggingface.co/Qwen/Qwen3.5-35B-A3B"
STUDENT_MODEL=${STUDENT_MODEL:-"${MODEL_35B}"}
MATH_DAPO_TEACHER_MODEL=${MATH_DAPO_TEACHER_MODEL:-"${MODEL_35B}"}
AIME_2024_TEACHER_MODEL=${AIME_2024_TEACHER_MODEL:-"${MODEL_35B}"}

DISTILLATION_LOSS_MODE="k1"
USE_POLICY_GRADIENT=True

MAX_PROMPT=${MAX_PROMPT:-1600}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# Fully async resources.
# Fused mode: teachers live inside the trainer process, so no separate teacher nodes.
ROLLOUT_NNODES=2          # 16 rollout GPUs
TRAINER_NNODES=2          # 16 trainer GPUs (student + 2 fused teachers share this pool)
N_GPUS_ROLLOUT=8
N_GPUS_TRAINING=8

# Megatron parallelism (35B-A3B MoE)
GEN_TP=8
TRAIN_TP=8
TRAIN_PP=2

STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/maoshizhuo/checkpoints"}
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/fully_async_opd_nonrouter_fused_addteacher"

############################ Data ############################

DAPO_PART1_TRAIN="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/maoshizhuo/dataset/dapo-math-part1.parquet"
DAPO_PART2_TRAIN="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/maoshizhuo/dataset/dapo-math-part2.parquet"

TRAIN_FILES="['${DAPO_PART1_TRAIN}','${DAPO_PART2_TRAIN}']"
TEST_FILES="['${DAPO_PART1_TRAIN}']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=64
    data.gen_batch_size=1
    data.return_raw_chat=True
    data.image_key=images
    data.shuffle=True
    data.seed=42
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

STUDENT=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.lr_decay_steps=10000000
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_NUM_TOKENS
    # V1 contexts load the actor on demand and return it to CPU afterwards.
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
)

# Shared Megatron runtime for all trainer-colocated fused teachers.
# Teacher identities/model paths/routing stay under distillation.teacher_models.
TEACHER_RUNTIME=(
    actor_rollout_ref.teacher.strategy=megatron
    actor_rollout_ref.teacher.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
    actor_rollout_ref.teacher.megatron.vanilla_mbridge=True
    actor_rollout_ref.teacher.megatron.param_offload=True
    actor_rollout_ref.teacher.megatron.optimizer_offload=False
    actor_rollout_ref.teacher.megatron.grad_offload=False
    actor_rollout_ref.teacher.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.teacher.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.teacher.megatron.expert_model_parallel_size=8
    actor_rollout_ref.teacher.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.teacher.megatron.context_parallel_size=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.prometheus.enable=True
    actor_rollout_ref.rollout.prometheus.port=44398
    ++actor_rollout_ref.rollout.prometheus.served_model_name=student_rollout
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl'
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
)

# ============================================================================
# Distillation config for FUSED (trainer-colocated) multi-teacher OPD.
#
# Teachers are standalone Megatron TrainingWorker instances inside the
# ActorRolloutRefWorker process. They do NOT use sglang inference configs.
# Their shared Megatron runtime is configured under actor_rollout_ref.teacher;
# only model_path and key remain here for model loading and routing.
# ============================================================================
DISTILLATION=(
    distillation.enabled=True
    +distillation.nonrouter=True
    distillation.teacher_execution=trainer
    distillation.teacher_key=data_source

    # Teachers share the trainer's resource pool — no dedicated GPUs/nodes.
    distillation.n_gpus_per_node=0
    distillation.nnodes=0

    # --- Teacher 1: math_dapo ---
    +distillation.teacher_models.t1.key="math_dapo"
    +distillation.teacher_models.t1.model_path="${MATH_DAPO_TEACHER_MODEL}"

    # --- Teacher 2: aime_2024 ---
    +distillation.teacher_models.t2.key="aime_2024"
    +distillation.teacher_models.t2.model_path="${AIME_2024_TEACHER_MODEL}"

    # --- Loss ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=True
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","tensorboard"]'
    trainer.project_name='fully-async-opd-nonrouter'
    trainer.experiment_name="v1_separate_async_nonrouter_fused"
    trainer.val_before_train=False
    trainer.save_freq=1000000
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=disable
    trainer.nnodes=${TRAINER_NNODES}
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=0
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-100}
    trainer.test_freq=-1
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=1
    trainer.v1.separate_async.parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}
    trainer.v1.sampler.max_off_policy_threshold=8
    trainer.v1.sampler.max_off_policy_strategy=drop
    transfer_queue.enable=True
)

ASYNC_TRAINING=(
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
)

############################ Launch ############################

echo "Running V1 separate_async nonrouter multi-teacher OPD: FUSED teachers"
echo "Student: ${STUDENT_MODEL}"
echo "Teachers: math_dapo -> ${MATH_DAPO_TEACHER_MODEL}, aime_2024 -> ${AIME_2024_TEACHER_MODEL}"
echo "Datasets: ${DAPO_PART1_TRAIN} (math_dapo), ${DAPO_PART2_TRAIN} (aime_2024)"
echo "Single-turn: prompt=${MAX_PROMPT}, response=${MAX_RESPONSE_LENGTH}, total_tokens=${MAX_NUM_TOKENS}"
echo "MTP/speculative decoding: disabled"
echo "GPUs: ${N_GPUS_ROLLOUT}x${ROLLOUT_NNODES} rollout + ${N_GPUS_TRAINING}x${TRAINER_NNODES} fused (student + 2 trainer-colocated teachers)"
echo "Student offload: V1 context-managed param_offload=True plus pre-teacher safety offload"
echo "Final checkpoint: ${CHECKPOINT_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer.yaml' \
    actor_rollout_ref.hybrid_engine=True \
    critic.strategy=megatron \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${STUDENT[@]}" \
    "${TEACHER_RUNTIME[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "${mtp_params[@]}" \
    "$@"

echo "V1 separate_async nonrouter multi-teacher OPD completed successfully"
