#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# DEBUG VERSION - Lightweight configuration for debugging
# Based on run_tis_sglang.sh dataset and model paths
# ============================================================================

project_name='DAPO-FP8-ROLLOUT-DEBUG'
exp_name='DAPO-Qwen2.5-7B-SGLANG-FP8-ROLLOUT-DEBUG'

# ============================================================================
# MODIFIED: Algorithm parameters (reduced for debug)
# ============================================================================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Rollout Correction parameters for FP8 rollout
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

# ============================================================================
# MODIFIED: Reduced sequence lengths for debug
# ============================================================================
max_prompt_length=$((2048))        # MODIFIED: Reduced from 1024
max_response_length=$((1024 * 4)) # MODIFIED: Reduced from 1024 * 20
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# ============================================================================
# MODIFIED: Reduced batch sizes for debug
# ============================================================================
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5             # MODIFIED: Reduced from 10
train_prompt_bsz=8                 # MODIFIED: Reduced from 32
n_resp_per_prompt=4                # MODIFIED: Reduced from 16
train_prompt_mini_bsz=8            # MODIFIED: Reduced from 32
gen_prompt_bsz=16                  # MODIFIED: Reduced from 96

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env_fp8.yaml"}
echo "RUNTIME_ENV: ${RUNTIME_ENV}"

# ============================================================================
# MODIFIED: Single node for debug
# ============================================================================
NNODES=${NNODES:-1}                # MODIFIED: Changed from 2 to 1
echo "NNODES: ${NNODES}"

# ============================================================================
# MODIFIED: Dataset and model paths from run_tis_sglang.sh
# ============================================================================
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# MODIFIED: Using smaller model from run_tis_sglang.sh
MODEL_PATH="/workdir/quant-rollout/models/Qwen2.5-7B-Instruct"  # MODIFIED: Changed from "Qwen/Qwen3-30B-A3B-Base"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
# MODIFIED: Using datasets from run_tis_sglang.sh
TRAIN_FILE="/workdir/data/BytedTsinghua-SIA/DAPO-Math-17k/train.parquet"  # MODIFIED: Changed from "${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
TEST_FILE="/workdir/data/gsm8k/test.parquet"                             # MODIFIED: Changed from "${RAY_DATA_HOME}/data/aime-2024.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# ============================================================================
# MODIFIED: Reduced parallelism for debug
# ============================================================================
# Performance Related Parameter
sp_size=2                          # MODIFIED: Reduced from 4
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=true
gen_tp=1
train_tp=1
train_pp=1

# ============================================================================
# MODIFIED: Debug logging level
# ============================================================================
# Set Flash-RL environment variables
export VERL_LOGGING_LEVEL=DEBUG    # Keep DEBUG for debugging
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1


#data.max_prompt_length=${max_prompt_length} \
# ============================================================================
# MODIFIED: Using local Ray address for debug (from run_tis_sglang.sh)
# ============================================================================
# RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --runtime-env=${RUNTIME_ENV} \
PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.val_max_samples=10 \
    actor_rollout_ref.nccl_timeout=1800 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 16 )) \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=False \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=2 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=1 \
    trainer.total_training_steps=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    +trainer.dump_high_diff_tokens=False \
    +trainer.dump_high_diff_dir="${CKPTS_DIR}/debug_logprob_diff_dumps" \
    actor_rollout_ref.rollout.enforce_eager=True
