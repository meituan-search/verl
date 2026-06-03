#!/usr/bin/env bash
# Test script for router strategy comparison
#
# Purpose: Compare two router strategies for LLM rollout replica scheduling.
#
#   ROUTER=capacity_aware  (default)
#     CapacityAwareScheduler: gates new-group dispatch on KV-cache utilization.
#     New groups block until a replica's effective token usage < CAPACITY_THRESHOLD.
#     Established groups are sticky (prefix-cache warm) regardless of load.
#
#   ROUTER=inflight
#     Legacy GlobalRequestLoadBalancer behaviour, emulated by setting
#     capacity_threshold=1.0 so the capacity gate never fires.
#     Routing is sticky-session only; no KV-cache awareness.
#
# Usage:
#   bash test_capacity_aware_scheduler.sh                           # capacity_aware, threshold=0.85
#   ROUTER=inflight bash test_capacity_aware_scheduler.sh
#   ROUTER=capacity_aware CAPACITY_THRESHOLD=0.70 bash test_capacity_aware_scheduler.sh
#   ROUTER=capacity_aware LB_BACKEND=sglang bash test_capacity_aware_scheduler.sh
#
# GPU layout: 3 nodes x 8 GPUs (colocated, Megatron backend)
#
# What to observe (capacity_aware):
#   - Training completes without deadlock (acquire_server never permanently blocked)
#   - No "poll failed" WARNING spam in logs (backends reachable)
#   - Same group_id consistently hits the same replica (sticky affinity)
#   - When all replicas saturated, acquire_server blocks then unblocks after poll

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1  # For Megatron communication/computation overlapping
project_name=test_router
exp_name=test_capacity_aware_scheduler

# ── Paths ──────────────────────────────────────────────────────────────────────
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/model/Qwen2___5-Math-7B
TRAIN_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet

# Rollout settings
rollout_mode="async"
rollout_name="sglang"  # vllm → VLLMLoadBackend（Prometheus /metrics 路径）
export VLLM_USE_V1=1
return_raw_chat="True"

# ── Router strategy ────────────────────────────────────────────────────────────
# capacity_aware : CapacityAwareScheduler — KV-cache–gated (default, recommended)
# inflight       : GlobalRequestLoadBalancer — legacy sticky + least-in-flight
ROUTER=${ROUTER:-"capacity_aware"}

case "${ROUTER}" in
    capacity_aware)
        load_balance_scheduler="capacity_aware"
        load_balance_threshold=${CAPACITY_THRESHOLD:-0.85}
        load_balance_poll_ms=${POLL_INTERVAL_MS:-200}
        load_balance_backend=${rollout_name:-"vllm"}
        ;;
    inflight)
        load_balance_scheduler="inflight"
        # threshold / poll_ms / backend are unused by GlobalRequestLoadBalancer
        # but must still be valid values for config parsing
        load_balance_threshold=0.85
        load_balance_poll_ms=200
        load_balance_backend="vllm"
        ;;
    *)
        echo "ERROR: unknown ROUTER='${ROUTER}'. Choose 'capacity_aware' or 'inflight'."
        exit 1
        ;;
esac

# Algorithm
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Training
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance: use_dynamic_bsz=True enables sequence packing for long-context efficiency
use_dynamic_bsz=True
# offload=True to handle memory pressure from long sequences (30k tokens) + dynamic batching
offload=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=2
train_tp=4
train_pp=2

# Cluster layout: 4 nodes x 8 GPUs (colocated, shared)
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Batch sizes
n_resp_per_prompt=16
train_prompt_bsz=512   # standard sync PPO batch size
train_prompt_mini_bsz=32
total_training_steps=400
test_freq=20

exp_name="router-${ROUTER}-e2e"

echo "============================================================"
echo " Router Strategy E2E Test"
echo "   router          : ${ROUTER}"
echo "   rollout backend : ${rollout_name}"
echo "   lb backend      : ${load_balance_backend}"
echo "   capacity_threshold: ${load_balance_threshold}"
echo "   poll_interval_ms  : ${load_balance_poll_ms}"
echo "============================================================"

python -X faulthandler -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.load_balance.scheduler=${load_balance_scheduler} \
    actor_rollout_ref.rollout.load_balance.capacity_threshold=${load_balance_threshold} \
    actor_rollout_ref.rollout.load_balance.poll_interval_ms=${load_balance_poll_ms} \
    actor_rollout_ref.rollout.load_balance.backend=${load_balance_backend} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_epochs=10 \
    trainer.test_freq="${test_freq}" \
    trainer.total_training_steps="${total_training_steps}" \
    $@

echo "============================================================"
echo " Router Strategy E2E Test PASSED  (ROUTER=${ROUTER})"
echo "============================================================"
