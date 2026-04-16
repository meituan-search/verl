#!/usr/bin/env bash
# Experiment 4: Fully Async + Recompute via ModelEngineServer (Dedicated GPU Pool)
#
# Purpose: Validate the model_engine_server approach where old_log_prob computation
#          is offloaded to a dedicated, standalone GPU pool that runs in parallel with
#          training. The trainer receives pre-computed engine_server_logprobs and incurs
#          zero recomputation overhead on the training GPUs.
#
# GPU layout:
#   - Rollout:            2 nodes x 8 GPUs = 16 GPUs  (vLLM generation)
#   - Trainer:            2 nodes x 8 GPUs = 16 GPUs  (Megatron training)
#   - ModelEngineServer:  1 node  x 8 GPUs =  8 GPUs  (dedicated old_log_prob server)
#   - Total: 40 GPUs  (+8 vs Exp2/3)
#
# Key settings:
#   - algorithm.rollout_correction.bypass_mode=False   (enable recompute)
#   - model_engine_server.enable_standalone=True        (dedicated server pool)
#   - model_engine_server.nnodes=1 / n_gpus_per_node=8
#   - model_engine_server.batch_size=-1, timeout=5     (async batching)
#
# Metrics to record (vs Exp3):
#   - Whether trainer-side recompute overhead is eliminated
#   - Server GPU utilization and batch fill rate (timeout-triggered vs full-batch)
#   - End-to-end throughput vs Exp3 (extra 8 GPUs cost vs latency gain)
#   - Final model quality (AIME 2024 pass@1) — should match Exp3 (same correctness)
#
# Optional fair-GPU variant: reduce NNODES_ROLLOUT or NNODES_TRAIN by 1 to keep
# total GPUs at 32, then compare throughput with Exp2/3.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1  # For Megatron communication/computation overlapping

project_name='model_engine_server_exp'
exp_name='exp4_fully-async_model-engine-server_dapo_qwen2.5-7B-math_megatron'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/model/Qwen2___5-Math-7B
TRAIN_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet

# Rollout settings
rollout_mode="async"
rollout_name="vllm"
export VLLM_USE_V1=1
return_raw_chat="True"

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
max_response_length=$((1024 * 28))
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
gen_tp=4
train_tp=4
train_pp=2

# Cluster layout
NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}     # 16 GPUs for rollout
NNODES_TRAIN=${NNODES_TRAIN:-2}         # 16 GPUs for training
NNODES_LOG_PROB=${NNODES_LOG_PROB:-1}   #  8 GPUs for model_engine_server
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Batch / async parameters
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512 * 400)))
test_freq=20
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True
lr_decay_steps=$(((4 * 4 * 400)))

python -X faulthandler -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=0 \
    data.gen_batch_size=${gen_prompt_bsz} \
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
    actor_rollout_ref.hybrid_engine=False \
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
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
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
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.total_epochs=10 \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    algorithm.rollout_correction.bypass_mode=False \
    model_engine_server.enable_standalone=True \
    model_engine_server.nnodes="${NNODES_LOG_PROB}" \
    model_engine_server.n_gpus_per_node="${NGPUS_PER_NODE}" \
    model_engine_server.use_dynamic_bsz=${use_dynamic_bsz} \
    model_engine_server.megatron.pipeline_model_parallel_size=1 \
    model_engine_server.megatron.tensor_model_parallel_size=2 \
    model_engine_server.batch_size=32 \
    model_engine_server.timeout=5
