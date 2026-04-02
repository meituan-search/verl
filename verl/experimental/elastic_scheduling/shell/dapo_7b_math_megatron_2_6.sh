#!/usr/bin/env bash
# Elastic Scheduling: Single-node 8-GPU scenario
#   Fixed Rollout  : 2 GPUs  (rollout.nnodes=1, rollout.n_gpus_per_node=2)
#   Elastic Trainer: 6 GPUs total  (trainer.nnodes=1, trainer.n_gpus_per_node=6)

set -xeuo pipefail

project_name='DAPO'
exp_name='DAPO-7b-elastic'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Ray
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
#MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
MODEL_PATH="/home/hadoop-djst-algoplat/models/Qwen/Qwen2.5-0.5B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# ── Resource topology ──────────────────────────────────────────
# Fixed Rollout : 2 GPUs, TP=2 → 1 rollout replica
# Elastic       : 6 GPUs, TP=2 → elastic_dp=3 (6/2)
# Total         : 8 GPUs (1 node)
offload=False
gen_tp=1       # rollout tensor parallel size (fixed rollout: 2 GPUs)
train_tp=1     # train tensor parallel size   (elastic:       6 GPUs / TP=2 → dp=3)
train_pp=1
train_cp=1

# Fixed rollout resource size
rollout_gpus=2
elastic_gpus=6

# ── Batch / mini-batch sizes ───────────────────────────────────
train_prompt_bsz=128
n_resp_per_prompt=16
train_prompt_mini_bsz=48

elastic_params=(
  # ---- Safety constraints ----
  # At least 1 elastic resource stays in train so training never stops
  elastic_scheduling.min_train_resources=1
  elastic_scheduling.min_rollout_resources=0
  # ---- Scheduling thresholds ----
  elastic_scheduling.rollout_queue_high_watermark=0.8
  elastic_scheduling.rollout_queue_low_watermark=0.3
  elastic_scheduling.cooldown_seconds=30.0
  elastic_scheduling.confidence_threshold=0.6
  # ---- [DEBUG] Force Train→Rollout switch every step (system sanity check) ----
  +elastic_scheduling.debug_force_switch_every_step=True
)

fully_async=(
  data.train_batch_size=0
  data.gen_batch_size=1
  trainer.test_freq=10
  actor_rollout_ref.hybrid_engine=False
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.actor.optim.lr_decay_steps=51200
  rollout.total_rollout_steps=$((512 * 100))
  # ---- Rollout cluster: fixed 2-GPU replica ----
  rollout.nnodes=1
  rollout.n_gpus_per_node=${rollout_gpus}
  trainer.nnodes=1
  trainer.n_gpus_per_node=${elastic_gpus}
  # ---- Async training knobs ----
  async_training.staleness_threshold=0.5
  async_training.trigger_parameter_sync_step=4
  async_training.require_batches=1
  async_training.partial_rollout=True
)

python -m verl.experimental.elastic_scheduling.main \
    --config-path=config \
    --config-name='elastic_ppo_megatron_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
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
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=44398 \
    actor_rollout_ref.model.trust_remote_code=True \
    data.trust_remote_code=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.total_epochs=10 \
    "${elastic_params[@]}" \
    "${fully_async[@]}"