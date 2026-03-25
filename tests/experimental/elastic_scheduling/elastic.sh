#!/usr/bin/env bash

#===============================================================================
# Elastic Scheduling PPO Training Test Script
#
# Single Node (8 GPUs) Test Configuration:
#   - Rollout: 4 GPUs (GPU 0-3)
#   - Train: 2 GPUs (GPU 4-5)
#   - Elastic: 2 GPUs (GPU 6-7), initial mode: rollout
#
# Usage:
#   RAY_ADDRESS='http://10.148.11.18:8420' \
#   RAY_DATA_HOME=${HOME}/verl \
#   MODEL_PATH=${RAY_DATA_HOME}/models/MiMo-7B-RL \
#   TRAIN_FILE=${RAY_DATA_HOME}/data/dapo-math-17k.parquet \
#   bash tests/experimental/elastic_scheduling/elastic.sh
#===============================================================================

set -xeuo pipefail

# Project name
project_name='ElasticPPO'
exp_name='elastic-ppo-single-node-8g'

# Ray configuration
#RAY_ADDRESS=${RAY_ADDRESS:-"http://10.148.11.18:8420"}
#RUNTIME_ENV=${RUNTIME_ENV:-"${PWD}/examples/mtp_trainer/runtime_env.yaml"}
#WORKING_DIR=${WORKING_DIR:-"${PWD}"}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/MiMo-7B-RL"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Resource allocation (8 GPUs total)
#   - Rollout: 4 GPUs (GPU 0-3)
#   - Train: 2 GPUs (GPU 4-5)
#   - Elastic: 2 GPUs (GPU 6-7), initial mode: rollout
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TRAIN_GPUS=${TRAIN_GPUS:-2}
ELASTIC_GPUS=${ELASTIC_GPUS:-2}

# Training parameters
max_prompt_length=${max_prompt_length:-$((1024 * 2))}
max_response_length=${max_response_length:-$((1024 * 8))}
n_resp_per_prompt=${n_resp_per_prompt:-16}

# Tensor parallel settings
gen_tp=${gen_tp:-1}
train_tp=${train_tp:-1}
train_pp=${train_pp:-1}
train_cp=${train_cp:-1}

# Fully async configuration
fully_async=(
  data.train_batch_size=0
  data.gen_batch_size=1
  trainer.test_freq=10
  trainer.nnodes=1
  trainer.n_gpus_per_node=${TRAIN_GPUS}
  rollout.nnodes=1
  rollout.n_gpus_per_node=${ROLLOUT_GPUS}
  async_training.staleness_threshold=0.5
  async_training.trigger_parameter_sync_step=4
  async_training.require_batches=1
  async_training.partial_rollout=true
  actor_rollout_ref.hybrid_engine=false
  actor_rollout_ref.rollout.calculate_log_probs=true
)

# Elastic scheduling configuration
elastic_scheduling=(
  elastic_scheduling.rollout_gpus=${ROLLOUT_GPUS}
  elastic_scheduling.train_gpus=${TRAIN_GPUS}
  elastic_scheduling.elastic_gpus=${ELASTIC_GPUS}
  elastic_scheduling.dp_size_per_resource=1
  elastic_scheduling.elastic_initial_mode=rollout
  elastic_scheduling.rollout_queue_high_watermark=0.8
  elastic_scheduling.rollout_queue_low_watermark=0.3
  elastic_scheduling.cooldown_seconds=10.0
  elastic_scheduling.sync_trigger_interval=4
  elastic_scheduling.enable_incremental_sync=true
  elastic_scheduling.check_interval=1.0
  elastic_scheduling.congestion_window_size=20
)

# Logging
echo "============================================================"
echo "Elastic Scheduling PPO Training Test"
echo "============================================================"
echo "Model Path: ${MODEL_PATH}"
echo "Train File: ${TRAIN_FILE}"
echo "============================================================"
echo "Resource Allocation (8 GPUs total):"
echo "  - Rollout GPUs: ${ROLLOUT_GPUS}"
echo "  - Train GPUs: ${TRAIN_GPUS}"
echo "  - Elastic GPUs: ${ELASTIC_GPUS} (initial mode: rollout)"
echo "============================================================"

# Submit job to Ray
python -m verl.experimental.elastic_scheduling.main \
        --config-path=config \
        --config-name='elastic_ppo_trainer_single_node.yaml' \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.trust_remote_code=True \
        data.trust_remote_code=True \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.nnodes=1 \
        actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
        trainer.logger=['console'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.total_epochs=10 \
        trainer.val_before_train=True \
        trainer.resume_mode=auto \
        "${fully_async[@]}" \
        "${elastic_scheduling[@]}"

