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
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        elastic_scheduling.rollout_gpus=${ROLLOUT_GPUS} \
        elastic_scheduling.train_gpus=${TRAIN_GPUS} \
        elastic_scheduling.elastic_gpus=${ELASTIC_GPUS} \
        trainer.default_local_dir="${CKPTS_DIR}"

