#!/bin/bash
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Elastic Scheduling PPO Training Launch Script
# Usage: bash run_elastic_ppo.sh [MODEL_PATH] [NUM_GPUS]

set -e

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_GPUS=${2:-24}

# Calculate resource allocation
# Total GPUs: 24
# Rollout: 8, Train: 8, Elastic: 8
ROLLOUT_GPUS=8
TRAIN_GPUS=8
ELASTIC_GPUS=8
DP_SIZE=8

echo "=========================================="
echo "Elastic Scheduling PPO Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Total GPUs: $NUM_GPUS"
echo "Rollout GPUs: $ROLLOUT_GPUS"
echo "Train GPUs: $TRAIN_GPUS"
echo "Elastic GPUs: $ELASTIC_GPUS"
echo "DP Size: $DP_SIZE"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23

# Ray setup
ray stop 2>/dev/null || true
ray start --head --num-gpus=$NUM_GPUS --port=6379 --include-dashboard=false

# Launch training
python -m verl.experimental.elastic_scheduling.main \
    actor_rollout_ref.model.path=$MODEL_PATH \
    elastic_scheduling.rollout_gpus=$ROLLOUT_GPUS \
    elastic_scheduling.train_gpus=$TRAIN_GPUS \
    elastic_scheduling.elastic_gpus=$ELASTIC_GPUS \
    elastic_scheduling.dp_size_per_resource=$DP_SIZE \
    elastic_scheduling.rollout_queue_high_watermark=0.8 \
    elastic_scheduling.rollout_queue_low_watermark=0.3 \
    elastic_scheduling.cooldown_seconds=10.0 \
    elastic_scheduling.sync_trigger_interval=4 \
    elastic_scheduling.check_interval=1.0 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=$NUM_GPUS \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=$NUM_GPUS \
    wandb.enable=false \
    training.total_epochs=100

