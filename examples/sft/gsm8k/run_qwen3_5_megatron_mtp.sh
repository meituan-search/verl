#!/usr/bin/env bash
# Qwen3.5 SFT with Megatron backend + MTP (Multi-Token Prediction)
#
# Requirements:
#     pip install --upgrade transformers==5.3.0   (only 5.3.0 is supported)
#     mbridge: make sure https://github.com/ISEEKYAN/mbridge/pull/98 this pr has merged
#
# MTP (Multi-Token Prediction) notes:
#   - model.mtp.enable=True               enables MTP module
#   - model.mtp.enable_train=True         enables MTP training loss
#   - model.mtp.detach_encoder=True       detaches encoder gradients for MTP
#   - model.mtp.mtp_loss_scaling_factor   weight of MTP auxiliary loss (e.g. 0.1)
#
# Example parallelism configs (in h20-144g):
#   Qwen3.5-35B-A3B  (8 GPUs / 1 node):   TP=4 PP=2 EP=4
#   Qwen3.5-397B-A17B (32 GPUs / 4 nodes): TP=8 PP=4 EP=8

set -xeuo pipefail

# ============================================================
# Distributed
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# ============================================================
# Data
# ============================================================
DATASET_DIR=${DATASET_DIR:-~/dataset}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/val.parquet}

# ============================================================
# Model
# ============================================================
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}

# ============================================================
# Parallelism (defaults for Qwen3.5-35B-A3B on 1 node / 8 GPUs)
# ============================================================
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-4}
ETP_SIZE=${ETP_SIZE:-1}

# ============================================================
# Training hyper-parameters
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-40960}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-7}
DTYPE=${DTYPE:-bfloat16}

BACKEND=megatron
RESUME_MODE=${RESUME_MODE:-disable}

# ============================================================
# MTP hyper-parameters
# ============================================================
MTP_ENABLE=${MTP_ENABLE:-True}
MTP_ENABLE_TRAIN=${MTP_ENABLE_TRAIN:-True}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-True}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# ============================================================
# Paths
# ============================================================
project_name=verl_sft_qwen3_5_mtp
exp_name=qwen3_5-mtp-${BACKEND}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}
ckpts_home=${ckpts_home:-~/verl/checkpoints/${project_name}/${exp_name}}
mkdir -p "${ckpts_home}"

# ============================================================
# Engine config
# ============================================================
# Key Qwen3.5 settings:
#   use_dynamic_bsz=True is only support for megatron with https://github.com/NVIDIA/Megatron-LM/pull/2644
#   engine.vanilla_mbridge=True       - use mbridge (not megatron-bridge)
#   +engine.override_transformer_config.mtp_loss_scaling_factor
#                                     - MTP auxiliary loss weight in Megatron
ENGINE_CONFIG="\
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    +optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +optim.override_optimizer_config.optimizer_cpu_offload=True \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=True \
    engine.override_transformer_config.attention_backend=auto \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1 \
    +engine.override_transformer_config.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}"

# ============================================================
# Launch
# ============================================================
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.max_length=${MAX_LENGTH} \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=8 \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=True \
    model.trust_remote_code=True \
    model.mtp.enable=${MTP_ENABLE} \
    model.mtp.enable_train=${MTP_ENABLE_TRAIN} \
    model.mtp.detach_encoder=${MTP_DETACH_ENCODER} \
    model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=10 \
    trainer.max_ckpt_to_keep=1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    checkpoint.save_contents=[model,extra]
