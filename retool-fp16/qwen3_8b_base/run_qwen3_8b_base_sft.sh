#!/bin/bash
set -x

nnodes=1
nproc_per_node=8

project_name=retool
experiment_name=multiturn-sft-qwen3-8b-base

DATA_ROOT=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/wangshulin02/data

TRAIN_DATA=$DATA_ROOT/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=$DATA_ROOT/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/huggingface.co/Qwen/Qwen3-8B-Base
SAVE_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/checkpoint/$experiment_name
# SAVE_PATH=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/wangshulin02/model/checkpoint/$experiment_name

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","tensorboard"]' \
    trainer.total_epochs=10 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true