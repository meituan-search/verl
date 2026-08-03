#!/usr/bin/env bash
# Qwen3.5-35B-A3B MoE GRPO RL with Megatron (single node, 8 GPUs, gsm8k/dapo dataset, colocate_async)
#
# notes on vllm:
#     by 20260225, the latest vllm nightly does not support qwen3.5 rollout, to use this script, you need to
#         1. wait until vllm supports qwen3.5 officially, and build a verl docker with that version of vllm
#         2. self build a verl docker image with vllm from source code with qwen3.5 support (main branch 20260225 is OK)
#     I succeeded in running this script with the main branch of vllm on 20260225, yet there are still some minor issues
#     the vllm qwen3.5 during initialization, need to be fixed. Also, the cuda_graph is somehow not working, need to be
#     fixed, either by verl team with supoorts to vllm0.16, or by vllm team.
# Requirements:
#   - 8 GPUs (80GB each, e.g. 1x8 H100/H200)
#   - Additional packages on top of the base image:
#       pip install --upgrade transformers
#       pip install flash-linear-attention
#       pip install -U git+https://github.com/ISEEKYAN/mbridge.git
#   - Megatron-LM==0.16.0
#
# Requirements on Ascend:
#   - 8 NPUs (2*64GB each, e.g. 1x8 A3)
#   - Additional packages on base image(v0.8.0-cann9.0.0-torch2.9.0post2-a3-ubuntu22.04-py3.11-vllm):
#       pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon
#   - Megatron-LM==0.16.0
#   - MindSpeed==0.16.0
#   - Megatron-Bridge==de93536e
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention. Earlier Megatron-LM
#   did NOT support packed sequences (THD format) for GDN, forcing bshd mode
#   (use_remove_padding=False, use_dynamic_bsz=False, ppo_micro_batch_size_per_gpu=1).
#   With newer Megatron-LM that adds GDN THD support, this script now uses:
#     - model.use_remove_padding=True            (THD packing)
#     - actor.megatron.use_remove_padding=True
#     - actor.use_dynamic_bsz=True
#   If your Megatron-LM does not support GDN THD yet, flip the three flags back
#   to False/False/False and add `ppo_micro_batch_size_per_gpu=1`.
#
# Tested parallelism config (8 GPUs / 1 node):
#   TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8
#
# colocate_async: trainer and rollout share the same GPUs; rollout is paused
# (sleep_replicas) during training and resumed after weight sync. NUM_WARMUP_BATCHES
# pre-fills the rollout/training pipeline so step #1 can sample while later
# batches are still generating.
#

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

set -xeuo pipefail

########################### Quick Config ###########################

# ---- user-adjustable ----
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
case "${DEVICE}" in
    gpu)
        TP=${TP:-2}
        PP=${PP:-1}
        CP=${CP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        ;;
    npu)
        TP=${TP:-2}
        PP=${PP:-2}
        CP=${CP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}   # pre-fill pipeline for rollout/training overlap

# INFER_BACKEND controls rollout backend: vllm | sglang.
# vllm needs source build with qwen3.5 support (main branch 20260225+).
INFER_BACKEND=${INFER_BACKEND:-sglang}
rollout_name=${INFER_BACKEND}
adv_estimator=grpo

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen3.5-35B-A3B"}

# Per-dataset defaults. Any knob can still be overridden via its env var
# (TRAIN_FILES, TRAIN_BATCH_SIZE, ...). DATASET just picks the base.
DATASET=${DATASET:-gsm8k}
GSM8K_DIR=${GSM8K_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/gsm8k}
DAPO_DIR=${DAPO_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo}
case "$DATASET" in
    gsm8k)
        train_path=${TRAIN_FILES:-"$GSM8K_DIR/train.parquet"}
        test_path=${VAL_FILES:-"$GSM8K_DIR/test.parquet"}
        train_batch_size=${TRAIN_BATCH_SIZE:-32}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
        max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
        max_response_length=${MAX_RESPONSE_LENGTH:-2048}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
        rollout_n=${ROLLOUT_N:-5}
        ;;
    dapo)
        train_path=${TRAIN_FILES:-"$DAPO_DIR/dapo-math-17k.parquet"}
        test_path=${VAL_FILES:-"$DAPO_DIR/aime-2024.parquet"}
        train_batch_size=${TRAIN_BATCH_SIZE:-16}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
        max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
        max_response_length=${MAX_RESPONSE_LENGTH:-32768}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-34816}
        rollout_n=${ROLLOUT_N:-8}
        ;;
    *)
        echo "Unknown DATASET=$DATASET (expected: gsm8k | dapo)" >&2
        exit 1
        ;;
esac

project_name=${PROJECT_NAME:-verl_grpo_${DATASET}_math}
exp_name=${EXPERIMENT_NAME:-qwen3_5_35b_megatron_${DATASET}_colocate_async}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.calculate_log_probs=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.val_before_train=False
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    transfer_queue.enable=True
)

EXTRA=(
    model_engine=megatron
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export CPU_AFFINITY_CONF=1
        ACTOR+=(
            actor_rollout_ref.actor.megatron.vanilla_mbridge=False
            actor_rollout_ref.actor.checkpoint.strict=False
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
        )
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
