#!/usr/bin/env bash
set -xeuo pipefail

ulimit -n 65535
# export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

ID=${1:-"0209_QWen3_8B_SGLang_FP8_Megatron_FP8-fully-async_1-2-16_8gpu_tis_on"}
CAP=${2:-"5"}

project_name='FP8-fully-async-e2e'
exp_name=$ID

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"
export WANDB_API_KEY=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
export http_proxy=http://10.229.18.27:8412 && export https_proxy=http://10.229.18.27:8412
# Show all worker logs (including SGLang [SGLang] entries); set to 0 to see SGLang server logs in same terminal
# export RAY_DEDUP_LOGS=0

CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${project_name}/${exp_name}"}
export SGLANG_ROOT=/root/sglang
# Prepend local Megatron-LM so megatron.core.utils (e.g. unwrap_model) is used instead of site-packages
MEGATRON_LM_ROOT="${MEGATRON_LM_ROOT:-/root/Megatron-LM}"
if [ -d /root/sglang/python ]; then
    SGLANG_ROOT="${SGLANG_ROOT:-/root/sglang}"
else
    SGLANG_ROOT="${SGLANG_ROOT:-/workdir/fp8-training-rollout/sglang}"
fi
echo "SGLANG_ROOT=${SGLANG_ROOT} (PYTHONPATH will use this for sglang)"
if [ -d "$MEGATRON_LM_ROOT" ]; then
    export PYTHONPATH="${MEGATRON_LM_ROOT}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"
else
    export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"
fi
CONFIG_PATH="${CONFIG_PATH:-$WORKING_DIR/recipe/dapo/config}"

# FP8 environment variables
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

export SGLANG_PATCH=1
export SGLANG_PORT=30300

# [SGLang] debug log: tail -f ${SGLANG_DEBUG_LOG} to see generate entry/exit (even when SGLang runs in Ray worker)
# export SGLANG_DEBUG_LOG="${SGLANG_DEBUG_LOG:-/workdir/sglang_debug.log}"
# echo "SGLANG_DEBUG_LOG=${SGLANG_DEBUG_LOG} (tail -f this file to see [SGLang] generate logs)"\
# export FLASHRL_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export VERL_POST_WAKEUP_DELAY_SEC=25
export RAY_TEMP_DIR=/workdir/ray_temp
mkdir -p ${RAY_TEMP_DIR}


# Start a new Ray cluster using GPUs 2,3,4,5 on a different port
# This allows running multiple training tasks simultaneously
RAY_PORT=6811
RAY_DASHBOARD_PORT=8897

# Check if Ray server is already running on the specified port
# Use localhost instead of IP address to avoid IP mismatch issues
echo "Checking if Ray server is already running on localhost:${RAY_PORT}..."
if ray status --address="localhost:${RAY_PORT}" >/dev/null 2>&1; then
    echo "Ray server is already running on localhost:${RAY_PORT}, using existing cluster."
    RAY_RUNNING=true
else
    echo "Ray server is not running, starting new Ray cluster..."
    # Start new Ray cluster with 8 GPUs
    ray start --head \
        --port=${RAY_PORT} \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=8 \
        --dashboard-host=0.0.0.0 \
        --temp-dir=${RAY_TEMP_DIR} \
        --disable-usage-stats || true
    echo "Ray cluster started successfully on localhost:${RAY_PORT}"
    # Wait a moment for Ray to fully initialize
    echo "Waiting for Ray cluster to be ready..."
    sleep 5
    # Verify Ray is actually running
    if ray status --address="localhost:${RAY_PORT}" >/dev/null 2>&1; then
        echo "Ray cluster is ready and accessible."
        RAY_RUNNING=true
    else
        echo "Warning: Ray cluster may not be fully ready yet, but continuing..."
        RAY_RUNNING=false
    fi
fi

# Set RAY_ADDRESS for Python code to connect
# Since we're running on the same machine, we can either:
# 1. Not set RAY_ADDRESS and let Ray auto-connect to local cluster (recommended)
# 2. Or use GCS address format for explicit connection
# We'll unset RAY_ADDRESS to let Ray auto-connect, which is more reliable
unset RAY_ADDRESS
echo "RAY_ADDRESS unset - Ray will auto-connect to local cluster at localhost:${RAY_PORT}"

export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH=/workdir/data/huggingface.co/Qwen/Qwen3-8B-Base
TRAIN_PATH=/workdir/data/BytedTsinghua-SIA/DAPO-Math-17k/train.parquet
TEST_PATH=/workdir/data/AIME-2024/train.parquet

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
# if [ "$rollout_mode" = "async" ]; then
#     export VLLM_USE_V1=1
#     return_raw_chat="True"
# fi
return_raw_chat="True"

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=16
total_rollout_steps=$(((32*400))) # gbs * step
test_freq=5
staleness_threshold=0.2
trigger_parameter_sync_step=2
require_batches=1 # rb * trigger * ppombz = gbs
partial_rollout=True
total_epochs=1

PYTHONUNBUFFERED=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="$TRAIN_PATH" \
    data.val_files="$TEST_PATH" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.val_max_samples=-1 \
    data.max_response_length=$((1024 * 20)) \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    +algorithm.filter_groups.metric=acc \
    reward_model.reward_manager=dapo \
    +reward_model.overlong_buffer.enable=True \
    +reward_model.overlong_buffer.len=4096 \
    +reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.min_lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_steps=51200 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_warmup_init=1e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.load_format='flash_rl' \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=44398 \
    actor_rollout_ref.actor.strategy="megatron" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs="${total_epochs}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \
    $@ 2>&1 | tee fp8-fully-async-1-2-16.log



    # data.filter_overlong_prompts=True \
    # data.return_raw_chat=${return_raw_chat} \
    # actor_rollout_ref.rollout.max_model_len=32768 \
    # actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    # actor_rollout_ref.actor.kl_loss_type=False \
    # actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    # actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    # actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    # trainer.critic_warmup=0 \