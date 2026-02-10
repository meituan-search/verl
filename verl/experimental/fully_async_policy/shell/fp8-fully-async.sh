#!/usr/bin/env bash
set -xeuo pipefail

ulimit -n 65535

ID=${1:-"0209_qwen3_8b_sglang_8gpu_megatron_fp8_fully-async_tis_on"}
CAP=${2:-"5"}

project_name='DAPO-fp8-training-rollout'
exp_name=$ID

export RAY_tmp_dir="/workdir/ray_temp"
mkdir -p "$RAY_tmp_dir"

echo "🔍 Checking SGLang Location..."
python3 -c "import sglang; print(f'📍 SGLang Path: {sglang.__file__}'); print(f'🔢 Version: {sglang.__version__}')"
echo "-----------------------------------------------------"


# export WANDB_MODE=offline

WORKING_DIR=${WORKING_DIR:-"${PWD}"}

echo "WORKING_DIR: ${WORKING_DIR}"

# export WANDB_API_KEY=c6a505ac97983529cf303349f68c7658350a2bb8
CONFIG_PATH="$WORKING_DIR/recipe/dapo/config"

# FP8 environment variables
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export SGLANG_PATCH=1
export SGLANG_PORT=30300
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${project_name}/${exp_name}"}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/huggingface.co/Qwen/Qwen3-8B-Base
TRAIN_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet

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
    trainer.logger='["console","tensorboard"]' \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.quantization=fp8 \
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
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
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
    ++ray_init._temp_dir="$RAY_tmp_dir" \
    $@



    # data.filter_overlong_prompts=True \
    # data.return_raw_chat=${return_raw_chat} \
    # actor_rollout_ref.rollout.max_model_len=32768 \
    # actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    # actor_rollout_ref.actor.kl_loss_type=False \
    # actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    # actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    # trainer.critic_warmup=0 \
    # actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    # actor_rollout_ref.rollout.load_format='flash_rl' \
    # actor_rollout_ref.rollout.quantization=fp8 \