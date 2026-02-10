#!/usr/bin/env bash

set -xeuo pipefail


ulimit -n 65535



ID=${1:-"0209_qwen3_8b_sglang_fp8_8gpu_megatron_tis_on"}

CAP=${2:-"5"}

export RAY_tmp_dir="/workdir/ray_temp"
mkdir -p "$RAY_tmp_dir"

echo "🔍 Checking SGLang Location..."
python3 -c "import sglang; print(f'📍 SGLang Path: {sglang.__file__}'); print(f'🔢 Version: {sglang.__version__}')"
echo "-----------------------------------------------------"

project_name='DAPO-fp8-training-rollout'

exp_name=$ID

export WANDB_MODE=offline

WORKING_DIR=${WORKING_DIR:-"${PWD}"}

echo "WORKING_DIR: ${WORKING_DIR}"

export WANDB_API_KEY=c6a505ac97983529cf303349f68c7658350a2bb8



CONFIG_PATH="$WORKING_DIR/recipe/dapo/config"



# FP8 environment variables

export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1


PYTHONUNBUFFERED=1 python3 -X faulthandler -m recipe.dapo.main_dapo \
    --config-path="$CONFIG_PATH" \
    --config-name='dapo_megatron_trainer' \
    data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet \
    data.val_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.val_max_samples=-1 \
    data.max_response_length=$((1024 * 20)) \
    data.gen_batch_size=96 \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.model.path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/huggingface.co/Qwen/Qwen3-8B-Base \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.min_lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_warmup_init=1e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=400 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    actor_rollout_ref.rollout.quantization=fp8 \
    \
    actor_rollout_ref.actor.strategy="megatron" \
    \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    ++ray_init._temp_dir="$RAY_tmp_dir" \
    $@