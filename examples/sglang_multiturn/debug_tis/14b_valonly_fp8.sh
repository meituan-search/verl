# run on 4xGPU  
# make sure your current working directory is the root of the project
# Using QuantizedRL with DYNAMIC INT8 quantization (BF16→INT8 like FlashRL)
# Automatically quantizes BF16 weights to INT8 during weight reloads
# This provides memory savings while preserving model quality
# Model: Qwen2.5-7B (same as FlashRL's sglang_patch testing)
#
# Requirements:
#   1. FlashRL installed: pip install flash-rl
#   2. Profile file at /root/profile.7b.pt (auto-detected)
#   3. Set load_format=quantized_rl (see line 66 below)
#
# How it works:
#   - Initial load: BF16/FP16 model (no quantization)
#   - Weight reloads: Automatic INT8 quantization using FlashRL's profile
#
# PyTorch Profiling:
#   - Enable with SGLANG_ENABLE_PROFILER=1
#   - Profile first N steps with SGLANG_PROFILE_STEPS=5
#   - Output directory: SGLANG_PROFILER_DIR=/root/sglang_profiles
#   - View traces in Chrome: chrome://tracing
#   - Or use NSight Systems: nsys-ui

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="blockwise_fp8_$(now)"
export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY=83886215c84d216a3b0539e4c08cf6320e2f08fb
export TENSORBOARD_DIR=/workdir/quant-rollout/work_logs/tsb_log
export http_proxy=http://10.229.18.27:8412 && export https_proxy=http://10.229.18.27:8412

# /workdir/quant-rollout/Qwen2.5-14B-Instruct



python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/workdir/quant-rollout/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_only=True \
    trainer.log_val_generations=10 \
    trainer.logger='["console"]' \
    trainer.project_name='flash_rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=200 \
    trainer.test_freq=20 \
    data.train_files=/workdir/data/gsm8k/train.parquet \
    data.val_files=/workdir/data/gsm8k/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    trainer.total_epochs=15 $@



    # actor_rollout_ref.rollout.load_format=flash_rl \