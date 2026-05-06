#!/usr/bin/env bash
# FP8 End-to-End Training Script (NVIDIA GPU)
# Description: Train with FP8 precision and export FP8 weights directly to SGLang Rollout
# Dependencies: Environment variables defined in exp/fp8/runtime_env.yaml
# Usage: ./nv_fp8_e2e.sh [additional_args...]

set -xeuo pipefail

# ============================ User Configuration ============================

project_name='Qwen3-8B-Base-fp8'
exp_name="$(date +%Y%m%d%H)_exp"
BASE_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/wangshulin02
CKPTS_DIR=$BASE_PATH/checkpoints/${project_name}/${exp_name}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/huggingface.co/Qwen/Qwen3-8B-Base
TRAIN_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet

# ============================ Training Parameters ============================

# ---- Data Parameters ----
DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=2048
    data.val_max_samples=-1
    data.max_response_length=$((1024 * 20))
    # +data.gen_batch_size=96
    data.train_batch_size=32
)

# ---- Model Parameters ----
MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=True
)

# ---- Model Override Parameters ----
OVERRIDE_ARGS=(
    +actor_rollout_ref.model.override_config.attention_dropout=0.
    +actor_rollout_ref.model.override_config.embd_pdrop=0.
    +actor_rollout_ref.model.override_config.resid_pdrop=0.
)

# ---- Optimizer Parameters ----
OPTIM_ARGS=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.min_lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_warmup_init=1e-7
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
)

# ---- PPO Algorithm Parameters ----
PPO_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.rollout.n=16
)

# ---- Dynamic Batch Processing Parameters ----
DYNAMIC_BSZ_ARGS=(
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=22528
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=22528
)

# ---- Group Filtering Parameters ----
FILTER_ARGS=(
    +algorithm.filter_groups.enable=True
    +algorithm.filter_groups.max_num_gen_batches=10
    +algorithm.filter_groups.metric=acc
)

# ---- Rollout Parameters ----
ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

# ---- Reward Model Parameters ----
REWARD_ARGS=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=20480
)

# ---- Trainer Parameters ----
TRAINER_ARGS=(
    trainer.use_legacy_worker_impl=disable
    trainer.logger=['console','tensorboard']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.val_before_train=True
    trainer.test_freq=10
    trainer.save_freq=-1
    trainer.max_actor_ckpt_to_keep=1
    trainer.total_epochs=10
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
)

# ---- Rollout Correction Parameters ----
ROLLOUT_CORRECTION_ARGS=(
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=2.0
    algorithm.rollout_correction.rollout_is_batch_normalize=False
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.rollout_rs_threshold=null
)

# ============================ FP8 Core Configuration ============================
# Key configurations:
# 1. export_weight_dtype="fp8" - Export FP8 weights directly after training
# 2. skip_mid_quantization=True - Skip redundant quantization in Rollout
# 3. Support blockwise FP8 precision training

FP8_ARGS=(
    # Strategy
    actor_rollout_ref.actor.strategy="megatron"

    # FP8 Transformer Configuration (blockwise precision)
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3"
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param=True

    # FP8 Optimizer Configuration
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"

    # FP8 Parameter Gathering
    +actor_rollout_ref.actor.megatron.override_ddp_config.fp8_param_gather=True

    # Additional Transformer Settings
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32

    # MBridge Configuration (required for FP8 weight export)
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

    # ===== FP8 Weight Export Core Configuration =====
    +actor_rollout_ref.actor.megatron.export_weight_dtype="fp8"
    +actor_rollout_ref.rollout.skip_mid_quantization=True
    actor_rollout_ref.rollout.quantization="fp8"

    # Megatron Parallel Configuration
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    # actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    # actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=True

    # Memory Optimization
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
)

# ============================ Launch Training ============================

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer' \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${OPTIM_ARGS[@]}" \
    "${PPO_ARGS[@]}" \
    "${DYNAMIC_BSZ_ARGS[@]}" \
    "${FILTER_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "${ROLLOUT_CORRECTION_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    "$@"
