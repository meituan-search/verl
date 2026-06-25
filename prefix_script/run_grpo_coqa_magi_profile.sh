#!/usr/bin/env bash
# GRPO CoQA magi — actor-only profiling run (MAGI_TIMING + optional nsys)
# Steps: 30 only. Use PROFILE_MODE=timing (default) or PROFILE_MODE=nsys
#
# PROFILE_MODE=timing  → MAGI_TIMING=1 prints dispatch/calc_attn/undispatch breakdown
# PROFILE_MODE=nsys    → nsys wraps actor update_actor only at steps 20,25
#
# Usage:
#   bash run_grpo_coqa_magi_profile.sh                      # timing mode
#   PROFILE_MODE=nsys bash run_grpo_coqa_magi_profile.sh   # nsys mode

set -xeuo pipefail
export PATH=/usr/local/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export HYDRA_FULL_ERROR=1

# Timing: print dispatch/calc_attn/undispatch for layer 0 on every micro-batch
export MAGI_TIMING=1

# Diag: additional MAGI diagnostic output
export MAGI_DIAG=1

MODEL_PATH="${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen/Qwen3-4B-Base}"
# Dataset: coqa_grpo.parquet prepared from CoQA via prefix_script/data/coqa/prepare_coqa_grpo.py
TRAIN_FILES="${TRAIN_FILES:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-djst-algoplat/yumingxuan/prefix_script/data/coqa/coqa_grpo.parquet}"
REWARD_FN="${REWARD_FN:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-djst-algoplat/yumingxuan/prefix_script/data/coqa/coqa_reward.py}"
PROFILE_MODE="${PROFILE_MODE:-timing}"

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="/tmp/verl_submit/profiles/magi/${TS}"
mkdir -p "$OUTDIR"

echo "================================================================"
echo "MAGI profile run  mode=$PROFILE_MODE  TS=$TS"
echo "  output: $OUTDIR"
echo "================================================================"

# Base args (same as magi run)
BASE_ARGS=(
    model_engine=megatron
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$TRAIN_FILES"
    data.val_max_samples=32
    data.train_batch_size=128
    data.max_prompt_length=1024
    data.max_response_length=128
    data.filter_overlong_prompts=True
    data.truncation=left
    data.prompt_key=prompt
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_prefix_tree=True
    actor_rollout_ref.model.prefix_tree_attention=magi
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.max_model_len=2048
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    reward.custom_reward_function.path="$REWARD_FN"
    reward.num_workers=2
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.total_training_steps=30
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=grpo_coqa_4b_profile
    trainer.experiment_name="magi_profile_${TS}"
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.balance_batch=True
)

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HOME/profiles/magi_${TS}/tb}"
mkdir -p "$TENSORBOARD_DIR"

if [ "$PROFILE_MODE" = "nsys" ]; then
    # nsys: profile actor update_actor only at steps 20,25
    NSYS_ARGS=(
        +global_profiler.tool=nsys
        "+global_profiler.steps=[20,25]"
        ++global_profiler.save_path="$OUTDIR/nsys"
        ++global_profiler.global_tool_config.nsys.discrete=True
        ++actor_rollout_ref.actor.profiler.enable=True
        ++actor_rollout_ref.actor.profiler.all_ranks=True
    )
    python3 -m verl.trainer.main_ppo \
        "${BASE_ARGS[@]}" "${NSYS_ARGS[@]}" 2>&1 | tee "$OUTDIR/run.log"
else
    # timing mode: MAGI_TIMING=1 already set above
    python3 -m verl.trainer.main_ppo \
        "${BASE_ARGS[@]}" 2>&1 | tee "$OUTDIR/run.log"
fi

echo "================================================================"
echo "Profile done → $OUTDIR"
if [ "$PROFILE_MODE" = "timing" ]; then
    echo "Extract timing: grep 'MAGI-TIMING' $OUTDIR/run.log"
fi
echo "================================================================"
