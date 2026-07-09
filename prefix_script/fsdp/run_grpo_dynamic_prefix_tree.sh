#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# E2E correctness test for dynamic prefix-tree + Magi attention on FSDP.
# Stack: Qwen2.5-0.5B-Instruct + GSM8K + GRPO + FSDP + Magi FFA + vLLM.
#
# Strategy: two short training runs (~10 steps each), identical seed and config,
# one with use_prefix_tree_dynamic=true and one without. Compare reward trajectories
# — dynamic-trie + Magi should track the dense baseline within tolerance.
#
# Hardware:
#   - CP_SIZE=1 → 1× H100, ~10-15 min (single-rank Magi, no dispatch)
#   - CP_SIZE=2 → 2× H100, ~10-15 min (real Magi CP dispatch)
#
# Usage:
#   bash prefix_script/fsdp/run_grpo_dynamic_prefix_tree.sh                       # cp=1
#   CP_SIZE=2 N_GPUS=2 bash prefix_script/fsdp/run_grpo_dynamic_prefix_tree.sh    # cp=2
#   TOTAL_STEPS=20 ROLLOUT_N=4 bash prefix_script/fsdp/run_grpo_dynamic_prefix_tree.sh
set -ex

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/dynamic_prefix_tree_e2e.XXXXXX)}
echo "[E2E] Logs → $LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10}
ROLLOUT_N=${ROLLOUT_N:-8}
SEED=${SEED:-42}
CP_SIZE=${CP_SIZE:-1}            # Magi context parallel size (1, 2, 4, ...)
N_GPUS=${N_GPUS:-${CP_SIZE}}     # GPUs per node — must be >= CP_SIZE
SKIP_BASELINE=${SKIP_BASELINE:-0}  # 1 = skip dense baseline run (debug iteration speedup)
ENABLE_GC=${ENABLE_GC:-1}          # 1 = enable gradient checkpointing (default), 0 = disable for ppo_kl debug

# Common Hydra args shared by both runs.
# Notes:
#   - use_dynamic_bsz=False keeps per-step token budget stable, so the two runs
#     see the same micro-batch shapes (any reward divergence is attributable to
#     the prefix-tree forward, not batching).
#   - ulysses_sequence_parallel_size=1 because dynamic-trie + Magi+CP is mutually exclusive
#     with Ulysses SP (see FSDPEngine._init_device_mesh assertion).
#   - rollout.n=$ROLLOUT_N gives each prompt N rollouts that share a prefix,
#     which is exactly the shape dynamic prefix tree accelerates.
COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files=$DATA_DIR/train.parquet
    data.val_files=$DATA_DIR/test.parquet
    data.train_batch_size=8
    data.max_prompt_length=512
    data.max_response_length=256
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    # dynamic-trie + Magi REQUIRES FSDP2 — FSDP1 produces EVAL/TRAIN forward divergence,
    # see verl/workers/engine/fsdp/transformer_impl.py:_build_module assert.
    # NOTE: must set TOP-LEVEL actor.strategy (not engine.strategy). ActorConfig
    # __post_init__ syncs top-level → engine.strategy, so any "+engine.strategy"
    # override gets clobbered. ref.strategy inherits from actor via OmegaConf.
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.project_name=dynamic_prefix_tree_e2e
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=$TOTAL_STEPS
    +trainer.seed=$SEED
)

# bf16 override knob (debug). Setting these CAN trigger a hang at ref's first
# forward, suspected NCCL/Magi cp_group interaction. Leave disabled by default
# until root cause is understood.
if [ "${FORCE_BF16:-0}" = "1" ]; then
    COMMON_ARGS+=(
        +actor_rollout_ref.actor.engine.model_dtype=bfloat16
        +actor_rollout_ref.ref.engine.model_dtype=bfloat16
    )
    echo "[E2E] WARNING: FORCE_BF16=1, model_dtype=bfloat16. May hang."
fi

# Gradient checkpointing toggle (debug knob for ppo_kl mismatch investigation)
if [ "$ENABLE_GC" = "0" ]; then
    COMMON_ARGS+=(actor_rollout_ref.model.enable_gradient_checkpointing=False)
    echo "[E2E] gradient_checkpointing DISABLED (ENABLE_GC=0)"
fi

echo "=========================================="
echo "[E2E] RUN 1: dynamic prefix tree + Magi (cp=${CP_SIZE})"
echo "=========================================="
# ref inherits use_prefix_tree_dynamic/prefix_tree_attention/context_parallel_size
# from actor automatically (see verl/trainer/config/ref/ref.yaml).
python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_prefix_tree_dynamic=True \
    actor_rollout_ref.actor.prefix_tree_attention=magi \
    actor_rollout_ref.actor.context_parallel_size=$CP_SIZE \
    trainer.experiment_name=qwen2_5_05b_grpo_dynamic_magi_cp${CP_SIZE} \
    2>&1 | tee "$LOG_DIR/dyn.log"

if [ "$SKIP_BASELINE" = "1" ]; then
    echo "=========================================="
    echo "[E2E] SKIP_BASELINE=1 → skipping dense baseline run"
    echo "[E2E] dynamic-trie + Magi log: $LOG_DIR/dyn.log"
    echo "=========================================="
    exit 0
fi

echo "=========================================="
echo "[E2E] RUN 2: Dense baseline (no prefix tree)"
echo "=========================================="
python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_prefix_tree_dynamic=False \
    trainer.experiment_name=qwen2_5_05b_grpo_dense \
    2>&1 | tee "$LOG_DIR/dense.log"

echo "[E2E] Both runs completed — log dir: $LOG_DIR"
echo "[E2E]   dynamic-trie + Magi: $LOG_DIR/dyn.log"
echo "[E2E]   dense baseline:      $LOG_DIR/dense.log"
