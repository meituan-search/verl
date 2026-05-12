#!/usr/bin/env bash
set -xeuo pipefail

# End-to-end regression test for the rollout load-balancing feature.
#
# Tests two load-balance strategies back-to-back using the same model / data:
#   1. least_inflight  — original behaviour, no extra deps (baseline)
#   2. sglang_router   — Rust Router with cache_aware policy
#                        (skipped automatically if sglang_router pkg is absent)
#
# Usage:
#   bash test_rollout_lb_e2e.sh                    # fsdp2, sglang rollout
#   ACTOR_STRATEGY=megatron bash test_rollout_lb_e2e.sh
#   LB_STRATEGY=least_inflight bash test_rollout_lb_e2e.sh  # single strategy
#
# Required env vars (override defaults as needed):
#   MODEL_PATH   – path to a small instruction-tuned model checkpoint
#   TRAIN_FILE   – parquet with training prompts
#   TEST_FILE    – parquet with validation prompts
#   NUM_GPUS     – total GPU count on this node (default: 8)

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}   # fsdp2 or megatron
LB_STRATEGY=${LB_STRATEGY:-"least_inflight"} # least_inflight | sglang_router | both

# Extra Hydra overrides forwarded verbatim to python — captured here at script level
# so run_trial()'s internal "$@" (which would be lb_strategy/router_policy args) is not used.
EXTRA_ARGS=("$@")

MODEL_PATH=${MODEL_PATH:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/Qwen/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet"}

# ---------------------------------------------------------------------------
# Rollout settings
# ---------------------------------------------------------------------------
rollout_name="sglang"
rollout_mode="async"

# In hybrid_engine=True mode, verl creates num_replicas = NUM_GPUS / gen_tp
# separate SGLangHttpServer actors, each occupying gen_tp GPUs.
# data_parallel_size is sglang-internal DP and must stay at 1 (default);
# verl handles multi-replica routing at its own layer via LLMServerManager.
GEN_TP=${GEN_TP:-2}
gen_tp=${GEN_TP}
num_replicas=$(( NUM_GPUS / gen_tp ))

if [ $(( gen_tp * num_replicas )) -ne "${NUM_GPUS}" ]; then
    echo "Error: GEN_TP=${gen_tp} does not evenly divide NUM_GPUS=${NUM_GPUS}." >&2
    echo "       Set GEN_TP to a divisor of NUM_GPUS (e.g. GEN_TP=1)." >&2
    exit 1
fi

if [ "${num_replicas}" -lt 2 ]; then
    echo "Warning: num_replicas=${num_replicas} — load balancing has no effect with a single replica." >&2
fi

# ---------------------------------------------------------------------------
# Algorithm / training parameters  (kept minimal for a regression test)
# ---------------------------------------------------------------------------
adv_estimator=grpo
max_prompt_length=2048
max_response_length=4096
train_prompt_bsz=512
n_resp_per_prompt=4
train_prompt_mini_bsz=8
n_gpus_training=${NUM_GPUS}

# ---------------------------------------------------------------------------
# Helper: check sglang_router package availability
# ---------------------------------------------------------------------------
_has_sglang_router() {
    python3 -c "import sglang_router" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Helper: run one training trial
#   $1 = lb_strategy   (least_inflight | sglang_router)
#   $2 = router_policy (cache_aware | round_robin | …)  — ignored for least_inflight
# ---------------------------------------------------------------------------
run_trial() {
    local lb_strategy="$1"
    local router_policy="${2:-cache_aware}"
    local exp_tag="lb-${lb_strategy}-${router_policy}"

    echo ""
    echo "========================================================="
    echo " Running trial: strategy=${lb_strategy}  policy=${router_policy}"
    echo "========================================================="

    # Common Hydra overrides shared by both trials
    local common_params=(
        data.train_files="${TRAIN_FILE}"
        data.val_files="${TEST_FILE}"
        data.prompt_key=prompt
        data.truncation='left'
        data.max_prompt_length=${max_prompt_length}
        data.max_response_length=${max_response_length}
        data.train_batch_size=${train_prompt_bsz}
        data.return_raw_chat=True
        data.filter_overlong_prompts=False
        actor_rollout_ref.model.path="${MODEL_PATH}"
        actor_rollout_ref.hybrid_engine=True
        actor_rollout_ref.rollout.name=${rollout_name}
        actor_rollout_ref.rollout.mode=${rollout_mode}
        actor_rollout_ref.rollout.n=${n_resp_per_prompt}
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
        actor_rollout_ref.rollout.enable_prefix_caching=True
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.disable_log_stats=False
        actor_rollout_ref.rollout.temperature=1.0
        actor_rollout_ref.rollout.top_p=1.0
        actor_rollout_ref.rollout.top_k=-1
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.7
        actor_rollout_ref.rollout.val_kwargs.top_k=-1
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.rollout.checkpoint_engine.backend=naive
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1
        actor_rollout_ref.actor.optim.weight_decay=0.1
        actor_rollout_ref.actor.clip_ratio_low=0.2
        actor_rollout_ref.actor.clip_ratio_high=0.28
        actor_rollout_ref.actor.clip_ratio_c=10.0
        actor_rollout_ref.actor.entropy_coeff=0
        actor_rollout_ref.actor.loss_agg_mode=token-mean
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.kl_loss_coef=0.0
        algorithm.adv_estimator=${adv_estimator}
        algorithm.use_kl_in_reward=False
        algorithm.kl_ctrl.kl_coef=0.0
        reward.reward_manager.name=dapo
        "trainer.logger=['console', 'tensorboard']"
        trainer.project_name='verl-rollout-lb-test'
        trainer.experiment_name="${exp_tag}"
        trainer.val_before_train=True
        trainer.save_freq=-1
        trainer.resume_mode=disable
        trainer.nnodes=1
        trainer.n_gpus_per_node=${n_gpus_training}
        trainer.log_val_generations=4
        trainer.total_epochs=1
        trainer.test_freq=-1
        # Load-balance strategy under test
        actor_rollout_ref.rollout.load_balance.strategy=${lb_strategy}
    )

    # Append router policy only when using sglang_router
    if [ "${lb_strategy}" = "sglang_router" ]; then
        common_params+=(
            actor_rollout_ref.rollout.load_balance.router.policy=${router_policy}
            actor_rollout_ref.rollout.load_balance.router.cache_threshold=0.3
            actor_rollout_ref.rollout.load_balance.router.balance_abs_threshold=64
            actor_rollout_ref.rollout.load_balance.router.balance_rel_threshold=1.5
            actor_rollout_ref.rollout.load_balance.router.health_check_interval_secs=60
        )
    fi

    if [ "${ACTOR_STRATEGY}" = "fsdp2" ]; then
        export VLLM_USE_V1=1
        python -X faulthandler -m verl.trainer.main_ppo \
            "${common_params[@]}" \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
            critic.strategy=fsdp2 \
            actor_rollout_ref.actor.grad_clip=1.0 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
            "${EXTRA_ARGS[@]}"

    elif [ "${ACTOR_STRATEGY}" = "megatron" ]; then
        python -X faulthandler -m verl.trainer.main_ppo \
            --config-name='ppo_megatron_trainer.yaml' \
            "${common_params[@]}" \
            actor_rollout_ref.actor.strategy=megatron \
            critic.strategy=megatron \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.actor.optim.lr_decay_steps=10000000 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.megatron.param_offload=False \
            actor_rollout_ref.actor.megatron.optimizer_offload=False \
            actor_rollout_ref.actor.megatron.grad_offload=False \
            actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
            actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
            actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
            actor_rollout_ref.ref.megatron.param_offload=True \
            "${EXTRA_ARGS[@]}"
    else
        echo "Error: unknown ACTOR_STRATEGY=${ACTOR_STRATEGY}. Use 'fsdp2' or 'megatron'."
        exit 1
    fi

    echo "Trial [${exp_tag}] completed successfully."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "Rollout LB E2E test  |  strategy=${ACTOR_STRATEGY}  |  lb=${LB_STRATEGY}  |  gpus=${NUM_GPUS}"

case "${LB_STRATEGY}" in
    least_inflight)
        run_trial least_inflight
        ;;

    sglang_router)
        if ! _has_sglang_router; then
            echo "SKIP: sglang_router strategy requested but 'sglang_router' package not installed."
            echo "      Install with: pip install sglang[router]"
            exit 0
        fi
        run_trial sglang_router cache_aware
        ;;

    both)
        # Trial 1 — baseline
        run_trial least_inflight

        # Trial 2 — Rust Router (skip gracefully if package absent)
        if _has_sglang_router; then
            run_trial sglang_router cache_aware
        else
            echo ""
            echo "INFO: sglang_router package not found — skipping sglang_router trial."
            echo "      Install with: pip install sglang[router]  to run the full suite."
        fi
        ;;

    *)
        echo "Error: unknown LB_STRATEGY=${LB_STRATEGY}. Use 'least_inflight', 'sglang_router', or 'both'."
        exit 1
        ;;
esac

echo ""
echo "All requested rollout-lb E2E trials completed successfully."
