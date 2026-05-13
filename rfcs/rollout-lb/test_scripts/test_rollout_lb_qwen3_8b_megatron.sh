#!/usr/bin/env bash
# Rollout LB E2E test — Qwen3-8B | Megatron training | sglang rollout
#
# Adapted from examples/grpo_trainer/run_qwen3_8b_megatron.sh with rollout
# load-balancing (least_inflight and sglang_router) layered on top.
#
# Usage:
#   bash test_rollout_lb_qwen3_8b_megatron.sh
#   LB_STRATEGY=sglang_router bash test_rollout_lb_qwen3_8b_megatron.sh
#   LB_STRATEGY=both bash test_rollout_lb_qwen3_8b_megatron.sh
#
# Required env vars (override defaults as needed):
#   MODEL_PATH   – path to Qwen3-8B checkpoint
#   TRAIN_FILE   – parquet with training prompts
#   TEST_FILE    – parquet with validation prompts
#   NUM_GPUS     – total GPU count on this node (default: 8)
# IMAGE: afo.docker.image.name": "registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/ai-search/training_ubuntu22_cuda12.9_python3.12_fp8e2e_a8708562:1.0.1

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
LB_STRATEGY=${LB_STRATEGY:-"sglang_router"}  # least_inflight | sglang_router | both

EXTRA_ARGS=("$@")

MODEL_PATH=${MODEL_PATH:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/wangshulin02/models/huggingface.co/Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/gsm8k/test.parquet"}

# ---------------------------------------------------------------------------
# Training hyperparameters (from run_qwen3_8b_megatron.sh)
# ---------------------------------------------------------------------------
train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-2}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_math_lb}

# ---------------------------------------------------------------------------
# Rollout settings
# ---------------------------------------------------------------------------
GEN_TP=${rollout_tp}
num_replicas=$(( NUM_GPUS / GEN_TP ))

if [ $(( GEN_TP * num_replicas )) -ne "${NUM_GPUS}" ]; then
    echo "Error: rollout_tp=${GEN_TP} does not evenly divide NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi

if [ "${num_replicas}" -lt 2 ]; then
    echo "Warning: num_replicas=${num_replicas} — load balancing has no effect with a single replica." >&2
fi

# ---------------------------------------------------------------------------
# Helper: check sglang_router package availability
# ---------------------------------------------------------------------------
_has_sglang_router() {
    python3 -c "import sglang_router" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Helper: run one training trial
#   $1 = lb_strategy   (least_inflight | sglang_router)
#   $2 = router_policy (cache_aware | round_robin | …)
# ---------------------------------------------------------------------------
run_trial() {
    local lb_strategy="$1"
    local router_policy="${2:-cache_aware}"
    local exp_tag="lb-${lb_strategy}-qwen3-8b-megatron"

    echo ""
    echo "========================================================="
    echo " Running trial: strategy=${lb_strategy}  policy=${router_policy}"
    echo "========================================================="

    local common_params=(
        data.train_files="${TRAIN_FILE}"
        data.val_files="${TEST_FILE}"
        data.train_batch_size=${train_batch_size}
        data.max_prompt_length=${max_prompt_length}
        data.max_response_length=${max_response_length}
        data.filter_overlong_prompts=True
        data.truncation='error'
        actor_rollout_ref.model.path="${MODEL_PATH}"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.hybrid_engine=True
        actor_rollout_ref.rollout.name=sglang
        actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
        actor_rollout_ref.rollout.n=${rollout_n}
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.rollout.checkpoint_engine.backend=naive
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
        actor_rollout_ref.actor.optim.lr=${actor_lr}
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
        actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
        actor_rollout_ref.actor.strategy=megatron
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
        critic.strategy=megatron
        algorithm.adv_estimator=grpo
        algorithm.use_kl_in_reward=False
        "trainer.logger=['console','tensorboard']"
        trainer.project_name=${project_name}
        trainer.experiment_name="${exp_tag}"
        trainer.n_gpus_per_node=${NUM_GPUS}
        trainer.nnodes=1
        trainer.save_freq=${save_freq}
        trainer.test_freq=${test_freq}
        trainer.total_epochs=${total_epochs}
        trainer.balance_batch=True
        model_engine=megatron
        actor_rollout_ref.rollout.load_balance.strategy=${lb_strategy}
    )

    if [ "${lb_strategy}" = "sglang_router" ]; then
        common_params+=(
            actor_rollout_ref.rollout.load_balance.router.policy=${router_policy}
            actor_rollout_ref.rollout.load_balance.router.cache_threshold=0.3
            actor_rollout_ref.rollout.load_balance.router.balance_abs_threshold=64
            actor_rollout_ref.rollout.load_balance.router.balance_rel_threshold=1.5
            actor_rollout_ref.rollout.load_balance.router.health_check_interval_secs=60
        )
    fi

    python3 -X faulthandler -m verl.trainer.main_ppo \
        --config-name='ppo_megatron_trainer.yaml' \
        "${common_params[@]}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

    echo "Trial [${exp_tag}] completed successfully."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "Rollout LB E2E test  |  model=Qwen3-8B  |  actor=megatron  |  lb=${LB_STRATEGY}  |  gpus=${NUM_GPUS}"

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
        run_trial least_inflight
        if _has_sglang_router; then
            run_trial sglang_router cache_aware
        else
            echo "INFO: sglang_router package not found — skipping sglang_router trial."
        fi
        ;;
    *)
        echo "Error: unknown LB_STRATEGY=${LB_STRATEGY}. Use 'least_inflight', 'sglang_router', or 'both'."
        exit 1
        ;;
esac

echo ""
echo "All requested rollout-lb E2E trials completed successfully."
