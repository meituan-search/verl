#!/usr/bin/env bash
# CP=2 prefix-tree benchmark: TP=4, CP=2 (8 GPUs total)
# Usage (from repo root): bash scripts/run_cp2_test.sh [fa3|magi] [single|multi] [steps]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

TAG=${1:-fa3}       # fa3 or magi
MODE=${2:-single}   # single or multi
STEPS=${3:-3}

export RAY_DEDUP_LOGS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}:/home/hadoop-djst-algoplat/.cache/huggingface/modules"
export HF_MODULES_CACHE=/home/hadoop-djst-algoplat/.cache/huggingface/modules
export RAY_JOB_CONFIG_JSON_ENV_VAR='{"runtime_env": {"env_vars": {
  "PYTHONPATH": "'"${PYTHONPATH}"'",
  "HF_MODULES_CACHE": "/home/hadoop-djst-algoplat/.cache/huggingface/modules",
  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
  "RAY_DEDUP_LOGS": "0"
}}}'

MIMO_PATH=/tmp/claude/MiMo-7B-RL

if [ "$MODE" = "single" ]; then
    DATASET_DIR="${REPO_DIR}/data/gsm8k_sft_10240"
    GBS=16; MBS=4; MAX_LEN=10240; MAX_TOK=45000
    SHUFFLE=True; BALANCE=True
else
    DATASET_DIR=/tmp/claude/gsm8k_tree_2branch
    GBS=4; MBS=4; MAX_LEN=14000; MAX_TOK=60000
    SHUFFLE=False; BALANCE=False
fi

USE_PT=False; PT_ATTN=""
[ "$TAG" = "magi" ] && USE_PT=True && PT_ATTN="data.prefix_tree_attention=magi"

OUTDIR=/tmp/claude/cp2_submit_test/${TAG}_${MODE}
mkdir -p "$OUTDIR"

echo "=== CP=2 $TAG $MODE (gbs=$GBS mbs=$MBS) === repo: $REPO_DIR ==="

cd "$REPO_DIR"
python -m verl.trainer.sft_trainer_ray \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 \
    data.train_files=$DATASET_DIR/train.parquet \
    data.val_files=$DATASET_DIR/test.parquet \
    data.train_batch_size=$GBS data.micro_batch_size_per_gpu=$MBS \
    data.pad_mode=no_padding data.truncation=right data.max_length=$MAX_LEN \
    data.use_dynamic_bsz=False data.max_token_len_per_gpu=$MAX_TOK \
    data.messages_key=messages data.num_workers=0 \
    data.use_prefix_tree=$USE_PT $PT_ATTN \
    data.shuffle=$SHUFFLE \
    model.path=$MIMO_PATH model.use_remove_padding=True model.trust_remote_code=True \
    engine=megatron optim=megatron \
    optim.lr=1e-5 optim.min_lr=1e-6 optim.lr_warmup_steps=0 \
    optim.weight_decay=0.1 "optim.betas=[0.9,0.95]" \
    optim.clip_grad=1.0 optim.lr_warmup_init=0 optim.lr_decay_style=cosine \
    engine.tensor_model_parallel_size=4 engine.pipeline_model_parallel_size=1 \
    engine.virtual_pipeline_model_parallel_size=null \
    engine.context_parallel_size=2 \
    engine.use_mbridge=True \
    "engine.override_transformer_config.recompute_modules=[]" \
    trainer.test_freq=-1 trainer.save_freq=-1 \
    "trainer.logger=['console']" \
    trainer.project_name=cp2_submit_test trainer.experiment_name=${TAG}_${MODE} \
    trainer.total_training_steps=$STEPS trainer.seed=42 \
    trainer.resume_mode=disable trainer.default_local_dir=/dev/null \
    trainer.balance_batch=$BALANCE profiler.enable=False \
    2>&1 | tee "$OUTDIR/run.log" | \
    grep "step:[0-9]\|prefix_sharing\|PT-TIME\|FA3-TIME\|OutOfMemory\|s/it"

echo "=== Done: $TAG $MODE. Log: $OUTDIR/run.log ==="
