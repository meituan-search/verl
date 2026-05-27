#!/usr/bin/env bash
# Ray multi-node launcher for 4×8 GRPO
#
# Usage:
#   # On head node:
#   HEAD_ADDR=<head_ip> bash launch_ray_4x8.sh head
#
#   # On each worker node (3 times):
#   HEAD_ADDR=<head_ip> bash launch_ray_4x8.sh worker
#
#   # After all nodes joined, on head:
#   HEAD_ADDR=<head_ip> bash launch_ray_4x8.sh run [extra_args]

set -euo pipefail
export PATH=/usr/local/miniconda3/bin:$PATH

HEAD_ADDR="${HEAD_ADDR:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
NGPUS="${NGPUS:-8}"

MODE="${1:-run}"
shift 2>/dev/null || true

case "$MODE" in
  head)
    echo "Starting Ray head on $HEAD_ADDR:$RAY_PORT"
    ray start --head \
      --node-ip-address="$HEAD_ADDR" \
      --port=$RAY_PORT \
      --num-gpus=$NGPUS \
      --block &
    ;;

  worker)
    echo "Joining Ray cluster at $HEAD_ADDR:$RAY_PORT"
    ray start \
      --address="$HEAD_ADDR:$RAY_PORT" \
      --num-gpus=$NGPUS \
      --block &
    ;;

  run)
    echo "Submitting job to Ray cluster at $HEAD_ADDR:$RAY_PORT"
    SCRIPT_DIR="$(dirname "$0")"
    RAY_ADDRESS="$HEAD_ADDR:$RAY_PORT" \
    NNODES=4 \
    NGPUS_PER_NODE=8 \
      bash "$SCRIPT_DIR/run_qwen3_30b_a3b_grpo_4x8.sh" "$@"
    ;;

  *)
    echo "Usage: $0 head|worker|run [extra_args]"
    exit 1
    ;;
esac
