#!/usr/bin/env bash
# Run all 3 GRPO variants sequentially for fair comparison.
# Qwen3-30B-A3B, GSM8K, 4×8 GPUs.

set -euo pipefail
SCRIPT_DIR="$(dirname "$0")"

for variant in "normal" "pg" "magi"; do
  case "$variant" in
    normal) USE_PG=false  USE_MAGI=false ;;
    pg)     USE_PG=true   USE_MAGI=false ;;
    magi)   USE_PG=false  USE_MAGI=true  ;;
  esac

  echo ""
  echo "=== START $variant $(date) ==="
  USE_PG=$USE_PG USE_MAGI=$USE_MAGI \
    bash "$SCRIPT_DIR/run_qwen3_30b_a3b_grpo_4x8.sh"
  echo "=== DONE $variant $(date) ==="
done
