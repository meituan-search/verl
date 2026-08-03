``` shell
RAY_ADDRESS='http://33.32.44.108:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_8b/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_8b/run_qwen3_8b_megatron.sh

```