``` shell
# qwen3_8b sync
RAY_ADDRESS='http://33.32.44.108:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_8b/sync/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_8b/sync/run_qwen3_8b_megatron.sh

# qwen3_8b colocate_async
RAY_ADDRESS='http://33.18.249.36:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_8b/colocate_async/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_8b/colocate_async/run_qwen3_8b_megatron_colocate_async.sh

# qwen3_5_35b sync
RAY_ADDRESS='http://33.32.44.108:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_5_35b/sync/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_5_35b/sync/run_qwen3_5_35b_megatron.sh

# qwen3_5_35b colocate_async
RAY_ADDRESS='http://33.18.249.36:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_5_35b/colocate_async/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_5_35b/colocate_async/run_qwen3_5_35b_megatron_colocate_async.sh

# qwen3_30b_a3b sync
RAY_ADDRESS='http://33.32.44.108:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_30b_a3b/sync/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_30b_a3b/sync/run_qwen3_30b_a3b_megatron.sh

# qwen3_30b_a3b colocate_async
RAY_ADDRESS='http://33.18.249.36:44390' ray job submit \
  --runtime-env exp_scripts/qwen3_30b_a3b/colocate_async/runtime_env.yaml \
  --working-dir . \
  -- bash exp_scripts/qwen3_30b_a3b/colocate_async/run_qwen3_30b_a3b_megatron_colocate_async.sh
```
