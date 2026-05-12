## FSDP ## Megatron + sglang_router Qwen2.5-0.5B
```shell
RAY_ADDRESS='http://33.32.36.11:44390' \
  ray job submit \
  --runtime-env rfcs/rollout-lb/test_scripts/runtime_env.yaml \
  --working-dir . \
  -- bash rfcs/rollout-lb/test_scripts/test_rollout_lb_e2e.sh
```


## Megatron + sglang_router Qwen3-8B
```shell
RAY_ADDRESS='http://33.32.36.11:44390' \
  ray job submit \
  --runtime-env rfcs/rollout-lb/test_scripts/runtime_env.yaml \
  --working-dir . \
  -- bash rfcs/rollout-lb/test_scripts/test_rollout_lb_qwen3_8b_megatron.sh
```