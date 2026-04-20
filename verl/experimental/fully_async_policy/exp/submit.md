
```shell

RAY_ADDRESS='http://33.18.250.9:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp1_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp1_colocate_baseline.sh

RAY_ADDRESS='http://33.18.250.9:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp2_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp2_fully_async_no_recompute.sh

RAY_ADDRESS='http://33.32.21.49:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp3_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp3_fully_async_cpu_recompute.sh

RAY_ADDRESS='http://33.18.246.72:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp4_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp4_fully_async_model_engine_server.sh

RAY_ADDRESS='http://33.18.236.7:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp4_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp4_fully_async_model_engine_server.sh

RAY_ADDRESS='http://33.32.35.36:44390' \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/exp/exp5_runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/exp/exp5_fully_async_model_engine_server_fsdp.sh

RAY_ADDRESS='http://33.18.250.9:44390'  \
ray job submit \
--runtime-env verl/experimental/fully_async_policy/shell/runtime_env.yaml \
--working-dir . \
-- bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_8_8_8.sh



