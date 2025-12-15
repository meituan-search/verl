``` shell
git clone https://github.com/volcengine/verl.git
```

1.将verl-recipe/retool/retool.py中的dataframe = dataframe.map(self.map_fn2, num_proc=16)改为dataframe = dataframe.map(self.map_fn2)
2.在verl/experimental/reward/reward_loop/dapo.py中添加：

        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

curl  "http://10.110.137.235:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.134.100:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'

``` shell
sandbox:
curl  "http://10.110.134.100:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.137.235:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.135.200:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.135.240:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.138.150:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
curl  "http://10.110.138.172:8080/run_code"   -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'


```


``` shell
1.fp16 qwen2.5-7b
RAY_ADDRESS='http://33.32.26.71:44390' \
ray job submit \
--runtime-env retool-fp16/qwen2.5_7b/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen2.5_7b/fp16/test_dapo_2.5_7b_fsdp_fp16.sh

2.bf16
RAY_ADDRESS='http://33.253.158.82:44390' \
ray job submit \
--runtime-env retool-fp16/qwen2.5_7b/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen2.5_7b/bf16/test_dapo_2.5_7b_fsdp_bf16.sh
```


qwen3-8b-base

``` shell
1.bf16 
RAY_ADDRESS='http://33.18.249.103:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/bf16/run_qwen3_8b_dapo_bf16.sh

2.bf16 qwen3-8b-base-sft372
RAY_ADDRESS='http://33.32.40.27:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/bf16/run_qwen3_8b_dapo_bf16.sh

2.bf16 qwen3-8b-base-sft620
RAY_ADDRESS='http://33.32.53.67:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/bf16/run_qwen3_8b_dapo_bf16.sh

3.fp16 
RAY_ADDRESS='http://33.253.158.82:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/fp16/run_qwen3_8b_dapo_fp16.sh

4.fp16 qwen3-8b-base-sft372
RAY_ADDRESS='http://33.32.4.100:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/fp16/run_qwen3_8b_dapo_fp16.sh

4.fp16 qwen3-8b-base-sft620
RAY_ADDRESS='http://33.18.250.44:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/fp16/run_qwen3_8b_dapo_fp16.sh
```


``` shell fsdp
2.bf16 qwen3-8b-base-sft620-ckpt
RAY_ADDRESS='http://33.32.53.67:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base/bf16/run_qwen3_8b_dapo_bf16.sh

4.fp16 qwen3-8b-base-sft620-ckpt
RAY_ADDRESS='http://33.253.195.189:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/fp16/run_qwen3_8b_dapo_fp16.sh

5.fp16 qwen3-8b-base-fsdpsft620-fsdp
RAY_ADDRESS='http://33.18.244.88:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/fp16/run_qwen3_8b_dapo_fp16.sh
```

todo !!

``` shell
bf16-fsdp-620 "http://10.110.138.150:8080/run_code"
RAY_ADDRESS='http://33.32.4.100:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/bf16/run_qwen3_8b_dapo_bf16.sh

fp16-fsdp-620-retry qwen3-8b-base-fsdpsft620-fsdp-retry
"http://10.110.134.100:8080/run_code"

RAY_ADDRESS='http://33.32.40.27:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/fp16/run_qwen3_8b_dapo_fp16.sh

bf16-fsdp-372
"http://10.110.137.235:8080/run_code"
RAY_ADDRESS='http://33.18.250.44:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/bf16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/bf16/run_qwen3_8b_dapo_bf16.sh

fp16-fsdp-372
"http://10.110.138.172:8080/run_code"
RAY_ADDRESS='http://33.18.244.88:44390' \
ray job submit \
--runtime-env retool-fp16/qwen3_8b_base_fsdp/fp16/runtime_env.yaml \
--working-dir . \
-- bash retool-fp16/qwen3_8b_base_fsdp/fp16/run_qwen3_8b_dapo_fp16.sh

```