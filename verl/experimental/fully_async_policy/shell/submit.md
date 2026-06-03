
```shell
RAY_ADDRESS='http://33.253.255.222:44390' \
ray job submit \
--runtime-env test_router/runtime_env.yaml \
--working-dir . \
-- bash test_router/test_capacity_aware_scheduler.sh

```


