# Load Balance RFC — 文件索引

## 总览

| 文件 | 内容 | 改动文件 |
|---|---|---|
| [方案B_sglang_rust_router_sidecar.md](方案B_sglang_rust_router_sidecar.md) | 方案 B 完整分析（架构、优缺点、实施路径） | — |
| [task_01_sglang_router_actor.md](task_01_sglang_router_actor.md) | Rust Router 的 Ray actor 封装 | 新增 `verl/workers/rollout/sglang_router_actor.py` |
| [task_02_llm_server_manager_dual_track.md](task_02_llm_server_manager_dual_track.md) | LLMServerManager 双轨改造（数据平面/控制平面分离） | `verl/workers/rollout/llm_server.py` |
| [task_03_llm_server_client_http.md](task_03_llm_server_client_http.md) | LLMServerClient 接口层改造（Ray RPC → aiohttp） | `verl/workers/rollout/llm_server.py` |
| [task_04_pd_router_integration.md](task_04_pd_router_integration.md) | PD 模式集成（子方案 B：Router 只路由 prefill） | `verl/workers/rollout/sglang_rollout/sglang_pd_replica.py` |
| [task_05_config_schema.md](task_05_config_schema.md) | Config schema 设计（rollout.yaml + dataclass） | `verl/trainer/config/rollout/rollout.yaml`，`verl/workers/config.py` |
| [usage_guide.md](usage_guide.md) | **用户使用说明**（快速上手、所有配置项、典型示例、注意事项、故障排查） | — |
| [test_coverage.md](test_coverage.md) | **测试覆盖说明**（116 个用例，每条用例的测试点逐一说明） | — |

## 实施顺序

```
Task 05（Config）→ Task 01（RouterActor）→ Task 02（Manager）→ Task 03（Client）→ Task 04（PD）
```

Task 05 最先确认（其他 task 参照配置字段名），Task 04 最后（依赖 01+02 完成）。
