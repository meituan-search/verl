# 测试覆盖说明

共 3 个测试文件，**116 个测试用例**（含参数化展开后），全部无需 GPU / Ray 集群 / Rust binary。

---

## 一、`test_sglang_router_actor.py` — 30 个用例

测试对象：`SGLangRouterActor`（[verl/workers/rollout/sglang_router_actor.py](../../verl/workers/rollout/sglang_router_actor.py)）

### TestSGLangRouterActorInit（8 个）

验证构造函数将参数正确传递给内部 `RouterArgs`。

| 用例 | 测试点 |
|---|---|
| `test_address_format` | `_address` 字段格式为 `http://{host}:{port}` |
| `test_thread_initially_none` | 构造后线程为 `None`，未提前启动 |
| `test_default_request_id_headers` | 未传 `request_id_headers` 时默认值为 `["x-verl-request-id"]` |
| `test_custom_request_id_headers` | 自定义 header 被透传到 `RouterArgs` |
| `test_worker_urls_passed_through` | `worker_urls` 列表原样传给 `RouterArgs.worker_urls` |
| `test_policy_passed_through` | `policy` 字段被透传（如 `round_robin`） |
| `test_cache_aware_thresholds` | `cache_threshold`、`balance_abs_threshold`、`balance_rel_threshold` 全部透传 |
| `test_health_check_params` | `health_check_interval_secs`、`health_failure_threshold`、`health_success_threshold` 透传 |

### TestSGLangRouterActorPDMode（4 个）

验证 PD disaggregation 模式下参数路径正确。

| 用例 | 测试点 |
|---|---|
| `test_pd_mode_worker_urls_cleared` | `pd_disaggregation=True` 时 `worker_urls` 必须为空（不能混入普通 replica 地址） |
| `test_pd_mode_prefill_decode_urls` | prefill/decode 地址分别传入并保留 |
| `test_pd_mode_separate_policies` | `prefill_policy` 和 `decode_policy` 可独立配置 |
| `test_pd_mode_empty_prefill_urls_defaults` | 未传 prefill/decode urls 时默认为空列表，不报错 |

### TestSGLangRouterActorStart（4 个）

验证 `start()` 的 daemon thread 行为。

| 用例 | 测试点 |
|---|---|
| `test_start_spawns_daemon_thread` | 调用后 `_thread` 非空且 `is_alive()==True`，且 `daemon==True` |
| `test_start_calls_router_start` | daemon thread 实际执行了 `router.start()`（通过 flag 验证） |
| `test_start_sets_thread_name` | 线程名为 `"sglang-rust-router"`，便于监控和排查 |
| `test_thread_exits_after_router_stops` | router stop 后线程退出，`is_alive()==False` |

### TestSGLangRouterActorIdempotent（1 个）

| 用例 | 测试点 |
|---|---|
| `test_double_start_does_not_spawn_second_thread` | 连续两次调用 `start()` 不创建第二个线程，返回同一线程对象 |

### TestSGLangRouterActorReady（6 个）

验证 `wait_until_ready()` 的轮询逻辑和超时行为。

| 用例 | 测试点 |
|---|---|
| `test_wait_until_ready_success` | `/health` 返回 200 时立即返回 `True` |
| `test_wait_until_ready_timeout` | `/health` 持续返回非 200 时抛出 `TimeoutError` |
| `test_wait_until_ready_connection_errors_then_success` | `ConnectionError` 被静默重试，最终 200 则成功 |
| `test_wait_until_ready_non_200_then_success` | 非 200 → 非 200 → 200 的序列，最终成功 |
| `test_wait_until_ready_polls_correct_url` | 请求的 URL 格式为 `http://{host}:{port}/health` |
| `test_wait_until_ready_request_timeout_retried` | `requests.Timeout`（`RequestException` 子类）也被重试 |

### TestSGLangRouterActorAccessors（5 个）

验证只读访问器方法。

| 用例 | 测试点 |
|---|---|
| `test_get_address_format` | 返回 `http://{host}:{port}` 格式字符串 |
| `test_get_address_localhost` | 默认测试用地址 `127.0.0.1:30001` |
| `test_is_alive_before_start` | 未调用 `start()` 时返回 `False` |
| `test_is_alive_after_start` | `start()` 后返回 `True` |
| `test_is_alive_after_router_stops` | router 停止后返回 `False` |

### TestWaitUntilReadyRealHTTP（2 个）

使用真实 `HTTPServer` 做端到端验证，覆盖真实 socket 行为。

| 用例 | 测试点 |
|---|---|
| `test_polls_real_health_endpoint` | 有真实 HTTP server 监听 `/health` 时，`wait_until_ready` 成功返回 |
| `test_timeout_with_no_server` | 目标端口无监听进程时，按时超时并抛出 `TimeoutError` |

---

## 二、`test_llm_server_router.py` — 43 个用例

测试对象：`_parse_token_output()`、`LLMServerClient`、`LLMServerManager`（[verl/workers/rollout/llm_server.py](../../verl/workers/rollout/llm_server.py)）

### TestParseTokenOutput（14 个）

验证 sglang HTTP `/generate` 响应 JSON → `TokenOutput` 的解析逻辑，是 Router 路径下唯一的反序列化层。

| 用例 | 测试点 |
|---|---|
| `test_basic_token_ids` | `output_ids` 被正确提取为 `token_ids` 列表 |
| `test_empty_output_ids` | 空 `output_ids` 不报错，返回空列表 |
| `test_missing_output_ids` | 响应中无 `output_ids` 键时默认空列表 |
| `test_stop_reason_from_finish_reason` | `meta_info.finish_reason.type` 映射到 `stop_reason` |
| `test_stop_reason_none_when_absent` | 无 `finish_reason` 时 `stop_reason` 为 `None` |
| `test_log_probs_extracted_when_lengths_match` | `output_token_logprobs` 与 `output_ids` 等长时提取 logprob 值 |
| `test_log_probs_none_when_lengths_mismatch` | 长度不匹配时 `log_probs` 为 `None`（防止索引越界） |
| `test_log_probs_none_when_absent` | 无 `output_token_logprobs` 时 `log_probs` 为 `None` |
| `test_num_preempted_default_zero` | 无 `num_preempted` 字段时默认为 0 |
| `test_num_preempted_extracted` | `meta_info.num_preempted` 被正确提取 |
| `test_global_steps_in_extra_fields` | `meta_info.global_steps` 被放入 `extra_fields["global_steps"]` |
| `test_global_steps_absent_means_empty_extra_fields` | 无 `global_steps` 时 `extra_fields` 为空字典 |
| `test_routed_experts_none_when_absent` | 无 `routed_experts` 时字段为 `None` |
| `test_routed_experts_converted` | `routed_experts` 被转换（MoE 路由回放用） |

### TestLLMServerClientRouterInit（3 个）

验证 `LLMServerClient.__init__` 的双模式构造。

| 用例 | 测试点 |
|---|---|
| `test_router_address_stored` | `router_address` 模式：地址被存储，`_load_balancer` 为 `None`，`servers` 为空字典 |
| `test_legacy_path_stored` | 旧模式：`_load_balancer` 和 `servers` 被存储，`_router_address` 为 `None` |
| `test_session_initially_none` | `_session` 初始为 `None`，懒初始化 |

### TestGenerateViaRouter（6 个）

验证 `_generate_via_router()` 的 HTTP 调用行为和重试策略。

| 用例 | 测试点 |
|---|---|
| `test_happy_path_returns_token_output` | 正常响应 200 时返回正确的 `TokenOutput` |
| `test_request_id_header_injected` | `x-verl-request-id` header 被注入，值为传入的 `request_id` |
| `test_url_constructed_from_router_address` | 请求 URL 为 `{router_address}/generate` |
| `test_4xx_raises_immediately_no_retry` | 4xx 错误立即抛出 `ClientResponseError`，不重试（一次调用） |
| `test_5xx_retries_then_raises_runtime_error` | 5xx 连续失败 3 次后抛出 `RuntimeError("3 attempts")`，共调用 3 次 |
| `test_connector_error_retries_then_raises` | 连接错误（`ClientConnectorError`）同样重试 3 次后抛出 |

### TestGenerateViaRouterLogprobs（2 个）

验证 vLLM 风格的 `prompt_logprobs` 参数向 sglang 参数的转换。

| 用例 | 测试点 |
|---|---|
| `test_prompt_logprobs_translated_to_sglang_params` | `sampling_params["prompt_logprobs"]=K` 被转换为 `return_logprob=True` + `logprob_start_len=0` + `top_logprobs_num=K`，原 key 不出现在 payload 中 |
| `test_logprobs_zero_sets_return_logprob` | `sampling_params["logprobs"]=True` 被转换为 `return_logprob=True`，不注入 `top_logprobs_num` |

### TestSessionManagement（6 个）

验证 aiohttp session 的生命周期管理，确保连接池复用且无资源泄漏。

| 用例 | 测试点 |
|---|---|
| `test_session_created_lazily` | 第一次调用 `_get_or_create_session()` 时创建 session |
| `test_session_reused_on_second_call` | 第二次调用返回同一 session 对象，不创建新 `ClientSession` |
| `test_closed_session_is_recreated` | session 已关闭时自动重新创建 |
| `test_close_closes_session` | `close()` 调用 `session.close()` |
| `test_close_noop_when_session_none` | session 为 `None` 时 `close()` 不报错 |
| `test_close_noop_when_already_closed` | session 已关闭时 `close()` 不重复调用 |

### TestLLMServerManagerInitLB（4 个）

验证 `_init_global_load_balancer()` 的路由决策逻辑——核心分支判断。

| 用例 | 测试点 |
|---|---|
| `test_least_inflight_creates_global_lb` | `strategy=least_inflight` 时创建 `GlobalRequestLoadBalancer`，`router_address` 为 `None` |
| `test_sglang_router_strategy_with_non_sglang_engine_falls_back` | `strategy=sglang_router` 但引擎为 vllm 时，回退到 least_inflight（不启动 Router） |
| `test_missing_load_balance_config_defaults_to_least_inflight` | 完全缺失 `load_balance` 配置时，默认走 least_inflight |
| `test_sglang_router_strategy_calls_init_sglang_router` | `strategy=sglang_router` + `name=sglang` 时，调用 `_init_sglang_router()` |

### TestLLMServerManagerGetClient（6 个）

验证 `get_client()` 根据路由模式返回正确的 client 类型和配置。

| 用例 | 测试点 |
|---|---|
| `test_least_inflight_returns_llm_server_client` | 旧模式返回 `LLMServerClient`，持有 `_load_balancer` |
| `test_least_inflight_fully_async_returns_fully_client` | `fully_async=True` 返回 `FullyLLMServerClient` |
| `test_router_mode_returns_client_with_router_address` | Router 模式返回带 `_router_address` 的 `LLMServerClient` |
| `test_router_mode_fully_async_returns_fully_client` | Router 模式 `fully_async=True` 返回 `FullyLLMServerClient` |
| `test_router_mode_client_has_no_load_balancer` | Router 模式 client 的 `_load_balancer` 为 `None` |
| `test_multiple_get_client_calls_return_independent_objects` | 每次调用返回新对象（不共享 session 状态），但 `router_address` 相同 |

### TestLLMServerManagerInitRouter（2 个）

验证 `_init_sglang_router()` 正确启动 Router actor 并存储地址。

| 用例 | 测试点 |
|---|---|
| `test_router_actor_started_and_address_stored` | actor 被创建，`start()`/`wait_until_ready()`/`get_address()` 均被调用；`router_address` 存储，`global_load_balancer` 置 `None` |
| `test_worker_urls_are_server_addresses` | 传给 Router 的 `worker_urls` 精确等于 `server_addresses`；`request_id_headers` 为 `["x-verl-request-id"]` |

---

## 三、`test_load_balance_config.py` — 43 个用例

测试对象：`SGLangRouterConfig`、`LoadBalanceConfig`（[verl/workers/config/load_balance.py](../../verl/workers/config/load_balance.py)）

### TestSGLangRouterConfigDefaults（6 个）

| 用例 | 测试点 |
|---|---|
| `test_default_policy` | 默认 `policy="cache_aware"` |
| `test_default_decode_policy` | 默认 `decode_policy="power_of_two"` |
| `test_default_cache_threshold` | 默认 `cache_threshold=0.3` |
| `test_default_balance_abs_threshold` | 默认 `balance_abs_threshold=64` |
| `test_default_balance_rel_threshold` | 默认 `balance_rel_threshold=1.5` |
| `test_default_health_check_interval` | 默认 `health_check_interval_secs=60` |

### TestSGLangRouterConfigValidPolicies（7 个，参数化）

| 用例 | 测试点 |
|---|---|
| `test_valid_policy[cache_aware/round_robin/consistent_hashing/prefix_hash]` | 4 个合法 `policy` 值均不抛异常 |
| `test_valid_decode_policy[power_of_two/round_robin/random]` | 3 个合法 `decode_policy` 值均不抛异常 |

### TestSGLangRouterConfigInvalidPolicies（3 个）

| 用例 | 测试点 |
|---|---|
| `test_invalid_policy_raises` | 非法 `policy` 值抛 `ValueError`，错误信息含 `"policy="` |
| `test_invalid_decode_policy_raises` | 将 router policy 值（如 `cache_aware`）赋给 `decode_policy` 抛 `ValueError` |
| `test_empty_policy_raises` | 空字符串 `policy` 抛 `ValueError` |

### TestSGLangRouterConfigThresholdValidation（10 个）

验证数值型字段的边界约束。

| 用例 | 测试点 |
|---|---|
| `test_cache_threshold_zero` | 边界值 `0.0` 合法 |
| `test_cache_threshold_one` | 边界值 `1.0` 合法 |
| `test_cache_threshold_above_one_raises` | `1.1` 超出 `[0,1]` 抛 `ValueError` |
| `test_cache_threshold_negative_raises` | 负值抛 `ValueError` |
| `test_balance_abs_threshold_zero` | `0` 合法（禁用负载均衡覆盖） |
| `test_balance_abs_threshold_negative_raises` | 负值抛 `ValueError` |
| `test_balance_rel_threshold_exactly_one` | `1.0` 合法（最小有效值） |
| `test_balance_rel_threshold_below_one_raises` | `< 1.0` 抛 `ValueError` |
| `test_health_check_interval_zero_raises` | `0` 抛 `ValueError`（间隔必须 > 0） |
| `test_health_check_interval_negative_raises` | 负值抛 `ValueError` |

### TestSGLangRouterConfigCustomValues（1 个）

| 用例 | 测试点 |
|---|---|
| `test_custom_all_fields` | 一次性设置所有字段为非默认值，全部正确存储 |

### TestLoadBalanceConfigDefaults（3 个）

| 用例 | 测试点 |
|---|---|
| `test_default_strategy` | 默认 `strategy="least_inflight"` |
| `test_default_router_is_sglang_router_config` | 默认 `router` 是 `SGLangRouterConfig` 实例（非 dict） |
| `test_default_router_has_default_policy` | 嵌套默认 `router.policy="cache_aware"` |

### TestLoadBalanceConfigValidStrategies（2 个，参数化）

| 用例 | 测试点 |
|---|---|
| `test_valid_strategy[least_inflight/sglang_router]` | 两个合法策略均不抛异常 |

### TestLoadBalanceConfigInvalidStrategy（2 个）

| 用例 | 测试点 |
|---|---|
| `test_invalid_strategy_raises` | Router 的 policy 值（如 `round_robin`）不能作为 strategy，抛 `ValueError` |
| `test_empty_strategy_raises` | 空字符串 strategy 抛 `ValueError` |

### TestLoadBalanceConfigNestedRouterCoercion（4 个）

验证 Hydra/OmegaConf 传入 dict 时的自动类型转换（`__post_init__` coerce 逻辑）。

| 用例 | 测试点 |
|---|---|
| `test_router_from_dict` | `router={"policy": "round_robin"}` 被自动转为 `SGLangRouterConfig` 实例 |
| `test_router_from_dict_with_all_fields` | dict 包含所有字段时，全部正确转换 |
| `test_router_from_invalid_type_raises` | `router=42`（非 dict/DictConfig/SGLangRouterConfig）抛 `TypeError` |
| `test_router_invalid_policy_in_dict_raises` | dict 中含非法 policy 值时，在构造 `SGLangRouterConfig` 阶段抛 `ValueError` |

### TestLoadBalanceConfigMapping（4 个）

验证 `BaseConfig` 的 `Mapping` 接口，确保配置对象可像字典一样使用。

| 用例 | 测试点 |
|---|---|
| `test_getitem_strategy` | `cfg["strategy"]` 正常返回字段值 |
| `test_get_with_default` | `cfg.get("不存在", "fallback")` 返回默认值 |
| `test_iter_yields_field_names` | 迭代产出所有字段名（包含 `strategy`、`router`） |
| `test_len` | `len(cfg)` 返回字段数量（≥ 2） |

### TestLoadBalanceConfigOmegaConfCoercion（1 个）

| 用例 | 测试点 |
|---|---|
| `test_omegaconf_dict_config_coercion` | OmegaConf `DictConfig` 对象经 `to_container` 转换后能正确构造 `LoadBalanceConfig`（模拟 Hydra 实际传参路径） |

---

## 四、覆盖矩阵

| 功能层 | 测试文件 | 测试组 | 用例数 |
|---|---|---|---|
| Router actor 构造/参数透传 | `test_sglang_router_actor` | Init + PDMode | 12 |
| Router actor 线程生命周期 | `test_sglang_router_actor` | Start + Idempotent | 5 |
| Router actor 就绪探测 | `test_sglang_router_actor` | Ready + RealHTTP | 8 |
| Router actor 访问器 | `test_sglang_router_actor` | Accessors | 5 |
| JSON 响应解析 | `test_llm_server_router` | ParseTokenOutput | 14 |
| LLMServerClient 初始化 | `test_llm_server_router` | ClientRouterInit | 3 |
| HTTP generate 路径 | `test_llm_server_router` | GenerateViaRouter | 6 |
| logprobs 参数转换 | `test_llm_server_router` | GenerateViaRouterLogprobs | 2 |
| aiohttp session 管理 | `test_llm_server_router` | SessionManagement | 6 |
| Manager 路由分支决策 | `test_llm_server_router` | ManagerInitLB | 4 |
| Manager get_client 返回类型 | `test_llm_server_router` | ManagerGetClient | 6 |
| Manager Router 初始化参数 | `test_llm_server_router` | ManagerInitRouter | 2 |
| SGLangRouterConfig 字段与校验 | `test_load_balance_config` | RouterConfig 系列 | 27 |
| LoadBalanceConfig 字段与校验 | `test_load_balance_config` | LBConfig 系列 | 16 |
| **合计** | | | **116** |
