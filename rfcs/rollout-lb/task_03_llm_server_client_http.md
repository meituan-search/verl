# Task 03：LLMServerClient 接口层改造

> 改动文件：`verl/workers/rollout/llm_server.py`
> 依赖：Task 02（LLMServerManager）
> 风险等级：**高**（接口变化最大的一处）

---

## 背景

当前 `LLMServerClient.generate()` 的调用链：

```
acquire_server(request_id)          # Ray remote call → GlobalRequestLoadBalancer
  → server.generate.remote(...)     # Ray remote call → SGLangHttpServer actor
  → release_server(server_id)       # Ray remote call（fire-and-forget）
```

引入 Router 后，数据平面改为：

```
aiohttp.post(router_url, json=payload, headers={"x-verl-request-id": request_id})
  → Router 内部路由 → sglang server HTTP /generate
```

这涉及三个核心变化：
1. **`__init__` 签名**：支持 `router_address` 模式和旧的 `servers + load_balancer_handle` 模式
2. **`generate()` 逻辑**：从 Ray RPC 改为 aiohttp，包含序列化、错误处理、tracing
3. **`TokenOutput` 反序列化**：从 Ray 自动序列化改为手动 JSON → Pydantic

---

## 详细设计

### 1. `LLMServerClient.__init__` 支持双模式

```python
class LLMServerClient:
    def __init__(
        self,
        config: DictConfig,
        # 旧模式（least_inflight）
        servers: Optional[dict[str, ray.actor.ActorHandle]] = None,
        load_balancer_handle: Optional[ray.actor.ActorHandle] = None,
        # 新模式（sglang_router）
        router_address: Optional[str] = None,
    ):
        self.config = config
        # 新模式
        self._router_address = router_address
        # 旧模式
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle = servers or {}
        # aiohttp session（懒初始化，避免在非 async 上下文创建）
        self._session: Optional[aiohttp.ClientSession] = None
```

### 2. aiohttp session 管理

session 需要在 async 上下文中创建，且应在整个 client 生命周期内复用（避免每次请求新建连接）。

```python
async def _get_or_create_session(self) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
        connector = aiohttp.TCPConnector(
            limit=2000,
            limit_per_host=500,
            ttl_dns_cache=300,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=600.0),  # LLM 生成可能很慢
        )
    return self._session

async def close(self) -> None:
    """关闭 session，在 LLMServerManager 析构时调用。"""
    if self._session and not self._session.closed:
        await self._session.close()
```

**注意**：`AsyncHttpServerAdapter` 在 `_get_session()` 中每次请求新建 session（[http_server_engine.py:628](../../verl/workers/rollout/sglang_rollout/http_server_engine.py)），这是为了避免跨协程共享状态。`LLMServerClient` 场景下每个 client 实例由单个 `AgentLoopWorker` 独占，可以安全地复用 session。

### 3. `generate()` 核心逻辑

```python
@rollout_trace_op
async def generate(
    self,
    request_id: str,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    **kwargs: Any,
) -> TokenOutput:
    if self._router_address:
        return await self._generate_via_router(
            request_id, prompt_ids, sampling_params, image_data, video_data, **kwargs
        )
    else:
        return await self._generate_via_ray(
            request_id, prompt_ids, sampling_params, image_data, video_data, **kwargs
        )
```

### 4. `_generate_via_router()` 详细实现

```python
async def _generate_via_router(
    self,
    request_id: str,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    **kwargs: Any,
) -> TokenOutput:
    url = f"{self._router_address}/generate"

    payload = {
        "rid": uuid4().hex,      # sglang 内部 request id，每次新建
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
    }
    if image_data:
        payload["image_data"] = image_data
    # video_data 暂不支持（sglang TODO 同步）
    payload.update({k: v for k, v in kwargs.items() if v is not None})

    headers = {
        "x-verl-request-id": request_id,   # Router sticky session 依据
        "Content-Type": "application/json",
    }

    session = await self._get_or_create_session()
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return _parse_token_output(data)
        except aiohttp.ClientResponseError as e:
            if e.status < 500:
                raise   # 4xx：客户端错误，不重试
            logger.warning(f"Router returned {e.status}, attempt {attempt + 1}/3")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            logger.warning(f"Router connection error: {e}, attempt {attempt + 1}/3")
        if attempt < 2:
            await asyncio.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s

    raise RuntimeError(f"generate via Router failed after 3 attempts for request {request_id}")
```

### 5. `_parse_token_output()` — JSON → TokenOutput

当前 `SGLangHttpServer.generate()` 返回的是一个 `TokenOutput` Pydantic 对象，Ray 序列化透明处理。改为 HTTP 后，sglang server 的 `/generate` 端点返回的是 sglang 自己的 JSON 格式，需要手动解析。

**sglang `/generate` 响应格式**（参考 `async_sglang_server.py:590-640`）：

```json
{
  "output_ids": [101, 202, ...],
  "meta_info": {
    "finish_reason": {"type": "stop"},
    "output_token_logprobs": [[logprob, token_id, rank], ...],
    "input_token_logprobs": [...],
    "input_top_logprobs": [...],
    "num_preempted": 0,
    "routed_experts": null,
    "global_steps": 42
  }
}
```

```python
def _parse_token_output(data: dict) -> TokenOutput:
    """Parse sglang /generate HTTP response into TokenOutput."""
    meta_info = data.get("meta_info", {})
    token_ids = list(data.get("output_ids", []))

    # logprobs
    log_probs = None
    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
        log_probs = [float(lp) for lp, _, _ in output_token_logprobs]

    # stop reason
    finish_reason = meta_info.get("finish_reason")
    stop_reason = finish_reason["type"] if finish_reason else None

    # routed experts (MoE routing replay)
    routed_experts = meta_info.get("routed_experts")
    if routed_experts is not None:
        import torch
        routed_experts = torch.tensor(routed_experts)

    # num_preempted
    num_preempted = meta_info.get("num_preempted", 0)

    # extra fields
    extra_fields = {}
    global_steps = meta_info.get("global_steps")
    if global_steps is not None:
        extra_fields["global_steps"] = global_steps

    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs,
        routed_experts=routed_experts,
        stop_reason=stop_reason,
        num_preempted=num_preempted,
        extra_fields=extra_fields,
    )
```

**关键差异**：`SGLangHttpServer.generate()` 当前还会处理 `prompt_logprobs`（调用 `_extract_prompt_logprobs_sglang`）。这部分逻辑在 HTTP 路径下需要同样在 `_parse_token_output` 中实现，根据 `sampling_params` 中是否有 `prompt_logprobs` 键来决定。

### 6. `_generate_via_ray()` — 旧逻辑无改动

```python
async def _generate_via_ray(self, request_id, prompt_ids, sampling_params, image_data, video_data, **kwargs):
    # 原有逻辑原封不动，仅重命名
    server_id, server = await self._acquire_server(request_id)
    try:
        output: TokenOutput = await server.generate.remote(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )
        return output
    finally:
        self._release_server(server_id)
```

### 7. `rollout_trace_op` 装饰器兼容性

`@rollout_trace_op` 装饰器（[verl/utils/rollout_trace.py](../../verl/utils/rollout_trace.py)）目前包裹在 `generate()` 上，两条路径共享同一个 span，无需改动。Router 的 HTTP 调用发生在 span 内部，span 上下文不需要传播到 Router（Router 不做 tracing）。

---

## prompt_logprobs 的特殊处理

`FullyLLMServerClient` 被 distillation teacher 使用时，`sampling_params` 可能包含 `prompt_logprobs=K`。当走 Router 路径时：

- payload 中需要包含 `logprob_start_len=0` 和 `top_logprobs_num=K`（sglang 参数），不能直接传 `prompt_logprobs`
- `_generate_via_router()` 需要在构造 payload 前将 `sampling_params["prompt_logprobs"]` 转换为 sglang 参数
- 响应解析时调用 `_extract_prompt_logprobs_sglang()`，逻辑与 `async_sglang_server.py:62-108` 完全相同

```python
# 在 _generate_via_router 中，payload 构造前：
prompt_logprobs = sampling_params.pop("prompt_logprobs", None)
return_logprob = sampling_params.pop("logprobs", False)
if prompt_logprobs is not None:
    return_logprob = True
    payload["logprob_start_len"] = 0
    if prompt_logprobs > 0:
        payload["top_logprobs_num"] = prompt_logprobs
payload["return_logprob"] = return_logprob
```

---

## 风险点汇总

| 风险 | 描述 | 缓解 |
|---|---|---|
| JSON 解析遗漏字段 | sglang 响应格式随版本变化 | 用 `.get()` 防御性解析，缺失字段返回 None |
| prompt_logprobs 路径未测 | teacher distillation 路径复杂 | 单独写集成测试覆盖此路径 |
| session 未关闭 | 程序退出时 aiohttp 告警 | `LLMServerManager` 在 `__del__` 或 `shutdown()` 中调用 `client.close()` |
| 并发 session 创建 | 多个协程同时进入 `_get_or_create_session()` | 加 `asyncio.Lock` 保护初始化 |
