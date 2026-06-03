# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo: 直接向运行中的 vLLM 推理服务测试 CapacityAwareScheduler 的两个核心功能：
  1. VLLMLoadBackend.fetch() —— 拉取 /metrics 并解析 KV-cache 使用率
  2. _CapacityAwareScheduler —— poll 循环、acquire_server 亲和路由、容量门控

用法:
    python demo_scheduler.py                          # 默认地址 33.32.1.89:39079
    python demo_scheduler.py 33.32.1.89:39079         # 指定地址
    python demo_scheduler.py 33.32.1.89:39079 33.32.1.89:39080  # 多副本
"""

from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import MagicMock

# 把项目根目录加入路径，直接复用源码
sys.path.insert(0, "/opt/meituan/dolphinfs_wangshulin02/Projects/verl")

from verl.workers.rollout.capacity_aware_scheduler import (
    VLLMLoadBackend,
    _CapacityAwareScheduler,
)

ADDRESSES = sys.argv[1:] or ["33.32.1.89:38909"]


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: 直接测试 VLLMLoadBackend.fetch()
# ─────────────────────────────────────────────────────────────────────────────
async def test_fetch(addresses: list[str]) -> None:
    print("\n" + "=" * 60)
    print("Part 1: VLLMLoadBackend.fetch()")
    print("=" * 60)
    backend = VLLMLoadBackend()
    for addr in addresses:
        normalized = addr if addr.startswith("http") else f"http://{addr}"
        try:
            metrics = await backend.fetch(normalized)
            print(f"  {addr}")
            print(f"    token_usage      : {metrics.token_usage:.4f}  ({metrics.token_usage * 100:.1f}%)")
            print(f"    num_total_tokens : {metrics.num_total_tokens}")
        except Exception as e:
            print(f"  {addr}  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: 测试 _CapacityAwareScheduler poll 循环 + acquire_server 亲和路由
# ─────────────────────────────────────────────────────────────────────────────
async def test_scheduler(addresses: list[str]) -> None:
    print("\n" + "=" * 60)
    print("Part 2: _CapacityAwareScheduler — poll + acquire_server")
    print("=" * 60)

    # 用 MagicMock 替代 Ray actor handle（不需要真实 Ray 集群）
    servers = {addr: MagicMock(name=f"handle_{addr}") for addr in addresses}

    sched = _CapacityAwareScheduler(
        servers=servers,
        capacity_threshold=0.95,  # 阈值设高，确保测试期间不阻塞
        poll_interval_ms=500,
        load_backend="vllm",
    )

    # 启动 poll 循环
    tasks = [asyncio.ensure_future(sched._poll_loop(addr)) for addr in addresses]
    sched._capacity_event.set()  # 初始允许调度

    # 等 2 个 poll 周期让状态更新
    print("  等待 poll 更新（~1s）...")
    await asyncio.sleep(1.2)

    # 打印 poll 结果
    print("\n  [poll 结果]")
    for addr, state in sched._states.items():
        print(f"  {addr}")
        print(f"    token_usage       : {state.token_usage:.4f}  ({state.token_usage * 100:.1f}%)")
        print(f"    effective_usage   : {sched._effective_usage(state):.4f}")
        print(f"    num_requests_running : {state.num_requests_running}")
        print(f"    num_requests_waiting : {state.num_requests_waiting}")
        print(f"    healthy           : {state.healthy}")
        print(f"    last_polled       : {time.strftime('%H:%M:%S', time.localtime(state.last_polled))}")

    # 测试 acquire_server —— 同一 group_id 应始终路由到同一副本
    print("\n  [acquire_server 亲和路由测试]")
    results: dict[str, str] = {}
    for i in range(6):
        gid = f"group-{i % 3}"  # 3 个 group，每个请求 2 次，验证 sticky
        sid, _ = await sched.acquire_server(f"req-{i}", group_id=gid)
        sched.release_server(sid)
        prev = results.get(gid)
        sticky = "✓ sticky" if prev == sid else ("(first)" if prev is None else f"✗ CHANGED from {prev}")
        results[gid] = sid
        print(f"  req-{i}  group={gid}  → {sid}  {sticky}")

    # 打印 get_status
    print("\n  [scheduler status]")
    status = sched.get_status()
    print(f"  affinity_count    : {status['affinity_count']}")
    print(f"  capacity_threshold: {status['capacity_threshold']}")
    for sid, info in status["replicas"].items():
        print(f"  {sid}  usage={info['token_usage']:.3f}  healthy={info['healthy']}")

    # 清理
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    print(f"目标地址: {ADDRESSES}")
    await test_fetch(ADDRESSES)
    await test_scheduler(ADDRESSES)
    print("\n✓ demo 完成")


if __name__ == "__main__":
    asyncio.run(main())
