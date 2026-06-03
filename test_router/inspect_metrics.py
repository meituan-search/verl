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

"""打印 vLLM /metrics 所有字段（metric 名 + 当前值）。

用法:
    python inspect_metrics.py                   # 默认 33.32.1.89:35547
    python inspect_metrics.py 33.32.1.89:35547
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/opt/meituan/dolphinfs_wangshulin02/Projects/verl")
import aiohttp

ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "33.32.1.89:35547"
URL = ADDRESS if ADDRESS.startswith("http") else f"http://{ADDRESS}"


async def main() -> None:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{URL}/metrics") as resp:
            text = await resp.text()

    print(f"{'METRIC':<70}  VALUE")
    print("-" * 90)
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # 格式: metric_name{labels} value  或  metric_name value
        parts = line.rsplit(None, 1)  # 从右切出最后一个 token（value）
        if len(parts) == 2:
            name, value = parts
            print(f"{name:<70}  {value}")


if __name__ == "__main__":
    asyncio.run(main())
