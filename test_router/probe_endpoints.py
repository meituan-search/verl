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

"""探测 vLLM 服务支持哪些端点，重点查找能暴露 KV-cache token 数的接口。

用法:
    python probe_endpoints.py                          # 默认 33.32.1.89:35547，输出到终端
    python probe_endpoints.py 33.32.1.89:38909         # 指定地址
    python probe_endpoints.py 33.32.1.89:38909 out.txt # 同时写到文件
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/opt/meituan/dolphinfs_wangshulin02/Projects/verl")
import aiohttp

ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "33.32.1.89:35547"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else None
URL = ADDRESS if ADDRESS.startswith("http") else f"http://{ADDRESS}"


async def main() -> None:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{URL}/metrics") as resp:
            text = await resp.text()

    lines = []
    lines.append(f"{'METRIC':<80}  VALUE")
    lines.append("-" * 95)
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            name, value = parts
            lines.append(f"{name:<80}  {value}")

    output = "\n".join(lines)
    print(output)

    if OUT_FILE:
        with open(OUT_FILE, "w") as f:
            f.write(output + "\n")
        print(f"\n→ 已写入 {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
