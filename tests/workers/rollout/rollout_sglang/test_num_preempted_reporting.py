# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for SGLang retraction reporting as num_preempted.

SGLang surfaces per-request retraction counts as meta_info["num_retractions"]
(Req.retraction_count -> BatchTokenIDOutput.retraction_counts ->
TokenizerManager._handle_batch_output). The rollout adapter must forward it
as TokenOutput.num_preempted so agent_loop/num_preempted/* metrics can
quantify how often KV-pressure retraction stretches rollout-n groups across
weight updates. Older SGLang builds omit the key — the field must stay None
(reported as -1, not available) rather than crash.
"""

from __future__ import annotations

import asyncio
import importlib.util
from types import SimpleNamespace

import pytest


def _sglang_available() -> bool:
    return importlib.util.find_spec("sglang") is not None


pytestmark = pytest.mark.skipif(not _sglang_available(), reason="sglang not installed")


def _make_server(meta_info: dict):
    """Build an SGLangHttpServer with a fake tokenizer_manager yielding `meta_info`."""
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    server = SGLangHttpServer.__new__(SGLangHttpServer)
    server.node_rank = 0
    server.replica_rank = 0
    server._disagggregation_role = None
    server._pd_decode_peers = []
    server.model_config = SimpleNamespace(lora_rank=0)
    server.config = SimpleNamespace(
        max_model_len=8192,
        prompt_length=1024,
        response_length=2048,
        enable_rollout_routing_replay=False,
        skip_tokenizer_init=False,
        mtp=None,
    )

    class _Gen:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return {"output_ids": [1, 2, 3], "meta_info": meta_info}

    server.tokenizer_manager = SimpleNamespace(generate_request=lambda req, *a: _Gen())
    return server


class TestNumPreemptedReporting:
    def test_num_retractions_forwarded(self):
        # Two retractions on this request -> TokenOutput.num_preempted == 2,
        # accumulated across resume segments by FullyAsyncLLMServerClient.
        server = _make_server({"finish_reason": {"type": "length"}, "num_retractions": 2})
        out = asyncio.run(server.generate(prompt_ids=[10, 11], sampling_params={"temperature": 1.0}, request_id="r0"))
        assert out.token_ids == [1, 2, 3]
        assert out.num_preempted == 2

    def test_zero_retractions_forwarded(self):
        server = _make_server({"finish_reason": {"type": "length"}, "num_retractions": 0})
        out = asyncio.run(server.generate(prompt_ids=[10, 11], sampling_params={"temperature": 1.0}, request_id="r0"))
        assert out.num_preempted == 0

    def test_missing_key_leaves_none(self):
        # Older SGLang builds don't set num_retractions; the sentinel contract
        # is None -> agent loop reports -1 ("not available"), never a crash.
        server = _make_server({"finish_reason": {"type": "length"}})
        out = asyncio.run(server.generate(prompt_ids=[10, 11], sampling_params={"temperature": 1.0}, request_id="r0"))
        assert out.num_preempted is None
