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
"""CPU tests for shared re-prefill helpers."""

import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    reprefill_trajectories,
    slice_response_logprobs,
    to_nested_jagged,
)


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


class TestSliceResponseLogprobs:
    def test_middle_sequence(self):
        # S = prompt_len + response_len = 8; response logprobs at idx 4..6
        prompt_logprobs_ls = [[float(i)] for i in range(8)]
        out = slice_response_logprobs(prompt_logprobs_ls, prompt_len=5, response_len=3)
        assert out == [4.0, 5.0, 6.0]

    def test_prompt_len_one(self):
        prompt_logprobs_ls = [[float(i)] for i in range(3)]
        out = slice_response_logprobs(prompt_logprobs_ls, prompt_len=1, response_len=2)
        assert out == [0.0, 1.0]

    def test_zero_response(self):
        prompt_logprobs_ls = [[1.0], [2.0]]
        assert slice_response_logprobs(prompt_logprobs_ls, prompt_len=2, response_len=0) == []


class TestToNestedJagged:
    def test_roundtrip_lengths(self):
        t = to_nested_jagged([[0.5, 1.5], [2.0], []])
        assert t.layout == torch.jagged
        assert t.shape[0] == 3

    def test_padded_tensor(self):
        t = to_nested_jagged([[0.5, 1.5], [2.0]])
        padded = t.to_padded_tensor(padding=0.0)
        assert padded.shape == (2, 2)
        assert padded[1, 1].item() == 0.0


class _FakeClient:
    """Yields prompt_logprobs of length len(prompt_ids) (one per position)."""

    def __init__(self):
        self.requests = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.requests.append((request_id, prompt_ids, sampling_params))
        result = type("R", (), {})()
        result.extra_fields = {"prompt_logprobs": [[float(i)] for i in range(len(prompt_ids))]}
        return result


class TestReprefillTrajectories:
    def test_issues_requests_with_logprob_sampling(self):
        import asyncio

        client = _FakeClient()
        results = asyncio.run(reprefill_trajectories(client, [[4], [1, 2, 3]], request_prefix="test_prefix"))
        assert len(results) == 2
        assert client.requests[0][0] == "test_prefix_0"
        assert client.requests[0][2] == {"prompt_logprobs": 0, "max_new_tokens": 0}
        assert results[1].extra_fields["prompt_logprobs"] == [[0.0], [1.0], [2.0]]


class TestBuildReprefillInputs:
    def test_concatenates_prompt_and_response(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0", f"{uuid.uuid4().hex}_0_0"]
        fields = TensorDict(
            {
                "prompts": to_nested_jagged([[101, 102, 103], [201]]),
                "responses": to_nested_jagged([[11, 12], [21, 22, 23]]),
            },
            batch_size=len(keys),
        )
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields)

        prompt_ids_list, real_lens, data = build_reprefill_inputs(keys=keys, partition_id=partition_id, pad_id=0)
        assert prompt_ids_list[0] == [101, 102, 103, 11, 12]
        assert prompt_ids_list[1] == [201, 21, 22, 23]
        assert real_lens == [(3, 2), (1, 3)]
