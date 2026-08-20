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
"""CPU tests for the reprefill_decoupled trainer (P1 + P2 pipelined path)."""

import concurrent.futures
import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1.reprefill_utils import to_nested_jagged
from verl.trainer.ppo.v1.trainer_reprefill_decoupled import PPOTrainerReprefillDecoupled


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _make_trainer(rollout_correction=None) -> PPOTrainerReprefillDecoupled:
    trainer = PPOTrainerReprefillDecoupled.__new__(PPOTrainerReprefillDecoupled)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": rollout_correction},
            "trainer": {"v1": {"reprefill_decoupled": {"enable_prefill_pipeline": False}}},
        }
    )
    trainer.global_steps = 3
    trainer.timing_raw = {}
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    return trainer


def _make_pipelined_trainer() -> PPOTrainerReprefillDecoupled:
    """Trainer with the prefill pipeline armed (P2 path).

    `_pending_prefill` is initialized as an empty dict exactly as
    `on_train_begin` would do when `enable_prefill_pipeline=True`. The
    dispatcher itself is only exercised by the dispatcher unit test; the
    pipelined on_sampled tests inject completed futures directly.
    """
    trainer = _make_trainer()
    trainer.config.trainer.v1.reprefill_decoupled.enable_prefill_pipeline = True
    trainer._pending_prefill = {}
    return trainer


class _FakeClient:
    def __init__(self, expected_lens):
        self.expected_lens = expected_lens  # list[(prompt_len, response_len)]

    async def generate(self, request_id, prompt_ids, sampling_params):
        plen, rlen = self.expected_lens.pop(0)
        assert len(prompt_ids) == plen + rlen
        return SimpleNamespace(
            extra_fields={"prompt_logprobs": [[-0.1 * i] for i in range(len(prompt_ids))]}
        )


def _make_batch(keys, partition_id):
    return tq.KVBatchMeta(
        keys=keys, partition_id=partition_id, tags=[None] * len(keys)
    )


def _write_trajectories(keys, partition_id, prompt_lens, response_lens):
    fields = TensorDict(
        {
            "prompts": to_nested_jagged([[100 + j for j in range(pl)] for pl in prompt_lens]),
            "responses": to_nested_jagged([[200 + j for j in range(rl)] for rl in response_lens]),
        },
        batch_size=len(keys),
    )
    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields)


class TestOnSampled:
    def test_writes_new_rollout_log_probs_and_tags(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0" for _ in range(2)]
        _write_trajectories(keys, partition_id, prompt_lens=[2, 3], response_lens=[2, 1])
        trainer = _make_trainer()
        expected = [(2, 2), (3, 1)]
        trainer.get_llm_client = lambda: _FakeClient(list(expected))
        batch = _make_batch(keys, partition_id)
        metrics = {}

        batch = trainer.on_sampled(batch, metrics)

        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["new_rollout_log_probs"]
        )
        padded = data["new_rollout_log_probs"].to_padded_tensor(padding=0.0)
        assert padded.shape[0] == 2
        assert padded.shape[1] == 2  # max response len
        assert all(tag["resume_version"] == 2 for tag in batch.tags)  # global_steps - 1
        assert metrics["reprefill_decoupled/resume_version"] == 2.0


class TestComputeOldLogProb:
    def test_decoupled_path_copies_new_rollout(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0" for _ in range(2)]
        expected = to_nested_jagged([[-0.5, -0.6], [-0.7]])
        tq.kv_batch_put(
            keys=keys, partition_id=partition_id,
            fields=TensorDict({"new_rollout_log_probs": expected}, batch_size=len(keys)),
        )
        trainer = _make_trainer()  # no rollout_correction → decoupled path
        batch = _make_batch(keys, partition_id)

        trainer._compute_old_log_prob(batch, {})

        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["old_log_probs"]
        )
        padded = data["old_log_probs"].to_padded_tensor(padding=0.0)
        assert torch.allclose(padded, expected.to_padded_tensor(padding=0.0))

    def test_bypass_mode_delegates_to_parent(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0" for _ in range(2)]
        expected = to_nested_jagged([[-1.5, -1.6], [-1.7]])
        tq.kv_batch_put(
            keys=keys, partition_id=partition_id,
            fields=TensorDict({"rollout_log_probs": expected}, batch_size=len(keys)),
        )
        trainer = _make_trainer(rollout_correction={"bypass_mode": True})
        batch = _make_batch(keys, partition_id)

        trainer._compute_old_log_prob(batch, {})

        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["old_log_probs"]
        )
        padded = data["old_log_probs"].to_padded_tensor(padding=0.0)
        assert torch.allclose(padded, expected.to_padded_tensor(padding=0.0))


class TestPrefillDispatcher:
    def test_submit_runs_coroutine_and_returns_future(self):
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PrefillDispatcher

        dispatcher = _PrefillDispatcher()
        dispatcher.start()
        try:
            async def coro():
                return 42

            future = dispatcher.submit(coro())
            assert future.result(timeout=10) == 42
        finally:
            dispatcher.shutdown()


class TestPipelinedOnSampled:
    """P2 path: on_sampled consumes pre-dispatched re-prefill entries when
    their version matches the current step window, and falls back to a
    synchronous re-issue otherwise (stale version or engine-version mismatch).
    """

    def test_valid_pending_entry_is_consumed(self, tq_init, partition_id):
        # Contract 2: valid entry (version == global_steps, engine
        # global_steps == global_steps - 1) → consumed, NO synchronous
        # re-issue, TQ write with correct shape, dict cleared.
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 2
        uid = uuid.uuid4().hex
        key = f"{uid}_0_0"
        _write_trajectories([key], partition_id, [prompt_len], [response_len])

        trainer = _make_pipelined_trainer()
        # global_steps == 3 → resume_version == 2 → engine global_steps must
        # equal 2 to be accepted.
        engine_global_steps = trainer.global_steps - 1
        fut = concurrent.futures.Future()
        fut.set_result([
            SimpleNamespace(extra_fields={
                "prompt_logprobs": [[-1.0 * i] for i in range(prompt_len + response_len)],
                "global_steps": engine_global_steps,
            })
        ])
        trainer._pending_prefill[uid] = _PendingPrefill(
            version=trainer.global_steps, future=fut
        )

        def _fail(*args, **kwargs):
            raise AssertionError("should not re-issue synchronous prefill")

        trainer._reprefill_all = _fail

        batch = _make_batch([key], partition_id)
        metrics = {}

        batch = trainer.on_sampled(batch, metrics)

        data = tq.kv_batch_get(
            keys=[key], partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        padded = data["new_rollout_log_probs"].to_padded_tensor(padding=0.0)
        assert padded.shape == (1, response_len)
        assert trainer._pending_prefill == {}
        assert all(tag["resume_version"] == trainer.global_steps - 1 for tag in batch.tags)
        assert metrics["reprefill_decoupled/resume_version"] == float(trainer.global_steps - 1)

    def test_stale_version_entry_triggers_synchronous_reissue(self, tq_init, partition_id):
        # Contract 3: stale-version entry (version == global_steps - 1) is
        # treated as invalid and re-issued synchronously via _reprefill_all.
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 2
        uid = uuid.uuid4().hex
        key = f"{uid}_0_0"
        _write_trajectories([key], partition_id, [prompt_len], [response_len])

        trainer = _make_pipelined_trainer()
        stale_fut = concurrent.futures.Future()
        stale_fut.set_result([SimpleNamespace(extra_fields={"prompt_logprobs": []})])
        trainer._pending_prefill[uid] = _PendingPrefill(
            version=trainer.global_steps - 1, future=stale_fut
        )

        trainer._reprefill_all = lambda prompt_ids_list: [
            SimpleNamespace(extra_fields={
                "prompt_logprobs": [[-0.5 * i] for i in range(prompt_len + response_len)],
            })
        ]

        batch = _make_batch([key], partition_id)
        metrics = {}

        batch = trainer.on_sampled(batch, metrics)

        data = tq.kv_batch_get(
            keys=[key], partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        padded = data["new_rollout_log_probs"].to_padded_tensor(padding=0.0)
        assert padded.shape == (1, response_len)
        assert all(tag["resume_version"] == trainer.global_steps - 1 for tag in batch.tags)
