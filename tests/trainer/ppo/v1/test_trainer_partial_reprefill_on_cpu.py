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
"""CPU tests for the partial_reprefill trainer (case dispatch)."""

import concurrent.futures
import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1.reprefill_utils import to_nested_jagged
from verl.trainer.ppo.v1.trainer_partial_reprefill import (
    _PendingPrefill,
    PPOTrainerPartialReprefill,
)


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _make_trainer(enable_case_skip=True, enable_piggyback=True, enable_prefill_pipeline=False, num_warmup_batches=0):
    trainer = PPOTrainerPartialReprefill.__new__(PPOTrainerPartialReprefill)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": None},
            "trainer": {
                "v1": {
                    "partial_reprefill": {
                        "enable_prefill_pipeline": enable_prefill_pipeline,
                        "enable_case_skip": enable_case_skip,
                        "enable_piggyback": enable_piggyback,
                        "compare_trainer_old_log_prob": False,
                        "num_warmup_batches": num_warmup_batches,
                    }
                }
            },
            "data": {"train_batch_size": 4},
        }
    )
    trainer.global_steps = 6
    trainer.timing_raw = {}
    trainer._pending_prefill = None
    return trainer


def _make_batch(partition_id, keys, tags=None):
    """Build a KVBatchMeta-like with keys, partition_id, tags."""
    from transfer_queue import KVBatchMeta

    return KVBatchMeta(
        keys=keys,
        partition_id=partition_id,
        tags=tags or [None] * len(keys),
    )


class TestComputeNewRolloutLogProbCase3:
    def test_case3_copies_rollout_log_probs(self, tq_init, partition_id):
        # Trajectory decoded at W_5 (= global_steps - 1 = 5); sampled at step 6
        # with rollout engine still at W_5 → case 3 (fully fresh).
        trainer = _make_trainer()
        key = f"traj-{uuid.uuid4().hex}"
        rollout_lp = [float(i) for i in range(4)]
        fields = TensorDict(
            {
                "rollout_log_probs": to_nested_jagged([rollout_lp]),
                "token_versions": torch.nested.as_nested_tensor(
                    [torch.tensor([5, 5, 5, 5], dtype=torch.int32)], layout=torch.jagged
                ),
            },
            batch_size=1,
        )
        tq.kv_batch_put(keys=[key], partition_id=partition_id, fields=fields)
        batch = _make_batch(partition_id, [key])
        metrics = {}
        # Bypass the client call by stubbing _reprefill_all to assert it's NOT called
        called = {"yes": False}

        def _reprefill_all(_):
            called["yes"] = True
            return []

        trainer._reprefill_all = _reprefill_all
        trainer._compute_new_rollout_log_prob(batch, metrics)
        assert called["yes"] is False, "case 3 must not call reprefill"
        assert metrics["partial_reprefill/case_distribution.case_3"] == 1.0
        # Verify new_rollout_log_probs == rollout_log_probs
        data = tq.kv_batch_get(
            keys=[key],
            partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        result = data["new_rollout_log_probs"][0].tolist()
        assert result == rollout_lp


class TestComputeNewRolloutLogProbCase1:
    def test_case1_skips_reprefill_when_marker_set(self, tq_init, partition_id):
        # Client already wrote new_rollout_log_probs + piggyback marker.
        trainer = _make_trainer()
        key = f"traj-{uuid.uuid4().hex}"
        expected_lp = [-0.1, -0.2, -0.3]
        tq.kv_batch_put(
            keys=[key],
            partition_id=partition_id,
            fields=TensorDict({"new_rollout_log_probs": to_nested_jagged([expected_lp])}, batch_size=1),
        )
        batch = _make_batch(
            partition_id,
            [key],
            tags=[{"piggyback_marker": True, "resume_version": 5}],
        )
        metrics = {}
        called = {"yes": False}
        trainer._reprefill_all = lambda _: called.__setitem__("yes", True) or []
        out = trainer._compute_new_rollout_log_prob(batch, metrics)
        assert called["yes"] is False
        assert metrics["partial_reprefill/case_distribution.case_1"] == 1.0
        assert metrics["partial_reprefill/case_distribution.case_2"] == 0.0
        assert metrics["partial_reprefill/case_distribution.case_3"] == 0.0


class TestComputeNewRolloutLogProbCase2:
    def test_case2_full_reprefill(self, tq_init, partition_id, monkeypatch):
        # Trajectory decoded at W_3 (stale), no piggyback → case 2.
        trainer = _make_trainer()
        key = f"traj-{uuid.uuid4().hex}"
        tq.kv_batch_put(
            keys=[key],
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "prompts": to_nested_jagged([[1, 2]]),
                    "responses": to_nested_jagged([[10, 11, 12]]),
                    "token_versions": torch.nested.as_nested_tensor(
                        [torch.tensor([3, 3, 3], dtype=torch.int32)],
                        layout=torch.jagged,
                    ),
                },
                batch_size=1,
            ),
        )
        batch = _make_batch(partition_id, [key])
        metrics = {}

        # Stub _reprefill_all to return a fake result with prompt_logprobs.
        fake_result = SimpleNamespace(extra_fields={"prompt_logprobs": [[-0.0], [-0.1], [-0.2], [-0.3], [-0.4]]})
        trainer._reprefill_all = lambda _: [fake_result]
        # Stub tokenizer.pad_token_id
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)

        out = trainer._compute_new_rollout_log_prob(batch, metrics)
        assert metrics["partial_reprefill/case_distribution.case_2"] == 1.0
        data = tq.kv_batch_get(
            keys=[key],
            partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        # prompt_len=2, response_len=3 → slice [1:4] → [-0.1, -0.2, -0.3]
        # (approx: to_nested_jagged stores float32, so values round-trip with
        # float32 precision, e.g. -0.1 → -0.10000000149011612)
        assert data["new_rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, -0.3])


class TestOnNewFinishedCaseAware:
    def test_skips_case1_and_case3_only_dispatches_case2(self, tq_init):
        # Uses partition "train" to match the production guard
        # `if partition_id != "train": return`, with try/finally cleanup so
        # the shared partition isn't contaminated.
        # current_parameter_version = global_steps - 1 = 5.
        trainer = _make_trainer()
        submitted = []

        def _submit(coro):
            coro.close()  # never awaited by this stub; close to silence RuntimeWarning
            future = concurrent.futures.Future()
            submitted.append(future)
            return future

        trainer._prefill_dispatcher = SimpleNamespace(submit=_submit)
        trainer._pending_prefill = {}
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer.get_llm_client = lambda: None
        # Uids must not contain underscores: production resolves trajectory
        # keys via `k.split("_")[0] == uid` (traj key = `{uid}_{session}_{index}`).
        # Unique suffix keeps the shared "train" partition clean across runs.
        sfx = uuid.uuid4().hex[:6]
        uid_a, uid_b, uid_c = f"uida{sfx}", f"uidb{sfx}", f"uidc{sfx}"
        key_a = f"{uid_a}_0_0"  # case 2 (stale: token_versions 3 < 5, no piggyback)
        key_b = f"{uid_b}_0_0"  # case 1 (piggyback marker in the partition tag)
        key_c = f"{uid_c}_0_0"  # case 3 (fresh: token_versions == 5)
        trainer.replay_buffer = SimpleNamespace(
            prompt_global_steps={"train": {}},
            partitions={
                "train": {
                    key_a: None,
                    key_b: {"piggyback_marker": True},
                    key_c: None,
                }
            },
        )
        # All three trajectories exist in TQ (as they do in production once
        # finished). token_versions drives decide_case; prompts/responses are
        # only consumed by the dispatched case-2 request.
        tq.kv_batch_put(
            keys=[key_a, key_b, key_c],
            partition_id="train",
            fields=TensorDict(
                {
                    "prompts": to_nested_jagged([[1, 2], [1, 2], [1, 2]]),
                    "responses": to_nested_jagged([[10, 11], [10, 11], [10, 11]]),
                    "token_versions": torch.nested.as_nested_tensor(
                        [
                            torch.tensor([3, 3], dtype=torch.int32),
                            torch.tensor([3, 3], dtype=torch.int32),
                            torch.tensor([5, 5], dtype=torch.int32),
                        ],
                        layout=torch.jagged,
                    ),
                },
                batch_size=3,
            ),
        )

        try:
            trainer._on_new_finished("train", [uid_a, uid_b, uid_c])

            assert list(trainer._pending_prefill.keys()) == [key_a]
            assert len(submitted) == 1
            assert trainer._pending_prefill[key_a].version == trainer.global_steps
        finally:
            tq.kv_clear(partition_id="train", keys=[key_a, key_b, key_c])


class TestComputeNewRolloutLogProbPipelined:
    def _write_trajs(self, partition_id, stale_key, fresh_key=None):
        # stale: token_versions 3 (case 2); fresh (if given): token_versions 5
        # (case 3). All keys get prompts/responses so build_reprefill_inputs
        # works for the case-2 key; rollout_log_probs serves the case-3 copy.
        trajs = [(stale_key, [3, 3, 3])]
        if fresh_key is not None:
            trajs.append((fresh_key, [5, 5, 5]))
        n = len(trajs)
        tq.kv_batch_put(
            keys=[k for k, _ in trajs],
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "prompts": to_nested_jagged([[1, 2]] * n),
                    "responses": to_nested_jagged([[10, 11, 12]] * n),
                    "token_versions": torch.nested.as_nested_tensor(
                        [torch.tensor(tv, dtype=torch.int32) for _, tv in trajs],
                        layout=torch.jagged,
                    ),
                    "rollout_log_probs": to_nested_jagged([[0.1, 0.2, 0.3]] * n),
                },
                batch_size=n,
            ),
        )

    def test_case2_consumes_prefill_future_case3_fast_path(self, tq_init, partition_id):
        trainer = _make_trainer()
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer._pending_prefill = {}
        stale_key = f"uid_s_{uuid.uuid4().hex[:6]}_0"
        fresh_key = f"uid_f_{uuid.uuid4().hex[:6]}_0"
        self._write_trajs(partition_id, stale_key, fresh_key)

        # Pre-dispatched future for the stale key: done + version-aligned
        # (resume_version = global_steps - 1 = 5).
        future = concurrent.futures.Future()
        future.set_result(
            [
                SimpleNamespace(
                    extra_fields={
                        "prompt_logprobs": [[-0.0], [-0.1], [-0.2], [-0.3], [-0.4]],
                        "global_steps": 5,
                    }
                )
            ]
        )
        trainer._pending_prefill[stale_key] = _PendingPrefill(version=trainer.global_steps, future=future)

        reprefill_called = []

        def _reprefill_all(_):
            reprefill_called.append(True)
            return []

        trainer._reprefill_all = _reprefill_all
        batch = _make_batch(partition_id, [stale_key, fresh_key])
        metrics = {}

        trainer._compute_new_rollout_log_prob_pipelined(batch, metrics)

        assert reprefill_called == [], "aligned prefill future must be consumed, not re-issued"
        assert metrics["partial_reprefill/case_distribution.case_2"] == 1.0
        assert metrics["partial_reprefill/case_distribution.case_3"] == 1.0
        assert metrics["partial_reprefill/prefill_consumed"] == 1.0
        data = tq.kv_batch_get(
            keys=[stale_key, fresh_key],
            partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        # prompt_len=2, response_len=3 → slice [1:4]
        assert data["new_rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, -0.3])
        assert data["new_rollout_log_probs"][1].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert trainer._pending_prefill == {}

    def test_stale_version_future_triggers_reissue(self, tq_init, partition_id):
        trainer = _make_trainer()
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer._pending_prefill = {}
        stale_key = f"uid_s_{uuid.uuid4().hex[:6]}_0"
        self._write_trajs(partition_id, stale_key)

        # Future resolved at the wrong engine version → must not be consumed.
        future = concurrent.futures.Future()
        future.set_result(
            [SimpleNamespace(extra_fields={"prompt_logprobs": [[-9.0]], "global_steps": 4})]
        )
        trainer._pending_prefill[stale_key] = _PendingPrefill(version=trainer.global_steps, future=future)

        fake_result = SimpleNamespace(extra_fields={"prompt_logprobs": [[-0.0], [-0.1], [-0.2], [-0.3], [-0.4]]})
        trainer._reprefill_all = lambda _: [fake_result]
        batch = _make_batch(partition_id, [stale_key])
        metrics = {}

        trainer._compute_new_rollout_log_prob_pipelined(batch, metrics)

        assert metrics["partial_reprefill/prefill_consumed"] == 0.0
        data = tq.kv_batch_get(
            keys=[stale_key],
            partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        assert data["new_rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, -0.3])
        assert trainer._pending_prefill == {}


class TestEnablePrefillPipelineFlag:
    """Verify `enable_prefill_pipeline` gates P2 dispatcher creation.

    Flag off (default): `_prefill_dispatcher` stays None, `_pending_prefill`
    stays None — `on_sampled` takes the non-pipelined branch. Flag on: dispatcher
    is started, `_pending_prefill` becomes `{}`, and the replay-buffer callback
    is registered. This is the carry-forward test from Task 10's review.
    """

    def test_flag_off_does_not_create_dispatcher(self):
        trainer = _make_trainer(enable_prefill_pipeline=False)
        added = []
        trainer._add_batch_to_generate = lambda: added.append(True)
        trainer.replay_buffer = SimpleNamespace()
        trainer.on_train_begin()
        assert getattr(trainer, "_prefill_dispatcher", None) is None
        assert getattr(trainer, "_pending_prefill", None) is None

    def test_flag_on_creates_dispatcher_and_pending_dict(self):
        trainer = _make_trainer(enable_prefill_pipeline=True, num_warmup_batches=1)
        added = []
        trainer._add_batch_to_generate = lambda: added.append(True)
        callback_registered = []
        rb = SimpleNamespace(
            set_on_new_finished_callback=lambda cb: callback_registered.append(cb)
        )
        trainer.replay_buffer = rb
        trainer.on_train_begin()
        assert len(added) == 1, "warmup batches must be added"
        assert trainer._prefill_dispatcher is not None
        assert trainer._pending_prefill == {}
        assert callback_registered == [trainer._on_new_finished]
        trainer.on_train_end()
        assert trainer._prefill_dispatcher._loop is None, "dispatcher loop must be closed on_train_end"

    def test_on_sampled_non_pipelined_when_flag_off(self):
        trainer = _make_trainer(enable_prefill_pipeline=False)
        called = []
        trainer._compute_new_rollout_log_prob = lambda batch, metrics: (called.append("non_pipelined") or batch)
        trainer._compute_new_rollout_log_prob_pipelined = lambda batch, metrics: (called.append("pipelined") or batch)
        trainer.timing_raw = {}
        batch = _make_batch("p", ["k1"])
        trainer.on_sampled(batch, {})
        assert called == ["non_pipelined"]

    def test_on_sampled_pipelined_when_flag_on(self):
        trainer = _make_trainer(enable_prefill_pipeline=True)
        # `on_train_begin` sets `_pending_prefill = {}` when flag is on;
        # simulate that state without actually starting the dispatcher thread.
        trainer._pending_prefill = {}
        called = []
        trainer._compute_new_rollout_log_prob = lambda batch, metrics: (called.append("non_pipelined") or batch)
        trainer._compute_new_rollout_log_prob_pipelined = lambda batch, metrics: (called.append("pipelined") or batch)
        trainer.timing_raw = {}
        batch = _make_batch("p", ["k1"])
        trainer.on_sampled(batch, {})
        assert called == ["pipelined"]

