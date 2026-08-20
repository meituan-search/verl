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
        # P2 keys `_pending_prefill` by trajectory key (matches what
        # `_on_new_finished` resolves uids to).
        trainer._pending_prefill[key] = _PendingPrefill(
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
        # Keyed by trajectory key, matching `_on_new_finished`'s resolution.
        trainer._pending_prefill[key] = _PendingPrefill(
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


def _make_pipelined_trainer_with_dispatcher(expected_lens):
    """Trainer with a real dispatcher and fake client for end-to-end C1 tests.

    Wires up `_pending_prefill`, `_prefill_dispatcher`, `replay_buffer`
    (a real `ReprefillReplayBuffer` so `_on_new_finished` can read its
    `partitions` and `prompt_global_steps` snapshots), and `get_llm_client`
    returning a `_FakeClient` with the given expected lens.
    """
    from verl.trainer.ppo.v1.replay_buffer import ReprefillReplayBuffer
    from verl.trainer.ppo.v1.trainer_reprefill_decoupled import (
        _PendingPrefill,
        _PrefillDispatcher,
    )

    trainer = _make_trainer()
    trainer.config.trainer.v1.reprefill_decoupled.enable_prefill_pipeline = True
    trainer.config.data = OmegaConf.create({"train_batch_size": 8})
    trainer._pending_prefill = {}
    trainer._prefill_dispatcher = _PrefillDispatcher()
    trainer._prefill_dispatcher.start()
    trainer.replay_buffer = ReprefillReplayBuffer(
        trainer_mode="reprefill_decoupled",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.05,
        refill_fn=None,
    )
    trainer.get_llm_client = lambda: _FakeClient(list(expected_lens))
    return trainer


def _seed_finished_prompt(partition_id, uid, sessions=1, global_steps=0,
                          prompt_len=2, response_len=2):
    """Write trajectory data + flip the prompt tag to finished, mirroring
    the agent_loop_tq contract (trajectories written BEFORE the prompt
    status flips to finished). Returns the list of trajectory keys."""
    keys = [f"{uid}_{s}_0" for s in range(sessions)]
    _write_trajectories(
        keys, partition_id,
        prompt_lens=[prompt_len] * sessions,
        response_lens=[response_len] * sessions,
    )
    tq.kv_put(
        key=uid, partition_id=partition_id,
        tag={"is_prompt": True, "status": "finished", "global_steps": global_steps},
    )
    return keys


def _clear_train_partition():
    """Best-effort cleanup of the shared 'train' partition used by
    TestOnNewFinished (which must use partition_id='train' because
    `_on_new_finished` guards on `partition_id != 'train'`)."""
    items = tq.kv_list() or {}
    train_keys = list(items.get("train", {}).keys())
    if train_keys:
        tq.kv_clear(keys=train_keys, partition_id="train")


class TestOnNewFinished:
    """C1: `_on_new_finished` resolves prompt uids to trajectory keys and
    pre-dispatches one re-prefill per trajectory key, keying `_pending_prefill`
    by trajectory key so the pipelined on_sampled lookup is direct.

    These tests use partition_id='train' (matching the production guard
    `if partition_id != 'train': return`) and clean up the shared 'train'
    partition via try/finally so they don't contaminate each other.
    """

    def test_resolves_uid_to_trajectory_key_and_predispatches(self, tq_init):
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 2
        uid = uuid.uuid4().hex
        try:
            keys = _seed_finished_prompt(
                "train", uid, sessions=1, global_steps=0,
                prompt_len=prompt_len, response_len=response_len,
            )
            key = keys[0]

            # Populate the replay buffer snapshot exactly as the poll loop
            # would before the callback fires. The callback reads
            # partitions + prompt_global_steps.
            trainer = _make_pipelined_trainer_with_dispatcher(
                expected_lens=[(prompt_len, response_len)],
            )
            trainer.replay_buffer._sync_metadata_from_transfer_queue()

            # Callback receives the uid (what ReprefillReplayBuffer delivers).
            trainer._on_new_finished("train", {uid})

            # _pending_prefill is keyed by the trajectory key, not the uid.
            assert key in trainer._pending_prefill
            assert uid not in trainer._pending_prefill
            entry = trainer._pending_prefill[key]
            assert isinstance(entry, _PendingPrefill)
            # Version == trainer.global_steps when the re-prefill was issued.
            assert entry.version == trainer.global_steps
            # Future resolves to a 1-element list whose result carries prompt_logprobs.
            result = entry.future.result(timeout=10)
            assert len(result) == 1
            assert "prompt_logprobs" in result[0].extra_fields

            trainer._prefill_dispatcher.shutdown()
        finally:
            _clear_train_partition()

    def test_multi_session_uid_predispatches_one_per_trajectory_key(self, tq_init):
        # A uid with multiple sessions → multiple trajectory keys, each gets
        # its own pre-dispatched entry. Verifies the resolution loop covers
        # all trajectories of a finished uid (terminal-group completeness).
        prompt_len, response_len = 2, 1
        sessions = 3
        uid = uuid.uuid4().hex
        try:
            keys = _seed_finished_prompt(
                "train", uid, sessions=sessions, global_steps=0,
                prompt_len=prompt_len, response_len=response_len,
            )

            trainer = _make_pipelined_trainer_with_dispatcher(
                expected_lens=[(prompt_len, response_len)] * sessions,
            )
            trainer.replay_buffer._sync_metadata_from_transfer_queue()

            trainer._on_new_finished("train", {uid})

            # Every trajectory key has an entry; the uid itself does not.
            assert set(trainer._pending_prefill.keys()) == set(keys)
            for key in keys:
                entry = trainer._pending_prefill[key]
                assert entry.version == trainer.global_steps
                # Each future resolves independently to a 1-element list.
                result = entry.future.result(timeout=10)
                assert len(result) == 1

            trainer._prefill_dispatcher.shutdown()
        finally:
            _clear_train_partition()

    def test_skips_non_train_partition(self, tq_init):
        # Callback is a no-op for validation partitions (partition_id != "train").
        trainer = _make_pipelined_trainer_with_dispatcher(expected_lens=[])
        try:
            trainer._on_new_finished("val", {uuid.uuid4().hex})
            assert trainer._pending_prefill == {}
        finally:
            trainer._prefill_dispatcher.shutdown()

    def test_bounds_to_train_batch_size_uids(self, tq_init):
        # I3: pre-dispatch is bounded to `train_batch_size` newly-finished
        # uids per poll iteration, oldest-first (smallest prompt_global_steps).
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 1
        # 4 finished uids with distinct global_steps; batch size capped at 2.
        uids = [uuid.uuid4().hex for _ in range(4)]
        steps_by_uid = {uids[0]: 0, uids[1]: 1, uids[2]: 2, uids[3]: 3}
        try:
            for uid in uids:
                _seed_finished_prompt(
                    "train", uid, sessions=1, global_steps=steps_by_uid[uid],
                    prompt_len=prompt_len, response_len=response_len,
                )

            trainer = _make_pipelined_trainer_with_dispatcher(
                expected_lens=[(prompt_len, response_len)] * 2,
            )
            # Override train_batch_size to 2 to force bounding without
            # needing 8 fake results.
            trainer.config.data.train_batch_size = 2
            trainer.replay_buffer._sync_metadata_from_transfer_queue()

            trainer._on_new_finished("train", set(uids))

            # Only the 2 oldest uids (steps 0, 1) were pre-dispatched; their
            # single trajectory keys are present.
            expected_keys = {f"{uids[0]}_0_0", f"{uids[1]}_0_0"}
            assert set(trainer._pending_prefill.keys()) == expected_keys
            for key in expected_keys:
                entry = trainer._pending_prefill[key]
                assert isinstance(entry, _PendingPrefill)
                assert entry.version == trainer.global_steps

            trainer._prefill_dispatcher.shutdown()
        finally:
            _clear_train_partition()


class TestPrefillCancellation:
    """I1: unconsumed pre-dispatched futures are cancelled on on_sampled."""

    def test_unconsumed_future_is_cancelled(self, tq_init, partition_id):
        # An entry whose key is NOT in the sampled batch (unselected key)
        # must have its future cancelled so its coroutine doesn't linger.
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 2
        sampled_uid = uuid.uuid4().hex
        sampled_key = f"{sampled_uid}_0_0"
        _write_trajectories(
            [sampled_key], partition_id, [prompt_len], [response_len],
        )

        trainer = _make_pipelined_trainer()
        # A pending entry for a DIFFERENT key that won't be in the batch.
        # Use a real dispatcher so we get a real `run_coroutine_threadsafe`
        # future whose cancel() is meaningful.
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PrefillDispatcher
        trainer._prefill_dispatcher = _PrefillDispatcher()
        trainer._prefill_dispatcher.start()
        trainer.get_llm_client = lambda: _FakeClient([(prompt_len, response_len)])

        orphan_uid = uuid.uuid4().hex
        orphan_key = f"{orphan_uid}_0_0"

        async def _hang_forever():
            # A coroutine that awaits indefinitely so it stays cancellable.
            import asyncio as _a
            await _a.sleep(3600)
            return [SimpleNamespace(extra_fields={"prompt_logprobs": []})]

        orphan_future = trainer._prefill_dispatcher.submit(_hang_forever())
        trainer._pending_prefill[orphan_key] = _PendingPrefill(
            version=trainer.global_steps, future=orphan_future,
        )
        # Also add a consumed entry for the sampled key so the re-issue path
        # is NOT triggered (valid entry → consumed).
        consumed_fut = concurrent.futures.Future()
        consumed_fut.set_result([SimpleNamespace(extra_fields={
            "prompt_logprobs": [[-1.0 * i] for i in range(prompt_len + response_len)],
            "global_steps": trainer.global_steps - 1,
        })])
        trainer._pending_prefill[sampled_key] = _PendingPrefill(
            version=trainer.global_steps, future=consumed_fut,
        )
        trainer._reprefill_all = lambda prompt_ids_list: [
            SimpleNamespace(extra_fields={
                "prompt_logprobs": [[-0.5 * i] for i in range(prompt_len + response_len)],
            })
        ]

        batch = _make_batch([sampled_key], partition_id)
        metrics = {}
        try:
            batch = trainer.on_sampled(batch, metrics)
        finally:
            trainer._prefill_dispatcher.shutdown()

        # The orphan future was cancelled (cancel() returns True for a
        # pending run_coroutine_threadsafe future whose coroutine hasn't
        # completed; .cancelled() reflects the wrapper state).
        assert orphan_future.cancelled(), (
            "orphan prefill future must be cancelled by on_sampled"
        )
        # Metric was emitted.
        assert metrics["reprefill_decoupled/prefill_cancelled"] == 1.0
        # Dict cleared.
        assert trainer._pending_prefill == {}

    def test_already_done_future_not_cancelled(self, tq_init, partition_id):
        # A consumed entry whose future already resolved is skipped by
        # cancellation (it's done); prefill_cancelled == 0.
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill

        prompt_len, response_len = 2, 2
        sampled_uid = uuid.uuid4().hex
        sampled_key = f"{sampled_uid}_0_0"
        _write_trajectories(
            [sampled_key], partition_id, [prompt_len], [response_len],
        )

        trainer = _make_pipelined_trainer()
        fut = concurrent.futures.Future()
        fut.set_result([SimpleNamespace(extra_fields={
            "prompt_logprobs": [[-1.0 * i] for i in range(prompt_len + response_len)],
            "global_steps": trainer.global_steps - 1,
        })])
        trainer._pending_prefill[sampled_key] = _PendingPrefill(
            version=trainer.global_steps, future=fut,
        )
        trainer._reprefill_all = lambda prompt_ids_list: [
            SimpleNamespace(extra_fields={
                "prompt_logprobs": [[-0.5 * i] for i in range(prompt_len + response_len)],
            })
        ]

        batch = _make_batch([sampled_key], partition_id)
        metrics = {}
        batch = trainer.on_sampled(batch, metrics)

        # The consumed future was already done → not cancelled.
        assert not fut.cancelled()
        assert metrics["reprefill_decoupled/prefill_cancelled"] == 0.0


class TestOnTrainEnd:
    """I2: on_train_end shuts down the dispatcher if it was started."""

    def test_shutdown_dispatcher(self):
        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PrefillDispatcher

        trainer = _make_trainer()
        trainer.config.trainer.v1.reprefill_decoupled.enable_prefill_pipeline = True
        trainer._pending_prefill = {}
        trainer._prefill_dispatcher = _PrefillDispatcher()
        trainer._prefill_dispatcher.start()
        # Sanity: loop is running.
        assert trainer._prefill_dispatcher._loop is not None

        trainer.on_train_end()

        # Loop and thread are torn down.
        assert trainer._prefill_dispatcher._loop is None
        assert trainer._prefill_dispatcher._thread is None

    def test_noop_when_pipeline_disabled(self):
        # P1 path: `_prefill_dispatcher` is never set; on_train_end is a no-op.
        trainer = _make_trainer()
        # No `_prefill_dispatcher` attribute — must not raise.
        trainer.on_train_end()
