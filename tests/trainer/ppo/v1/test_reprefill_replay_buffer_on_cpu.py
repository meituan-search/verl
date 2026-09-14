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
"""CPU tests for ReprefillReplayBuffer's on_new_finished hook.

The hook fires when keys transition to ``finished`` during a metadata sync inside
``sample``'s poll loop, so the reprefill_decoupled trainer can pre-dispatch
re-prefill requests while the remaining generation finishes. Producer helpers are
copied verbatim from test_replay_buffer_on_cpu.py (they are file-local there).
"""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from tensordict import NonTensorData
from transfer_queue import KVBatchMeta

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.replay_buffer import ReprefillReplayBuffer

# Small poll interval so the blocking consumer reacts to producer writes quickly.
POLL_INTERVAL = 0.05


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _make_rb(poll_interval: float = POLL_INTERVAL) -> ReprefillReplayBuffer:
    """Construct a ReprefillReplayBuffer with the async constructor contract."""
    return ReprefillReplayBuffer(
        trainer_mode="reprefill_decoupled",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=poll_interval,
        refill_fn=None,
    )


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


def _set_prompt_status(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Transition an existing prompt to a new status (e.g. running -> finished).

    Mirrors the rollout side flipping a GRPO group's status once it terminates.
    The prompt tag is updated in place; its trajectory values are untouched.
    """
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
    )


@dataclass
class PromptSpec:
    """A prompt group and the trajectories that precede its status update."""

    uid: str
    status: str
    sessions: int = 1
    global_steps: int = 0
    rewards: list[float] | None = None
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Write complete trajectory groups before publishing their prompt status."""

    def __init__(self, partition_id: str, specs: list[PromptSpec]):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.specs = specs
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for spec in self.specs:
                for session_id in range(spec.sessions):
                    key = _trajectory_key(spec.uid, session_id)
                    fields = {"input_ids": torch.tensor([1, 2, 3])}
                    tag = {"is_prompt": False, "seq_len": 3, "global_steps": spec.global_steps}
                    if spec.rewards is not None:
                        fields["extra_fields"] = {"reward_extra_info": {"acc": float(spec.rewards[session_id])}}
                    tq.kv_put(
                        key=key,
                        partition_id=self.partition_id,
                        fields=fields,
                        tag=tag,
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status, "global_steps": spec.global_steps},
                )
        except Exception as e:  # surfaced to the test via join_and_check()
            self.error = e

    def join_and_check(self, timeout: float = 10.0) -> None:
        self.join(timeout)
        assert not self.is_alive(), "RolloutProducer thread did not finish in time"
        if self.error is not None:
            raise self.error


class SampleConsumer(threading.Thread):
    """Runs the blocking ``ReplayBuffer.sample`` in a background thread so the test
    can assert that it stays blocked until the producer supplies enough data."""

    def __init__(self, rb: ReprefillReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.global_steps = global_steps
        self.result: KVBatchMeta | None = None
        self.metrics: dict | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result, self.metrics = self.rb.sample(
                global_steps=self.global_steps,
                partition_id=self.partition_id,
                batch_size=self.batch_size,
            )
        except Exception as e:
            self.error = e

    def result_or_raise(self, timeout: float = 10.0) -> KVBatchMeta:
        self.join(timeout)
        assert not self.is_alive(), "SampleConsumer thread did not finish in time"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _produce(partition_id: str, specs: list[PromptSpec]) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs)
    producer.start()
    return producer


def _clear_partition(partition_id: str) -> None:
    """Best-effort cleanup of every key written into a partition."""
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


# --------------------------------------------------------------------------- #
# on_new_finished callback: fires for keys that newly transition to finished.
# --------------------------------------------------------------------------- #


def test_callback_fires_for_newly_finished_keys(tq_init, partition_id):
    """The callback receives every prompt uid that is finished when sample's
    first metadata sync runs."""
    rb = _make_rb()
    seen: list[tuple[str, set]] = []
    rb.set_on_new_finished_callback(lambda pid, keys: seen.append((pid, keys)))

    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(2)]
    _produce(partition_id, specs).join_and_check()

    try:
        rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        all_seen = {k for _, keys in seen for k in keys}
        assert all_seen == {spec.uid for spec in specs}
    finally:
        _clear_partition(partition_id)


def test_callback_does_not_refire_for_old_keys(tq_init, partition_id):
    """A key that is already finished on one sync must not refire on the next sync
    inside the same ``sample()`` poll loop.

    The first spec is produced before sampling starts, so the first metadata sync
    fires the callback for it. ``sample`` then blocks (batch_size=2, only 1 ready)
    and polls again; the still-finished first uid must NOT refire. A second spec
    is produced so the poll loop advances and the new uid fires exactly once.
    """
    rb = _make_rb()
    seen: list[set] = []
    rb.set_on_new_finished_callback(lambda pid, keys: seen.append(keys))

    first = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    _produce(partition_id, [first]).join_and_check()

    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=1)
    try:
        consumer.start()
        # Wait for the first poll to register first.uid; it must stay blocked.
        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "sample returned before the second spec was produced"

        second = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=1)
        _produce(partition_id, [second]).join_and_check()
        consumer.result_or_raise()

        # The first uid fired on the first sync; the second on a later sync.
        # Neither refired on the intermediate syncs where both were already finished.
        flattened = [k for keys in seen for k in keys]
        assert flattened.count(first.uid) == 1
        assert flattened.count(second.uid) == 1
    finally:
        if consumer.is_alive():
            consumer.join(timeout=2)
        _clear_partition(partition_id)


def test_callback_exception_does_not_break_sampling(tq_init, partition_id):
    """A failing callback must be swallowed so sampling still returns a batch."""
    rb = _make_rb()
    rb.set_on_new_finished_callback(lambda pid, keys: (_ for _ in ()).throw(RuntimeError("boom")))

    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(1)]
    _produce(partition_id, specs).join_and_check()

    try:
        batch, _ = rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        assert len(batch.keys) == 1
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# partial_rollout piggyback field propagation through agent_loop_tq.
# --------------------------------------------------------------------------- #


class _PostprocessWorker:
    """Minimal stand-in for AgentLoopWorkerTQ covering only the attributes
    ``_agent_loop_postprocess`` touches (reward/teacher hooks are no-ops)."""

    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    reward_loop_worker_handles = None
    distillation_enabled = False

    def __init__(self):
        self.processor = None


def test_partial_rollout_fields_propagate_to_tq(tq_init):
    """A TokenOutput carrying partial_rollout piggyback fields must land in
    TransferQueue when it flows through ``_agent_loop_postprocess``:
    - ``token_versions`` / ``new_rollout_log_probs`` as per-trajectory
      (nested-jagged) fields, readable with the trainer's access pattern.
    - ``piggyback_marker`` / ``resume_version`` as tag entries.
    - ``prompt_logprobs`` (an intermediate already consumed into
      ``new_rollout_log_probs``) must NOT be propagated as a TQ field.

    NOTE: ``_agent_loop_postprocess`` hardcodes the partition to "train"
    (validate=False), so the test reads and cleans that partition; the uid
    is unique so it cannot collide with other tests.
    """

    async def run():
        uid = _uid()
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[11, 12],
            response_mask=[1, 1],
            metrics=AgentLoopMetrics(generate_sequences=1.5, tool_calls=0.25, compute_score=0.5, num_preempted=2),
            extra_fields={
                "token_versions": torch.tensor([3, 3, 3, 4], dtype=torch.int32),
                "new_rollout_log_probs": [-0.5, -1.5],
                "piggyback_marker": True,
                "resume_version": 4,
                # Intermediate field; consumed into new_rollout_log_probs by the client.
                "prompt_logprobs": [-9.0] * 4,
            },
        )
        worker = _PostprocessWorker()
        await AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
            worker,
            output,
            validate=False,
            uid=uid,
            session_id=0,
            global_steps=4,
        )
        return uid

    uid = asyncio.run(run())
    key = _trajectory_key(uid, session_id=0, index=0)
    try:
        # Mirror the trainer's consumer access pattern (trainer_partial_reprefill).
        meta = tq.kv_batch_get(
            keys=[key],
            partition_id="train",
            select_fields=["token_versions", "new_rollout_log_probs"],
        )
        token_versions = meta["token_versions"][0]
        assert token_versions.tolist() == [3, 3, 3, 4]
        assert token_versions.dtype == torch.int32
        new_rollout = meta["new_rollout_log_probs"][0]
        torch.testing.assert_close(new_rollout, torch.tensor([-0.5, -1.5], dtype=torch.float32))

        # prompt_logprobs must not be promoted to a TQ field.
        full = tq.kv_batch_get(keys=[key], partition_id="train")
        assert "prompt_logprobs" not in full

        tag = tq.kv_list()["train"][key]
        assert tag["piggyback_marker"] is True
        assert tag["resume_version"] == 4
        assert tag["agent_loop_metrics"] == {
            "generate_sequences": 1.5,
            "tool_calls": 0.25,
            "compute_score": 0.5,
            "num_preempted": 2,
        }
    finally:
        tq.kv_clear(keys=[key], partition_id="train")


def test_no_piggyback_fields_no_extra_tq_writes(tq_init):
    """Trajectories without piggyback fields must not grow new TQ fields/tags."""

    async def run():
        uid = _uid()
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[11, 12],
            response_mask=[1, 1],
            metrics=AgentLoopMetrics(),
            extra_fields={},
        )
        worker = _PostprocessWorker()
        await AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
            worker,
            output,
            validate=False,
            uid=uid,
            session_id=0,
            global_steps=0,
        )
        return uid

    uid = asyncio.run(run())
    key = _trajectory_key(uid, session_id=0, index=0)
    try:
        full = tq.kv_batch_get(keys=[key], partition_id="train")
        assert "token_versions" not in full
        assert "new_rollout_log_probs" not in full
        tag = tq.kv_list()["train"][key]
        assert "piggyback_marker" not in tag
        assert "resume_version" not in tag
    finally:
        tq.kv_clear(keys=[key], partition_id="train")


# --------------------------------------------------------------------------- #
# generate_sequences must forward the piggyback gate to the rollout client.
# --------------------------------------------------------------------------- #


class _SamplingParamsWorker:
    """Minimal stand-in for AgentLoopWorkerTQ covering only the attributes
    ``generate_sequences`` touches; ``_run_prompt`` captures its arguments."""

    generate_sequences = AgentLoopWorkerTQ.__ray_actor_class__.generate_sequences

    def __init__(self, rollout_cfg):
        from types import SimpleNamespace

        self.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout_cfg))
        self.background_tasks = set()
        self.captured = []

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        self.captured.append(dict(sampling_params))


class _SingleSampleBatch(dict):
    """Dict-shaped batch stand-in whose ``len`` is the batch size, not the key
    count (mirrors TensorDict semantics for the ``range(len(batch))`` loop)."""

    def __len__(self):
        return 1


def _run_generate_sequences(monkeypatch, rollout_cfg) -> list[dict]:
    async def fake_trajectory_info(*args, **kwargs):
        return [{"validate": False}]

    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.get_trajectory_info", fake_trajectory_info)
    batch = _SingleSampleBatch(
        validate=False,
        agent_name=NonTensorData("single_turn_agent"),
        global_steps=torch.tensor([1]),
        index=torch.tensor([0]),
    )
    worker = _SamplingParamsWorker(rollout_cfg)

    async def run():
        await worker.generate_sequences(batch)
        await asyncio.gather(*worker.background_tasks)

    asyncio.run(run())
    return worker.captured


@pytest.mark.parametrize("gate_value", [True, False])
def test_generate_sequences_forwards_piggyback_gate(monkeypatch, tq_init, gate_value):
    """The trainer's enable_piggyback flag is synced into
    actor_rollout_ref.rollout.enable_piggyback; generate_sequences must forward
    it in sampling_params so the rollout client's resume branch computes
    prefix_prompt_logprobs. Without it, every aborted-and-resumed trajectory
    falls to case 2 with reason "no_resume_version"."""
    from types import SimpleNamespace

    rollout_cfg = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        calculate_log_probs=False,
        enable_piggyback=gate_value,
        emit_token_versions=gate_value,
    )
    captured = _run_generate_sequences(monkeypatch, rollout_cfg)
    assert len(captured) == 1
    assert captured[0]["enable_piggyback"] is gate_value
    assert captured[0]["emit_token_versions"] is gate_value


def test_generate_sequences_piggyback_gate_defaults_off(monkeypatch, tq_init):
    """Configs without the mirror flag (e.g. non-partial_reprefill trainers)
    must default the gate to False — matching AgentLoopWorker's getattr."""
    from types import SimpleNamespace

    rollout_cfg = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        calculate_log_probs=False,
    )
    captured = _run_generate_sequences(monkeypatch, rollout_cfg)
    assert len(captured) == 1
    assert captured[0]["enable_piggyback"] is False
    assert captured[0]["emit_token_versions"] is False


# --------------------------------------------------------------------------- #
# AgentLoopManagerTQ._performance_metrics — agent_loop/* timing aggregation.
# --------------------------------------------------------------------------- #


def _agent_loop_manager_tq():
    """Construct an AgentLoopManagerTQ without running __init__ (Ray actors,
    config and LLM clients are irrelevant to _performance_metrics)."""
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

    return AgentLoopManagerTQ.__new__(AgentLoopManagerTQ)


def test_performance_metrics_aggregates_tags():
    """_performance_metrics must emit the same agent_loop/* keys as the base
    AgentLoopManager, aggregating the per-trajectory agent_loop_metrics tag
    entries of the sampled batch."""
    manager = _agent_loop_manager_tq()
    tags = [
        {
            "prompt_len": 10,
            "response_len": 20,
            "agent_loop_metrics": {
                "generate_sequences": 1.0,
                "tool_calls": 0.0,
                "compute_score": 0.1,
                "num_preempted": 0,
            },
        },
        {
            "prompt_len": 5,
            "response_len": 50,
            # slowest: 3.0 + 1.0 + 0.2 = 4.2
            "agent_loop_metrics": {
                "generate_sequences": 3.0,
                "tool_calls": 1.0,
                "compute_score": 0.2,
                "num_preempted": 2,
            },
        },
        None,  # padding / reissued trajectories without metrics are skipped
    ]
    timing = manager._performance_metrics(tags)
    assert timing["agent_loop/generate_sequences/min"] == 1.0
    assert timing["agent_loop/generate_sequences/max"] == 3.0
    assert timing["agent_loop/generate_sequences/mean"] == 2.0
    assert timing["agent_loop/tool_calls/max"] == 1.0
    assert timing["agent_loop/compute_score/max"] == 0.2
    assert timing["agent_loop/num_preempted/mean"] == 1.0
    assert timing["agent_loop/slowest/generate_sequences"] == 3.0
    assert timing["agent_loop/slowest/tool_calls"] == 1.0
    assert timing["agent_loop/slowest/num_preempted"] == 2
    assert timing["agent_loop/slowest/prompt_length"] == 5
    assert timing["agent_loop/slowest/response_length"] == 50


def test_performance_metrics_empty_tags():
    """No agent_loop_metrics entries (e.g. a batch fully made of padding)
    must emit nothing rather than raise."""
    manager = _agent_loop_manager_tq()
    assert manager._performance_metrics([None, {}, {"prompt_len": 1}]) == {}
