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

import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

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
