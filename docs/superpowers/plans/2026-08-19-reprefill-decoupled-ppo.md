# Re-prefill Decoupled PPO (`reprefill_decoupled`) Implementation Plan

Last updated: 08/20/2026

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `reprefill_decoupled` V1 PPO trainer (colocate async) where π_b (`old_log_probs`) comes from re-prefilling consumed trajectories on the rollout engine at its current weight, replacing the trainer-side forward pass — with P2 pipelining that pre-dispatches re-prefills while the replay buffer waits for remaining generation.

**Architecture:** Extract the re-prefill machinery validated in `trainer_staleness_sweep.py` into a shared module `reprefill_utils.py`; add `PPOTrainerReprefillDecoupled(PPOTrainerColocateAsync)` overriding `on_sampled` (re-prefill → write `new_rollout_log_probs` + `resume_version` tags to TransferQueue) and `_compute_old_log_prob` (copy `new_rollout_log_probs` as `old_log_probs`, skip trainer forward). P2 adds a `ReprefillReplayBuffer(ReplayBufferAsync)` hook that fires on newly-finished keys during the `sample()` poll loop, plus a background event-loop dispatcher and version guard in the trainer.

**Tech Stack:** Python 3.12, PyTorch (nested jagged tensors), TransferQueue (`transfer_queue` / `tq`), asyncio, Hydra/OmegaConf config, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-reprefill-decoupled-ppo-design.md`

## Global Constraints

- Python environment: `uv` venv at `.venv` (`source .venv/bin/activate`); run tests with `pytest`.
- All commits include trailer `Co-authored-by: Claude` (project CLAUDE.md rule).
- Run `pre-commit` hooks pass before each commit (installed via `uv pip install pre-commit; pre-commit install`).
- The nested-jagged storage format (`torch.nested.as_nested_tensor(..., layout=torch.jagged)`) is mandatory for any field written to TransferQueue that later goes through `to_padded_tensor()` — NonTensorStack breaks downstream `torch.exp` (see `trainer_staleness_sweep.py:163-171` comment).
- Engine version invariant used throughout: during step k's window (after `on_step_end` of step k-1 called `update_weights(self.global_steps)`, before `on_sample_end` of step k), the rollout engine holds weight version `self.global_steps - 1` (constant across the window). `resume_version = self.global_steps - 1` is the π_b version tag.
- Config namespace: `trainer.v1.reprefill_decoupled` in `verl/trainer/config/ppo_trainer.yaml`. Enable via `trainer.v1.trainer_mode=reprefill_decoupled`.
- Test fixtures that need a real TransferQueue follow the pattern in `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py`: module-scoped `tq.init()` fixture, per-test unique `partition_id` (uuid hex).
- `staleness_sweep` trainer behavior must not change (pure refactor in Task 2).

---

### Task 1: Shared re-prefill helpers module

**Files:**
- Create: `verl/trainer/ppo/v1/reprefill_utils.py`
- Test: `tests/trainer/ppo/v1/test_reprefill_utils_on_cpu.py`

**Interfaces:**
- Consumes: `transfer_queue` (`tq.kv_batch_get`), torch.
- Produces (exact signatures, used by Tasks 2, 3, 6):
  - `slice_response_logprobs(prompt_logprobs_ls: list, prompt_len: int, response_len: int) -> list[float]`
  - `to_nested_jagged(nested_list: list[list[float]]) -> torch.Tensor` (jagged layout)
  - `build_reprefill_inputs(keys: list[str], partition_id: str, pad_id: int) -> tuple[list[list[int]], list[tuple[int, int]], Any]` (returns `(prompt_ids_list, real_lens, data)` where `data` is the TQ fetch result supporting `.select(...)`)
  - `async reprefill_trajectories(client, prompt_ids_list: list[list[int]], request_prefix: str = "reprefill") -> list` (results with `.extra_fields["prompt_logprobs"]`)
  - `compute_and_emit_staleness_metrics(batch, metrics: dict, global_steps: int) -> None` (emits `staleness_sweep/…`-style keys under `offpolicy/` prefix)

- [ ] **Step 1: Write failing tests for the pure functions**

Create `tests/trainer/ppo/v1/test_reprefill_utils_on_cpu.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py)
"""CPU tests for shared re-prefill helpers."""

import uuid

import pytest
import torch
import transfer_queue as tq

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
        results = asyncio.run(
            reprefill_trajectories(client, [[1, 2], [3]], request_prefix="test_prefix")
        )
        assert len(results) == 2
        assert client.requests[0][0] == "test_prefix_0"
        assert client.requests[0][2] == {"prompt_logprobs": 0, "max_new_tokens": 0}
        assert results[1].extra_fields["prompt_logprobs"] == [[0.0], [1.0], [2.0]]


class TestBuildReprefillInputs:
    def test_concatenates_prompt_and_response(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0", f"{uuid.uuid4().hex}_0_0"]
        fields = {
            "prompts": to_nested_jagged([[101, 102, 103], [201]]),
            "responses": to_nested_jagged([[11, 12], [21, 22, 23]]),
        }
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields)

        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=keys, partition_id=partition_id, pad_id=0
        )
        assert prompt_ids_list[0] == [101, 102, 103, 11, 12]
        assert prompt_ids_list[1] == [201, 21, 22, 23]
        assert real_lens == [(3, 2), (1, 3)]
```

Note: if `tq.kv_batch_put` rejects plain dicts of nested tensors, consult the `_produce`/`PromptSpec` helpers in `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py` for the canonical trajectory-write format and adapt the write (the assertion contract stays the same).

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_reprefill_utils_on_cpu.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl.trainer.ppo.v1.reprefill_utils'`

- [ ] **Step 3: Implement `reprefill_utils.py`**

Create `verl/trainer/ppo/v1/reprefill_utils.py` (code moved verbatim from `trainer_staleness_sweep.py:144-228` where possible):

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from verl/trainer/ppo/v1/trainer_staleness_sweep.py)
"""Shared re-prefill helpers: compute response-token log probs of existing
trajectories under the rollout engine's current weight
(max_new_tokens=0, prompt_logprobs=0).

Used by staleness_sweep (diagnostics) and reprefill_decoupled (π_b for
Decoupled PPO)."""
import asyncio
import logging
import os

import numpy as np
import torch
import transfer_queue as tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def slice_response_logprobs(prompt_logprobs_ls, prompt_len, response_len):
    # sglang prompt_logprobs_ls has length S = prompt_len + response_len.
    # Entry i is the logprob of the token at position i+1 predicted by
    # tokens [0..i]. Response tokens occupy positions
    # [prompt_len, prompt_len + response_len - 1]; their logprobs are at
    # indices [prompt_len - 1, prompt_len + response_len - 2]. Each entry is
    # a single-element list when prompt_logprobs=0.
    start = max(prompt_len - 1, 0)
    end = prompt_len + response_len - 1
    return [float(entry[0]) for entry in prompt_logprobs_ls[start:end]]


def to_nested_jagged(nested_list):
    # Nested jagged layout is required for KVBatch.to_padded_tensor() to
    # recognize the field downstream (see trainer_staleness_sweep.py).
    return torch.nested.as_nested_tensor(
        [torch.tensor(lst, dtype=torch.float32) for lst in nested_list],
        layout=torch.jagged,
    )


def build_reprefill_inputs(keys, partition_id, pad_id):
    data = tq.kv_batch_get(
        keys=keys, partition_id=partition_id, select_fields=["prompts", "responses"]
    )
    prompts_padded = data["prompts"].to_padded_tensor(padding=pad_id)
    responses_padded = data["responses"].to_padded_tensor(padding=pad_id)

    prompt_ids_list: list[list[int]] = []
    real_lens: list[tuple[int, int]] = []
    for i in range(len(keys)):
        prompt_ids = [int(x) for x in prompts_padded[i].tolist() if x != pad_id]
        response_ids = [int(x) for x in responses_padded[i].tolist() if x != pad_id]
        prompt_ids_list.append(prompt_ids + response_ids)
        real_lens.append((len(prompt_ids), len(response_ids)))
    return prompt_ids_list, real_lens, data


async def reprefill_trajectories(client, prompt_ids_list, request_prefix="reprefill"):
    sampling_params_list = [{"prompt_logprobs": 0, "max_new_tokens": 0}] * len(prompt_ids_list)
    results = await asyncio.gather(*[
        client.generate(
            request_id=f"{request_prefix}_{i}",
            prompt_ids=pids,
            sampling_params=sp,
        )
        for i, (pids, sp) in enumerate(zip(prompt_ids_list, sampling_params_list))
    ])
    return results


def compute_and_emit_staleness_metrics(batch, metrics, global_steps):
    """Emit offpolicy/* diagnostics from the three logprobs (rollout / new_rollout / old).

    Shared by staleness_sweep and reprefill_decoupled. Fetches fields from TQ;
    silently returns on fetch failure (metrics are best-effort).
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

    fields = ["rollout_log_probs", "new_rollout_log_probs", "old_log_probs", "response_mask"]
    try:
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=fields,
        )
    except Exception as e:
        logger.warning(f"reprefill: failed to fetch logprobs for metrics: {e}")
        return

    from verl import DataProto
    data = DataProto(batch=data.to_padded_tensor())

    rollout_lp = data.batch["rollout_log_probs"]
    new_rollout_lp = data.batch["new_rollout_log_probs"]
    old_lp = data.batch["old_log_probs"]
    response_mask = data.batch["response_mask"]

    resume_versions = np.array(
        [tag.get("resume_version", global_steps - 1) for tag in batch.tags],
        dtype=np.int64,
    )
    staleness = (global_steps - 1) - resume_versions
    metrics["staleness_sweep/sample_staleness_mean"] = float(staleness.mean())
    metrics["staleness_sweep/sample_staleness_max"] = float(staleness.max())

    corr_metrics = compute_offpolicy_metrics(
        old_log_prob=old_lp,
        rollout_log_prob=rollout_lp,
        response_mask=response_mask,
        new_rollout_log_prob=new_rollout_lp,
    )
    for key, value in corr_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[f"offpolicy/{key}"] = value.item()
        else:
            metrics[f"offpolicy/{key}"] = value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_reprefill_utils_on_cpu.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/reprefill_utils.py tests/trainer/ppo/v1/test_reprefill_utils_on_cpu.py
git commit -m "feat(v1): extract shared re-prefill helpers from staleness_sweep

Co-authored-by: Claude"
```

---

### Task 2: Refactor `staleness_sweep` to consume the shared helpers

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_staleness_sweep.py` (methods `_compute_new_rollout_log_prob`, `_reprefill_trajectories_async`, `_build_reprefill_inputs`, `_slice_response_logprobs`, `_compute_staleness_metrics`)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: unchanged public behavior of `staleness_sweep` (same TQ fields, same metric names, same request IDs).

- [ ] **Step 1: Replace helper method bodies with shared-module calls**

In `trainer_staleness_sweep.py`:

1. Add imports, remove now-unused ones (`numpy`, `asyncio` if unused elsewhere):

```python
from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    compute_and_emit_staleness_metrics,
    reprefill_trajectories,
    slice_response_logprobs,
    to_nested_jagged,
)
```

2. Delete `_build_reprefill_inputs` and `_slice_response_logprobs` methods entirely.
3. Replace `_reprefill_trajectories_async` body:

```python
    @auto_await
    async def _reprefill_trajectories_async(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix="staleness_sweep_reprefill",
        )
```

4. In `_compute_new_rollout_log_prob`, replace the corresponding blocks:

```python
        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id, pad_id=self.tokenizer.pad_token_id
        )
        # (sampling_params_list construction removed — handled in reprefill_trajectories)
        results = self._reprefill_trajectories_async(prompt_ids_list)
```

and:

```python
        data["new_rollout_log_probs"] = to_nested_jagged(new_rollout_log_probs_nested)
```

(the loop appending to `new_rollout_log_probs_nested` keeps calling `slice_response_logprobs(...)` — now the imported function).

5. Replace `_compute_staleness_metrics` body with a call (metric names must stay identical, including the `staleness_sweep/` meta keys):

```python
    def _compute_staleness_metrics(self, batch: KVBatchMeta, metrics: dict) -> None:
        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
```

- [ ] **Step 2: Verify no behavior change**

Run: `source .venv/bin/activate && python -c "from verl.trainer.ppo.v1.trainer_staleness_sweep import PPOTrainerStalenessSweep; print('import ok')" && python -m pytest tests/trainer/ppo/v1/ -v -k "replay_buffer or trainer_base or reprefill_utils"`
Expected: import ok; all tests PASS (no staleness_sweep-specific CPU tests exist; this task is behavior-preserving refactor guarded by the import + surrounding suite).

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_staleness_sweep.py
git commit -m "refactor(v1): staleness_sweep consumes shared reprefill helpers

Co-authored-by: Claude"
```

---

### Task 3: `reprefill_decoupled` trainer — P1 core (synchronous)

**Files:**
- Create: `verl/trainer/ppo/v1/trainer_reprefill_decoupled.py`
- Modify: `verl/trainer/config/ppo_trainer.yaml` (add `reprefill_decoupled:` section after `staleness_sweep:`, around line 250)
- Test: `tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py`

**Interfaces:**
- Consumes: Task 1 helpers; `PPOTrainerColocateAsync` (registered as `colocate_async`); `register_trainer` from `trainer_base.py`.
- Produces: trainer class `PPOTrainerReprefillDecoupled` registered as `reprefill_decoupled`; TQ field `new_rollout_log_probs` (nested jagged); per-key tag `resume_version: int`; config keys `trainer.v1.reprefill_decoupled.num_warmup_batches` (int, default 1), `trainer.v1.reprefill_decoupled.enable_prefill_pipeline` (bool, default false — read in Task 6).

- [ ] **Step 1: Write failing tests**

Create `tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py`. Fixture pattern follows `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py` (`__new__`-constructed trainer + minimal OmegaConf; no Ray/GPU needed). Check `transfer_queue.KVBatchMeta`'s constructor signature via `inspect.signature(tq.KVBatchMeta)` before writing the batch factory — it must carry at least `keys`, `partition_id`, `tags`.

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from tests/trainer/ppo/v1/test_trainer_base_on_cpu.py)
"""CPU tests for the reprefill_decoupled trainer (P1 synchronous path)."""

import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

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
    fields = {
        "prompts": to_nested_jagged([[100 + j for j in range(pl)] for pl in prompt_lens]),
        "responses": to_nested_jagged([[200 + j for j in range(rl)] for rl in response_lens]),
    }
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
            fields={"new_rollout_log_probs": expected},
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
            fields={"rollout_log_probs": expected},
        )
        trainer = _make_trainer(rollout_correction={"bypass_mode": True})
        batch = _make_batch(keys, partition_id)

        trainer._compute_old_log_prob(batch, {})

        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["old_log_probs"]
        )
        padded = data["old_log_probs"].to_padded_tensor(padding=0.0)
        assert torch.allclose(padded, expected.to_padded_tensor(padding=0.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `trainer_reprefill_decoupled`.

- [ ] **Step 3: Implement the trainer**

Create `verl/trainer/ppo/v1/trainer_reprefill_decoupled.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from verl/trainer/ppo/v1/trainer_colocate_async.py)
"""Re-prefill Decoupled PPO trainer: π_b (old_log_probs) from rollout-side re-prefill.

Extends PPOTrainerColocateAsync:
1. `on_sampled()`: re-prefill all consumed trajectories on the rollout engine
   (which holds W_{k-1} throughout the step window) and write
   `new_rollout_log_probs` + `resume_version` tags to TransferQueue.
2. `_compute_old_log_prob()`: copy `new_rollout_log_probs` as `old_log_probs`
   — no trainer-side forward pass. Downstream TIS/MIS/RS correction
   (compute_rollout_correction_and_add_to_batch) and the actor loss are
   unchanged; they consume `old_log_probs` as π_b regardless of origin.

P2 (`enable_prefill_pipeline=true`): pre-dispatch re-prefills while the
replay buffer poll loop waits for remaining generation (see Task 6).

Enable via: trainer.v1.trainer_mode=reprefill_decoupled
"""
import logging
import os

import transfer_queue as tq

from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    compute_and_emit_staleness_metrics,
    reprefill_trajectories,
    slice_response_logprobs,
    to_nested_jagged,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("reprefill_decoupled")
class PPOTrainerReprefillDecoupled(PPOTrainerColocateAsync):
    """Re-prefill Decoupled PPO trainer (colocate async)."""

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.reprefill_decoupled.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_sampled(self, batch: "KVBatchMeta", metrics: dict) -> "KVBatchMeta":
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    @auto_await
    async def _reprefill_all(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix=f"reprefill_decoupled_{self.global_steps}",
        )

    def _compute_new_rollout_log_prob(self, batch, metrics):
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        results = self._reprefill_all(prompt_ids_list)
        nested = [
            slice_response_logprobs(
                results[i].extra_fields["prompt_logprobs"], real_lens[i][0], real_lens[i][1]
            )
            for i in range(len(batch.keys))
        ]
        data["new_rollout_log_probs"] = to_nested_jagged(nested)
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=data.select("new_rollout_log_probs"),
        )
        for i in range(len(batch.keys)):
            if batch.tags[i] is None:
                batch.tags[i] = {}
            batch.tags[i]["resume_version"] = int(resume_version)
        metrics["reprefill_decoupled/resume_version"] = float(resume_version)
        return batch

    def _compute_old_log_prob(self, batch, metrics: dict):
        # Bypass mode stays available for A/B experiments: parent copies
        # rollout_log_probs as old_log_probs (2-policy semantics).
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass:
            return super()._compute_old_log_prob(batch, metrics)

        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        data["old_log_probs"] = data.pop("new_rollout_log_probs")
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data
        )
        metrics["reprefill_decoupled/old_log_prob_source"] = 1.0  # 1.0 = reprefill
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        # offpolicy/* diagnostics (staleness / mismatch / combined) from the
        # three logprobs; then parent computes the actual advantage. PPO loss
        # is unchanged (parent uses the combined ratio).
        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
        return super()._compute_advantage(batch, metrics)

    def _debug_log_prob_extra_fields(self) -> list[str]:
        # Expose new_rollout_log_probs to calculate_debug_metrics.
        return ["new_rollout_log_probs"]
```

Also add to `verl/trainer/config/ppo_trainer.yaml` after the `staleness_sweep:` block (before `sampler:`):

```yaml
    # Re-prefill Decoupled PPO trainer: colocate_async where π_b
    # (old_log_probs) is computed by re-prefilling consumed trajectories on
    # the rollout engine at its current weight W_{k-1}, replacing the
    # trainer-side old_log_prob forward pass.
    reprefill_decoupled:

      # Number of warmup batches to add before training loop starts
      num_warmup_batches: 1

      # Pre-dispatch re-prefill for samples as they finish during the
      # replay-buffer poll loop, overlapping with remaining generation.
      enable_prefill_pipeline: false
```

Note: `_build_replay_buffer` in `trainer_base.py` currently falls back to `ReplayBufferAsync` for any mode not in `sync_modes = ("sync", "staleness_sweep")` — `reprefill_decoupled` therefore already gets `ReplayBufferAsync` in P1. No change needed until Task 5.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py -v`
Expected: all PASS. If `tq.KVBatchMeta(...)` constructor differs, inspect its signature and adapt `_make_batch` only.

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_reprefill_decoupled.py verl/trainer/config/ppo_trainer.yaml tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py
git commit -m "feat(v1): add reprefill_decoupled trainer (P1 synchronous re-prefill π_b)

Co-authored-by: Claude"
```

---

### Task 4: `ReprefillReplayBuffer` with `on_new_finished` hook

**Files:**
- Modify: `verl/trainer/ppo/v1/replay_buffer.py` (add class after `ReplayBufferAsync`)
- Modify: `verl/trainer/ppo/v1/trainer_base.py:158-165` (mode → class dispatch)
- Test: `tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py`

**Interfaces:**
- Consumes: `ReplayBufferAsync` (same constructor kwargs — dispatched via the existing else-branch in `_build_replay_buffer`).
- Produces: class `ReprefillReplayBuffer(ReplayBufferAsync)` with method `set_on_new_finished_callback(callback: Callable[[str, set[str]], None])` — callback receives `(partition_id, newly_finished_keys)` and is invoked after each metadata sync inside `sample()`'s poll loop. Callback exceptions must not break sampling.

- [ ] **Step 1: Write failing tests**

Create `tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py`. Reuse the producer helpers from `test_replay_buffer_on_cpu.py` — copy the file-local `_produce`/`PromptSpec`/`_uid`/`_trajectory_key` helpers (they are file-local; copy, don't import):

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py)
"""CPU tests for ReprefillReplayBuffer's on_new_finished hook."""

import uuid

import pytest
import transfer_queue as tq

from verl.trainer.ppo.v1.replay_buffer import ReprefillReplayBuffer

# ... copy _uid, _trajectory_key, PromptSpec, _produce, _set_prompt_status
# helpers verbatim from tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py ...


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _make_rb(poll_interval=0.05):
    return ReprefillReplayBuffer(
        trainer_mode="reprefill_decoupled",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=poll_interval,
        refill_fn=None,
    )


def test_callback_fires_for_newly_finished_keys(tq_init, partition_id):
    rb = _make_rb()
    seen: list[tuple[str, set]] = []
    rb.set_on_new_finished_callback(lambda pid, keys: seen.append((pid, keys)))

    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(2)]
    _produce(partition_id, specs).join_and_check()

    rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

    all_seen = {k for _, keys in seen for k in keys}
    assert all_seen == {spec.uid for spec in specs}


def test_callback_does_not_refire_for_old_keys(tq_init, partition_id):
    rb = _make_rb()
    seen: list[set] = []
    rb.set_on_new_finished_callback(lambda pid, keys: seen.append(keys))

    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(2)]
    _produce(partition_id, specs).join_and_check()
    rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

    # Second sample call: same keys are already finished — no new callback events.
    seen.clear()
    specs2 = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=1) for _ in range(1)]
    _produce(partition_id, specs2).join_and_check()
    rb.sample(global_steps=2, partition_id=partition_id, batch_size=3)
    all_seen = {k for keys in seen for k in keys}
    assert all_seen == {specs2[0].uid}


def test_callback_exception_does_not_break_sampling(tq_init, partition_id):
    rb = _make_rb()
    rb.set_on_new_finished_callback(lambda pid, keys: (_ for _ in ()).throw(RuntimeError("boom")))

    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(1)]
    _produce(partition_id, specs).join_and_check()

    batch, _ = rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
    assert len(batch) == 1
```

Adapt the constructor kwargs / `PromptSpec` fields to match the actual signatures in `test_replay_buffer_on_cpu.py` when copying (they are the source of truth).

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReprefillReplayBuffer'`.

- [ ] **Step 3: Implement `ReprefillReplayBuffer` and dispatch**

In `replay_buffer.py`, after `ReplayBufferAsync`:

```python
class ReprefillReplayBuffer(ReplayBufferAsync):
    """ReplayBufferAsync that notifies a callback when keys transition to finished.

    Used by reprefill_decoupled to pre-dispatch re-prefill requests while the
    sample() poll loop waits for the remaining trajectories. The rollout
    engine weight is constant within a step window, so re-prefills issued
    here are version-aligned with the π_b the upcoming on_sampled needs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_new_finished_callback = None

    def set_on_new_finished_callback(self, callback):
        self._on_new_finished_callback = callback

    def _sync_metadata_from_transfer_queue(self):
        before = {pid: set(keys) for pid, keys in self.finished_keys.items()}
        super()._sync_metadata_from_transfer_queue()
        if self._on_new_finished_callback is None:
            return
        for partition_id, keys in self.finished_keys.items():
            new_keys = keys - before.get(partition_id, set())
            if not new_keys:
                continue
            try:
                self._on_new_finished_callback(partition_id, new_keys)
            except Exception as e:
                # Diagnostics/prefill must never break sampling.
                logger.warning(f"ReprefillReplayBuffer: on_new_finished callback failed: {e}")
```

In `trainer_base.py` `_build_replay_buffer` (the `sync_modes` branch at trainer_base.py:158-165), change the dispatch:

```python
            sync_modes = ("sync", "staleness_sweep")
            if self.trainer_mode == "reprefill_decoupled":
                from verl.trainer.ppo.v1.replay_buffer import ReprefillReplayBuffer
                sampler_cls = ReprefillReplayBuffer
            else:
                sampler_cls = ReplayBuffer if self.trainer_mode in sync_modes else ReplayBufferAsync
```

(`replay_buffer` is already imported at module top in trainer_base.py — use the existing import instead of a function-local one if the name is available.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py tests/trainer/ppo/v1/test_trainer_base_on_cpu.py -v`
Expected: all PASS (including the existing `test_builtin_sampler_class_follows_trainer_mode`, which must still pass — `reprefill_decoupled` is a new branch it doesn't cover).

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/replay_buffer.py verl/trainer/ppo/v1/trainer_base.py tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py
git commit -m "feat(v1): ReprefillReplayBuffer with on_new_finished hook

Co-authored-by: Claude"
```

---

### Task 5: P2 — pipelined pre-dispatch in the trainer

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_reprefill_decoupled.py` (add `_PrefillDispatcher`, `on_train_begin` registration, pipelined `on_sampled`)
- Test: `tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py` (append)

**Interfaces:**
- Consumes: `ReprefillReplayBuffer.set_on_new_finished_callback` (Task 4), Task 1 helpers, config flag `trainer.v1.reprefill_decoupled.enable_prefill_pipeline`.
- Produces: `_PrefillDispatcher` (methods `start()`, `submit(coro) -> concurrent.futures.Future`, `shutdown()`); trainer attributes `_pending_prefill: dict[str, _PendingPrefill]` where `_PendingPrefill` is a dataclass with fields `version: int` (value of `self.global_steps` at issue time) and `future: concurrent.futures.Future` (result is a 1-element list of engine results).

**Version-guard rules (implement exactly):**
1. A pending entry is valid at `on_sampled` time iff `entry.version == self.global_steps` (issued during the current step window; engine weight is constant within the window).
2. Secondary guard: if the engine result carries `extra_fields["global_steps"]` and it differs from `self.global_steps - 1`, log a warning and treat the entry as invalid (re-issue synchronously).
3. After each `on_sampled`, clear `_pending_prefill` entirely (entries for unselected keys would be wrong-version next step anyway).

- [ ] **Step 1: Write failing tests**

Append to `tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py`:

```python
import concurrent.futures


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
    def _make_pipelined_trainer(self):
        trainer = _make_trainer()
        trainer.config.trainer.v1.reprefill_decoupled.enable_prefill_pipeline = True
        trainer._prefill_dispatcher = None  # reissue path must not need it
        trainer._pending_prefill = {}
        return trainer

    def test_hit_valid_entry_skips_reissue(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0"]
        _write_trajectories(keys, partition_id, prompt_lens=[2], response_lens=[2])
        trainer = self._make_pipelined_trainer()

        async def ok():
            return [SimpleNamespace(extra_fields={
                "prompt_logprobs": [[-1.0 * i] for i in range(4)],  # prompt+response = 4
                "global_steps": 2,  # == trainer.global_steps - 1
            })]

        fut = concurrent.futures.Future()
        fut.set_result_from_thread = None  # use set_result directly
        fut.set_result(None)
        # Build a real completed future via the dispatcher-free path:
        import asyncio
        fut = asyncio.new_event_loop().run_until_complete(_wrap(ok()))

        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill
        trainer._pending_prefill[keys[0]] = _PendingPrefill(version=trainer.global_steps, future=fut)

        reissued = []
        trainer._reprefill_all = lambda prompt_ids_list: (
            reissued.append(prompt_ids_list) or []
        ) and [SimpleNamespace(extra_fields={"prompt_logprobs": [[0.0]]})]

        batch = _make_batch(keys, partition_id)
        trainer.on_sampled(batch, {})

        assert reissued == []  # valid entry consumed, no synchronous re-prefill
        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["new_rollout_log_probs"]
        )
        assert data["new_rollout_log_probs"].to_padded_tensor(padding=0.0).shape == (1, 2)
        assert trainer._pending_prefill == {}  # cleared after consume

    def test_stale_version_entry_reissues_synchronously(self, tq_init, partition_id):
        keys = [f"{uuid.uuid4().hex}_0_0"]
        _write_trajectories(keys, partition_id, prompt_lens=[2], response_lens=[2])
        trainer = self._make_pipelined_trainer()

        from verl.trainer.ppo.v1.trainer_reprefill_decoupled import _PendingPrefill
        stale = concurrent.futures.Future()
        stale.set_result([SimpleNamespace(extra_fields={"prompt_logprobs": [[0.0]]})])
        trainer._pending_prefill[keys[0]] = _PendingPrefill(
            version=trainer.global_steps - 1, future=stale  # issued in a previous window
        )

        trainer._reprefill_all = lambda prompt_ids_list: [SimpleNamespace(extra_fields={
            "prompt_logprobs": [[-0.5 * i] for i in range(4)]
        })]

        batch = _make_batch(keys, partition_id)
        trainer.on_sampled(batch, {})

        data = tq.kv_batch_get(
            keys=keys, partition_id=partition_id, select_fields=["new_rollout_log_probs"]
        )
        assert data["new_rollout_log_probs"].to_padded_tensor(padding=0.0).shape == (1, 2)
```

Where `_wrap` is a tiny helper at module level:

```python
async def _wrap(coro):
    future = concurrent.futures.Future()
    result = await coro
    future.set_result(result)
    return future
```

(The `test_hit_valid_entry_skips_reissue` body contains exploratory lines from drafting — clean them up while writing: the final entry future must be a `concurrent.futures.Future` whose result is the 1-element result list `[SimpleNamespace(...)]`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py -v -k "PrefillDispatcher or Pipelined"`
Expected: FAIL with `ImportError: cannot import name '_PrefillDispatcher'`.

- [ ] **Step 3: Implement the dispatcher and pipelined path**

Add to `trainer_reprefill_decoupled.py`:

```python
import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass


@dataclass
class _PendingPrefill:
    version: int  # trainer.global_steps when the re-prefill was issued
    future: concurrent.futures.Future  # result: 1-element list of engine results


class _PrefillDispatcher:
    """Background event-loop thread for issuing re-prefill requests off the
    trainer thread (called from the replay buffer poll loop)."""

    def __init__(self):
        self._loop = None
        self._thread = None

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            self._loop = None
            self._thread = None
```

In `PPOTrainerReprefillDecoupled`:

```python
    def on_train_begin(self):
        super().on_train_begin()
        if self.config.trainer.v1.reprefill_decoupled.get("enable_prefill_pipeline", False):
            self._prefill_dispatcher = _PrefillDispatcher()
            self._prefill_dispatcher.start()
            self._pending_prefill: dict[str, _PendingPrefill] = {}
            self.replay_buffer.set_on_new_finished_callback(self._on_new_finished)
            logger.info("reprefill_decoupled: pipelined re-prefill pre-dispatch enabled")

    def _on_new_finished(self, partition_id, new_keys):
        if partition_id != "train":
            return
        for key in new_keys:
            if key in self._pending_prefill:
                continue
            try:
                prompt_ids_list, _, _ = build_reprefill_inputs(
                    keys=[key], partition_id=partition_id,
                    pad_id=self.tokenizer.pad_token_id,
                )
                future = self._prefill_dispatcher.submit(
                    reprefill_trajectories(
                        client=self.get_llm_client(),
                        prompt_ids_list=prompt_ids_list,
                        request_prefix=f"reprefill_p2_{self.global_steps}_{key}",
                    )
                )
                self._pending_prefill[key] = _PendingPrefill(
                    version=self.global_steps, future=future
                )
            except Exception as e:
                logger.warning(f"reprefill_decoupled: pre-dispatch failed for {key}: {e}")

    def on_sampled(self, batch: "KVBatchMeta", metrics: dict) -> "KVBatchMeta":
        if getattr(self, "_pending_prefill", None) is not None:
            with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
                batch = self._compute_new_rollout_log_prob_pipelined(batch, metrics)
        else:
            with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
                batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    def _compute_new_rollout_log_prob_pipelined(self, batch, metrics):
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        results_by_index: dict[int, object] = {}
        reissue_indices: list[int] = []
        for i, key in enumerate(batch.keys):
            entry = self._pending_prefill.get(key)
            if entry is not None and entry.version == self.global_steps:
                try:
                    result = entry.future.result(timeout=600)[0]
                    engine_version = result.extra_fields.get("global_steps", None)
                    if engine_version is not None and engine_version != resume_version:
                        logger.warning(
                            f"reprefill_decoupled: prefill version mismatch for {key} "
                            f"(engine={engine_version}, expected={resume_version}); re-issuing"
                        )
                    else:
                        results_by_index[i] = result
                        continue
                except Exception as e:
                    logger.warning(f"reprefill_decoupled: prefill future failed for {key}: {e}")
            reissue_indices.append(i)

        if reissue_indices:
            keys = [batch.keys[i] for i in reissue_indices]
            prompt_ids_sub, _, _ = build_reprefill_inputs(
                keys=keys, partition_id=batch.partition_id,
                pad_id=self.tokenizer.pad_token_id,
            )
            reissue_results = self._reprefill_all(prompt_ids_sub)
            for j, i in enumerate(reissue_indices):
                results_by_index[i] = reissue_results[j]

        nested = [
            slice_response_logprobs(
                results_by_index[i].extra_fields["prompt_logprobs"],
                real_lens[i][0], real_lens[i][1],
            )
            for i in range(len(batch.keys))
        ]
        data["new_rollout_log_probs"] = to_nested_jagged(nested)
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=data.select("new_rollout_log_probs"),
        )
        for i in range(len(batch.keys)):
            if batch.tags[i] is None:
                batch.tags[i] = {}
            batch.tags[i]["resume_version"] = int(resume_version)
        metrics["reprefill_decoupled/resume_version"] = float(resume_version)
        # Entries for unselected keys are wrong-version next step; drop all.
        self._pending_prefill.clear()
        return batch
```

Note: `on_train_begin` calls `super().on_train_begin()` which reads `trainer.v1.colocate_async.num_warmup_batches` — the parent `PPOTrainerColocateAsync.on_train_begin` accesses the `colocate_async` section. Keep the Task 3 override that reads `reprefill_decoupled.num_warmup_batches` directly and inline the warmup logic (do not call super) — reconcile with the Task 3 implementation so there is exactly one `on_train_begin` that (a) adds `reprefill_decoupled.num_warmup_batches` warmup batches, (b) starts the dispatcher + registers the callback when pipelining is enabled.

- [ ] **Step 4: Run the full new-trainer test file**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py -v`
Expected: all PASS (P1 tests must still pass — pipeline disabled path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_reprefill_decoupled.py tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py
git commit -m "feat(v1): pipelined re-prefill pre-dispatch for reprefill_decoupled (P2)

Co-authored-by: Claude"
```

---

### Task 6: Run script + acceptance harness (manual GPU)

**Files:**
- Create: `exp_scripts/qwen3_8b/reprefill_decoupled/run_qwen3_8b_megatron_reprefill_decoupled.sh`
- Create: `exp_scripts/qwen3_8b/reprefill_decoupled/README.md`

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: cluster launch script mirroring `exp_scripts/qwen3_8b/staleness_sweep/run_qwen3_8b_megatron_staleness_sweep.sh` with mode swapped; README documenting the three acceptance arms.

- [ ] **Step 1: Create the run script**

Copy `exp_scripts/qwen3_8b/staleness_sweep/run_qwen3_8b_megatron_staleness_sweep.sh` to `exp_scripts/qwen3_8b/reprefill_decoupled/run_qwen3_8b_megatron_reprefill_decoupled.sh` and change:
- `trainer.v1.trainer_mode=staleness_sweep` → `trainer.v1.trainer_mode=reprefill_decoupled`
- remove `trainer.v1.staleness_sweep.num_steps=...` if present
- add `trainer.v1.reprefill_decoupled.enable_prefill_pipeline=${ENABLE_PREFILL_PIPELINE:-false}` (env-overridable so P1/P2 comparison uses one script)
- add `algorithm.rollout_correction.rollout_is=...` / correction preset passthrough if the staleness_sweep script has one; otherwise document in README that correction presets are passed via `+algorithm.rollout_correction=...` overrides

- [ ] **Step 2: Create the README**

Document: (1) enable flags, (2) the three acceptance arms from spec §4 — bypass mode with high staleness (negative control, expect collapse), `reprefill_decoupled` (expect stable: no entropy collapse / KL explosion, reward grows), existing decoupled trainer-π_b (upper-bound reference), (3) the metrics to watch: `offpolicy/*`, `staleness_sweep/sample_staleness_*`, `reprefill_decoupled/*`, `new_rollout_log_prob` timer (P1 vs P2), (4) how to force staleness (large `num_warmup_batches`, e.g. 4-8× train_batch).

- [ ] **Step 3: Verify script syntax**

Run: `bash -n exp_scripts/qwen3_8b/reprefill_decoupled/run_qwen3_8b_megatron_reprefill_decoupled.sh`
Expected: no output (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add exp_scripts/qwen3_8b/reprefill_decoupled/
git commit -m "feat(exp): reprefill_decoupled run script + acceptance README

Co-authored-by: Claude"
```

- [ ] **Step 5: Full local suite gate**

Run: `source .venv/bin/activate && python -m pytest tests/trainer/ppo/v1/ -v`
Expected: all PASS.

---

## Out of scope (spec §6 / roadmap B & C)

- separate_async port of `reprefill_decoupled` (P3).
- Trajectory-level selective re-prefill (方案 B) and token-level partial_rollout-resume re-prefill (方案 C) — hooks reserved (`resume_version` tags, `new_rollout_log_probs` TQ field) but not implemented.

## Self-Review Notes

- Spec coverage: §3.1 helpers → Task 1-2; §3.2 trainer + config → Task 3; §4 P1 → Task 3 (sync path is the default `enable_prefill_pipeline=false`); §5 P2 → Tasks 4-5; §8 testing → per-task CPU tests + Task 6 manual acceptance; §7 risks → version guards (Task 5), timer metric (Task 3/5 marked_timer), T/R mismatch observability (`offpolicy/*` in `_compute_advantage`).
- Type consistency: `_PendingPrefill(version, future)` used identically in Task 5 implementation and tests; `build_reprefill_inputs` signature `(keys, partition_id, pad_id)` consistent across Tasks 1, 2, 3, 5.
- Known execution-time unknowns (flagged inline): `tq.KVBatchMeta` constructor signature, `tq.kv_batch_put` acceptance of nested-jagged dicts, `PromptSpec`/`_produce` exact shapes — each test instructs the executor to verify against the existing `test_replay_buffer_on_cpu.py` source of truth rather than guess.
