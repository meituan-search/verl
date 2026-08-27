# Partial Re-prefill (`partial_reprefill` trainer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `partial_reprefill` trainer that per-trajectory dispatches between skip-reprefill (fully-fresh), consume-client-piggyback (partial_rollout), and full-reprefill (fully-stale), reducing wasted re-prefill work in colocate async training.

**Architecture:** New trainer `PPOTrainerPartialReprefill(PPOTrainerColocateAsync)` in `verl/trainer/ppo/v1/trainer_partial_reprefill.py`, sibling to `reprefill_decoupled`. Client (`FullyAsyncLLMServerClient.generate`) is modified so partial_rollout resume requests `prompt_logprobs`, builds a per-token `token_versions` array, and writes `new_rollout_log_probs` for partial trajectories (prefix from resume-prefill prompt_logprobs, suffix copied from `rollout_log_probs`). Trainer at sample time dispatches per-trajectory: piggyback marker → case 1 skip; `token_versions[-1] == current_version` → case 3 copy; else → case 2 full reprefill.

**Tech Stack:** Python 3.12, PyTorch (nested jagged tensors), transfer_queue (TQ), SGLang, pytest, OmegaConf.

**Spec:** `docs/superpowers/specs/2026-08-27-partial-reprefill-design.md`

## Global Constraints

- Python environment managed via `uv` (Python 3.12). Pre-commit installed.
- All new code follows verl's pre-commit (ruff format + ruff check).
- TDD: every code task writes a failing CPU test first, verifies fail, implements minimal, verifies pass, commits.
- Commit trailers include `Co-authored-by: Claude` per CLAUDE.md.
- Config namespace: `trainer.v1.partial_reprefill.*` — must be added to `verl/trainer/config/ppo_trainer.yaml` AND all four `_generated_ppo_*_trainer.yaml` files (they are auto-generated mirrors; regenerate via `python -m verl.trainer.config.generate` if available, otherwise edit by hand).
- Tests follow the pattern in `tests/trainer/ppo/v1/test_trainer_reprefill_decoupled_on_cpu.py`: construct trainer via `__new__`, set `config`/`global_steps` manually, call methods directly.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `verl/trainer/ppo/v1/trainer_partial_reprefill.py` | Create | New trainer class; case-decision dispatcher; P2 pipelined path |
| `verl/trainer/ppo/v1/reprefill_utils.py` | Modify | Add `decide_case`, `build_token_versions`, `build_partial_new_rollout_log_probs`, `compute_and_emit_token_staleness_metrics` helpers |
| `verl/workers/rollout/llm_server.py` | Modify | `FullyAsyncLLMServerClient.generate` partial_rollout resume path: request prompt_logprobs, build token_versions + new_rollout_log_probs, set piggyback marker |
| `verl/trainer/ppo/v1/__init__.py` | Modify | Export `PPOTrainerPartialReprefill` |
| `verl/trainer/config/ppo_trainer.yaml` | Modify | Add `partial_reprefill` config block |
| `verl/trainer/config/_generated_ppo_*_trainer.yaml` (×4) | Modify | Mirror config block |
| `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py` | Create | Unit tests for new helpers |
| `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py` | Create | Unit tests for trainer methods |
| `tests/workers/rollout/test_llm_server_partial_piggyback.py` | Create | Unit tests for client piggyback |

---

## Phase A: Trainer skeleton + case 3 (skip reprefill for fully-fresh)

### Task A1: Add config block + register trainer

**Files:**
- Modify: `verl/trainer/config/ppo_trainer.yaml` (after `reprefill_decoupled:` block, ~line 270)
- Modify: `verl/trainer/config/_generated_ppo_trainer.yaml`, `_generated_ppo_veomni_trainer.yaml`, `_generated_ppo_torchtitan_trainer.yaml`, `_generated_ppo_megatron_trainer.yaml` (mirror)
- Modify: `verl/trainer/ppo/v1/__init__.py`

**Interfaces:**
- Produces: config namespace `trainer.v1.partial_reprefill` with keys `num_warmup_batches` (int, default 1), `enable_prefill_pipeline` (bool, default false), `enable_case_skip` (bool, default true), `enable_piggyback` (bool, default true), `compare_trainer_old_log_prob` (bool, default false).
- Produces: `PPOTrainerPartialReprefill` importable from `verl.trainer.ppo.v1`.

- [ ] **Step 1: Add config block to `ppo_trainer.yaml`**

Insert after the `reprefill_decoupled:` block (around line 270):

```yaml
    # Partial Re-prefill trainer: per-trajectory case dispatch between
    # fully-fresh (skip reprefill), partial piggyback (consume client-side
    # resume-prefill prompt_logprobs), and fully-stale (full reprefill at
    # W_sample). Sibling to reprefill_decoupled.
    partial_reprefill:

      # Number of warmup batches to add before training loop starts
      num_warmup_batches: 1

      # Pre-dispatch re-prefill for samples as they finish during the
      # replay-buffer poll loop (P2 pipelined path).
      enable_prefill_pipeline: false

      # Case 3 fast path: skip reprefill for fully-fresh trajectories
      # (all tokens decoded at current rollout weight). Set false to
      # force full reprefill for A/B isolation.
      enable_case_skip: true

      # Case 1: consume client-side piggyback (partial_rollout resume
      # prefill prompt_logprobs for stale prefix). Set false to ignore
      # piggyback marker and always do full reprefill for A/B isolation.
      enable_piggyback: true

      # A/B timing comparison: keep computing the trainer-side old_log_prob
      # forward pass alongside the rollout-side computation, stored as
      # trainer_old_log_probs.
      compare_trainer_old_log_prob: false
```

- [ ] **Step 2: Mirror to the four `_generated_ppo_*_trainer.yaml` files**

Apply the same block insertion to each generated file. The exact line numbers differ; find the `reprefill_decoupled:` block in each and insert after.

- [ ] **Step 3: Register trainer in `__init__.py`**

Add to `verl/trainer/ppo/v1/__init__.py`:

```python
from .trainer_partial_reprefill import PPOTrainerPartialReprefill
```

And append `"PPOTrainerPartialReprefill"` to `__all__`.

- [ ] **Step 4: Create the trainer file skeleton (empty class)**

Create `verl/trainer/ppo/v1/trainer_partial_reprefill.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Apache-2.0 license header — copy from trainer_reprefill_decoupled.py)
"""Partial Re-prefill trainer: per-trajectory case dispatch.

Extends PPOTrainerColocateAsync. Sibling to reprefill_decoupled.
Enable via: trainer.v1.trainer_mode=partial_reprefill
"""

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync


@register_trainer("partial_reprefill")
class PPOTrainerPartialReprefill(PPOTrainerColocateAsync):
    """Partial reprefill trainer (colocate async)."""

    pass
```

- [ ] **Step 5: Verify import works**

Run: `python -c "from verl.trainer.ppo.v1 import PPOTrainerPartialReprefill; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add verl/trainer/config/ppo_trainer.yaml \
        verl/trainer/config/_generated_ppo_trainer.yaml \
        verl/trainer/config/_generated_ppo_veomni_trainer.yaml \
        verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml \
        verl/trainer/config/_generated_ppo_megatron_trainer.yaml \
        verl/trainer/ppo/v1/__init__.py \
        verl/trainer/ppo/v1/trainer_partial_reprefill.py
git commit -m "feat: add partial_reprefill trainer skeleton + config namespace"
```

---

### Task A2: `decide_case` helper in `reprefill_utils.py`

**Files:**
- Modify: `verl/trainer/ppo/v1/reprefill_utils.py` (append after existing helpers)
- Test: `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py` (create)

**Interfaces:**
- Produces: `decide_case(piggyback_marker: bool, last_token_version: int | None, current_parameter_version: int, enable_case_skip: bool, enable_piggyback: bool) -> int` — returns 1, 2, or 3.

- [ ] **Step 1: Write the failing test**

Create `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py`:

```python
# Apache-2.0 license header (copy from test_reprefill_utils_on_cpu.py)
"""CPU tests for partial reprefill helpers."""

import pytest

from verl.trainer.ppo.v1.reprefill_utils import decide_case


class TestDecideCase:
    def test_case1_piggyback_enabled(self):
        assert decide_case(
            piggyback_marker=True,
            last_token_version=5,
            current_parameter_version=6,
            enable_case_skip=True,
            enable_piggyback=True,
        ) == 1

    def test_case1_piggyback_disabled_falls_to_case2(self):
        # piggyback disabled: even if marker set, fall through to case dispatch
        assert decide_case(
            piggyback_marker=True,
            last_token_version=5,
            current_parameter_version=6,
            enable_case_skip=True,
            enable_piggyback=False,
        ) == 2

    def test_case3_fully_fresh(self):
        assert decide_case(
            piggyback_marker=False,
            last_token_version=6,
            current_parameter_version=6,
            enable_case_skip=True,
            enable_piggyback=True,
        ) == 3

    def test_case2_fully_stale(self):
        assert decide_case(
            piggyback_marker=False,
            last_token_version=5,
            current_parameter_version=6,
            enable_case_skip=True,
            enable_piggyback=True,
        ) == 2

    def test_case3_disabled_falls_to_case2(self):
        # case skip disabled: even if fresh, force case 2
        assert decide_case(
            piggyback_marker=False,
            last_token_version=6,
            current_parameter_version=6,
            enable_case_skip=False,
            enable_piggyback=True,
        ) == 2

    def test_missing_last_token_version_is_case2(self):
        # token_versions not populated (e.g. older client): default to case 2
        assert decide_case(
            piggyback_marker=False,
            last_token_version=None,
            current_parameter_version=6,
            enable_case_skip=True,
            enable_piggyback=True,
        ) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py -v`
Expected: FAIL with `ImportError: cannot import name 'decide_case'`

- [ ] **Step 3: Implement `decide_case`**

Append to `verl/trainer/ppo/v1/reprefill_utils.py`:

```python
def decide_case(
    piggyback_marker: bool,
    last_token_version: int | None,
    current_parameter_version: int,
    enable_case_skip: bool,
    enable_piggyback: bool,
) -> int:
    """Decide which reprefill case to apply to a trajectory.

    Returns 1 (piggyback), 2 (full reprefill), or 3 (skip — copy rollout_log_probs).
    Case 1 requires piggyback marker AND enable_piggyback. Case 3 requires
    last_token_version == current_parameter_version AND enable_case_skip.
    Everything else falls through to case 2.
    """
    if piggyback_marker and enable_piggyback:
        return 1
    if (
        enable_case_skip
        and last_token_version is not None
        and last_token_version == current_parameter_version
    ):
        return 3
    return 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/reprefill_utils.py \
        tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py
git commit -m "feat: add decide_case helper for partial reprefill case dispatch"
```

---

### Task A3: Trainer `_compute_new_rollout_log_prob` with case 3 logic

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_partial_reprefill.py` (replace skeleton)
- Test: `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py` (create)

**Interfaces:**
- Consumes: `decide_case` from `reprefill_utils`; `build_reprefill_inputs`, `slice_response_logprobs`, `to_nested_jagged` from `reprefill_utils`.
- Produces: `PPOTrainerPartialReprefill._compute_new_rollout_log_prob(batch, metrics)` that handles cases 1, 2, 3.

- [ ] **Step 1: Write the failing test**

Create `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py`:

```python
# Apache-2.0 license header
"""CPU tests for the partial_reprefill trainer (case dispatch)."""

import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1.reprefill_utils import to_nested_jagged
from verl.trainer.ppo.v1.trainer_partial_reprefill import PPOTrainerPartialReprefill


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _make_trainer(enable_case_skip=True, enable_piggyback=True):
    trainer = PPOTrainerPartialReprefill.__new__(PPOTrainerPartialReprefill)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": None},
            "trainer": {
                "v1": {
                    "partial_reprefill": {
                        "enable_prefill_pipeline": False,
                        "enable_case_skip": enable_case_skip,
                        "enable_piggyback": enable_piggyback,
                        "compare_trainer_old_log_prob": False,
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
        tq.kv_batch_put(
            keys=[key],
            partition_id=partition_id,
            fields={
                "rollout_log_probs": to_nested_jagged([rollout_lp]),
                "token_versions": torch.nested.as_nested_tensor(
                    [torch.tensor([5, 5, 5, 5], dtype=torch.int32)], layout=torch.jagged
                ),
            },
        )
        batch = _make_batch(partition_id, [key])
        metrics = {}
        # Bypass the client call by stubbing _reprefill_all to assert it's NOT called
        called = {"yes": False}

        def _reprefill_all(_):
            called["yes"] = True
            return []

        trainer._reprefill_all = _reprefill_all
        out = trainer._compute_new_rollout_log_prob(batch, metrics)
        assert called["yes"] is False, "case 3 must not call reprefill"
        assert metrics["partial_reprefill/case_distribution.case_3"] == 1.0
        # Verify new_rollout_log_probs == rollout_log_probs
        data = tq.kv_batch_get(
            keys=[key], partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        result = data["new_rollout_log_probs"][0].tolist()
        assert result == rollout_lp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py -v`
Expected: FAIL with `AttributeError: 'PPOTrainerPartialReprefill' object has no attribute '_compute_new_rollout_log_prob'`

- [ ] **Step 3: Implement the trainer with case 3 logic**

Replace `verl/trainer/ppo/v1/trainer_partial_reprefill.py`:

```python
# Apache-2.0 license header
"""Partial Re-prefill trainer: per-trajectory case dispatch.

Extends PPOTrainerColocateAsync. Sibling to reprefill_decoupled.

Three cases per trajectory at sample time:
- Case 1 (piggyback): client already populated new_rollout_log_probs during
  partial_rollout resume. Skip.
- Case 2 (fully-stale): full reprefill at W_sample. Existing _reprefill_all logic.
- Case 3 (fully-fresh): all tokens at current weight. Copy rollout_log_probs.

Enable via: trainer.v1.trainer_mode=partial_reprefill
"""

import logging
import os

import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    decide_case,
    slice_response_logprobs,
    to_nested_jagged,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await
from verl.trainer.ppo.v1.reprefill_utils import reprefill_trajectories

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("partial_reprefill")
class PPOTrainerPartialReprefill(PPOTrainerColocateAsync):
    """Partial reprefill trainer (colocate async)."""

    def on_train_begin(self):
        cfg = self.config.trainer.v1.partial_reprefill
        num_warmup_batches = cfg.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches (partial_reprefill)")

    def on_sampled(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    @auto_await
    async def _reprefill_all(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix=f"partial_reprefill_{self.global_steps}",
        )

    def _compute_new_rollout_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        cfg = self.config.trainer.v1.partial_reprefill
        current_version = self.global_steps - 1
        resume_version = current_version

        # Pre-fetch rollout_log_probs + token_versions for all keys (case 3 needs
        # rollout_log_probs; case dispatch needs token_versions[-1]).
        fields = ["rollout_log_probs", "token_versions"]
        meta = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=fields,
        )

        # Determine which trajectories need full reprefill (case 2).
        case2_indices: list[int] = []
        case3_indices: list[int] = []
        case1_count = 0
        for i, tag in enumerate(batch.tags):
            tag = tag or {}
            piggyback = bool(tag.get("piggyback_marker", False)) and cfg.enable_piggyback
            # token_versions is a nested-jagged int32 tensor; last element of trajectory i.
            tv = meta["token_versions"][i] if "token_versions" in meta else None
            last_tv = int(tv[-1].item()) if tv is not None and len(tv) > 0 else None
            case = decide_case(
                piggyback_marker=piggyback,
                last_token_version=last_tv,
                current_parameter_version=current_version,
                enable_case_skip=cfg.enable_case_skip,
                enable_piggyback=cfg.enable_piggyback,
            )
            if case == 1:
                case1_count += 1
            elif case == 3:
                case3_indices.append(i)
            else:
                case2_indices.append(i)

        # Case 2: full reprefill for selected keys.
        if case2_indices:
            keys2 = [batch.keys[i] for i in case2_indices]
            prompt_ids_list, real_lens, _ = build_reprefill_inputs(
                keys=keys2, partition_id=batch.partition_id,
                pad_id=self.tokenizer.pad_token_id,
            )
            results = self._reprefill_all(prompt_ids_list)
            nested = [
                slice_response_logprobs(
                    results[j].extra_fields["prompt_logprobs"],
                    real_lens[j][0], real_lens[j][1],
                )
                for j in range(len(keys2))
            ]
            tq.kv_batch_put(
                keys=keys2, partition_id=batch.partition_id,
                fields={"new_rollout_log_probs": to_nested_jagged(nested)},
            )
            for j, i in enumerate(case2_indices):
                if batch.tags[i] is None:
                    batch.tags[i] = {}
                batch.tags[i]["resume_version"] = int(resume_version)

        # Case 3: copy rollout_log_probs → new_rollout_log_probs.
        if case3_indices:
            keys3 = [batch.keys[i] for i in case3_indices]
            rollout_lp = meta["rollout_log_probs"]
            nested3 = [rollout_lp[i] for i in case3_indices]
            tq.kv_batch_put(
                keys=keys3, partition_id=batch.partition_id,
                fields={"new_rollout_log_probs": to_nested_jagged(nested3)},
            )
            for i in case3_indices:
                if batch.tags[i] is None:
                    batch.tags[i] = {}
                batch.tags[i]["resume_version"] = int(resume_version)

        # Case 1: nothing to do (client already wrote new_rollout_log_probs).

        metrics["partial_reprefill/case_distribution.case_1"] = float(case1_count)
        metrics["partial_reprefill/case_distribution.case_2"] = float(len(case2_indices))
        metrics["partial_reprefill/case_distribution.case_3"] = float(len(case3_indices))
        metrics["partial_reprefill/resume_version"] = float(resume_version)
        return batch

    def _compute_old_log_prob(self, batch, metrics: dict):
        # Identical to reprefill_decoupled: old_log_probs = new_rollout_log_probs.
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass:
            return super()._compute_old_log_prob(batch, metrics)
        compare = self.config.trainer.v1.partial_reprefill.get("compare_trainer_old_log_prob", False)
        if compare:
            batch = super()._compute_old_log_prob(batch, metrics)
            metrics["partial_reprefill/trainer_old_log_prob_computed"] = 1.0
        select_fields = ["new_rollout_log_probs"] + (["old_log_probs"] if compare else [])
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=select_fields,
        )
        if compare:
            data["trainer_old_log_probs"] = data.pop("old_log_probs")
        data["old_log_probs"] = data.pop("new_rollout_log_probs")
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
        metrics["partial_reprefill/old_log_prob_source"] = 1.0
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        from verl.trainer.ppo.v1.reprefill_utils import compute_and_emit_staleness_metrics
        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
        return super()._compute_advantage(batch, metrics)

    def _debug_log_prob_extra_fields(self) -> list[str]:
        return ["new_rollout_log_probs"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py::TestComputeNewRolloutLogProbCase3 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_partial_reprefill.py \
        tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py
git commit -m "feat: implement partial_reprefill trainer with case 3 skip logic"
```

---

## Phase B: Client-side partial_rollout piggyback

### Task B1: SGLang behavior verification spike

**Files:**
- Test (throwaway): `tests/workers/rollout/test_sglang_prompt_logprobs_on_resume.py`

This is a spike — not kept in the final codebase. Goal: verify SGLang emits `prompt_logprobs` during a `generate` call where `max_new_tokens > 0` (the partial_rollout resume path), not just when `max_new_tokens == 0` (the standalone reprefill path).

- [ ] **Step 1: Write the spike test**

```python
# tests/workers/rollout/test_sglang_prompt_logprobs_on_resume.py
"""Spike: verify SGLang emits prompt_logprobs during resume (max_new_tokens>0)."""

import pytest
from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer


@pytest.mark.asyncio
async def test_resume_emits_prompt_logprobs():
    # This test requires a running SGLang server; skip if not available.
    pytest.skip("requires running SGLang server; run manually for verification")
    server = AsyncSglangServer(...)
    # 1. Submit a generate call with prompt_logprobs=0, max_new_tokens=10.
    # 2. Verify result.extra_fields["prompt_logprobs"] has length == prompt_len.
    # 3. Submit a SECOND generate call (simulating resume) with
    #    prompt = original_prompt + first_output_tokens, prompt_logprobs=0,
    #    max_new_tokens=10.
    # 4. Verify the second result's prompt_logprobs has length ==
    #    len(original_prompt + first_output_tokens).
    # If SGLang does NOT emit prompt_logprobs when max_new_tokens>0, the
    # piggyback design must fall back to a separate reprefill call after
    # resume completes (see §9 open question in the spec).
```

- [ ] **Step 2: Run the spike manually against a local SGLang server**

Document the result in the commit message. If SGLang emits prompt_logprobs during resume → proceed with Phase B. If not → adjust the design: issue a separate `reprefill_trajectories` call for the prefix AFTER the resume completes, accepting the duplicate prefill cost (still saves the suffix re-prefill).

- [ ] **Step 3: Commit the spike finding (delete the throwaway test)**

```bash
git rm tests/workers/rollout/test_sglang_prompt_logprobs_on_resume.py
# Commit the design note in the spec's §9 resolution
git commit -m "docs: resolve SGLang prompt_logprobs-on-resume open question

Result: [EMITS / DOES NOT EMIT] prompt_logprobs when max_new_tokens>0.
[If DOES NOT: piggyback falls back to separate post-resume reprefill call.]"
```

---

### Task B2: `build_token_versions` + `build_partial_new_rollout_log_probs` helpers

**Files:**
- Modify: `verl/trainer/ppo/v1/reprefill_utils.py`
- Test: `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py` (extend)

**Interfaces:**
- Produces: `build_token_versions(segment_versions: list[int], segment_lengths: list[int]) -> torch.Tensor` — per-token int32 jagged.
- Produces: `build_partial_new_rollout_log_probs(prefix_prompt_logprobs: list[float], suffix_rollout_log_probs: list[float]) -> list[float]` — concatenate prefix + suffix into one trajectory's new_rollout_log_probs.

- [ ] **Step 1: Write the failing tests (append to test file)**

```python
# Append to tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py

from verl.trainer.ppo.v1.reprefill_utils import (
    build_token_versions,
    build_partial_new_rollout_log_probs,
)


class TestBuildTokenVersions:
    def test_single_segment(self):
        tv = build_token_versions(segment_versions=[5], segment_lengths=[4])
        assert tv.tolist() == [5, 5, 5, 5]

    def test_two_segments_partial_rollout(self):
        # prefix decoded at W_3 (len 4), suffix decoded at W_5 (len 3)
        tv = build_token_versions(
            segment_versions=[3, 5], segment_lengths=[4, 3]
        )
        assert tv.tolist() == [3, 3, 3, 3, 5, 5, 5]

    def test_three_segments_multi_interrupt(self):
        tv = build_token_versions(
            segment_versions=[3, 4, 5], segment_lengths=[2, 2, 2]
        )
        assert tv.tolist() == [3, 3, 4, 4, 5, 5]


class TestBuildPartialNewRolloutLogProbs:
    def test_concatenation(self):
        # prefix logprobs (from resume prefill prompt_logprobs) + suffix
        # rollout_log_probs (decode logprob at W_resume — copy is equivalent)
        result = build_partial_new_rollout_log_probs(
            prefix_prompt_logprobs=[-0.1, -0.2, -0.3],
            suffix_rollout_log_probs=[-0.4, -0.5],
        )
        assert result == [-0.1, -0.2, -0.3, -0.4, -0.5]

    def test_empty_prefix(self):
        # No prefix (single-segment trajectory — but this path shouldn't be
        # hit for single-segment; test for completeness).
        result = build_partial_new_rollout_log_probs(
            prefix_prompt_logprobs=[],
            suffix_rollout_log_probs=[-0.1, -0.2],
        )
        assert result == [-0.1, -0.2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_token_versions'`

- [ ] **Step 3: Implement the helpers**

Append to `verl/trainer/ppo/v1/reprefill_utils.py`:

```python
def build_token_versions(segment_versions: list[int], segment_lengths: list[int]) -> torch.Tensor:
    """Expand per-segment version into per-token int32 1D tensor.

    Used by client to record each token's decode-time global_steps.
    """
    assert len(segment_versions) == len(segment_lengths)
    tokens: list[int] = []
    for v, n in zip(segment_versions, segment_lengths, strict=True):
        tokens.extend([int(v)] * n)
    return torch.tensor(tokens, dtype=torch.int32)


def build_partial_new_rollout_log_probs(
    prefix_prompt_logprobs: list[float],
    suffix_rollout_log_probs: list[float],
) -> list[float]:
    """Concatenate prefix (re-prefilled at W_resume) + suffix (decode logprob
    at W_resume, copied as-is) into one trajectory's new_rollout_log_probs.

    For partial_rollout trajectories: prefix tokens were decoded at W_prefix
    and re-prefilled at W_resume; suffix tokens were decoded at W_resume, so
    their rollout_log_probs IS the W_resume logprob — no re-prefill needed.
    """
    return list(prefix_prompt_logprobs) + list(suffix_rollout_log_probs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/reprefill_utils.py \
        tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py
git commit -m "feat: add token_versions + partial new_rollout_log_probs builders"
```

---

### Task B3: Modify `FullyAsyncLLMServerClient.generate` to emit piggyback

**Files:**
- Modify: `verl/workers/rollout/llm_server.py:355-424` (partial_rollout loop)
- Test: `tests/workers/rollout/test_llm_server_partial_piggyback.py` (create)

**Interfaces:**
- Produces: `FullyAsyncLLMServerClient.generate` writes `extra_fields["token_versions"]` (1D int32 tensor) and `extra_fields["new_rollout_log_probs"]` (list[float] or nested-jagged) for partial trajectories. Sets a sentinel `extra_fields["piggyback_marker"] = True` when piggyback was successfully produced.
- Consumes: `build_token_versions`, `build_partial_new_rollout_log_probs`, `slice_response_logprobs` from `reprefill_utils`.

**Note on testing:** The full `generate` method is hard to unit-test (requires a live SGLang server). Test the **extracted helper** that builds piggyback fields from a list of segment outputs — this is the unit-testable piece. The `generate` method itself is integration-tested separately.

- [ ] **Step 1: Extract the piggyback-builder as a helper, write the failing test**

Create `tests/workers/rollout/test_llm_server_partial_piggyback.py`:

```python
# Apache-2.0 license header
"""CPU tests for the partial_rollout piggyback field builder."""

import pytest
import torch
from types import SimpleNamespace

from verl.workers.rollout.llm_server import _build_piggyback_fields


def _seg(token_ids, log_probs, prompt_logprobs, global_steps):
    """Make a fake segment output."""
    return SimpleNamespace(
        token_ids=token_ids,
        log_probs=log_probs,
        stop_reason="length",
        extra_fields={
            "prompt_logprobs": prompt_logprobs,
            "global_steps": global_steps,
        },
    )


class TestBuildPiggybackFields:
    def test_single_segment_no_piggyback(self):
        # Single-segment trajectory — no prefix to piggyback. Returns
        # piggyback_marker=False; no new_rollout_log_probs.
        segs = [_seg(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3],
                     prompt_logprobs=None, global_steps=5)]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        # token_versions still built
        assert result["token_versions"].tolist() == [5, 5, 5]

    def test_two_segment_partial_piggyback(self):
        # Prefix @ W_3 (len 2), suffix @ W_5 (len 3). Resume at W_5
        # re-prefilled prefix, emitted prompt_logprobs for prompt +
        # prefix (length = prompt_len + prefix_len = 2 + 2 = 4).
        prefix_prompt_logprobs = [[-1.0], [-0.5], [-0.4], [-0.3]]  # 4 entries
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8],
                 prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13, 14], log_probs=[-0.2, -0.15, -0.1],
                 prompt_logprobs=prefix_prompt_logprobs, global_steps=5),
        ]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is True
        # new_rollout = prefix_prompt_logprobs[1:] (skip prompt) + suffix log_probs
        # = [-0.5, -0.4] (prefix re-prefilled) + [-0.2, -0.15, -0.1] (suffix copy)
        assert result["new_rollout_log_probs"] == [-0.5, -0.4, -0.2, -0.15, -0.1]
        assert result["token_versions"].tolist() == [3, 3, 5, 5, 5]
        assert result["resume_version"] == 5

    def test_missing_prompt_logprobs_no_piggyback(self):
        # Resume didn't emit prompt_logprobs (e.g. SGLang error). Fall back:
        # no piggyback marker, only token_versions built.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8],
                 prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15],
                 prompt_logprobs=None, global_steps=5),
        ]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        assert result["token_versions"].tolist() == [3, 3, 5, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/workers/rollout/test_llm_server_partial_piggyback.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_piggyback_fields'`

- [ ] **Step 3: Implement the helper + wire into `generate`**

Add to `verl/workers/rollout/llm_server.py` (top-level function near `FullyAsyncLLMServerClient`):

```python
def _build_piggyback_fields(segments: list, prompt_len: int) -> dict:
    """Build piggyback fields from a partial_rollout's segment outputs.

    Called after the partial_rollout loop completes. `segments` is the list
    of per-segment outputs from super().generate() calls. The LAST segment's
    prompt_logprobs (if present) is the resume-prefill result covering
    `prompt + cumulative_prefix` at the final W_resume.

    Returns a dict with:
      - token_versions: per-token int32 1D tensor (always present).
      - piggyback_marker: bool (True only if prefix was successfully re-prefilled).
      - new_rollout_log_probs: list[float] (only if piggyback_marker=True).
      - resume_version: int (only if piggyback_marker=True).
    """
    from verl.trainer.ppo.v1.reprefill_utils import (
        build_token_versions,
        build_partial_new_rollout_log_probs,
        slice_response_logprobs,
    )

    segment_versions = [int(s.extra_fields.get("global_steps", 0)) for s in segments]
    segment_lengths = [len(s.token_ids) for s in segments]
    token_versions = build_token_versions(segment_versions, segment_lengths)

    result = {"token_versions": token_versions}

    # Piggyback only when: ≥2 segments AND last segment emitted prompt_logprobs.
    if len(segments) < 2:
        return result
    last_seg = segments[-1]
    last_pl = last_seg.extra_fields.get("prompt_logprobs")
    if last_pl is None:
        return result

    # Cumulative prefix length = total tokens before the last segment.
    prefix_len = sum(len(s.token_ids) for s in segments[:-1])
    # last_pl has length prompt_len + prefix_len (SGLang emits one logprob per
    # prompt token when prompt_logprobs is set). Slice out the prefix portion.
    prefix_prompt_logprobs = slice_response_logprobs(last_pl, prompt_len, prefix_len)
    # Suffix = last segment's decode log_probs (already at W_resume).
    suffix_rollout_logprobs = [
        float(x) for x in (last_seg.log_probs or [])
    ]
    new_rollout = build_partial_new_rollout_log_probs(
        prefix_prompt_logprobs, suffix_rollout_logprobs
    )
    result["piggyback_marker"] = True
    result["new_rollout_log_probs"] = new_rollout
    result["resume_version"] = segment_versions[-1]
    return result
```

Wire it into `generate` — modify the loop at `llm_server.py:358-424`. After the while-loop terminates (before `return final_output`), add:

```python
    # Build piggyback fields from segment outputs (partial_rollout).
    # segments list is collected by appending each `output` from the loop above.
    # To collect: declare `segments = []` before the while-loop and `segments.append(output)`
    # inside the loop. (See step 4 below for the exact diff.)
    piggyback = _build_piggyback_fields(segments, prompt_len=len(prompt_ids))
    final_output.extra_fields["token_versions"] = piggyback["token_versions"]
    if piggyback.get("piggyback_marker"):
        final_output.extra_fields["piggyback_marker"] = True
        final_output.extra_fields["new_rollout_log_probs"] = piggyback["new_rollout_log_probs"]
        final_output.extra_fields["resume_version"] = piggyback["resume_version"]
```

Also request `prompt_logprobs` during resume: inside the while-loop, when `output.stop_reason in ("aborted", "abort")` and `should_retry` is True, override the sampling_params for the next iteration to include `prompt_logprobs=0`:

```python
    if output.stop_reason in ("aborted", "abort") and should_retry:
        # Resume path: request prompt_logprobs so the resume prefill emits
        # logprobs for the cumulative prefix (piggyback target).
        sampling_params = {**sampling_params, "prompt_logprobs": 0}
        await asyncio.sleep(1)
        continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/workers/rollout/test_llm_server_partial_piggyback.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add verl/workers/rollout/llm_server.py \
        tests/workers/rollout/test_llm_server_partial_piggyback.py
git commit -m "feat: emit token_versions + piggyback new_rollout_log_probs from partial_rollout"
```

---

### Task B4: Propagate client fields through agent_loop_tq to TQ

**Files:**
- Modify: `verl/trainer/ppo/v1/agent_loop_tq.py:182-227` (the `_agent_loop_postprocess` field-put loop)

**Goal:** Ensure `extra_fields["token_versions"]`, `extra_fields["new_rollout_log_probs"]`, `extra_fields["piggyback_marker"]`, `extra_fields["resume_version"]` flow from `TokenOutput` into TQ alongside the existing `rollout_log_probs` field.

- [ ] **Step 1: Inspect the current field-put loop in `agent_loop_tq.py`**

Run: `sed -n '180,230p' verl/trainer/ppo/v1/agent_loop_tq.py`

Identify where `rollout_log_probs` is built from `output.log_probs` / `response_logprobs` and put to TQ. The new fields need to be put in the same loop.

- [ ] **Step 2: Add field puts**

In the same `_agent_loop_postprocess` loop where `rollout_log_probs` is put to TQ, add (per trajectory):

```python
    extra = output.extra_fields or {}
    fields_to_put = {}
    if "token_versions" in extra:
        fields_to_put["token_versions"] = extra["token_versions"]
    if "new_rollout_log_probs" in extra:
        # Already a list[float] from the client; convert to nested-jagged for TQ.
        fields_to_put["new_rollout_log_probs"] = to_nested_jagged([extra["new_rollout_log_probs"]])
    if "piggyback_marker" in extra and extra["piggyback_marker"]:
        # Tags carry the marker (per-trajectory scalar, not per-token).
        tag["piggyback_marker"] = True
        tag["resume_version"] = int(extra.get("resume_version", 0))
    if fields_to_put:
        tq.kv_batch_put(
            keys=[traj_key], partition_id=partition_id, fields=fields_to_put,
        )
```

Exact variable names (`output`, `traj_key`, `tag`) must match the surrounding code — read the file first.

- [ ] **Step 3: Write an integration-style test (or extend the existing replay_buffer test)**

Add to `tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py`:

```python
def test_partial_rollout_fields_propagate_to_tq(tq_init, partition_id):
    # Simulate a TokenOutput with piggyback fields; run through
    # _agent_loop_postprocess; verify TQ has the fields.
    # ... (construct a minimal TokenOutput, call _agent_loop_postprocess,
    #      then tq.kv_batch_get and assert fields present)
    pass  # TODO: fill in based on existing test patterns in this file
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py -v
git add verl/trainer/ppo/v1/agent_loop_tq.py \
        tests/trainer/ppo/v1/test_reprefill_replay_buffer_on_cpu.py
git commit -m "feat: propagate partial_rollout piggyback fields through agent_loop_tq to TQ"
```

---

## Phase C: Trainer case 1 consumption + token-level metrics

### Task C1: Extend trainer tests to cover case 1 + case 2

**Files:**
- Modify: `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py` (extend)

- [ ] **Step 1: Write failing tests for case 1 and case 2**

Append to `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py`:

```python
class TestComputeNewRolloutLogProbCase1:
    def test_case1_skips_reprefill_when_marker_set(self, tq_init, partition_id):
        # Client already wrote new_rollout_log_probs + piggyback marker.
        trainer = _make_trainer()
        key = f"traj-{uuid.uuid4().hex}"
        expected_lp = [-0.1, -0.2, -0.3]
        tq.kv_batch_put(
            keys=[key], partition_id=partition_id,
            fields={"new_rollout_log_probs": to_nested_jagged([expected_lp])},
        )
        batch = _make_batch(
            partition_id, [key],
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
            keys=[key], partition_id=partition_id,
            fields={
                "prompts": to_nested_jagged([[1, 2]]),
                "responses": to_nested_jagged([[10, 11, 12]]),
                "token_versions": torch.nested.as_nested_tensor(
                    [torch.tensor([3, 3, 3], dtype=torch.int32)],
                    layout=torch.jagged,
                ),
            },
        )
        batch = _make_batch(partition_id, [key])
        metrics = {}

        # Stub _reprefill_all to return a fake result with prompt_logprobs.
        fake_result = SimpleNamespace(
            extra_fields={"prompt_logprobs": [[-0.0], [-0.1], [-0.2], [-0.3], [-0.4]]}
        )
        trainer._reprefill_all = lambda _: [fake_result]
        # Stub tokenizer.pad_token_id
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)

        out = trainer._compute_new_rollout_log_prob(batch, metrics)
        assert metrics["partial_reprefill/case_distribution.case_2"] == 1.0
        data = tq.kv_batch_get(
            keys=[key], partition_id=partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        # prompt_len=2, response_len=3 → slice [1:4] → [-0.1, -0.2, -0.3]
        assert data["new_rollout_log_probs"][0].tolist() == [-0.1, -0.2, -0.3]
```

- [ ] **Step 2: Run tests — case 1 + case 2 should pass (already implemented in A3)**

Run: `pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py -v`
Expected: PASS (A3's `_compute_new_rollout_log_prob` already handles all three cases)

- [ ] **Step 3: Commit tests (no impl change — coverage extension)**

```bash
git add tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py
git commit -m "test: cover case 1 (piggyback skip) and case 2 (full reprefill) in partial_reprefill"
```

---

### Task C2: Token-level metrics function

**Files:**
- Modify: `verl/trainer/ppo/v1/reprefill_utils.py`
- Test: `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py` (extend)

**Interfaces:**
- Produces: `compute_and_emit_token_staleness_metrics(batch, metrics, global_steps)` — emits `offpolicy_token/*` metrics using per-token `token_versions` and the three logprob fields.

- [ ] **Step 1: Write the failing test**

Append to `tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py`:

```python
from verl.trainer.ppo.v1.reprefill_utils import compute_and_emit_token_staleness_metrics


class TestComputeTokenStalenessMetrics:
    def test_emits_all_expected_keys(self, tq_init, partition_id):
        # Two trajectories: one fully fresh (token_versions all = current),
        # one fully stale (all old).
        keys = [f"traj-{i}-{uuid.uuid4().hex}" for i in range(2)]
        # rollout == new_rollout for fresh; differ for stale.
        tq.kv_batch_put(
            keys=keys, partition_id=partition_id,
            fields={
                "rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.5, -0.6]]),
                "new_rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                "old_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                "response_mask": to_nested_jagged([[1.0, 1.0], [1.0, 1.0]]),
                "token_versions": torch.nested.as_nested_tensor(
                    [
                        torch.tensor([5, 5], dtype=torch.int32),
                        torch.tensor([3, 3], dtype=torch.int32),
                    ],
                    layout=torch.jagged,
                ),
            },
        )
        from transfer_queue import KVBatchMeta
        batch = KVBatchMeta(
            keys=keys, partition_id=partition_id,
            tags=[{"resume_version": 5}, {"resume_version": 5}],
        )
        metrics = {}
        compute_and_emit_token_staleness_metrics(batch, metrics, global_steps=6)
        assert "offpolicy_token/staleness_mean" in metrics
        assert "offpolicy_token/fresh_token_ratio" in metrics
        assert "offpolicy_token/stale_token_ratio" in metrics
        # 2 fresh tokens (traj 0) + 2 stale (traj 1)
        assert metrics["offpolicy_token/fresh_token_ratio"] == 0.5
        assert metrics["offpolicy_token/stale_token_ratio"] == 0.5
        # staleness_mean: only stale tokens contribute; diff = |(-0.5)-(-0.7)| + |(-0.6)-(-0.8)| = 0.2+0.2 = 0.4; mean = 0.2
        assert abs(metrics["offpolicy_token/staleness_mean"] - 0.2) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py::TestComputeTokenStalenessMetrics -v`
Expected: FAIL with `ImportError: cannot import name 'compute_and_emit_token_staleness_metrics'`

- [ ] **Step 3: Implement the function**

Append to `verl/trainer/ppo/v1/reprefill_utils.py`:

```python
def compute_and_emit_token_staleness_metrics(batch, metrics, global_steps):
    """Emit per-token staleness metrics using token_versions + the three logprobs.

    Metrics (all under `offpolicy_token/`):
      - staleness_mean: per-token mean of |log(π_rollout) - log(π_new_rollout)|
        over tokens where token_versions[i] < global_steps - 1.
      - fresh_token_ratio: fraction of tokens with version == global_steps - 1.
      - stale_token_ratio: complement.
      - staleness_by_version_gap_0: ratio of tokens with version gap 0.
      - staleness_by_version_gap_1: ratio with gap 1.
      - staleness_by_version_gap_2_3: ratio with gap 2-3.
      - staleness_by_version_gap_4plus: ratio with gap ≥4.
    """
    fields = ["rollout_log_probs", "new_rollout_log_probs", "old_log_probs",
              "response_mask", "token_versions"]
    try:
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=fields,
        )
    except Exception as e:
        logger.warning(f"token staleness metrics: failed to fetch: {e}")
        return

    current_version = global_steps - 1
    total_tokens = 0
    fresh_tokens = 0
    gap_buckets = {0: 0, 1: 0, "2_3": 0, "4plus": 0}
    staleness_diffs: list[float] = []

    for i in range(len(batch.keys)):
        rollout = data["rollout_log_probs"][i].tolist()
        new_rollout = data["new_rollout_log_probs"][i].tolist()
        mask = data["response_mask"][i].tolist()
        tv = data["token_versions"][i].tolist() if "token_versions" in data else None
        for j in range(len(rollout)):
            if mask[j] == 0:
                continue
            total_tokens += 1
            gap = current_version - (tv[j] if tv is not None else current_version)
            if gap == 0:
                fresh_tokens += 1
                gap_buckets[0] += 1
            elif gap == 1:
                gap_buckets[1] += 1
            elif gap <= 3:
                gap_buckets["2_3"] += 1
            else:
                gap_buckets["4plus"] += 1
            if gap > 0:
                staleness_diffs.append(abs(rollout[j] - new_rollout[j]))

    if total_tokens == 0:
        return
    metrics["offpolicy_token/fresh_token_ratio"] = fresh_tokens / total_tokens
    metrics["offpolicy_token/stale_token_ratio"] = 1.0 - fresh_tokens / total_tokens
    metrics["offpolicy_token/staleness_mean"] = (
        sum(staleness_diffs) / len(staleness_diffs) if staleness_diffs else 0.0
    )
    metrics["offpolicy_token/staleness_by_version_gap_0"] = gap_buckets[0] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_1"] = gap_buckets[1] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_2_3"] = gap_buckets["2_3"] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_4plus"] = gap_buckets["4plus"] / total_tokens
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py::TestComputeTokenStalenessMetrics -v`
Expected: PASS

- [ ] **Step 5: Wire into trainer `_compute_advantage`**

In `verl/trainer/ppo/v1/trainer_partial_reprefill.py`, update `_compute_advantage`:

```python
    def _compute_advantage(self, batch, metrics: dict):
        from verl.trainer.ppo.v1.reprefill_utils import (
            compute_and_emit_staleness_metrics,
            compute_and_emit_token_staleness_metrics,
        )
        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
        compute_and_emit_token_staleness_metrics(batch, metrics, self.global_steps)
        return super()._compute_advantage(batch, metrics)
```

- [ ] **Step 6: Commit**

```bash
git add verl/trainer/ppo/v1/reprefill_utils.py \
        verl/trainer/ppo/v1/trainer_partial_reprefill.py \
        tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py
git commit -m "feat: add per-token staleness metrics alongside per-trajectory metrics"
```

---

## Phase D: P2 pipelined pre-dispatch with case awareness

### Task D1: Duplicate `_PrefillDispatcher` + `_on_new_finished` in new trainer

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_partial_reprefill.py`

**Goal:** Port the P2 pipelined path from `reprefill_decoupled.py:60-333` into `partial_reprefill`, but make the pre-dispatch case-aware: skip case 1 (piggyback marker) and case 3 (fully fresh) trajectories; only pre-dispatch case 2.

- [ ] **Step 1: Write failing test for case-aware pre-dispatch**

Append to `tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py`:

```python
class TestOnNewFinishedCaseAware:
    def test_skips_case1_and_case3_only_dispatches_case2(
        self, tq_init, partition_id, monkeypatch
    ):
        trainer = _make_trainer()
        trainer._prefill_dispatcher = SimpleNamespace(
            submit=lambda coro: concurrent.futures.Future(),
        )
        trainer._pending_prefill = {}
        trainer.replay_buffer = SimpleNamespace(
            prompt_global_steps={partition_id: {}},
            partitions={
                partition_id: {
                    "uid_a_0_0": None,  # case 2 (stale)
                    "uid_b_0_0": None,  # case 1 (piggyback)
                    "uid_c_0_0": None,  # case 3 (fresh)
                }
            },
        )
        # Stub build_reprefill_inputs / decide_case via monkeypatch
        dispatched_keys = []
        # ... (set up tq fields for each trajectory: token_versions, piggyback_marker)
        # ... call trainer._on_new_finished(partition_id, ["uid_a", "uid_b", "uid_c"])
        # ... assert only "uid_a_0_0" is in trainer._pending_prefill
```

Fill in the test body with concrete TQ setup before running.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py::TestOnNewFinishedCaseAware -v`
Expected: FAIL with `AttributeError: 'PPOTrainerPartialReprefill' object has no attribute '_on_new_finished'`

- [ ] **Step 3: Port `_PrefillDispatcher`, `_PendingPrefill`, `on_train_begin` (P2 branch), `on_train_end`, `_on_new_finished`, `_cancel_pending_prefills`, `_compute_new_rollout_log_prob_pipelined` from `trainer_reprefill_decoupled.py`**

Copy verbatim, then modify `_on_new_finished` to skip case 1 + case 3:

```python
    def _on_new_finished(self, partition_id, new_keys):
        if partition_id != "train":
            return
        cfg = self.config.trainer.v1.partial_reprefill
        current_version = self.global_steps - 1
        train_batch_size = self.config.data.train_batch_size
        prompt_global_steps = self.replay_buffer.prompt_global_steps.get(partition_id, {})
        ordered_uids = sorted(new_keys, key=lambda u: prompt_global_steps.get(u, 0))
        bounded_uids = ordered_uids[:train_batch_size] if train_batch_size > 0 else ordered_uids

        partition = self.replay_buffer.partitions.get(partition_id, {})
        for uid in bounded_uids:
            traj_keys = [k for k in partition if k.split("_")[0] == uid]
            for traj_key in traj_keys:
                if traj_key in self._pending_prefill:
                    continue
                # Case-awareness: fetch tags + token_versions for this traj_key
                meta = tq.kv_batch_get(
                    keys=[traj_key], partition_id=partition_id,
                    select_fields=["token_versions"],
                )
                tag_dict = self.replay_buffer.tags.get(partition_id, {}).get(traj_key, {})
                piggyback = bool(tag_dict.get("piggyback_marker", False)) and cfg.enable_piggyback
                tv = meta.get("token_versions")
                last_tv = int(tv[0][-1].item()) if tv is not None and len(tv[0]) > 0 else None
                case = decide_case(
                    piggyback_marker=piggyback,
                    last_token_version=last_tv,
                    current_parameter_version=current_version,
                    enable_case_skip=cfg.enable_case_skip,
                    enable_piggyback=cfg.enable_piggyback,
                )
                if case != 2:
                    continue  # only pre-dispatch case 2
                try:
                    prompt_ids_list, _, _ = build_reprefill_inputs(
                        keys=[traj_key], partition_id=partition_id,
                        pad_id=self.tokenizer.pad_token_id,
                    )
                    future = self._prefill_dispatcher.submit(
                        reprefill_trajectories(
                            client=self.get_llm_client(),
                            prompt_ids_list=prompt_ids_list,
                            request_prefix=f"partial_reprefill_p2_{self.global_steps}_{traj_key}",
                        )
                    )
                    self._pending_prefill[traj_key] = _PendingPrefill(
                        version=self.global_steps, future=future,
                    )
                except Exception as e:
                    logger.warning(f"partial_reprefill: pre-dispatch failed for {traj_key}: {e}")
```

Also update `_compute_new_rollout_log_prob_pipelined` to integrate with the case-decision dispatcher from A3: case 2 keys consume pre-dispatched futures (with re-issue fallback); case 1 + case 3 take the fast path (no future lookup). Simplest approach: keep A3's dispatcher as the entry, and inside the case 2 branch, prefer pre-dispatched futures when available, falling back to synchronous `_reprefill_all` for re-issues.

- [ ] **Step 4: Run tests**

Run: `pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_partial_reprefill.py \
        tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py
git commit -m "feat: P2 pipelined pre-dispatch with case-aware skip for partial_reprefill"
```

---

### Task D2: Enable `enable_prefill_pipeline` flag wiring in `on_train_begin`/`on_train_end`

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_partial_reprefill.py`

- [ ] **Step 1: Update `on_train_begin` to start dispatcher when `enable_prefill_pipeline=true`**

```python
    def on_train_begin(self):
        cfg = self.config.trainer.v1.partial_reprefill
        for _ in range(cfg.num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {cfg.num_warmup_batches} warmup batches (partial_reprefill)")
        if cfg.get("enable_prefill_pipeline", False):
            self._prefill_dispatcher = _PrefillDispatcher()
            self._prefill_dispatcher.start()
            self._pending_prefill: dict[str, _PendingPrefill] = {}
            self.replay_buffer.set_on_new_finished_callback(self._on_new_finished)
            logger.info("partial_reprefill: pipelined pre-dispatch enabled")

    def on_train_end(self):
        dispatcher = getattr(self, "_prefill_dispatcher", None)
        if dispatcher is not None:
            self._cancel_pending_prefills(reason="on_train_end")
            dispatcher.shutdown()
            logger.info("partial_reprefill: prefill dispatcher shut down on_train_end")
```

- [ ] **Step 2: Wire `on_sampled` to dispatch pipelined vs non-pipelined**

```python
    def on_sampled(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            if getattr(self, "_pending_prefill", None) is not None:
                batch = self._compute_new_rollout_log_prob_pipelined(batch, metrics)
            else:
                batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch
```

- [ ] **Step 3: Run full test suite + commit**

```bash
pytest tests/trainer/ppo/v1/test_trainer_partial_reprefill_on_cpu.py -v
pytest tests/trainer/ppo/v1/test_reprefill_utils_partial_on_cpu.py -v
pytest tests/workers/rollout/test_llm_server_partial_piggyback.py -v
git add verl/trainer/ppo/v1/trainer_partial_refill.py
git commit -m "feat: wire enable_prefill_pipeline flag in partial_reprefill trainer"
```

---

## Self-Review Checklist (run after writing all tasks)

**1. Spec coverage:**
- §2 design overview (three cases) → Tasks A3, B3, C1
- §3 TQ schema (token_versions, piggyback_marker) → Tasks B2, B3, B4
- §4.1 client changes (prompt_logprobs, merge behavior) → Task B3
- §4.2 trainer case-decision dispatcher → Task A3
- §4.3 P2 pipelined case-aware → Task D1
- §4.4 staleness metrics unchanged → Task C1 (no change to existing function)
- §4.5 new token-level metrics → Task C2
- §5 data flow walkthrough → covered by tests in A3, B3, C1
- §6 edge cases (multi-interruption, piggyback failure, stale current_parameter_version) → covered by decide_case tests (A2) + _build_piggyback_fields tests (B3)
- §7 testing strategy → all test tasks
- §8 rollout / config → Task A1
- §9 open questions → Task B1 resolves SGLang behavior

**2. Placeholder scan:** Verify no "TBD" / "TODO" / "implement later" in any task body. (D1 Step 3 has "see A3's dispatcher" — that's a reference, not a placeholder; the dispatcher IS implemented in A3.)

**3. Type consistency:**
- `decide_case` signature in A2 matches usage in A3, D1.
- `build_token_versions` returns `torch.Tensor` (per A2 tests); used in B3 `_build_piggyback_fields`.
- `build_partial_new_rollout_log_probs` returns `list[float]`; used in B3.
- `compute_and_emit_token_staleness_metrics` signature in C2 matches usage in C2 Step 5.
- `_PendingPrefill`, `_PrefillDispatcher` class names in D1 match the duplicates from `reprefill_decoupled.py:60-88`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-27-partial-reprefill.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Required sub-skill: `superpowers:subagent-driven-development`.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
