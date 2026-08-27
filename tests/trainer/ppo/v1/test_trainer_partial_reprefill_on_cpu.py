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
            keys=[key], partition_id=partition_id,
            fields=TensorDict(
                {"new_rollout_log_probs": to_nested_jagged([expected_lp])}, batch_size=1
            ),
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
        # (approx: to_nested_jagged stores float32, so values round-trip with
        # float32 precision, e.g. -0.1 → -0.10000000149011612)
        assert data["new_rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, -0.3])
