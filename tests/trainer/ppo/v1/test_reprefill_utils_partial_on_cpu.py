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
"""CPU tests for partial reprefill helpers."""

import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1.reprefill_utils import (
    build_partial_new_rollout_log_probs,
    build_token_versions,
    compute_and_emit_token_staleness_metrics,
    decide_case,
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


class TestBuildTokenVersions:
    def test_single_segment(self):
        tv = build_token_versions(segment_versions=[5], segment_lengths=[4])
        assert tv == [5, 5, 5, 5]

    def test_two_segments_partial_rollout(self):
        # prefix decoded at W_3 (len 4), suffix decoded at W_5 (len 3)
        tv = build_token_versions(segment_versions=[3, 5], segment_lengths=[4, 3])
        assert tv == [3, 3, 3, 3, 5, 5, 5]

    def test_three_segments_multi_interrupt(self):
        tv = build_token_versions(segment_versions=[3, 4, 5], segment_lengths=[2, 2, 2])
        assert tv == [3, 3, 4, 4, 5, 5]


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


class TestDecideCase:
    def test_case1_piggyback_within_budget(self):
        # resume_version=5, current=6, budget=1: gap within budget → consume piggyback
        assert (
            decide_case(
                piggyback_marker=True,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=5,
                max_resume_staleness=1,
            )
            == 1
        )

    def test_case1_piggyback_beyond_budget_falls_to_case2(self):
        # resume_version=3, current=6, budget=1: gap 3 > budget → refresh via reprefill
        assert (
            decide_case(
                piggyback_marker=True,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=3,
                max_resume_staleness=1,
            )
            == 2
        )

    def test_case1_piggyback_zero_gap(self):
        assert (
            decide_case(
                piggyback_marker=True,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=6,
                max_resume_staleness=0,
            )
            == 1
        )

    def test_case1_piggyback_disabled_falls_to_case3_within_budget(self):
        # piggyback disabled: marker is ignored, but a within-budget
        # resume_version still means the logprobs are fresh enough →
        # copy fast path (case 3), not a reprefill.
        assert (
            decide_case(
                piggyback_marker=True,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=False,
                resume_version=5,
                max_resume_staleness=1,
            )
            == 3
        )

    def test_case3_copy_within_budget(self):
        # single-segment trajectory whose logprobs are fresh enough
        assert (
            decide_case(
                piggyback_marker=False,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=5,
                max_resume_staleness=1,
            )
            == 3
        )

    def test_case3_copy_beyond_budget_falls_to_case2(self):
        assert (
            decide_case(
                piggyback_marker=False,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=3,
                max_resume_staleness=1,
            )
            == 2
        )

    def test_missing_resume_version_is_case2(self):
        # no resume_version (resumed without piggyback — logprobs span
        # multiple versions; or an older client): full reprefill
        assert (
            decide_case(
                piggyback_marker=False,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=None,
            )
            == 2
        )

    def test_case3_disabled_falls_to_case2(self):
        # case skip disabled: even if fresh, force case 2
        assert (
            decide_case(
                piggyback_marker=False,
                current_parameter_version=6,
                enable_case_skip=False,
                enable_piggyback=True,
                resume_version=6,
                max_resume_staleness=1,
            )
            == 2
        )

    def test_missing_resume_version_with_marker_is_case2(self):
        # piggyback marker without resume_version (inconsistent): case 2
        assert (
            decide_case(
                piggyback_marker=True,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
                resume_version=None,
            )
            == 2
        )


def _token_versions_field(per_traj_versions: list[list[int]]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(
        [torch.tensor(v, dtype=torch.int32) for v in per_traj_versions],
        layout=torch.jagged,
    )


class TestComputeTokenStalenessMetrics:
    def test_emits_all_expected_keys(self, tq_init, partition_id):
        # Two trajectories: one fully fresh (token_versions all = current),
        # one fully stale (all old).
        keys = [f"traj-{i}-{uuid.uuid4().hex}" for i in range(2)]
        # rollout == new_rollout for fresh; differ for stale.
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.5, -0.6]]),
                    "new_rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                    "old_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                    "response_mask": to_nested_jagged([[1.0, 1.0], [1.0, 1.0]]),
                    "token_versions": _token_versions_field([[5, 5], [3, 3]]),
                },
                batch_size=len(keys),
            ),
        )
        from transfer_queue import KVBatchMeta

        batch = KVBatchMeta(
            keys=keys,
            partition_id=partition_id,
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
        # version-gap buckets: traj 0 has gap 0, traj 1 has gap 2
        assert metrics["offpolicy_token/staleness_by_version_gap_0"] == 0.5
        assert metrics["offpolicy_token/staleness_by_version_gap_1"] == 0.0
        assert metrics["offpolicy_token/staleness_by_version_gap_2_3"] == 0.5
        assert metrics["offpolicy_token/staleness_by_version_gap_4plus"] == 0.0

    def test_mixed_batch_missing_token_versions_no_crash(self, tq_init, partition_id):
        # Mixed batch: one key has token_versions (case-2/3-like), one does not
        # (e.g. case-1 piggyback or a trajectory written by an older client).
        # The function must not raise; keys without usable token_versions are
        # skipped. Note: the installed transfer_queue drops a field from the
        # batch-level get result when ANY key lacks it, so in this scenario no
        # per-token stats are emitted at all — the assertion below accepts
        # either that behavior or a hypothetical future TQ that returns
        # per-key entries (in which case only the key WITH token_versions
        # contributes: 2 fresh tokens).
        key_with = f"traj-0-{uuid.uuid4().hex}"
        key_without = f"traj-1-{uuid.uuid4().hex}"
        tq.kv_batch_put(
            keys=[key_with],
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "rollout_log_probs": to_nested_jagged([[-0.1, -0.2]]),
                    "new_rollout_log_probs": to_nested_jagged([[-0.1, -0.2]]),
                    "old_log_probs": to_nested_jagged([[-0.1, -0.2]]),
                    "response_mask": to_nested_jagged([[1.0, 1.0]]),
                    "token_versions": _token_versions_field([[5, 5]]),
                },
                batch_size=1,
            ),
        )
        tq.kv_batch_put(
            keys=[key_without],
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "rollout_log_probs": to_nested_jagged([[-0.5, -0.6]]),
                    "new_rollout_log_probs": to_nested_jagged([[-0.7, -0.8]]),
                    "old_log_probs": to_nested_jagged([[-0.7, -0.8]]),
                    "response_mask": to_nested_jagged([[1.0, 1.0]]),
                },
                batch_size=1,
            ),
        )
        from transfer_queue import KVBatchMeta

        batch = KVBatchMeta(
            keys=[key_with, key_without],
            partition_id=partition_id,
            tags=[{"resume_version": 5}, {"piggyback_marker": True, "resume_version": 5}],
        )
        metrics = {}
        compute_and_emit_token_staleness_metrics(batch, metrics, global_steps=6)  # must not raise
        if "offpolicy_token/fresh_token_ratio" in metrics:
            # Only the key WITH token_versions contributes: 2 fresh tokens.
            assert metrics["offpolicy_token/fresh_token_ratio"] == 1.0
            assert metrics["offpolicy_token/stale_token_ratio"] == 0.0

    def test_length_mismatch_skips_key(self, tq_init, partition_id):
        # token_versions shorter than the response (malformed) for key 1:
        # that key's per-token stats are skipped, key 0 is still counted.
        keys = [f"traj-{i}-{uuid.uuid4().hex}" for i in range(2)]
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.5, -0.6]]),
                    "new_rollout_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                    "old_log_probs": to_nested_jagged([[-0.1, -0.2], [-0.7, -0.8]]),
                    "response_mask": to_nested_jagged([[1.0, 1.0], [1.0, 1.0]]),
                    "token_versions": _token_versions_field([[5, 5], [3]]),
                },
                batch_size=len(keys),
            ),
        )
        from transfer_queue import KVBatchMeta

        batch = KVBatchMeta(
            keys=keys,
            partition_id=partition_id,
            tags=[{"resume_version": 5}, {"resume_version": 5}],
        )
        metrics = {}
        compute_and_emit_token_staleness_metrics(batch, metrics, global_steps=6)
        # Only key 0 (2 fresh tokens) is counted.
        assert metrics["offpolicy_token/fresh_token_ratio"] == 1.0
        assert metrics["offpolicy_token/stale_token_ratio"] == 0.0
        assert metrics["offpolicy_token/staleness_mean"] == 0.0

    def test_response_mask_excludes_padded_tokens(self, tq_init, partition_id):
        # Only unmasked tokens count; masked (pad) tokens with old versions
        # must not pollute the ratios.
        keys = [f"traj-{i}-{uuid.uuid4().hex}" for i in range(1)]
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "rollout_log_probs": to_nested_jagged([[-0.1, -0.2, -0.3]]),
                    "new_rollout_log_probs": to_nested_jagged([[-0.1, -0.2, -0.3]]),
                    "old_log_probs": to_nested_jagged([[-0.1, -0.2, -0.3]]),
                    "response_mask": to_nested_jagged([[1.0, 0.0, 1.0]]),
                    "token_versions": _token_versions_field([[5, 3, 5]]),
                },
                batch_size=len(keys),
            ),
        )
        from transfer_queue import KVBatchMeta

        batch = KVBatchMeta(keys=keys, partition_id=partition_id, tags=[{"resume_version": 5}])
        metrics = {}
        compute_and_emit_token_staleness_metrics(batch, metrics, global_steps=6)
        # Masked token (version 3) is excluded → both counted tokens are fresh.
        assert metrics["offpolicy_token/fresh_token_ratio"] == 1.0
        assert metrics["offpolicy_token/stale_token_ratio"] == 0.0
        assert metrics["offpolicy_token/staleness_by_version_gap_0"] == 1.0
