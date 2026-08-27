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

from verl.trainer.ppo.v1.reprefill_utils import (
    build_partial_new_rollout_log_probs,
    build_token_versions,
    decide_case,
)


class TestBuildTokenVersions:
    def test_single_segment(self):
        tv = build_token_versions(segment_versions=[5], segment_lengths=[4])
        assert tv.tolist() == [5, 5, 5, 5]

    def test_two_segments_partial_rollout(self):
        # prefix decoded at W_3 (len 4), suffix decoded at W_5 (len 3)
        tv = build_token_versions(segment_versions=[3, 5], segment_lengths=[4, 3])
        assert tv.tolist() == [3, 3, 3, 3, 5, 5, 5]

    def test_three_segments_multi_interrupt(self):
        tv = build_token_versions(segment_versions=[3, 4, 5], segment_lengths=[2, 2, 2])
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


class TestDecideCase:
    def test_case1_piggyback_enabled(self):
        assert (
            decide_case(
                piggyback_marker=True,
                last_token_version=5,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
            )
            == 1
        )

    def test_case1_piggyback_disabled_falls_to_case2(self):
        # piggyback disabled: even if marker set, fall through to case dispatch
        assert (
            decide_case(
                piggyback_marker=True,
                last_token_version=5,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=False,
            )
            == 2
        )

    def test_case3_fully_fresh(self):
        assert (
            decide_case(
                piggyback_marker=False,
                last_token_version=6,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
            )
            == 3
        )

    def test_case2_fully_stale(self):
        assert (
            decide_case(
                piggyback_marker=False,
                last_token_version=5,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
            )
            == 2
        )

    def test_case3_disabled_falls_to_case2(self):
        # case skip disabled: even if fresh, force case 2
        assert (
            decide_case(
                piggyback_marker=False,
                last_token_version=6,
                current_parameter_version=6,
                enable_case_skip=False,
                enable_piggyback=True,
            )
            == 2
        )

    def test_missing_last_token_version_is_case2(self):
        # token_versions not populated (e.g. older client): default to case 2
        assert (
            decide_case(
                piggyback_marker=False,
                last_token_version=None,
                current_parameter_version=6,
                enable_case_skip=True,
                enable_piggyback=True,
            )
            == 2
        )
