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

from verl.trainer.ppo.v1.reprefill_utils import decide_case


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
