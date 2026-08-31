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
"""CPU tests for the partial_rollout piggyback field builder."""

import logging
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
        segs = [_seg(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3], prompt_logprobs=None, global_steps=5)]
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
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prompt_logprobs=prefix_prompt_logprobs,
                global_steps=5,
            ),
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
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prompt_logprobs=None, global_steps=5),
        ]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        assert result["token_versions"].tolist() == [3, 3, 5, 5]

    def test_short_prompt_logprobs_no_piggyback(self, caplog):
        # M1: last segment's prompt_logprobs too short to cover the prefix
        # slice (SGLang emitted a different length than the prefill input).
        # Must bail out with piggyback_marker=False (case-2 fallback), not
        # silently emit a wrong-shaped new_rollout_log_probs.
        # prompt_len=2, prefix_len=2 -> last_pl needs >= 4 entries; give 2.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prompt_logprobs=[[-1.0], [-0.5]],  # 2 entries — too short
                global_steps=5,
            ),
        ]
        with caplog.at_level(logging.WARNING):
            result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        assert "resume_version" not in result
        # token_versions still built
        assert result["token_versions"].tolist() == [3, 3, 5, 5, 5]
        assert any("piggyback" in r.message.lower() for r in caplog.records)

    def test_missing_prompt_logprobs_logs_warning(self, caplog):
        # M2: piggyback failure must be observable — warning logged when the
        # last segment has no prompt_logprobs despite >= 2 segments.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prompt_logprobs=None, global_steps=5),
        ]
        with caplog.at_level(logging.WARNING):
            result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert any("piggyback" in r.message.lower() for r in caplog.records)
