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


def _seg(token_ids, log_probs, prefix_prompt_logprobs, global_steps):
    """Make a fake segment output.

    `prefix_prompt_logprobs` is the short list emitted by the SGLang adapter
    when logprob_start_len = prompt_len - 1: it has length prefix_len + 1
    (entry 0 is None per SGLang's [None] + [:-1] post-processing; entries
    [1, prefix_len + 1) are logprobs of D[0..prefix_len-1]). Each entry is a
    (logprob, token_id) tuple.
    """
    return SimpleNamespace(
        token_ids=token_ids,
        log_probs=log_probs,
        stop_reason="length",
        extra_fields={
            "prefix_prompt_logprobs": prefix_prompt_logprobs,
            "global_steps": global_steps,
        },
    )


class TestBuildPiggybackFields:
    def test_single_segment_no_piggyback(self):
        # Single-segment trajectory — no prefix to piggyback. Returns
        # piggyback_marker=False; no new_rollout_log_probs.
        segs = [_seg(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3], prefix_prompt_logprobs=None, global_steps=5)]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        # token_versions still built
        assert result["token_versions"].tolist() == [5, 5, 5]

    def test_two_segment_partial_piggyback(self):
        # Prefix @ W_3 (len 2), suffix @ W_5 (len 3). Resume at W_5
        # re-prefilled prefix; the adapter emitted prefix_prompt_logprobs
        # for the decoded prefix only (length = prefix_len + 1 = 2 + 1 = 3).
        # Entry 0 is None (SGLang [None] + [:-1] artifact); entries [1, 3)
        # are logprobs of D[0], D[1].
        prefix_prompt_logprobs = [(None, 99), (-0.5, 12), (-0.4, 13)]  # 3 entries
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prefix_prompt_logprobs=prefix_prompt_logprobs,
                global_steps=5,
            ),
        ]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is True
        # new_rollout = prefix logprobs (slice [1, prefix_len+1) of the short
        # list, taking [0] of each tuple) + suffix log_probs
        # = [-0.5, -0.4] (prefix re-prefilled) + [-0.2, -0.15, -0.1] (suffix copy)
        assert result["new_rollout_log_probs"] == [-0.5, -0.4, -0.2, -0.15, -0.1]
        assert result["token_versions"].tolist() == [3, 3, 5, 5, 5]
        assert result["resume_version"] == 5

    def test_missing_prompt_logprobs_no_piggyback(self):
        # Resume didn't emit prefix_prompt_logprobs (e.g. SGLang error). Fall back:
        # no piggyback marker, only token_versions built.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prefix_prompt_logprobs=None, global_steps=5),
        ]
        result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert "new_rollout_log_probs" not in result
        assert result["token_versions"].tolist() == [3, 3, 5, 5]

    def test_short_prompt_logprobs_no_piggyback(self, caplog):
        # M1: last segment's prefix_prompt_logprobs too short to cover the
        # prefix slice (SGLang emitted a different length than expected).
        # Must bail out with piggyback_marker=False (case-2 fallback), not
        # silently emit a wrong-shaped new_rollout_log_probs.
        # prefix_len=2 -> last_pl needs >= prefix_len + 1 = 3 entries; give 2.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prefix_prompt_logprobs=[(None, 99), (-0.5, 12)],  # 2 entries — too short
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
        # last segment has no prefix_prompt_logprobs despite >= 2 segments.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prefix_prompt_logprobs=None, global_steps=5),
        ]
        with caplog.at_level(logging.WARNING):
            result = _build_piggyback_fields(segs, prompt_len=2)
        assert result["piggyback_marker"] is False
        assert any("piggyback" in r.message.lower() for r in caplog.records)
