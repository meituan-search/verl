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

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout.llm_server import (
    FullyAsyncLLMServerClient,
    LLMServerClient,
    _build_piggyback_fields,
    _populate_new_rollout_fields,
)
from verl.workers.rollout.replica import TokenOutput


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
        assert len(result["new_rollout_log_probs"]) == 5  # == total generated tokens
        assert result["resume_version"] == 5

    def test_missing_prompt_logprobs_raises(self):
        # Caller gated on enable_piggyback=True and >=2 segments: the last
        # segment is a post-abort retry that carried the prompt_logprobs
        # request, so a missing prefix_prompt_logprobs violates the
        # contract — raise.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prefix_prompt_logprobs=None, global_steps=5),
        ]
        with pytest.raises(AssertionError, match="prefix_prompt_logprobs"):
            _build_piggyback_fields(segs, prompt_len=2)

    def test_short_prompt_logprobs_raises(self):
        # prefix_len=2 -> last_pl needs >= prefix_len + 1 = 3 entries; give 2.
        # Misaligned length must raise, never emit a wrong-shaped
        # new_rollout_log_probs.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prefix_prompt_logprobs=[(None, 99), (-0.5, 12)],  # 2 entries — too short
                global_steps=5,
            ),
        ]
        with pytest.raises(AssertionError, match="unexpected SGLang emission shape"):
            _build_piggyback_fields(segs, prompt_len=2)


class TestPopulateNewRolloutFields:
    @staticmethod
    def _final_output(token_ids, log_probs):
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=log_probs,
            extra_fields={},
        )

    def test_disabled_single_segment_writes_resume_version_only(self):
        # enable_piggyback=False: no new_rollout_log_probs, but resume_version
        # is still written — the trajectory's logprobs live at a single
        # version (decode version), which the trainer's case dispatch keys on.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
        ]
        final = self._final_output([10, 11], [-0.9, -0.8])
        _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=False, emit_token_versions=True)
        assert "new_rollout_log_probs" not in final.extra_fields
        assert final.extra_fields["resume_version"] == 3
        assert final.extra_fields["token_versions"] == [3, 3]

    def test_disabled_multi_segment_writes_nothing(self):
        # Resumed without piggyback: logprobs span multiple versions, so no
        # resume_version either — trainer falls it to a full reprefill.
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(token_ids=[12, 13], log_probs=[-0.2, -0.15], prefix_prompt_logprobs=None, global_steps=5),
        ]
        final = self._final_output([10, 11, 12, 13], [-0.9, -0.8, -0.2, -0.15])
        _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=False, emit_token_versions=True)
        assert "new_rollout_log_probs" not in final.extra_fields
        assert "resume_version" not in final.extra_fields
        assert final.extra_fields["token_versions"] == [3, 3, 5, 5]

    def test_single_segment_copies_rollout_log_probs(self):
        # enable_piggyback=True, never aborted (1 segment): the trajectory
        # was decoded at a single weight version, so new_rollout_log_probs
        # = rollout_log_probs (uniform all-or-nothing schema).
        segs = [_seg(token_ids=[10, 11, 12], log_probs=[-0.9, -0.8, -0.7], prefix_prompt_logprobs=None, global_steps=3)]
        final = self._final_output([10, 11, 12], [-0.9, -0.8, -0.7])
        _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=True, emit_token_versions=True)
        assert final.extra_fields["new_rollout_log_probs"] == [-0.9, -0.8, -0.7]
        assert final.extra_fields["resume_version"] == 3
        assert final.extra_fields["token_versions"] == [3, 3, 3]
        # copy, not piggyback — no marker
        assert "piggyback_marker" not in final.extra_fields

    def test_single_segment_misaligned_logprobs_raises(self):
        # enable_piggyback=True but decode log_probs are missing/misaligned —
        # contract violation, raise rather than emit a wrong-shaped copy.
        segs = [_seg(token_ids=[10, 11, 12], log_probs=[-0.9, -0.8, -0.7], prefix_prompt_logprobs=None, global_steps=3)]
        final = self._final_output([10, 11, 12], None)  # calculate_log_probs off
        with pytest.raises(AssertionError, match="aligned with token_ids"):
            _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=True, emit_token_versions=True)

    def test_multi_segment_piggyback(self):
        prefix_prompt_logprobs = [(None, 99), (-0.5, 12), (-0.4, 13)]
        segs = [
            _seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3),
            _seg(
                token_ids=[12, 13, 14],
                log_probs=[-0.2, -0.15, -0.1],
                prefix_prompt_logprobs=prefix_prompt_logprobs,
                global_steps=5,
            ),
        ]
        final = self._final_output([10, 11, 12, 13, 14], [-0.9, -0.8, -0.2, -0.15, -0.1])
        _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=True, emit_token_versions=True)
        assert final.extra_fields["piggyback_marker"] is True
        assert final.extra_fields["new_rollout_log_probs"] == [-0.5, -0.4, -0.2, -0.15, -0.1]
        assert final.extra_fields["resume_version"] == 5
        assert final.extra_fields["token_versions"] == [3, 3, 5, 5, 5]

    def test_emit_token_versions_off_skips_construction(self):
        # Trainers that don't consume token-level staleness diagnostics
        # (colocate_async, ...) pass emit_token_versions=False: no
        # token_versions field, but the piggyback/resume_version contract is
        # unchanged.
        segs = [_seg(token_ids=[10, 11], log_probs=[-0.9, -0.8], prefix_prompt_logprobs=None, global_steps=3)]
        final = self._final_output([10, 11], [-0.9, -0.8])
        _populate_new_rollout_fields(final, segs, prompt_len=2, enable_piggyback=False, emit_token_versions=False)
        assert "token_versions" not in final.extra_fields
        assert final.extra_fields["resume_version"] == 3


class TestBaseClientStripsGateKeys:
    def test_generate_strips_gate_keys(self):
        # Agent loop workers smuggle rollout-behavior gates through
        # sampling_params; only FullyAsyncLLMServerClient consumes them. The
        # base LLMServerClient must strip them, else the unknown keys reach
        # the engine's SamplingParams construction and raise TypeError
        # (sync trainers / staleness_sweep use this base client).
        client = LLMServerClient.__new__(LLMServerClient)
        client.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="sglang")))

        received = {}

        class _FakeRemote:
            def remote(self, **kwargs):
                received.update(kwargs)

                async def _ret():
                    return TokenOutput(token_ids=[1], log_probs=[-0.5], num_preempted=0)

                return _ret()

        fake_server = SimpleNamespace(generate=SimpleNamespace(remote=_FakeRemote().remote))

        async def _acquire(request_id):
            return "s0", fake_server

        client._acquire_server = _acquire
        client._release_server = lambda server_id: None

        sampling_params = {"temperature": 1.0, "top_p": 1.0, "enable_piggyback": True, "emit_token_versions": True}
        output = asyncio.run(client.generate("req-0", prompt_ids=[10, 11], sampling_params=sampling_params))
        assert output.token_ids == [1]
        assert received["sampling_params"] == {"temperature": 1.0, "top_p": 1.0}
        assert "enable_piggyback" not in received["sampling_params"]
        assert "emit_token_versions" not in received["sampling_params"]


class TestResumeTotalCap:
    @staticmethod
    def _client(response_length=20480):
        client = FullyAsyncLLMServerClient.__new__(FullyAsyncLLMServerClient)
        client.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="sglang", response_length=response_length))
        )
        client._only_hybrid = False
        return client

    def test_resume_total_capped_at_response_length(self):
        # The abort-resume loop must cap the ACCUMULATED token count at
        # rollout.response_length. Without the cap, each segment is
        # individually clamped by the engine but the total grows unbounded
        # (e.g. 20000 + 2400 = 22400 > 20480), and the piggyback
        # new_rollout_log_probs built from the raw total is longer than the
        # agent-loop-truncated response_mask — crashing the trainer's
        # offpolicy metrics with a tensor size mismatch.
        client = self._client(response_length=20480)
        calls = []

        def _fake_generate(**kwargs):
            # Snapshot: the client mutates sampling_params[limit_key] in place
            # between resume iterations (same dict object).
            calls.append({**kwargs, "sampling_params": dict(kwargs["sampling_params"])})

            async def _ret():
                nth = len(calls)
                if nth == 1:
                    return TokenOutput(
                        token_ids=[7] * 20000,
                        log_probs=[-0.1] * 20000,
                        stop_reason="aborted",
                        num_preempted=0,
                    )
                # Emulate the engine honoring max_tokens: ask for the
                # remaining budget, get exactly that.
                budget = kwargs["sampling_params"]["max_tokens"]
                return TokenOutput(
                    token_ids=[9] * budget,
                    log_probs=[-0.2] * budget,
                    stop_reason="completed",
                    num_preempted=0,
                )

            return _ret()

        fake_server = SimpleNamespace(generate=SimpleNamespace(remote=_fake_generate))

        async def _acquire(request_id):
            return "s0", fake_server

        client._acquire_server = _acquire
        client._release_server = lambda server_id: None

        out = asyncio.run(client.generate("req-0", prompt_ids=[1, 2, 3], sampling_params={"temperature": 1.0}))

        assert len(calls) == 2
        # First request carries the full budget; the resume request only the
        # remainder.
        assert calls[0]["sampling_params"]["max_tokens"] == 20480
        assert calls[1]["sampling_params"]["max_tokens"] == 480
        assert len(out.token_ids) == 20480
        assert out.stop_reason == "length"

    def test_explicit_max_tokens_respected(self):
        # A caller-provided limit key (e.g. the reprefill path's
        # max_new_tokens=0) keeps precedence over the injected cap.
        client = self._client(response_length=20480)
        calls = []

        def _fake_generate(**kwargs):
            calls.append(kwargs)

            async def _ret():
                return TokenOutput(token_ids=[], log_probs=[], stop_reason="length", num_preempted=0)

            return _ret()

        fake_server = SimpleNamespace(generate=SimpleNamespace(remote=_fake_generate))

        async def _acquire(request_id):
            return "s0", fake_server

        client._acquire_server = _acquire
        client._release_server = lambda server_id: None

        out = asyncio.run(client.generate("req-0", prompt_ids=[1, 2, 3], sampling_params={"max_new_tokens": 0}))
        assert len(calls) == 1
        assert out.token_ids == []
