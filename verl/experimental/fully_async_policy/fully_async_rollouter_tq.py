# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""Fully Async Policy with TQ support.

Provides three components that implement TQ writing for fully-async path:

1. FullyAsyncAgentLoopWorkerTQ — Ray worker that overrides generate_sequences() to perform
   **blocking TQ write** internally, with inline TQ field preparation logic (mirrors
   AgentLoopWorkerTQ._agent_loop_postprocess but self-contained).

2. FullyAsyncAgentLoopManagerTQ — Manager using FullyAsyncAgentLoopWorkerTQ workers.

3. TQFullyAsyncRollouter — Rollouter using FullyAsyncAgentLoopManagerTQ, with
   simplified _process_single_sample_streaming() that just calls the worker and releases slot.
"""

import asyncio
import logging
import os

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from verl.experimental.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)

# Private symbols imported directly from the submodule (not re-exported in __init__.py)
from verl.experimental.agent_loop.agent_loop import (
    DictConfigWrap,
    ToolListWrap,
    _agent_loop_registry,
    _InternalAgentLoopOutput,
    rollout_trace_attr,
)
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ==========================================================================
# FullyAsyncAgentLoopWorkerTQ: Ray worker returning unpadded DataProto
# ==========================================================================


class FullyAsyncAgentLoopWorkerTQ(AgentLoopWorker):
    """Agent loop worker for fully-async TQ path.

    Key difference from base AgentLoopWorker and AgentLoopWorkerTQ:
    - Overrides generate_sequences() to perform **blocking TQ write** internally,
      with inline TQ field preparation logic (mirrors AgentLoopWorkerTQ._agent_loop_postprocess).
    - Intercepts ``AgentLoopOutput`` **before** padding (in ``_run_agent_loop_with_raw_output``)
      so we can feed unpadded ``list[int]`` into TQ.
    - Returns a minimal DataProto since all meaningful data is already written to TQ.

    Data flow (self-contained, mirrors AgentLoopWorkerTQ._agent_loop_postprocess):
      AgentLoopOutput → _prepare_tq_fields → _write_to_tq
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()

    # ------------------------------------------------------------------
    # Override _run_agent_loop to capture raw AgentLoopOutput (pre-padding)
    # ------------------------------------------------------------------

    async def _run_agent_loop_with_raw_output(
        self,
        sampling_params: dict,
        trajectory: dict,
        *,
        trace: bool = False,
        **kwargs,
    ) -> tuple[AgentLoopOutput, _InternalAgentLoopOutput]:
        """Run agent loop and return both raw and padded outputs.

        Returns:
            ``(raw_output, padded_output)`` where ``raw_output`` is the
            unpadded :class:`AgentLoopOutput` (before ``_agent_loop_postprocess``)
            and ``padded_output`` is the :class:`_InternalAgentLoopOutput`
            (after padding, used for the return value of ``generate_sequences``).
        """
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            agent_name = kwargs.get("agent_name")
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            raw_output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            padded_output = await self._agent_loop_postprocess(raw_output, trajectory["validate"], **kwargs)
            return raw_output, padded_output

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences and write results to TQ (blocking).

        Mirrors AgentLoopWorkerTQ.generate_sequences() pattern:
        1. Run agent loop for each sample in batch (capturing raw AgentLoopOutput)
        2. Post-process: compute score, teacher logprobs, etc.
        3. Write each trajectory to TQ using shared ``AgentLoopWorkerTQ.prepare_tq_fields``
        4. Block until TQ write completes (unlike AgentLoopWorkerTQ which is fire-and-forget)

        Args:
            batch (DataProto): Input batch (single sample for fully-async path).

        Returns:
            DataProto: Minimal output batch (data already written to TQ).
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # Default agent loop name
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        global_steps = batch.meta_info.get("global_steps", -1)

        # Run agent loops for all samples, capturing both raw and padded outputs
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop_with_raw_output(sampling_params, trajectory_info[i], trace=False, **kwargs)
                )
            )
        results: list[tuple[AgentLoopOutput, _InternalAgentLoopOutput]] = await asyncio.gather(*tasks)

        raw_outputs = [r[0] for r in results]
        padded_outputs = [r[1] for r in results]

        # ★ CORE: Write all trajectories to TQ using shared AgentLoopWorkerTQ logic
        await self._write_outputs_to_tq(raw_outputs, padded_outputs, batch, global_steps, validate)

        # Return minimal DataProto (caller just needs to know it succeeded)
        return self._postprocess(padded_outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=validate)

    # ------------------------------------------------------------------
    # TQ write — self-contained implementation (mirrors AgentLoopWorkerTQ._agent_loop_postprocess)
    # ------------------------------------------------------------------

    async def _write_outputs_to_tq(
        self,
        raw_outputs: list[AgentLoopOutput],
        padded_outputs: list[_InternalAgentLoopOutput],
        batch: DataProto,
        global_steps: int,
        validate: bool,
    ) -> None:
        """Write agent loop outputs to TQ (self-contained, mirrors AgentLoopWorkerTQ._agent_loop_postprocess).

        This method:
        1. Computes reward scores & teacher logprobs (same as AgentLoopWorkerTQ._agent_loop_postprocess)
        2. Prepares (keys, fields, tags) inline (same logic as AgentLoopWorkerTQ lines 399-430)
        3. Writes to TQ via ``tq.async_kv_batch_put``
        """
        if not raw_outputs:
            return

        # --- Step 1: Score & teacher logprobs (mirrors AgentLoopWorkerTQ._agent_loop_postprocess) ---
        # Use sample_id as base for globally unique TQ keys (injected by Rollouter._process_single_sample_streaming)
        base_key = batch.meta_info.get("sample_id", f"unknown_{global_steps}")

        # Build kwargs dict matching what AgentLoopWorkerTQ passes to its internal logic
        first_kwargs = {k: v[0] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
        first_kwargs["global_steps"] = global_steps
        # Ensure uid is present (required by _compute_advantage and other downstream processing)
        if "uid" not in first_kwargs:
            first_kwargs["uid"] = base_key

        await self._compute_score(raw_outputs, kwargs=first_kwargs)

        final_output = raw_outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=first_kwargs,
        )

        if final_output.reward_score is not None:
            for output in raw_outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields.get("reward_extra_info", {})

        keys, fields, tags = [], [], []
        for i, output in enumerate(raw_outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            # Key format: {sample_id}_{i} (globally unique, sample_id from Rollouter like "sample_0_42")
            keys.append(f"{base_key}_{i}")

            # Field dict: start from output.as_dict(), then enrich
            field = output.as_dict()
            field.update(first_kwargs)
            # Do not store raw image/video
            field.pop("multi_modal_data", None)
            # Uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)

            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "global_steps": global_steps,
                    "current_status": "finish",  # Required by ReplayBuffer.wait_and_sample()
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                }
            )

        # --- Step 3: Write to TQ ---
        partition_id = "train" if not validate else "val"
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

        bsz = len(raw_outputs)
        keys_str = ", ".join(keys[:3]) + ("..." if len(keys) > 3 else "")
        print(
            f"[FullyAsyncAgentLoopWorkerTQ] Wrote {bsz} samples [{keys_str}] to TQ ({partition_id})",
            flush=True,
        )


class FullyAsyncAgentLoopManagerTQ(FullyAsyncAgentLoopManager):
    """Agent loop manager that uses FullyAsyncAgentLoopWorkerTQ workers.
    Overrides the worker class so that generate_sequences_single() returns
    unpadded DataProto with variable-length per-sample tensors.
    """

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(FullyAsyncAgentLoopWorkerTQ)
        super().__init__(*args, **kwargs)


class TQFullyAsyncRollouter(FullyAsyncRollouter):
    """
    FullyAsyncRollouter variant that uses TransferQueue + ReplayBuffer instead of MessageQueue.

    Core design:
    - ReplayBuffer.acquire_slot() in _feed_samples() provides source-level flow control
      (replaces _should_pause_generation + MessageQueue.queue_size backpressure)
    - Generated samples are written to TQ (zero-copy) instead of MessageQueue (pickle)
    - FullyAsyncAgentLoopManager is unchanged — still used for actual generation
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        # Call parent __init__ (sets up datasets, dataloader, basic config)
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)

        # ==================== TQ-specific overrides ====================
        self.replay_buffer = None  # Ray Actor handle, set via set_replay_buffer()

        # ★ Tell base class _init_async_rollout_manager() to use TQ agent loop manager
        self.agent_loop_manager_class = FullyAsyncAgentLoopManagerTQ

        print("[TQFullyAsyncRollouter] initialized (TQ mode)")

    # ======== ReplayBuffer injection (replaces set_message_queue_client) ========

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        async with self.lock:
            self.replay_buffer = replay_buffer

            tq.init()
            print("[TQFullyAsyncRollouter] TQ initialized in Rollouter actor process", flush=True)

    # ======== Override: _feed_samples — add acquire_slot() ========

    async def _feed_samples(self):
        """Feed samples from dataloader to pending_queue, with source-level flow control.

        Key difference from base class: acquire_slot() is called BEFORE putting
        to pending_queue. This blocks the dataloader when too many samples are
        in-flight, replacing the need for _should_pause_generation().
        """
        print("[TQFullyAsyncRollouter][_feed_samples] STARTING", flush=True)
        continuous_iterator = self._create_continuous_iterator()
        feed_count = 0

        for epoch, batch_dict in continuous_iterator:
            print(f"[TQFullyAsyncRollouter][Feed] Acquiring slot for sample {feed_count}...", flush=True)
            acquired = await self.replay_buffer.acquire_slot.remote(timeout=None)
            if not acquired:
                print(
                    f"[TQFullyAsyncRollouter][Feed] ReplayBuffer finished or closed, "
                    f"stop feeding after {feed_count} samples"
                )
                break

            # Prepare data (same as base class)
            full_batch = prepare_single_generation_data(batch_dict, self.config)
            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)
            feed_count += 1
            print(
                f"[TQFullyAsyncRollouter][Feed] Put sample {feed_count - 1} to pending_queue (total={feed_count})",
                flush=True,
            )

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[TQFullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[TQFullyAsyncRollouter][Feed] Sample addition is complete, {feed_count} samples have been added")

    async def _should_pause_generation(self) -> bool:
        return False

    # ======== Override: _process_single_sample_streaming — delegated TQ write ========

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample: generate via ALM worker (which writes to TQ blocking), then release slot.

        Simplified from base class:
        - Base class: generate → put to MessageQueue
        - Old TQ path: generate → unbind 2D padded → strip padding → manual TQ write (complex, error-prone)
        - New TQ path: generate + TQ write both happen INSIDE FullyAsyncAgentLoopWorkerTQ.generate_sequences()
          → we just call it and release the slot

        The TQ write inside FullyAsyncAgentLoopWorkerTQ._write_outputs_to_tq() follows
        AgentLoopWorkerTQ._agent_loop_postprocess format exactly:
          AgentLoopOutput.as_dict() → 1-D variable-length tensors → list_of_dict_to_tensordict → TQ
        """
        print(
            f"[TQFullyAsyncRollouter][_process_single] Starting generate for {rollout_sample.sample_id}...",
            flush=True,
        )
        try:
            # Inject sample_id into batch so worker can use it for unique TQ keys
            rollout_sample.full_batch.meta_info["sample_id"] = rollout_sample.sample_id

            # ★ This single call does ALL of:
            #   1. Run agent loop(s) for the sample
            #   2. Compute reward scores, teacher logprobs, etc.
            #   3. Write data to TQ in unpadded 1-D tensor format (blocking!)
            #   4. Return minimal DataProto (data already in TQ)
            ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
            print(
                f"[TQFullyAsyncRollouter][_process_single] generate + TQ write done for {rollout_sample.sample_id}",
                flush=True,
            )
            self.total_generated_samples += len(ret) if ret is not None else 1
        except Exception as e:
            logger.exception(f"[TQFullyAsyncRollouter] Failed to process {rollout_sample.sample_id}: {e}")
        finally:
            # Always release the slot regardless of success/failure
            await self.replay_buffer.release_slot.remote()

        self.processed_sample_count += 1

    # ======== Override: _streaming_generation_main — finish signal via RB ========

    async def _streaming_generation_main(self):
        """Main entry for stream processing.

        Key difference from base class:
        - Finish signal goes through RB.signal_finish() instead of MQ.put_sample(None)
        """
        print("[TQFullyAsyncRollouter][_streaming_generation_main] ENTER", flush=True)
        if self.async_rollout_manager is None:
            print(
                "[TQFullyAsyncRollouter][_streaming_generation_main] Calling _init_async_rollout_manager...", flush=True
            )
            await self._init_async_rollout_manager()
            print("[TQFullyAsyncRollouter][_streaming_generation_main] _init_async_rollout_manager DONE", flush=True)
        else:
            print(
                "[TQFullyAsyncRollouter][_streaming_generation_main] "
                "async_rollout_manager already exists, skipping init",
                flush=True,
            )

        print(
            f"[TQFullyAsyncRollouter] Start streaming mode, max concurrent: {self.max_concurrent_samples}", flush=True
        )

        # Start feed and processor tasks (same as base class)
        print("[TQFullyAsyncRollouter] Starting feed_task...", flush=True)
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        print("[TQFullyAsyncRollouter] Starting processor_task...", flush=True)
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[TQFullyAsyncRollouter] Sample feed completed")
            await self.processor_task
            print("[TQFullyAsyncRollouter] Streaming process completed")
            await self.pending_queue.join()
            print("[TQFullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Streaming process exception: {e}")
            raise e
        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            await self.replay_buffer.signal_finish.remote()

            async with self.lock:
                self.running = False

    # ======== Override: reset_staleness — resume + delegate to RB ========

    async def reset_staleness(self):
        """Reset version window after parameter update."""
        # Reset RB version window (no longer passes active_task_count —
        # RB's staleness control is purely via acquire_slot dual-condition)
        print("[RollouterTQ][reset_staleness] calling RB.reset_staleness (version window reset)", flush=True)
        rb_timing = await self.replay_buffer.reset_staleness.remote()
        print(f"[TQFullyAsyncRollouter][reset_staleness] RB.reset_staleness DONE {rb_timing}", flush=True)
        return rb_timing

    # ======== Override: get_statistics — source from RB instead of MQ ========

    async def get_statistics(self) -> dict:
        """Gather statistics from RB and local state."""
        rb_stats = {}
        if self.replay_buffer is not None:
            rb_stats = await self.replay_buffer.get_statistics.remote()

        stats = {
            # Monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            # Counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/processed_sample_count": self.processed_sample_count,
            # Static config
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            # RB stats (if available)
            **rb_stats,
        }

        return stats
