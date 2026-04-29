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

"""TQAgentLoopWorker: Independent Worker with built-in event loop for TQ-based fully async training.

Architecture:
- Worker actively pulls pending tasks from ReplayBuffer (instead of being called by AgentLoopManager)
- For each prompt, creates n sessions (rollout.n sampling), each running an AgentLoop
- Each session may produce multiple trajectories (for multi-output agent loops like prefix switching)
- Server reads prompt from TQ directly, generates response, writes back to TQ

Data Flow:
  Rollouter -> TQ(kv_batch_put, status=pending)
  -> ReplayBuffer(poll detects pending)
  -> Worker(get_pending_keys) -> TQ(kv_batch_put, status=running)
  -> Server(kv_batch_get prompt -> generate -> kv_batch_put response, status=finish)
  -> Trainer(wait_and_sample -> kv_batch_get data -> train -> kv_clear)
"""

import asyncio
import logging
import os
from typing import Any

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    TokenOutput,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(num_cpus=1)
class TQAgentLoopWorker:
    """Independent Worker with built-in event loop, actively pulling tasks from ReplayBuffer.

    Responsibilities:
    1. Pull pending tasks {uid} from ReplayBuffer
    2. Handle n samplings per prompt: {uid}_{session_id}
    3. Execute AgentLoop for each session, which may return multiple trajectories: {uid}_{session_id}_{trajectory_id}
    4. Copy {uid} prompt data to create {uid}_{session_id}_{trajectory_id} entries in TQ
    5. Dispatch key to Server side for generation

    The Worker acts as a bridge between ReplayBuffer (task scheduling) and LLM Servers (generation).
    """

    def __init__(
        self,
        config: DictConfig,
        replay_buffer_handle: ray.actor.ActorHandle,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        tokenizer,
        processor=None,
    ):
        self.config = config
        self.replay_buffer = replay_buffer_handle
        self.tokenizer = tokenizer
        self.processor = processor

        # LoadBalancer only handles scheduling
        self.load_balancer = load_balancer_handle
        self.servers = dict(servers)

        # AgentLoop configuration
        self.rollout_config = config.actor_rollout_ref.rollout
        self.partition_id = config.trainer.get("partition_id", "train")

        # State
        self.finished = False
        self.active_tasks: set[asyncio.Task] = set()

        # Initialize TQ on worker side
        tq.init()

        # Start the built-in event loop
        self._loop_task = None  # Will be set when start() is called

    def start(self):
        """Start the event loop (call after Ray actor is created)."""
        self._loop_task = asyncio.create_task(self._run_loop())
        print(f"[TQAgentLoopWorker] Event loop started, partition={self.partition_id}")

    async def _run_loop(self):
        """Main loop that actively pulls and processes tasks."""
        while not self.finished:
            try:
                # 1. Pull pending tasks from ReplayBuffer
                pending_tasks = await self.replay_buffer.get_pending_keys.remote(
                    partition_id=self.partition_id,
                    limit=self.rollout_config.batch_size if hasattr(self.rollout_config, "batch_size") else 8,
                    timeout=1.0,
                )

                if not pending_tasks:
                    await asyncio.sleep(0.1)
                    continue

                # 2. Create processing task for each pending item
                for key, meta in pending_tasks:
                    task = asyncio.create_task(self._process_single_prompt(key, meta))
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)

                # 3. Control concurrency: wait for some tasks to complete
                max_concurrent = getattr(self.rollout_config, "max_concurrent", 16)
                if len(self.active_tasks) >= max_concurrent:
                    done, _ = await asyncio.wait(
                        self.active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

            except Exception as e:
                logger.exception(f"[TQAgentLoopWorker] Error in _run_loop: {e}")
                await asyncio.sleep(1.0)

        # Wait for all active tasks to finish
        if self.active_tasks:
            print(f"[TQAgentLoopWorker] Waiting for {len(self.active_tasks)} active tasks to finish...")
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        print("[TQAgentLoopWorker] Event loop exited")

    async def _process_single_prompt(self, key: str, meta: dict):
        """Process a single prompt including n samplings.

        Args:
            key: The original key written by Rollouter (format: {partition_id}_{uid})
            meta: Metadata dictionary containing uid and other info
        """
        uid = meta["uid"]

        try:
            # 1. Update status to running
            await tq.async_kv_batch_put(
                keys=[key],
                tags=[{"current_status": "running"}],
                partition_id=self.partition_id,
            )

            # 2. n samplings (each sample is one AgentLoop execution)
            session_tasks = []
            for session_id in range(self.rollout_config.n):
                session_key = f"{self.partition_id}_{uid}_{session_id}_0"
                task = asyncio.create_task(self._run_session(session_key, key, session_id, meta))
                session_tasks.append(task)

            # 3. Wait for all sessions to complete
            await asyncio.gather(*session_tasks, return_exceptions=True)

            # 4. Update the original key status to finish (marking prompt processing complete)
            await tq.async_kv_batch_put(
                keys=[key],
                tags=[{"current_status": "finish"}],
                partition_id=self.partition_id,
            )

            logger.debug(f"[TQAgentLoopWorker] Completed processing prompt {uid}, {self.rollout_config.n} sessions")

        except Exception as e:
            logger.exception(f"[TQAgentLoopWorker] Error processing {key}: {e}")
            # Mark as error
            await tq.async_kv_batch_put(
                keys=[key],
                tags=[{"current_status": "error", "error": str(e)}],
                partition_id=self.partition_id,
            )

    async def _run_session(
        self,
        session_key: str,
        parent_key: str,
        session_id: int,
        parent_meta: dict,
    ):
        """Execute a single sampling session (one AgentLoop).

        This method:
        1. Creates an AgentLoop instance configured for this session
        2. Reads prompt data from TQ using parent_key
        3. Runs the AgentLoop which internally calls Server.generate_with_tq()
        4. Post-processes the output and writes trajectory data to TQ

        Args:
            session_key: Full key for this session (format: {partition_id}_{uid}_{session_id}_0)
            parent_key: Original key from Rollouter (format: {partition_id}_{uid})
            session_id: Session index (0 to n-1)
            parent_meta: Original metadata from Rollouter
        """
        # 1. Create AgentLoop instance
        agent_loop = self._create_agent_loop()

        # 2. Prepare sampling parameters
        sampling_params = {
            "temperature": self.rollout_config.temperature,
            "top_p": self.rollout_config.top_p,
            "top_k": self.rollout_config.top_k,
            "logprobs": self.rollout_config.calculate_log_probs,
        }
        # Handle max_tokens / max_new_tokens
        if hasattr(self.rollout_config, "max_tokens"):
            sampling_params["max_tokens"] = self.rollout_config.max_tokens
        if hasattr(self.rollout_config, "max_new_tokens"):
            sampling_params["max_new_tokens"] = self.rollout_config.max_new_tokens

        # 3. Copy prompt data from parent_key to session_key in TQ
        # This ensures each session has its own copy of the prompt data
        await self._copy_prompt_to_session(parent_key, session_key, parent_meta)

        # 4. Get prompt data for the AgentLoop
        prompt_data = await tq.async_kv_batch_get(
            keys=[parent_key],
            select_fields=["input_ids", "attention_mask", "position_ids"],
            partition_id=self.partition_id,
        )

        if not prompt_data or len(prompt_data.get("input_ids", [])) == 0:
            logger.warning(f"[TQAgentLoopWorker] No prompt data found for key {parent_key}")
            return

        # 5. Execute AgentLoop
        # The AgentLoop will call back into self.generate() which dispatches to Server
        try:
            output = await agent_loop.run(
                sampling_params=sampling_params,
                session_key=session_key,
                parent_key=parent_key,
                partition_id=self.partition_id,
                **parent_meta,
            )

            # 6. Post-process output: write trajectory data to TQ
            if output is not None:
                await self._postprocess_output(output, session_key, session_id, parent_meta)

        except Exception as e:
            logger.exception(f"[TQAgentLoopWorker] Error in _run_session for {session_key}: {e}")
            await tq.async_kv_batch_put(
                keys=[session_key],
                tags={"current_status": "error", "error": str(e)},
                partition_id=self.partition_id,
            )

    async def _copy_prompt_to_session(self, parent_key: str, session_key: str, parent_meta: dict):
        """Copy prompt data from parent key to session key in TQ.

        Each session needs its own copy of the input data so that different
        sessions can independently track their state without interfering.
        """
        # Read prompt fields from parent
        fields_to_copy = ["input_ids", "attention_mask", "position_ids"]
        # Also copy non-tensor metadata fields
        meta_fields = ["uid", "data_source", "raw_prompt"]

        prompt_data = await tq.async_kv_batch_get(
            keys=[parent_key],
            select_fields=fields_to_copy,
            partition_id=self.partition_id,
        )

        if prompt_data:
            # Write copied data to session key with initial tags
            session_tags = {
                "current_status": "pending",
                "uid": parent_meta.get("uid", ""),
                "session_id": 0,  # Will be updated per session
                "trajectory_id": 0,
                "start_model_version": parent_meta.get("start_model_version", 0),
                "end_model_version": parent_meta.get("end_model_version", 0),
                "prompt_len": parent_meta.get("prompt_len", 0),
            }

            # Add any additional metadata fields
            for mf in meta_fields:
                if mf in parent_meta:
                    session_tags[mf] = parent_meta[mf]

            await tq.async_kv_batch_put(
                keys=[session_key],
                fields=prompt_data,
                tags=session_tags,
                partition_id=self.partition_id,
            )

    async def _postprocess_output(self, output: Any, session_key: str, session_id: int, parent_meta: dict):
        """Post-process AgentLoop output and write results to TQ.

        Handles both single AgentLoopOutput and list of outputs (multiple trajectories).
        """
        if isinstance(output, AgentLoopOutput):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            logger.warning(f"[TQAgentLoopWorker] Unexpected output type: {type(output)}")
            return

        if not outputs:
            logger.warning(f"[TQAgentLoopWorker] Empty output for session {session_key}")
            return

        # Process each trajectory output
        keys, fields_list, tags_list = [], [], []

        for traj_idx, out in enumerate(outputs):
            # Build trajectory key
            if len(outputs) == 1:
                traj_key = session_key  # Single trajectory, keep session key
            else:
                # Multiple trajectories: modify trajectory_id in key
                base_parts = session_key.rsplit("_", 1)
                traj_key = f"{base_parts[0]}_{traj_idx}"

            # Prepare field data from output
            field_dict = self._output_to_field_dict(out, parent_meta)
            keys.append(traj_key)
            fields_list.append(field_dict)

            # Prepare tags
            prompt_len = len(out.prompt_ids) if hasattr(out, "prompt_ids") else 0
            response_len = len(out.response_ids) if hasattr(out, "response_ids") else 0
            tags_list.append(
                {
                    "current_status": "finish",
                    "uid": parent_meta.get("uid", ""),
                    "session_id": session_id,
                    "trajectory_id": traj_idx,
                    "start_model_version": parent_meta.get("start_model_version", 0),
                    "end_model_version": out.extra_fields.get("global_steps", parent_meta.get("end_model_version", 0))
                    if out.extra_fields
                    else parent_meta.get("end_model_version", 0),
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                }
            )

        # Batch write to TQ
        if keys:
            await tq.async_kv_batch_put(
                keys=keys,
                fields=fields_list,
                tags=tags_list,
                partition_id=self.partition_id,
            )

    def _output_to_field_dict(self, output: AgentLoopOutput, parent_meta: dict) -> dict:
        """Convert AgentLoopOutput to a field dictionary for TQ storage."""
        field_dict = {}

        # Core tensor fields
        if hasattr(output, "prompt_ids") and output.prompt_ids is not None:
            field_dict["prompts"] = torch.tensor(output.prompt_ids, dtype=torch.int64)
        if hasattr(output, "response_ids") and output.response_ids is not None:
            field_dict["responses"] = torch.tensor(output.response_ids, dtype=torch.int64)
        if hasattr(output, "response_mask") and output.response_mask is not None:
            rm = output.response_mask
            if not isinstance(rm, torch.Tensor):
                rm = torch.tensor(rm, dtype=torch.int64)
            field_dict["response_mask"] = rm
        if hasattr(output, "log_probs") and output.log_probs is not None:
            lp = output.log_probs
            if not isinstance(lp, torch.Tensor):
                lp = torch.tensor(lp, dtype=torch.float32)
            field_dict["rollout_log_probs"] = lp

        # Reward score
        if hasattr(output, "reward_score") and output.reward_score is not None:
            rs = output.reward_score
            if not isinstance(rs, torch.Tensor):
                rs = torch.tensor([rs], dtype=torch.float32)
            field_dict["rm_scores"] = rs

        # Compose full input_ids and attention_mask
        if "prompts" in field_dict and "responses" in field_dict:
            field_dict["input_ids"] = torch.cat([field_dict["prompts"], field_dict["responses"]], dim=0)
            seq_len = field_dict["input_ids"].shape[0]
            field_dict["attention_mask"] = torch.ones(seq_len, dtype=torch.int64)
            field_dict["position_ids"] = torch.arange(seq_len, dtype=torch.int64)

        # Extra fields from output
        if output.extra_fields:
            for k, v in output.extra_fields.items():
                if k not in field_dict and v is not None:
                    if isinstance(v, torch.Tensor):
                        field_dict[k] = v
                    elif isinstance(v, int | float):
                        field_dict[k] = torch.tensor([v], dtype=torch.float32 if isinstance(v, float) else torch.int64)

        # Copy non-tensor metadata from parent
        for meta_key in ["uid", "data_source", "agent_name"]:
            if meta_key in parent_meta:
                field_dict[meta_key] = parent_meta[meta_key]

        return field_dict

    def _create_agent_loop(self):
        """Create an AgentLoop instance configured for this worker's setup.

        The AgentLoop uses this worker's `generate` method as its server interface,
        enabling the AgentLoop to dispatch generation requests through our TQ-aware path.
        """
        agent_name = self.rollout_config.agent.default_agent_loop

        # Import the agent loop registry
        from verl.experimental.agent_loop import _agent_loop_registry

        agent_loop_config = _agent_loop_registry[agent_name]

        # Create a wrapper that passes this worker as the server_manager
        # so AgentLoop calls self.generate() for LLM inference
        return hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfig(OmegaConf.to_container(self.config, resolve=False)),
            server_manager=self,  # Worker itself acts as server_manager
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self._get_dataset_class(),
            data_config=DictConfig(OmegaConf.to_container(self.config.data, resolve=False)),
        )

    def _get_dataset_class(self):
        """Get the dataset class from config."""
        from verl.utils.dataset.rl_dataset import RLHFDataset

        # Try to import the actual dataset class based on config
        try:
            # Return a placeholder - actual dataset creation happens in main
            return RLHFDataset
        except Exception:
            return RLHFDataset

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict,
        **kwargs,
    ) -> TokenOutput:
        """Call Server to execute generation (called by AgentLoop).

        This method:
        1. Selects a Server via LoadBalancer
        2. Calls Server's generate_with_tq method
        3. Server reads prompt from TQ, generates, writes response back to TQ

        Args:
            request_id: Unique request ID for sticky session routing
            prompt_ids: List of prompt token IDs
            sampling_params: Sampling parameters (temperature, top_p, etc.)
            **kwargs: Additional arguments including partition_id, session_key, etc.

        Returns:
            TokenOutput with generated tokens and metadata
        """
        # 1. Acquire a server from LoadBalancer
        server_address = await self.load_balancer.acquire_server.remote(request_id)
        server_handle = self.servers.get(server_address)

        if server_handle is None:
            raise RuntimeError(f"[TQAgentLoopWorker] Server not found for address: {server_address}")

        try:
            # 2. Call Server's generate_with_tq method
            # Server will read prompt from TQ, generate, and write response back to TQ
            output = await server_handle.generate_with_tq.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                partition_id=kwargs.get("partition_id", self.partition_id),
                session_key=kwargs.get("session_key"),
                **kwargs,
            )
            return output
        finally:
            # 3. Release the server back to LoadBalancer
            try:
                await self.load_balancer.release_server.remote(server_address)
            except Exception as e:
                logger.warning(f"[TQAgentLoopWorker] Error releasing server {server_address}: {e}")

    def signal_finish(self):
        """Signal the Worker to stop processing new tasks."""
        print("[TQAgentLoopWorker] Received finish signal")
        self.finished = True

    def add_server(self, server_address: str, server_handle: ray.actor.ActorHandle) -> None:
        """Add a new server to the worker's server map."""
        self.servers[server_address] = server_handle
        logger.debug(f"[TQAgentLoopWorker] Added server: {server_address}")

    def remove_server(self, server_address: str) -> None:
        """Remove a server from the worker's server map."""
        self.servers.pop(server_address, None)
        logger.debug(f"[TQAgentLoopWorker] Removed server: {server_address}")

    def get_statistics(self) -> dict:
        """Return current worker statistics."""
        return {
            "active_tasks": len(self.active_tasks),
            "finished": self.finished,
            "servers": list(self.servers.keys()),
            "partition_id": self.partition_id,
        }
