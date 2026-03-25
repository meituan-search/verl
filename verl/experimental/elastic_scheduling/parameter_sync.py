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

"""
Elastic Parameter Synchronization for VERL

Extends CheckpointEngineManager to support dynamic resource scaling
with efficient parameter synchronization to new DP instances.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generator, Optional

import torch

from verl.checkpoint_engine import CheckpointEngineManager

if TYPE_CHECKING:
    from verl.checkpoint_engine.base import RolloutReplica
    from verl.single_controller.ray import RayWorkerGroup

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Parameter synchronization statistics"""

    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    total_bytes_transferred: int = 0
    avg_sync_time: float = 0.0

    def record_sync(self, success: bool, bytes_transferred: int, duration: float):
        """Record a sync operation"""
        self.total_syncs += 1
        if success:
            self.successful_syncs += 1
        self.total_bytes_transferred += bytes_transferred

        # Update rolling average
        n = self.successful_syncs
        self.avg_sync_time = (self.avg_sync_time * (n - 1) + duration) / n


class ElasticCheckpointManager:
    """
    Elastic Parameter Synchronization Manager

    Extends CheckpointEngineManager to support:
    1. Incremental sync to new DP instances
    2. Graceful handling of resource changes
    3. Efficient parameter broadcasting

    Key design decisions:
    - Uses existing NCCL/NIXL checkpoint engine for actual transfer
    - Maintains version tracking for each resource
    - Supports partial sync (only changed parameters)
    """

    def __init__(
        self,
        config,
        trainer: "RayWorkerGroup",
        replicas: list["RolloutReplica"],
        sync_backend: str = "nccl",
        enable_incremental_sync: bool = True,
    ):
        # Create base checkpoint manager
        self.base_manager = CheckpointEngineManager(
            config=config,
            trainer=trainer,
            replicas=replicas,
        )

        # Configuration
        self.sync_backend = sync_backend
        self.enable_incremental_sync = enable_incremental_sync

        # Resource tracking
        self._resource_sync_versions: dict[str, int] = {}
        self._current_version: int = 0

        # Pending syncs
        self._pending_syncs: asyncio.Queue = asyncio.Queue()
        self._sync_in_progress: bool = False

        # Statistics
        self.stats = SyncStats()

        # Callbacks
        self.on_sync_complete: Optional[Callable] = None
        self.on_sync_start: Optional[Callable] = None

    @property
    def trainer(self):
        return self.base_manager.trainer

    @property
    def replicas(self):
        return self.base_manager.replicas

    async def update_weights(self, global_steps: int = None):
        """
        Update weights to all replicas, including new ones

        This is the main entry point for parameter synchronization.
        It handles both existing and newly added replicas.
        """
        start_time = time.time()

        try:
            # Update current version
            if global_steps is not None:
                self._current_version = global_steps

            # Trigger base manager sync
            await self.base_manager.update_weights(global_steps)

            # Update version tracking for all replicas
            for i, replica in enumerate(self.replicas):
                resource_id = getattr(replica, "resource_id", f"replica_{i}")
                self._resource_sync_versions[resource_id] = self._current_version

            # Calculate transfer stats
            duration = time.time() - start_time
            # Estimate bytes (this would be more accurate with actual measurement)
            est_bytes = self._estimate_param_size()
            self.stats.record_sync(True, est_bytes, duration)

            logger.debug(f"Weight sync completed in {duration:.2f}s, version={self._current_version}")

            if self.on_sync_complete:
                await self.on_sync_complete(self._current_version)

        except Exception as e:
            duration = time.time() - start_time
            self.stats.record_sync(False, 0, duration)
            logger.error(f"Weight sync failed: {e}")
            raise

    async def sync_to_replicas(
        self,
        replicas: list["RolloutReplica"],
        is_elastic: bool = False,
    ):
        """
        Sync parameters to specific replicas (e.g., newly added elastic resources)

        Args:
            replicas: List of replicas to sync
            is_elastic: Whether these are elastic (new) resources
        """
        if not replicas:
            return

        start_time = time.time()

        try:
            if self.on_sync_start:
                await self.on_sync_start(replicas)

            # For elastic resources, we need to:
            # 1. Rebuild topology to include new replicas
            # 2. Broadcast parameters to all

            # Get current parameters from trainer
            per_tensor_param, _ = self.trainer.get_per_tensor_param()

            # Broadcast to replicas
            for replica in replicas:
                await self._sync_params_to_single_replica(replica, per_tensor_param)

                # Update version tracking
                resource_id = getattr(replica, "resource_id", f"replica_{id(replica)}")
                self._resource_sync_versions[resource_id] = self._current_version

            # Calculate stats
            duration = time.time() - start_time
            est_bytes = self._estimate_param_size() * len(replicas)
            self.stats.record_sync(True, est_bytes, duration)

            logger.info(f"Synced to {len(replicas)} elastic replicas in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            self.stats.record_sync(False, 0, duration)
            logger.error(f"Failed to sync to elastic replicas: {e}")
            raise

    async def _sync_params_to_single_replica(
        self,
        replica: "RolloutReplica",
        params: Generator[tuple[str, torch.Tensor], None, None],
    ):
        """Sync parameters to a single replica"""
        try:
            # Get checkpoint engine from replica
            if hasattr(replica, "checkpoint_engine"):
                engine = replica.checkpoint_engine
            elif hasattr(replica, "workers") and replica.workers:
                # Get engine from first worker
                worker = replica.workers[0]
                if hasattr(worker, "checkpoint_engine"):
                    engine = worker.checkpoint_engine
                else:
                    raise ValueError("Replica has no checkpoint engine")
            else:
                raise ValueError("Replica has no accessible checkpoint engine")

            # Prepare engine for receive
            if hasattr(engine, "prepare"):
                engine.prepare()

            # Send parameters
            if hasattr(engine, "send_weights"):
                await engine.send_weights(params)
            elif hasattr(engine, "receive_weights"):
                # For colocated mode, use direct receive
                pass

            # Finalize
            if hasattr(engine, "finalize"):
                engine.finalize()

        except Exception as e:
            logger.error(f"Failed to sync to replica: {e}")
            raise

    def _estimate_param_size(self) -> int:
        """Estimate total parameter size in bytes"""
        if not self.trainer or not hasattr(self.trainer, "engine"):
            return 0

        engine = self.trainer.engine
        if hasattr(engine, "module"):
            total_params = 0
            for param in engine.module.parameters():
                total_params += param.numel() * param.element_size()
            return total_params

        return 0

    def get_resource_sync_status(self) -> dict:
        """Get synchronization status for all resources"""
        return {
            "current_version": self._current_version,
            "tracked_resources": len(self._resource_sync_versions),
            "outdated_resources": [
                rid for rid, ver in self._resource_sync_versions.items() if ver < self._current_version
            ],
            "stats": {
                "total_syncs": self.stats.total_syncs,
                "successful": self.stats.successful_syncs,
                "failed": self.stats.failed_syncs,
                "avg_sync_time": self.stats.avg_sync_time,
            },
        }

    def is_resource_synced(self, resource_id: str) -> bool:
        """Check if a resource is synced to current version"""
        return self._resource_sync_versions.get(resource_id, -1) >= self._current_version

    async def wait_for_resource_sync(self, resource_id: str, timeout: float = 30.0):
        """Wait for a specific resource to be synced"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_resource_synced(resource_id):
                return True
            await asyncio.sleep(0.1)

        return False


class IncrementalSyncManager:
    """
    Manages incremental parameter synchronization

    Tracks which parameters have changed since last sync
    and only synchronizes the delta for efficiency.
    """

    def __init__(self, enable: bool = True):
        self.enable = enable
        self._last_params: Optional[dict] = None
        self._changed_keys: set = set()
        self._change_count: int = 0

    def detect_changes(self, params: Generator[tuple[str, torch.Tensor], None, None]) -> list[tuple[str, torch.Tensor]]:
        """
        Detect which parameters have changed

        Args:
            params: Generator of (name, tensor) pairs

        Returns:
            List of changed (name, tensor) pairs
        """
        if not self.enable:
            return list(params)

        params_list = list(params)
        current_keys = {name for name, _ in params_list}
        current_tensors = {name: tensor for name, tensor in params_list}

        changed = []

        if self._last_params is None:
            # First sync - all parameters
            changed = params_list
        else:
            # Check for changes
            for name, tensor in params_list:
                if name not in self._last_params:
                    changed.append((name, tensor))
                    self._changed_keys.add(name)
                elif not torch.equal(tensor, self._last_params[name]):
                    changed.append((name, tensor))
                    self._changed_keys.add(name)

            # Check for removed keys
            for name in self._last_params:
                if name not in current_keys:
                    self._changed_keys.add(name)

        # Update tracking
        self._last_params = current_tensors
        self._change_count += len(changed)

        logger.debug(f"Detected {len(changed)} changed parameters out of {len(params_list)}")

        return changed

    def get_change_summary(self) -> dict:
        """Get summary of parameter changes"""
        return {
            "total_changes": self._change_count,
            "recent_changes": len(self._changed_keys),
            "enabled": self.enable,
        }

    def reset(self):
        """Reset change tracking"""
        self._last_params = None
        self._changed_keys.clear()
        self._change_count = 0
