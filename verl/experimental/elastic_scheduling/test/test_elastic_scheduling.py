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
Unit tests for Elastic Scheduling Module
These tests only test core data structures and logic without importing the full module.
"""

import time

import pytest


class TestElasticResourceConfig:
    """Tests for ElasticResourceConfig"""

    def test_config_creation(self):
        """Test creating config with default values"""
        import pytest

        # Skip if peft is not available (needed for full module import)
        pytest.importorskip("peft", reason="peft module required for full module import")

        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
        )

        config = ElasticResourceConfig()

        assert config.min_elastic_gpus == 0
        assert config.max_elastic_gpus == 32
        assert config.elastic_dp_size == 8
        assert config.rollout_queue_high_watermark == 0.8
        assert config.rollout_queue_low_watermark == 0.3
        assert config.cooldown_seconds == 10.0

    def test_config_custom_values(self):
        """Test creating config with custom values"""
        import pytest

        pytest.importorskip("peft", reason="peft module required for full module import")

        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
        )

        config = ElasticResourceConfig(
            min_elastic_gpus=4,
            max_elastic_gpus=32,
            elastic_dp_size=4,
            rollout_queue_high_watermark=0.9,
        )

        assert config.min_elastic_gpus == 4
        assert config.max_elastic_gpus == 32
        assert config.elastic_dp_size == 4
        assert config.rollout_queue_high_watermark == 0.9


class TestHybridEngineMode:
    """Tests for HybridEngineMode enum"""

    def test_mode_values(self):
        """Test enum values"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            HybridEngineMode,
        )

        # auto() generates incrementing integers
        assert isinstance(HybridEngineMode.ROLLOUT.value, int)
        assert isinstance(HybridEngineMode.TRAIN.value, int)
        assert isinstance(HybridEngineMode.COLOCATED.value, int)
        assert HybridEngineMode.ROLLOUT != HybridEngineMode.TRAIN
        assert HybridEngineMode.ROLLOUT != HybridEngineMode.COLOCATED


class TestHybridEngineResource:
    """Tests for HybridEngineResource"""

    def test_resource_creation(self):
        """Test creating a resource"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            HybridEngineMode,
            HybridEngineResource,
        )

        resource = HybridEngineResource(
            resource_id="test_resource",
            gpu_ranks=[0, 1, 2, 3],
            world_size=4,
        )

        assert resource.resource_id == "test_resource"
        assert resource.gpu_ranks == [0, 1, 2, 3]
        assert resource.world_size == 4
        assert resource.current_mode == HybridEngineMode.ROLLOUT
        assert not resource.is_elastic
        assert not resource.is_active

    def test_resource_mode_properties(self):
        """Test mode properties"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            HybridEngineMode,
            HybridEngineResource,
        )

        resource = HybridEngineResource(
            resource_id="test",
            gpu_ranks=[0],
            world_size=1,
            current_mode=HybridEngineMode.TRAIN,
        )

        assert resource.is_train_mode
        assert not resource.is_rollout_mode

        resource.current_mode = HybridEngineMode.ROLLOUT
        assert resource.is_rollout_mode
        assert not resource.is_train_mode


class TestElasticResourceManager:
    """Tests for ElasticResourceManager"""

    def test_initialization(self):
        """Test manager initialization"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        rollout_spec = [0, 1, 2, 3, 4, 5, 6, 7]
        train_spec = [8, 9, 10, 11, 12, 13, 14, 15]
        elastic_spec = [16, 17, 18, 19, 20, 21, 22, 23]

        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=rollout_spec,
            train_resource_spec=train_spec,
            elastic_resource_spec=elastic_spec,
        )

        assert len(manager.rollout_resources) == 1
        assert len(manager.train_resources) == 1
        assert len(manager.elastic_resources) == 1

    def test_initial_modes(self):
        """Test initial resource modes"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
            HybridEngineMode,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[8, 9, 10, 11, 12, 13, 14, 15],
            elastic_resource_spec=[16, 17, 18, 19, 20, 21, 22, 23],
        )

        assert manager.rollout_resources[0].current_mode == HybridEngineMode.ROLLOUT
        assert manager.train_resources[0].current_mode == HybridEngineMode.TRAIN
        assert manager.elastic_resources[0].current_mode == HybridEngineMode.ROLLOUT

    def test_get_active_count(self):
        """Test getting active resource counts"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[8, 9, 10, 11, 12, 13, 14, 15],
            elastic_resource_spec=[16, 17, 18, 19, 20, 21, 22, 23],
        )

        # Mark all resources as active
        for r in manager.rollout_resources:
            r.is_active = True
        for r in manager.train_resources:
            r.is_active = True
        for r in manager.elastic_resources:
            r.is_active = True

        assert manager.get_active_rollout_count() == 2  # 1 fixed + 1 elastic (rollout mode)
        assert manager.get_active_train_count() == 1  # 1 fixed only

    def test_cooldown(self):
        """Test switch cooldown mechanism"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig(cooldown_seconds=2.0)
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[],
            train_resource_spec=[],
            elastic_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        resource = manager.elastic_resources[0]
        resource.is_active = True

        # First switch should be allowed
        assert manager._can_switch(resource.resource_id) is True

        # Add a switch record
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import HybridEngineMode, SwitchRecord

        manager._switch_history.append(
            SwitchRecord(
                resource_id=resource.resource_id,
                from_mode=HybridEngineMode.ROLLOUT,
                to_mode=HybridEngineMode.TRAIN,
                timestamp=time.time(),
            )
        )

        # Should be in cooldown
        assert manager._can_switch(resource.resource_id) is False

    def test_register_worker(self):
        """Test worker registration"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[],
            elastic_resource_spec=[],
        )

        resource = manager.rollout_resources[0]
        mock_handle = object()

        manager.register_worker(resource.resource_id, mock_handle)

        assert resource.worker_handle == mock_handle
        assert resource.is_active is True

    def test_get_status_summary(self):
        """Test status summary"""
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[8, 9, 10, 11, 12, 13, 14, 15],
            elastic_resource_spec=[16, 17, 18, 19, 20, 21, 22, 23],
        )

        status = manager.get_status_summary()

        assert "total_resources" in status
        assert "active_rollout" in status
        assert "active_train" in status
        assert "gpu_distribution" in status


class TestCongestionMonitor:
    """Tests for CongestionMonitor"""

    def test_initial_state(self):
        """Test initial monitor state"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
            SuggestedAction,
        )

        monitor = CongestionMonitor(
            window_size=10,
            high_watermark=0.8,
            low_watermark=0.3,
        )

        metrics = monitor.get_metrics()

        assert metrics.queue_utilization == 0.0
        assert metrics.production_rate == 0.0
        assert metrics.consumption_rate == 0.0
        assert metrics.suggested_action == SuggestedAction.STABLE

    def test_rollout_congestion(self):
        """Test rollout congestion detection"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
        )

        monitor = CongestionMonitor(
            window_size=10,
            high_watermark=0.8,
            low_watermark=0.3,
        )

        # Simulate high queue utilization
        monitor.record(queue_size=80, queue_capacity=100, produced=0, consumed=0)
        monitor.record(queue_size=85, queue_capacity=100, produced=0, consumed=0)
        monitor.record(queue_size=90, queue_capacity=100, produced=0, consumed=0)

        metrics = monitor.get_metrics()

        assert metrics.is_rollout_congested is True
        assert metrics.queue_utilization == 0.9

    def test_train_congestion(self):
        """Test train congestion detection"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
        )

        monitor = CongestionMonitor(
            window_size=10,
            high_watermark=0.8,
            low_watermark=0.3,
        )

        # Simulate low queue utilization
        monitor.record(queue_size=10, queue_capacity=100, produced=0, consumed=0)
        monitor.record(queue_size=5, queue_capacity=100, produced=0, consumed=0)
        monitor.record(queue_size=2, queue_capacity=100, produced=0, consumed=0)

        metrics = monitor.get_metrics()

        assert metrics.is_train_congested is True

    def test_reset(self):
        """Test monitor reset"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
        )

        monitor = CongestionMonitor(window_size=10)
        monitor.record(queue_size=50, queue_capacity=100, produced=10, consumed=10)
        monitor._ema_production = 100.0
        monitor._ema_consumption = 80.0

        monitor.reset()

        assert len(monitor.queue_sizes) == 0
        assert monitor._ema_production is None


class TestResourceCoordinator:
    """Tests for ResourceCoordinator"""

    def test_coordinator_creation(self):
        """Test coordinator can be created"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
            ResourceCoordinator,
        )
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[8, 9, 10, 11, 12, 13, 14, 15],
            elastic_resource_spec=[16, 17, 18, 19, 20, 21, 22, 23],
        )
        monitor = CongestionMonitor()

        coordinator = ResourceCoordinator(
            resource_manager=manager,
            congestion_monitor=monitor,
            sync_trigger_interval=4,
            auto_scale_enabled=True,
        )

        assert coordinator.resource_manager is manager
        assert coordinator.congestion_monitor is monitor
        assert coordinator.current_step == 0

    def test_get_status(self):
        """Test status summary"""
        from verl.experimental.elastic_scheduling.coordinator import (
            CongestionMonitor,
            ResourceCoordinator,
        )
        from verl.experimental.elastic_scheduling.model_engine.resource_manager import (
            ElasticResourceConfig,
            ElasticResourceManager,
        )

        config = ElasticResourceConfig()
        manager = ElasticResourceManager(
            config=config,
            rollout_resource_spec=[0, 1, 2, 3, 4, 5, 6, 7],
            train_resource_spec=[8, 9, 10, 11, 12, 13, 14, 15],
            elastic_resource_spec=[16, 17, 18, 19, 20, 21, 22, 23],
        )
        monitor = CongestionMonitor()

        coordinator = ResourceCoordinator(
            resource_manager=manager,
            congestion_monitor=monitor,
        )

        status = coordinator.get_current_status()

        assert "step" in status
        assert "is_coordinating" in status
        assert "stats" in status


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
