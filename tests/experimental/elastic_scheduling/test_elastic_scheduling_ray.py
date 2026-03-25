#!/usr/bin/env python
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
Unit tests for Elastic Scheduling Module - Ray GPU Version

This script is designed to run on a Ray GPU cluster.
Run with: ray job submit --address='http://10.148.11.18:8420' -- python test_elastic_scheduling_ray.py
"""

import time

import ray


def test_config_creation():
    """Test creating config with default values"""
    from verl.experimental.elastic_scheduling.resource_manager import (
        ElasticResourceConfig,
    )

    config = ElasticResourceConfig()

    assert config.min_elastic_gpus == 0
    assert config.max_elastic_gpus == 32
    assert config.elastic_dp_size == 8
    assert config.rollout_queue_high_watermark == 0.8
    assert config.rollout_queue_low_watermark == 0.3
    assert config.cooldown_seconds == 10.0
    print("✓ test_config_creation passed")


def test_config_custom_values():
    """Test creating config with custom values"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_config_custom_values passed")


def test_mode_values():
    """Test enum values"""
    from verl.experimental.elastic_scheduling.resource_manager import (
        HybridEngineMode,
    )

    # auto() generates incrementing integers
    assert isinstance(HybridEngineMode.ROLLOUT.value, int)
    assert isinstance(HybridEngineMode.TRAIN.value, int)
    assert isinstance(HybridEngineMode.COLOCATED.value, int)
    assert HybridEngineMode.ROLLOUT != HybridEngineMode.TRAIN
    assert HybridEngineMode.ROLLOUT != HybridEngineMode.COLOCATED
    print("✓ test_mode_values passed")


def test_resource_creation():
    """Test creating a resource"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_resource_creation passed")


def test_resource_mode_properties():
    """Test mode properties"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_resource_mode_properties passed")


def test_manager_initialization():
    """Test manager initialization"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_manager_initialization passed")


def test_initial_modes():
    """Test initial resource modes"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_initial_modes passed")


def test_get_active_count():
    """Test getting active resource counts"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_get_active_count passed")


def test_cooldown():
    """Test switch cooldown mechanism"""
    from verl.experimental.elastic_scheduling.resource_manager import (
        ElasticResourceConfig,
        ElasticResourceManager,
        HybridEngineMode,
        SwitchRecord,
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
    print("✓ test_cooldown passed")


def test_register_worker():
    """Test worker registration"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_register_worker passed")


def test_get_status_summary():
    """Test status summary"""
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_get_status_summary passed")


def test_congestion_monitor_initial_state():
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
    print("✓ test_congestion_monitor_initial_state passed")


def test_rollout_congestion():
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
    print("✓ test_rollout_congestion passed")


def test_train_congestion():
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
    print("✓ test_train_congestion passed")


def test_congestion_monitor_reset():
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
    print("✓ test_congestion_monitor_reset passed")


def test_coordinator_creation():
    """Test coordinator can be created"""
    from verl.experimental.elastic_scheduling.coordinator import (
        CongestionMonitor,
        ResourceCoordinator,
    )
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_coordinator_creation passed")


def test_coordinator_status():
    """Test status summary"""
    from verl.experimental.elastic_scheduling.coordinator import (
        CongestionMonitor,
        ResourceCoordinator,
    )
    from verl.experimental.elastic_scheduling.resource_manager import (
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
    print("✓ test_coordinator_status passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Elastic Scheduling Module - Unit Tests")
    print("=" * 60)
    print()

    tests = [
        # Config tests
        test_config_creation,
        test_config_custom_values,
        # Mode tests
        test_mode_values,
        # Resource tests
        test_resource_creation,
        test_resource_mode_properties,
        # Manager tests
        test_manager_initialization,
        test_initial_modes,
        test_get_active_count,
        test_cooldown,
        test_register_worker,
        test_get_status_summary,
        # Congestion monitor tests
        test_congestion_monitor_initial_state,
        test_rollout_congestion,
        test_train_congestion,
        test_congestion_monitor_reset,
        # Coordinator tests
        test_coordinator_creation,
        test_coordinator_status,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    print(f"Ray initialized: {ray.is_initialized()}")
    print(f"Ray version: {ray.__version__}")
    print()

    # Run tests
    success = run_all_tests()

    # Shutdown Ray
    ray.shutdown()

    # Exit with appropriate code
    exit(0 if success else 1)
