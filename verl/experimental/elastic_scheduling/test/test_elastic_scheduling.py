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
Unit tests for Elastic Scheduling Module.

Supports two run modes:
  1. pytest (recommended):
       pytest test_elastic_scheduling.py -v
  2. Ray job submission:
       ray job submit --address='http://<HEAD>:8265' -- python test_elastic_scheduling.py

Tests cover:
  - ElasticMode / ElasticState           (elastic_engine_workers)
  - ElasticResourceInfo                  (coordinator)
  - ElasticCoordinator._evaluate_action  (coordinator Ray Actor)
  - ElasticGlobalRequestLoadBalancer     (agent_loop.elastic_agent_loop)
"""

import time
from enum import Enum

import pytest
import ray

from verl.experimental.elastic_scheduling.agent_loop.elastic_agent_loop import ElasticGlobalRequestLoadBalancer
from verl.experimental.elastic_scheduling.coordinator import ElasticCoordinator, ElasticResourceInfo
from verl.experimental.elastic_scheduling.elastic_engine_workers import ElasticMode, ElasticState

# ---------------------------------------------------------------------------
# Ray session fixture (module-scoped – one cluster for all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lb(server_ids: list):
    """Create an ElasticGlobalRequestLoadBalancer Ray Actor."""

    return ElasticGlobalRequestLoadBalancer.remote(server_ids)


def _make_coordinator_actor(
    resources: dict,
    high_watermark: float = 0.8,
    low_watermark: float = 0.3,
    min_train_resources: int = 0,
    min_rollout_resources: int = 0,
    prod_rate: float = 0.0,
    cons_rate: float = 0.0,
):
    """
    Create an ElasticCoordinator Ray Actor pre-populated with the given
    resources and config.  Passes None for external handles (rollouter /
    trainer / message_queue) since _evaluate_action does not touch them.
    """

    elastic_resource_infos = [{"resource_id": rid, "initial_mode": mode} for rid, mode in resources.items()]
    config = {
        "high_watermark": high_watermark,
        "low_watermark": low_watermark,
        "min_train_resources": min_train_resources,
        "min_rollout_resources": min_rollout_resources,
    }

    actor = ElasticCoordinator.remote(
        elastic_rollouter=None,
        elastic_trainer=None,
        message_queue=None,
        elastic_resource_infos=elastic_resource_infos,
        config=config,
    )

    if prod_rate > 0 or cons_rate > 0:
        ray.get(actor._set_ema_rates_for_test.remote(prod_rate or None, cons_rate or None))

    return actor


# ---------------------------------------------------------------------------
# ElasticMode / ElasticState tests
# ---------------------------------------------------------------------------


class TestElasticMode:
    """Tests for ElasticMode enum."""

    def test_mode_values(self):
        assert hasattr(ElasticMode, "TRAIN")
        assert hasattr(ElasticMode, "ROLLOUT")
        assert hasattr(ElasticMode, "SWITCHING")
        assert ElasticMode.TRAIN != ElasticMode.ROLLOUT
        assert ElasticMode.ROLLOUT != ElasticMode.SWITCHING

    def test_mode_is_enum(self):
        assert issubclass(ElasticMode, Enum)


class TestElasticState:
    """Tests for ElasticState dataclass."""

    def test_default_values(self):
        state = ElasticState(resource_id="res0")

        assert state.resource_id == "res0"
        assert state.current_mode == ElasticMode.TRAIN
        assert state.param_version == -1
        assert state.train_world_ranks == []
        assert state.last_switch_time == 0.0
        assert state.total_switches == 0
        assert state.is_healthy is True

    def test_record_switch(self):
        state = ElasticState(resource_id="res0")
        before = time.time()
        state.record_switch(ElasticMode.ROLLOUT)
        after = time.time()

        assert state.current_mode == ElasticMode.ROLLOUT
        assert state.total_switches == 1
        assert before <= state.last_switch_time <= after

    def test_multiple_switches(self):
        state = ElasticState(resource_id="res0")
        for mode in [ElasticMode.ROLLOUT, ElasticMode.TRAIN, ElasticMode.ROLLOUT]:
            state.record_switch(mode)

        assert state.total_switches == 3
        assert state.current_mode == ElasticMode.ROLLOUT


# ---------------------------------------------------------------------------
# ElasticResourceInfo tests
# ---------------------------------------------------------------------------


class TestElasticResourceInfo:
    """Tests for ElasticResourceInfo dataclass in coordinator."""

    def test_default_values(self):
        info = ElasticResourceInfo(resource_id="r0", current_mode="rollout")

        assert info.resource_id == "r0"
        assert info.current_mode == "rollout"
        assert info.param_version == -1
        assert info.last_switch_time == 0.0
        assert info.is_healthy is True
        assert info.worker_handles == []

    def test_custom_values(self):
        info = ElasticResourceInfo(
            resource_id="r1",
            current_mode="train",
            param_version=7,
            is_healthy=False,
        )

        assert info.resource_id == "r1"
        assert info.current_mode == "train"
        assert info.param_version == 7
        assert info.is_healthy is False


# ---------------------------------------------------------------------------
# ElasticCoordinator._evaluate_action tests (via Ray Actor)
# ---------------------------------------------------------------------------


class TestElasticCoordinatorEvaluateAction:
    """Unit tests for ElasticCoordinator._evaluate_action (via Ray Actor)."""

    def test_stable_when_no_resources(self):
        """Empty resource set → stable."""
        actor = _make_coordinator_actor({})
        action, _ = ray.get(actor._evaluate_action.remote(0.5))
        assert action == "stable"

    def test_scale_rollout_high_queue(self):
        """Queue above high watermark + train resource available → scale_rollout."""
        actor = _make_coordinator_actor(
            {"r0": "train", "r1": "rollout"},
            high_watermark=0.8,
            min_train_resources=0,
        )
        action, confidence = ray.get(actor._evaluate_action.remote(0.9))
        assert action == "scale_rollout"
        assert confidence > 0.6

    def test_no_scale_rollout_when_min_train_constraint(self):
        """min_train_resources prevents switching the only train resource."""
        actor = _make_coordinator_actor(
            {"r0": "train"},
            high_watermark=0.8,
            min_train_resources=1,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.95))
        assert action == "stable"

    def test_scale_train_low_queue(self):
        """Queue below low watermark + prod > cons → scale_train."""
        actor = _make_coordinator_actor(
            {"r0": "rollout", "r1": "rollout"},
            low_watermark=0.3,
            min_rollout_resources=0,
            prod_rate=100.0,
            cons_rate=50.0,
        )
        action, confidence = ray.get(actor._evaluate_action.remote(0.05))
        assert action == "scale_train"
        assert confidence > 0.6

    def test_no_scale_train_when_prod_le_cons(self):
        """Low queue + prod ≤ cons, but cons < prod*2 so rate-based fallback is skipped → stable."""
        actor = _make_coordinator_actor(
            {"r0": "rollout"},
            low_watermark=0.3,
            prod_rate=10.0,
            cons_rate=15.0,  # prod(10) < cons(15), but cons < prod*2=20 → rate-based skipped
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.05))
        assert action == "stable"

    def test_rate_based_scale_rollout(self):
        """Rate-based fallback: prod >> cons, mid-queue → scale_rollout."""
        actor = _make_coordinator_actor(
            {"r0": "train"},
            high_watermark=0.8,
            min_train_resources=0,
            prod_rate=200.0,
            cons_rate=50.0,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.6))
        assert action == "scale_rollout"

    def test_rate_based_scale_train(self):
        """Rate-based fallback: cons >> prod, low queue → scale_train."""
        actor = _make_coordinator_actor(
            {"r0": "rollout"},
            low_watermark=0.3,
            min_rollout_resources=0,
            prod_rate=10.0,
            cons_rate=50.0,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.2))
        assert action == "scale_train"

    def test_high_watermark_boundary(self):
        """queue_util exactly at high_watermark is NOT > threshold → stable."""
        actor = _make_coordinator_actor(
            {"r0": "train", "r1": "rollout"},
            high_watermark=0.8,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.8))
        assert action == "stable"

    def test_just_below_high_watermark(self):
        """queue_util just below high_watermark → stable."""
        actor = _make_coordinator_actor(
            {"r0": "train", "r1": "rollout"},
            high_watermark=0.8,
            low_watermark=0.3,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.79))
        assert action == "stable"

    def test_above_high_watermark_triggers(self):
        """queue_util just above high_watermark → scale_rollout."""
        actor = _make_coordinator_actor(
            {"r0": "train", "r1": "rollout"},
            high_watermark=0.8,
        )
        action, _ = ray.get(actor._evaluate_action.remote(0.81))
        assert action == "scale_rollout"


# ---------------------------------------------------------------------------
# ElasticGlobalRequestLoadBalancer tests
# ---------------------------------------------------------------------------


class TestElasticGlobalRequestLoadBalancer:
    """Tests for ElasticGlobalRequestLoadBalancer (Ray Actor)."""

    def test_init_empty_raises(self):
        """Empty server list should raise ValueError (wrapped in RayTaskError)."""

        with pytest.raises(ray.exceptions.ActorDiedError):
            actor = ElasticGlobalRequestLoadBalancer.remote([])
            ray.get(actor.acquire_server.remote("probe"))

    def test_acquire_single_server(self):
        """Single server pool: every request routes to that server."""
        lb = _make_lb(["s1"])
        server = ray.get(lb.acquire_server.remote("req-1"))
        assert server == "s1"
        assert ray.get(lb.get_inflight_count.remote("s1")) == 1

    def test_acquire_least_loaded(self):
        """New requests go to the server with the fewest in-flight."""
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.set_inflight_for_test.remote("s1", 10))
        server = ray.get(lb.acquire_server.remote("req-new"))
        assert server == "s2"

    def test_sticky_routing(self):
        """Same request_id routes to the same server (sticky session)."""
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.set_inflight_for_test.remote("s2", 100))  # bias toward s1
        first = ray.get(lb.acquire_server.remote("req-sticky"))
        ray.get(lb.release_server.remote(first))
        second = ray.get(lb.acquire_server.remote("req-sticky"))
        assert first == second

    def test_release_decrements_inflight(self):
        lb = _make_lb(["s1"])
        ray.get(lb.acquire_server.remote("req-1"))
        assert ray.get(lb.get_inflight_count.remote("s1")) == 1
        ray.get(lb.release_server.remote("s1"))
        assert ray.get(lb.get_inflight_count.remote("s1")) == 0

    def test_release_unknown_server_is_noop(self):
        """Releasing an unknown server should not raise."""
        lb = _make_lb(["s1"])
        ray.get(lb.release_server.remote("unknown"))  # should not raise

    def test_release_at_zero_is_noop(self):
        """Releasing when count is 0 should not go negative."""
        lb = _make_lb(["s1"])
        ray.get(lb.release_server.remote("s1"))
        assert ray.get(lb.get_inflight_count.remote("s1")) == 0

    def test_add_server(self):
        lb = _make_lb(["s1"])
        ray.get(lb.add_server.remote("s2"))
        assert ray.get(lb.has_server.remote("s2"))
        assert ray.get(lb.get_inflight_count.remote("s2")) == 0

    def test_add_server_clears_removed_flag(self):
        lb = _make_lb(["s1"])
        ray.get(lb.remove_server.remote("s1"))
        assert ray.get(lb.is_server_removed.remote("s1"))
        ray.get(lb.add_server.remote("s1"))
        assert not ray.get(lb.is_server_removed.remote("s1"))

    def test_remove_server_marks_removal(self):
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.remove_server.remote("s1"))
        assert ray.get(lb.is_server_removed.remote("s1"))
        assert "s1" not in ray.get(lb.get_all_servers.remote())

    def test_removed_server_not_acquired(self):
        """Removed server must not receive new requests."""
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.remove_server.remote("s1"))
        for i in range(20):
            server = ray.get(lb.acquire_server.remote(f"req-{i}"))
            assert server == "s2", f"Acquired removed server on request {i}"

    def test_no_available_server_raises(self):
        """All servers removed → RuntimeError on acquire."""
        lb = _make_lb(["s1"])
        ray.get(lb.remove_server.remote("s1"))
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(lb.acquire_server.remote("req-x"))

    def test_cleanup_removed_server(self):
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.remove_server.remote("s1"))
        ray.get(lb.cleanup_removed_server.remote("s1"))
        assert not ray.get(lb.has_server.remote("s1"))
        assert not ray.get(lb.is_server_removed.remote("s1"))

    def test_get_all_servers_excludes_removed(self):
        lb = _make_lb(["s1", "s2", "s3"])
        ray.get(lb.remove_server.remote("s2"))
        active = ray.get(lb.get_all_servers.remote())
        assert "s1" in active and "s3" in active
        assert "s2" not in active

    def test_sticky_reroutes_after_removal(self):
        """Sticky session re-routes to healthy server when original is removed."""
        lb = _make_lb(["s1", "s2"])
        ray.get(lb.set_inflight_for_test.remote("s2", 100))  # bias toward s1
        first = ray.get(lb.acquire_server.remote("req-sticky2"))
        assert first == "s1"
        ray.get(lb.release_server.remote(first))
        ray.get(lb.remove_server.remote("s1"))
        second = ray.get(lb.acquire_server.remote("req-sticky2"))
        assert second == "s2"


# ---------------------------------------------------------------------------
# Main runner (for `python test_elastic_scheduling.py` / Ray job submit)
# ---------------------------------------------------------------------------


def _run_test(fn, passed, failed):
    try:
        fn()
        print(f"✓ {fn.__name__} passed")
        return passed + 1, failed
    except Exception as e:
        print(f"✗ {fn.__name__} FAILED: {e}")
        import traceback

        traceback.print_exc()
        return passed, failed + 1


def run_all_tests():
    print("=" * 60)
    print("Elastic Scheduling Module - Unit Tests")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    test_classes = [
        TestElasticMode,
        TestElasticState,
        TestElasticResourceInfo,
        TestElasticCoordinatorEvaluateAction,
        TestElasticGlobalRequestLoadBalancer,
    ]

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                passed, failed = _run_test(getattr(instance, name), passed, failed)

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    # Ray initialization is required for ElasticCoordinator actor tests.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Ray initialized: {ray.is_initialized()}")
    print(f"Ray version: {ray.__version__}")
    print()

    success = run_all_tests()

    ray.shutdown()
    exit(0 if success else 1)
