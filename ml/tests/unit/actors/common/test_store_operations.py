"""
Targeted unit tests for StoreOperationsComponent.

Extends the minimal smoke checks with focused coverage for:
- store initialization and fallback paths
- health status reporting
- persistence worker stop behavior
- synchronous store flushing

Tests implemented here:
1. test_import_and_instantiate - Proves component can be imported
2. test_api_surface - Proves all required methods exist
3. test_fallback_behavior - Proves DummyStore fallback works

REMAINING TESTS (from test design, not implemented yet):
- 13 additional unit tests
- 4 integration tests (require PostgreSQL)
- 3 performance tests

"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from ml.actors.common import StoreOperationsComponent, StoreOperationsProtocol


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@dataclass(frozen=True)
class _Config:
    allow_dummy_fallback: bool = True
    enable_async_persistence: bool = False
    persistence_queue_size: int = 10
    persistence_flush_interval: float = 0.1
    persistence_batch_size: int = 4


class _Engine:
    def __init__(self) -> None:
        self.pool = object()

    def dispose(self) -> None:
        return None


class _Store:
    def __init__(self, engine: object | None = None) -> None:
        self.engine = engine
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


class _Loop:
    def __init__(self, *, running: bool, closed: bool = False) -> None:
        self._running = running
        self._closed = closed
        self.created: list[Any] = []

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return self._closed

    def create_task(self, coro: Any) -> object:
        self.created.append(coro)
        if hasattr(coro, "close"):
            coro.close()
        return object()


class _Worker:
    def __init__(self, loop: _Loop) -> None:
        self._loop = loop

    async def stop(self, *, drain: bool, timeout: float) -> None:
        return None

    def queue_size(self) -> int:
        return 0


def test_import_and_instantiate():
    """
    Test: Component can be imported and basic structure is correct.

    This is a META-TEST that proves the component exists and has the right structure.
    It does NOT test functionality - just that the code is valid Python.
    """
    # Given: Component class exists
    assert StoreOperationsComponent is not None
    assert StoreOperationsProtocol is not None

    # Then: Component has expected attributes
    assert hasattr(StoreOperationsComponent, "__init__")
    assert hasattr(StoreOperationsComponent, "feature_store")
    assert hasattr(StoreOperationsComponent, "model_store")
    assert hasattr(StoreOperationsComponent, "strategy_store")
    assert hasattr(StoreOperationsComponent, "data_store")
    assert hasattr(StoreOperationsComponent, "get_health_status")
    assert hasattr(StoreOperationsComponent, "on_start")
    assert hasattr(StoreOperationsComponent, "on_stop")


def test_api_surface():
    """
    Test: All required public methods/properties exist.

    Verifies the component implements the expected API surface.
    """
    # Given: Component class
    component_methods = [m for m in dir(StoreOperationsComponent) if not m.startswith("_")]

    # Then: All required methods present
    required_methods = [
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "persistence_worker",
        "get_health_status",
        "on_start",
        "on_stop",
    ]

    for method in required_methods:
        assert method in component_methods, f"Missing required method: {method}"


def test_store_ops_uses_preinitialized_services_and_reports_healthy() -> None:
    engine = _Engine()
    store = _Store(engine=engine)
    services = SimpleNamespace(
        feature_store=store,
        model_store=store,
        strategy_store=store,
        data_store=store,
    )
    component = StoreOperationsComponent(_Config(), actor_id="actor", services=services)

    assert component.feature_store is store
    health = component.get_health_status()
    assert {item["status"] for item in health.values()} == {"healthy"}


def test_store_ops_health_status_degraded_and_unhealthy() -> None:
    healthy_store = _Store(engine=_Engine())
    degraded_store = SimpleNamespace()
    unhealthy_store = _Store(engine=None)
    services = SimpleNamespace(
        feature_store=healthy_store,
        model_store=degraded_store,
        strategy_store=unhealthy_store,
        data_store=None,
    )
    component = StoreOperationsComponent(_Config(), actor_id="actor", services=services)

    health = component.get_health_status()
    assert health["feature_store"]["status"] == "healthy"
    assert health["model_store"]["status"] == "degraded"
    assert health["strategy_store"]["status"] == "unhealthy"
    assert health["data_store"]["status"] == "unhealthy"


def test_store_ops_fallback_emits_metrics(monkeypatch) -> None:
    class _Counter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **_labels: str) -> "_Counter":
            return self

        def inc(self) -> None:
            self.count += 1

    counter = _Counter()
    monkeypatch.setattr(
        "ml.actors.common.store_operations.get_counter",
        lambda *args, **kwargs: counter,
    )

    store = _Store(engine=_Engine())
    services = SimpleNamespace(
        feature_store=store,
        model_store=store,
        strategy_store=store,
        data_store=store,
    )
    calls = {"count": 0}

    def _init_services(_config: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("primary failed")
        return services

    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        _init_services,
    )

    component = StoreOperationsComponent(_Config(), actor_id="actor")

    assert component.data_store is store
    assert calls["count"] == 2
    assert counter.count == 4


def test_store_ops_disallows_fallback(monkeypatch) -> None:
    def _init_services(_config: Any) -> Any:
        raise RuntimeError("primary failed")

    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        _init_services,
    )

    with pytest.raises(RuntimeError):
        StoreOperationsComponent(_Config(allow_dummy_fallback=False), actor_id="actor")


def test_stop_persistence_worker_schedules_on_running_loop(monkeypatch) -> None:
    store = _Store(engine=_Engine())
    services = SimpleNamespace(
        feature_store=store,
        model_store=store,
        strategy_store=store,
        data_store=store,
    )
    component = StoreOperationsComponent(_Config(), actor_id="actor", services=services)

    loop = _Loop(running=True)
    worker = _Worker(loop)
    component._persistence_worker = worker
    monkeypatch.setattr("asyncio.get_running_loop", lambda: loop)

    scheduled = component._stop_persistence_worker()

    assert scheduled is False
    assert len(loop.created) == 1


def test_stop_persistence_worker_returns_true_when_loop_closed() -> None:
    store = _Store(engine=_Engine())
    services = SimpleNamespace(
        feature_store=store,
        model_store=store,
        strategy_store=store,
        data_store=store,
    )
    component = StoreOperationsComponent(_Config(), actor_id="actor", services=services)

    loop = _Loop(running=False, closed=True)
    component._persistence_worker = _Worker(loop)

    assert component._stop_persistence_worker() is True


def test_on_stop_flushes_stores_when_worker_missing() -> None:
    feature_store = _Store(engine=_Engine())
    model_store = _Store(engine=_Engine())
    strategy_store = _Store(engine=_Engine())
    data_store = _Store(engine=_Engine())
    services = SimpleNamespace(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_store=data_store,
    )
    component = StoreOperationsComponent(_Config(), actor_id="actor", services=services)

    component.on_stop()

    assert feature_store.flush_calls == 1
    assert model_store.flush_calls == 1
    assert strategy_store.flush_calls == 1
    assert data_store.flush_calls == 1


# NOTE: The following tests from phase_2_3_1_CONSOLIDATED.md are NOT YET IMPLEMENTED:
#
# UNIT TESTS (16 total, only 2 above implemented):
# - test_store_initialization_all_stores
# - test_store_fallback_to_dummy
# - test_store_progressive_fallback_chain
# - test_store_health_check_all_healthy
# - test_store_health_check_degraded_state
# - test_store_circuit_breaker_propagation
# - test_store_property_accessors_cached
# - test_store_initialization_error_handling
# - test_async_worker_initialization_on_start
# - test_async_worker_enqueue_feature_write
# - test_async_worker_queue_full_warning
# - test_async_worker_flush_interval
# - test_cleanup_on_stop_drains_queue
# - test_cleanup_on_stop_synchronous_fallback
# - test_cleanup_on_stop_thread_joins
# - test_fallback_rejected_when_disallowed
#
# INTEGRATION TESTS (4 total):
# - test_store_integration_feature_store_write_read
# - test_store_integration_model_store_write_read
# - test_store_integration_strategy_store_write_read
# - test_store_integration_data_store_query
#
# PERFORMANCE TESTS (3 total):
# - test_performance_store_initialization_latency
# - test_performance_health_check_latency
# - test_performance_accessor_latency
#
# These should be implemented by the Test Implementation Agent or in a follow-up iteration.
