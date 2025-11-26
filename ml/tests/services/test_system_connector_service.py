"""Tests for the system connector integration service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pytest

from ml.dashboard.services.system_service import SystemConnectRequest
from ml.dashboard.services.system_service import SystemConnectorService


class DummyEngine:
    """Deterministic engine stub supporting connect/disconnect hooks."""

    def __init__(self, *, delay: float = 0.0, should_fail: bool = False) -> None:
        self.delay = delay
        self.should_fail = should_fail
        self.connected = False
        self.disconnect_calls = 0

    def connect(self) -> None:
        if self.delay:
            time.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("connect failed")
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False


@dataclass(slots=True)
class DummyKernel:
    """Kernel container mirroring TradingNode.kernel attributes used in tests."""

    data_engine: DummyEngine | None
    exec_engine: DummyEngine | None
    cache: object | None


class DummyTradingNode:
    """Trading node stub exposing build/stop/dispose lifecycle operations."""

    def __init__(
        self,
        *,
        data_engine: DummyEngine | None,
        exec_engine: DummyEngine | None,
        cache: object | None = None,
        build_delay: float = 0.0,
    ) -> None:
        self.kernel = DummyKernel(data_engine=data_engine, exec_engine=exec_engine, cache=cache)
        self.build_delay = build_delay
        self.built = False
        self.stopped = False
        self.disposed = False

    def build(self) -> None:
        if self.build_delay:
            time.sleep(self.build_delay)
        self.built = True

    def stop(self) -> None:
        self.stopped = True

    def dispose(self) -> None:
        self.disposed = True


class NodeFactory:
    """Callable that records every generated trading node stub."""

    def __init__(self, builder: Callable[[], DummyTradingNode]) -> None:
        self._builder = builder
        self.created: list[DummyTradingNode] = []

    def __call__(self) -> DummyTradingNode:
        node = self._builder()
        self.created.append(node)
        return node


@pytest.mark.asyncio
async def test_connect_system_success() -> None:
    factory = NodeFactory(
        lambda: DummyTradingNode(
            data_engine=DummyEngine(),
            exec_engine=DummyEngine(),
            cache={},
        )
    )
    service = SystemConnectorService(None, node_factory=factory)

    result = await service.connect_system(SystemConnectRequest(timeout_seconds=1.0))

    assert result.success is True
    assert result.status == "CONNECTED"
    assert {status.component for status in result.components} == {"build", "data_engine", "exec_engine", "cache"}
    node = factory.created[-1]
    assert node.built is True
    assert node.kernel.data_engine and node.kernel.data_engine.connected is True
    assert node.kernel.exec_engine and node.kernel.exec_engine.connected is True
    snapshot = service.get_system_status()
    assert snapshot.connected is True


@pytest.mark.asyncio
async def test_force_reconnect_disposes_previous_node() -> None:
    factory = NodeFactory(
        lambda: DummyTradingNode(
            data_engine=DummyEngine(),
            exec_engine=DummyEngine(),
            cache={},
        )
    )
    service = SystemConnectorService(None, node_factory=factory)

    await service.connect_system(SystemConnectRequest(timeout_seconds=1.0))
    first_node = factory.created[0]

    await service.connect_system(SystemConnectRequest(timeout_seconds=1.0, force_reconnect=True))

    assert len(factory.created) == 2
    assert first_node.stopped is True
    assert first_node.disposed is True
    snapshot = service.get_system_status()
    assert snapshot.connected is True


@pytest.mark.asyncio
async def test_component_failure_results_in_degraded_status() -> None:
    factory = NodeFactory(
        lambda: DummyTradingNode(
            data_engine=DummyEngine(should_fail=True),
            exec_engine=DummyEngine(),
            cache={},
        )
    )
    service = SystemConnectorService(None, node_factory=factory)

    result = await service.connect_system(SystemConnectRequest(timeout_seconds=1.0))

    assert result.success is False
    assert result.status == "DEGRADED"
    failing = {status.component: status for status in result.components}
    assert failing["data_engine"].connected is False
    assert failing["data_engine"].error is not None
    snapshot = service.get_system_status()
    assert snapshot.connected is False


@pytest.mark.asyncio
async def test_connect_timeout_reports_failed_component() -> None:
    factory = NodeFactory(
        lambda: DummyTradingNode(
            data_engine=DummyEngine(),
            exec_engine=DummyEngine(),
            cache={},
            build_delay=0.2,
        )
    )
    service = SystemConnectorService(None, node_factory=factory)

    result = await service.connect_system(
        SystemConnectRequest(timeout_seconds=0.05, component_timeout_seconds=0.05)
    )

    assert result.success is False
    assert result.status == "DEGRADED"
    failing = {status.component: status for status in result.components}
    build_status = failing["build"]
    assert build_status.connected is False
    assert build_status.error == "timeout"
    snapshot = service.get_system_status()
    assert snapshot.connected is False


@pytest.mark.asyncio
async def test_disconnect_without_connection_is_noop() -> None:
    service = SystemConnectorService(None, node_factory=lambda: DummyTradingNode(
        data_engine=DummyEngine(),
        exec_engine=DummyEngine(),
        cache={},
    ))

    result = await service.disconnect_system()

    assert result.success is True
    assert result.status == "NOOP"
    assert result.message == "System already disconnected"


@pytest.mark.asyncio
async def test_disconnect_after_success_clears_status() -> None:
    factory = NodeFactory(
        lambda: DummyTradingNode(
            data_engine=DummyEngine(),
            exec_engine=DummyEngine(),
            cache={},
        )
    )
    service = SystemConnectorService(None, node_factory=factory)

    await service.connect_system(SystemConnectRequest(timeout_seconds=1.0))
    result = await service.disconnect_system()

    assert result.success is True
    assert result.status == "DISCONNECTED"
    snapshot = service.get_system_status()
    assert snapshot.connected is False
    assert snapshot.components == tuple()
