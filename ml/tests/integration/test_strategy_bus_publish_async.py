"""
Integration tests for strategy bus publish safety.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from types import SimpleNamespace
from typing import Any

import pytest

from nautilus_trader.model.data import BarType

from ml.common.bus_bridge import DomainEventBridge
from ml.common.message_bus import NoopPublisher
from ml.config.base import MLStrategyConfig
from ml.strategies.base import SimpleMLStrategy


@contextmanager
def env(vars: dict[str, str]) -> Iterator[None]:
    """
    Temporarily set environment variables for the duration of a test.
    """
    old = {key: os.getenv(key) for key in vars}
    os.environ.update(vars)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _CapturePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:  # noqa: D401
        self.calls.append((topic, payload))
        return True


@pytest.mark.integration
def test_strategy_bus_uses_async_bridge_in_prod(
    monkeypatch: pytest.MonkeyPatch,
    test_bar_type: BarType,
) -> None:
    """
    Ensure strategy bus uses async bridge in prod mode and disables store publishing.
    """
    capture = _CapturePublisher()

    def fake_factory(_cfg: Any) -> Any:
        return capture

    class _StoreStub:
        def __init__(self) -> None:
            self.publisher: object | None = object()
            self._enable_publishing: bool = True

    stores = SimpleNamespace(strategy_store=_StoreStub())

    with env(
        {
            "ML_BUS_ENABLE": "1",
            "ML_BUS_FROM_STRATEGY": "1",
            "ML_BUS_FROM_STORE": "0",
        },
    ):
        monkeypatch.setattr("ml.common.message_bus.publisher_from_config", fake_factory)
        config = MLStrategyConfig(
            instrument_id=test_bar_type.instrument_id,
            ml_signal_source="ML_SIGNAL_ACTOR",
            use_strategy_store=False,
        )
        strategy = SimpleMLStrategy(config=config, stores=stores)

        assert isinstance(strategy._bus_bridge, DomainEventBridge)
        assert strategy._bus_publisher is strategy._bus_bridge
        assert stores.strategy_store.publisher is None
        assert stores.strategy_store._enable_publishing is False

        bridge = strategy._bus_bridge
        if bridge is not None:
            bridge.stop(drain=True, timeout=1.0)


@pytest.mark.integration
def test_strategy_bus_disabled_without_async_bridge(
    test_bar_type: BarType,
) -> None:
    """
    Ensure strategy bus publishing is disabled when no async bridge is configured.
    """
    with env(
        {
            "ML_BUS_ENABLE": "1",
            "ML_BUS_FROM_STRATEGY": "0",
            "ML_BUS_FROM_STORE": "0",
        },
    ):
        config = MLStrategyConfig(
            instrument_id=test_bar_type.instrument_id,
            ml_signal_source="ML_SIGNAL_ACTOR",
            use_strategy_store=False,
        )
        strategy = SimpleMLStrategy(config=config)

        assert strategy._bus_bridge is None
        assert isinstance(strategy._bus_publisher, NoopPublisher)
