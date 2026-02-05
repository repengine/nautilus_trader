"""
Error-path tests for actor bus bridge initialization.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml.actors.ml_domain_events import init_actor_bus_bridge
from ml.common.message_bus import NoopPublisher


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@pytest.mark.unit
def test_actor_bus_bridge_disabled_returns_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ml.config.actor_bus.ActorBusConfig.from_env",
        lambda: SimpleNamespace(from_actor=False),
    )
    monkeypatch.setattr(
        "ml.config.bus.MessageBusConfig.from_env",
        lambda: SimpleNamespace(enabled=True),
    )

    actor = SimpleNamespace()
    bridge, scheme, prefix = init_actor_bus_bridge(actor)

    assert bridge is None
    assert scheme == "domain_op"
    assert prefix == "events.ml"


@pytest.mark.unit
def test_actor_bus_bridge_handles_store_disable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Store:
        def __init__(self) -> None:
            object.__setattr__(self, "publisher", object())
            object.__setattr__(self, "_enable_publishing", True)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_enable_publishing":
                raise RuntimeError("blocked")
            object.__setattr__(self, name, value)

    class _Bridge:
        def __init__(
            self,
            publisher: object,
            max_queue: int,
            throttler: object | None,
            per_topic_throttles: dict[str, object],
            component_id: str,
        ) -> None:
            self.publisher = publisher
            self.max_queue = max_queue
            self.throttler = throttler
            self.per_topic_throttles = per_topic_throttles
            self.component_id = component_id
            self.started = False

        def start(self) -> None:
            self.started = True

    actor_bus_cfg = SimpleNamespace(
        from_actor=True,
        throttle_enabled=False,
        throttle_rate_per_sec=1.0,
        throttle_burst=1,
        max_queue=5,
        scheme="domain_op",
        prefix="events.ml",
    )
    bus_cfg = SimpleNamespace(enabled=True)

    monkeypatch.setattr(
        "ml.config.actor_bus.ActorBusConfig.from_env",
        lambda: actor_bus_cfg,
    )
    monkeypatch.setattr(
        "ml.config.bus.MessageBusConfig.from_env",
        lambda: bus_cfg,
    )
    monkeypatch.setattr(
        "ml.actors.ml_domain_events.publisher_from_config",
        lambda _cfg: NoopPublisher(),
    )
    monkeypatch.setattr(
        "ml.actors.ml_domain_events._parse_per_topic_throttles",
        lambda: {"events.ml.*": object()},
    )
    monkeypatch.setattr("ml.actors.ml_domain_events.DomainEventBridge", _Bridge)

    actor = SimpleNamespace(
        _feature_store=_Store(),
        _model_store=None,
        _strategy_store=_Store(),
        _data_store=_Store(),
    )

    bridge, scheme, prefix = init_actor_bus_bridge(actor)

    assert bridge is not None
    assert scheme == "domain_op"
    assert prefix == "events.ml"
    assert actor._feature_store.publisher is None
    assert actor._strategy_store.publisher is None
    assert actor._data_store.publisher is None
    assert actor._feature_store._enable_publishing is True


@pytest.mark.unit
def test_actor_bus_bridge_records_warning_metric_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> SimpleNamespace:
        raise RuntimeError("boom")

    class _Metrics:
        def __init__(self) -> None:
            self.calls = 0

        def inc(self, *_args: object, **_kwargs: object) -> None:
            self.calls += 1
            raise RuntimeError("metrics down")

    metrics = _Metrics()

    monkeypatch.setattr("ml.config.actor_bus.ActorBusConfig.from_env", _raise)
    monkeypatch.setattr(
        "ml.common.metrics_manager.MetricsManager.default",
        lambda: metrics,
    )

    actor = SimpleNamespace()
    bridge, scheme, prefix = init_actor_bus_bridge(actor)

    assert bridge is None
    assert scheme == "domain_op"
    assert prefix == "events.ml"
    assert metrics.calls == 1
