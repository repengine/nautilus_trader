from __future__ import annotations

from typing import Any

from ml.common.in_memory_bus import InMemoryPublisher
from ml.config.events import EventStatus, Stage
from ml.strategies.services import StrategyDecisionPublisher


def _capture_factory(out: list[tuple[str, dict[str, Any]]]):
    def _cap(topic: str, payload: dict[str, Any]) -> None:
        out.append((topic, payload))

    return _cap


def test_publisher_builds_domain_op_topic_and_payload() -> None:
    bus = InMemoryPublisher()
    captured: list[tuple[str, dict[str, Any]]] = []
    bus.subscribe("ml.strategies.created.#", _capture_factory(captured))
    pub = StrategyDecisionPublisher(bus, scheme="domain_op")

    ok = pub.publish(
        strategy_id="STRAT1",
        instrument_id="EUR/USD.SIM",
        signal_type="BUY",
        strength=0.8,
        model_predictions={"M": 0.9},
        risk_metrics={"r": 1.0},
        execution_params={"e": 2},
        ts_event=123,
        is_live=True,
        status=EventStatus.SUCCESS,
    )
    assert ok is True
    assert captured, "Expected at least one event"
    topic, payload = captured[0]
    assert topic.startswith("ml.strategies.created."), topic
    assert payload["stage"] == Stage.SIGNAL_EMITTED.value
    assert payload["status"] == EventStatus.SUCCESS.value
    assert payload["strategy_id"] == "STRAT1"
    assert payload["instrument_id"].startswith("EUR/USD")
    assert payload["signal_type"] == "BUY"


def test_publisher_builds_stage_first_topic() -> None:
    bus = InMemoryPublisher()
    captured: list[tuple[str, dict[str, Any]]] = []
    bus.subscribe("events.ml.SIGNAL_EMITTED.#", _capture_factory(captured))
    pub = StrategyDecisionPublisher(bus, scheme="stage_first", prefix="events.ml")

    ok = pub.publish(
        strategy_id="STRAT1",
        instrument_id="BTC-USD",
        signal_type="SELL",
        strength=0.6,
        model_predictions={"M": 0.4},
        risk_metrics={},
        execution_params={},
        ts_event=999,
        is_live=False,
        status=EventStatus.SUCCESS,
    )
    assert ok is True
    assert captured, "Expected stage-first event"
    topic, _ = captured[0]
    assert topic.startswith("events.ml.SIGNAL_EMITTED."), topic


class _BrokenPublisher(InMemoryPublisher):
    def publish(self, topic: str, payload: dict[str, Any]) -> bool:  # type: ignore[override]
        raise RuntimeError("boom")


def test_publisher_is_non_blocking_on_failure() -> None:
    bus = _BrokenPublisher()
    pub = StrategyDecisionPublisher(bus, scheme="domain_op")
    ok = pub.publish(
        strategy_id="S",
        instrument_id="X",
        signal_type="HOLD",
        strength=0.0,
        model_predictions={},
        risk_metrics=None,
        execution_params=None,
        ts_event=0,
        is_live=True,
        status=EventStatus.SUCCESS,
    )
    assert ok is False

