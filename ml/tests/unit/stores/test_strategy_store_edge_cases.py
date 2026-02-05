from __future__ import annotations

from typing import Any

import pytest

from ml.stores.base import StrategyOrderEvent
from ml.stores.base import StrategyRiskHaltEvent
from ml.stores.strategy_store import StrategyStore


class _Clock:
    def __init__(self, value: int) -> None:
        self._value = value

    def timestamp_ns(self) -> int:
        return self._value


def _make_store(monkeypatch: pytest.MonkeyPatch, clock: _Clock | None = None) -> StrategyStore:
    monkeypatch.setattr(StrategyStore, "_init_engine_and_tables", lambda self: None)
    return StrategyStore(connection_string=None, clock=clock)


def test_write_order_event_skips_unparseable(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch, clock=_Clock(0))
    monkeypatch.setattr(StrategyOrderEvent, "from_event", lambda *_, **__: None)

    store.write_order_event(object())

    assert store._order_event_buffer == []


def test_write_order_event_fills_run_id_and_ingested_at_ns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(42)
    store = _make_store(monkeypatch, clock=clock)
    event = StrategyOrderEvent(
        event_id="evt-1",
        strategy_id="s1",
        instrument_id="AAPL",
        client_order_id="client-1",
        venue_order_id=None,
        event_type="OrderSubmitted",
        payload={"status": "ok"},
        _ts_event=10,
        _ts_init=10,
        run_id=None,
        ingested_at_ns=None,
    )

    store.write_order_event(event, run_id="   ")

    assert event.run_id is None
    assert event.ingested_at_ns == 42
    assert store._order_event_buffer == [event]


def test_write_risk_halt_event_normalizes_ts_init(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(0)
    store = _make_store(monkeypatch, clock=clock)

    store.write_risk_halt_event(
        strategy_id="s1",
        instrument_id="AAPL",
        event_type="halted",
        reason="risk",
        detail=None,
        ts_event=1_000_000,
        run_id="",
    )

    assert len(store._risk_halt_buffer) == 1
    event = store._risk_halt_buffer[0]
    assert isinstance(event, StrategyRiskHaltEvent)
    assert event._ts_init == event._ts_event
    assert event.run_id is None


def test_flush_order_events_emits_and_updates_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(123)
    store = _make_store(monkeypatch, clock=clock)
    event = StrategyOrderEvent(
        event_id="evt-2",
        strategy_id="s1",
        instrument_id="AAPL",
        client_order_id="client-2",
        venue_order_id=None,
        event_type="OrderFilled",
        payload={"status": "filled"},
        _ts_event=11,
        _ts_init=11,
    )
    store._order_event_buffer.append(event)

    written: list[list[StrategyOrderEvent]] = []
    emitted: list[list[StrategyOrderEvent]] = []

    monkeypatch.setattr(
        store,
        "write_order_events",
        lambda data, publish_bus=True: written.append(list(data)),
    )
    monkeypatch.setattr(
        store,
        "_emit_order_event_events",
        lambda events: emitted.append(list(events)),
    )

    store._flush_order_events()

    assert written == [[event]]
    assert emitted == [[event]]
    assert store._order_event_buffer == []
    assert store._last_flush_ns == 123


def test_record_observability_stage_boundary_calls_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(monkeypatch)
    calls: list[dict[str, Any]] = []

    def _record_stage_boundary(
        obs_service: object | None,
        *,
        component: str,
        instrument_id: str,
        stage: str,
        ts_stage_start: int,
        ts_stage_end: int,
        row_count: int,
    ) -> None:
        calls.append(
            {
                "obs_service": obs_service,
                "component": component,
                "instrument_id": instrument_id,
                "stage": stage,
                "ts_stage_start": ts_stage_start,
                "ts_stage_end": ts_stage_end,
                "row_count": row_count,
            },
        )

    monkeypatch.setattr(
        "ml.common.observability_utils.record_stage_boundary",
        _record_stage_boundary,
    )
    store._observability_service = object()

    store._record_observability_stage_boundary(
        stage="strategy_signal_storage",
        instrument_id="SPY",
        ts_stage_start=1,
        ts_stage_end=2,
        row_count=3,
    )

    assert calls == [
        {
            "obs_service": store._observability_service,
            "component": "strategy_store",
            "instrument_id": "SPY",
            "stage": "strategy_signal_storage",
            "ts_stage_start": 1,
            "ts_stage_end": 2,
            "row_count": 3,
        },
    ]


def test_flush_risk_halt_events_clears_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch, clock=_Clock(0))
    event = StrategyRiskHaltEvent(
        event_id="risk-1",
        strategy_id="s1",
        instrument_id="AAPL",
        event_type="halted",
        reason="risk",
        detail="details",
        _ts_event=99,
        _ts_init=99,
    )
    store._risk_halt_buffer.append(event)

    written: list[list[StrategyRiskHaltEvent]] = []
    emitted: list[list[StrategyRiskHaltEvent]] = []

    monkeypatch.setattr(
        store,
        "write_risk_halt_events",
        lambda data, publish_bus=True: written.append(list(data)),
    )
    monkeypatch.setattr(
        store,
        "_emit_risk_halt_event_events",
        lambda events: emitted.append(list(events)),
    )

    store._flush_risk_halt_events()

    assert written == [[event]]
    assert emitted == [[event]]
    assert store._risk_halt_buffer == []
