from __future__ import annotations

import pytest
from typing import Any

from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import (
    DashboardService,
    _EVENT_CACHE_HITS,
    _EVENT_CACHE_MISSES,
    _EVENT_FAILURES_TOTAL,
    _EVENT_POLLS_TOTAL,
)

pytestmark = pytest.mark.usefixtures("mock_tracing_backend")


@pytest.fixture(autouse=True)
def _isolated_prom_registry(isolated_prometheus_registry: Any) -> None:
    """Keep event metrics isolated per test case."""
    del isolated_prometheus_registry


def _metric_value(counter: Any, **labels: str) -> float:
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


def _reset_counter(counter: Any, **labels: str) -> None:
    if labels:
        counter.labels(**labels)._value.set(0.0)
    else:
        counter._value.set(0.0)


def _make_service(
    *,
    ttl: float = 60.0,
    max_entries: int = 10,
) -> DashboardService:
    cfg = DashboardConfig(
        events_cache_ttl_seconds=ttl,
        events_cache_max_entries=max_entries,
        events_poll_interval_seconds=0.0,
    )
    svc = DashboardService.from_config(cfg)
    return svc


def test_list_events_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service()
    events = [
        {"id": "1-0", "topic": "STAGE.EVENT", "payload": {"source": "live"}},
    ]
    calls: list[int] = []

    def _poll(self: DashboardService, *, limit: int) -> list[dict[str, Any]]:  # noqa: D401
        calls.append(limit)
        return events

    monkeypatch.setattr(DashboardService, "_poll_events", _poll)

    _reset_counter(_EVENT_CACHE_HITS)
    _reset_counter(_EVENT_CACHE_MISSES)
    _reset_counter(_EVENT_POLLS_TOTAL)
    _reset_counter(_EVENT_FAILURES_TOTAL, reason="error")
    _reset_counter(_EVENT_FAILURES_TOTAL, reason="disabled")

    first = svc.list_events(limit=10)
    assert first == events
    assert _metric_value(_EVENT_CACHE_MISSES) == 1.0
    assert _metric_value(_EVENT_POLLS_TOTAL) == 1.0

    second = svc.list_events(limit=10)
    assert second == events
    assert _metric_value(_EVENT_CACHE_HITS) == 1.0
    assert len(calls) == 1


def test_list_events_handles_poll_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(ttl=0.0)
    calls: list[int] = []

    def _poll(self: DashboardService, *, limit: int) -> list[dict[str, Any]]:
        calls.append(limit)
        raise RuntimeError("events_error")

    monkeypatch.setattr(DashboardService, "_poll_events", _poll)
    svc._event_cache.update(
        [
            {"id": "old", "topic": "STAGE.OLD", "payload": {"source": "live"}},
        ],
    )

    _reset_counter(_EVENT_CACHE_HITS)
    _reset_counter(_EVENT_CACHE_MISSES)
    _reset_counter(_EVENT_POLLS_TOTAL)
    _reset_counter(_EVENT_FAILURES_TOTAL, reason="error")

    observed = svc.list_events(limit=5)
    assert observed[0]["id"] == "old"
    expected_limit = max(5, svc.config.events_cache_max_entries)
    assert calls == [expected_limit]
    assert _metric_value(_EVENT_CACHE_MISSES) == 1.0
    assert _metric_value(_EVENT_FAILURES_TOTAL, reason="error") == 1.0


def test_list_events_filters_by_stage_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service()
    payloads = [
        {"id": "1", "topic": "STAGE.DATA", "payload": {"source": "live"}},
        {"id": "2", "topic": "STAGE.DATA", "payload": {"source": "backfill", "params": {"instrument": "SPY"}}},
    ]

    def _poll(self: DashboardService, *, limit: int) -> list[dict[str, Any]]:
        return payloads

    monkeypatch.setattr(DashboardService, "_poll_events", _poll)
    _reset_counter(_EVENT_CACHE_MISSES)
    _reset_counter(_EVENT_POLLS_TOTAL)

    items = svc.list_events(limit=10, source="backfill", instrument_substr="SPY")
    assert [item["id"] for item in items] == ["2"]
    assert _metric_value(_EVENT_POLLS_TOTAL) == 1.0


def test_list_events_disabled_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service()

    def _disabled(self: DashboardService, *, limit: int) -> list[dict[str, Any]]:
        raise RuntimeError("events_disabled")

    monkeypatch.setattr(DashboardService, "_poll_events", _disabled)
    _reset_counter(_EVENT_FAILURES_TOTAL, reason="disabled")
    _reset_counter(_EVENT_CACHE_MISSES)
    _reset_counter(_EVENT_CACHE_HITS)

    result = svc.list_events(limit=5)
    assert result == []
    assert _metric_value(_EVENT_FAILURES_TOTAL, reason="disabled") == 1.0
