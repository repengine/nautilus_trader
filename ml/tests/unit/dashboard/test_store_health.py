from __future__ import annotations

import datetime as dt
from datetime import datetime

from ml.dashboard.store_health import StoreHealthSummary
from ml.dashboard.store_health import summarize_data_store
from ml.dashboard.store_health import summarize_feature_store


class _FakeCursor:
    def __init__(self, *, first_row: dict[str, int | None] | None = None, rows: list[dict[str, int | str | None]] | None = None) -> None:
        self._first = first_row
        self._rows = rows or []

    def first(self) -> dict[str, int | None] | None:
        return self._first

    def fetchall(self) -> list[dict[str, int | str | None]]:
        return self._rows


class _SequenceEngine:
    def __init__(self, cursors: list[_FakeCursor]) -> None:
        self._cursors = cursors

    def connect(self) -> _SequenceCtx:
        return _SequenceCtx(self)

    def execute(self, *_args: object, **_kwargs: object) -> _FakeCursor:
        if not self._cursors:
            raise RuntimeError("No cursors available")
        return self._cursors.pop(0)


class _SequenceCtx:
    def __init__(self, engine: _SequenceEngine) -> None:
        self._engine = engine

    def __enter__(self) -> _SequenceEngine:
        return self._engine

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
        return False


class _HealthyStore:
    def health_details(self) -> dict[str, object]:
        return {"connectivity_ok": True, "write_ok": True, "buffer_backlog": 0}


def test_summarize_feature_store_success() -> None:
    cursor = _FakeCursor(first_row={"ts_event": 1_000_000_000})
    engine = _SequenceEngine([cursor])
    summary = summarize_feature_store(_HealthyStore(), engine, now=datetime(2025, 1, 1, tzinfo=dt.UTC))
    assert isinstance(summary, StoreHealthSummary)
    assert summary.healthy is True
    assert summary.latest_event_ns == 1_000_000_000
    assert summary.fallback_active is False


def test_summarize_feature_store_fallback_on_error() -> None:
    cursor = _FakeCursor(first_row={"ts_event": "abc"})
    engine = _SequenceEngine([cursor])
    summary = summarize_feature_store(_HealthyStore(), engine)
    assert summary.error == "invalid_timestamp"
    assert summary.healthy is True


def test_summarize_data_store_top_datasets() -> None:
    cursors = [
        _FakeCursor(first_row={"ts_event": 2_000_000_000}),
        _FakeCursor(
            rows=[
                {"dataset_type": "BARS", "ts_event": 2_000_000_000},
                {"dataset_type": "TRADES", "ts_event": 1_500_000_000},
            ],
        ),
    ]
    engine = _SequenceEngine(cursors)
    summary = summarize_data_store(
        engine,
        top_limit=5,
        now=datetime(2025, 1, 1, tzinfo=dt.UTC),
    )
    assert summary.items
    assert summary.items[0].key == "BARS"
    assert summary.healthy is True


def test_summarize_data_store_engine_missing() -> None:
    summary = summarize_data_store(None, top_limit=3)
    assert summary.healthy is False
    assert summary.fallback_active is True
