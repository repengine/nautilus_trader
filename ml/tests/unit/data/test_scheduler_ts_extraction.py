from __future__ import annotations

from datetime import UTC, datetime

import pytest

import pandas as pd

from ml.data.scheduler import DataScheduler

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

class _AttrItem:
    def __init__(self, value: int) -> None:
        self.ts_event = value

class _CallableItem:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def ts_event(self) -> datetime:
        return self._value

class _DictLike:
    def __init__(self, value: int) -> None:
        self._value = value

    def to_dict(self) -> dict[str, int]:
        return {"ts_event": self._value}

def test_extract_ts_bounds_mixed_sources() -> None:
    attr_item = _AttrItem(100)
    dict_item = {"ts_event": "250"}
    dict_like = _DictLike(400)

    ts_min, ts_max = DataScheduler._extract_ts_bounds([attr_item, dict_item, dict_like])

    assert ts_min == 100
    assert ts_max == 400

def test_extract_ts_bounds_handles_pandas_timestamp() -> None:
    ts = pd.Timestamp("2024-01-02T03:04:05Z")
    item = _AttrItem(ts)  # type: ignore[arg-type]

    ts_min, ts_max = DataScheduler._extract_ts_bounds([item])

    expected = int(ts.value)
    assert ts_min == expected
    assert ts_max == expected

def test_extract_ts_bounds_handles_callable_returning_datetime() -> None:
    value = datetime(2025, 1, 1, tzinfo=UTC)
    item = _CallableItem(value)

    ts_min, ts_max = DataScheduler._extract_ts_bounds([item])

    expected = DataScheduler._coerce_ns(value)
    assert ts_min == expected
    assert ts_max == expected

def test_extract_ts_bounds_returns_zero_when_missing() -> None:
    class _Empty:
        pass

    result = DataScheduler._extract_ts_bounds([_Empty()])

    assert result == (0, 0)
