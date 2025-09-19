from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import pytest

from ml.stores.base import StrategySignal
from ml.stores.services.strategy_services import (
    StrategySignalClearService,
    StrategySignalEventService,
    StrategySignalQueryService,
    StrategySignalStatsService,
    StrategySignalWriteService,
)


class _DummyTable:
    def __init__(self) -> None:
        class _Cols:
            strategy_id = "strategy_id"
            instrument_id = "instrument_id"

        self.c = _Cols()

    def delete(self) -> _DummyTable:  # noqa: D401
        return self

    def where(self, *_: Any) -> _DummyTable:  # noqa: D401
        return self


class _DummyEngine:
    @contextmanager
    def begin(self):  # type: ignore[no-untyped-def]
        yield self

    def execute(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None


class _WriteDeps:
    def __init__(self) -> None:
        self.strategy_signals_table = _DummyTable()
        self.last_values: list[dict[str, object]] | None = None

    def _execute_upsert_and_publish(self, *, values, **_: Any) -> None:  # type: ignore[no-untyped-def]
        self.last_values = values


class _ReadDeps:
    def __init__(self) -> None:
        self._last_sql: Any = None
        self._last_params: dict[str, object] | None = None

    def _safe_table(self, base: str, allowed: set[str]) -> str:  # noqa: D401
        assert base in allowed
        return base

    def _execute_read(self, sql: Any, params: dict[str, object], *, columns: list[str]) -> object:  # noqa: D401
        self._last_sql = sql
        self._last_params = params
        return {"columns": columns, "params": params}

    def _fetch_one(self, sql: Any, params: dict[str, object]) -> tuple[object, ...] | None:  # noqa: D401
        # return dummy stats row: counts + min/max ts
        return (10, 2, 3, 4, 5, 6, 1.5, 100, 200)

    def _fetch_all(self, sql: Any, params: dict[str, object]) -> list[tuple[object, ...]]:  # noqa: D401
        return [("BUY", 5), ("SELL", 3)]


class _EventRegistry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self._manifests: set[str] = set()

    def emit_event(self, *, dataset_id: str, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        self.events.append((dataset_id, dict(kwargs)))

    def update_watermark(self, *, dataset_id: str, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        self.events.append((dataset_id + ":wm", dict(kwargs)))

    def get_manifest(self, dataset_id: str) -> object:  # noqa: D401
        if dataset_id not in self._manifests:
            raise ValueError("not registered")
        return object()

    def register_dataset(self, manifest: object) -> str:  # noqa: D401
        self._manifests.add(getattr(manifest, "dataset_id", "unknown"))
        return "ok"


class _EventDeps:
    def __init__(self) -> None:
        self._registry = _EventRegistry()

    def _get_data_registry(self) -> _EventRegistry | None:  # noqa: D401
        return self._registry


class _ClearDeps:
    def __init__(self) -> None:
        self.engine = _DummyEngine()
        self.strategy_signals_table = _DummyTable()


def test_strategy_write_service_strict_protocol_roundtrip() -> None:
    deps = _WriteDeps()
    svc = StrategySignalWriteService(deps, logging.getLogger(__name__))
    ss = StrategySignal(
        strategy_id="s1",
        instrument_id="i1",
        signal_type="BUY",
        strength=0.9,
        model_predictions={"m": 0.7},
        risk_metrics={"r": 1.0},
        execution_params={"e": 1},
        _ts_event=1,
        _ts_init=1,
    )
    svc.write_batch([ss])
    assert deps.last_values is not None
    assert deps.last_values[0]["strategy_id"] == "s1"


def test_strategy_read_services_strict_protocol_basic() -> None:
    deps = _ReadDeps()
    q = StrategySignalQueryService(deps)
    res = q.read_range(start_ns=0, end_ns=10)
    assert isinstance(res, object)

    s = StrategySignalStatsService(deps)
    stats = s.get_statistics()
    assert stats["total_signals"] == 10


def test_strategy_clear_and_events_strict_protocol_smoke() -> None:
    ceps = _ClearDeps()
    cs = StrategySignalClearService(ceps)
    cs.clear()  # no error

    eeps = _EventDeps()
    es = StrategySignalEventService(eeps, logging.getLogger(__name__))
    es.emit_signal_events([])  # no error on empty input

