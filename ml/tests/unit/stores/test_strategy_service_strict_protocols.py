from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import DataContract, DatasetManifest
from ml.registry.protocols import RegistryProtocol
from ml.stores.base import StrategySignal
from ml.stores.services.strategy_services import (
    StrategyOrderEventEventService,
    StrategySignalClearService,
    StrategySignalEventService,
    StrategySignalQueryService,
    StrategySignalStatsService,
    StrategySignalWriteService,
)
from ml.stores.protocols import (
    LoggerLike,
    StrategyClearDepsStrict,
    StrategyEventDepsStrict,
    StrategyReadDepsStrict,
    StrategyWriteDepsStrict,
    TableLike,
)


class _DummyTable(TableLike):
    def __init__(self) -> None:
        class _Cols:
            strategy_id = "strategy_id"
            instrument_id = "instrument_id"

        self.c = _Cols()

    def delete(self) -> None:  # noqa: D401
        return None

    def where(self, *_: Any) -> _DummyTable:  # noqa: D401
        return self


class _DummyEngine:
    @contextmanager
    def begin(self) -> Iterator[_DummyEngine]:
        yield self

    def execute(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None


class _WriteDeps(StrategyWriteDepsStrict):
    def __init__(self) -> None:
        self.strategy_signals_table = _DummyTable()
        self.strategy_order_events_table = _DummyTable()
        self.last_values: list[dict[str, object]] | None = None

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, object]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: TableLike,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str],
        dataset_id: str,
        stage: object,
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: LoggerLike,
        publish_bus: bool = True,
    ) -> None:
        self.last_values = values
        logger.debug(
            "write",
            extra={"ts_event_field": ts_event_field, "count": len(values)},
        )


class _ReadDeps(StrategyReadDepsStrict):
    def __init__(self) -> None:
        self._last_sql: Any = None
        self._last_params: dict[str, object] | None = None

    def _safe_table(self, base: str, allowed: set[str]) -> str:  # noqa: D401
        assert base in allowed
        return base

    def _execute_read(
        self,
        sql: Any,
        params: Mapping[str, object],
        *,
        columns: Sequence[str],
    ) -> object:  # noqa: D401
        self._last_sql = sql
        self._last_params = dict(params)
        return {"columns": list(columns), "params": dict(params)}

    def _fetch_one(
        self,
        sql: Any,
        params: Mapping[str, object],
    ) -> tuple[object, ...] | None:  # noqa: D401
        # return dummy stats row: counts + min/max ts
        return (10, 2, 3, 4, 5, 6, 1.5, 100, 200)

    def _fetch_all(
        self,
        sql: Any,
        params: Mapping[str, object],
    ) -> list[tuple[object, ...]]:  # noqa: D401
        return [("BUY", 5), ("SELL", 3)]


class _EventRegistry(RegistryProtocol):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self._manifests: set[str] = set()

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            (
                dataset_id,
                {
                    "instrument_id": instrument_id,
                    "stage": stage,
                    "source": source,
                    "run_id": run_id,
                    "ts_min": ts_min,
                    "ts_max": ts_max,
                    "count": count,
                    "status": status,
                    "error": error,
                    "metadata": metadata,
                },
            ),
        )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.events.append(
            (
                dataset_id + ":wm",
                {
                    "instrument_id": instrument_id,
                    "source": source,
                    "last_success_ns": last_success_ns,
                    "count": count,
                    "completeness_pct": completeness_pct,
                },
            ),
        )

    def get_manifest(self, dataset_id: str) -> DatasetManifest:  # noqa: D401
        if dataset_id not in self._manifests:
            raise ValueError("not registered")
        return cast(DatasetManifest, object())

    def get_contract(self, dataset_id: str) -> DataContract:  # noqa: D401
        return cast(DataContract, object())

    def register_dataset(self, manifest: object) -> str:  # noqa: D401
        self._manifests.add(getattr(manifest, "dataset_id", "unknown"))
        return "ok"

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        del dataset_id
        del changes
        return None

    def get_contracts(self) -> list[object]:  # pragma: no cover - optional helper
        return []


class _EventDeps(StrategyEventDepsStrict):
    def __init__(self) -> None:
        self._registry = _EventRegistry()

    def _get_data_registry(self) -> RegistryProtocol | None:  # noqa: D401
        return self._registry


class _ClearDeps(StrategyClearDepsStrict):
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

    oes = StrategyOrderEventEventService(eeps, logging.getLogger(__name__))
    oes.emit_order_events([])  # no error on empty input
