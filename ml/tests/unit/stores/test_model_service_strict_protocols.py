from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import DataContract, DatasetManifest
from ml.registry.protocols import RegistryProtocol
from ml.stores.base import ModelPrediction
from ml.stores.services.model_services import (
    ModelClearService,
    ModelEventService,
    ModelQueryService,
    ModelStatsService,
    ModelWriteService,
)
from ml.stores.protocols import (
    LoggerLike,
    ModelClearDepsStrict,
    ModelEventDepsStrict,
    ModelReadDepsStrict,
    ModelWriteDepsStrict,
    TableLike,
)


class _DummyTable(TableLike):
    def __init__(self) -> None:
        class _Cols:
            model_id = "model_id"
            instrument_id = "instrument_id"

        self.c = _Cols()

    def delete(self) -> None:  # noqa: D401
        return None

    # Support chaining of .where(...)
    def where(self, *_: Any) -> _DummyTable:  # noqa: D401
        return self


class _DummyEngine:
    @contextmanager
    def begin(self) -> Iterator[_DummyEngine]:
        yield self

    def execute(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None


class _LoggerStub(LoggerLike):  # pragma: no cover - trivial
    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        return None

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        return None

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        return None

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        return None


class _WriteDeps(ModelWriteDepsStrict):
    def __init__(self) -> None:
        self.model_predictions_table = _DummyTable()
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
        logger.debug("model_write", extra={"count": len(values)})


class _ReadDeps(ModelReadDepsStrict):
    def __init__(self) -> None:
        self._last_sql: Any = None
        self._last_params: dict[str, object] | None = None

    def _qualified_table(self, base: str) -> str:  # noqa: D401
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
        return (10, 2, 3, 1.5, 5.0, 100, 200)


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


class _EventDeps(ModelEventDepsStrict):
    def __init__(self) -> None:
        self._registry = _EventRegistry()

    def _get_data_registry(self) -> RegistryProtocol | None:  # noqa: D401
        return self._registry


class _ClearDeps(ModelClearDepsStrict):
    def __init__(self) -> None:
        self.engine = _DummyEngine()
        self.model_predictions_table = _DummyTable()


def test_model_write_service_strict_protocol_roundtrip() -> None:
    deps = _WriteDeps()
    svc = ModelWriteService(deps, logging.getLogger(__name__))
    mp = ModelPrediction(
        model_id="m1",
        instrument_id="i1",
        prediction=0.7,
        confidence=0.9,
        features_used={"f": 1.0},
        inference_time_ms=1.2,
        _ts_event=1,
        _ts_init=1,
        is_live=False,
    )
    svc.write_batch([mp])
    assert deps.last_values is not None
    assert deps.last_values[0]["model_id"] == "m1"


def test_model_read_services_strict_protocol_basic() -> None:
    deps = _ReadDeps()
    q = ModelQueryService(deps)
    res = q.read_range(start_ns=0, end_ns=10)
    assert isinstance(res, object)

    s = ModelStatsService(deps)
    stats = s.get_statistics()
    assert stats["total_predictions"] == 10


def test_model_clear_and_events_strict_protocol_smoke() -> None:
    # Ensure these services accept strict deps and do not raise
    ceps = _ClearDeps()
    cs = ModelClearService(ceps)
    cs.clear()  # no error

    eeps = _EventDeps()
    es = ModelEventService(eeps, logging.getLogger(__name__))
    es.emit_prediction_events([])  # no error on empty input
