from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ml.config.events import Stage
from ml.stores.base import ModelPrediction
from ml.stores.services.model_services import ModelQueryService, ModelWriteService
from ml.stores.protocols import (
    LoggerLike,
    ModelReadDepsStrict,
    ModelWriteDepsStrict,
    TableLike,
)


class _TableStub(TableLike):
    def __init__(self) -> None:
        self.c = object()

    def delete(self) -> None:  # pragma: no cover - unused
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


class _FakeWriteDeps(ModelWriteDepsStrict):
    def __init__(self) -> None:
        self.model_predictions_table: TableLike = _TableStub()
        self.last_call: dict[str, Any] | None = None

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
        self.last_call = {
            "values": values,
            "ts_event_field": ts_event_field,
            "ts_init_field": ts_init_field,
            "context": context,
            "key_fields": key_fields,
            "table": table,
            "conflict_cols": list(conflict_cols),
            "update_cols": list(update_cols),
            "dataset_id": dataset_id,
            "stage": stage,
            "instrument_key": instrument_key,
            "ts_field": ts_field,
            "run_id_batch": run_id_batch,
            "run_id_row": run_id_row,
            "source": source,
            "publish_bus": publish_bus,
        }
        logger.debug("model_write", extra={"count": len(values)})


class _FakeReadDeps(ModelReadDepsStrict):
    def __init__(self) -> None:
        self.qualified_bases: list[str] = []
        self.last_execute: dict[str, Any] | None = None

    def _qualified_table(self, base: str) -> str:  # noqa: D401
        self.qualified_bases.append(base)
        return base

    def _execute_read(
        self,
        sql: Any,
        params: Mapping[str, object],
        *,
        columns: Sequence[str],
    ) -> Any:  # noqa: D401
        self.last_execute = {
            "sql": str(sql),
            "params": dict(params),
            "columns": list(columns),
        }

        class _DF:
            pass

        return _DF()

    def _fetch_one(
        self,
        sql: Any,
        params: Mapping[str, object],
    ) -> tuple[Any, ...] | None:  # pragma: no cover
        return None


def test_model_write_service_calls_upsert_and_publish() -> None:
    deps = _FakeWriteDeps()
    svc = ModelWriteService(deps, logger=_LoggerStub())
    pred = ModelPrediction(
        model_id="m",
        instrument_id="i",
        prediction=0.1,
        confidence=0.2,
        features_used={"f": 1.0},
        inference_time_ms=1.5,
        _ts_event=1,
        _ts_init=2,
    )
    svc.write_batch([pred])

    assert deps.last_call is not None
    assert deps.last_call["dataset_id"] == "predictions"
    assert deps.last_call["stage"] == Stage.PREDICTION_EMITTED
    assert deps.last_call["key_fields"] == ("model_id", "instrument_id", "ts_event")


def test_model_query_service_read_range_columns() -> None:
    deps = _FakeReadDeps()
    svc = ModelQueryService(deps)
    _ = svc.read_range(start_ns=0, end_ns=5, instrument_id=None)
    assert deps.qualified_bases == ["ml_model_predictions"]
    assert deps.last_execute is not None
    cols = deps.last_execute["columns"]
    assert cols == [
        "model_id",
        "instrument_id",
        "ts_event",
        "prediction",
        "confidence",
        "inference_time_ms",
    ]
