from __future__ import annotations

from typing import Any

from ml.config.events import Stage
from ml.stores.base import ModelPrediction
from ml.stores.services.model_services import ModelQueryService, ModelWriteService


class _FakeWriteDeps:
    def __init__(self) -> None:
        self.model_predictions_table = object()
        self.last_call: dict[str, Any] | None = None

    def _execute_upsert_and_publish(self, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        self.last_call = kwargs


class _FakeReadDeps:
    def __init__(self) -> None:
        self.qualified_bases: list[str] = []
        self.last_execute: dict[str, Any] | None = None

    def _qualified_table(self, base: str) -> str:  # noqa: D401
        self.qualified_bases.append(base)
        return base

    def _execute_read(self, sql: Any, params: dict[str, Any], *, columns: list[str]) -> Any:  # noqa: D401
        self.last_execute = {"sql": str(sql), "params": params, "columns": columns}
        class _DF:
            pass

        return _DF()

    def _fetch_one(self, sql: Any, params: dict[str, Any]) -> tuple[Any, ...] | None:  # pragma: no cover
        return None


def test_model_write_service_calls_upsert_and_publish() -> None:
    deps = _FakeWriteDeps()
    svc = ModelWriteService(deps, logger=object())
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

