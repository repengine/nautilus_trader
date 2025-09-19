from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import pytest

from ml.stores.base import ModelPrediction
from ml.stores.services.model_services import (
    ModelClearService,
    ModelEventService,
    ModelQueryService,
    ModelStatsService,
    ModelWriteService,
)


class _DummyTable:
    def __init__(self) -> None:
        class _Cols:
            model_id = "model_id"
            instrument_id = "instrument_id"

        self.c = _Cols()

    def delete(self) -> _DummyTable:  # noqa: D401
        return self

    # Support chaining of .where(...)
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
        self.model_predictions_table = _DummyTable()
        self.last_values: list[dict[str, object]] | None = None

    def _execute_upsert_and_publish(self, *, values, **_: Any) -> None:  # type: ignore[no-untyped-def]
        self.last_values = values


class _ReadDeps:
    def __init__(self) -> None:
        self._last_sql: Any = None
        self._last_params: dict[str, object] | None = None

    def _qualified_table(self, base: str) -> str:  # noqa: D401
        return base

    def _execute_read(self, sql: Any, params: dict[str, object], *, columns: list[str]) -> object:  # noqa: D401
        self._last_sql = sql
        self._last_params = params
        return {"columns": columns, "params": params}

    def _fetch_one(self, sql: Any, params: dict[str, object]) -> tuple[object, ...] | None:  # noqa: D401
        # return dummy stats row: counts + min/max ts
        return (10, 2, 3, 1.5, 5.0, 100, 200)


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

