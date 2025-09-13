"""
ModelStore registry event emission on flush (DB-free via monkeypatch).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ml.stores.model_store import ModelStore


class _FakeRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Any,
        source: Any,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": count,
                "status": status,
                "metadata": metadata or {},
            },
        )

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Any,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "source": source,
                "last_success_ns": last_success_ns,
                "count": count,
                "completeness_pct": completeness_pct,
            },
        )


def test_model_store_emits_registry_events_on_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid DB setup by no-op init and upsert execution
    monkeypatch.setattr(ModelStore, "_init_engine_and_tables", lambda self: None)
    store = ModelStore(connection_string=None)
    monkeypatch.setattr(store, "_execute_write", lambda values: None)

    fake = _FakeRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: fake)

    now = time.time_ns()
    # Append two predictions (same instrument/model)
    store.write_prediction(
        model_id="MODEL1",
        instrument_id="SPY",
        prediction=0.5,
        confidence=0.8,
        features={},
        inference_time_ms=2.0,
        ts_event=now,
        is_live=True,
    )
    store.write_prediction(
        model_id="MODEL1",
        instrument_id="SPY",
        prediction=0.6,
        confidence=0.7,
        features={},
        inference_time_ms=2.2,
        ts_event=now + 1,
        is_live=True,
    )

    # Flush to trigger registry emission via BufferedStoreMixin
    store.flush()

    assert len(fake.events) == 1  # grouped by (model,instrument)
    evt = fake.events[0]
    assert evt["dataset_id"] == "predictions"
    assert evt["instrument_id"] == "SPY"
    assert evt["count"] == 2
    assert len(fake.watermarks) == 1

