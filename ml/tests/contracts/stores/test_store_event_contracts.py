from __future__ import annotations

import time
from typing import Any

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


class FakeRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

    def emit_event(
        self,
        *,
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
                "metadata": dict(metadata or {}),
            },
        )

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Source,
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


def _patch_db_init_and_write(monkeypatch: pytest.MonkeyPatch, store: Any) -> None:
    # Avoid DB setup and upserts in unit tests
    monkeypatch.setattr(
        store.__class__,
        "_init_engine_and_tables",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(store, "_execute_write", lambda values: None, raising=False)


def test_strategy_store_registry_event_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup store with no DB and fake registry
    store = StrategyStore(connection_string=None)
    _patch_db_init_and_write(monkeypatch, store)
    reg = FakeRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: reg)

    now = time.time_ns()
    # Group 1 (historical): STRAT1/SPY with two entries
    store.write_signal(
        strategy_id="STRAT1",
        instrument_id="SPY",
        signal_type="BUY",
        strength=0.7,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now,
        is_live=False,
    )
    store.write_signal(
        strategy_id="STRAT1",
        instrument_id="SPY",
        signal_type="SELL",
        strength=0.3,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now + 1,
        is_live=False,
    )
    # Ensure the buffered objects indicate historical
    for obj in store._write_buffer:
        if obj.instrument_id == "SPY":
            setattr(obj, "is_live", False)

    # Group 2 (live): STRAT2/QQQ with two entries
    store.write_signal(
        strategy_id="STRAT2",
        instrument_id="QQQ",
        signal_type="BUY",
        strength=0.6,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now + 2,
        is_live=True,
    )
    store.write_signal(
        strategy_id="STRAT2",
        instrument_id="QQQ",
        signal_type="HOLD",
        strength=0.1,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now + 3,
        is_live=True,
    )
    for obj in store._write_buffer:
        if obj.instrument_id == "QQQ":
            setattr(obj, "is_live", True)

    store.flush()

    # Two grouped events expected (SPY historical, QQQ live)
    assert {e["instrument_id"] for e in reg.events} == {"SPY", "QQQ"}

    # Validate event shapes and enums
    spy_evt = next(e for e in reg.events if e["instrument_id"] == "SPY")
    assert spy_evt["dataset_id"] == "signals"
    assert spy_evt["stage"] is Stage.SIGNAL_EMITTED
    assert spy_evt["source"] is Source.HISTORICAL
    assert spy_evt["status"] is EventStatus.SUCCESS
    assert spy_evt["count"] == 2
    assert spy_evt["ts_min"] == now
    assert spy_evt["ts_max"] == now + 1

    qqq_evt = next(e for e in reg.events if e["instrument_id"] == "QQQ")
    assert qqq_evt["dataset_id"] == "signals"
    assert qqq_evt["stage"] is Stage.SIGNAL_EMITTED
    assert qqq_evt["source"] is Source.LIVE
    assert qqq_evt["status"] is EventStatus.SUCCESS
    assert qqq_evt["count"] == 2
    assert qqq_evt["ts_min"] == now + 2
    assert qqq_evt["ts_max"] == now + 3

    # Watermarks: last_success_ns uses group max timestamp and correct source
    wm_spy = next(w for w in reg.watermarks if w["instrument_id"] == "SPY")
    assert wm_spy["dataset_id"] == "signals"
    assert wm_spy["source"] is Source.HISTORICAL
    assert wm_spy["last_success_ns"] == now + 1
    assert wm_spy["count"] == 2

    wm_qqq = next(w for w in reg.watermarks if w["instrument_id"] == "QQQ")
    assert wm_qqq["dataset_id"] == "signals"
    assert wm_qqq["source"] is Source.LIVE
    assert wm_qqq["last_success_ns"] == now + 3
    assert wm_qqq["count"] == 2


def test_model_store_registry_event_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ModelStore(connection_string=None)
    _patch_db_init_and_write(monkeypatch, store)
    reg = FakeRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: reg)

    now = time.time_ns()
    # Group 1 (historical): model m1 on SPY
    store.write_prediction(
        model_id="m1",
        instrument_id="SPY",
        prediction=0.1,
        confidence=0.9,
        features={},
        inference_time_ms=1.0,
        ts_event=now,
        is_live=False,
    )
    store.write_prediction(
        model_id="m1",
        instrument_id="SPY",
        prediction=0.2,
        confidence=0.8,
        features={},
        inference_time_ms=1.0,
        ts_event=now + 1,
        is_live=False,
    )
    for obj in store._write_buffer:
        if obj.instrument_id == "SPY":
            setattr(obj, "is_live", False)

    # Group 2 (live): model m2 on QQQ
    store.write_prediction(
        model_id="m2",
        instrument_id="QQQ",
        prediction=1.2,
        confidence=0.5,
        features={},
        inference_time_ms=1.0,
        ts_event=now + 2,
        is_live=True,
    )
    store.write_prediction(
        model_id="m2",
        instrument_id="QQQ",
        prediction=1.3,
        confidence=0.6,
        features={},
        inference_time_ms=1.0,
        ts_event=now + 3,
        is_live=True,
    )
    for obj in store._write_buffer:
        if obj.instrument_id == "QQQ":
            setattr(obj, "is_live", True)

    store.flush()

    assert {e["instrument_id"] for e in reg.events} == {"SPY", "QQQ"}

    spy_evt = next(e for e in reg.events if e["instrument_id"] == "SPY")
    assert spy_evt["dataset_id"] == "predictions"
    assert spy_evt["stage"] is Stage.PREDICTION_EMITTED
    assert spy_evt["source"] is Source.HISTORICAL
    assert spy_evt["status"] is EventStatus.SUCCESS
    assert spy_evt["count"] == 2
    assert spy_evt.get("metadata", {}).get("model_id") == "m1"
    assert spy_evt["ts_min"] == now
    assert spy_evt["ts_max"] == now + 1

    qqq_evt = next(e for e in reg.events if e["instrument_id"] == "QQQ")
    assert qqq_evt["dataset_id"] == "predictions"
    assert qqq_evt["stage"] is Stage.PREDICTION_EMITTED
    assert qqq_evt["source"] is Source.LIVE
    assert qqq_evt["status"] is EventStatus.SUCCESS
    assert qqq_evt["count"] == 2
    assert qqq_evt.get("metadata", {}).get("model_id") == "m2"
    assert qqq_evt["ts_min"] == now + 2
    assert qqq_evt["ts_max"] == now + 3

    wm_spy = next(w for w in reg.watermarks if w["instrument_id"] == "SPY")
    assert wm_spy["dataset_id"] == "predictions"
    assert wm_spy["source"] is Source.HISTORICAL
    assert wm_spy["last_success_ns"] == now + 1
    assert wm_spy["count"] == 2

    wm_qqq = next(w for w in reg.watermarks if w["instrument_id"] == "QQQ")
    assert wm_qqq["dataset_id"] == "predictions"
    assert wm_qqq["source"] is Source.LIVE
    assert wm_qqq["last_success_ns"] == now + 3
    assert wm_qqq["count"] == 2
