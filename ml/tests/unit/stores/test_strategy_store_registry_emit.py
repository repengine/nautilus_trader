from __future__ import annotations

import time
from typing import Any

import pytest

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
        stage: Any,
        source: Any,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: Any,
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


def test_strategy_store_emits_registry_events_on_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid DB setup by no-op'ing engine/table init and execute path
    monkeypatch.setattr(StrategyStore, "_init_engine_and_tables", lambda self: None)
    store = StrategyStore(connection_string=None)
    # Stub execute to avoid DB; rely on flush's registry emission only
    monkeypatch.setattr(store, "_execute_write", lambda values: None)

    fake = FakeRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: fake)

    now = time.time_ns()
    # Append two buffered items
    store.write_signal(
        strategy_id="STRAT1",
        instrument_id="SPY",
        signal_type="BUY",
        strength=0.5,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        decision_metadata={"version": "v1"},
        ts_event=now,
        is_live=False,
    )
    store.write_signal(
        strategy_id="STRAT1",
        instrument_id="SPY",
        signal_type="SELL",
        strength=0.4,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        decision_metadata={"version": "v1"},
        ts_event=now + 1,
        is_live=False,
    )

    # Flush to trigger registry emission
    store.flush()

    assert len(fake.events) == 1  # grouped by (strategy,instrument)
    evt = fake.events[0]
    assert evt["dataset_id"] == "signals"
    assert evt["instrument_id"] == "SPY"
    assert evt["count"] == 2
    # Watermark updated once
    assert len(fake.watermarks) == 1
