"""
Unit tests for DataStore canonical dataset IDs for events/watermarks.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ml.stores.base import FeatureData, ModelPrediction, StrategySignal
from ml.stores.data_store import DataStore


class _MockRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

    def emit_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def update_watermark(self, **kwargs: Any) -> None:
        self.watermarks.append(kwargs)


def test_data_store_canonical_ids_for_events(monkeypatch: Any) -> None:
    registry = _MockRegistry()

    # Use simple mocks for underlying stores to avoid DB usage
    class _Dummy:
        def __getattr__(self, _name: str) -> Any:  # pragma: no cover - trivial
            return lambda *a, **k: None

    feature_store = _Dummy()
    model_store = _Dummy()
    strategy_store = _Dummy()

    # Use DATABASE_URL from environment or fall back to test database
    connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_test")
    
    store = DataStore(
        registry=registry,  # type: ignore[arg-type]
        connection_string=connection_string,
        feature_store=feature_store,  # type: ignore[arg-type]
        model_store=model_store,  # type: ignore[arg-type]
        strategy_store=strategy_store,  # type: ignore[arg-type]
    )

    # Avoid auto-registration path during unit test
    monkeypatch.setattr(store, "_ensure_dataset_registered", lambda *a, **k: None)  # type: ignore[misc]

    ts = int(time.time() * 1e9)

    # Features
    fd = FeatureData(
        feature_set_id="fs_test",
        instrument_id="EUR/USD",
        values={"f": 1.0},
        _ts_event=ts,
        _ts_init=ts,
    )
    store.write_features("EUR/USD", [fd], source="computed")
    assert any(e.get("dataset_id") == "features" for e in registry.events)

    # Predictions
    mp = ModelPrediction(
        model_id="m1",
        instrument_id="EUR/USD",
        prediction=0.1,
        confidence=0.9,
        features_used={"f": 1.0},
        inference_time_ms=0.5,
        _ts_event=ts + 1,
        _ts_init=ts + 1,
    )
    store.write_predictions([mp], source="inference")
    assert any(e.get("dataset_id") == "predictions" for e in registry.events)

    # Signals
    sig = StrategySignal(
        strategy_id="s1",
        instrument_id="EUR/USD",
        signal_type="BUY",
        strength=0.7,
        model_predictions={"m1": 0.1},
        risk_metrics={"risk": 0.2},
        execution_params={},
        _ts_event=ts + 2,
        _ts_init=ts + 2,
    )
    store.write_signals([sig], source="strategy")
    assert any(e.get("dataset_id") == "signals" for e in registry.events)
