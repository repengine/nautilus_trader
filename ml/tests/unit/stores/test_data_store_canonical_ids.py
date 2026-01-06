"""
Unit tests for DataStore canonical dataset IDs for events/watermarks.
"""

from __future__ import annotations

import os
import time
from typing import Any
from types import MethodType

import pytest

from typing import cast

from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_store_facade import DataStore
from ml.stores.feature_store_facade import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.utils.stubs import FeatureStoreNoOp, ModelStoreNoOp, RegistryTestStub, StrategyStoreNoOp


@pytest.mark.unit
def test_data_store_canonical_ids_for_events(monkeypatch: Any) -> None:
    registry = RegistryTestStub()
    feature_store = cast(FeatureStore, FeatureStoreNoOp())
    model_store = cast(ModelStore, ModelStoreNoOp())
    strategy_store = cast(StrategyStore, StrategyStoreNoOp())

    # Use an in-memory SQLite engine to keep this unit test DB-free
    connection_string = "sqlite:///:memory:"

    store = DataStore(
        registry=registry,
        connection_string=connection_string,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
    )

    # Avoid auto-registration path during unit test
    def _noop(_self: DataStore, *_args: object, **_kwargs: object) -> None:
        return None

    cast(Any, store)._ensure_dataset_registered = MethodType(_noop, store)

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
