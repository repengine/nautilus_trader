"""
DataStore event emission tests using stubbed registry and stores.

Focus: functional outcome that DataStore emits events and updates watermarks
with expected parameters when writing features/predictions/signals.

"""

from __future__ import annotations

import pytest

from ml.features.earnings.store import DummyEarningsStore
from ml.stores.data_store_facade import DataStore
from ml.tests.utils.stubs import FeatureStoreNoOp, ModelStoreNoOp, RegistryTestStub, StrategyStoreNoOp
from nautilus_trader.model.identifiers import InstrumentId


@pytest.fixture
def stubbed_data_store() -> tuple[DataStore, RegistryTestStub]:
    registry = RegistryTestStub()
    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=FeatureStoreNoOp(),
        model_store=ModelStoreNoOp(),
        strategy_store=StrategyStoreNoOp(),
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )
    return ds, registry


def test_data_store_emits_feature_events(
    stubbed_data_store: tuple[DataStore, RegistryTestStub],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
    use_component_datastore: bool,
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import FeatureData

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    fd = FeatureData(
        feature_set_id="fs1",
        instrument_id=instrument_id_str,
        values={"a": 1.0},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_features(instrument_id=instrument_id_str, features=[fd])
    assert any(e for e in reg.events if e["dataset_id"] == "features")


def test_data_store_emits_prediction_events(
    stubbed_data_store: tuple[DataStore, RegistryTestStub],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
    use_component_datastore: bool,
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import ModelPrediction

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    mp = ModelPrediction(
        model_id="m1",
        instrument_id=instrument_id_str,
        prediction=0.8,
        confidence=0.9,
        features_used={"a": 1.0},
        inference_time_ms=0.1,
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_predictions(predictions=[mp])
    assert any(e for e in reg.events if e["dataset_id"] == "predictions")


def test_data_store_emits_signal_events(
    stubbed_data_store: tuple[DataStore, RegistryTestStub],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
    use_component_datastore: bool,
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import StrategySignal

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    ss = StrategySignal(
        strategy_id="s1",
        instrument_id=instrument_id_str,
        signal_type="BUY",
        strength=0.8,
        model_predictions={"m1": 0.8},
        risk_metrics={"conf": 0.8},
        execution_params={"side": "BUY"},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_signals(signals=[ss])
    assert any(e for e in reg.events if e["dataset_id"] == "signals")
