#!/usr/bin/env python3

"""
Test to validate the new fixture system works correctly.
"""

import pytest
from pathlib import Path

from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from ml.tests.builders import MLConfigBuilder, MockBuilder, DataBuilder


def test_default_fixtures(default_bar_type, default_instrument_id):
    """
    Test that default fixtures work.
    """
    assert isinstance(default_bar_type, BarType)
    assert str(default_bar_type) == "EUR/USD.SIM-1-MINUTE-MID-INTERNAL"

    assert isinstance(default_instrument_id, InstrumentId)
    assert str(default_instrument_id) == "EUR/USD.SIM"


def test_ml_config_fixtures(base_ml_config, base_signal_config):
    """
    Test ML configuration fixtures.
    """
    assert isinstance(base_ml_config, MLActorConfig)
    assert base_ml_config.model_id == "test_model"
    assert base_ml_config.warm_up_period == 10
    assert base_ml_config.batch_size == 1

    assert isinstance(base_signal_config, MLSignalActorConfig)
    assert base_signal_config.model_id == "test_signal_model"
    assert base_signal_config.prediction_threshold == 0.5
    assert base_signal_config.signal_strategy == "threshold"


def test_model_fixtures(dummy_onnx_model, dummy_xgboost_model):
    """
    Test model file fixtures.
    """
    assert isinstance(dummy_onnx_model, Path)
    assert dummy_onnx_model.exists()
    assert dummy_onnx_model.suffix == ".onnx"

    assert isinstance(dummy_xgboost_model, Path)
    assert dummy_xgboost_model.exists()
    assert dummy_xgboost_model.suffix == ".json"


def test_mock_fixtures(mock_model_registry, mock_feature_registry, mock_data_store):
    """
    Test mock fixtures.
    """
    # Test model registry mock
    model_info = mock_model_registry.get_model("test_model")
    assert model_info.manifest.model_id == "test_model_v1"
    assert model_info.manifest.version == "1.0.0"

    # Test feature registry mock
    feature_info = mock_feature_registry.get_feature_set("test_features")
    assert feature_info.manifest.feature_set_id == "test_features_v1"
    assert "price_sma_20" in feature_info.manifest.feature_names  # Use correct field name

    # Test data store mock
    assert mock_data_store.write("test", {}) is True
    assert mock_data_store.read_features() == {}


def test_config_builder():
    """
    Test MLConfigBuilder.
    """
    # Test actor config builder
    config = MLConfigBuilder.actor_config(model_id="custom_model")
    assert config.model_id == "custom_model"
    assert isinstance(config.bar_type, BarType)

    # Test signal config builder
    signal_config = MLConfigBuilder.signal_config(prediction_threshold=0.7)
    assert signal_config.prediction_threshold == 0.7
    assert signal_config.feature_config is not None

    # Test strategy config builder
    strategy_config = MLConfigBuilder.strategy_config(max_positions=5)
    assert strategy_config.max_positions == 5
    assert strategy_config.execute_trades is False  # Safe default


def test_mock_builder():
    """
    Test MockBuilder.
    """
    # Test model registry builder
    registry = MockBuilder.model_registry(model_id="custom_model", version="2.0.0")
    model_info = registry.get_model("custom_model")
    assert model_info.manifest.model_id == "custom_model"
    assert model_info.manifest.version == "2.0.0"

    # Test feature registry builder
    feature_reg = MockBuilder.feature_registry(feature_names=["custom_feature"])
    feature_info = feature_reg.get_feature_set("test")
    assert "custom_feature" in feature_info.manifest.feature_names

    # Test all registries bundle
    registries = MockBuilder.all_registries()
    assert "model_registry" in registries
    assert "feature_registry" in registries
    assert "strategy_registry" in registries


def test_data_builder():
    """
    Test DataBuilder.
    """
    # Test feature data generation
    features = DataBuilder.feature_data(n_samples=50, n_features=5)
    assert features.shape == (50, 5)

    # Test predictions generation
    predictions = DataBuilder.predictions(n_samples=100, bounded=True)
    assert len(predictions) == 100
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0

    # Test OHLCV data generation
    ohlcv = DataBuilder.ohlcv_data(n_bars=20, as_dataframe=True)
    assert len(ohlcv) == 20
    assert all(col in ohlcv.columns for col in ["open", "high", "low", "close", "volume"])

    # Test signal data generation
    signals = DataBuilder.signal_data(n_signals=5)
    assert len(signals) == 5
    assert all("instrument_id" in s for s in signals)
    assert all("confidence" in s for s in signals)


def test_sample_data_fixtures(sample_features, sample_predictions, test_timestamps):
    """
    Test sample data fixtures.
    """
    assert isinstance(sample_features, dict)
    assert "price_sma_20" in sample_features
    assert "rsi" in sample_features

    assert len(sample_predictions) == 3
    assert sample_predictions[0] == 0.65

    ts_event, ts_init = test_timestamps
    assert ts_init > ts_event
    assert isinstance(ts_event, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
