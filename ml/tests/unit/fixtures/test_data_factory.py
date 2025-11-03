#!/usr/bin/env python3

"""
Unit tests for TestDataFactory methods.

Tests the bars(), features(), and predictions() methods of the TestDataFactory
class for correct behavior, error handling, and edge cases.

"""

from __future__ import annotations

import numpy as np
import pytest

from ml.tests.fixtures.model_factory import TestDataFactory
from nautilus_trader.model.data import Bar


# ============================================================================
# Happy Path Tests
# ============================================================================


def test_test_data_factory_bars_returns_list_of_bars() -> None:
    """Verify bars() method returns valid Bar objects."""
    factory = TestDataFactory()
    bars = factory.bars(n=10)

    assert isinstance(bars, list)
    assert len(bars) == 10
    assert all(isinstance(b, Bar) for b in bars)

    # Verify OHLC relationships
    for bar in bars:
        assert float(bar.high) >= float(bar.open)
        assert float(bar.high) >= float(bar.close)
        assert float(bar.low) <= float(bar.open)
        assert float(bar.low) <= float(bar.close)
        assert float(bar.high) >= float(bar.low)

    # Verify timestamp monotonicity
    timestamps = [b.ts_event for b in bars]
    assert timestamps == sorted(timestamps)


def test_test_data_factory_bars_accepts_custom_instrument() -> None:
    """Verify bars() can generate for different instruments."""
    factory = TestDataFactory()

    eurusd_bars = factory.bars(n=5, instrument_id="EUR/USD.SIM")
    btcusd_bars = factory.bars(n=5, instrument_id="BTC/USD.SIM")

    assert eurusd_bars[0].bar_type.instrument_id.value == "EUR/USD.SIM"
    assert btcusd_bars[0].bar_type.instrument_id.value == "BTC/USD.SIM"


def test_test_data_factory_features_returns_array() -> None:
    """Verify features() method returns numpy array."""
    factory = TestDataFactory()
    features = factory.features(n=50, n_features=10)

    assert isinstance(features, np.ndarray)
    assert features.shape == (50, 10)
    assert features.dtype == np.float32
    assert not np.any(np.isnan(features))
    assert np.abs(features).max() < 100  # Reasonable range


def test_test_data_factory_features_uses_seed_for_reproducibility() -> None:
    """Verify seed parameter produces reproducible results."""
    factory = TestDataFactory()

    features1 = factory.features(n=20, seed=42)
    features2 = factory.features(n=20, seed=42)
    features3 = factory.features(n=20, seed=99)

    np.testing.assert_array_equal(features1, features2)
    assert not np.allclose(features1, features3)


def test_test_data_factory_predictions_returns_list_of_dicts() -> None:
    """Verify predictions() method returns valid predictions."""
    factory = TestDataFactory()
    predictions = factory.predictions(n=20, instrument="EUR/USD.SIM")

    assert isinstance(predictions, list)
    assert len(predictions) == 20

    for pred in predictions:
        assert isinstance(pred, dict)
        assert "instrument_id" in pred
        assert "timestamp" in pred
        assert "prediction" in pred
        assert "confidence" in pred

        # Validate ranges
        assert -1 <= pred["prediction"] <= 1
        assert 0 <= pred["confidence"] <= 1
        assert pred["timestamp"] > 0


# ============================================================================
# Error Condition Tests
# ============================================================================


def test_test_data_factory_bars_rejects_invalid_n() -> None:
    """Verify bars() validates n parameter."""
    factory = TestDataFactory()

    with pytest.raises(ValueError, match="n must be greater than 0"):
        factory.bars(n=0)

    with pytest.raises(ValueError, match="n must be greater than 0"):
        factory.bars(n=-1)


def test_test_data_factory_features_rejects_invalid_dimensions() -> None:
    """Verify features() validates dimensions."""
    factory = TestDataFactory()

    with pytest.raises(ValueError, match="n_samples must be greater than 0"):
        factory.features(n=0, n_features=10)

    with pytest.raises(ValueError, match="n_features must be greater than 0"):
        factory.features(n=10, n_features=0)


def test_test_data_factory_predictions_rejects_invalid_instrument() -> None:
    """Verify predictions() validates instrument parameter."""
    factory = TestDataFactory()

    with pytest.raises(ValueError, match="instrument must not be empty"):
        factory.predictions(n=20, instrument="")


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_test_data_factory_bars_single_bar() -> None:
    """Verify bars() works with n=1."""
    factory = TestDataFactory()
    bars = factory.bars(n=1)

    assert len(bars) == 1
    assert isinstance(bars[0], Bar)
    assert float(bars[0].high) >= float(bars[0].low)


def test_test_data_factory_features_minimal_dimensions() -> None:
    """Verify features() works with n=1, n_features=1."""
    factory = TestDataFactory()
    features = factory.features(n=1, n_features=1)

    assert features.shape == (1, 1)
    assert isinstance(features[0, 0], (np.floating, float))


def test_test_data_factory_predictions_single_prediction() -> None:
    """Verify predictions() handles edge cases gracefully."""
    factory = TestDataFactory()
    predictions = factory.predictions(n=1)

    assert len(predictions) == 1
    assert "prediction" in predictions[0]
    assert -1 <= predictions[0]["prediction"] <= 1
