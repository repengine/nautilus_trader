#!/usr/bin/env python3

"""
Backward compatibility tests for test data fixtures.

Ensures that existing fixtures (generate_test_bars, sample_features,
sample_predictions) still work after introducing test_data_factory.

"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import numpy as np
import pytest

from nautilus_trader.model.data import Bar


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@pytest.mark.integration
def test_generate_test_bars_still_works(generate_test_bars: list[Bar]) -> None:
    """Verify existing generate_test_bars fixture still works."""
    assert isinstance(generate_test_bars, list)
    assert len(generate_test_bars) > 0
    assert all(isinstance(b, Bar) for b in generate_test_bars)

    # Verify data quality hasn't changed
    for bar in generate_test_bars:
        assert float(bar.high) >= float(bar.low)
        assert float(bar.high) >= float(bar.open)
        assert float(bar.high) >= float(bar.close)


@pytest.mark.integration
def test_sample_features_backward_compatibility(
    sample_features: dict[str, float],
) -> None:
    """Verify sample_features fixture still works."""
    assert isinstance(sample_features, dict)
    assert "price_sma_20" in sample_features
    assert "rsi" in sample_features

    # Values should be reasonable
    assert 0 <= sample_features["rsi"] <= 100


@pytest.mark.integration
def test_sample_predictions_backward_compatibility(
    sample_predictions: np.ndarray,
) -> None:
    """Verify sample_predictions fixture still works."""
    assert isinstance(sample_predictions, np.ndarray)
    assert sample_predictions.shape == (3,)
    assert sample_predictions.dtype == np.float32
