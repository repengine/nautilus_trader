"""
Property tests for fixture invariants across environments.

This module uses hypothesis to test fixture properties hold regardless of
HAS_PROMETHEUS value and across multiple invocations.

Tests verify environment-independent behavior and consistent patching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Mark all tests in this module as property tests (excluded from parallel runs)
pytestmark = pytest.mark.property


class TestMockPrometheusProperties:
    """Property tests for mock_prometheus_when_unavailable fixture."""

    @given(test_run=st.integers(min_value=1, max_value=10))
    def test_fixture_isolation_across_runs(
        self, test_run: int, mock_prometheus_when_unavailable: Any
    ) -> None:
        """
        Verify fixture provides consistent behavior across multiple test runs.

        Property: Fixture behavior is deterministic and isolated per test.
        """
        from ml._imports import HAS_PROMETHEUS

        if not HAS_PROMETHEUS:
            # Should always be a dict with mocks
            assert isinstance(mock_prometheus_when_unavailable, dict)
            assert "Counter" in mock_prometheus_when_unavailable
            assert "Gauge" in mock_prometheus_when_unavailable
            assert "Histogram" in mock_prometheus_when_unavailable
        else:
            # Should always be None
            assert mock_prometheus_when_unavailable is None

        # Property holds regardless of test_run value


class TestFixtureValueProperties:
    """Property tests for data fixture values."""

    @given(key=st.sampled_from(["price_sma_20", "rsi", "volume_ratio_20", "price_change", "volatility"]))
    def test_sample_features_keys_always_present(
        self, key: str, sample_features: dict[str, float]
    ) -> None:
        """
        Verify all expected keys are always present in sample_features.

        Property: Fixture always contains complete set of expected keys.
        """
        assert key in sample_features, f"Key '{key}' missing from sample_features"
        assert isinstance(sample_features[key], (float, int))

    @given(index=st.integers(min_value=0, max_value=2))
    def test_sample_predictions_valid_range(
        self, index: int, sample_predictions: np.ndarray
    ) -> None:
        """
        Verify all prediction values are in valid range.

        Property: All predictions are bounded within [0, 1].
        """
        assert 0.0 <= sample_predictions[index] <= 1.0, (
            f"Prediction at index {index} out of range: {sample_predictions[index]}"
        )

    @given(
        feature_dict=st.fixed_dictionaries(
            {
                "price_sma_20": st.floats(min_value=1.0, max_value=2.0, allow_nan=False),
                "rsi": st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
                "volume_ratio_20": st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
                "price_change": st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
                "volatility": st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
            }
        )
    )
    def test_feature_dict_structure_matches_fixture(
        self, feature_dict: dict[str, float], sample_features: dict[str, float]
    ) -> None:
        """
        Verify fixture has same structure as generated feature dicts.

        Property: Fixture structure matches expected feature dictionary pattern.
        """
        # Fixture should have same keys as generated dict
        assert set(sample_features.keys()) == set(feature_dict.keys()), (
            f"Fixture keys {set(sample_features.keys())} don't match "
            f"expected {set(feature_dict.keys())}"
        )

        # All values should be numeric
        for key in sample_features:
            assert isinstance(sample_features[key], (float, int)), (
                f"Value for key '{key}' is not numeric: {type(sample_features[key])}"
            )

    @given(
        prediction_array=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=1, max_size=10
        ).map(lambda x: np.array(x, dtype=np.float32))
    )
    def test_prediction_array_properties(
        self, prediction_array: np.ndarray, sample_predictions: np.ndarray
    ) -> None:
        """
        Verify fixture has expected array properties.

        Property: Fixture is always a valid numpy array with float32 dtype.
        """
        # Fixture should be numpy array
        assert isinstance(sample_predictions, np.ndarray)

        # Fixture should have float32 dtype (like generated arrays)
        assert sample_predictions.dtype == np.float32

        # Fixture should be 1D
        assert len(sample_predictions.shape) == 1

        # All values should be finite
        assert np.all(np.isfinite(sample_predictions))


class TestFixtureDeterminism:
    """Property tests for fixture determinism."""

    @given(invocation=st.integers(min_value=1, max_value=5))
    def test_sample_features_deterministic(
        self, invocation: int, sample_features: dict[str, float]
    ) -> None:
        """
        Verify fixture returns consistent values across invocations within same test.

        Property: Fixture is deterministic (same values every time for same test).
        """
        # Expected values from fixture definition
        expected_values = {
            "price_sma_20": 1.0900,
            "rsi": 55.5,
            "volume_ratio_20": 1.2,
            "price_change": 0.002,
            "volatility": 0.015,
        }

        # Verify values match expected (regardless of invocation number)
        for key, expected in expected_values.items():
            assert abs(sample_features[key] - expected) < 1e-6, (
                f"Value for '{key}' changed: expected {expected}, "
                f"got {sample_features[key]} on invocation {invocation}"
            )

    @given(invocation=st.integers(min_value=1, max_value=5))
    def test_sample_predictions_deterministic(
        self, invocation: int, sample_predictions: np.ndarray
    ) -> None:
        """
        Verify fixture returns consistent arrays across invocations within same test.

        Property: Fixture is deterministic (same array every time for same test).
        """
        # Expected values from fixture definition
        expected = np.array([0.65, 0.3, 0.8], dtype=np.float32)

        # Verify values match expected (regardless of invocation number)
        assert np.allclose(sample_predictions, expected, rtol=1e-6), (
            f"Predictions changed: expected {expected}, "
            f"got {sample_predictions} on invocation {invocation}"
        )
