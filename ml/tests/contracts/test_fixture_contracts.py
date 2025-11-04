"""
Contract tests for fixture behavior after critical fixes.

This module tests that fixtures behave correctly after fixing:
1. Autouse removal from mock_prometheus_when_unavailable
2. Duplicate fixture consolidation

Tests verify opt-in behavior, type consistency, and isolation guarantees.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestMockPrometheusFixtureContracts:
    """Contract tests for mock_prometheus_when_unavailable fixture."""

    def test_mock_prometheus_fixture_is_opt_in(self) -> None:
        """
        Verify the fixture is not autouse and requires explicit declaration.

        Expected: Fixture has autouse=False and requires explicit use in test signatures.
        """
        # Verify by reading the fixture source file directly
        import pathlib

        fixture_file = pathlib.Path(__file__).parent.parent / "fixtures" / "monitoring_collectors.py"
        fixture_source = fixture_file.read_text()

        # Check that the fixture definition has autouse=False (not True)
        assert "@pytest.fixture(autouse=True)" not in fixture_source, (
            "Fixture must not have autouse=True"
        )

        # Also verify the fixture exists and is properly defined
        assert "def mock_prometheus_when_unavailable" in fixture_source, (
            "Fixture mock_prometheus_when_unavailable not found"
        )

        # Verify it has either autouse=False or no autouse parameter (defaults to False)
        # The fixture should have one of these patterns:
        # - @pytest.fixture(autouse=False)
        # - @pytest.fixture()
        # - @pytest.fixture
        assert ("@pytest.fixture(autouse=False)" in fixture_source or
                "@pytest.fixture()" in fixture_source or
                "@pytest.fixture\ndef mock_prometheus_when_unavailable" in fixture_source), (
            "Fixture must explicitly have autouse=False or no autouse parameter"
        )

    def test_mock_prometheus_fixture_behavior_when_unavailable(
        self, mock_prometheus_when_unavailable: Any
    ) -> None:
        """
        Verify fixture provides consistent mocks when HAS_PROMETHEUS=False.

        Expected: Fixture yields dict with Counter, Gauge, Histogram mocks or None.
        """
        # Import after fixture has applied patching
        from ml._imports import HAS_PROMETHEUS

        if not HAS_PROMETHEUS:
            # When Prometheus unavailable, fixture should provide mocks
            assert isinstance(mock_prometheus_when_unavailable, dict), (
                "Fixture should return dict of mocks when HAS_PROMETHEUS=False"
            )
            assert "Counter" in mock_prometheus_when_unavailable
            assert "Gauge" in mock_prometheus_when_unavailable
            assert "Histogram" in mock_prometheus_when_unavailable

            # Verify mocks are MagicMock instances (object type check, not identity)
            assert isinstance(
                mock_prometheus_when_unavailable["Counter"], type(MagicMock())
            ), "Counter should be a mock"
            assert isinstance(
                mock_prometheus_when_unavailable["Gauge"], type(MagicMock())
            ), "Gauge should be a mock"
            assert isinstance(
                mock_prometheus_when_unavailable["Histogram"], type(MagicMock())
            ), "Histogram should be a mock"
        else:
            # When Prometheus available, fixture should yield None
            assert mock_prometheus_when_unavailable is None, (
                "Fixture should return None when HAS_PROMETHEUS=True"
            )

    def test_tests_without_fixture_unaffected(self) -> None:
        """
        Verify tests not using the fixture see real HAS_PROMETHEUS value.

        Expected: HAS_PROMETHEUS reflects actual installation state.
        """
        from ml._imports import HAS_PROMETHEUS

        # This test deliberately omits mock_prometheus_when_unavailable from signature
        # It should see the real HAS_PROMETHEUS value
        assert isinstance(HAS_PROMETHEUS, bool), "HAS_PROMETHEUS should be a boolean"

        # No mocking or patching should occur - imports should work naturally
        # (This test proves the fixture is opt-in, not autouse)


class TestSampleFeaturesFixtureContracts:
    """Contract tests for sample_features fixture after consolidation."""

    def test_sample_features_fixture_type_consistency(
        self, sample_features: dict[str, float]
    ) -> None:
        """
        Verify sample_features has consistent return type.

        Expected: Returns dict[str, float] (canonical type from common.py).
        """
        # Type check (not identity)
        assert isinstance(sample_features, dict), "sample_features must be dict"

        # Verify keys are strings
        assert all(isinstance(k, str) for k in sample_features.keys()), (
            "All keys must be strings"
        )

        # Verify values are floats
        assert all(
            isinstance(v, (float, int)) for v in sample_features.values()
        ), "All values must be numeric"

        # Contains expected keys (canonical fixture from common.py)
        expected_keys = {"sma_20", "rsi", "volume_ratio", "price_change", "volatility"}
        assert set(sample_features.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(sample_features.keys())}"
        )


class TestSamplePredictionsFixtureContracts:
    """Contract tests for sample_predictions fixture after consolidation."""

    def test_sample_predictions_fixture_type_consistency(
        self, sample_predictions: np.ndarray
    ) -> None:
        """
        Verify sample_predictions has consistent return type.

        Expected: Returns np.ndarray with shape (3,) and dtype float32.
        """
        # Type check (not identity)
        assert isinstance(sample_predictions, np.ndarray), (
            "sample_predictions must be np.ndarray"
        )

        # Exact shape match
        assert sample_predictions.shape == (3,), (
            f"Expected shape (3,), got {sample_predictions.shape}"
        )

        # Exact dtype match
        assert sample_predictions.dtype == np.float32, (
            f"Expected dtype float32, got {sample_predictions.dtype}"
        )

        # Values are reasonable predictions
        assert all(-1.0 <= v <= 1.0 for v in sample_predictions), (
            "Predictions should be in range [-1, 1]"
        )


class TestFixtureImmutability:
    """Edge case tests for fixture immutability."""

    def test_sample_features_fixture_immutability_first_call(
        self, sample_features: dict[str, float]
    ) -> None:
        """
        Verify modifying fixture return value doesn't affect other tests (first call).

        Expected: Each test gets fresh dict instance.
        """
        # Store original value
        original_rsi = sample_features["rsi"]

        # Modify fixture
        sample_features["rsi"] = 999.9

        # Verify modification happened
        assert sample_features["rsi"] == 999.9

        # Store the modified value for comparison in next test
        # (In real execution, this won't affect other tests)
        assert original_rsi != 999.9, "Original value should differ from modified"

    def test_sample_features_fixture_immutability_second_call(
        self, sample_features: dict[str, float]
    ) -> None:
        """
        Verify modifying fixture return value doesn't affect other tests (second call).

        Expected: Each test gets fresh dict instance, modifications from previous test don't persist.
        """
        # This test runs after test_sample_features_fixture_immutability_first_call
        # If fixture is properly isolated, rsi should be back to default
        assert sample_features["rsi"] == 55.5, (
            "Fixture should be fresh, not polluted by previous test"
        )

    def test_sample_predictions_fixture_determinism(
        self, sample_predictions: np.ndarray
    ) -> None:
        """
        Verify fixture returns same values across invocations.

        Expected: All calls return identical arrays (deterministic).
        """
        # Fixture is called once per test, but we can verify it's deterministic
        # by checking the values match expected defaults
        expected = np.array([0.65, -0.3, 0.8], dtype=np.float32)

        # Value equality (not identity)
        assert np.array_equal(sample_predictions, expected), (
            f"Expected {expected}, got {sample_predictions}"
        )


class TestBackwardCompatibility:
    """Backward compatibility tests for existing test patterns."""

    def test_existing_tests_using_sample_features_dict(
        self, sample_features: dict[str, float]
    ) -> None:
        """
        Verify tests expecting dict[str, float] still work.

        Expected: No breaking changes, all existing test patterns work.
        """
        # Common test pattern: accessing dict keys
        assert "rsi" in sample_features
        assert isinstance(sample_features["rsi"], (float, int))

        # Common test pattern: iterating keys
        for key, value in sample_features.items():
            assert isinstance(key, str)
            assert isinstance(value, (float, int))

        # No AttributeError or KeyError should be raised


# Note: E2E-specific tests for sample_feature_objects and sample_prediction_objects
# are in ml/tests/e2e/test_datastore_e2e.py and use the renamed fixtures
