#!/usr/bin/env python3
"""
Unit tests for AdaptiveThresholdComponent.

This module tests the adaptive threshold component which manages threshold
adaptation based on volatility and market regime detection for MLSignalActor
decomposition.

Test Categories (26 tests total):
- Threshold Calculation: 10 tests
- Regime Detection: 8 tests
- Context Building: 5 tests
- Property Tests: 3 tests (Hypothesis-based)

This is TDD - tests define the contract for implementation.

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (zero allocations in warm path)
- Pattern 2: Protocol-First Interface Design (property accessors)

"""

from __future__ import annotations

import logging
import tracemalloc
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Mock logger for testing.
    """
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def default_component():
    """
    Provides a default AdaptiveThresholdComponent with standard parameters.
    """
    from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

    return AdaptiveThresholdComponent(
        base_threshold=0.7,
        volatility_factor=2.0,
        min_threshold=0.1,
        max_threshold=0.95,
        actor_id="test_actor",
        log=None,
    )


@pytest.fixture
def component_factory(mock_logger: logging.Logger):
    """
    Factory fixture for creating AdaptiveThresholdComponent instances.

    Returns a callable that creates components with specified parameters.

    """
    from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

    def _create_component(
        base_threshold: float = 0.7,
        volatility_factor: float = 2.0,
        min_threshold: float = 0.1,
        max_threshold: float = 0.95,
        actor_id: str | None = "test_actor",
        log: logging.Logger | None = None,
    ) -> AdaptiveThresholdComponent:
        """
        Create a component with specified parameters.
        """
        return AdaptiveThresholdComponent(
            base_threshold=base_threshold,
            volatility_factor=volatility_factor,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            actor_id=actor_id,
            log=log,
        )

    return _create_component


@pytest.fixture
def sample_volatility_window() -> npt.NDArray[np.float32]:
    """
    Provides a sample volatility window for regime detection tests.
    """
    return np.array([0.002, 0.003, 0.001, 0.004, 0.002], dtype=np.float32)


# =============================================================================
# Threshold Calculation Tests (10 tests)
# =============================================================================


class TestThresholdCalculation:
    """
    Tests for threshold calculation functionality.
    """

    def test_threshold_starts_at_base_value(self, component_factory) -> None:
        """
        Verify threshold initializes to base_threshold.

        Test ensures:
        - Initial threshold equals base_threshold
        - No modification before first update

        """
        # Arrange & Act
        component = component_factory(base_threshold=0.7)

        # Assert
        assert component.current_threshold == 0.7, "Initial threshold must equal base_threshold"

    def test_threshold_adjusts_for_volatility(self, component_factory) -> None:
        """
        Verify threshold adjusts based on volatility.

        Test ensures:
        - Threshold increases when volatility > 0
        - Formula: base + volatility * factor applied

        """
        # Arrange
        component = component_factory(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Act
        new_threshold = component.update_threshold(avg_volatility=0.01)

        # Assert
        assert new_threshold > 0.7, "Threshold should increase with positive volatility and factor"
        # Expected: 0.7 + (0.01 * 2.0) = 0.72
        assert new_threshold == pytest.approx(
            0.72, rel=1e-5
        ), "Threshold should be base + volatility * factor"

    def test_threshold_clamped_to_min(self, component_factory) -> None:
        """
        Verify threshold never below min_threshold.

        Test ensures:
        - Very low volatility with negative factor clamped
        - Threshold >= min_threshold always

        """
        # Arrange
        component = component_factory(
            base_threshold=0.3,
            volatility_factor=-10.0,  # Negative factor
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Act: High volatility with negative factor should push below min
        new_threshold = component.update_threshold(avg_volatility=0.05)

        # Assert
        # Expected: 0.3 + (0.05 * -10.0) = -0.2, but clamped to 0.1
        assert new_threshold >= 0.1, "Threshold must never be below min_threshold"
        assert (
            component.current_threshold >= component.min_threshold
        ), "Current threshold must respect min_threshold"

    def test_threshold_clamped_to_max(self, component_factory) -> None:
        """
        Verify threshold never above max_threshold.

        Test ensures:
        - Very high volatility clamped
        - Threshold <= max_threshold always

        """
        # Arrange
        component = component_factory(
            base_threshold=0.7,
            volatility_factor=100.0,  # Very high factor
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Act: High volatility should push above max
        new_threshold = component.update_threshold(avg_volatility=0.1)

        # Assert
        # Expected: 0.7 + (0.1 * 100.0) = 10.7, but clamped to 0.95
        assert new_threshold <= 0.95, "Threshold must never exceed max_threshold"
        assert (
            component.current_threshold <= component.max_threshold
        ), "Current threshold must respect max_threshold"

    def test_threshold_update_formula(self, component_factory) -> None:
        """
        Verify threshold update formula: base + volatility * factor.

        Test ensures:
        - Correct formula applied
        - Result clamped to [min, max]

        """
        # Arrange
        base = 0.5
        factor = 3.0
        volatility = 0.02
        min_t = 0.1
        max_t = 0.95
        component = component_factory(
            base_threshold=base,
            volatility_factor=factor,
            min_threshold=min_t,
            max_threshold=max_t,
        )

        # Act
        result = component.update_threshold(avg_volatility=volatility)

        # Assert
        # Expected: min(max(0.5 + 0.02 * 3.0, 0.1), 0.95) = 0.56
        expected = min(max(base + volatility * factor, min_t), max_t)
        assert result == pytest.approx(
            expected, rel=1e-5
        ), f"Threshold should match formula: {expected}, got {result}"

    def test_threshold_zero_volatility(self, component_factory) -> None:
        """
        Verify threshold with zero volatility stays at base.

        Test ensures:
        - Zero volatility produces no adjustment
        - Result equals base_threshold

        """
        # Arrange
        component = component_factory(
            base_threshold=0.6,
            volatility_factor=5.0,
        )

        # Act
        result = component.update_threshold(avg_volatility=0.0)

        # Assert
        assert result == 0.6, "Zero volatility should produce threshold equal to base_threshold"

    def test_threshold_negative_volatility_factor(self, component_factory) -> None:
        """
        Verify negative factor decreases threshold with volatility.

        Test ensures:
        - Negative factor causes threshold to decrease
        - Threshold < base_threshold when volatility > 0

        """
        # Arrange
        component = component_factory(
            base_threshold=0.7,
            volatility_factor=-1.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Act
        result = component.update_threshold(avg_volatility=0.1)

        # Assert
        # Expected: 0.7 + (0.1 * -1.0) = 0.6
        assert result < 0.7, "Negative volatility_factor should decrease threshold"
        assert result == pytest.approx(
            0.6, rel=1e-5
        ), "Threshold should be base - |volatility * factor|"

    def test_threshold_min_max_validation(self) -> None:
        """
        Verify min_threshold <= max_threshold enforced.

        Test ensures:
        - ValueError raised for invalid bounds
        - Clear error message

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Act & Assert
        with pytest.raises(ValueError, match="min_threshold"):
            AdaptiveThresholdComponent(
                base_threshold=0.5,
                volatility_factor=2.0,
                min_threshold=0.9,  # Greater than max
                max_threshold=0.3,
            )

    def test_threshold_base_within_bounds(self) -> None:
        """
        Verify base_threshold within [min, max].

        Test ensures:
        - ValueError raised for base outside bounds
        - Clear error message

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Act & Assert: base below min
        with pytest.raises(ValueError, match="base_threshold"):
            AdaptiveThresholdComponent(
                base_threshold=0.05,  # Below min
                volatility_factor=2.0,
                min_threshold=0.1,
                max_threshold=0.95,
            )

        # Act & Assert: base above max
        with pytest.raises(ValueError, match="base_threshold"):
            AdaptiveThresholdComponent(
                base_threshold=0.99,  # Above max
                volatility_factor=2.0,
                min_threshold=0.1,
                max_threshold=0.95,
            )

    def test_threshold_update_idempotent_with_same_volatility(
        self,
        component_factory,
    ) -> None:
        """
        Verify updating with same volatility doesn't change threshold.

        Test ensures:
        - Idempotent behavior
        - Threshold unchanged for repeated same input

        """
        # Arrange
        component = component_factory(base_threshold=0.7)
        volatility = 0.02

        # Act
        threshold1 = component.update_threshold(avg_volatility=volatility)
        threshold2 = component.update_threshold(avg_volatility=volatility)

        # Assert
        assert (
            threshold1 == threshold2
        ), "Updating with same volatility should produce same threshold"


# =============================================================================
# Regime Detection Tests (8 tests)
# =============================================================================


class TestRegimeDetection:
    """
    Tests for market regime detection functionality.
    """

    def test_regime_detection_low_volatility(self, component_factory) -> None:
        """
        Verify "low_volatility" regime for avg_vol < 0.001.

        Test ensures:
        - Low volatility detected correctly
        - Regime label is "low_volatility"

        """
        # Arrange
        component = component_factory()
        window = np.array([0.0005, 0.0004, 0.0006, 0.0005, 0.0005], dtype=np.float32)
        count = 5

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        # Average = 0.0005 < 0.001
        assert (
            regime == "low_volatility"
        ), f"Expected 'low_volatility' regime for avg_vol < 0.001, got '{regime}'"

    def test_regime_detection_normal_volatility(self, component_factory) -> None:
        """
        Verify "normal" regime for 0.001 <= avg_vol < 0.005.

        Test ensures:
        - Normal volatility detected correctly
        - Regime label is "normal"

        """
        # Arrange
        component = component_factory()
        window = np.array([0.003, 0.003, 0.003, 0.003, 0.003], dtype=np.float32)
        count = 5

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        # Average = 0.003, which is in [0.001, 0.005)
        assert (
            regime == "normal"
        ), f"Expected 'normal' regime for 0.001 <= avg_vol < 0.005, got '{regime}'"

    def test_regime_detection_high_volatility(self, component_factory) -> None:
        """
        Verify "high_volatility" regime for avg_vol >= 0.005.

        Test ensures:
        - High volatility detected correctly
        - Regime label is "high_volatility"

        """
        # Arrange
        component = component_factory()
        window = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        count = 5

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        # Average = 0.01 >= 0.005
        assert (
            regime == "high_volatility"
        ), f"Expected 'high_volatility' regime for avg_vol >= 0.005, got '{regime}'"

    def test_regime_detection_unknown_insufficient_data(
        self,
        component_factory,
    ) -> None:
        """
        Verify "unknown" regime when count < min_count.

        Test ensures:
        - Insufficient data produces "unknown" regime
        - MIN_REGIME_COUNT (3) enforced

        """
        # Arrange
        component = component_factory()
        window = np.array([0.003, 0.003, 0.0, 0.0, 0.0], dtype=np.float32)
        count = 2  # Less than MIN_REGIME_COUNT (3)

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        assert (
            regime == "unknown"
        ), f"Expected 'unknown' regime for count < min_count, got '{regime}'"

    def test_regime_detection_uses_count_not_capacity(
        self,
        component_factory,
    ) -> None:
        """
        Verify regime uses count (not capacity) for average.

        Test ensures:
        - Average calculated from window[:count]
        - Padding zeros ignored

        """
        # Arrange
        component = component_factory()
        # Window with capacity 100, but only 10 valid values
        window = np.zeros(100, dtype=np.float32)
        window[:10] = 0.003  # Only first 10 values are valid
        count = 10

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert: Average over 10 values = 0.003 -> "normal"
        # If it used all 100 values, avg would be 0.0003 -> "low_volatility"
        assert regime == "normal", "Regime should use count-based average, not full array"

    def test_regime_detection_boundary_0_001(self, component_factory) -> None:
        """
        Verify boundary case: avg_vol exactly 0.001.

        Test ensures:
        - Boundary is inclusive for "normal"
        - avg_vol == 0.001 -> "normal"

        """
        # Arrange
        component = component_factory()
        window = np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float32)
        count = 5

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        # avg_vol = 0.001, boundary is inclusive for "normal"
        assert regime == "normal", f"Expected 'normal' regime at boundary 0.001, got '{regime}'"

    def test_regime_detection_boundary_0_005(self, component_factory) -> None:
        """
        Verify boundary case: avg_vol at/above 0.005.

        Test ensures:
        - Values >= 0.005 produce "high_volatility"
        - Boundary check uses strict less-than (<) for normal

        Note: float32 precision means 0.005 stored as ~0.00499999989...
        which is < 0.005 (float64). To get true "high_volatility", use
        a value slightly above the boundary.

        """
        # Arrange
        component = component_factory()
        # Use value slightly above boundary to ensure "high_volatility"
        # (float32 precision makes exact 0.005 problematic)
        window = np.array([0.006, 0.006, 0.006, 0.006, 0.006], dtype=np.float32)
        count = 5

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert
        # avg_vol = 0.006 > 0.005 -> "high_volatility"
        assert (
            regime == "high_volatility"
        ), f"Expected 'high_volatility' regime for avg_vol > 0.005, got '{regime}'"

    def test_regime_detection_with_zero_padding(self, component_factory) -> None:
        """
        Verify regime ignores zero-padding in ring buffer.

        Test ensures:
        - Only valid values (window[:count]) used
        - Zero padding doesn't affect average

        """
        # Arrange
        component = component_factory()
        # Simulate ring buffer: 50 real values, rest zeros
        window = np.zeros(100, dtype=np.float32)
        window[:50] = 0.006  # High volatility values
        count = 50

        # Act
        regime = component.detect_regime(volatility_window=window, count=count)

        # Assert: Average over 50 values = 0.006 -> "high_volatility"
        assert regime == "high_volatility", "Regime calculation should ignore zero-padded portion"
        # Verify state updated
        assert (
            component.current_regime == "high_volatility"
        ), "current_regime property should reflect detected regime"


# =============================================================================
# Context Building Tests (5 tests)
# =============================================================================


class TestContextBuilding:
    """
    Tests for context building functionality.
    """

    def test_context_includes_adaptive_threshold(self, component_factory) -> None:
        """
        Verify build_context() includes adaptive_threshold.

        Test ensures:
        - Context has adaptive_threshold key
        - Value matches current threshold

        """
        # Arrange
        component = component_factory(base_threshold=0.75)
        component.update_threshold(avg_volatility=0.0)  # Keep at base

        # Act
        context = component.build_context()

        # Assert
        assert "adaptive_threshold" in context, "Context must include 'adaptive_threshold' key"
        assert (
            context["adaptive_threshold"] == 0.75
        ), "adaptive_threshold should match current threshold"

    def test_context_includes_market_regime(self, component_factory) -> None:
        """
        Verify build_context() includes market_regime.

        Test ensures:
        - Context has market_regime key
        - Value matches current regime

        """
        # Arrange
        component = component_factory()
        window = np.array([0.003, 0.003, 0.003, 0.003, 0.003], dtype=np.float32)
        component.detect_regime(volatility_window=window, count=5)

        # Act
        context = component.build_context()

        # Assert
        assert "market_regime" in context, "Context must include 'market_regime' key"
        assert context["market_regime"] == "normal", "market_regime should match detected regime"

    def test_context_includes_prediction_history(self, component_factory) -> None:
        """
        Verify context includes prediction_history from buffer.

        Test ensures:
        - Prediction history passed through
        - Value matches input

        """
        # Arrange
        component = component_factory()
        prediction_history = [0.5, 0.6, 0.7, 0.8]

        # Act
        context = component.build_context(prediction_history=prediction_history)

        # Assert
        assert (
            "prediction_history" in context
        ), "Context must include 'prediction_history' when provided"
        assert (
            context["prediction_history"] is prediction_history
        ), "prediction_history should be reference to input (not copy)"

    def test_context_includes_confidence_history(self, component_factory) -> None:
        """
        Verify context includes confidence_history.

        Test ensures:
        - Confidence history passed through
        - Value matches input

        """
        # Arrange
        component = component_factory()
        confidence_history = [0.9, 0.85, 0.92, 0.88]

        # Act
        context = component.build_context(confidence_history=confidence_history)

        # Assert
        assert (
            "confidence_history" in context
        ), "Context must include 'confidence_history' when provided"
        assert (
            context["confidence_history"] is confidence_history
        ), "confidence_history should be reference to input (not copy)"

    def test_context_building_no_allocations(self, component_factory) -> None:
        """
        Verify build_context() doesn't allocate excessively (hot path).

        Test ensures:
        - Minimal allocations (only dict creation)
        - Values are references, not copies

        Note: Dictionary creation is unavoidable, but values should be references.

        """
        # Arrange
        component = component_factory(base_threshold=0.7)
        component.update_threshold(avg_volatility=0.01)
        window = np.array([0.003, 0.003, 0.003], dtype=np.float32)
        component.detect_regime(volatility_window=window, count=3)

        # Warm up
        _ = component.build_context()

        # Act: Measure allocations during context building
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            _ = component.build_context()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Dict creation has some overhead, allow reasonable amount
        # 100 iterations * ~100-200 bytes per dict = ~10000-20000 bytes
        assert total_new_bytes < 30000, (
            f"Context building should have minimal allocations, "
            f"but allocated {total_new_bytes} bytes"
        )


# =============================================================================
# Property Tests (3 tests)
# =============================================================================


class TestPropertyBased:
    """
    Hypothesis property-based tests for adaptive threshold invariants.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        volatility=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_threshold_always_within_bounds_property(
        self,
        volatility: float,
    ) -> None:
        """
        Verify threshold in [min, max] for any volatility.

        Property: min_threshold <= threshold <= max_threshold for all states.

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Arrange
        min_threshold = 0.1
        max_threshold = 0.95
        component = AdaptiveThresholdComponent(
            base_threshold=0.5,
            volatility_factor=5.0,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )

        # Act
        threshold = component.update_threshold(avg_volatility=volatility)

        # Assert: Invariant must hold
        assert min_threshold <= threshold <= max_threshold, (
            f"Threshold ({threshold}) must be in [{min_threshold}, {max_threshold}] "
            f"for volatility={volatility}"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        volatilities=st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
            min_size=2,
            max_size=20,
        ),
    )
    def test_threshold_monotonic_with_volatility_property(
        self,
        volatilities: list[float],
    ) -> None:
        """
        Verify threshold increases monotonically with volatility (positive factor).

        Property: For sorted volatilities, thresholds are non-decreasing.

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Arrange
        component = AdaptiveThresholdComponent(
            base_threshold=0.5,
            volatility_factor=2.0,  # Positive factor
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Sort volatilities for monotonicity test
        sorted_volatilities = sorted(volatilities)

        # Act
        thresholds = [component.update_threshold(avg_volatility=vol) for vol in sorted_volatilities]

        # Assert: Thresholds should be non-decreasing (monotonic)
        assert thresholds == sorted(thresholds), (
            "Thresholds must be non-decreasing for sorted volatilities "
            "with positive volatility_factor"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        base_volatility=st.floats(min_value=0.001, max_value=0.004, allow_nan=False),
        delta=st.floats(min_value=-0.0001, max_value=0.0001, allow_nan=False),
    )
    def test_regime_stability_property(
        self,
        base_volatility: float,
        delta: float,
    ) -> None:
        """
        Verify regime doesn't change for small volatility changes.

        Property: Same regime for volatility +/- small delta.

        Note: This tests stability within the middle of regime boundaries.
        Very small deltas near boundaries may cause regime changes.

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Arrange
        component = AdaptiveThresholdComponent(
            base_threshold=0.5,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Use base_volatility in middle of "normal" regime (0.001 to 0.005)
        assume(0.0015 <= base_volatility <= 0.0045)
        assume(0.0015 <= base_volatility + delta <= 0.0045)

        # Create volatility windows
        window1 = np.full(5, base_volatility, dtype=np.float32)
        window2 = np.full(5, base_volatility + delta, dtype=np.float32)

        # Act
        regime1 = component.detect_regime(volatility_window=window1, count=5)
        regime2 = component.detect_regime(volatility_window=window2, count=5)

        # Assert: Same regime for small delta within regime boundaries
        assert regime1 == regime2, (
            f"Regime should be stable for small delta: "
            f"base={base_volatility}, delta={delta}, "
            f"regime1={regime1}, regime2={regime2}"
        )


# =============================================================================
# Edge Case and Error Condition Tests
# =============================================================================


class TestEdgeCasesAndErrors:
    """
    Tests for edge cases and error conditions.
    """

    def test_property_accessors_return_correct_values(
        self,
        component_factory,
    ) -> None:
        """
        Verify property accessors return correct values.

        Test all property accessors for consistency.

        """
        # Arrange
        component = component_factory(
            base_threshold=0.65,
            volatility_factor=3.0,
            min_threshold=0.2,
            max_threshold=0.9,
        )

        # Assert
        assert component.base_threshold == 0.65, "base_threshold property incorrect"
        assert component.volatility_factor == 3.0, "volatility_factor property incorrect"
        assert component.min_threshold == 0.2, "min_threshold property incorrect"
        assert component.max_threshold == 0.9, "max_threshold property incorrect"
        assert component.current_threshold == 0.65, "current_threshold should start at base"
        assert component.current_regime == "unknown", "current_regime should start unknown"

    def test_update_threshold_updates_internal_state(
        self,
        component_factory,
    ) -> None:
        """
        Verify update_threshold() updates internal state.

        Test ensures:
        - _threshold internal state updated
        - current_threshold property reflects change

        """
        # Arrange
        component = component_factory(base_threshold=0.5)

        # Act
        result = component.update_threshold(avg_volatility=0.05)

        # Assert
        assert (
            component.current_threshold == result
        ), "current_threshold property should match update result"

    def test_detect_regime_updates_internal_state(self, component_factory) -> None:
        """
        Verify detect_regime() updates internal state.

        Test ensures:
        - _market_regime internal state updated
        - current_regime property reflects change

        """
        # Arrange
        component = component_factory()
        window = np.array([0.01, 0.01, 0.01], dtype=np.float32)

        # Act
        result = component.detect_regime(volatility_window=window, count=3)

        # Assert
        assert (
            component.current_regime == result
        ), "current_regime property should match detect result"

    def test_context_without_optional_histories(self, component_factory) -> None:
        """
        Verify build_context() works without optional histories.

        Test ensures:
        - Context built successfully without histories
        - Only adaptive_threshold and market_regime present

        """
        # Arrange
        component = component_factory(base_threshold=0.7)

        # Act
        context = component.build_context()

        # Assert
        assert "adaptive_threshold" in context, "adaptive_threshold must be present"
        assert "market_regime" in context, "market_regime must be present"
        assert (
            "prediction_history" not in context
        ), "prediction_history should not be present when not provided"
        assert (
            "confidence_history" not in context
        ), "confidence_history should not be present when not provided"

    def test_component_with_logger(self, mock_logger) -> None:
        """
        Verify component works with logger.

        Test ensures:
        - Logger accepted and used
        - Initialization logged

        """
        from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent

        # Arrange & Act
        component = AdaptiveThresholdComponent(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
            actor_id="test_actor",
            log=mock_logger,
        )

        # Assert
        mock_logger.info.assert_called_once()
        assert "AdaptiveThresholdComponent initialized" in str(
            mock_logger.info.call_args,
        ), "Initialization should be logged"

    def test_regime_with_empty_valid_window(self, component_factory) -> None:
        """
        Verify regime detection with count=0.

        Test ensures:
        - Zero count produces "unknown" regime
        - No division by zero

        """
        # Arrange
        component = component_factory()
        window = np.zeros(10, dtype=np.float32)

        # Act
        regime = component.detect_regime(volatility_window=window, count=0)

        # Assert
        assert regime == "unknown", "Zero count should produce 'unknown' regime"
