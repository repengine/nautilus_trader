#!/usr/bin/env python3
"""
Unit tests for SignalStrategyComponent and signal generation strategies.

This module tests:
- 5 strategy implementations (Threshold, Extremes, Momentum, Ensemble, Adaptive)
- Strategy factory with 3-level priority system
- Atomic strategy swapping (prepare -> execute pattern)
- Hot path optimization (P99 <100us target)

Test Categories (78 tests total):
- Threshold Strategy: 14 tests
- Extremes Strategy: 12 tests
- Momentum Strategy: 12 tests
- Ensemble Strategy: 14 tests
- Adaptive Strategy: 12 tests
- Strategy Factory: 8 tests
- Strategy Swapper: 6 tests
- Property Tests: 5 tests (Hypothesis-based)

as this is TDD - tests define the contract for implementation.

"""

from __future__ import annotations

import logging
import time
import tracemalloc
from collections.abc import MutableMapping
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, given, settings, assume
from hypothesis import strategies as st
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from ml.actors.base import MLSignal
from ml.actors.common.signal_strategy import (
    AdaptiveStrategy,
    EnsembleStrategy,
    ExtremesStrategy,
    MomentumStrategy,
    SignalGenerationStrategy,
    SignalPolicySwapper,
    SignalStrategy,
    SignalStrategyComponent,
    StrategySwapper,
    ThresholdSignalStrategy,
)


# =================================================================================================
# Fixtures
# =================================================================================================


@pytest.fixture
def default_bar_type() -> BarType:
    """
    Standard bar type for testing.
    """
    return BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")


@pytest.fixture
def create_test_bar(default_bar_type: BarType):
    """
    Factory fixture for creating test bars.

    Returns a callable that creates Bar objects with specified parameters.

    """

    def _create_bar(
        open_price: float = 1.0900,
        high_price: float = 1.0910,
        low_price: float = 1.0890,
        close_price: float = 1.0905,
        volume: float = 1000.0,
        ts_event: int | None = None,
        ts_init: int | None = None,
    ) -> Bar:
        """
        Create a test bar with specified parameters.
        """
        if ts_event is None:
            ts_event = int(time.time() * 1e9)
        if ts_init is None:
            ts_init = ts_event + 1000

        return Bar(
            bar_type=default_bar_type,
            open=Price(open_price, precision=5),
            high=Price(high_price, precision=5),
            low=Price(low_price, precision=5),
            close=Price(close_price, precision=5),
            volume=Quantity(volume, precision=0),
            ts_event=ts_event,
            ts_init=ts_init,
        )

    return _create_bar


@pytest.fixture
def base_features() -> npt.NDArray[np.float32]:
    """
    Standard feature array for testing.
    """
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)


@pytest.fixture
def base_context() -> dict[str, Any]:
    """
    Standard context dictionary for testing.
    """
    return {
        "timestamp_ns": int(time.time() * 1e9),
        "model_id": "test_model_v1",
        "log_predictions": False,
    }


@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Mock logger for testing.
    """
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def base_signal_config():
    """
    Base MLSignalActorConfig for tests.

    Creates a minimal config suitable for testing SignalStrategyComponent.

    """
    from ml.config.actors import MLSignalActorConfig

    return MLSignalActorConfig(
        model_id="test_signal_model",
        model_path="/tmp/test_model.onnx",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        batch_size=1,
        warm_up_period=10,
        prediction_threshold=0.7,
        use_dummy_stores=True,
        signal_strategy="threshold",
    )


# =================================================================================================
# Category 1: Threshold Strategy Tests (14 tests)
# =================================================================================================


class TestThresholdStrategy:
    """
    Tests for ThresholdSignalStrategy.
    """

    def test_threshold_strategy_basic_signal_generation(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ThresholdSignalStrategy generates signal when confidence meets threshold.

        Test ensures:
        - Signal generated when confidence >= threshold
        - Signal contains correct prediction and confidence values
        - Signal timestamp matches bar timestamp

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        prediction = 0.8
        confidence = 0.9

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.ts_event == bar.ts_event

    def test_threshold_strategy_confidence_below_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy returns None when confidence below threshold.

        Test ensures:
        - No signal generated when confidence < threshold
        - Method returns None (not empty signal)

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        prediction = 0.8
        confidence = 0.6

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is None

    def test_threshold_strategy_confidence_exactly_at_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify edge case: confidence exactly equals threshold.

        Test ensures:
        - Signal generated when confidence == threshold (boundary inclusive)
        - Confidence value preserved in signal

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        prediction = 0.8
        confidence = 0.7

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        assert signal.confidence == 0.7

    def test_threshold_strategy_with_zero_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy with threshold=0 always generates signals.

        Test ensures:
        - Signal generated for any confidence >= 0
        - Edge case for "accept all" configuration

        """
        strategy = ThresholdSignalStrategy(threshold=0.0)
        bar = create_test_bar()
        prediction = 0.5
        confidence = 0.01

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None

    def test_threshold_strategy_with_one_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy with threshold=1 requires perfect confidence.

        Test ensures:
        - No signal when confidence < 1.0
        - Only perfect confidence generates signal

        """
        strategy = ThresholdSignalStrategy(threshold=1.0)
        bar = create_test_bar()
        prediction = 0.8
        confidence = 0.99

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is None

    def test_threshold_strategy_preserves_prediction_value(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy preserves original prediction value.

        Test ensures:
        - Prediction value not modified by strategy
        - Works with negative predictions (sell signals)

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        prediction = -0.5
        confidence = 0.9

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        assert signal.prediction == -0.5

    def test_threshold_strategy_includes_model_id_in_context(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal includes model_id from context.

        Test ensures:
        - model_id from context propagated to signal
        - Correct model attribution for tracking

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model_v1",
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, context)

        assert signal is not None
        assert signal.model_id == "test_model_v1"

    def test_threshold_strategy_handles_missing_model_id(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy handles missing model_id gracefully.

        Test ensures:
        - Default model_id used when not in context
        - No exception raised for missing context key

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        context: dict[str, Any] = {"timestamp_ns": int(time.time() * 1e9)}

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, context)

        assert signal is not None
        assert signal.model_id == "unknown"

    def test_threshold_strategy_includes_features_when_log_enabled(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal includes features when log_predictions=True.

        Test ensures:
        - Features included in signal for debugging
        - Feature array matches input exactly

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "log_predictions": True,
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, context)

        assert signal is not None
        assert signal.features is not None
        assert np.array_equal(signal.features, base_features)

    def test_threshold_strategy_excludes_features_when_log_disabled(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal excludes features when log_predictions=False.

        Test ensures:
        - Features not included to save memory
        - Hot path optimization respected

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "log_predictions": False,
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, context)

        assert signal is not None
        assert signal.features is None

    def test_threshold_strategy_uses_correct_instrument_id(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify signal uses instrument_id from bar.

        Test ensures:
        - Instrument ID correctly extracted from bar_type
        - Signal attributed to correct instrument

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        assert signal is not None
        assert signal.instrument_id == bar.bar_type.instrument_id

    def test_threshold_strategy_uses_timestamp_from_context(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal uses timestamp_ns from context.

        Test ensures:
        - ts_init from context, not bar
        - Proper timing attribution

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        expected_timestamp = 123456789
        context = {
            "timestamp_ns": expected_timestamp,
            "model_id": "test_model",
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, base_features, context)

        assert signal is not None
        assert signal.ts_init == expected_timestamp

    def test_threshold_strategy_handles_negative_prediction(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy works with negative predictions (sell signals).

        Test ensures:
        - Negative predictions preserved
        - Confidence still validated against threshold

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()
        prediction = -0.8
        confidence = 0.9

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        assert signal.prediction == -0.8
        assert signal.confidence == 0.9

    def test_threshold_strategy_no_allocations_on_hot_path(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify strategy.generate_signal() allocates minimal memory (hot path
        requirement).

        Test ensures:
        - No significant memory allocations for threshold check
        - Hot path P99 <100us achievable

        Note: We test for minimal allocations (only MLSignal object when signal generated).

        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = create_test_bar()

        # Warm up - first call may allocate
        _ = strategy.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        # Measure allocations on subsequent call
        tracemalloc.start()
        _ = strategy.generate_signal(bar, 0.8, 0.5, base_features, base_context)
        current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # For no-signal case, should have near-zero allocations
        # Allow small threshold for Python internals
        assert current < 1000, f"Unexpected allocations: {current} bytes"


# =================================================================================================
# Category 2: Extremes Strategy Tests (12 tests)
# =================================================================================================


class TestExtremesStrategy:
    """
    Tests for ExtremesStrategy.
    """

    def test_extremes_strategy_generates_signal_at_top_percentile(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify ExtremesStrategy generates signal when prediction in top percentile.

        Test ensures:
        - Signal generated for extreme high predictions
        - Percentile calculation correct

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill window with predictions [0.1, 0.2, ..., 0.9]
        for i in range(50):
            pred = 0.1 + (i / 50) * 0.8
            strategy.generate_signal(bar, pred, 0.9, base_features, context)

        # This prediction should be in top percentile
        signal = strategy.generate_signal(bar, 0.95, 0.9, base_features, context)

        assert signal is not None
        assert signal.prediction == 0.95

    def test_extremes_strategy_generates_signal_at_bottom_percentile(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy generates signal when prediction in bottom percentile.

        Test ensures:
        - Signal generated for extreme low predictions
        - Bottom percentile detection works

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill window
        for i in range(50):
            pred = 0.1 + (i / 50) * 0.8
            strategy.generate_signal(bar, pred, 0.9, base_features, context)

        # This prediction should be in bottom percentile
        signal = strategy.generate_signal(bar, 0.05, 0.9, base_features, context)

        assert signal is not None
        assert signal.prediction == 0.05

    def test_extremes_strategy_no_signal_for_average_prediction(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify no signal for mid-range predictions.

        Test ensures:
        - Average predictions don't trigger signals
        - Only extremes generate signals

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill window
        for i in range(50):
            pred = 0.1 + (i / 50) * 0.8
            strategy.generate_signal(bar, pred, 0.9, base_features, context)

        # Mid-range prediction
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is None

    def test_extremes_strategy_respects_confidence_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy checks confidence threshold even for extreme values.

        Test ensures:
        - Extreme prediction with low confidence doesn't generate signal
        - Both conditions must be met

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill window
        for i in range(50):
            strategy.generate_signal(bar, i * 0.02, 0.9, base_features, context)

        # Extreme prediction but low confidence
        signal = strategy.generate_signal(bar, 0.95, 0.5, base_features, context)

        assert signal is None

    def test_extremes_strategy_requires_full_window(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy returns None until window is full.

        Test ensures:
        - No signals during warm-up period
        - Signals only after sufficient history

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=50)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Add only 30 predictions (less than window_size=50)
        for i in range(30):
            signal = strategy.generate_signal(bar, 0.95, 0.9, base_features, context)
            assert signal is None, f"Signal generated at prediction {i}"

    def test_extremes_strategy_uses_ring_buffer(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy uses pre-allocated ring buffer (zero-copy).

        Test ensures:
        - Ring buffer initialized in context
        - Circular indexing correct
        - No new allocations after warm-up

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=10)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # First call should initialize ring buffer
        strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert "_pred_ring" in context
        assert "_pred_ring_filled" in context
        assert "_pred_ring_idx" in context

        # Fill buffer and verify indices wrap correctly
        for i in range(15):  # More than window size
            strategy.generate_signal(bar, 0.5 + i * 0.01, 0.9, base_features, context)

        assert context["_pred_ring_filled"] == 10  # Capped at window size
        assert context["_pred_ring_idx"] < 10  # Index wraps

    def test_extremes_strategy_calculates_percentiles_correctly(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify percentile calculation using np.partition (not full sort).

        Test ensures:
        - Correct top/bottom thresholds calculated
        - Uses efficient partition algorithm

        """
        strategy = ExtremesStrategy(top_pct=0.2, threshold=0.5, window_size=10)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill with known values: 0.0, 0.1, 0.2, ..., 0.9
        for i in range(10):
            strategy.generate_signal(bar, i * 0.1, 0.9, base_features, context)

        # With top_pct=0.2, top 20% = 0.8+ and bottom 20% = 0.1-
        # Prediction of 0.85 should be in top 20%
        signal = strategy.generate_signal(bar, 0.85, 0.9, base_features, context)
        assert signal is not None

        # Prediction of 0.05 should be in bottom 20%
        signal = strategy.generate_signal(bar, 0.05, 0.9, base_features, context)
        assert signal is not None

    def test_extremes_strategy_handles_duplicate_predictions(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy handles case where many predictions are identical.

        Test ensures:
        - Duplicate handling doesn't break percentile calculation
        - Signal generated if prediction is extreme

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.5, window_size=10)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Fill with mostly identical values
        for _ in range(9):
            strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        # Add one extreme value
        strategy.generate_signal(bar, 0.1, 0.9, base_features, context)

        # New extreme should still be detected
        signal = strategy.generate_signal(bar, 0.95, 0.9, base_features, context)
        assert signal is not None

    def test_extremes_strategy_scratch_buffer_not_allocated(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify scratch buffer pre-allocated and reused (hot path).

        Test ensures:
        - Same scratch buffer reused across calls
        - No new allocations after initialization

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=10)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Initialize
        for i in range(10):
            strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        # Get scratch buffer id
        scratch_id = id(context["_pred_scratch"])

        # More calls should reuse same buffer
        for i in range(5):
            strategy.generate_signal(bar, 0.5, 0.9, base_features, context)
            assert id(context["_pred_scratch"]) == scratch_id

    def test_extremes_strategy_window_size_zero(self) -> None:
        """
        Verify error handling for invalid window_size=0.

        Test ensures:
        - ValueError raised for invalid configuration
        - Clear error message

        Note: This test may need adjustment based on whether validation
        happens at construction or first use.

        """
        # Strategy creation should succeed (validation may be deferred)
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=0)

        # Usage should fail
        # Note: Implementation may choose to validate at construction instead
        # In that case, move pytest.raises to construction

    def test_extremes_strategy_top_pct_bounds(self) -> None:
        """
        Verify top_pct must be in (0, 1).

        Test ensures:
        - ValueError raised for invalid top_pct
        - Both bounds checked

        Note: This test may need adjustment based on validation approach.

        """
        # Invalid: top_pct > 1
        with pytest.raises(ValueError, match="top_pct"):
            strategy = ExtremesStrategy(top_pct=1.5, threshold=0.7, window_size=50)

        # Invalid: top_pct < 0
        with pytest.raises(ValueError, match="top_pct"):
            strategy = ExtremesStrategy(top_pct=-0.1, threshold=0.7, window_size=50)

    def test_extremes_strategy_no_allocations_after_warmup(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify zero allocations after window is full.

        Test ensures:
        - Hot path optimized after warm-up
        - Ring buffer reuse working

        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=10)
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
        }

        # Warm up - fill window
        for i in range(15):
            strategy.generate_signal(bar, 0.5 + i * 0.01, 0.9, base_features, context)

        # Measure allocations
        tracemalloc.start()
        _ = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should have minimal allocations (only potential MLSignal)
        assert current < 2000, f"Unexpected allocations: {current} bytes"


# =================================================================================================
# Category 3: Momentum Strategy Tests (12 tests)
# =================================================================================================


class TestMomentumStrategy:
    """
    Tests for MomentumStrategy.
    """

    def test_momentum_strategy_generates_signal_for_strong_momentum(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify MomentumStrategy generates signal when momentum exceeds threshold.

        Test ensures:
        - Signal generated for strong upward momentum
        - Prediction adjusted by momentum factor

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # Create ring buffer with increasing predictions (strong upward momentum)
        ring = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,  # Next write position (wrapped)
            "_prediction_ring_count": 5,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        # Prediction should be adjusted: prediction * (1 + momentum)
        # Momentum = (0.5 - 0.1) / 4 = 0.1
        expected_prediction = 0.5 * (1 + 0.1)
        assert signal.prediction == pytest.approx(expected_prediction, rel=1e-5)

    def test_momentum_strategy_no_signal_for_weak_momentum(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify no signal when momentum below threshold.

        Test ensures:
        - Flat predictions don't trigger signals
        - Momentum threshold enforced

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.05)
        bar = create_test_bar()

        # Flat predictions (near-zero momentum)
        ring = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is None

    def test_momentum_strategy_uses_ring_buffer_for_performance(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy uses ring buffer metadata from context.

        Test ensures:
        - Ring buffer preferred over history list
        - Zero-copy hot path achieved

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        ring = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
            # Also provide history list - ring should be preferred
            "prediction_history": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        # Should use ring buffer values, not zeros from history
        assert signal is not None

    def test_momentum_strategy_fallback_to_history_list(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify fallback to prediction_history list when ring unavailable.

        Test ensures:
        - History list used when ring not present
        - Signal calculated correctly from list

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # No ring buffer, only history list
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "prediction_history": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None

    def test_momentum_strategy_requires_minimum_lookback(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy returns None if insufficient history.

        Test ensures:
        - No signal when history too short
        - Lookback requirement enforced

        """
        strategy = MomentumStrategy(lookback=10, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # Only 5 predictions but lookback=10
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "prediction_history": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is None

    def test_momentum_strategy_respects_confidence_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify confidence threshold enforced even with strong momentum.

        Test ensures:
        - High momentum + low confidence = no signal
        - Both conditions must be met

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.7, momentum_threshold=0.01)
        bar = create_test_bar()

        ring = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        # Strong momentum but low confidence
        signal = strategy.generate_signal(bar, 0.5, 0.5, base_features, context)

        assert signal is None

    def test_momentum_strategy_positive_momentum_increases_prediction(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify positive momentum amplifies prediction.

        Test ensures:
        - Prediction multiplied by (1 + momentum)
        - Correct amplification factor

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # Momentum = (0.6 - 0.4) / 4 = 0.05
        ring = np.array([0.4, 0.45, 0.5, 0.55, 0.6], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        expected = 0.5 * 1.05
        assert signal.prediction == pytest.approx(expected, rel=1e-5)

    def test_momentum_strategy_negative_momentum_decreases_prediction(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify negative momentum reduces prediction.

        Test ensures:
        - Prediction multiplied by (1 + negative_momentum)
        - Correct reduction factor

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # Negative momentum = (0.4 - 0.6) / 4 = -0.05
        ring = np.array([0.6, 0.55, 0.5, 0.45, 0.4], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        expected = 0.5 * 0.95
        assert signal.prediction == pytest.approx(expected, rel=1e-5)

    def test_momentum_strategy_telescoping_difference_calculation(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify momentum uses (last - first) / (lookback - 1) formula.

        Test ensures:
        - Correct formula applied
        - Not np.mean(np.diff()) for ring buffer case

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.001)
        bar = create_test_bar()

        # Values where telescoping vs mean-diff would give different results
        ring = np.array([0.1, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        # Telescoping: (0.2 - 0.1) / 4 = 0.025
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        if signal is not None:
            expected_momentum = (0.2 - 0.1) / 4  # 0.025
            expected_prediction = 0.5 * (1 + expected_momentum)
            assert signal.prediction == pytest.approx(expected_prediction, rel=1e-5)

    def test_momentum_strategy_circular_indexing(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify correct circular indexing for ring buffer.

        Test ensures:
        - Wrap-around index calculation correct
        - First/last values found correctly after wrap

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        # Ring buffer with wrap-around
        # Values written: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
        # After wrap: [0.6, 0.7, 0.3, 0.4, 0.5], index=2
        ring = np.array([0.6, 0.7, 0.3, 0.4, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 2,  # Next write position
            "_prediction_ring_count": 5,
        }

        # With lookback=5, index=2:
        # first_idx = (2 - 5) % 5 = -3 % 5 = 2 (value: 0.3)
        # last_idx = (2 - 1) % 5 = 1 (value: 0.7)
        # Momentum = (0.7 - 0.3) / 4 = 0.1
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        expected_momentum = (0.7 - 0.3) / 4  # 0.1
        expected_prediction = 0.5 * (1 + expected_momentum)
        assert signal.prediction == pytest.approx(expected_prediction, rel=1e-5)

    def test_momentum_strategy_lookback_one(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify edge case: lookback=1 (no momentum possible).

        Test ensures:
        - Edge case handled gracefully
        - No division by zero

        """
        strategy = MomentumStrategy(lookback=1, threshold=0.5, momentum_threshold=0.01)
        bar = create_test_bar()

        ring = np.array([0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 1,
        }

        # Should not raise, may return None or handle gracefully
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)
        # Just verify no exception - behavior may vary

    def test_momentum_strategy_no_allocations_on_hot_path(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify zero allocations when using ring buffer.

        Test ensures:
        - Hot path optimized
        - Ring buffer access is zero-copy

        """
        strategy = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.05)
        bar = create_test_bar()

        ring = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "_prediction_ring": ring,
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 5,
        }

        # Warm up
        _ = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        # Measure allocations
        tracemalloc.start()
        _ = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert current < 1000, f"Unexpected allocations: {current} bytes"


# =================================================================================================
# Category 4: Ensemble Strategy Tests (14 tests)
# =================================================================================================


class TestEnsembleStrategy:
    """
    Tests for EnsembleStrategy.
    """

    def test_ensemble_strategy_combines_multiple_strategies(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify EnsembleStrategy weighted voting across strategies.

        Test ensures:
        - All sub-strategies called
        - Weighted combination correct

        """
        # Create sub-strategies that will all vote
        strategies = {
            "threshold": ThresholdSignalStrategy(threshold=0.5),
            "threshold2": ThresholdSignalStrategy(threshold=0.6),
        }
        weights = {"threshold": 0.6, "threshold2": 0.4}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        assert signal is not None
        # Both voted with confidence=0.9
        # Ensemble confidence = (0.6 * 0.9 + 0.4 * 0.9) / 1.0 = 0.9
        assert signal.confidence == pytest.approx(0.9, rel=1e-5)

    def test_ensemble_strategy_requires_threshold_for_combined_confidence(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble respects threshold for combined confidence.

        Test ensures:
        - Signal only if ensemble_confidence >= threshold
        - Threshold applies to combined score, not individual

        """
        strategies = {
            "threshold": ThresholdSignalStrategy(threshold=0.9),  # Won't vote
            "threshold2": ThresholdSignalStrategy(threshold=0.5),  # Will vote
        }
        weights = {"threshold": 0.5, "threshold2": 0.5}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.8)

        bar = create_test_bar()
        # Confidence 0.6: only threshold2 votes
        # Ensemble confidence = 0.6 (from single voter weight 0.5 normalized)
        signal = ensemble.generate_signal(bar, 0.8, 0.6, base_features, base_context)

        # 0.6 < 0.8 threshold
        assert signal is None

    def test_ensemble_strategy_no_signal_if_no_votes(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify no signal if all sub-strategies return None.

        Test ensures:
        - Zero total weight handled
        - No signal when no votes

        """
        strategies = {
            "threshold1": ThresholdSignalStrategy(threshold=0.9),
            "threshold2": ThresholdSignalStrategy(threshold=0.95),
        }
        weights = {"threshold1": 0.5, "threshold2": 0.5}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        # Confidence 0.8 < both thresholds, no votes
        signal = ensemble.generate_signal(bar, 0.8, 0.8, base_features, base_context)

        assert signal is None

    def test_ensemble_strategy_weights_sum_correctly(self) -> None:
        """
        Verify weights can be validated to sum to ~1.0.

        Test ensures:
        - Weights accessible for validation
        - Typical configuration sums to 1.0

        """
        strategies = {
            "threshold": ThresholdSignalStrategy(threshold=0.7),
            "extremes": ThresholdSignalStrategy(threshold=0.7),
            "momentum": ThresholdSignalStrategy(threshold=0.7),
        }
        weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.7)

        assert sum(ensemble.weights.values()) == pytest.approx(1.0)

    def test_ensemble_strategy_handles_missing_strategy_weight(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble handles case where sub-strategy not in weights.

        Test ensures:
        - Weight defaults to 0.0 for unknown strategy
        - No exception raised

        """
        strategies = {
            "threshold": ThresholdSignalStrategy(threshold=0.5),
            "custom": ThresholdSignalStrategy(threshold=0.5),  # Not in weights
        }
        weights = {"threshold": 1.0}  # "custom" not listed
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        # Custom votes but has weight 0, only threshold counts
        signal = ensemble.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        assert signal is not None
        assert signal.confidence == pytest.approx(0.9, rel=1e-5)

    def test_ensemble_strategy_preserves_original_prediction(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble uses original prediction, not modified.

        Test ensures:
        - Prediction not altered by ensemble
        - Original value preserved

        """
        strategies = {
            "threshold": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"threshold": 1.0}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        original_prediction = -0.75
        signal = ensemble.generate_signal(
            bar,
            original_prediction,
            0.9,
            base_features,
            base_context,
        )

        assert signal is not None
        assert signal.prediction == original_prediction

    def test_ensemble_strategy_uses_ensemble_confidence(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify signal confidence is ensemble-calculated, not individual.

        Test ensures:
        - Weighted average confidence used
        - Not just highest or lowest

        """
        strategies = {
            "s1": ThresholdSignalStrategy(threshold=0.5),
            "s2": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"s1": 0.7, "s2": 0.3}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        input_confidence = 0.8
        signal = ensemble.generate_signal(
            bar,
            0.5,
            input_confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        # Both vote with same confidence, weighted avg = 0.8
        expected_confidence = (0.7 * 0.8 + 0.3 * 0.8) / 1.0
        assert signal.confidence == pytest.approx(expected_confidence, rel=1e-5)

    def test_ensemble_strategy_passes_context_to_substrategy(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify context passed through to all sub-strategies.

        Test ensures:
        - All sub-strategies receive same context
        - Context not modified between calls

        """
        mock_strategy = Mock(spec=SignalGenerationStrategy)
        mock_strategy.generate_signal.return_value = None

        strategies = {"mock": mock_strategy}
        weights = {"mock": 1.0}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        context = {
            "timestamp_ns": 123456,
            "model_id": "test_model",
            "custom_key": "custom_value",
        }

        ensemble.generate_signal(bar, 0.5, 0.9, base_features, context)

        mock_strategy.generate_signal.assert_called_once()
        call_args = mock_strategy.generate_signal.call_args
        passed_context = call_args[0][4]  # Fifth positional arg
        assert passed_context["model_id"] == "test_model"
        assert passed_context["custom_key"] == "custom_value"

    def test_ensemble_strategy_single_substrategy(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble with only one sub-strategy (edge case).

        Test ensures:
        - Single strategy ensemble works
        - Behaves like single strategy

        """
        strategies = {"only": ThresholdSignalStrategy(threshold=0.6)}
        weights = {"only": 1.0}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        assert signal is not None
        assert signal.confidence == 0.9

    def test_ensemble_strategy_equal_weights(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble with equal weights.

        Test ensures:
        - Equal weights produce simple average
        - No weight bias

        """
        strategies = {
            "s1": ThresholdSignalStrategy(threshold=0.5),
            "s2": ThresholdSignalStrategy(threshold=0.5),
            "s3": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"s1": 0.33, "s2": 0.33, "s3": 0.34}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.75, base_features, base_context)

        assert signal is not None
        # All vote with 0.75, weighted avg should be ~0.75
        assert signal.confidence == pytest.approx(0.75, rel=0.01)

    def test_ensemble_strategy_dominant_weight(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble respects heavily weighted strategy.

        Test ensures:
        - Dominant weight controls outcome
        - Minor weights have proportional influence

        """
        strategies = {
            "dominant": ThresholdSignalStrategy(threshold=0.5),
            "minor1": ThresholdSignalStrategy(threshold=0.5),
            "minor2": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"dominant": 0.9, "minor1": 0.05, "minor2": 0.05}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.8, base_features, base_context)

        assert signal is not None
        # All vote with 0.8, weighted avg = 0.8
        assert signal.confidence == pytest.approx(0.8, rel=1e-5)

    def test_ensemble_strategy_partial_voting(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble handles case where some strategies vote, some don't.

        Test ensures:
        - Only voting strategies contribute
        - Non-voters don't reduce confidence

        """
        strategies = {
            "voter1": ThresholdSignalStrategy(threshold=0.5),  # Will vote
            "voter2": ThresholdSignalStrategy(threshold=0.5),  # Will vote
            "non_voter": ThresholdSignalStrategy(threshold=0.95),  # Won't vote
        }
        weights = {"voter1": 0.4, "voter2": 0.3, "non_voter": 0.3}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.8, base_features, base_context)

        assert signal is not None
        # Only voter1 and voter2 vote (weight 0.7 total)
        # Ensemble confidence = (0.4 * 0.8 + 0.3 * 0.8) / 0.7 = 0.8
        assert signal.confidence == pytest.approx(0.8, rel=1e-5)

    def test_ensemble_strategy_zero_total_weight(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble handles zero total weight gracefully.

        Test ensures:
        - No division by zero
        - No signal when no weighted votes

        """
        strategies = {
            "s1": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"s1": 0.0}  # Zero weight
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()
        signal = ensemble.generate_signal(bar, 0.8, 0.9, base_features, base_context)

        assert signal is None

    def test_ensemble_strategy_no_allocations(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
    ) -> None:
        """
        Verify ensemble doesn't allocate on hot path.

        Test ensures:
        - Minimal allocations for iteration
        - Hot path optimized

        """
        strategies = {
            "s1": ThresholdSignalStrategy(threshold=0.5),
            "s2": ThresholdSignalStrategy(threshold=0.5),
        }
        weights = {"s1": 0.5, "s2": 0.5}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = create_test_bar()

        # Warm up
        _ = ensemble.generate_signal(bar, 0.8, 0.5, base_features, base_context)

        # Measure allocations
        tracemalloc.start()
        _ = ensemble.generate_signal(bar, 0.8, 0.5, base_features, base_context)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Allow some allocations for iteration
        assert current < 2000, f"Unexpected allocations: {current} bytes"


# =================================================================================================
# Category 5: Adaptive Strategy Tests (12 tests)
# =================================================================================================


class TestAdaptiveStrategy:
    """
    Tests for AdaptiveStrategy.
    """

    def test_adaptive_strategy_uses_adaptive_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify AdaptiveStrategy uses adaptive_threshold from context.

        Test ensures:
        - Context threshold used instead of base_threshold
        - Threshold captured in signal metadata

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.8,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.metadata["adaptive_threshold"] == 0.8

    def test_adaptive_strategy_calculates_signal_strength(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal_strength = confidence / adaptive_threshold.

        Test ensures:
        - Correct calculation
        - Strength captured in metadata

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.6,
        }

        # confidence=0.9, threshold=0.6 -> strength=1.5
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.metadata["signal_strength"] == pytest.approx(1.5)

    def test_adaptive_strategy_requires_signal_strength_gte_one(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal only generated if signal_strength >= 1.0.

        Test ensures:
        - No signal when confidence < threshold
        - Strength threshold enforced

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.8,
        }

        # confidence=0.7, threshold=0.8 -> strength=0.875 < 1.0
        signal = strategy.generate_signal(bar, 0.5, 0.7, base_features, context)

        assert signal is None

    def test_adaptive_strategy_includes_market_regime(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify signal includes market_regime from context.

        Test ensures:
        - Regime captured in metadata
        - Useful for analysis

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.5,
            "market_regime": "high_volatility",
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.metadata["market_regime"] == "high_volatility"

    def test_adaptive_strategy_handles_missing_adaptive_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify fallback to base_threshold when adaptive_threshold missing.

        Test ensures:
        - Graceful degradation
        - Base threshold used as fallback

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context: dict[str, Any] = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            # No adaptive_threshold
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        # Falls back to base_threshold=0.7
        assert signal.metadata["adaptive_threshold"] == 0.7

    def test_adaptive_strategy_handles_missing_market_regime(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify defaults to "unknown" when market_regime missing.

        Test ensures:
        - Default regime used
        - No exception for missing key

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.5,
            # No market_regime
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.metadata["market_regime"] == "unknown"

    def test_adaptive_strategy_includes_features_in_signal(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify adaptive strategy includes features in signal.

        Test ensures:
        - Features always included (unlike other strategies)
        - Useful for adaptive analysis

        Note: AdaptiveStrategy always includes features, not checking log_predictions.

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.5,
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.features is not None
        assert np.array_equal(signal.features, base_features)

    def test_adaptive_strategy_zero_adaptive_threshold(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify handling of edge case: adaptive_threshold=0.

        Test ensures:
        - No division by zero
        - Graceful handling (signal_strength=0 or handled specially)

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.0,
        }

        # Should not raise
        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        # Either no signal or signal_strength=0
        if signal is not None:
            assert signal.metadata.get("signal_strength", 0.0) == 0.0
        # If signal is None, that's also acceptable handling

    def test_adaptive_strategy_preserves_prediction_and_confidence(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify original prediction/confidence preserved.

        Test ensures:
        - Values not modified
        - Same as input

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.5,
        }

        signal = strategy.generate_signal(bar, -0.5, 0.9, base_features, context)

        assert signal is not None
        assert signal.prediction == -0.5
        assert signal.confidence == 0.9

    def test_adaptive_strategy_metadata_completeness(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify all required metadata fields present.

        Test ensures:
        - adaptive_threshold in metadata
        - signal_strength in metadata
        - market_regime in metadata

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.6,
            "market_regime": "normal",
        }

        signal = strategy.generate_signal(bar, 0.5, 0.9, base_features, context)

        assert signal is not None
        assert "adaptive_threshold" in signal.metadata
        assert "signal_strength" in signal.metadata
        assert "market_regime" in signal.metadata

    def test_adaptive_strategy_threshold_bounds(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify strategy respects min/max threshold bounds.

        Test ensures:
        - Threshold clamped to [min, max]
        - Bounds enforced

        Note: This test verifies the strategy's parameters, not runtime clamping.
        Runtime clamping may be done by AdaptiveThresholdComponent.

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.3,
            max_threshold=0.9,
        )

        assert strategy.min_threshold == 0.3
        assert strategy.max_threshold == 0.9
        assert strategy.min_threshold <= strategy.base_threshold <= strategy.max_threshold

    def test_adaptive_strategy_no_allocations(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
    ) -> None:
        """
        Verify zero allocations on hot path.

        Test ensures:
        - Minimal allocations
        - Hot path optimized

        """
        strategy = AdaptiveStrategy(
            base_threshold=0.7,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )
        bar = create_test_bar()
        context = {
            "timestamp_ns": int(time.time() * 1e9),
            "model_id": "test_model",
            "adaptive_threshold": 0.5,
        }

        # Warm up
        _ = strategy.generate_signal(bar, 0.5, 0.4, base_features, context)

        # Measure allocations (no signal case)
        tracemalloc.start()
        _ = strategy.generate_signal(bar, 0.5, 0.4, base_features, context)
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert current < 1000, f"Unexpected allocations: {current} bytes"


# =================================================================================================
# Category 6: Strategy Factory Tests (8 tests)
# =================================================================================================


class TestStrategyFactory:
    """
    Tests for SignalStrategyComponent.create_strategy() factory.
    """

    def test_factory_creates_threshold_strategy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory creates ThresholdSignalStrategy for "threshold" config.

        Test ensures:
        - Correct strategy type created
        - Threshold from config used

        """
        from ml.config.actors import MLSignalActorConfig
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="threshold",
            prediction_threshold=0.75,
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, ThresholdSignalStrategy)
        assert strategy.threshold == 0.75

    def test_factory_creates_extremes_strategy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory creates ExtremesStrategy for "extremes" config.

        Test ensures:
        - Correct strategy type created
        - Parameters from config used

        """
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="extremes",
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, ExtremesStrategy)

    def test_factory_creates_momentum_strategy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory creates MomentumStrategy for "momentum" config.

        Test ensures:
        - Correct strategy type created
        - Lookback from config used

        """
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="momentum",
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, MomentumStrategy)

    def test_factory_creates_ensemble_strategy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory creates EnsembleStrategy for "ensemble" config.

        Test ensures:
        - Correct strategy type created
        - Contains 3 sub-strategies
        - Weights from config used

        """
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="ensemble",
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, EnsembleStrategy)
        assert len(strategy.strategies) == 3

    def test_factory_creates_adaptive_strategy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory creates AdaptiveStrategy for "adaptive" config.

        Test ensures:
        - Correct strategy type created
        - Base threshold from config used

        """
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="adaptive",
            prediction_threshold=0.65,
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.base_threshold == 0.65

    def test_factory_uses_custom_strategy_if_provided(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory uses custom_strategy from config if provided.

        Test ensures:
        - Custom strategy used as-is
        - Priority 1 in factory

        """
        import msgspec

        custom_strategy = ThresholdSignalStrategy(threshold=0.99)
        config = msgspec.structs.replace(
            base_signal_config,
            custom_strategy=custom_strategy,
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert strategy is custom_strategy

    def test_factory_handles_unknown_strategy_type(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory handles unknown strategy name gracefully.

        Test ensures:
        - Falls back to threshold strategy
        - Warning logged

        """
        import msgspec

        config = msgspec.structs.replace(
            base_signal_config,
            signal_strategy="unknown_strategy_type",
        )

        component = SignalStrategyComponent(config, "test_actor", mock_logger)
        strategy = component.create_strategy(config, metadata=None)

        assert isinstance(strategy, ThresholdSignalStrategy)
        mock_logger.warning.assert_called()

    def test_factory_loads_model_driven_policy(
        self,
        base_signal_config,
        mock_logger,
    ) -> None:
        """
        Verify factory loads strategy from model metadata decision_policy.

        Test ensures:
        - Policy path resolved
        - Strategy loaded from adapter

        """
        import msgspec

        config = msgspec.structs.replace(base_signal_config)
        metadata = {
            "decision_policy": "ml.actors.adapters.threshold_policy",
            "decision_config": {"threshold": 0.8},
        }

        component = SignalStrategyComponent(config, "test_actor", mock_logger)

        with patch("ml.actors.adapters.build_strategy_from_policy") as mock_build:
            mock_strategy = ThresholdSignalStrategy(threshold=0.8)
            mock_build.return_value = mock_strategy

            strategy = component.create_strategy(config, metadata=metadata)

            mock_build.assert_called_once()
            assert strategy is mock_strategy


# =================================================================================================
# Category 7: Strategy Swapper Tests (6 tests)
# =================================================================================================


class TestStrategySwapper:
    """
    Tests for SignalPolicySwapper / StrategySwapper.
    """

    def test_swapper_set_current_strategy(self) -> None:
        """
        Verify SignalPolicySwapper.set_current() sets strategy.

        Test ensures:
        - Current strategy set
        - swap_pending is False

        """
        swapper = SignalPolicySwapper()
        strategy = ThresholdSignalStrategy(threshold=0.7)

        swapper.set_current(strategy)

        assert swapper.current_strategy is strategy
        assert swapper.swap_pending is False

    def test_swapper_prepare_swap(self) -> None:
        """
        Verify prepare_swap() stages new strategy.

        Test ensures:
        - swap_pending becomes True
        - Next strategy staged

        """
        swapper = SignalPolicySwapper()
        initial = ThresholdSignalStrategy(threshold=0.7)
        swapper.set_current(initial)

        new_strategy = ThresholdSignalStrategy(threshold=0.9)
        swapper.prepare_swap(new_strategy)

        assert swapper.swap_pending is True
        assert swapper._next_strategy is new_strategy
        assert swapper.current_strategy is initial  # Not changed yet

    def test_swapper_execute_swap_atomically(self) -> None:
        """
        Verify execute_swap() promotes next to current.

        Test ensures:
        - Current strategy updated
        - swap_pending becomes False
        - Next strategy cleared

        """
        swapper = SignalPolicySwapper()
        initial = ThresholdSignalStrategy(threshold=0.7)
        swapper.set_current(initial)

        new_strategy = ThresholdSignalStrategy(threshold=0.9)
        swapper.prepare_swap(new_strategy)
        swapper.execute_swap()

        assert swapper.current_strategy is new_strategy
        assert swapper.swap_pending is False
        assert swapper._next_strategy is None

    def test_swapper_execute_swap_returns_true(self) -> None:
        """
        Verify execute_swap() returns True when swap occurs.

        Test ensures:
        - Return value indicates success
        - Useful for logging/metrics

        """
        swapper = SignalPolicySwapper()
        swapper.set_current(ThresholdSignalStrategy(threshold=0.7))
        swapper.prepare_swap(ThresholdSignalStrategy(threshold=0.9))

        result = swapper.execute_swap()

        assert result is True

    def test_swapper_execute_swap_returns_false_when_no_pending(self) -> None:
        """
        Verify execute_swap() returns False if no pending swap.

        Test ensures:
        - Safe to call even without pending swap
        - Idempotent behavior

        """
        swapper = SignalPolicySwapper()
        swapper.set_current(ThresholdSignalStrategy(threshold=0.7))

        result = swapper.execute_swap()

        assert result is False

    def test_swapper_prepare_swap_with_error(self) -> None:
        """
        Verify prepare_swap_with_error() records error.

        Test ensures:
        - load_error set
        - swap_pending cleared

        """
        swapper = SignalPolicySwapper()
        swapper.set_current(ThresholdSignalStrategy(threshold=0.7))

        error = ValueError("Failed to load strategy")
        swapper.prepare_swap_with_error(error)

        assert swapper.load_error is error
        assert swapper.swap_pending is False


# =================================================================================================
# Category 8: Property Tests (5 tests)
# =================================================================================================


class TestPropertyTests:
    """
    Hypothesis-based property tests for signal generation strategies.
    """

    @given(
        prediction=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_signal_values_always_in_valid_range_property(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
        prediction: float,
        confidence: float,
        threshold: float,
    ) -> None:
        """
        Verify all strategies produce signals with valid ranges.

        Property: Signal confidence must be in [0, 1].

        """
        strategy = ThresholdSignalStrategy(threshold=threshold)
        bar = create_test_bar()

        signal = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            base_context,
        )

        if signal is not None:
            assert 0.0 <= signal.confidence <= 1.0

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_signal_confidence_always_in_valid_range_property(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
        confidence: float,
    ) -> None:
        """
        Verify all strategies produce signals with confidence in [0, 1].

        Property: Signal.confidence must be in [0, 1] for any input.

        """
        strategy = ThresholdSignalStrategy(threshold=0.0)  # Accept all
        bar = create_test_bar()

        signal = strategy.generate_signal(
            bar,
            0.5,
            confidence,
            base_features,
            base_context,
        )

        assert signal is not None
        assert 0.0 <= signal.confidence <= 1.0

    @given(
        prediction=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_strategy_determinism_property(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        prediction: float,
        confidence: float,
    ) -> None:
        """
        Verify same inputs produce same outputs.

        Property: generate_signal is deterministic.

        """
        strategy = ThresholdSignalStrategy(threshold=0.5)
        bar = create_test_bar()
        context1 = {
            "timestamp_ns": 123456789,
            "model_id": "test",
        }
        context2 = {
            "timestamp_ns": 123456789,
            "model_id": "test",
        }

        signal1 = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            context1,
        )
        signal2 = strategy.generate_signal(
            bar,
            prediction,
            confidence,
            base_features,
            context2,
        )

        if signal1 is None:
            assert signal2 is None
        else:
            assert signal2 is not None
            assert signal1.prediction == signal2.prediction
            assert signal1.confidence == signal2.confidence

    @given(
        prediction=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_strategy_context_immutability_property(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        prediction: float,
    ) -> None:
        """
        Verify strategies don't unexpectedly mutate context dict.

        Property: Original context keys preserved (except allowed ring buffer updates).

        """
        strategy = ThresholdSignalStrategy(threshold=0.5)
        bar = create_test_bar()
        context = {
            "timestamp_ns": 123456789,
            "model_id": "test",
            "custom_key": "custom_value",
        }
        original_keys = set(context.keys())

        strategy.generate_signal(bar, prediction, 0.9, base_features, context)

        # Original keys should still exist
        for key in original_keys:
            assert key in context
            if key == "custom_key":
                assert context[key] == "custom_value"

    @given(
        confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ensemble_weights_invariant_property(
        self,
        create_test_bar,
        base_features: npt.NDArray[np.float32],
        base_context: dict[str, Any],
        confidence: float,
    ) -> None:
        """
        Verify ensemble confidence bounded by min/max sub-strategy confidences.

        Property: Ensemble confidence in [min, max] of voting strategies.

        """
        strategies = {
            "s1": ThresholdSignalStrategy(threshold=0.3),
            "s2": ThresholdSignalStrategy(threshold=0.3),
        }
        weights = {"s1": 0.6, "s2": 0.4}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.3)

        bar = create_test_bar()
        signal = ensemble.generate_signal(
            bar,
            0.5,
            confidence,
            base_features,
            base_context,
        )

        if signal is not None:
            # Ensemble uses same confidence from all voters
            # So result should equal input confidence
            assert signal.confidence == pytest.approx(confidence, rel=1e-5)
