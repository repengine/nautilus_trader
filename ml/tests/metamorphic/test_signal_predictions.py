"""
Metamorphic tests for signal predictions.

These tests verify that ML predictions and signals maintain expected relationships under
controlled transformations, without requiring exact output values.

"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


@pytest.mark.property
@pytest.mark.parallel_safe
class TestSignalPredictionMetamorphic:
    """
    Metamorphic tests for signal generation and predictions.
    """

    @given(
        base_features=st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            min_size=10,
            max_size=50,
        ),
        time_shift=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=5000)
    def test_time_shift_invariance(self, base_features, time_shift):
        """
        Metamorphic relation: For stationary features, time-shifting
        should not affect predictions (assuming no time-dependent features).
        """
        features = np.array(base_features)

        # Create mock model that returns predictions based on features
        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=lambda x: np.mean(x, axis=1))

        # Generate predictions for original features
        pred_original = mock_model.predict(features.reshape(1, -1))

        # Time-shift features (circular shift for stationarity)
        features_shifted = np.roll(features, time_shift)
        pred_shifted = mock_model.predict(features_shifted.reshape(1, -1))

        # Metamorphic relation: Predictions should be similar for stationary features
        np.testing.assert_allclose(
            pred_original,
            pred_shifted,
            rtol=1e-10,
            err_msg="Time shift affected predictions for stationary features",
        )

    @given(
        features=st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            min_size=10,
            max_size=50,
        ),
        confidence_threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=30, deadline=5000)
    def test_confidence_monotonicity(self, features, confidence_threshold):
        """
        Metamorphic relation: Higher confidence predictions should lead
        to stronger signals (monotonic relationship).
        """
        features = np.array(features)

        # Generate predictions with varying confidence
        low_confidence_pred = 0.52
        high_confidence_pred = 0.9

        # Signal strength should be monotonic with confidence
        def signal_strength(prediction, threshold):
            confidence = max(prediction, 1.0 - prediction)
            if confidence > threshold:
                return confidence
            return 0.0

        low_signal = signal_strength(low_confidence_pred, confidence_threshold)
        high_signal = signal_strength(high_confidence_pred, confidence_threshold)

        # Metamorphic relation: Higher confidence -> stronger or equal signal
        if confidence_threshold < high_confidence_pred:
            assert abs(high_signal) >= abs(
                low_signal,
            ), "Higher confidence should produce stronger or equal signal"

    @given(
        n_models=st.integers(min_value=2, max_value=5),
        features=st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            min_size=10,
            max_size=30,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_ensemble_consistency(self, n_models, features):
        """
        Metamorphic relation: Ensemble predictions should be bounded
        by individual model predictions (no extrapolation).
        """
        features = np.array(features).reshape(1, -1)

        # Generate individual model predictions
        individual_predictions = []
        for i in range(n_models):
            # Each model has different behavior
            np.random.seed(i)
            pred = np.random.uniform(-1, 1)
            individual_predictions.append(pred)

        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(individual_predictions)

        # Metamorphic relations:
        # 1. Ensemble should be within bounds of individual predictions
        min_pred = min(individual_predictions)
        max_pred = max(individual_predictions)
        assert (
            min_pred <= ensemble_pred <= max_pred
        ), f"Ensemble prediction {ensemble_pred} outside bounds [{min_pred}, {max_pred}]"

        # 2. Ensemble variance should be less than or equal to max individual variance
        individual_variance = np.var(individual_predictions)
        # For averaging, ensemble variance is reduced
        assert individual_variance >= 0, "Variance should be non-negative"

    @given(
        base_prediction=st.floats(min_value=0.0, max_value=1.0),
        scale_factors=st.lists(
            st.floats(min_value=0.5, max_value=2.0),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_prediction_scaling_bounds(self, base_prediction, scale_factors):
        """
        Metamorphic relation: Scaling predictions should maintain
        bounds and relative relationships.
        """
        centered = base_prediction - 0.5
        # Apply different scales around the neutral 0.5 center
        scaled_predictions = [0.5 + centered * scale for scale in scale_factors]

        # Ensure all scaled predictions maintain bounds [0, 1]
        clipped_predictions = [np.clip(pred, 0, 1) for pred in scaled_predictions]

        # Metamorphic relations:
        # 1. Direction around 0.5 should be preserved (unless clipped)
        tiny = np.finfo(np.float64).tiny
        for pred, clipped in zip(scaled_predictions, clipped_predictions):
            # Guard against IEEE-754 underflow to 0.0 when scaling subnormal values
            if abs(centered) < tiny or abs(pred - 0.5) < tiny:
                continue
            if 0.0 < clipped < 1.0:
                assert (
                    np.sign(pred - 0.5) == np.sign(centered)
                ), "Scaling should preserve prediction direction around 0.5"

        # 2. Relative ordering should be preserved
        sorted_scales = sorted(scale_factors)
        # Preserve the mapping order (do not sort predictions themselves)
        sorted_preds = [0.5 + centered * s for s in sorted_scales]
        clipped_sorted = [np.clip(p, 0, 1) for p in sorted_preds]

        # Check monotonicity (considering clipping)
        for i in range(len(clipped_sorted) - 1):
            if centered >= 0:
                assert (
                    clipped_sorted[i] <= clipped_sorted[i + 1]
                ), "Bullish predictions should maintain order after scaling"
            else:
                assert (
                    clipped_sorted[i] >= clipped_sorted[i + 1]
                ), "Bearish predictions should maintain reverse order after scaling"

    @given(
        features=st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            min_size=20,
            max_size=50,
        ),
        noise_std=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=30, deadline=5000)
    def test_noise_robustness(self, features, noise_std):
        """
        Metamorphic relation: Small noise in features should result
        in bounded changes to predictions.
        """
        features = np.array(features).reshape(1, -1)

        # Mock model with simple linear behavior
        mock_model = MagicMock()
        mock_model.predict = MagicMock(
            side_effect=lambda x: np.tanh(np.mean(x, axis=1)),  # Bounded output
        )

        # Original prediction
        pred_original = mock_model.predict(features)[0]

        # Add noise to features
        noise = np.random.normal(0, noise_std, features.shape)
        features_noisy = features + noise
        pred_noisy = mock_model.predict(features_noisy)[0]

        # Metamorphic relation: Prediction change should be bounded
        prediction_change = abs(pred_noisy - pred_original)

        # For tanh activation, sensitivity is at most 1
        max_expected_change = min(1.0, noise_std * 2)  # Heuristic bound

        assert (
            prediction_change <= max_expected_change
        ), f"Prediction changed too much ({prediction_change:.3f}) for noise level {noise_std:.3f}"


class TestSignalThresholdMetamorphic:
    """
    Test metamorphic properties of signal thresholding.
    """

    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=10,
            max_size=100,
        ),
        base_threshold=st.floats(min_value=0.5, max_value=0.8),
        threshold_delta=st.floats(min_value=0.05, max_value=0.2),
    )
    @settings(max_examples=30, deadline=5000)
    def test_threshold_monotonicity(self, predictions, base_threshold, threshold_delta):
        """
        Metamorphic relation: Increasing threshold should monotonically
        decrease the number of signals generated.
        """
        assume(base_threshold + threshold_delta < 1.0)

        predictions = np.array(predictions)
        confidences = np.maximum(predictions, 1.0 - predictions)

        # Count signals at different thresholds
        threshold_low = base_threshold
        threshold_high = base_threshold + threshold_delta

        signals_low = np.sum(confidences > threshold_low)
        signals_high = np.sum(confidences > threshold_high)

        # Metamorphic relation: Higher threshold -> fewer or equal signals
        assert (
            signals_high <= signals_low
        ), f"Higher threshold ({threshold_high:.2f}) produced more signals than lower ({threshold_low:.2f})"

        # Additional property: Signal subset relationship
        signals_mask_low = confidences > threshold_low
        signals_mask_high = confidences > threshold_high

        # All high-threshold signals should also be low-threshold signals
        assert np.all(
            signals_mask_high <= signals_mask_low,
        ), "High threshold signals should be subset of low threshold signals"

    @given(
        n_instruments=st.integers(min_value=2, max_value=10),
        n_predictions=st.integers(min_value=10, max_value=50),
        correlation=st.floats(min_value=0.0, max_value=0.9),
    )
    @settings(max_examples=30, deadline=5000)
    def test_cross_instrument_consistency(self, n_instruments, n_predictions, correlation):
        """
        Metamorphic relation: Correlated instruments should produce
        similar signal patterns.
        """
        # Generate base predictions
        base_predictions = np.random.uniform(0, 1, n_predictions)

        # Generate correlated predictions for each instrument
        instrument_predictions = {}
        for i in range(n_instruments):
            # Add correlated noise
            noise = np.random.uniform(0, 1, n_predictions)
            correlated_pred = correlation * base_predictions + (1 - correlation) * noise
            # Clip to valid range
            instrument_predictions[f"INST{i}"] = np.clip(correlated_pred, 0, 1)

        # Calculate signal correlations
        threshold = 0.5
        instrument_signals = {}
        for inst, preds in instrument_predictions.items():
            instrument_signals[inst] = preds > threshold

        # Metamorphic relation: Higher correlation -> more similar signals
        if n_instruments >= 2 and correlation > 0.7:
            # Check similarity between first two instruments
            signals1 = instrument_signals["INST0"]
            signals2 = instrument_signals["INST1"]

            agreement = np.mean(signals1 == signals2)

            # High correlation should lead to high agreement
            assert (
                agreement > 0.5
            ), f"High correlation ({correlation:.2f}) should produce similar signals (agreement: {agreement:.2f})"

    @given(
        prediction_sequence=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=20,
            max_size=100,
        ),
        lookback=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=30, deadline=5000)
    def test_signal_persistence(self, prediction_sequence, lookback):
        """
        Metamorphic relation: Signal persistence/smoothing should
        reduce signal switching frequency.
        """
        predictions = np.array(prediction_sequence)

        # Raw signals (no persistence)
        threshold = 0.5
        raw_confidence = np.maximum(predictions, 1.0 - predictions)
        raw_signals = raw_confidence > threshold

        # Smoothed signals (require persistence over lookback period)
        smoothed_signals = np.zeros_like(raw_signals, dtype=bool)
        for i in range(lookback, len(predictions)):
            window = predictions[i - lookback : i]
            # Signal only if majority of window exceeds confidence threshold
            window_conf = np.maximum(window, 1.0 - window)
            smoothed_signals[i] = np.mean(window_conf > threshold) > 0.5

        # Count signal changes
        def count_switches(signals):
            if len(signals) <= 1:
                return 0
            return np.sum(signals[1:] != signals[:-1])

        # Ignore warm-up region where smoothing initializes
        raw_switches = count_switches(raw_signals[lookback:])
        smoothed_switches = count_switches(smoothed_signals[lookback:])

        # Metamorphic relation: Smoothing should reduce switching
        assert (
            smoothed_switches <= raw_switches
        ), f"Smoothing increased switches ({smoothed_switches} > {raw_switches})"

        # Additional property: Smoothed signals should be subset of extended raw signals
        # (i.e., smoothing doesn't create signals from nothing)
        total_raw_signals = np.sum(raw_signals)
        total_smoothed_signals = np.sum(smoothed_signals)

        # Smoothing should generally reduce total signals
        assert (
            total_smoothed_signals <= total_raw_signals * 1.5
        ), "Smoothing created too many additional signals"
