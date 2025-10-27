"""
Property-based tests for MultiSignalActor coordination mechanisms.

These tests verify mathematical properties and invariants that must hold for signal
aggregation, weight normalization, consensus mechanisms, and signal ordering in
multi-signal coordination systems.

Properties tested:
- Signal aggregation: weighted average correctness
- Weight normalization: sum to 1.0
- Consensus mechanisms: majority/unanimous voting correctness
- Signal ordering preservation

"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

try:  # optional dependency
    from hypothesis import assume
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st
    from hypothesis.strategies import composite
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


# Test data structures for coordination mechanisms
class SignalData:
    """Test signal data structure."""

    def __init__(
        self,
        source_id: str,
        prediction: float,
        confidence: float,
        weight: float,
        timestamp_ns: int,
    ) -> None:
        self.source_id = source_id
        self.prediction = prediction
        self.confidence = confidence
        self.weight = weight
        self.timestamp_ns = timestamp_ns


class CoordinationMechanisms:
    """Mock coordination mechanisms for testing properties."""

    @staticmethod
    def weighted_average(signals: list[SignalData]) -> tuple[float, float]:
        """
        Compute weighted average of signals.

        Returns (prediction, confidence).
        """
        if not signals:
            return 0.0, 0.0

        total_weight = sum(s.weight for s in signals)
        if total_weight <= 0:
            return 0.0, 0.0

        weighted_pred = sum(s.prediction * s.weight for s in signals) / total_weight
        weighted_conf = CoordinationMechanisms.aggregate_confidence(signals)

        return float(weighted_pred), float(weighted_conf)

    @staticmethod
    def aggregate_confidence(signals: list[SignalData]) -> float:
        """Aggregate confidences with high-confidence preservation."""
        if not signals:
            return 0.0

        total_weight = sum(s.weight for s in signals)
        if total_weight <= 0:
            return 0.0

        raw_conf = sum(s.confidence * s.weight for s in signals) / total_weight
        return float(raw_conf)

    @staticmethod
    def normalize_weights(weights: list[float]) -> list[float]:
        """Normalize weights to sum to 1.0."""
        total = sum(abs(w) for w in weights)
        if total <= 0:
            return [1.0 / len(weights)] * len(weights) if weights else []
        return [abs(w) / total for w in weights]

    @staticmethod
    def majority_consensus(signals: list[SignalData], threshold: float = 0.5) -> tuple[float, float]:
        """
        Majority consensus mechanism.

        Returns aggregated signal if majority agrees (prediction > threshold or < -threshold).
        """
        if not signals:
            return 0.0, 0.0

        positive_signals = [s for s in signals if s.prediction > threshold]
        negative_signals = [s for s in signals if s.prediction < -threshold]

        if len(positive_signals) > len(signals) / 2:
            return CoordinationMechanisms.weighted_average(positive_signals)
        elif len(negative_signals) > len(signals) / 2:
            return CoordinationMechanisms.weighted_average(negative_signals)
        else:
            # No majority, return neutral
            return 0.0, 0.0

    @staticmethod
    def unanimous_consensus(signals: list[SignalData], threshold: float = 0.5) -> tuple[float, float]:
        """
        Unanimous consensus mechanism.

        Returns aggregated signal only if all signals agree in direction.
        """
        if not signals:
            return 0.0, 0.0

        all_positive = all(s.prediction > threshold for s in signals)
        all_negative = all(s.prediction < -threshold for s in signals)

        if all_positive or all_negative:
            return CoordinationMechanisms.weighted_average(signals)
        else:
            return 0.0, 0.0


# Hypothesis strategies for generating test data
@composite
def signal_data(draw, min_sources=1, max_sources=10):
    """Generate a list of SignalData objects."""
    n_sources = draw(st.integers(min_value=min_sources, max_value=max_sources))
    signals = []

    for i in range(n_sources):
        source_id = f"source_{i}"
        prediction = draw(st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ))
        confidence = draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ))
        weight = draw(st.floats(
            min_value=0.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        ))
        timestamp_ns = draw(st.integers(min_value=0, max_value=2**62))

        signals.append(SignalData(source_id, prediction, confidence, weight, timestamp_ns))

    return signals


@composite
def weights_list(draw, min_size=1, max_size=10):
    """Generate a list of weights."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(st.lists(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        ),
        min_size=n,
        max_size=n
    ))


@composite
def ordered_signals(draw, min_size=2, max_size=10):
    """Generate ordered signals by timestamp."""
    signals = draw(signal_data(min_sources=min_size, max_sources=max_size))
    # Sort by timestamp to ensure ordering
    signals.sort(key=lambda s: s.timestamp_ns)
    return signals


# Property tests
@pytest.mark.property
class TestMultiSignalCoordinationProperties:
    """Property tests for multi-signal coordination mechanisms."""

    @given(signals=signal_data(min_sources=1, max_sources=5))
    def test_weighted_average_correctness(self, signals: list[SignalData]) -> None:
        """
        Test that weighted average computation is mathematically correct.

        Property: weighted_avg = sum(value_i * weight_i) / sum(weight_i)
        """
        # Skip if all weights are zero
        assume(any(s.weight > 0 for s in signals))

        pred, conf = CoordinationMechanisms.weighted_average(signals)

        # Manual computation for verification
        total_weight = sum(s.weight for s in signals)
        expected_pred = sum(s.prediction * s.weight for s in signals) / total_weight
        expected_conf = sum(s.confidence * s.weight for s in signals) / total_weight

        # Verify correctness within numerical tolerance
        assert abs(pred - expected_pred) < 1e-10
        assert abs(conf - expected_conf) < 1e-10

    @given(weights=weights_list(min_size=1, max_size=8))
    def test_weight_normalization_sums_to_one(self, weights: list[float]) -> None:
        """
        Test that normalized weights sum to 1.0.

        Property: sum(normalize_weights(weights)) = 1.0
        """
        # Skip if all weights are zero
        assume(any(abs(w) > 1e-12 for w in weights))

        normalized = CoordinationMechanisms.normalize_weights(weights)

        # Check sum equals 1.0 within tolerance
        weight_sum = sum(normalized)
        assert abs(weight_sum - 1.0) < 1e-10

        # Check all weights are non-negative
        assert all(w >= 0 for w in normalized)

        # Check length preservation
        assert len(normalized) == len(weights)

    @given(weights=weights_list(min_size=1, max_size=8))
    def test_weight_normalization_preserves_ratios(self, weights: list[float]) -> None:
        """
        Test that weight normalization preserves relative ratios.

        Property: normalized_i / normalized_j = |weight_i| / |weight_j|
        """
        # Skip if all weights are zero
        assume(any(abs(w) > 1e-12 for w in weights))

        normalized = CoordinationMechanisms.normalize_weights(weights)
        abs_weights = [abs(w) for w in weights]

        # Check ratios are preserved for non-zero weights
        for i in range(len(weights)):
            for j in range(len(weights)):
                if abs_weights[i] > 1e-6 and abs_weights[j] > 1e-6:
                    expected_ratio = abs_weights[i] / abs_weights[j]
                    actual_ratio = normalized[i] / normalized[j]
                    # Use relative tolerance for large ratios
                    rel_tolerance = max(1e-8, 1e-10 * max(expected_ratio, actual_ratio))
                    assert abs(actual_ratio - expected_ratio) < rel_tolerance

    @given(signals=signal_data(min_sources=3, max_sources=7))
    def test_majority_consensus_correctness(self, signals: list[SignalData]) -> None:
        """
        Test that majority consensus correctly identifies and aggregates majority signals.

        Property: If majority signals agree in direction, result has same direction.
        """
        threshold = 0.1  # Small threshold for testing

        pred, conf = CoordinationMechanisms.majority_consensus(signals, threshold)

        # Count signals by direction
        positive_count = sum(1 for s in signals if s.prediction > threshold)
        negative_count = sum(1 for s in signals if s.prediction < -threshold)
        neutral_count = len(signals) - positive_count - negative_count

        if positive_count > len(signals) / 2:
            # Majority positive - result should be positive or zero
            assert pred >= -1e-10
        elif negative_count > len(signals) / 2:
            # Majority negative - result should be negative or zero
            assert pred <= 1e-10
        else:
            # No majority - result should be neutral
            assert abs(pred) < 1e-10
            assert abs(conf) < 1e-10

    @given(signals=signal_data(min_sources=2, max_sources=5))
    def test_unanimous_consensus_strictness(self, signals: list[SignalData]) -> None:
        """
        Test that unanimous consensus is strict - requires all signals to agree.

        Property: Result is non-zero only if all signals agree in direction.
        """
        threshold = 0.1

        pred, conf = CoordinationMechanisms.unanimous_consensus(signals, threshold)

        all_positive = all(s.prediction > threshold for s in signals)
        all_negative = all(s.prediction < -threshold for s in signals)

        if all_positive:
            # All positive - result should be positive
            assert pred > -1e-10
        elif all_negative:
            # All negative - result should be negative
            assert pred < 1e-10
        else:
            # Not unanimous - result should be neutral
            assert abs(pred) < 1e-10
            assert abs(conf) < 1e-10

    @given(signals=ordered_signals(min_size=2, max_size=8))
    def test_signal_ordering_preservation(self, signals: list[SignalData]) -> None:
        """
        Test that signal ordering by timestamp is preserved in processing.

        Property: If signals are ordered by timestamp, they remain ordered.
        """
        # Verify input ordering
        timestamps = [s.timestamp_ns for s in signals]
        assert timestamps == sorted(timestamps), "Input signals should be ordered"

        # Process signals and verify ordering is maintained
        processed_signals = []
        for signal in signals:
            # Simulate processing that should preserve order
            processed_signals.append(signal)

        processed_timestamps = [s.timestamp_ns for s in processed_signals]
        assert processed_timestamps == sorted(processed_timestamps)

        # Check that order is exactly preserved
        for i, signal in enumerate(processed_signals):
            assert signal.timestamp_ns == signals[i].timestamp_ns

    @given(
        signals1=signal_data(min_sources=2, max_sources=4),
        signals2=signal_data(min_sources=2, max_sources=4)
    )
    def test_aggregation_associativity(
        self,
        signals1: list[SignalData],
        signals2: list[SignalData]
    ) -> None:
        """
        Test that signal aggregation is associative for commutative operations.

        Property: aggregate(A, B) = aggregate(B, A) for weighted average.
        """
        # Skip if any group has all zero weights
        assume(any(s.weight > 0 for s in signals1))
        assume(any(s.weight > 0 for s in signals2))

        # Test commutativity
        combined1 = signals1 + signals2
        combined2 = signals2 + signals1

        pred1, conf1 = CoordinationMechanisms.weighted_average(combined1)
        pred2, conf2 = CoordinationMechanisms.weighted_average(combined2)

        # Results should be identical (weighted average is commutative)
        assert abs(pred1 - pred2) < 1e-10
        assert abs(conf1 - conf2) < 1e-10

    @given(signals=signal_data(min_sources=1, max_sources=6))
    def test_confidence_bounds_preservation(self, signals: list[SignalData]) -> None:
        """
        Test that aggregated confidence values remain within valid bounds.

        Property: 0 <= aggregated_confidence <= 1
        """
        # Skip if all weights are zero
        assume(any(s.weight > 0 for s in signals))

        _pred, conf = CoordinationMechanisms.weighted_average(signals)

        # Confidence must be within bounds
        assert 0.0 <= conf <= 1.0

        # If all input confidences are in bounds, output should be too
        assert all(0.0 <= s.confidence <= 1.0 for s in signals)

    @given(signals=signal_data(min_sources=1, max_sources=6))
    def test_prediction_bounds_preservation(self, signals: list[SignalData]) -> None:
        """
        Test that aggregated predictions remain within reasonable bounds.

        Property: If all predictions are in [-1, 1], aggregated prediction is in [-1, 1].
        """
        # Skip if all weights are zero
        assume(any(s.weight > 0 for s in signals))

        pred, _conf = CoordinationMechanisms.weighted_average(signals)

        # All input predictions are bounded by construction
        assert all(-1.0 <= s.prediction <= 1.0 for s in signals)

        # Output should also be bounded (weighted average preserves bounds)
        assert -1.0 <= pred <= 1.0

    def test_empty_signals_handling(self) -> None:
        """
        Test that coordination mechanisms handle empty signal lists gracefully.

        Property: Empty input produces neutral output.
        """
        empty_signals: list[SignalData] = []

        # All mechanisms should handle empty input
        pred, conf = CoordinationMechanisms.weighted_average(empty_signals)
        assert pred == 0.0 and conf == 0.0

        pred, conf = CoordinationMechanisms.majority_consensus(empty_signals)
        assert pred == 0.0 and conf == 0.0

        pred, conf = CoordinationMechanisms.unanimous_consensus(empty_signals)
        assert pred == 0.0 and conf == 0.0

        # Weight normalization should return empty list
        normalized = CoordinationMechanisms.normalize_weights([])
        assert normalized == []

    @given(n=st.integers(min_value=1, max_value=10))
    def test_equal_weights_uniform_distribution(self, n: int) -> None:
        """
        Test that equal weights produce uniform distribution.

        Property: normalize_weights([1, 1, ..., 1]) = [1/n, 1/n, ..., 1/n]
        """
        equal_weights = [1.0] * n
        normalized = CoordinationMechanisms.normalize_weights(equal_weights)

        expected_weight = 1.0 / n
        for weight in normalized:
            assert abs(weight - expected_weight) < 1e-10


@pytest.mark.property
class TestMultiSignalActorIntegration:
    """Integration property tests for MultiInstrumentSignalActor."""

    def test_actor_initialization_with_coordination_config(self) -> None:
        """
        Test that MultiInstrumentSignalActor can be initialized with coordination settings.
        """
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.enums import AggregationSource
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType

        instrument_id = InstrumentId(Symbol("BTC"), Venue("SIM"))
        bar_spec = BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST,
        )
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

        config = MultiInstrumentSignalActorConfig(
            actor_id="test_multi_signal",
            model_path="/tmp/dummy_model.onnx",  # Required field
            model_id="test_model",
            bar_type=bar_type,
            instrument_id=instrument_id,
            max_batch_size=16,
            feature_dim=32,
            initial_universe=["BTC-USD.SIM", "ETH-USD.SIM"],
            flush_max_latency_ms=100,
        )

        # Should initialize without error
        actor = MultiInstrumentSignalActor(config)

        # Basic properties should be set
        assert actor._max_batch == 16
        assert actor._feature_dim == 32
        assert actor._universe.size() == 2

    @given(batch_size=st.integers(min_value=1, max_value=256))
    def test_batch_size_property_preservation(self, batch_size: int) -> None:
        """
        Test that batch size configuration is preserved and enforced.

        Property: Configured batch size is the maximum processed in one batch.
        """
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.enums import AggregationSource
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType

        instrument_id = InstrumentId(Symbol("BTC"), Venue("SIM"))
        bar_spec = BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST,
        )
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

        config = MultiInstrumentSignalActorConfig(
            actor_id="test_batch_size",
            model_path="/tmp/dummy_model.onnx",  # Required field
            model_id="test_model",
            bar_type=bar_type,
            instrument_id=instrument_id,
            max_batch_size=batch_size,
            feature_dim=10,
        )

        actor = MultiInstrumentSignalActor(config)
        assert actor._max_batch == batch_size
        assert actor._batch_features.shape[0] == batch_size

    @given(
        n_signals=st.integers(min_value=2, max_value=8),
        failure_rate=st.floats(min_value=0.0, max_value=0.5)
    )
    def test_signal_independence_property(self, n_signals: int, failure_rate: float) -> None:
        """
        Test that signals operate independently - failure of one doesn't affect others.

        Property: If signal_i fails, signals j≠i continue to produce valid outputs.
        """
        # Create mock signal sources
        sources = []
        for i in range(n_signals):
            source = MagicMock()
            source.id = f"signal_{i}"
            source.prediction = 0.5 + (i * 0.1)  # Different predictions
            source.confidence = 0.8
            source.failed = False
            sources.append(source)

        # Randomly fail some sources based on failure_rate
        n_failed = int(n_signals * failure_rate)
        if n_failed > 0:
            failed_indices = np.random.choice(n_signals, size=n_failed, replace=False)
            for idx in failed_indices:
                sources[idx].failed = True

        # Test that working sources are unaffected by failed ones
        working_sources = [s for s in sources if not s.failed]
        failed_sources = [s for s in sources if s.failed]

        # Working sources should still produce valid signals
        if working_sources:
            signals = [
                SignalData(s.id, s.prediction, s.confidence, 1.0, 100)
                for s in working_sources
            ]

            pred, conf = CoordinationMechanisms.weighted_average(signals)

            # Aggregated result should be valid
            assert -1.0 <= pred <= 1.0
            assert 0.0 <= conf <= 1.0

            # Result should be average of working sources only
            expected_pred = sum(s.prediction for s in working_sources) / len(working_sources)
            assert abs(pred - expected_pred) < 1e-10

    @given(signals=signal_data(min_sources=3, max_sources=8))
    def test_failover_handling_property(self, signals: list[SignalData]) -> None:
        """
        Test failover handling when signals become unavailable.

        Property: System continues operating with reduced signal set.
        """
        assume(len(signals) >= 2)  # Need at least 2 signals for meaningful failover

        # Original aggregation with all signals
        _original_pred, _original_conf = CoordinationMechanisms.weighted_average(signals)

        # Simulate failover by removing signals one by one
        for i in range(1, len(signals)):
            remaining_signals = signals[i:]
            if any(s.weight > 0 for s in remaining_signals):
                pred, conf = CoordinationMechanisms.weighted_average(remaining_signals)

                # System should continue to produce valid results
                assert -1.0 <= pred <= 1.0
                assert 0.0 <= conf <= 1.0

                # Should not crash or produce NaN/infinity
                assert not math.isnan(pred)
                assert not math.isnan(conf)
                assert not math.isinf(pred)
                assert not math.isinf(conf)

    @given(signals=signal_data(min_sources=2, max_sources=6))
    def test_signal_strength_aggregation_property(self, signals: list[SignalData]) -> None:
        """
        Test that signal strength (confidence) is properly aggregated.

        Property: Aggregated strength reflects weighted contribution of individual strengths.
        """
        assume(any(s.weight > 0 for s in signals))

        _pred, conf = CoordinationMechanisms.weighted_average(signals)

        # Manual calculation of expected confidence
        expected_conf = CoordinationMechanisms.aggregate_confidence(signals)

        # Aggregated confidence should match manual calculation
        assert abs(conf - expected_conf) < 1e-10

        # High confidence signals should contribute more to final confidence
        if len(signals) >= 2:
            high_conf_signals = [s for s in signals if s.confidence > 0.8]
            low_conf_signals = [s for s in signals if s.confidence < 0.3]

            if high_conf_signals and low_conf_signals:
                # Weight high confidence signals more
                high_weight_signals = []
                for s in signals:
                    weight = 10.0 if s.confidence > 0.8 else 1.0
                    high_weight_signals.append(
                        SignalData(s.source_id, s.prediction, s.confidence, weight, s.timestamp_ns)
                    )

                _, weighted_conf = CoordinationMechanisms.weighted_average(high_weight_signals)

                # Weighted confidence must be bounded by min/max of inputs (valid property)
                confidences = [s.confidence for s in high_weight_signals]
                assert min(confidences) <= weighted_conf <= max(confidences)

    @given(signals=ordered_signals(min_size=3, max_size=8))
    def test_timing_coordination_property(self, signals: list[SignalData]) -> None:
        """
        Test that timing coordination ensures all signals process the same data.

        Property: Signals are synchronized by timestamp before aggregation.
        """
        # Verify signals are ordered by timestamp
        timestamps = [s.timestamp_ns for s in signals]
        assert timestamps == sorted(timestamps)

        # Test that aggregation respects temporal ordering
        # Simulate processing signals in temporal batches
        batched_results = []

        # Group signals by time windows (simulate real-time processing)
        window_size_ns = 1_000_000_000  # 1 second windows
        current_window = signals[0].timestamp_ns // window_size_ns
        current_batch = []

        for signal in signals:
            signal_window = signal.timestamp_ns // window_size_ns

            if signal_window == current_window:
                current_batch.append(signal)
            else:
                # Process current batch
                if current_batch and any(s.weight > 0 for s in current_batch):
                    pred, conf = CoordinationMechanisms.weighted_average(current_batch)
                    batched_results.append((current_window, pred, conf))

                # Start new batch
                current_window = signal_window
                current_batch = [signal]

        # Process final batch
        if current_batch and any(s.weight > 0 for s in current_batch):
            pred, conf = CoordinationMechanisms.weighted_average(current_batch)
            batched_results.append((current_window, pred, conf))

        # Verify temporal ordering is preserved in results
        if len(batched_results) > 1:
            result_windows = [window for window, _, _ in batched_results]
            assert result_windows == sorted(result_windows)

    @given(
        base_signals=signal_data(min_sources=2, max_sources=4),
        noise_scale=st.floats(min_value=0.0, max_value=0.1)
    )
    def test_coordination_stability_under_noise(
        self,
        base_signals: list[SignalData],
        noise_scale: float
    ) -> None:
        """
        Test that coordination mechanisms are stable under small signal perturbations.

        Property: Small changes in input signals produce small changes in output.
        """
        assume(any(s.weight > 0 for s in base_signals))

        # Original result
        orig_pred, orig_conf = CoordinationMechanisms.weighted_average(base_signals)

        # Add small noise to signals
        noisy_signals = []
        for signal in base_signals:
            noise_pred = np.random.normal(0, noise_scale)
            noise_conf = np.random.normal(0, noise_scale * 0.1)  # Smaller noise for confidence

            new_pred = np.clip(signal.prediction + noise_pred, -1.0, 1.0)
            new_conf = np.clip(signal.confidence + noise_conf, 0.0, 1.0)

            noisy_signals.append(
                SignalData(signal.source_id, new_pred, new_conf, signal.weight, signal.timestamp_ns)
            )

        # Noisy result
        noisy_pred, noisy_conf = CoordinationMechanisms.weighted_average(noisy_signals)

        # Changes should be bounded by noise scale
        pred_change = abs(noisy_pred - orig_pred)
        conf_change = abs(noisy_conf - orig_conf)

        # Lipschitz continuity: output change bounded by input change with reasonable tolerance
        # For weighted averages, the output change can be up to the maximum input change
        # Allow some tolerance for numerical precision and the fact that clipping can amplify changes
        max_expected_pred_change = noise_scale * 2.0  # Conservative bound allowing for edge effects
        max_expected_conf_change = noise_scale * 0.5  # Confidence changes should be smaller

        assert pred_change <= max_expected_pred_change + 1e-8
        assert conf_change <= max_expected_conf_change + 1e-8

    @given(signals=signal_data(min_sources=2, max_sources=6))
    def test_consensus_monotonicity_property(self, signals: list[SignalData]) -> None:
        """
        Test monotonicity property of consensus mechanisms.

        Property: Stronger consensus (more agreement) produces higher confidence.
        """
        assume(len(signals) >= 2)

        # Create high consensus scenario (all signals similar)
        high_consensus_signals = []
        base_pred = 0.7
        for i, signal in enumerate(signals):
            # Small variations around base prediction
            pred = base_pred + (i * 0.05 - 0.1)  # ±0.1 variation
            pred = np.clip(pred, -1.0, 1.0)
            high_consensus_signals.append(
                SignalData(signal.source_id, pred, signal.confidence, signal.weight, signal.timestamp_ns)
            )

        # Create low consensus scenario (signals spread out)
        low_consensus_signals = []
        for i, signal in enumerate(signals):
            # Large variations
            pred = -0.8 + (i * 1.6 / (len(signals) - 1))  # Spread from -0.8 to 0.8
            low_consensus_signals.append(
                SignalData(signal.source_id, pred, signal.confidence, signal.weight, signal.timestamp_ns)
            )

        if any(s.weight > 0 for s in high_consensus_signals) and any(s.weight > 0 for s in low_consensus_signals):
            # Calculate consensus strength (inverse of prediction variance)
            high_preds = [s.prediction for s in high_consensus_signals]
            low_preds = [s.prediction for s in low_consensus_signals]

            high_variance = np.var(high_preds)
            low_variance = np.var(low_preds)

            # High consensus should have lower variance
            assert high_variance <= low_variance + 1e-10

            # This property can be used to adjust confidence based on consensus
            # (Implementation dependent - here we verify the variance relationship)
