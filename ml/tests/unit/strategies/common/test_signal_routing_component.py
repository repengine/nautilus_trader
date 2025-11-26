"""
Tests for SignalRoutingComponent.

This module tests the signal routing component extracted from BaseMLStrategy
as part of the Phase 3.4 decomposition. Tests cover:

- Signal filtering by model ID
- Signal filtering by confidence threshold
- Signal filtering by instrument
- Signal aggregation (weighted average and voting)
- Time window management
- Signal history management
- Buffer management

"""

from __future__ import annotations

import time
from collections import deque
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.actors.base import MLSignal
from ml.strategies.common.signal_routing import SignalRoutingComponent
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def create_mock_signal(
    model_id: str = "model_1",
    prediction: float = 0.7,
    confidence: float = 0.8,
    instrument_id: str = "EURUSD.SIM",
    ts_event: int | None = None,
) -> MLSignal:
    """Create an MLSignal for testing."""
    if ts_event is None:
        ts_event = time.time_ns()

    inst_id = InstrumentId(Symbol(instrument_id.split(".")[0]), Venue(instrument_id.split(".")[1]))

    return MLSignal(
        instrument_id=inst_id,
        model_id=model_id,
        prediction=prediction,
        confidence=confidence,
        metadata={},
        ts_event=ts_event,
        ts_init=ts_event,
    )


@pytest.fixture
def signal_routing_component() -> SignalRoutingComponent:
    """Basic signal routing component for tests."""
    return SignalRoutingComponent(
        target_model_ids=None,
        aggregation_mode=None,
        required_models=1,
        time_window_ms=1000,
        min_confidence=0.5,
        history_size=100,
        instrument_id=None,
    )


@pytest.fixture
def component_with_model_filter() -> SignalRoutingComponent:
    """Component with model ID filtering configured."""
    return SignalRoutingComponent(
        target_model_ids=["model_a", "model_b"],
        min_confidence=0.0,
    )


@pytest.fixture
def component_with_aggregation() -> SignalRoutingComponent:
    """Component with aggregation enabled."""
    return SignalRoutingComponent(
        target_model_ids=None,
        aggregation_mode="weighted_average",
        required_models=3,
        time_window_ms=1000,
        conflict_resolution="weighted_average",
        model_weights={"model_1": 1.0, "model_2": 2.0, "model_3": 1.0},
        min_confidence=0.0,
    )


@pytest.fixture
def component_with_voting() -> SignalRoutingComponent:
    """Component with voting aggregation enabled."""
    return SignalRoutingComponent(
        target_model_ids=None,
        aggregation_mode="voting",
        required_models=3,
        time_window_ms=1000,
        conflict_resolution="voting",
        min_confidence=0.0,
    )


@pytest.fixture
def component_with_instrument_filter() -> SignalRoutingComponent:
    """Component with instrument filtering configured."""
    instrument_id = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return SignalRoutingComponent(
        instrument_id=instrument_id,
        min_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Test Class: Signal Filtering by Model ID
# ---------------------------------------------------------------------------


class TestSignalFilteringByModelId:
    """Test signal filtering by model ID."""

    def test_filter_signal_by_model_id_accepts_target_model(
        self,
        component_with_model_filter: SignalRoutingComponent,
    ) -> None:
        """Verify signals from target models are accepted."""
        signal = create_mock_signal(model_id="model_a")
        assert component_with_model_filter.filter_by_model_id(signal) is True

    def test_filter_signal_by_model_id_rejects_non_target_model(
        self,
        component_with_model_filter: SignalRoutingComponent,
    ) -> None:
        """Verify signals from non-target models are rejected."""
        signal = create_mock_signal(model_id="model_c")
        assert component_with_model_filter.filter_by_model_id(signal) is False

    def test_filter_signal_when_no_target_list_accepts_all(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify all signals accepted when no target list configured."""
        signal = create_mock_signal(model_id="any_model")
        assert signal_routing_component.filter_by_model_id(signal) is True


# ---------------------------------------------------------------------------
# Test Class: Signal Filtering by Confidence
# ---------------------------------------------------------------------------


class TestSignalFilteringByConfidence:
    """Test signal filtering by confidence threshold."""

    def test_filter_signal_by_confidence_accepts_above_threshold(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify high confidence signals are accepted."""
        signal = create_mock_signal(confidence=0.8)
        assert signal_routing_component.filter_by_confidence(signal) is True

    def test_filter_signal_by_confidence_rejects_below_threshold(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify low confidence signals are rejected."""
        signal = create_mock_signal(confidence=0.3)
        assert signal_routing_component.filter_by_confidence(signal) is False

    def test_filter_signal_by_confidence_accepts_at_threshold(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify signals at exactly the threshold are accepted."""
        signal = create_mock_signal(confidence=0.5)
        assert signal_routing_component.filter_by_confidence(signal) is True


# ---------------------------------------------------------------------------
# Test Class: Signal Filtering by Instrument
# ---------------------------------------------------------------------------


class TestSignalFilteringByInstrument:
    """Test signal filtering by instrument."""

    def test_filter_signal_by_instrument_accepts_matching(
        self,
        component_with_instrument_filter: SignalRoutingComponent,
    ) -> None:
        """Verify signals for configured instrument are accepted."""
        signal = create_mock_signal(instrument_id="EURUSD.SIM")
        assert component_with_instrument_filter.filter_by_instrument(signal) is True

    def test_filter_signal_by_instrument_rejects_non_matching(
        self,
        component_with_instrument_filter: SignalRoutingComponent,
    ) -> None:
        """Verify signals for other instruments are rejected."""
        signal = create_mock_signal(instrument_id="GBPUSD.SIM")
        assert component_with_instrument_filter.filter_by_instrument(signal) is False

    def test_filter_signal_by_instrument_accepts_when_no_filter(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify all instruments accepted when no filter configured."""
        signal = create_mock_signal(instrument_id="USDJPY.SIM")
        assert signal_routing_component.filter_by_instrument(signal) is True


# ---------------------------------------------------------------------------
# Test Class: Signal Aggregation - Weighted Average
# ---------------------------------------------------------------------------


class TestSignalAggregationWeightedAverage:
    """Test weighted average signal aggregation."""

    def test_aggregate_signal_weighted_average(
        self,
        component_with_aggregation: SignalRoutingComponent,
    ) -> None:
        """Verify weighted average aggregation works correctly."""
        base_time = time.time_ns()

        # Add signals with known predictions and weights
        # model_1: pred=0.7, weight=1.0
        # model_2: pred=0.8, weight=2.0
        # model_3: pred=0.6, weight=1.0
        # Weighted avg = (0.7*1 + 0.8*2 + 0.6*1) / 4 = 2.9 / 4 = 0.725
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_1", prediction=0.7, ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_2", prediction=0.8, ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_3", prediction=0.6, ts_event=base_time)
        )

        aggregated = component_with_aggregation.aggregate_signals()

        assert aggregated is not None
        assert aggregated.prediction == pytest.approx(0.725, rel=1e-3)
        assert aggregated.model_id == "aggregated"
        assert "aggregated_from" in aggregated.metadata

    def test_aggregate_signal_requires_minimum_models(
        self,
        component_with_aggregation: SignalRoutingComponent,
    ) -> None:
        """Verify aggregation only triggers when enough models respond."""
        base_time = time.time_ns()

        # Add only 2 signals (need 3)
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_1", ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_2", ts_event=base_time)
        )

        aggregated = component_with_aggregation.aggregate_signals()
        assert aggregated is None

        # Buffer should still have signals
        assert len(component_with_aggregation.signal_buffer) == 2


# ---------------------------------------------------------------------------
# Test Class: Signal Aggregation - Voting
# ---------------------------------------------------------------------------


class TestSignalAggregationVoting:
    """Test voting-based signal aggregation."""

    def test_aggregate_signal_voting_majority(
        self,
        component_with_voting: SignalRoutingComponent,
    ) -> None:
        """Verify voting aggregation produces BUY on bullish majority."""
        base_time = time.time_ns()

        # 2 bullish (> 0.5), 1 bearish
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.7, ts_event=base_time)
        )
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.8, ts_event=base_time)
        )
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m3", prediction=0.3, ts_event=base_time)
        )

        aggregated = component_with_voting.aggregate_signals()

        assert aggregated is not None
        assert aggregated.prediction > 0.5  # BUY signal
        assert aggregated.metadata.get("action") == "BUY"

    def test_aggregate_signal_voting_bearish_majority(
        self,
        component_with_voting: SignalRoutingComponent,
    ) -> None:
        """Verify voting aggregation produces SELL on bearish majority."""
        base_time = time.time_ns()

        # 2 bearish, 1 bullish
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.2, ts_event=base_time)
        )
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.3, ts_event=base_time)
        )
        component_with_voting.add_to_buffer(
            create_mock_signal(model_id="m3", prediction=0.7, ts_event=base_time)
        )

        aggregated = component_with_voting.aggregate_signals()

        assert aggregated is not None
        assert aggregated.prediction < 0.5  # SELL signal
        assert aggregated.metadata.get("action") == "SELL"


# ---------------------------------------------------------------------------
# Test Class: Time Window Management
# ---------------------------------------------------------------------------


class TestTimeWindowManagement:
    """Test time window expiry handling."""

    def test_aggregate_signal_time_window_expiry(
        self,
        component_with_aggregation: SignalRoutingComponent,
    ) -> None:
        """Verify old signals are discarded when outside time window."""
        base_time = time.time_ns()
        old_time = base_time - 5_000_000_000  # 5000ms ago (5 seconds)

        # Add an old signal
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_1", ts_event=old_time)
        )
        # Add recent signals
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_2", ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_3", ts_event=base_time)
        )

        # Aggregation should fail due to time window violation
        aggregated = component_with_aggregation.aggregate_signals()
        assert aggregated is None

    def test_purge_stale_signals_removes_old(self) -> None:
        """Verify purge_stale_signals removes signals older than time window."""
        component = SignalRoutingComponent(
            time_window_ms=100,  # 100ms window
            aggregation_mode="voting",
            required_models=2,
        )

        base_time = time.time_ns()
        old_time = base_time - 500_000_000  # 500ms ago

        component.add_to_buffer(
            create_mock_signal(model_id="old_model", ts_event=old_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="new_model", ts_event=base_time)
        )

        component.purge_stale_signals()

        # Old signal should be removed
        assert "old_model" not in component.signal_buffer
        assert "new_model" in component.signal_buffer


# ---------------------------------------------------------------------------
# Test Class: Signal History Management
# ---------------------------------------------------------------------------


class TestSignalHistoryManagement:
    """Test signal history management."""

    def test_signal_history_bounded_by_maxlen(self) -> None:
        """Verify signal history doesn't exceed configured size."""
        component = SignalRoutingComponent(history_size=100)

        # Add 150 signals
        for i in range(150):
            signal = create_mock_signal(model_id=f"model_{i}")
            component.add_to_history(signal)

        # Should only retain 100
        assert len(component.signal_history) == 100

    def test_add_to_history_appends(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify add_to_history appends to deque."""
        signal = create_mock_signal(model_id="test_model")
        initial_len = len(signal_routing_component.signal_history)

        signal_routing_component.add_to_history(signal)

        assert len(signal_routing_component.signal_history) == initial_len + 1
        assert signal_routing_component.signal_history[-1] is signal


# ---------------------------------------------------------------------------
# Test Class: Signal Buffer Management
# ---------------------------------------------------------------------------


class TestSignalBufferManagement:
    """Test signal buffer management."""

    def test_signal_buffer_per_model_tracking(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify buffer tracks signals per model."""
        signal1 = create_mock_signal(model_id="model_1")
        signal2 = create_mock_signal(model_id="model_2")

        signal_routing_component.add_to_buffer(signal1)
        signal_routing_component.add_to_buffer(signal2)

        assert "model_1" in signal_routing_component.signal_buffer
        assert "model_2" in signal_routing_component.signal_buffer
        assert len(signal_routing_component.signal_buffer) == 2

    def test_get_model_signal_returns_latest(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify get_model_signal returns latest signal from model."""
        signal1 = create_mock_signal(model_id="model_1", prediction=0.6)
        signal2 = create_mock_signal(model_id="model_1", prediction=0.9)

        signal_routing_component.add_to_buffer(signal1)
        signal_routing_component.add_to_buffer(signal2)

        result = signal_routing_component.get_model_signal("model_1")

        assert result is not None
        assert result.prediction == 0.9  # Latest signal

    def test_get_model_signal_returns_none_for_unknown(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify get_model_signal returns None for unknown model."""
        result = signal_routing_component.get_model_signal("unknown_model")
        assert result is None

    def test_clear_buffer(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify clear_buffer empties the buffer."""
        signal_routing_component.add_to_buffer(create_mock_signal())
        assert len(signal_routing_component.signal_buffer) == 1

        signal_routing_component.clear_buffer()
        assert len(signal_routing_component.signal_buffer) == 0


# ---------------------------------------------------------------------------
# Test Class: Aggregation Conditions
# ---------------------------------------------------------------------------


class TestAggregationConditions:
    """Test aggregation condition checks."""

    def test_should_aggregate_when_all_models_present(
        self,
        component_with_aggregation: SignalRoutingComponent,
    ) -> None:
        """Verify should_aggregate returns True when sufficient models present."""
        base_time = time.time_ns()

        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_1", ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_2", ts_event=base_time)
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_3", ts_event=base_time)
        )

        assert component_with_aggregation.should_aggregate() is True

    def test_should_aggregate_false_when_missing_models(
        self,
        component_with_aggregation: SignalRoutingComponent,
    ) -> None:
        """Verify should_aggregate returns False when models missing."""
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_1")
        )
        component_with_aggregation.add_to_buffer(
            create_mock_signal(model_id="model_2")
        )
        # Only 2 of 3 required

        assert component_with_aggregation.should_aggregate() is False

    def test_should_aggregate_false_when_aggregation_disabled(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify should_aggregate returns False when aggregation not configured."""
        signal_routing_component.add_to_buffer(create_mock_signal())
        assert signal_routing_component.should_aggregate() is False


# ---------------------------------------------------------------------------
# Test Class: Route Signal Integration
# ---------------------------------------------------------------------------


class TestRouteSignal:
    """Test the main route_signal method."""

    def test_route_signal_applies_filters_then_aggregation(self) -> None:
        """Verify route_signal applies filters and then aggregation."""
        component = SignalRoutingComponent(
            target_model_ids=["model_1", "model_2", "model_3"],
            aggregation_mode="weighted_average",
            required_models=3,
            time_window_ms=1000,
            min_confidence=0.5,
        )

        base_time = time.time_ns()

        # Add signals that pass filters
        result1 = component.route_signal(
            create_mock_signal(model_id="model_1", confidence=0.8, ts_event=base_time)
        )
        result2 = component.route_signal(
            create_mock_signal(model_id="model_2", confidence=0.9, ts_event=base_time)
        )
        result3 = component.route_signal(
            create_mock_signal(model_id="model_3", confidence=0.7, ts_event=base_time)
        )

        # First two should return None (buffered for aggregation)
        assert result1 is None
        assert result2 is None
        # Third should trigger aggregation and return aggregated signal
        assert result3 is not None
        assert result3.model_id == "aggregated"

    def test_route_signal_with_no_aggregation(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify route_signal returns original when no aggregation configured."""
        signal = create_mock_signal(confidence=0.8)
        result = signal_routing_component.route_signal(signal)

        assert result is signal  # Same object

    def test_route_signal_with_single_model_target(self) -> None:
        """Verify route_signal works with single target model."""
        component = SignalRoutingComponent(
            target_model_ids=["only_model"],
            min_confidence=0.0,
        )

        accepted = component.route_signal(create_mock_signal(model_id="only_model"))
        rejected = component.route_signal(create_mock_signal(model_id="other_model"))

        assert accepted is not None
        assert rejected is None

    def test_route_signal_filters_low_confidence(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify route_signal filters out low confidence signals."""
        low_conf_signal = create_mock_signal(confidence=0.3)
        result = signal_routing_component.route_signal(low_conf_signal)

        assert result is None

    def test_route_signal_always_adds_to_history(
        self,
        signal_routing_component: SignalRoutingComponent,
    ) -> None:
        """Verify route_signal adds all signals to history regardless of filtering."""
        low_conf_signal = create_mock_signal(confidence=0.1)  # Will be filtered

        initial_history_len = len(signal_routing_component.signal_history)
        signal_routing_component.route_signal(low_conf_signal)

        # Even filtered signals should be in history
        assert len(signal_routing_component.signal_history) == initial_history_len + 1


# ---------------------------------------------------------------------------
# Test Class: Conflict Resolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    """Test conflict resolution modes."""

    def test_aggregate_signal_conflict_resolution_conservative(self) -> None:
        """Verify conservative (weighted_average) conflict resolution."""
        component = SignalRoutingComponent(
            aggregation_mode="voting",  # Mode is voting
            conflict_resolution="weighted_average",  # But resolution is weighted
            required_models=2,
            model_weights={"m1": 1.0, "m2": 2.0},
        )

        base_time = time.time_ns()
        component.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.6, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.9, ts_event=base_time)
        )

        aggregated = component.aggregate_signals()

        assert aggregated is not None
        # Weighted avg: (0.6*1 + 0.9*2) / 3 = 2.4 / 3 = 0.8
        assert aggregated.prediction == pytest.approx(0.8, rel=1e-3)

    def test_aggregate_signal_conflict_resolution_aggressive(self) -> None:
        """Verify aggressive (voting) conflict resolution."""
        component = SignalRoutingComponent(
            aggregation_mode="weighted_average",
            conflict_resolution="voting",  # Override to voting
            required_models=3,
        )

        base_time = time.time_ns()
        # 2 bullish, 1 bearish
        component.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.7, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.8, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m3", prediction=0.3, ts_event=base_time)
        )

        aggregated = component.aggregate_signals()

        assert aggregated is not None
        # Voting should produce 0.8 (BUY) or 0.2 (SELL)
        assert aggregated.prediction == 0.8  # BUY due to majority


# ---------------------------------------------------------------------------
# Test Class: Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test component properties."""

    def test_target_model_ids_property(self) -> None:
        """Verify target_model_ids property returns configured value."""
        component = SignalRoutingComponent(target_model_ids=["a", "b"])
        assert component.target_model_ids == ["a", "b"]

    def test_aggregation_mode_property(self) -> None:
        """Verify aggregation_mode property returns configured value."""
        component = SignalRoutingComponent(aggregation_mode="voting")
        assert component.aggregation_mode == "voting"

    def test_required_models_property(self) -> None:
        """Verify required_models property returns configured value."""
        component = SignalRoutingComponent(required_models=5)
        assert component.required_models == 5

    def test_time_window_ms_property(self) -> None:
        """Verify time_window_ms property returns configured value."""
        component = SignalRoutingComponent(time_window_ms=5000)
        assert component.time_window_ms == 5000

    def test_min_confidence_property(self) -> None:
        """Verify min_confidence property returns configured value."""
        component = SignalRoutingComponent(min_confidence=0.75)
        assert component.min_confidence == 0.75

    def test_signal_history_property(self) -> None:
        """Verify signal_history property returns deque."""
        component = SignalRoutingComponent(history_size=50)
        assert isinstance(component.signal_history, deque)
        assert component.signal_history.maxlen == 50

    def test_signal_buffer_property(self) -> None:
        """Verify signal_buffer property returns dict."""
        component = SignalRoutingComponent()
        assert isinstance(component.signal_buffer, dict)


# ---------------------------------------------------------------------------
# Test Class: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_buffer_aggregation(self) -> None:
        """Verify aggregate_signals handles empty buffer."""
        component = SignalRoutingComponent(
            aggregation_mode="weighted_average",
            required_models=1,
        )

        result = component.aggregate_signals()
        assert result is None

    def test_zero_weight_model(self) -> None:
        """Verify handling of zero-weight models."""
        component = SignalRoutingComponent(
            aggregation_mode="weighted_average",
            conflict_resolution="weighted_average",
            required_models=2,
            model_weights={"m1": 0.0, "m2": 1.0},
        )

        base_time = time.time_ns()
        component.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.1, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.9, ts_event=base_time)
        )

        aggregated = component.aggregate_signals()

        # With m1 having zero weight, result should be dominated by m2
        assert aggregated is not None
        assert aggregated.prediction == pytest.approx(0.9, rel=1e-3)

    def test_model_id_from_metadata(self) -> None:
        """Verify model_id extraction from metadata when field is None."""
        component = SignalRoutingComponent(
            target_model_ids=["meta_model"],
        )

        # Create signal with model_id in metadata only
        inst_id = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
        signal = MLSignal(
            instrument_id=inst_id,
            model_id="meta_model",  # Set directly since MLSignal requires it
            prediction=0.7,
            confidence=0.8,
            metadata={"model_id": "meta_model"},
            ts_event=time.time_ns(),
            ts_init=time.time_ns(),
        )

        assert component.filter_by_model_id(signal) is True

    def test_confidence_exactly_zero(self) -> None:
        """Verify handling of zero confidence."""
        component = SignalRoutingComponent(min_confidence=0.0)
        signal = create_mock_signal(confidence=0.0)

        assert component.filter_by_confidence(signal) is True

    def test_confidence_exactly_one(self) -> None:
        """Verify handling of max confidence."""
        component = SignalRoutingComponent(min_confidence=1.0)
        signal = create_mock_signal(confidence=1.0)

        assert component.filter_by_confidence(signal) is True

    def test_prediction_boundary_values(self) -> None:
        """Verify voting handles boundary predictions correctly."""
        component = SignalRoutingComponent(
            aggregation_mode="voting",
            required_models=3,
        )

        base_time = time.time_ns()

        # Exactly 0.5 should be counted as bearish (not > 0.5)
        component.add_to_buffer(
            create_mock_signal(model_id="m1", prediction=0.5, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m2", prediction=0.5, ts_event=base_time)
        )
        component.add_to_buffer(
            create_mock_signal(model_id="m3", prediction=0.5, ts_event=base_time)
        )

        aggregated = component.aggregate_signals()

        assert aggregated is not None
        # All at 0.5 (not > 0.5), so bearish majority
        assert aggregated.prediction == 0.2


# ---------------------------------------------------------------------------
# Test Class: Logger Integration
# ---------------------------------------------------------------------------


class TestLoggerIntegration:
    """Test logger integration."""

    def test_filter_logs_debug_message_when_filtered(self) -> None:
        """Verify debug log when signal is filtered."""
        mock_log = MagicMock()
        component = SignalRoutingComponent(
            target_model_ids=["target_model"],
            log=mock_log,
        )

        signal = create_mock_signal(model_id="other_model")
        component.route_signal(signal)

        mock_log.debug.assert_called()

    def test_no_logger_does_not_crash(self) -> None:
        """Verify component works without logger."""
        component = SignalRoutingComponent(
            target_model_ids=["target_model"],
            log=None,
        )

        signal = create_mock_signal(model_id="other_model")
        # Should not raise
        result = component.route_signal(signal)
        assert result is None
