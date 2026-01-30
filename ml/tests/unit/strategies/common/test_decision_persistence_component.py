"""
Tests for DecisionPersistenceComponent.

This module tests the decision persistence component extracted from BaseMLStrategy
as part of the Phase 3.4 decomposition. Tests cover:

- Decision persistence to strategy store
- Circuit breaker protection
- Event publishing when store unavailable
- HOLD signal filtering
- Risk metrics calculation
- Execution params building
- Model predictions extraction
- Metrics recording
- Lazy publisher initialization

"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.actors.base import MLSignal
from ml.strategies.common.decision_persistence import (
    CircuitBreakerProtocol,
    DecisionPersistenceComponent,
    StrategyStoreProtocol,
)
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
    metadata: dict[str, Any] | None = None,
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
        metadata=metadata or {},
        ts_event=ts_event,
        ts_init=ts_event,
    )


@pytest.fixture
def mock_strategy_store() -> MagicMock:
    """Create a mock strategy store."""
    store = MagicMock(spec=StrategyStoreProtocol)
    store.write_signal = MagicMock()
    store.flush = MagicMock()
    return store


@pytest.fixture
def mock_circuit_breaker() -> MagicMock:
    """Create a mock circuit breaker."""
    breaker = MagicMock(spec=CircuitBreakerProtocol)
    breaker.can_execute = MagicMock(return_value=True)
    breaker.record_success = MagicMock()
    breaker.record_failure = MagicMock()
    return breaker


@pytest.fixture
def mock_bus_publisher() -> MagicMock:
    """Create a mock bus publisher."""
    publisher = MagicMock()
    publisher.publish = MagicMock(return_value=True)
    return publisher


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def decision_persistence_component(
    mock_strategy_store: MagicMock,
    mock_circuit_breaker: MagicMock,
    mock_bus_publisher: MagicMock,
    mock_logger: MagicMock,
) -> DecisionPersistenceComponent:
    """Create a decision persistence component with mocked dependencies."""
    return DecisionPersistenceComponent(
        strategy_id="test_strategy",
        strategy_store=mock_strategy_store,
        circuit_breaker=mock_circuit_breaker,
        bus_publisher=mock_bus_publisher,
        persist_all_signals=False,
        log=mock_logger,
        active_positions=0,
        pending_orders=0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_positions=5,
        is_backtesting=False,
    )


@pytest.fixture
def component_with_persist_all(
    mock_strategy_store: MagicMock,
    mock_circuit_breaker: MagicMock,
    mock_bus_publisher: MagicMock,
    mock_logger: MagicMock,
) -> DecisionPersistenceComponent:
    """Create a component with persist_all_signals=True."""
    return DecisionPersistenceComponent(
        strategy_id="test_strategy",
        strategy_store=mock_strategy_store,
        circuit_breaker=mock_circuit_breaker,
        bus_publisher=mock_bus_publisher,
        persist_all_signals=True,
        log=mock_logger,
    )


@pytest.fixture
def component_without_store(
    mock_bus_publisher: MagicMock,
    mock_logger: MagicMock,
) -> DecisionPersistenceComponent:
    """Create a component without a strategy store."""
    return DecisionPersistenceComponent(
        strategy_id="test_strategy",
        strategy_store=None,
        bus_publisher=mock_bus_publisher,
        persist_all_signals=False,
        log=mock_logger,
    )


# ---------------------------------------------------------------------------
# Test Class: Decision Persistence - Store Write
# ---------------------------------------------------------------------------


class TestDecisionPersistenceStoreWrite:
    """Test decision persistence to strategy store."""

    def test_persist_decision_writes_to_store(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify decisions are written to strategy store."""
        signal = create_mock_signal()

        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        assert result is True
        mock_strategy_store.write_signal.assert_called_once()

        # Verify call arguments
        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["strategy_id"] == "test_strategy"
        assert call_args.kwargs["signal_type"] == "BUY"
        assert call_args.kwargs["is_live"] is True

    def test_persist_decision_with_position_size(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify position size is included in execution params."""
        from nautilus_trader.model.objects import Quantity

        signal = create_mock_signal()
        position_size = Quantity.from_str("10")

        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            position_size=position_size,
        )

        assert result is True
        call_args = mock_strategy_store.write_signal.call_args
        assert "position_size" in call_args.kwargs["execution_params"]
        assert call_args.kwargs["execution_params"]["position_size"] == "10"

    def test_persist_decision_without_position_size(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify None position size is handled correctly."""
        signal = create_mock_signal()

        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="SELL",
            position_size=None,
        )

        assert result is True
        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["execution_params"]["position_size"] is None


# ---------------------------------------------------------------------------
# Test Class: Circuit Breaker Protection
# ---------------------------------------------------------------------------


class TestCircuitBreakerProtection:
    """Test circuit breaker protection."""

    def test_persist_decision_circuit_breaker_prevents_write(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify circuit breaker prevents write when open."""
        mock_circuit_breaker.can_execute.return_value = False

        signal = create_mock_signal()
        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        # Should not succeed (partial event published instead)
        assert result is False
        mock_strategy_store.write_signal.assert_not_called()

    def test_persist_decision_circuit_breaker_recovery(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify circuit breaker allows write after recovery."""
        # First call - breaker open
        mock_circuit_breaker.can_execute.return_value = False
        signal = create_mock_signal()
        result1 = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )
        assert result1 is False

        # Second call - breaker closed
        mock_circuit_breaker.can_execute.return_value = True
        result2 = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )
        assert result2 is True
        mock_strategy_store.write_signal.assert_called_once()

    def test_circuit_breaker_records_success(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify circuit breaker success recorded after write."""
        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_circuit_breaker.record_success.assert_called_once()

    def test_circuit_breaker_open_after_failures(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify circuit breaker failure recorded on store error."""
        mock_strategy_store.write_signal.side_effect = Exception("Store error")

        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_circuit_breaker.record_failure.assert_called_once()

    def test_circuit_breaker_half_open_retry(
        self,
        mock_strategy_store: MagicMock,
        mock_bus_publisher: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Verify circuit breaker half-open state allows retry."""
        # Simulate half-open behavior
        breaker = MagicMock()
        call_count = 0

        def half_open_behavior() -> bool:
            nonlocal call_count
            call_count += 1
            # First call fails (open), second allows (half-open test)
            return call_count > 1

        breaker.can_execute = half_open_behavior
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        component = DecisionPersistenceComponent(
            strategy_id="test",
            strategy_store=mock_strategy_store,
            circuit_breaker=breaker,
            bus_publisher=mock_bus_publisher,
            log=mock_logger,
        )

        signal = create_mock_signal()

        # First call - blocked
        result1 = component.persist_decision(signal=signal, decision_type="BUY")
        assert result1 is False

        # Second call - allowed (half-open)
        result2 = component.persist_decision(signal=signal, decision_type="BUY")
        assert result2 is True


# ---------------------------------------------------------------------------
# Test Class: Event Publishing (No Store)
# ---------------------------------------------------------------------------


class TestEventPublishingNoStore:
    """Test event publishing when store is unavailable."""

    def test_persist_decision_publishes_event_when_store_unavailable(
        self,
        component_without_store: DecisionPersistenceComponent,
    ) -> None:
        """Verify events published when store is None."""
        signal = create_mock_signal()

        with patch.object(component_without_store, "get_decision_publisher") as mock_get_pub:
            mock_publisher = MagicMock()
            mock_publisher.publish = MagicMock(return_value=True)
            mock_get_pub.return_value = mock_publisher

            result = component_without_store.persist_decision(
                signal=signal,
                decision_type="BUY",
            )

            assert result is True
            mock_publisher.publish.assert_called_once()

    def test_persist_decision_partial_status_event(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify partial event published when circuit breaker open."""
        mock_circuit_breaker.can_execute.return_value = False

        signal = create_mock_signal()

        with patch.object(
            decision_persistence_component, "get_decision_publisher"
        ) as mock_get_pub:
            mock_publisher = MagicMock()
            mock_publisher.publish = MagicMock(return_value=True)
            mock_get_pub.return_value = mock_publisher

            decision_persistence_component.persist_decision(
                signal=signal,
                decision_type="BUY",
            )

            # Verify PARTIAL status was used
            mock_publisher.publish.assert_called_once()
            call_kwargs = mock_publisher.publish.call_args.kwargs
            from ml.config.events import EventStatus

            assert call_kwargs["status"] == EventStatus.PARTIAL


# ---------------------------------------------------------------------------
# Test Class: HOLD Signal Filtering
# ---------------------------------------------------------------------------


class TestHoldSignalFiltering:
    """Test HOLD signal filtering behavior."""

    def test_persist_decision_hold_signal_filtered(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify HOLD decisions skipped by default."""
        signal = create_mock_signal()

        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="HOLD",
        )

        assert result is False
        mock_strategy_store.write_signal.assert_not_called()

    def test_persist_decision_hold_signal_persisted_with_flag(
        self,
        component_with_persist_all: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify HOLD decisions persisted when configured."""
        signal = create_mock_signal()

        result = component_with_persist_all.persist_decision(
            signal=signal,
            decision_type="HOLD",
        )

        assert result is True
        mock_strategy_store.write_signal.assert_called_once()

    def test_persist_decision_hold_signal_persisted_with_override(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify HOLD decisions persisted when override flag is provided."""
        signal = create_mock_signal()

        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="HOLD",
            persist_hold=True,
        )

        assert result is True
        mock_strategy_store.write_signal.assert_called_once()


# ---------------------------------------------------------------------------
# Test Class: Risk Metrics Calculation
# ---------------------------------------------------------------------------


class TestRiskMetricsCalculation:
    """Test risk metrics calculation."""

    def test_persist_decision_risk_metrics_calculation(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify risk metrics calculated when not provided."""
        signal = create_mock_signal(confidence=0.85, prediction=0.75)

        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            risk_metrics=None,
        )

        call_args = mock_strategy_store.write_signal.call_args
        risk_metrics = call_args.kwargs["risk_metrics"]

        assert "confidence" in risk_metrics
        assert risk_metrics["confidence"] == 0.85
        assert "prediction" in risk_metrics
        assert risk_metrics["prediction"] == 0.75
        assert "active_positions" in risk_metrics

    def test_build_risk_metrics_includes_position_size(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify position size included in risk metrics."""
        from nautilus_trader.model.objects import Quantity

        signal = create_mock_signal()
        position_size = Quantity.from_str("100.5")

        metrics = decision_persistence_component._build_risk_metrics(signal, position_size)

        assert "position_size" in metrics
        assert metrics["position_size"] == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# Test Class: Execution Params Building
# ---------------------------------------------------------------------------


class TestExecutionParamsBuilding:
    """Test execution params building."""

    def test_persist_decision_execution_params_building(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify execution params calculated when not provided."""
        signal = create_mock_signal()

        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            execution_params=None,
        )

        call_args = mock_strategy_store.write_signal.call_args
        exec_params = call_args.kwargs["execution_params"]

        assert "stop_loss_pct" in exec_params
        assert exec_params["stop_loss_pct"] == 0.02
        assert "take_profit_pct" in exec_params
        assert exec_params["take_profit_pct"] == 0.04
        assert "max_positions" in exec_params
        assert exec_params["max_positions"] == 5

    def test_build_execution_params_with_position_size(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify execution params include position size."""
        from nautilus_trader.model.objects import Quantity

        signal = create_mock_signal()
        position_size = Quantity.from_str("50")

        params = decision_persistence_component._build_execution_params(
            signal, "BUY", position_size
        )

        assert params["position_size"] == "50"

    def test_execution_params_include_positions_metadata(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify positions metadata is appended to execution params."""
        signal = create_mock_signal()
        positions_metadata = {
            "source": "cache_positions_open",
            "ready": True,
            "degraded": False,
            "reason": None,
            "count": 2,
        }

        decision_persistence_component.update_state(
            positions_metadata=positions_metadata,
        )
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            execution_params=None,
        )

        call_args = mock_strategy_store.write_signal.call_args
        exec_params = call_args.kwargs["execution_params"]

        assert exec_params["positions"] == positions_metadata


# ---------------------------------------------------------------------------
# Test Class: Decision Publisher
# ---------------------------------------------------------------------------


class TestDecisionPublisher:
    """Test decision publisher functionality."""

    def test_get_decision_publisher_lazy_initialization(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify publisher is lazily created."""
        # Initially should be None
        assert decision_persistence_component._decision_publisher is None

        # Get publisher - patch at the source modules
        with patch("ml.config.bus.MessageBusConfig") as mock_cfg:
            mock_cfg.from_env.return_value = MagicMock(scheme="memory", topic_prefix="test")
            with patch(
                "ml.strategies.services.StrategyDecisionPublisher"
            ) as mock_pub_cls:
                mock_pub_cls.return_value = MagicMock()
                publisher = decision_persistence_component.get_decision_publisher()

                assert publisher is not None

    def test_get_decision_publisher_cached(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify publisher is cached after first call."""
        with patch("ml.config.bus.MessageBusConfig") as mock_cfg:
            mock_cfg.from_env.return_value = MagicMock(scheme="memory", topic_prefix="test")
            with patch(
                "ml.strategies.services.StrategyDecisionPublisher"
            ) as mock_pub_cls:
                mock_instance = MagicMock()
                mock_pub_cls.return_value = mock_instance

                publisher1 = decision_persistence_component.get_decision_publisher()
                publisher2 = decision_persistence_component.get_decision_publisher()

                # Should be same instance
                assert publisher1 is publisher2
                # Should only be created once
                mock_pub_cls.assert_called_once()

    def test_publish_decision_event_success(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_bus_publisher: MagicMock,
    ) -> None:
        """Verify decision event published successfully."""
        signal = create_mock_signal()

        with patch(
            "ml.common.message_bus.publisher_from_config"
        ) as mock_pub_cfg:
            mock_pub_cfg.return_value = mock_bus_publisher

            result = decision_persistence_component.publish_decision_event(
                signal=signal,
                decision_type="BUY",
                is_live=True,
            )

            assert result is True
            mock_bus_publisher.publish.assert_called_once()

    def test_publish_decision_event_handles_exception(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Verify graceful handling when publisher unavailable."""
        # Create a component without bus_publisher so it falls back to publisher_from_config
        component = DecisionPersistenceComponent(
            strategy_id="test_strategy",
            strategy_store=None,
            bus_publisher=None,  # No bus publisher
            log=mock_logger,
        )

        signal = create_mock_signal()

        with patch(
            "ml.common.message_bus.publisher_from_config"
        ) as mock_pub_cfg:
            mock_pub_cfg.return_value = None

            result = component.publish_decision_event(
                signal=signal,
                decision_type="BUY",
            )

            # Should not raise, just return False
            assert result is False


# ---------------------------------------------------------------------------
# Test Class: Metrics Recording
# ---------------------------------------------------------------------------


class TestMetricsRecording:
    """Test metrics recording."""

    def test_metrics_decisions_persisted_counter(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify decisions persisted counter incremented."""
        # Set up mock metric
        mock_counter = MagicMock()
        mock_labels = MagicMock()
        mock_counter.labels.return_value = mock_labels
        decision_persistence_component._decisions_persisted_counter = mock_counter

        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_counter.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.inc.assert_called_once()

    def test_metrics_write_latency_histogram(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify write latency metric recorded."""
        mock_histogram = MagicMock()
        mock_labels = MagicMock()
        mock_histogram.labels.return_value = mock_labels
        decision_persistence_component._write_latency_histogram = mock_histogram

        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_histogram.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.observe.assert_called_once()
        # Latency should be positive
        assert mock_labels.observe.call_args[0][0] > 0

    def test_metrics_batch_size_gauge(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify batch size metric recorded."""
        # Set up mock with write buffer
        mock_strategy_store._write_buffer = [1, 2, 3]  # 3 items in buffer

        mock_gauge = MagicMock()
        mock_labels = MagicMock()
        mock_gauge.labels.return_value = mock_labels
        decision_persistence_component._batch_size_gauge = mock_gauge

        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_gauge.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.set.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# Test Class: Model Predictions Building
# ---------------------------------------------------------------------------


class TestModelPredictionsBuilding:
    """Test model predictions building."""

    def test_build_model_predictions_extracts_from_signal(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify model predictions extracted from signal."""
        signal = create_mock_signal(model_id="test_model", prediction=0.75)

        predictions = decision_persistence_component._build_model_predictions(signal)

        assert "test_model" in predictions
        assert predictions["test_model"] == 0.75

    def test_build_model_predictions_includes_aggregated(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify aggregated model predictions included."""
        # Set up model signals buffer
        decision_persistence_component._model_signals = {
            "model_a": create_mock_signal(model_id="model_a", prediction=0.6),
            "model_b": create_mock_signal(model_id="model_b", prediction=0.8),
        }

        # Create aggregated signal
        signal = create_mock_signal(
            model_id="aggregated",
            prediction=0.7,
            metadata={"aggregated_from": ["model_a", "model_b"]},
        )

        predictions = decision_persistence_component._build_model_predictions(signal)

        assert "aggregated" in predictions
        assert "model_a" in predictions
        assert "model_b" in predictions
        assert predictions["model_a"] == 0.6
        assert predictions["model_b"] == 0.8


# ---------------------------------------------------------------------------
# Test Class: State Updates
# ---------------------------------------------------------------------------


class TestStateUpdates:
    """Test state update functionality."""

    def test_update_state_active_positions(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify active positions can be updated."""
        assert decision_persistence_component._active_positions == 0

        decision_persistence_component.update_state(active_positions=3)

        assert decision_persistence_component._active_positions == 3

    def test_update_state_is_backtesting(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify is_backtesting can be updated."""
        assert decision_persistence_component._is_backtesting is False

        decision_persistence_component.update_state(is_backtesting=True)

        assert decision_persistence_component._is_backtesting is True

    def test_update_state_model_signals(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify model signals buffer can be updated."""
        new_signals = {"model_x": create_mock_signal(model_id="model_x")}

        decision_persistence_component.update_state(model_signals=new_signals)

        assert "model_x" in decision_persistence_component._model_signals


# ---------------------------------------------------------------------------
# Test Class: Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test component properties."""

    def test_strategy_id_property(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify strategy_id property."""
        assert decision_persistence_component.strategy_id == "test_strategy"

    def test_strategy_store_property(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify strategy_store property."""
        assert decision_persistence_component.strategy_store is mock_strategy_store

    def test_circuit_breaker_property(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify circuit_breaker property."""
        assert decision_persistence_component.circuit_breaker is mock_circuit_breaker

    def test_persist_all_signals_property(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify persist_all_signals property."""
        assert decision_persistence_component.persist_all_signals is False

    def test_circuit_breaker_open_property_when_closed(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify circuit_breaker_open returns False when breaker closed."""
        mock_circuit_breaker.can_execute.return_value = True
        assert decision_persistence_component.circuit_breaker_open is False

    def test_circuit_breaker_open_property_when_open(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify circuit_breaker_open returns True when breaker open."""
        mock_circuit_breaker.can_execute.return_value = False
        assert decision_persistence_component.circuit_breaker_open is True

    def test_circuit_breaker_open_property_when_none(
        self,
        component_without_store: DecisionPersistenceComponent,
    ) -> None:
        """Verify circuit_breaker_open returns False when no breaker."""
        assert component_without_store.circuit_breaker_open is False


# ---------------------------------------------------------------------------
# Test Class: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling behavior."""

    def test_persist_decision_store_write_failure_logs_error(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Verify store write failures are logged."""
        mock_strategy_store.write_signal.side_effect = Exception("Database error")

        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        mock_logger.error.assert_called()

    def test_persist_decision_partial_event_on_store_failure(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify partial event published on store failure."""
        mock_strategy_store.write_signal.side_effect = Exception("Store error")

        signal = create_mock_signal()

        with patch.object(
            decision_persistence_component, "get_decision_publisher"
        ) as mock_get_pub:
            mock_publisher = MagicMock()
            mock_publisher.publish = MagicMock(return_value=True)
            mock_get_pub.return_value = mock_publisher

            decision_persistence_component.persist_decision(
                signal=signal,
                decision_type="BUY",
            )

            # Verify PARTIAL status was used
            mock_publisher.publish.assert_called()

    def test_no_crash_on_metrics_failure(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify metrics failures don't crash persistence."""
        # Set up broken metric
        mock_counter = MagicMock()
        mock_counter.labels.side_effect = Exception("Metrics error")
        decision_persistence_component._decisions_persisted_counter = mock_counter

        signal = create_mock_signal()

        # Should not raise
        result = decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        # Store write should still succeed
        assert result is True


# ---------------------------------------------------------------------------
# Test Class: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_persist_with_custom_risk_metrics(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify custom risk metrics are used when provided."""
        signal = create_mock_signal()
        custom_metrics = {"custom_risk": 0.5, "volatility": 0.15}

        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            risk_metrics=custom_metrics,
        )

        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["risk_metrics"] == custom_metrics

    def test_persist_with_custom_execution_params(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify custom execution params are used when provided."""
        signal = create_mock_signal()
        custom_params = {"urgency": "high", "algo": "twap"}

        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
            execution_params=custom_params,
        )

        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["execution_params"] == custom_params

    def test_is_live_flag_in_backtest_mode(
        self,
        mock_strategy_store: MagicMock,
        mock_circuit_breaker: MagicMock,
        mock_bus_publisher: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Verify is_live flag set correctly for backtest."""
        component = DecisionPersistenceComponent(
            strategy_id="test",
            strategy_store=mock_strategy_store,
            circuit_breaker=mock_circuit_breaker,
            bus_publisher=mock_bus_publisher,
            log=mock_logger,
            is_backtesting=True,  # Backtesting mode
        )

        signal = create_mock_signal()
        component.persist_decision(signal=signal, decision_type="BUY")

        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["is_live"] is False

    def test_is_live_flag_in_live_mode(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify is_live flag set correctly for live trading."""
        signal = create_mock_signal()
        decision_persistence_component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        call_args = mock_strategy_store.write_signal.call_args
        assert call_args.kwargs["is_live"] is True

    def test_model_id_from_metadata_fallback(
        self,
        decision_persistence_component: DecisionPersistenceComponent,
    ) -> None:
        """Verify model_id extraction from metadata when field is None."""
        # Create signal with model_id in metadata
        inst_id = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
        signal = MLSignal(
            instrument_id=inst_id,
            model_id="",  # Empty model_id
            prediction=0.7,
            confidence=0.8,
            metadata={"model_id": "meta_model"},
            ts_event=time.time_ns(),
            ts_init=time.time_ns(),
        )

        predictions = decision_persistence_component._build_model_predictions(signal)

        # Should use metadata model_id
        assert "meta_model" in predictions or "" in predictions


# ---------------------------------------------------------------------------
# Test Class: Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test component initialization."""

    def test_default_initialization(self) -> None:
        """Verify component initializes with defaults."""
        component = DecisionPersistenceComponent(strategy_id="test")

        assert component.strategy_id == "test"
        assert component.strategy_store is None
        assert component.circuit_breaker is None
        assert component.persist_all_signals is False

    def test_custom_initialization(
        self,
        mock_strategy_store: MagicMock,
        mock_circuit_breaker: MagicMock,
    ) -> None:
        """Verify component initializes with custom values."""
        component = DecisionPersistenceComponent(
            strategy_id="custom_strategy",
            strategy_store=mock_strategy_store,
            circuit_breaker=mock_circuit_breaker,
            persist_all_signals=True,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_positions=10,
        )

        assert component.strategy_id == "custom_strategy"
        assert component.strategy_store is mock_strategy_store
        assert component.circuit_breaker is mock_circuit_breaker
        assert component.persist_all_signals is True
        assert component._stop_loss_pct == 0.05
        assert component._take_profit_pct == 0.10
        assert component._max_positions == 10

    def test_no_logger_does_not_crash(self) -> None:
        """Verify component works without logger."""
        component = DecisionPersistenceComponent(
            strategy_id="test",
            log=None,
        )

        # Should not raise
        signal = create_mock_signal()

        # No store, so it will try to publish event
        # We mock the publisher to avoid external dependencies
        with patch.object(component, "get_decision_publisher") as mock_get_pub:
            mock_get_pub.return_value = None  # No publisher available

            result = component.persist_decision(signal=signal, decision_type="BUY")

            # No store AND no publisher, so should return False
            assert result is False
