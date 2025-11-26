"""
Tests for LifecycleComponent.

This module tests the lifecycle component extracted from BaseMLStrategy
as part of the Phase 3.4 decomposition. Tests cover:

- Signal subscription with and without client_id
- Instrument subscription
- Configuration logging on startup
- Statistics logging on stop
- Strategy store flush on stop
- Flush exception handling
- Statistics calculation

"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from ml.strategies.common.lifecycle import (
    LifecycleComponent,
    StrategyStoreProtocol,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def create_instrument_id(instrument_str: str = "EURUSD.SIM") -> InstrumentId:
    """Create an InstrumentId for testing."""
    parts = instrument_str.split(".")
    return InstrumentId(Symbol(parts[0]), Venue(parts[1]))


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
def mock_strategy_store() -> MagicMock:
    """Create a mock strategy store."""
    store = MagicMock(spec=StrategyStoreProtocol)
    store.flush = MagicMock()
    return store


@pytest.fixture
def mock_subscribe_data() -> MagicMock:
    """Create a mock subscribe_data callback."""
    return MagicMock()


@pytest.fixture
def mock_subscribe_instrument() -> MagicMock:
    """Create a mock subscribe_instrument callback."""
    return MagicMock()


@pytest.fixture
def lifecycle_component(
    mock_logger: MagicMock,
    mock_subscribe_data: MagicMock,
    mock_subscribe_instrument: MagicMock,
) -> LifecycleComponent:
    """Create a lifecycle component with mocked dependencies."""
    return LifecycleComponent(
        strategy_id="test_strategy",
        instrument_id=create_instrument_id("EURUSD.SIM"),
        signal_client_id=None,
        target_model_ids=["model_a", "model_b"],
        aggregation_mode="weighted_average",
        position_size_pct=0.05,
        min_confidence=0.6,
        execute_trades=True,
        subscribe_data_callback=mock_subscribe_data,
        subscribe_instrument_callback=mock_subscribe_instrument,
        log=mock_logger,
    )


@pytest.fixture
def lifecycle_component_with_client_id(
    mock_logger: MagicMock,
    mock_subscribe_data: MagicMock,
    mock_subscribe_instrument: MagicMock,
) -> LifecycleComponent:
    """Create a lifecycle component with a specific client_id."""
    return LifecycleComponent(
        strategy_id="test_strategy",
        instrument_id=create_instrument_id("EURUSD.SIM"),
        signal_client_id="actor_1",
        target_model_ids=["model_a"],
        aggregation_mode=None,
        position_size_pct=0.02,
        min_confidence=0.5,
        execute_trades=True,
        subscribe_data_callback=mock_subscribe_data,
        subscribe_instrument_callback=mock_subscribe_instrument,
        log=mock_logger,
    )


@pytest.fixture
def lifecycle_component_dry_run(
    mock_logger: MagicMock,
    mock_subscribe_data: MagicMock,
    mock_subscribe_instrument: MagicMock,
) -> LifecycleComponent:
    """Create a lifecycle component in dry run mode."""
    return LifecycleComponent(
        strategy_id="test_strategy",
        instrument_id=create_instrument_id("GBPUSD.SIM"),
        signal_client_id=None,
        execute_trades=False,
        subscribe_data_callback=mock_subscribe_data,
        subscribe_instrument_callback=mock_subscribe_instrument,
        log=mock_logger,
    )


@pytest.fixture
def lifecycle_component_no_callbacks(
    mock_logger: MagicMock,
) -> LifecycleComponent:
    """Create a lifecycle component without subscription callbacks."""
    return LifecycleComponent(
        strategy_id="test_strategy",
        instrument_id=create_instrument_id("EURUSD.SIM"),
        subscribe_data_callback=None,
        subscribe_instrument_callback=None,
        log=mock_logger,
    )


# ---------------------------------------------------------------------------
# Test Class: on_start - ML Signal Subscriptions
# ---------------------------------------------------------------------------


class TestOnStartMLSignalSubscriptions:
    """Test on_start ML signal subscription behavior."""

    def test_on_start_subscribes_to_ml_signals_with_client_id(
        self,
        lifecycle_component_with_client_id: LifecycleComponent,
        mock_subscribe_data: MagicMock,
    ) -> None:
        """Verify signal subscription with specific client ID."""
        lifecycle_component_with_client_id.on_start()

        # Should have been called once
        mock_subscribe_data.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_subscribe_data.call_args.kwargs
        assert "data_type" in call_kwargs
        assert "client_id" in call_kwargs
        assert call_kwargs["client_id"] is not None
        # Verify the client_id value
        assert str(call_kwargs["client_id"]) == "actor_1"

    def test_on_start_subscribes_to_ml_signals_without_client_id(
        self,
        lifecycle_component: LifecycleComponent,
        mock_subscribe_data: MagicMock,
    ) -> None:
        """Verify signal subscription for all sources (no client_id)."""
        lifecycle_component.on_start()

        # Should have been called once
        mock_subscribe_data.assert_called_once()

        # Verify call arguments - client_id should be None
        call_kwargs = mock_subscribe_data.call_args.kwargs
        assert "data_type" in call_kwargs
        assert "client_id" in call_kwargs
        assert call_kwargs["client_id"] is None


# ---------------------------------------------------------------------------
# Test Class: on_start - Instrument Subscriptions
# ---------------------------------------------------------------------------


class TestOnStartInstrumentSubscription:
    """Test on_start instrument subscription behavior."""

    def test_on_start_subscribes_to_instrument(
        self,
        lifecycle_component: LifecycleComponent,
        mock_subscribe_instrument: MagicMock,
    ) -> None:
        """Verify instrument subscription is called."""
        lifecycle_component.on_start()

        # Should have been called once
        mock_subscribe_instrument.assert_called_once()

        # Verify the instrument_id was passed
        call_args = mock_subscribe_instrument.call_args
        instrument_id = call_args[0][0]  # First positional argument
        assert str(instrument_id) == "EURUSD.SIM"


# ---------------------------------------------------------------------------
# Test Class: on_start - Configuration Logging
# ---------------------------------------------------------------------------


class TestOnStartConfigurationLogging:
    """Test on_start configuration logging behavior."""

    def test_on_start_logs_configuration(
        self,
        lifecycle_component: LifecycleComponent,
        mock_logger: MagicMock,
    ) -> None:
        """Verify configuration is logged on startup."""
        lifecycle_component.on_start()

        # Should have info calls for startup and configuration
        assert mock_logger.info.call_count >= 2

        # Check that config log contains key information
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        config_log_found = False
        for call_str in info_calls:
            if "ML Strategy configured" in call_str:
                config_log_found = True
                assert "EURUSD.SIM" in call_str
                assert "position_size" in call_str
                assert "min_confidence" in call_str
                break

        assert config_log_found, "Configuration log message not found"


# ---------------------------------------------------------------------------
# Test Class: on_stop - Statistics Logging
# ---------------------------------------------------------------------------


class TestOnStopStatisticsLogging:
    """Test on_stop statistics logging behavior."""

    def test_on_stop_logs_statistics(
        self,
        lifecycle_component: LifecycleComponent,
        mock_logger: MagicMock,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify statistics are logged on stop."""
        lifecycle_component.on_stop(
            strategy_store=mock_strategy_store,
            signals_received=100,
            trades_executed=50,
            winning_trades=30,
            total_pnl=Decimal("1250.50"),
        )

        # Should have logged statistics
        mock_logger.info.assert_called()

        # Check that log contains statistics
        last_call = mock_logger.info.call_args
        log_message = str(last_call)
        assert "Signals: 100" in log_message
        assert "Trades: 50" in log_message
        assert "Win rate: 60.0%" in log_message
        assert "1250.50" in log_message

    def test_on_stop_logs_statistics_dry_run_mode(
        self,
        lifecycle_component_dry_run: LifecycleComponent,
        mock_logger: MagicMock,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify statistics logged in dry run mode."""
        lifecycle_component_dry_run.on_stop(
            strategy_store=mock_strategy_store,
            signals_received=100,
            trades_executed=0,
            winning_trades=0,
            total_pnl=Decimal("0"),
            dry_run_trades=75,
        )

        # Check that log contains dry run info
        last_call = mock_logger.info.call_args
        log_message = str(last_call)
        assert "DRY RUN MODE" in log_message
        assert "Dry Run Trades: 75" in log_message
        assert "execute_trades=False" in log_message


# ---------------------------------------------------------------------------
# Test Class: on_stop - Strategy Store Flush
# ---------------------------------------------------------------------------


class TestOnStopStoreFlush:
    """Test on_stop strategy store flush behavior."""

    def test_on_stop_flushes_store_buffer(
        self,
        lifecycle_component: LifecycleComponent,
        mock_strategy_store: MagicMock,
    ) -> None:
        """Verify strategy store flush is called on stop."""
        lifecycle_component.on_stop(
            strategy_store=mock_strategy_store,
            signals_received=10,
            trades_executed=5,
            winning_trades=3,
            total_pnl=Decimal("100"),
        )

        # Verify flush was called
        mock_strategy_store.flush.assert_called_once()

    def test_on_stop_handles_flush_exception(
        self,
        lifecycle_component: LifecycleComponent,
        mock_strategy_store: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Verify flush exceptions are caught and logged."""
        # Make flush raise an exception
        mock_strategy_store.flush.side_effect = RuntimeError("Database connection lost")

        # Should not raise - exception should be caught
        lifecycle_component.on_stop(
            strategy_store=mock_strategy_store,
            signals_received=10,
            trades_executed=5,
            winning_trades=3,
            total_pnl=Decimal("100"),
        )

        # Verify error was logged
        mock_logger.error.assert_called()
        error_call = str(mock_logger.error.call_args)
        assert "flush_failed" in error_call or "Database connection lost" in error_call

    def test_on_stop_without_store_does_not_flush(
        self,
        lifecycle_component: LifecycleComponent,
        mock_logger: MagicMock,
    ) -> None:
        """Verify on_stop works without a strategy store."""
        # Should not raise when store is None
        lifecycle_component.on_stop(
            strategy_store=None,
            signals_received=10,
            trades_executed=5,
            winning_trades=3,
            total_pnl=Decimal("100"),
        )

        # Just verify it completed without error
        mock_logger.info.assert_called()


# ---------------------------------------------------------------------------
# Test Class: get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Test get_statistics method."""

    def test_get_statistics_returns_correct_values(
        self,
        lifecycle_component: LifecycleComponent,
    ) -> None:
        """Verify statistics dictionary contains correct values."""
        stats = lifecycle_component.get_statistics(
            signals_received=100,
            trades_executed=50,
            winning_trades=30,
            total_pnl=Decimal("1250.50"),
        )

        assert stats["strategy_id"] == "test_strategy"
        assert stats["instrument_id"] == "EURUSD.SIM"
        assert stats["signals_received"] == 100
        assert stats["trades_executed"] == 50
        assert stats["winning_trades"] == 30
        assert stats["win_rate"] == pytest.approx(60.0)
        assert stats["total_pnl"] == pytest.approx(1250.50)
        assert stats["execute_trades"] is True

    def test_get_statistics_handles_zero_trades(
        self,
        lifecycle_component: LifecycleComponent,
    ) -> None:
        """Verify win rate calculation handles zero trades."""
        stats = lifecycle_component.get_statistics(
            signals_received=10,
            trades_executed=0,
            winning_trades=0,
            total_pnl=Decimal("0"),
        )

        # Win rate should be 0% with no trades (not divide by zero)
        assert stats["win_rate"] == 0.0

    def test_get_statistics_100_percent_win_rate(
        self,
        lifecycle_component: LifecycleComponent,
    ) -> None:
        """Verify 100% win rate is calculated correctly."""
        stats = lifecycle_component.get_statistics(
            signals_received=20,
            trades_executed=10,
            winning_trades=10,
            total_pnl=Decimal("500"),
        )

        assert stats["win_rate"] == pytest.approx(100.0)

    def test_get_statistics_dry_run_mode(
        self,
        lifecycle_component_dry_run: LifecycleComponent,
    ) -> None:
        """Verify execute_trades flag is correct for dry run."""
        stats = lifecycle_component_dry_run.get_statistics(
            signals_received=50,
            trades_executed=0,
            winning_trades=0,
            total_pnl=Decimal("0"),
        )

        assert stats["execute_trades"] is False


# ---------------------------------------------------------------------------
# Test Class: Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_on_start_without_callbacks_logs_warnings(
        self,
        lifecycle_component_no_callbacks: LifecycleComponent,
        mock_logger: MagicMock,
    ) -> None:
        """Verify warnings logged when callbacks not configured."""
        lifecycle_component_no_callbacks.on_start()

        # Should have logged warnings for missing callbacks
        assert mock_logger.warning.call_count >= 2

    def test_on_start_subscription_exception_is_caught(
        self,
        mock_logger: MagicMock,
        mock_subscribe_instrument: MagicMock,
    ) -> None:
        """Verify subscription exceptions are caught and logged."""
        # Make subscribe_data raise
        mock_subscribe_data = MagicMock(side_effect=RuntimeError("Subscription failed"))

        component = LifecycleComponent(
            strategy_id="test_strategy",
            instrument_id=create_instrument_id("EURUSD.SIM"),
            subscribe_data_callback=mock_subscribe_data,
            subscribe_instrument_callback=mock_subscribe_instrument,
            log=mock_logger,
        )

        # Should not raise
        component.on_start()

        # Should have logged error
        mock_logger.error.assert_called()

    def test_properties_return_correct_values(
        self,
        lifecycle_component_with_client_id: LifecycleComponent,
    ) -> None:
        """Verify property accessors return correct values."""
        assert lifecycle_component_with_client_id.strategy_id == "test_strategy"
        assert str(lifecycle_component_with_client_id.instrument_id) == "EURUSD.SIM"
        assert lifecycle_component_with_client_id.signal_client_id == "actor_1"
        assert lifecycle_component_with_client_id.target_model_ids == ["model_a"]
        assert lifecycle_component_with_client_id.aggregation_mode is None
        assert lifecycle_component_with_client_id.execute_trades is True

    def test_component_without_logger_uses_noop(
        self,
        mock_subscribe_data: MagicMock,
        mock_subscribe_instrument: MagicMock,
    ) -> None:
        """Verify component works without a logger (uses no-op)."""
        component = LifecycleComponent(
            strategy_id="test_strategy",
            instrument_id=create_instrument_id("EURUSD.SIM"),
            subscribe_data_callback=mock_subscribe_data,
            subscribe_instrument_callback=mock_subscribe_instrument,
            log=None,  # No logger provided
        )

        # Should not raise
        component.on_start()
        component.on_stop(
            strategy_store=None,
            signals_received=10,
            trades_executed=5,
            winning_trades=3,
            total_pnl=Decimal("100"),
        )
