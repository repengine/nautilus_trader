"""
Unit tests for StrategyStore integration in MLTradingStrategy.

Tests the persistence of strategy decisions to the StrategyStore,
including configuration, error handling, and metrics.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.stores.strategy_store import StrategyStore
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.fixtures.model_factory import TestModelFactory
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


class TestStrategyStoreIntegration:
    """Test cases for StrategyStore integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TESTER-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )
        self.cache = TestComponentStubs.cache()
        self.portfolio = Portfolio(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Create test instrument
        self.instrument_id = InstrumentId.from_str("EUR/USD.SIM")

        # Create test model
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_path = TestModelFactory.create_minimal_xgboost_model(
            n_features=23,
            output_path=self.temp_dir / "test_model.pkl"
        )

    def test_strategy_store_initialization_with_config(self):
        """Test that StrategyStore is initialized when configured."""
        # Create config with StrategyStore enabled
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
            strategy_store_config={
                "connection_string": "postgresql://test:test@localhost:5432/test",
                "batch_size": 50,
                "flush_interval_ms": 500,
            }
        )

        # Create strategy with mocked store
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Verify StrategyStore was created with correct parameters
            MockStore.assert_called_once()
            call_args = MockStore.call_args
            assert call_args.kwargs["connection_string"] == "postgresql://test:test@localhost:5432/test"
            assert call_args.kwargs["batch_size"] == 50
            assert call_args.kwargs["flush_interval_ms"] == 500
            # Clock should be present but might be the strategy's clock
            assert "clock" in call_args.kwargs

            assert strategy.strategy_store is not None

    def test_strategy_store_disabled_by_config(self):
        """Test that StrategyStore is not initialized when disabled."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=False,  # Disabled
        )

        strategy = MLTradingStrategy(config)
        strategy.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        assert strategy.strategy_store is None

    def test_persist_buy_decision(self):
        """Test persisting a BUY decision to StrategyStore."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Create test signal
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.8,  # > 0.5 = BUY
            confidence=0.9,
            metadata={"features": {"rsi": 45}},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Process signal
        strategy._process_ml_signal(signal)

        # Verify write_signal was called
        mock_store.write_signal.assert_called()
        call_args = mock_store.write_signal.call_args

        assert call_args.kwargs["signal_type"] == "BUY"
        assert call_args.kwargs["strength"] == 0.9
        assert call_args.kwargs["model_predictions"] == {"test_model": 0.8}
        assert call_args.kwargs["instrument_id"] == str(self.instrument_id)

    def test_persist_sell_decision(self):
        """Test persisting a SELL decision to StrategyStore."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Create test signal
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.2,  # < 0.5 = SELL
            confidence=0.85,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Process signal
        strategy._process_ml_signal(signal)

        # Verify write_signal was called
        mock_store.write_signal.assert_called()
        call_args = mock_store.write_signal.call_args

        assert call_args.kwargs["signal_type"] == "SELL"
        assert call_args.kwargs["strength"] == 0.85
        assert call_args.kwargs["model_predictions"] == {"test_model": 0.2}

    def test_persist_hold_decision_with_config(self):
        """Test that HOLD decisions are persisted when configured."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
            persist_all_signals=True,  # Persist HOLD signals
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Mock existing position that aligns with signal
        mock_position = MagicMock()
        mock_position.side.name = "LONG"
        strategy._get_current_position = MagicMock(return_value=mock_position)

        # Create BUY signal (aligns with LONG position = HOLD)
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.75,  # > 0.5 = BUY, but we have LONG position
            confidence=0.8,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Process signal
        strategy._process_ml_signal(signal)

        # Verify HOLD was persisted
        mock_store.write_signal.assert_called()
        call_args = mock_store.write_signal.call_args
        assert call_args.kwargs["signal_type"] == "HOLD"

    def test_skip_hold_decision_when_not_configured(self):
        """Test that HOLD decisions are not persisted when not configured."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
            persist_all_signals=False,  # Don't persist HOLD signals
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Create signal and call helper directly with HOLD
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.5,
            confidence=0.7,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        strategy._persist_strategy_decision(
            signal=signal,
            decision_type="HOLD",
            position_size=None,
        )

        # Verify write_signal was NOT called for HOLD
        mock_store.write_signal.assert_not_called()

    def test_error_handling_in_persistence(self):
        """Test that errors in persistence are handled gracefully."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            mock_store_instance.write_signal.side_effect = Exception("Database connection lost")
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Create test signal
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.8,
            confidence=0.9,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Process signal - should not raise exception
        strategy._process_ml_signal(signal)

        # Verify error was handled
        mock_store.write_signal.assert_called()

    def test_flush_on_stop(self):
        """Test that StrategyStore is flushed when strategy stops."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store

        # Stop the strategy
        strategy.on_stop()

        # Verify flush was called
        mock_store.flush.assert_called_once()

    def test_risk_metrics_calculation(self):
        """Test that risk metrics are properly calculated and persisted."""
        config = MLStrategyConfig(
            strategy_id="TEST-001",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            use_strategy_store=True,
            position_size_pct=0.02,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
        )

        # Mock StrategyStore to prevent real connection
        with patch("ml.strategies.base.StrategyStore") as MockStore:
            mock_store_instance = MagicMock(spec=StrategyStore)
            MockStore.return_value = mock_store_instance

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )

            # Store should be the mocked instance
            mock_store = strategy.strategy_store
            strategy._active_positions = 2

        # Create test signal
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="test_model",
            prediction=0.75,
            confidence=0.85,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Process signal
        strategy._process_ml_signal(signal)

        # Verify risk metrics were included
        call_args = mock_store.write_signal.call_args
        risk_metrics = call_args.kwargs["risk_metrics"]

        assert risk_metrics["confidence"] == 0.85
        assert risk_metrics["prediction"] == 0.75
        assert risk_metrics["active_positions"] == 2
        assert risk_metrics["signal_direction"] == "LONG"

        # Verify execution params
        exec_params = call_args.kwargs["execution_params"]
        assert exec_params["target_side"] == "BUY"
        assert exec_params["model_id"] == "test_model"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
