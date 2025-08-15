"""
Unit tests for dry run mode in MLTradingStrategy.

Tests the execute_trades flag that allows running the strategy
without actually submitting orders to the broker.
"""

from unittest.mock import MagicMock, patch

import pytest

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.common.component import MessageBus, TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


class TestDryRunMode:
    """Test cases for dry run mode in MLTradingStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("DRY-RUN-TESTER")
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
        self.instrument_id = InstrumentId.from_str("BTC/USDT.BINANCE")
    
    def test_dry_run_mode_enabled(self):
        """Test that strategy does not execute trades when execute_trades=False."""
        config = MLStrategyConfig(
            strategy_id="DRY-RUN-TEST",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            execute_trades=False,  # Dry run mode enabled
            use_strategy_store=True,
        )
        
        # Create strategy with mocked store
        with patch('ml.strategies.base.StrategyStore') as MockStore:
            mock_store_instance = MagicMock()
            MockStore.return_value = mock_store_instance
            
            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )
            
            # Mock the order submission methods
            strategy._place_market_order = MagicMock()
            strategy._calculate_position_size = MagicMock(return_value=100)
            
            # Create BUY signal
            signal = MLSignal(
                instrument_id=self.instrument_id,
                model_id="test_model",
                prediction=0.8,  # > 0.5 = BUY
                confidence=0.9,
                metadata={},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()),
                ts_init=dt_to_unix_nanos(self.clock.utc_now()),
            )
            
            # Process signal
            strategy._handle_ml_signal(signal)
            
            # Verify no actual orders were placed
            strategy._place_market_order.assert_not_called()
            
            # Verify dry run counter incremented
            assert strategy._dry_run_trades == 1
            
            # Verify decision was still persisted
            mock_store_instance.write_signal.assert_called()
    
    def test_normal_execution_mode(self):
        """Test that strategy executes trades normally when execute_trades=True."""
        config = MLStrategyConfig(
            strategy_id="NORMAL-TEST",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            execute_trades=True,  # Normal execution mode
            use_strategy_store=True,
        )
        
        # Create strategy with mocked store
        with patch('ml.strategies.base.StrategyStore') as MockStore:
            mock_store_instance = MagicMock()
            MockStore.return_value = mock_store_instance
            
            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )
            
            # Mock the order submission methods
            strategy._place_market_order = MagicMock(return_value="ORDER-123")
            strategy._calculate_position_size = MagicMock(return_value=100)
            
            # Create BUY signal
            signal = MLSignal(
                instrument_id=self.instrument_id,
                model_id="test_model",
                prediction=0.8,  # > 0.5 = BUY
                confidence=0.9,
                metadata={},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()),
                ts_init=dt_to_unix_nanos(self.clock.utc_now()),
            )
            
            # Process signal
            strategy._handle_ml_signal(signal)
            
            # Verify order was placed
            strategy._place_market_order.assert_called_once()
            
            # Verify dry run counter not incremented
            assert strategy._dry_run_trades == 0
            
            # Verify decision was persisted
            mock_store_instance.write_signal.assert_called()
    
    def test_dry_run_persistence_and_metrics(self):
        """Test that persistence and metrics work correctly in dry run mode."""
        config = MLStrategyConfig(
            strategy_id="METRICS-TEST",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            execute_trades=False,  # Dry run mode
            use_strategy_store=True,
            persist_all_signals=True,
        )
        
        # Create strategy with mocked store
        with patch('ml.strategies.base.StrategyStore') as MockStore:
            mock_store_instance = MagicMock()
            MockStore.return_value = mock_store_instance
            
            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )
            
            # Mock position sizing
            strategy._calculate_position_size = MagicMock(return_value=100)
            
            # Process multiple signals
            for i in range(3):
                signal = MLSignal(
                    instrument_id=self.instrument_id,
                    model_id=f"model_{i}",
                    prediction=0.6 + i * 0.1,  # 0.6, 0.7, 0.8
                    confidence=0.8,
                    metadata={"iteration": i},
                    ts_event=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
                    ts_init=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
                )
                # Call the handler which increments counters
                strategy._handle_ml_signal(signal)
                self.clock.advance_time(1000000000)  # 1 second
            
            # Verify dry run trades counted
            assert strategy._dry_run_trades == 3
            
            # Verify all decisions were persisted
            assert mock_store_instance.write_signal.call_count == 3
            
            # Verify signals were tracked
            assert strategy._signals_received == 3
            
            # Stop the strategy and check final log
            strategy.on_stop()
            
            # Verify flush was called
            mock_store_instance.flush.assert_called_once()
    
    def test_dry_run_with_position_reversal(self):
        """Test dry run mode handles position reversals correctly."""
        config = MLStrategyConfig(
            strategy_id="REVERSAL-TEST",
            instrument_id=self.instrument_id,
            ml_signal_source="TEST_ACTOR",
            execute_trades=False,  # Dry run mode
            use_strategy_store=True,
        )
        
        # Create strategy with mocked store
        with patch('ml.strategies.base.StrategyStore') as MockStore:
            mock_store_instance = MagicMock()
            MockStore.return_value = mock_store_instance
            
            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )
            
            # Mock position sizing and existing position
            strategy._calculate_position_size = MagicMock(return_value=100)
            mock_position = MagicMock()
            mock_position.side.name = "LONG"
            mock_position.quantity = 100
            
            # First signal - enter position
            signal1 = MLSignal(
                instrument_id=self.instrument_id,
                model_id="test_model",
                prediction=0.8,  # BUY
                confidence=0.9,
                metadata={},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()),
                ts_init=dt_to_unix_nanos(self.clock.utc_now()),
            )
            
            strategy._handle_ml_signal(signal1)
            assert strategy._dry_run_trades == 1
            
            # Mock existing position for reversal
            strategy._get_current_position = MagicMock(return_value=mock_position)
            
            # Second signal - reverse position
            signal2 = MLSignal(
                instrument_id=self.instrument_id,
                model_id="test_model",
                prediction=0.2,  # SELL (reverse from LONG)
                confidence=0.9,
                metadata={},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()) + 1000000000,
                ts_init=dt_to_unix_nanos(self.clock.utc_now()) + 1000000000,
            )
            
            strategy._handle_ml_signal(signal2)
            assert strategy._dry_run_trades == 2
            
            # Verify both decisions were persisted
            assert mock_store_instance.write_signal.call_count == 2
            
            # Check the persisted decisions
            calls = mock_store_instance.write_signal.call_args_list
            
            # First call should be BUY
            assert calls[0].kwargs["signal_type"] == "BUY"
            
            # Second call should be SELL (reversal)
            assert calls[1].kwargs["signal_type"] == "SELL"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])