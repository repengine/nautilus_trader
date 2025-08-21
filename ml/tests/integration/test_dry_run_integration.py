"""
Integration test demonstrating dry run mode in a realistic scenario.

This test shows how the strategy processes signals, makes decisions, persists to stores,
and updates metrics without executing actual trades.

"""

from typing import Any, cast
from unittest.mock import MagicMock
from unittest.mock import patch

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


def test_dry_run_production_scenario() -> None:
    """
    Test a realistic production scenario with dry run mode.

    This simulates:
    1. Multiple ML signals coming in
    2. Strategy processing and making decisions
    3. Persistence to StrategyStore
    4. Metrics being updated
    5. No actual orders being placed

    """
    # Setup
    clock = TestClock()
    trader_id = TraderId("PROD-DRY-RUN")
    msgbus = MessageBus(trader_id=trader_id, clock=clock)
    cache = TestComponentStubs.cache()
    portfolio = Portfolio(msgbus=msgbus, cache=cache, clock=clock)

    instrument_id = InstrumentId.from_str("ETH/USDT.BINANCE")

    # Configure strategy with dry run mode
    config = MLStrategyConfig(
        strategy_id="ML-PROD-DRY",
        instrument_id=instrument_id,
        ml_signal_source="ML_SIGNAL_ACTOR",
        execute_trades=False,  # DRY RUN MODE
        use_strategy_store=True,
        persist_all_signals=True,
        position_size_pct=0.05,
        min_confidence=0.6,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
    )

    # Create strategy with mocked store
    with patch("ml.strategies.base.StrategyStore") as MockStore:
        mock_store = MagicMock()
        MockStore.return_value = mock_store

        strategy = MLTradingStrategy(config)
        strategy.register_base(
            portfolio=portfolio,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Mock position sizing
        cast(Any, strategy)._calculate_position_size = MagicMock(return_value=50)
        cast(Any, strategy)._place_market_order = MagicMock()

        # Mock position tracking for reversal logic
        mock_position = None

        def get_position_mock() -> Any:
            return mock_position

        cast(Any, strategy)._get_current_position = get_position_mock

        # Simulate production signals coming in
        signals = [
            # Initial BUY signal
            MLSignal(
                instrument_id=instrument_id,
                model_id="xgb_trend_v2",
                prediction=0.75,
                confidence=0.82,
                metadata={"features": {"rsi": 45, "macd": 0.002}},
                ts_event=dt_to_unix_nanos(clock.utc_now()),
                ts_init=dt_to_unix_nanos(clock.utc_now()),
            ),
            # Confirming BUY signal (HOLD)
            MLSignal(
                instrument_id=instrument_id,
                model_id="xgb_trend_v2",
                prediction=0.68,
                confidence=0.71,
                metadata={"features": {"rsi": 52, "macd": 0.003}},
                ts_event=dt_to_unix_nanos(clock.utc_now()) + 10_000_000_000,
                ts_init=dt_to_unix_nanos(clock.utc_now()) + 10_000_000_000,
            ),
            # SELL signal (reversal)
            MLSignal(
                instrument_id=instrument_id,
                model_id="xgb_trend_v2",
                prediction=0.25,
                confidence=0.85,
                metadata={"features": {"rsi": 72, "macd": -0.001}},
                ts_event=dt_to_unix_nanos(clock.utc_now()) + 20_000_000_000,
                ts_init=dt_to_unix_nanos(clock.utc_now()) + 20_000_000_000,
            ),
        ]

        # Process signals as they would come in production
        for i, signal in enumerate(signals):
            print(f"\n--- Processing Signal {i+1} ---")
            print(f"Model: {signal.model_id}")
            print(f"Prediction: {signal.prediction:.3f}")
            print(f"Confidence: {signal.confidence:.3f}")

            # Process the signal
            strategy._handle_ml_signal(signal)

            # After first signal, mock a position exists
            if i == 0:
                mock_position = MagicMock()
                mock_position.side.name = "LONG"
                mock_position.quantity = 50

            # Advance time slightly
            clock.advance_time(1_000_000_000)  # 1 second

        # Verify dry run behavior
        print("\n--- Dry Run Results ---")
        print(f"Signals received: {strategy._signals_received}")
        print(f"Dry run trades: {strategy._dry_run_trades}")
        _mock_place = cast(Any, strategy)._place_market_order
        print(f"Actual orders placed: {_mock_place.call_count}")

        # Assertions
        assert strategy._signals_received == 3
        assert strategy._dry_run_trades == 2  # Initial entry + reversal (HOLD doesn't increment)
        assert _mock_place.call_count == 0  # No actual orders

        # Verify all decisions were persisted
        assert mock_store.write_signal.call_count == 3

        # Check persisted decision types
        calls = mock_store.write_signal.call_args_list
        signal_types = [call.kwargs["signal_type"] for call in calls]
        assert signal_types == ["BUY", "HOLD", "SELL"]

        # Verify metrics were updated despite dry run
        for call in calls:
            assert "confidence" in call.kwargs["risk_metrics"]
            assert "prediction" in call.kwargs["risk_metrics"]
            assert "action" in call.kwargs["execution_params"]

        # Stop strategy and verify flush
        strategy.on_stop()
        mock_store.flush.assert_called_once()

        print("\n✅ Dry run integration test passed!")
        print("Strategy successfully processed signals, made decisions,")
        print("persisted to stores, and updated metrics without placing orders.")


if __name__ == "__main__":
    test_dry_run_production_scenario()
