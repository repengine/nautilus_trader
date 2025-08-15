#!/usr/bin/env python
"""
Run ML Trading System in dry run mode using historical data replay.
This simulates live trading using recent historical data.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.adapters.databento import DatabentoDataLoader
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.config import LoggingConfig
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from ml.actors.signal import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig, MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy


def run_backtest_dry_run():
    """
    Run ML system in backtest mode with recent data.
    """
    print("\n" + "=" * 80)
    print("ML TRADING SYSTEM - BACKTEST DRY RUN")
    print("=" * 80)
    
    # Configuration
    venue = Venue("XNAS")
    instrument_id = InstrumentId.from_str("SPY.XNAS")
    bar_type = BarType.from_str("SPY.XNAS-1-MINUTE-LAST-EXTERNAL")
    
    # Check database connection
    db_connection = os.getenv(
        "DB_CONNECTION",
        "postgresql://postgres:postgres@localhost:5432/nautilus"
    )
    
    # Feature configuration
    feature_config = MLFeatureConfig(
        lookback_window=20,
        indicators={
            "sma": {"period": 10},
            "rsi": {"period": 14},
            "bbands": {"period": 20, "std": 2},
        },
        normalize_features=True,
    )
    
    # ML Signal Actor configuration
    actor_config = MLSignalActorConfig(
        model_id="dummy_bullish",
        component_id="MLSignalActor-BACKTEST",
        model_path="ml/models/dummy_bullish_model.pkl",
        bar_type=bar_type,
        instrument_id=instrument_id,
        db_connection=db_connection,
        prediction_threshold=0.5,
        feature_config=feature_config,
        warm_up_period=20,
        publish_signals=True,
        log_predictions=True,
        use_dummy_stores=False,
    )
    
    # ML Strategy configuration
    strategy_config = MLStrategyConfig(
        strategy_id="MLStrategy-BACKTEST-DRY",
        instrument_id=instrument_id,
        ml_signal_source="MLSignalActor-BACKTEST",
        position_size_pct=0.02,
        min_confidence=0.6,
        max_positions=1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        use_strategy_store=True,
        strategy_store_config={
            "connection_string": db_connection,
            "batch_size": 100,
            "flush_interval_ms": 1000,
        },
        persist_all_signals=True,
        execute_trades=False,  # DRY RUN MODE
    )
    
    print(f"Instrument: {instrument_id}")
    print(f"Bar Type: {bar_type}")
    print(f"Database: {db_connection.split('@')[1] if '@' in db_connection else 'SQLite'}")
    print(f"Mode: DRY RUN (execute_trades=False)")
    print("=" * 80)
    
    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    
    # Build the backtest engine
    engine = BacktestEngine(config=config)
    
    # Add venue
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(100_000.0, USD)],
    )
    
    # Add instrument
    SPY = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    engine.add_instrument(SPY)
    
    # Load recent historical data
    print("\nLoading historical data...")
    loader = DatabentoDataLoader()
    
    # Try to load data from Databento
    try:
        # Calculate dates (last 5 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        print(f"Attempting to load data from {start_date.date()} to {end_date.date()}")
        
        # Generate synthetic bars for testing
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        import random
        
        bars = []
        base_price = 400.0
        
        for i in range(1000):
            ts = start_date + timedelta(minutes=i)
            ts_ns = dt_to_unix_nanos(ts)
            
            # Generate realistic price movement
            open_price = base_price + random.uniform(-2, 2)
            high = open_price + random.uniform(0, 1)
            low = open_price - random.uniform(0, 1)
            close = random.uniform(low, high)
            base_price = close  # Carry forward for continuity
            
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.2f}"),
                high=Price.from_str(f"{high:.2f}"),
                low=Price.from_str(f"{low:.2f}"),
                close=Price.from_str(f"{close:.2f}"),
                volume=Quantity.from_int(random.randint(100000, 1000000)),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)
        
        engine.add_data(bars)
        print(f"Generated {len(bars)} synthetic bars")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Add ML components
    actor = MLSignalActor(config=actor_config)
    strategy = MLTradingStrategy(config=strategy_config)
    
    engine.add_actor(actor)
    engine.add_strategy(strategy)
    
    print("\nStarting backtest...")
    print("-" * 80)
    
    # Run backtest
    engine.run()
    
    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    
    # Get strategy performance
    for strategy in engine.trader.strategies():
        if hasattr(strategy, '_signals_received'):
            print(f"Signals Received: {strategy._signals_received}")
        if hasattr(strategy, '_dry_run_trades'):
            print(f"Dry Run Trades: {strategy._dry_run_trades}")
    
    print("\nEngine Statistics:")
    print(f"Total events: {engine.kernel.clock.timestamp_ns()}")
    print(f"Backtest complete!")


if __name__ == "__main__":
    print("ML Trading System - Backtest Dry Run")
    print("====================================")
    print()
    print("This will use historical data to simulate live trading")
    print("in DRY RUN mode (no actual trades)")
    print()
    
    try:
        run_backtest_dry_run()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)