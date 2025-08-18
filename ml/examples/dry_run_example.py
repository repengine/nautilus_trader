"""
Complete dry run example for ML trading system.

This script demonstrates how to set up and run the ML trading pipeline
in dry run mode without requiring:
- Real broker connection
- Trained production models
- PostgreSQL database

What it DOES require:
- Nautilus Trader installation
- ML module components
- Market data (can use historical data)
"""

from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import BacktestVenueConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money


def create_dry_run_config() -> dict[str, Any] | None:
    """
    Create configuration for dry run testing.
    """
    # First, ensure we have a dummy model
    models_dir = Path("ml/models")
    model_path = models_dir / "dummy_bullish_model.pkl"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run: python ml/examples/create_dummy_model.py")
        return None

    # Instrument configuration
    instrument_id = InstrumentId.from_str("BTC-USDT.BINANCE")
    bar_type = BarType.from_str("BTC-USDT.BINANCE-1-MINUTE")

    # Feature configuration (must match model expectations)
    feature_config = MLFeatureConfig(
        lookback_window=20,
        indicators={
            "sma": {"period": 10},
            "rsi": {"period": 14},
        },
        feature_names=[f"feature_{i}" for i in range(10)],  # Match dummy model
        normalize_features=True,
        fill_missing_with=0.0,
    )

    # ML Signal Actor configuration
    actor_config = MLSignalActorConfig(
        model_id="dummy_bullish",
        component_id="MLSignalActor-001",
        model_path=str(model_path),
        bar_type=bar_type,
        instrument_id=instrument_id,
        prediction_threshold=0.5,
        max_inference_latency_ms=5.0,
        feature_config=feature_config,
        warm_up_period=20,
        publish_signals=True,
        log_predictions=True,
        use_dummy_stores=True,  # Use dummy stores for testing
    )

    # ML Strategy configuration
    strategy_config = MLStrategyConfig(
        strategy_id="MLStrategy-DRY-RUN",
        instrument_id=instrument_id,
        ml_signal_source="MLSignalActor-001",
        position_size_pct=0.02,  # 2% of account per trade
        min_confidence=0.6,
        max_positions=1,
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.04,  # 4% take profit
        use_strategy_store=True,
        persist_all_signals=True,
        execute_trades=False,  # DRY RUN MODE - No actual trades!
    )

    return {
        "actor_config": actor_config,
        "strategy_config": strategy_config,
        "instrument_id": instrument_id,
        "bar_type": bar_type,
    }


def setup_backtest_engine(configs: dict[str, Any]) -> BacktestEngine:
    """
    Set up backtest engine for dry run testing.

    We use the backtest engine to simulate market conditions
    and test our ML pipeline without live market connection.
    """
    # Engine configuration
    engine_config = BacktestEngineConfig()

    # Create engine
    engine = BacktestEngine(config=engine_config)

    # Add venue
    venue_config = BacktestVenueConfig(
        name="BINANCE",
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(100_000, USD)],
        base_currency=USD,
        fill_model=FillModel(),
        latency_model=LatencyModel(0),
    )
    engine.add_venue(venue_config)

    # Add instrument (you'd normally load this from catalog)
    # For now, we'll use a simple stub
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    provider = TestInstrumentProvider()
    btcusdt = provider.btcusdt_binance()
    engine.add_instrument(btcusdt)

    # Add actor and strategy
    engine.add_actor(MLSignalActor(configs["actor_config"]))
    engine.add_strategy(MLTradingStrategy(configs["strategy_config"]))

    return engine


def run_dry_run_backtest() -> None:
    """
    Run a dry run using historical data in backtest mode.

    This allows testing the complete ML pipeline without:
    - Live market connection
    - Real broker
    - Real money risk
    """
    print("=" * 80)
    print("ML TRADING SYSTEM - DRY RUN MODE")
    print("=" * 80)

    # Create configuration
    configs = create_dry_run_config()
    if not configs:
        return

    print("\nConfiguration:")
    print(f"- Model: {configs['actor_config'].model_path}")
    print(f"- Instrument: {configs['instrument_id']}")
    print("- Strategy Mode: DRY RUN (execute_trades=False)")
    print("- Using Dummy Stores: True")

    # Set up engine
    print("\nSetting up backtest engine...")
    engine = setup_backtest_engine(configs)

    # Load or generate sample data
    print("\nGenerating sample market data...")
    # In production, you would load real historical data
    # For this example, we'll generate synthetic data
    from nautilus_trader.test_kit.providers import TestDataProvider
    provider = TestDataProvider()

    # Generate sample bars
    bars = cast(Any, provider).generate_bar_data(
        instrument_id=configs["instrument_id"],
        bar_type=configs["bar_type"],
        start_time=datetime.utcnow() - timedelta(hours=2),
        end_time=datetime.utcnow(),
        frequency=timedelta(minutes=1),
    )

    # Add data to engine
    engine.add_data(bars)

    # Run backtest
    print("\nRunning dry run backtest...")
    print("-" * 40)

    engine.run()

    # Print results
    print("\n" + "=" * 80)
    print("DRY RUN COMPLETE")
    print("=" * 80)

    # Get strategy performance
    strategy = engine.actors.get("MLStrategy-DRY-RUN")
    if strategy:
        print("\nStrategy Statistics:")
        print(f"- Signals Received: {strategy._signals_received}")
        print(f"- Dry Run Trades: {strategy._dry_run_trades}")
        print(f"- Execute Trades Flag: {strategy._config.execute_trades}")
        print("\nNote: No actual orders were placed (dry run mode)")

    print("\n" + "=" * 80)
    print("NEXT STEPS FOR PRODUCTION:")
    print("=" * 80)
    print("1. Train a real model with your data")
    print("2. Set up PostgreSQL for persistence (or continue with dummy stores)")
    print("3. Connect to live/paper market data feed")
    print("4. Configure proper risk management parameters")
    print("5. Set execute_trades=True when ready for live trading")
    print("6. Monitor metrics via Prometheus/Grafana")


def run_dry_run_live() -> None:
    """
    Run a dry run with live market data connection.

    This requires setting up a live data feed but still
    operates in dry run mode (no actual trades).
    """
    print("=" * 80)
    print("ML TRADING SYSTEM - LIVE DRY RUN MODE")
    print("=" * 80)

    # Create configuration
    configs = create_dry_run_config()
    if not configs:
        return

    print("\nConfiguration:")
    print(f"- Model: {configs['actor_config'].model_path}")
    print(f"- Instrument: {configs['instrument_id']}")
    print("- Strategy Mode: DRY RUN (execute_trades=False)")
    print("- Using Dummy Stores: True")

    # Here you would set up live trading node
    # with real market data connection
    # For example:
    """
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.config import TradingNodeConfig

    node_config = TradingNodeConfig(
        trader_id="TRADER-001",
        data_clients={
            "BINANCE": BinanceDataClientConfig(...),
        },
        exec_clients={},  # No execution client for dry run
        ...
    )

    node = TradingNode(config=node_config)
    node.add_actor(MLSignalActor(configs["actor_config"]))
    node.add_strategy(MLTradingStrategy(configs["strategy_config"]))
    node.run()
    """

    print("\nLive dry run requires market data connection setup.")
    print("See Nautilus Trader documentation for data client configuration.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        run_dry_run_live()
    else:
        run_dry_run_backtest()
