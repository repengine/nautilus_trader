#!/usr/bin/env python
"""
Entrypoint for ML Trading Strategy container.

This script runs the ML Trading Strategy that consumes signals
from the ML Signal Actor and makes trading decisions (in dry run mode).
"""

import os
import sys
import asyncio
import signal
from decimal import Decimal

from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId, TraderId

from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy


class MLStrategyNode:
    """
    Container-ready ML Trading Strategy node.
    """
    
    def __init__(self):
        self.node = None
        self.running = False
        
    def setup(self):
        """
        Set up the trading node with ML Trading Strategy.
        """
        # Get configuration from environment
        db_connection = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@localhost:5432/nautilus")
        
        strategy_id = os.getenv("STRATEGY_ID", "MLStrategy-DRY-001")
        ml_signal_source = os.getenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
        instrument_str = os.getenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        
        # DRY RUN MODE
        execute_trades = os.getenv("EXECUTE_TRADES", "false").lower() == "true"
        
        # Risk parameters
        position_size_pct = float(os.getenv("POSITION_SIZE_PCT", "0.02"))
        min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.6"))
        max_positions = int(os.getenv("MAX_POSITIONS", "1"))
        stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.02"))
        take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
        
        # Persistence
        use_strategy_store = os.getenv("USE_STRATEGY_STORE", "true").lower() == "true"
        persist_all_signals = os.getenv("PERSIST_ALL_SIGNALS", "true").lower() == "true"
        
        # Parse identifiers
        instrument_id = InstrumentId.from_str(instrument_str)
        
        print("=" * 80)
        print("ML TRADING STRATEGY - CONTAINER MODE")
        print("=" * 80)
        print(f"Database: {db_connection.split('@')[1] if '@' in db_connection else 'local'}")
        print(f"Strategy ID: {strategy_id}")
        print(f"Signal Source: {ml_signal_source}")
        print(f"Instrument: {instrument_id}")
        print(f"Execute Trades: {execute_trades} {'(DRY RUN MODE)' if not execute_trades else '(LIVE MODE)'}")
        print(f"Position Size: {position_size_pct*100:.1f}%")
        print(f"Min Confidence: {min_confidence:.2f}")
        print(f"Stop Loss: {stop_loss_pct*100:.1f}%")
        print(f"Take Profit: {take_profit_pct*100:.1f}%")
        print("=" * 80)
        
        if not execute_trades:
            print("\n⚠️  DRY RUN MODE ACTIVE ⚠️")
            print("Strategy will process signals and make decisions")
            print("but will NOT submit actual orders to the exchange")
            print("=" * 80)
        
        # Strategy configuration
        strategy_config = MLStrategyConfig(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            ml_signal_source=ml_signal_source,
            position_size_pct=position_size_pct,
            min_confidence=min_confidence,
            max_positions=max_positions,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_strategy_store=use_strategy_store,
            strategy_store_config={
                "connection_string": db_connection,
                "batch_size": 100,
                "flush_interval_ms": 1000,
            } if use_strategy_store else None,
            persist_all_signals=persist_all_signals,
            execute_trades=execute_trades,  # DRY RUN CONTROL
        )
        
        # Get Databento API key if we need market data
        databento_api_key = os.getenv("DATABENTO_API_KEY")
        
        # Trading node configuration
        node_config_dict = {
            "trader_id": TraderId("ML-STRATEGY-001"),
            "exec_clients": {},  # No execution in dry run
            "default_data_client": None,
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
            },
            "timeout_connection": 30.0,
            "timeout_reconciliation": 10.0,
            "timeout_portfolio": 10.0,
            "timeout_disconnection": 10.0,
        }
        
        # Add data client if we have API key (for market data subscription)
        if databento_api_key:
            data_config = DatabentoDataClientConfig(
                api_key=databento_api_key,
                http_gateway="https://hist.databento.com",
                streaming_gateway="wss://stream.databento.com",
                historical_gateway="https://hist.databento.com",
                dataset=os.getenv("DATABENTO_DATASET", "GLBX.MDP3"),
            )
            node_config_dict["data_clients"] = {"DATABENTO": data_config}
            node_config_dict["default_data_client"] = "DATABENTO"
        
        node_config = TradingNodeConfig(**node_config_dict)
        
        # Create trading node
        self.node = TradingNode(config=node_config)
        
        # Add ML Trading Strategy
        strategy = MLTradingStrategy(config=strategy_config)
        self.node.add_strategy(strategy)
        
        print("\nML Trading Strategy configured and ready")
        print(f"Listening for signals from {ml_signal_source}...")
        
        # Log initial state
        print(f"\nInitial State:")
        print(f"- Signals Received: 0")
        print(f"- Dry Run Trades: 0")
        print(f"- Active Positions: 0")
        
    async def run(self):
        """
        Run the strategy node.
        """
        self.running = True
        
        # Set up graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
        
        # Run the node
        try:
            await self.node.run_async()
        except Exception as e:
            print(f"Error running node: {e}")
            await self.shutdown()
    
    async def shutdown(self, sig=None):
        """
        Gracefully shutdown the node.
        """
        if sig:
            print(f"\nReceived signal {sig.name}, shutting down...")
        else:
            print("\nShutting down...")
        
        self.running = False
        
        if self.node:
            # Get final statistics
            strategies = self.node.trader.strategies()
            if strategies:
                strategy = list(strategies.values())[0]
                print("\n" + "=" * 80)
                print("FINAL STATISTICS")
                print("=" * 80)
                print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                print(f"Execute Trades Setting: {getattr(strategy._config, 'execute_trades', False)}")
                
            await self.node.stop_async()
            await self.node.dispose_async()
        
        print("\nML Trading Strategy shutdown complete")


def main():
    """
    Main entry point.
    """
    # Create and run the strategy node
    strategy_node = MLStrategyNode()
    strategy_node.setup()
    
    # Run async event loop
    try:
        asyncio.run(strategy_node.run())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()