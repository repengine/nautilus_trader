#!/usr/bin/env python
"""
Run ML Trading System locally in dry run mode with real components.

This script runs the ML actor and strategy locally (not in containers)
but connects to real PostgreSQL and Databento data feed.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.adapters.databento.factories import DatabentoLiveDataClientFactory
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Money

from ml.actors.signal import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig, MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy


class LocalDryRunSystem:
    """
    Run ML trading system locally with real data feed.
    """
    
    def __init__(self):
        self.node = None
        self.databento_key = None
        self.db_connection = None
        
    def check_prerequisites(self):
        """
        Check that all prerequisites are met.
        """
        print("Checking prerequisites...")
        
        # Check Databento API key
        self.databento_key = os.getenv("DATABENTO_API_KEY")
        if not self.databento_key:
            print("ERROR: DATABENTO_API_KEY not set")
            print("Please run: export DATABENTO_API_KEY=your_key_here")
            return False
        print("✓ Databento API key found")
        
        # Check PostgreSQL connection
        self.db_connection = os.getenv(
            "DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus"
        )
        
        # Test PostgreSQL connection
        try:
            import psycopg2
            conn_params = self._parse_connection_string(self.db_connection)
            conn = psycopg2.connect(**conn_params)
            conn.close()
            print(f"✓ PostgreSQL connected: {conn_params['host']}:{conn_params['port']}/{conn_params['database']}")
        except Exception as e:
            print(f"⚠ PostgreSQL not available: {e}")
            print("  Will use SQLite fallback for persistence")
            self.db_connection = "sqlite:///ml_dry_run.db"
        
        # Check model file
        model_path = Path("ml/models/dummy_bullish_model.pkl")
        if not model_path.exists():
            print("Creating dummy model...")
            self._create_dummy_model(str(model_path))
        print(f"✓ Model found: {model_path}")
        
        return True
    
    def _parse_connection_string(self, conn_str):
        """
        Parse PostgreSQL connection string.
        """
        # postgresql://user:password@host:port/database
        import re
        pattern = r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)"
        match = re.match(pattern, conn_str)
        if match:
            return {
                "user": match.group(1),
                "password": match.group(2),
                "host": match.group(3),
                "port": int(match.group(4)),
                "database": match.group(5),
            }
        return {}
    
    def _create_dummy_model(self, model_path: str):
        """
        Create a dummy model for testing.
        """
        import pickle
        import numpy as np
        
        class DummyModel:
            def __init__(self):
                self.feature_names = [f"feature_{i}" for i in range(10)]
                
            def predict(self, X):
                # Generate slightly bullish predictions
                base = 0.55 + np.random.randn(len(X) if len(X.shape) > 1 else 1) * 0.1
                return np.clip(base, 0, 1)
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(DummyModel(), f)
    
    async def setup_and_run(self):
        """
        Set up and run the trading system.
        """
        print("\n" + "=" * 80)
        print("ML TRADING SYSTEM - LOCAL DRY RUN WITH REAL DATA")
        print("=" * 80)
        
        # Configuration
        instrument_id = InstrumentId.from_str("ES-USD-FUT.CME")  # E-mini S&P 500
        bar_type = BarType.from_str("ES-USD-FUT.CME-1-MINUTE")
        
        # Use SQLite if PostgreSQL not available
        use_dummy_stores = "sqlite" in self.db_connection
        
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
            component_id="MLSignalActor-LOCAL",
            model_path="ml/models/dummy_bullish_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            db_connection=self.db_connection if not use_dummy_stores else None,
            prediction_threshold=0.5,
            feature_config=feature_config,
            warm_up_period=20,
            publish_signals=True,
            log_predictions=True,
            use_dummy_stores=use_dummy_stores,
        )
        
        # ML Strategy configuration
        strategy_config = MLStrategyConfig(
            strategy_id="MLStrategy-LOCAL-DRY",
            instrument_id=instrument_id,
            ml_signal_source="MLSignalActor-LOCAL",
            position_size_pct=0.02,
            min_confidence=0.6,
            max_positions=1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            use_strategy_store=not use_dummy_stores,
            strategy_store_config={
                "connection_string": self.db_connection,
                "batch_size": 100,
                "flush_interval_ms": 1000,
            } if not use_dummy_stores else None,
            persist_all_signals=True,
            execute_trades=False,  # DRY RUN MODE
        )
        
        print(f"Instrument: {instrument_id}")
        print(f"Bar Type: {bar_type}")
        print(f"Database: {self.db_connection.split('@')[1] if '@' in self.db_connection else 'SQLite'}")
        print(f"Mode: DRY RUN (execute_trades=False)")
        print("=" * 80)
        
        # Databento configuration
        data_config = DatabentoDataClientConfig(
            api_key=self.databento_key,
            http_gateway="https://hist.databento.com",
            streaming_gateway="wss://stream.databento.com",
            dataset="GLBX.MDP3",  # CME dataset
        )
        
        # Trading node configuration
        node_config = TradingNodeConfig(
            trader_id=TraderId("ML-LOCAL-001"),
            data_clients={
                "DATABENTO": data_config,
            },
            exec_clients={},  # No execution for dry run
            default_data_client="DATABENTO",
            logging={
                "level": "INFO",
            },
        )
        
        # Create trading node
        self.node = TradingNode(config=node_config)
        
        # Add components
        actor = MLSignalActor(config=actor_config)
        strategy = MLTradingStrategy(config=strategy_config)
        
        self.node.add_actor(actor)
        self.node.add_strategy(strategy)
        
        # Subscribe to market data
        actor.subscribe_bars(bar_type)
        strategy.subscribe_bars(bar_type)  # Strategy might need bars too
        
        print("\nSystem initialized. Starting data feed...")
        print("Waiting for market data...")
        print("\nPress Ctrl+C to stop\n")
        print("-" * 80)
        
        # Run the node
        try:
            await self.node.run_async()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            await self.shutdown()
    
    async def shutdown(self):
        """
        Gracefully shutdown the system.
        """
        if self.node:
            # Get statistics
            for strategy in self.node.trader.strategies().values():
                print("\n" + "=" * 80)
                print("FINAL STATISTICS")
                print("=" * 80)
                print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                print(f"Mode: {'DRY RUN' if not strategy._config.execute_trades else 'LIVE'}")
            
            await self.node.stop_async()
            await self.node.dispose_async()
        
        print("\nShutdown complete")


async def main():
    """
    Main entry point.
    """
    system = LocalDryRunSystem()
    
    if not system.check_prerequisites():
        sys.exit(1)
    
    try:
        await system.setup_and_run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("ML Trading System - Local Dry Run")
    print("==================================")
    print()
    print("This will connect to real market data (Databento)")
    print("but run in DRY RUN mode (no actual trades)")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)