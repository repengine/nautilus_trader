#!/usr/bin/env python
"""
Run ML Trading System locally in dry run mode with real components.

This script runs the ML actor and strategy locally (not in containers) but connects to
real PostgreSQL and Databento data feed.

"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, cast


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

from ml.actors import MLSignalActor
from ml.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.adapters.databento.factories import DatabentoLiveDataClientFactory
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode


class LocalDryRunSystem:
    """
    Run ML trading system locally with real data feed.
    """

    def __init__(self) -> None:
        self.node: TradingNode | None = None
        self.databento_key: str | None = None
        self.db_connection: str = "postgresql://postgres:postgres@localhost:5432/nautilus"

    def check_prerequisites(self) -> bool:
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
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        )

        # Test PostgreSQL connection
        try:
            import psycopg2

            conn_params = self._parse_connection_string(self.db_connection)
            conn = psycopg2.connect(**conn_params)
            conn.close()
            print(
                f"✓ PostgreSQL connected: {conn_params['host']}:{conn_params['port']}/{conn_params['database']}",
            )
        except Exception as e:
            print(f"⚠ PostgreSQL not available: {e}")
            print("  Will use SQLite fallback for persistence")
            self.db_connection = "sqlite:///ml_dry_run.db"

        # Check model file (ONNX only)
        model_path = Path("ml/models/dummy_bullish_model.onnx")
        if not model_path.exists():
            print("⚠ ONNX model not found at ml/models/dummy_bullish_model.onnx")
            print("  Please export or place an ONNX model at this path before running.")
        else:
            print(f"✓ Model found: {model_path}")

        return True

    def _parse_connection_string(self, conn_str: str) -> dict[str, object]:
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

    # Removed insecure pickle-based dummy model creation; require ONNX model instead.

    async def setup_and_run(self) -> None:
        """
        Set up and run the trading system.
        """
        print("\n" + "=" * 80)
        print("ML TRADING SYSTEM - LOCAL DRY RUN WITH REAL DATA")
        print("=" * 80)

        # Configuration for US equities
        # Using SPY (S&P 500 ETF) as our test instrument
        instrument_id = InstrumentId.from_str("SPY.XNAS")  # SPY on NASDAQ
        bar_type = BarType.from_str(
            "SPY.XNAS-1-MINUTE-LAST-EXTERNAL",
        )  # 1-minute bars with EXTERNAL suffix

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
        actor_kwargs = {
            "model_id": "dummy_bullish",
            "component_id": "MLSignalActor-LOCAL",
            # Expect an ONNX model at this path for dry run
            "model_path": "ml/models/dummy_bullish_model.onnx",
            "bar_type": bar_type,
            "instrument_id": instrument_id,
            "prediction_threshold": 0.5,
            "feature_config": feature_config,
            "warm_up_period": 20,
            "publish_signals": True,
            "log_predictions": True,
            "use_dummy_stores": use_dummy_stores,
        }
        if not use_dummy_stores:
            actor_kwargs["db_connection"] = self.db_connection
        actor_config = MLSignalActorConfig(**actor_kwargs)

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
            strategy_store_config=(
                {
                    "connection_string": self.db_connection,
                    "batch_size": 100,
                    "flush_interval_ms": 1000,
                }
                if not use_dummy_stores
                else None
            ),
            persist_all_signals=True,
            execute_trades=False,  # DRY RUN MODE
        )

        print(f"Instrument: {instrument_id}")
        print(f"Bar Type: {bar_type}")
        print(
            f"Database: {self.db_connection.split('@')[1] if '@' in self.db_connection else 'SQLite'}",
        )
        print("Mode: DRY RUN (execute_trades=False)")
        print("=" * 80)

        # Databento configuration for US equities
        data_config = DatabentoDataClientConfig(
            api_key=self.databento_key,
            http_gateway="https://hist.databento.com",
            live_gateway="wss://stream.databento.com",
            venue_dataset_map={"XNAS": "EQUS.MINI"},  # Use EQUS.MINI for live data
        )

        # Trading node configuration
        node_config = TradingNodeConfig(
            trader_id=TraderId("ML-LOCAL-001"),
            data_clients={
                "DATABENTO": data_config,
            },
            exec_clients={},  # No execution for dry run
        )

        # Create trading node
        self.node = TradingNode(config=node_config)

        # Register Databento factory
        self.node.add_data_client_factory("DATABENTO", DatabentoLiveDataClientFactory)

        # Build the node first
        self.node.build()

        # Add components
        actor = MLSignalActor(config=actor_config)
        strategy = MLTradingStrategy(config=strategy_config)

        self.node.trader.add_actor(actor)
        self.node.trader.add_strategy(strategy)

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

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the system.
        """
        if self.node:
            # Get statistics
            from typing import Any as _Any

            strategies = self.node.trader.strategies()
            strategies_dict: dict[str, Any] = strategies if isinstance(strategies, dict) else {}
            for strategy in strategies_dict.values():
                print("\n" + "=" * 80)
                print("FINAL STATISTICS")
                print("=" * 80)
                print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                print(f"Mode: {'DRY RUN' if not strategy._config.execute_trades else 'LIVE'}")

            await self.node.stop_async()
            node_any = cast(_Any, self.node)
            if hasattr(node_any, "dispose_async"):
                await node_any.dispose_async()

        print("\nShutdown complete")


async def main() -> None:
    """
    Run entry point.
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
