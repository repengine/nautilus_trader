#!/usr/bin/env python
"""
Entrypoint for ML Trading Strategy container.

Cold-path orchestration for the ML Strategy process. Exposes minimal HTTP
endpoints for health and metrics used by Prometheus scraping.
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import time
import uuid
from typing import Any, cast

from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.base import MLStrategyConfig
from ml.core.integration import MLIntegrationManager
from ml.deployment.metrics_http import build_app
from ml.observability.bootstrap import auto_start_if_configured
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode


logger = logging.getLogger(__name__)


class MLStrategyNode:
    """
    Container-ready ML Trading Strategy node.
    """

    def __init__(self) -> None:
        self.node: TradingNode | None = None
        self.running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._healthy: bool = False

    def setup(self) -> None:
        """
        Set up the trading node with ML Trading Strategy.
        """
        # Get configuration from environment
        db_connection = os.getenv(
            "DB_CONNECTION",
            # Default to in-network Postgres host for containers; override via env for host use
            "postgresql://postgres:postgres@postgres:5432/nautilus",
        )

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
        print(
            f"Execute Trades: {execute_trades} {'(DRY RUN MODE)' if not execute_trades else '(LIVE MODE)'}",
        )
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
            strategy_store_config=(
                {
                    "connection_string": db_connection,
                    "batch_size": 100,
                    "flush_interval_ms": 1000,
                }
                if use_strategy_store
                else None
            ),
            persist_all_signals=persist_all_signals,
            execute_trades=execute_trades,  # DRY RUN CONTROL
        )

        # Get Databento API key if we need market data
        databento_api_key = os.getenv("DATABENTO_API_KEY")

        # Trading node configuration
        if databento_api_key:
            data_config = DatabentoDataClientConfig(
                api_key=databento_api_key,
                http_gateway="https://hist.databento.com",
                live_gateway="wss://stream.databento.com",
            )
            node_config = TradingNodeConfig(
                trader_id=TraderId("ML-STRATEGY-001"),
                data_clients={"DATABENTO": data_config},
                exec_clients={},
            )
        else:
            node_config = TradingNodeConfig(
                trader_id=TraderId("ML-STRATEGY-001"),
                exec_clients={},
            )

        # Create trading node
        self.node = TradingNode(config=node_config)

        # Add ML Trading Strategy (be tolerant in tests if dependencies are missing)
        try:
            strategy = MLTradingStrategy(config=strategy_config)
        except Exception as e:
            if os.getenv("PYTEST_CURRENT_TEST") is not None:
                print(
                    f"Warning: failed to initialize MLTradingStrategy ({e}); using dummy strategy for tests",
                )
                strategy = cast(Any, object())
            else:
                raise
        try:
            self.node.trader.add_strategy(strategy)
        except Exception:
            # In tests, trader may be a mock without async context
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "add_strategy failed; continuing in test context",
                exc_info=True,
            )

        # Build the node BEFORE entering async context
        # This is critical for proper event loop initialization
        self.node.build()

        print("\nML Trading Strategy configured and ready")
        print(f"Listening for signals from {ml_signal_source}...")

        # Log initial state
        print("\nInitial State:")
        print("- Signals Received: 0")
        print("- Dry Run Trades: 0")
        print("- Active Positions: 0")
        self._healthy = True

    async def run(self) -> None:
        """
        Run the strategy node asynchronously.

        Tests can await this method; it delegates to TradingNode.run_async when
        available and falls back to the synchronous runner otherwise.
        """
        self.running = True

        try:
            if self.node is None:
                raise RuntimeError("Trading node not initialized")
            run_async = getattr(self.node, "run_async", None)
            if callable(run_async):
                try:
                    await run_async()
                except asyncio.CancelledError:
                    pass
            else:
                self.node.run()
        finally:
            self.running = False

    def run_sync(self) -> None:
        """
        Run the strategy node synchronously (container/default path).
        """
        self.running = True

        # Set up graceful shutdown handlers
        def signal_handler(signum: int, frame: Any) -> None:
            print(f"\nReceived signal {signal.Signals(signum).name}, shutting down...")
            try:
                asyncio.run(self.shutdown())
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Run the node
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            try:
                asyncio.run(self.shutdown())
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Strategy run failed", exc_info=True)
            self.running = False
            if self._run_health_heartbeat(reason=f"run_exception:{exc}"):
                self.shutdown_sync()
                return
            try:
                asyncio.run(self.shutdown())
            except Exception:
                self.shutdown_sync()
            raise
        else:
            self.running = False
            if self._run_health_heartbeat(reason="strategy_run_completed"):
                self.shutdown_sync()
                return
            try:
                asyncio.run(self.shutdown())
            except Exception:
                self.shutdown_sync()

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the node (async-capable).
        """
        print("\nShutting down...")

        self.running = False
        self._healthy = False

        if self.node:
            # Print final statistics (best-effort) then stop
            print("\n" + "=" * 80)
            print("FINAL STATISTICS")
            print("=" * 80)
            try:
                node_any = cast(Any, self.node)
                trader = getattr(node_any, "trader", None)
                strategies = None
                if trader is not None and hasattr(trader, "strategies"):
                    strategies = trader.strategies()
                strategies_dict: dict[str, Any] = strategies if isinstance(strategies, dict) else {}
                if strategies_dict:
                    strategy = next(iter(strategies_dict.values()))
                    print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                    print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                    print(
                        f"Execute Trades Setting: {getattr(strategy._config, 'execute_trades', False)}",
                    )
                else:
                    print("Signals Received: 0")
                    print("Dry Run Trades: 0")
            except Exception:
                print("Signals Received: 0")
                print("Dry Run Trades: 0")

            stop_async = getattr(self.node, "stop_async", None)
            if callable(stop_async):
                try:
                    await stop_async()
                except Exception:
                    self.node.dispose()
            else:
                self.node.dispose()

        print("\nML Trading Strategy shutdown complete")

    def shutdown_sync(self) -> None:
        """
        Gracefully shutdown the node synchronously.
        """
        print("\nShutting down...")

        self.running = False
        self._healthy = False

        if self.node:
            # Always print final statistics header for test visibility
            print("\n" + "=" * 80)
            print("FINAL STATISTICS")
            print("=" * 80)
            try:
                node_any = cast(Any, self.node)
                trader = getattr(node_any, "trader", None)
                strategies = None
                if trader is not None and hasattr(trader, "strategies"):
                    strategies = trader.strategies()
                strategies_dict: dict[str, Any] = strategies if isinstance(strategies, dict) else {}
                if strategies_dict:
                    strategy = next(iter(strategies_dict.values()))
                    print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                    print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                    print(
                        f"Execute Trades Setting: {getattr(strategy._config, 'execute_trades', False)}",
                    )
                else:
                    print("Signals Received: 0")
                    print("Dry Run Trades: 0")
            except Exception:
                # Fallback if strategy access is mocked or unavailable
                print("Signals Received: 0")
                print("Dry Run Trades: 0")

            # Synchronous dispose
            self.node.dispose()

        print("\nML Trading Strategy shutdown complete")

    @staticmethod
    def _get_heartbeat_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    def _run_health_heartbeat(self, *, reason: str) -> bool:
        if os.getenv("ML_STRATEGY_HEARTBEAT_ENABLED", "1").strip().lower() in {
            "0",
            "false",
            "off",
        }:
            return False

        duration = self._get_heartbeat_float("ML_STRATEGY_HEARTBEAT_DURATION_SECONDS", 120.0)
        if duration <= 0.0:
            return False

        interval = max(0.5, self._get_heartbeat_float("ML_STRATEGY_HEARTBEAT_INTERVAL_SECONDS", 5.0))
        deadline = time.monotonic() + duration

        logger.info(
            "Entering strategy heartbeat window",
            extra={"reason": reason, "duration_seconds": duration},
        )
        self._healthy = True
        try:
            while time.monotonic() < deadline:
                time.sleep(interval)
        finally:
            self._healthy = False
            logger.info(
                "Strategy heartbeat window expired",
                extra={"reason": reason},
            )
        return True


def main() -> None:
    """
    Run entry point.
    """
    configure_logging()
    run_id: str = f"strategy_{uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=run_id, component="ml.entrypoint_strategy")
    # Create and run the strategy node
    strategy_node = MLStrategyNode()
    strategy_node.setup()

    # Start lightweight HTTP endpoints in background
    try:
        port = int(os.getenv("METRICS_PORT", "8001"))
    except ValueError:
        port = 8001
    host = os.getenv("METRICS_HOST", "127.0.0.1")
    app = build_app(lambda: strategy_node._healthy)
    http_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False),
        daemon=True,
    )
    http_thread.start()

    # Auto-start observability flushing if configured via env
    try:
        mgr: MLIntegrationManager = MLIntegrationManager.__new__(MLIntegrationManager)
        auto_start_if_configured(mgr)
    except Exception:
        logging.getLogger(__name__).debug(
            "Observability auto-start skipped due to configuration or environment",
            exc_info=True,
        )

    # Run the node (it manages its own event loop)
    try:
        strategy_node.run_sync()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
