#!/usr/bin/env python
"""
Entrypoint for ML Signal Actor container.

Cold-path orchestration for the ML Signal Actor process. Exposes minimal
HTTP endpoints for health and metrics and enforces ONNX-only model artifacts
for security compliance.
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, cast

from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.base import MLFeatureConfig
from ml.core.integration import MLIntegrationManager
from ml.deployment.metrics_http import build_app
from ml.deployment.security import assert_allowed_model_path
from ml.observability.bootstrap import auto_start_if_configured
from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode


class MLSignalActorNode:
    """
    Container-ready ML Signal Actor node.
    """

    def __init__(self) -> None:
        self.node: TradingNode | None = None
        self.running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._healthy: bool = False

    def setup(self) -> None:
        """
        Set up the trading node with ML Signal Actor.
        """
        # Get configuration from environment
        db_connection = os.getenv(
            "DB_CONNECTION",
            # Default to in-network Postgres host for containers; override via env for host use
            "postgresql://postgres:postgres@postgres:5432/nautilus",
        )
        databento_api_key = os.getenv("DATABENTO_API_KEY")

        model_path = os.getenv("MODEL_PATH", "/app/models/model.onnx")
        instrument_str = os.getenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        bar_type_str = os.getenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE")
        actor_id = os.getenv("ACTOR_ID", "MLSignalActor-001")
        use_dummy_stores = os.getenv("USE_DUMMY_STORES", "false").lower() == "true"

        # Parse identifiers
        instrument_id = InstrumentId.from_str(instrument_str)
        bar_type = BarType.from_str(bar_type_str)

        # Check for API key
        if not databento_api_key:
            print("ERROR: DATABENTO_API_KEY environment variable not set")
            print("Please set your Databento API key to connect to market data")
            sys.exit(1)

        # Enforce ONNX-only and existence
        try:
            assert_allowed_model_path(model_path)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        if not Path(model_path).exists():
            print(f"ERROR: Model not found at {model_path}")
            print("Provide a valid ONNX model path.")
            sys.exit(1)

        print("=" * 80)
        print("ML SIGNAL ACTOR - CONTAINER MODE")
        print("=" * 80)
        print(f"Database: {db_connection.split('@')[1] if '@' in db_connection else 'local'}")
        print(f"Model: {model_path}")
        print(f"Instrument: {instrument_id}")
        print(f"Bar Type: {bar_type}")
        print(f"Actor ID: {actor_id}")
        print(f"Dummy Stores: {use_dummy_stores}")
        print("=" * 80)

        # Feature configuration
        feature_config = MLFeatureConfig(
            lookback_window=20,
            indicators={
                "sma": {"period": 10},
                "rsi": {"period": 14},
                "bbands": {"period": 20, "std": 2},
            },
            normalize_features=True,
            fill_missing_with=0.0,
        )

        # Actor configuration
        actor_kwargs = {
            "model_id": actor_id.replace("Actor", "Model"),
            "component_id": actor_id,
            "model_path": model_path,
            "bar_type": bar_type,
            "instrument_id": instrument_id,
            "prediction_threshold": 0.5,
            "max_inference_latency_ms": 5.0,
            "feature_config": feature_config,
            "warm_up_period": 20,
            "publish_signals": True,
            "log_predictions": True,
            "use_dummy_stores": use_dummy_stores,
            "enable_health_monitoring": True,
        }
        if not use_dummy_stores:
            actor_kwargs["db_connection"] = db_connection
        actor_config = MLSignalActorConfig(**actor_kwargs)

        # Databento data client configuration
        data_config = DatabentoDataClientConfig(
            api_key=databento_api_key,
            http_gateway="https://hist.databento.com",
            live_gateway="wss://stream.databento.com",
        )

        # Trading node configuration
        node_config = TradingNodeConfig(
            trader_id=TraderId("ML-ACTOR-001"),
            data_clients={
                "DATABENTO": data_config,
            },
            exec_clients={},  # No execution for signal actor
        )

        # Create trading node
        self.node = TradingNode(config=node_config)

        # Add ML Signal Actor
        actor = MLSignalActor(config=actor_config)
        self.node.trader.add_actor(actor)

        # Subscribe to market data
        actor.subscribe_bars(bar_type)

        print("\nML Signal Actor configured and ready")
        print("Waiting for market data...")
        # Mark healthy after successful setup
        self._healthy = True

    def _create_dummy_model(self, model_path: str) -> None:  # pragma: no cover - deprecated
        raise RuntimeError("Dummy pickle models are no longer supported.")

    async def run(self) -> None:
        """
        Run the actor node.
        """
        self.running = True

        # Set up graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):

            def _handler(sig_local: signal.Signals = sig) -> None:
                task = asyncio.create_task(self.shutdown(sig_local))
                self._tasks.append(task)

            loop.add_signal_handler(sig, _handler)

        # Run the node
        try:
            if self.node is None:
                raise RuntimeError("Trading node not initialized")
            await self.node.run_async()
        except Exception as e:
            print(f"Error running node: {e}")
            await self.shutdown()

    async def shutdown(self, sig: signal.Signals | None = None) -> None:
        """
        Gracefully shutdown the node.
        """
        if sig:
            print(f"\nReceived signal {sig.name}, shutting down...")
        else:
            print("\nShutting down...")

        self.running = False
        self._healthy = False

        if self.node:
            await self.node.stop_async()
            # Dispose may not exist in some versions; guard at runtime
            node_any = cast(Any, self.node)
            if hasattr(node_any, "dispose_async"):
                await node_any.dispose_async()

        print("ML Signal Actor shutdown complete")


def main() -> None:
    """
    Run entry point.
    """
    configure_logging()
    run_id: str = f"actor_{uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=run_id, component="ml.entrypoint_actor")
    # Create and run the actor node
    actor_node = MLSignalActorNode()
    actor_node.setup()

    # Start lightweight HTTP endpoints in background
    try:
        port = int(os.getenv("METRICS_PORT", "8000"))
    except ValueError:
        port = 8000
    app = build_app(lambda: actor_node._healthy)
    http_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False),  # noqa: S104
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

    # Run async event loop
    try:
        asyncio.run(actor_node.run())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
