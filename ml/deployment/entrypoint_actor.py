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
from typing import Any

from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig
from ml.actors.recorder import RecorderActor
from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.base import MLFeatureConfig
from ml.core.integration import MLIntegrationManager
from ml.deployment.metrics_http import build_app
from ml.deployment.security import assert_allowed_model_path
from ml.observability.bootstrap import auto_start_if_configured
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.writers import CatalogWriteFacade
from ml.stores.writers import LiveDataRecorder
from ml.stores.writers import ParquetCatalogMarketDataWriter
from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.adapters.databento.factories import DatabentoLiveDataClientFactory
from nautilus_trader.config import LiveDataEngineConfig
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
        use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

        model_path = os.getenv("MODEL_PATH", "/app/models/model.onnx")
        # Aggregated EQUS.MINI path (provider OHLCV + anonymized venue)
        instrument_str = os.getenv("INSTRUMENT_ID", "SPY.EQUS")
        # Use EXTERNAL aggregation - Databento provides OHLCV-1m bars during market hours
        bar_type_str = os.getenv("BAR_TYPE", "SPY.EQUS-1-MINUTE-LAST-EXTERNAL")
        actor_id = os.getenv("ACTOR_ID", "MLSignalActor-001")
        use_dummy_stores = os.getenv("USE_DUMMY_STORES", "false").lower() == "true"
        # Optional behavior overrides for testing/ops
        def _get_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "y"}

        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _get_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        # Parse identifiers
        instrument_id = InstrumentId.from_str(instrument_str)
        bar_type = BarType.from_str(bar_type_str)

        # Check for API key unless running with mock data
        if not databento_api_key and not use_mock_data:
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
        if use_mock_data:
            print("Mock Data: ENABLED (Databento disabled)")
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

        # Actor configuration (base kwargs shared with multi-instrument config)
        actor_kwargs: dict[str, Any] = {
            "model_id": actor_id.replace("Actor", "Model"),
            "component_id": actor_id,
            "model_path": model_path,
            "bar_type": bar_type,
            "instrument_id": instrument_id,
            "prediction_threshold": _get_float("PREDICTION_THRESHOLD", 0.5),
            "max_inference_latency_ms": _get_float("MAX_INFERENCE_LATENCY_MS", 5.0),
            "max_feature_latency_ms": _get_float("MAX_FEATURE_LATENCY_MS", 0.5),
            "feature_config": feature_config,
            "warm_up_period": _get_int("WARM_UP_PERIOD", 20),
            "publish_signals": _get_bool("PUBLISH_SIGNALS", True),
            "log_predictions": _get_bool("LOG_PREDICTIONS", True),
            "use_dummy_stores": use_dummy_stores,
            "enable_health_monitoring": True,
        }
        if not use_dummy_stores:
            actor_kwargs["db_connection"] = db_connection

        # Optional: force a permissive signal policy for testing (always emit)
        if _get_bool("FORCE_SIGNAL_MODE", False):
            try:
                from collections.abc import MutableMapping

                import numpy as _np
                from nautilus_trader.model.data import Bar as _Bar

                from ml.actors.base import MLSignal as _MLSignal

                class _AlwaysSignalStrategy:  # minimal protocol-compatible
                    def generate_signal(
                        self,
                        bar: _Bar,
                        prediction: float,
                        confidence: float,
                        features: object,
                        context: MutableMapping[str, Any],
                    ) -> _MLSignal:
                        feat32 = _np.asarray(features, dtype=_np.float32)
                        return _MLSignal(
                            instrument_id=bar.bar_type.instrument_id,
                            model_id=str(context.get("model_id", "test_model")),
                            prediction=float(prediction),
                            confidence=float(confidence),
                            features=feat32 if context.get("log_predictions", False) else None,
                            ts_event=bar.ts_event,
                            ts_init=int(context.get("timestamp_ns", 0)),
                        )

                actor_kwargs["custom_strategy"] = _AlwaysSignalStrategy()
                # Remove minimum spacing between signals in test mode
                actor_kwargs["min_signal_separation_bars"] = 0
            except Exception:
                # Non-fatal; fall back to configured strategy
                pass
        # Multi-instrument extensions (batching + universe)
        def _get_list(name: str) -> list[str] | None:
            raw = os.getenv(name)
            if not raw:
                return None
            items = [t.strip() for t in raw.split(",") if t.strip()]
            return items or None

        universe: list[str] | None = (
            _get_list("ACTOR_UNIVERSE")
            or _get_list("UNIVERSE_SYMBOLS")
        )
        if not universe:
            # Default multi-instrument universe aligned with common US listings
            # ETFs use EQUS aggregated venue; equities use XNAS symbols
            universe = [
                "SPY.EQUS",
                "QQQ.EQUS",
                "AAPL.XNAS",
                "MSFT.XNAS",
                "NVDA.XNAS",
            ]

        max_batch_size = _get_int("MAX_BATCH_SIZE", 128)
        feature_dim = _get_int("FEATURE_DIM", 64)
        flush_max_latency_ms = _get_int("FLUSH_MAX_LATENCY_MS", 0)

        actor_config = MultiInstrumentSignalActorConfig(
            **actor_kwargs,
            max_batch_size=max_batch_size,
            feature_dim=feature_dim,
            initial_universe=universe,
            flush_max_latency_ms=flush_max_latency_ms,
        )

        # Trading node configuration
        data_engine_cfg: LiveDataEngineConfig = LiveDataEngineConfig(
            time_bars_timestamp_on_close=True,
            time_bars_build_with_no_updates=True,
            time_bars_skip_first_non_full_bar=True,
        )
        if use_mock_data:
            node_config: TradingNodeConfig = TradingNodeConfig(
                trader_id=TraderId("ML-ACTOR-001"),
                data_engine=data_engine_cfg,
                data_clients={},  # No live data clients in mock mode
                exec_clients={},
            )
        else:
            # Databento data client configuration (aggregated dataset mode)
            # Use anonymized EQUS venue for EQUS.MINI (live OHLCV aggregated bars, MBP-1/TBBO/Trades available)
            venue_dataset_map = {"EQUS": "EQUS.MINI"}

            data_config: DatabentoDataClientConfig = DatabentoDataClientConfig(
                api_key=databento_api_key or "",  # guarded above
                http_gateway="https://hist.databento.com",
                live_gateway="wss://stream.databento.com",
                use_exchange_as_venue=False,
                venue_dataset_map=venue_dataset_map,
            )

            node_config = TradingNodeConfig(
                trader_id=TraderId("ML-ACTOR-001"),
                data_engine=data_engine_cfg,
                data_clients={
                    "DATABENTO": data_config,
                },
                exec_clients={},  # No execution for signal actor
            )

        # Create trading node
        self.node = TradingNode(config=node_config)

        # Register Databento factory only when using real data
        if not use_mock_data:
            # This must happen after node creation but before build()
            self.node.add_data_client_factory("DATABENTO", DatabentoLiveDataClientFactory)

        # Add Multi‑Instrument ML Signal Actor by default
        actor = MultiInstrumentSignalActor(config=actor_config)
        self.node.trader.add_actor(actor)

        # Subscribe to market data when using real feed
        if not use_mock_data:
            # If a universe is defined, subscribe each instrument to the same bar parameters
            try:
                bars_suffix = "-".join(str(bar_type).split("-")[1:])
            except Exception:
                bars_suffix = None
            if universe:
                for sym in universe:
                    bt_str = f"{sym}-{bars_suffix}" if bars_suffix else str(bar_type)
                    try:
                        actor.subscribe_bars(BarType.from_str(bt_str))
                    except Exception:
                        actor.subscribe_bars(bar_type)
            else:
                actor.subscribe_bars(bar_type)

        # Optional: attach a lightweight RecorderActor to persist live bars
        live_record_enable = os.getenv("ML_LIVE_RECORD_ENABLE", "1").strip().lower() in {"1", "true", "yes"}
        if live_record_enable:
            datasets_csv = os.getenv("ML_LIVE_RECORD_DATASETS", "bars").strip()
            dataset_tokens = {t.strip().lower() for t in datasets_csv.split(",") if t.strip()}
            record_bars = "bars" in dataset_tokens
            record_quotes = "quotes" in dataset_tokens
            record_trades = "trades" in dataset_tokens

            try:
                mgr = MLIntegrationManager(
                    db_connection=db_connection,
                    auto_start_postgres=False,
                    auto_migrate=False,
                    ensure_healthy=False,
                    strict_protocol_validation=False,
                )
                recorder = LiveDataRecorder(
                    data_store=mgr.data_store,  # type: ignore[arg-type]
                    data_registry=mgr.data_registry,  # type: ignore[arg-type]
                    buffer_size=int(os.getenv("ML_LIVE_RECORD_BUFFER", "1000")),
                    flush_interval_ms=int(os.getenv("ML_LIVE_RECORD_FLUSH_MS", "1000")),
                )
                rec_actor = RecorderActor(
                    recorder=recorder,
                    record_bars=record_bars,
                    record_quotes=record_quotes,
                    record_trades=record_trades,
                )
                self.node.trader.add_actor(rec_actor)
                if not use_mock_data and record_bars:
                    rec_actor.subscribe_bars(bar_type)
                # Best-effort start of periodic flush inside node loop when available
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(recorder.start())
                except Exception:
                    # Recorder still flushes on size thresholds and shutdown
                    logging.getLogger(__name__).debug("Recorder start task not scheduled", exc_info=True)
            except Exception as exc:
                # JSON/catalog fallback when PostgreSQL is unavailable
                try:
                    catalog_path = os.getenv("CATALOG_PATH", "").strip()
                    if not catalog_path:
                        raise RuntimeError("CATALOG_PATH is required for file-backed live recording fallback")
                    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

                    catalog = ParquetDataCatalog(catalog_path)
                    writer = ParquetCatalogMarketDataWriter(catalog)
                    data_store_fallback = CatalogWriteFacade(writer)

                    # JSON DataRegistry for events/watermarks (bootstrap manifests/contracts)
                    registry_path = Path.home() / ".nautilus" / "ml" / "registry"
                    persistence = PersistenceConfig(backend=BackendType.JSON, json_path=registry_path)
                    try:
                        from ml.registry.bootstrap_datasets import bootstrap_datasets

                        bootstrap_datasets(backend=BackendType.JSON, registry_path=registry_path)
                    except Exception:
                        logging.getLogger(__name__).debug("Bootstrap datasets skipped in fallback", exc_info=True)
                    data_registry = DataRegistry(registry_path=registry_path, persistence_config=persistence)

                    recorder = LiveDataRecorder(
                        data_store=data_store_fallback,  # type: ignore[arg-type]
                        data_registry=data_registry,
                        buffer_size=int(os.getenv("ML_LIVE_RECORD_BUFFER", "1000")),
                        flush_interval_ms=int(os.getenv("ML_LIVE_RECORD_FLUSH_MS", "1000")),
                    )
                    rec_actor = RecorderActor(
                        recorder=recorder,
                        record_bars=record_bars,
                        record_quotes=record_quotes,
                        record_trades=record_trades,
                    )
                    self.node.trader.add_actor(rec_actor)
                    if not use_mock_data and record_bars:
                        rec_actor.subscribe_bars(bar_type)
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(recorder.start())
                    except Exception:
                        logging.getLogger(__name__).debug("Recorder start task not scheduled (fallback)", exc_info=True)

                    # Optional: backfill on start in fallback mode (catalog-only)
                    if os.getenv("ML_BACKFILL_ON_START", "").lower() in {"1", "true", "yes"}:
                        import shlex
                        import subprocess

                        from ml.config.coverage import CoveragePolicy
                        from ml.config.coverage import get_max_lookback_days

                        lookback = os.getenv("BACKFILL_LOOKBACK_DAYS")
                        if not lookback:
                            lookback = str(get_max_lookback_days("bars", CoveragePolicy.from_env()))
                        dataset_id_env = os.getenv("BACKFILL_DATASET_ID", "EQUS.MINI")
                        instruments_env = os.getenv("BACKFILL_INSTRUMENTS", str(instrument_id))
                        schema_env = os.getenv("BACKFILL_SCHEMA", "bars")
                        cmd = [
                            "python",
                            "-m",
                            "ml.cli.ingest_backfill",
                            "--db",
                            db_connection,
                            "--dataset-id",
                            dataset_id_env,
                            "--schema",
                            schema_env,
                            "--instruments",
                            instruments_env,
                            "--lookback-days",
                            lookback,
                            "--coverage-mode",
                            "catalog",
                            "--write-mode",
                            "sql",
                            "--client-mode",
                            "catalog",
                            "--catalog-path",
                            catalog_path,
                        ]
                        logging.getLogger(__name__).info("Running fallback backfill: %s", shlex.join(cmd))
                        try:
                            subprocess.run(cmd, check=True)
                        except Exception as bf_exc:
                            logging.getLogger(__name__).warning("Fallback backfill failed: %s", bf_exc)
                except Exception as inner:
                    logging.getLogger(__name__).warning(
                        "Live recording disabled (no fallback available): %s; root=%s",
                        inner,
                        exc,
                    )

        # Build the node BEFORE entering async context
        # This is critical for proper event loop initialization
        self.node.build()

        print("\nML Signal Actor configured and ready")
        print("Waiting for market data...")
        # Mark healthy after successful setup
        self._healthy = True

    def _create_dummy_model(self, model_path: str) -> None:  # pragma: no cover - deprecated
        raise RuntimeError("Dummy pickle models are no longer supported.")

    async def run(self) -> None:
        """
        Run the actor node asynchronously.

        Uses the TradingNode async API when available so tests can await
        concurrent startup and controlled cancellation.
        """
        self.running = True

        try:
            if self.node is None:
                raise RuntimeError("Trading node not initialized")
            # TradingNode provides an async runner in test contexts
            run_async = getattr(self.node, "run_async", None)
            if callable(run_async):
                try:
                    await run_async()
                except asyncio.CancelledError:
                    # Expected during tests which cancel tasks
                    pass
            else:
                # Fallback to synchronous runner
                self.node.run()
        finally:
            self.running = False

    def run_sync(self) -> None:
        """
        Run the actor node synchronously (container/default path).
        """
        self.running = True

        # Set up graceful shutdown handlers
        def signal_handler(signum: int, frame: Any) -> None:
            print(f"\nReceived signal {signal.Signals(signum).name}, shutting down...")
            self.shutdown_sync()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Run the node
        try:
            if self.node is None:
                raise RuntimeError("Trading node not initialized")
            # Node was already built in setup()
            # Use node.run() which manages its own event loop
            self.node.run()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            self.shutdown_sync()
        except Exception as e:
            print(f"Error running node: {e}")
            self.shutdown_sync()
            sys.exit(1)

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the node (async-capable).
        """
        print("\nShutting down...")

        self.running = False
        self._healthy = False

        if self.node:
            stop_async = getattr(self.node, "stop_async", None)
            if callable(stop_async):
                try:
                    await stop_async()
                except Exception:
                    # Fall back to synchronous dispose
                    self.node.dispose()
            else:
                self.node.dispose()

        print("ML Signal Actor shutdown complete")

    def shutdown_sync(self) -> None:
        """
        Gracefully shutdown the node synchronously.
        """
        print("\nShutting down...")

        self.running = False
        self._healthy = False

        if self.node:
            # Synchronous dispose
            self.node.dispose()

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

    # Run the node (it manages its own event loop)
    try:
        actor_node.run_sync()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
