#!/usr/bin/env python
"""
ML Pipeline Docker entrypoint.

This script serves as the entry point for the ML pipeline container, handling
environment configuration and launching the appropriate pipeline mode.

"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from flask import Flask
from flask import jsonify


# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml._imports import check_ml_dependencies
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig
from ml.data.scheduler import DataScheduler
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


# Configure logging
def _get_log_handlers() -> list[logging.Handler]:
    """
    Get logging handlers based on environment.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("LOG_FILE")
    if log_file:
        # Only add file handler if LOG_FILE is explicitly set
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    return handlers


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_get_log_handlers(),
)
logger = logging.getLogger(__name__)


# Health check Flask app
class PipelineStatus(TypedDict):
    healthy: bool
    last_run: str | None
    errors: list[str]


app = Flask(__name__)
# Simple runtime status structure used by health endpoint
pipeline_status: PipelineStatus = {"healthy": False, "last_run": None, "errors": []}


@app.route("/health")
def health_check() -> tuple[Any, int]:
    """
    Health check endpoint for Docker.
    """
    return jsonify(pipeline_status), 200 if pipeline_status["healthy"] else 503


class PipelineRunner:
    """
    ML Pipeline runner for Docker deployment.
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline runner.
        """
        self.scheduler: DataScheduler | None = None
        self.running: bool = False
        self._shutdown_event = threading.Event()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object | None) -> None:
        """
        Handle shutdown signals gracefully.
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()
        if self.scheduler is not None:
            try:
                # Scheduler implements cooperative stop
                self.scheduler.stop()
            except Exception:
                pass

    def _create_config(self) -> SchedulerConfig:
        """
        Create scheduler configuration from environment variables.
        """
        # Parse universe symbols
        raw_symbols = os.environ.get("UNIVERSE_SYMBOLS", "SPY.XNAS")
        symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]

        # Create Databento config (API key consumed by underlying loader via env)
        databento_config = DatabentoConfig(
            dataset=os.environ.get("DATABENTO_DATASET", "EQUS.MINI"),
            schema=os.environ.get("DATABENTO_SCHEMA", "ohlcv-1m"),
            stype_in=os.environ.get("DATABENTO_STYPE_IN", "raw_symbol"),
        )

        # Universe expansion (optional)
        universe = UniverseConfig(
            expansion_mode=os.environ.get("UNIVERSE_MODE", "moderate"),  # type: ignore[arg-type]
        )
        # Merge user-provided symbols with expanded lists
        full_universe = list(dict.fromkeys(symbols + universe.get_full_universe()))

        # Create scheduler config
        config = SchedulerConfig(
            symbols=full_universe,
            databento=databento_config,
            feature_store_enabled=True,
            feature_store_connection=os.environ.get(
                "FEATURE_STORE_CONNECTION",
                os.environ.get("DATABASE_URL"),
            ),
        )

        return config

    def _initialize_stores(self, config: SchedulerConfig) -> tuple[FeatureStore, ModelStore]:
        """
        Initialize the feature and model stores.
        """
        # Initialize feature store
        fs_conn = config.feature_store_connection or os.environ.get(
            "FEATURE_STORE_CONNECTION",
            os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/nautilus"),
        )
        assert fs_conn is not None
        feature_store = FeatureStore(connection_string=fs_conn)

        # Initialize model store
        ms_conn = os.environ.get(
            "MODEL_STORE_CONNECTION",
            os.environ.get("DATABASE_URL", fs_conn),
        )
        model_store = ModelStore(connection_string=ms_conn)

        return feature_store, model_store

    def _initialize_catalog(self, config: SchedulerConfig) -> ParquetDataCatalog:
        """
        Initialize the data catalog.
        """
        catalog_path = Path(os.environ.get("CATALOG_PATH", "/app/data/catalog"))
        catalog_path.mkdir(parents=True, exist_ok=True)

        return ParquetDataCatalog(str(catalog_path))

    def run(self) -> None:
        """
        Run the ML pipeline based on environment configuration.
        """
        try:
            # Check dependencies
            check_ml_dependencies(["databento", "polars", "pandas", "numpy"])

            # Get pipeline mode
            mode = os.environ.get("PIPELINE_MODE", "daily").lower()

            # Create configuration
            config = self._create_config()
            logger.info(f"Starting ML pipeline in {mode} mode")
            logger.info(f"Universe symbols: {len(config.symbols)}")

            # Initialize components
            feature_store, model_store = self._initialize_stores(config)
            catalog = self._initialize_catalog(config)

            # Create scheduler
            self.scheduler = DataScheduler(catalog=catalog, config=config)

            # Update health status
            pipeline_status["healthy"] = True
            pipeline_status["last_run"] = datetime.now().isoformat()

            # Run based on mode
            if mode == "backfill":
                self._run_backfill()
            elif mode == "daily":
                self._run_daily()
            elif mode == "realtime":
                self._run_realtime()
            else:
                raise ValueError(f"Unknown pipeline mode: {mode}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            pipeline_status["healthy"] = False
            pipeline_status["errors"].append(str(e))
            sys.exit(1)

    def _run_backfill(self) -> None:
        """
        Run pipeline in backfill mode.
        """
        logger.info("Running backfill mode")

        # Backfill is currently mapped to a single daily update for simplicity.
        # A full backfill loop should iterate dates and call collection per day.
        assert self.scheduler is not None
        self.scheduler.run_daily_update()

        logger.info("Backfill completed successfully")

    def _run_daily(self) -> None:
        """
        Run pipeline in daily scheduled mode.
        """
        logger.info("Running daily scheduled mode")
        self.running = True

        # Configure schedule (no-op placeholder for now)
        assert self.scheduler is not None
        cron = os.environ.get("PIPELINE_SCHEDULE", "0 17 * * *")
        self.scheduler.schedule_updates(cron)

        # Keep running until shutdown signal
        while self.running and not self._shutdown_event.is_set():
            time.sleep(60)  # Check every minute
            pipeline_status["last_run"] = datetime.now().isoformat()

        logger.info("Daily scheduler stopped")

    def _run_realtime(self) -> None:
        """
        Run pipeline in realtime mode.
        """
        logger.info("Running realtime mode")
        self.running = True

        # Run continuous updates
        while self.running and not self._shutdown_event.is_set():
            try:
                # Collect and process latest data (best-effort realtime)
                assert self.scheduler is not None
                self.scheduler.run_daily_update()
                pipeline_status["last_run"] = datetime.now().isoformat()

                # Wait before next update (configurable)
                interval = int(os.environ.get("REALTIME_INTERVAL", "300"))  # 5 minutes default
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Realtime update failed: {e}")
                pipeline_status["errors"].append(str(e))
                time.sleep(60)  # Wait before retry

        logger.info("Realtime mode stopped")


def main() -> None:
    """
    Main entry point.
    """
    # Start health check server in background
    health_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=int(os.environ.get("HEALTH_CHECK_PORT", "8080")),
            debug=False,
        ),
    )
    health_thread.daemon = True
    health_thread.start()

    # Run pipeline
    runner = PipelineRunner()
    runner.run()


if __name__ == "__main__":
    main()
