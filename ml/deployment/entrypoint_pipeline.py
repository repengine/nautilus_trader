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
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.environ.get("LOG_FILE", "/app/logs/ml_pipeline.log")),
    ],
)
logger = logging.getLogger(__name__)

# Health check Flask app
app = Flask(__name__)
pipeline_status = {"healthy": False, "last_run": None, "errors": []}


@app.route("/health")
def health_check():
    """
    Health check endpoint for Docker.
    """
    return jsonify(pipeline_status), 200 if pipeline_status["healthy"] else 503


class PipelineRunner:
    """
    ML Pipeline runner for Docker deployment.
    """

    def __init__(self):
        """
        Initialize the pipeline runner.
        """
        self.scheduler = None
        self.running = False
        self._shutdown_event = threading.Event()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()
        if self.scheduler:
            self.scheduler.stop()

    def _create_config(self) -> SchedulerConfig:
        """
        Create scheduler configuration from environment variables.
        """
        # Parse universe symbols
        symbols = os.environ.get("UNIVERSE_SYMBOLS", "SPY.XNAS").split(",")
        symbols = [s.strip() for s in symbols if s.strip()]

        # Create Databento config
        databento_config = DatabentoConfig(
            api_key=os.environ.get("DATABENTO_API_KEY", ""),
            dataset=os.environ.get("DATABENTO_DATASET", "EQUS.MINI"),
        )

        # Create universe config
        universe_config = UniverseConfig(
            symbols=symbols,
            sectors={},  # Can be extended via config file
            market_caps={},  # Can be extended via config file
        )

        # Create scheduler config
        config = SchedulerConfig(
            databento=databento_config,
            universe=universe_config,
            catalog_path=Path(os.environ.get("CATALOG_PATH", "/app/data/catalog")),
            feature_store_type=os.environ.get("FEATURE_STORE_TYPE", "postgres"),
            model_store_type=os.environ.get("MODEL_STORE_TYPE", "postgres"),
            schedule=os.environ.get("PIPELINE_SCHEDULE", "0 17 * * *"),
            max_workers=int(os.environ.get("MAX_WORKERS", "4")),
            batch_size=int(os.environ.get("BATCH_SIZE", "1000")),
        )

        return config

    def _initialize_stores(self, config: SchedulerConfig) -> tuple[FeatureStore, ModelStore]:
        """
        Initialize the feature and model stores.
        """
        # Initialize feature store
        feature_store = FeatureStore(
            store_type=config.feature_store_type,
            connection_string=os.environ.get("FEATURE_STORE_CONNECTION"),
        )

        # Initialize model store
        model_store = ModelStore(
            store_type=config.model_store_type,
            connection_string=os.environ.get("MODEL_STORE_CONNECTION"),
        )

        return feature_store, model_store

    def _initialize_catalog(self, config: SchedulerConfig) -> ParquetDataCatalog:
        """
        Initialize the data catalog.
        """
        catalog_path = config.catalog_path
        catalog_path.mkdir(parents=True, exist_ok=True)

        return ParquetDataCatalog(str(catalog_path))

    def run(self):
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
            logger.info(f"Universe: {config.universe.symbols}")

            # Initialize components
            feature_store, model_store = self._initialize_stores(config)
            catalog = self._initialize_catalog(config)

            # Create scheduler
            self.scheduler = DataScheduler(
                config=config,
                catalog=catalog,
                feature_store=feature_store,
                model_store=model_store,
            )

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

    def _run_backfill(self):
        """
        Run pipeline in backfill mode.
        """
        logger.info("Running backfill mode")

        # Get date range from environment or use defaults
        start_date = os.environ.get("BACKFILL_START", "2024-01-01")
        end_date = os.environ.get("BACKFILL_END", "2024-01-31")

        # Run backfill
        self.scheduler.run_backfill(start_date, end_date)

        logger.info("Backfill completed successfully")

    def _run_daily(self):
        """
        Run pipeline in daily scheduled mode.
        """
        logger.info("Running daily scheduled mode")
        self.running = True

        # Start scheduler
        self.scheduler.start()

        # Keep running until shutdown signal
        while self.running and not self._shutdown_event.is_set():
            time.sleep(60)  # Check every minute
            pipeline_status["last_run"] = datetime.now().isoformat()

        logger.info("Daily scheduler stopped")

    def _run_realtime(self):
        """
        Run pipeline in realtime mode.
        """
        logger.info("Running realtime mode")
        self.running = True

        # Run continuous updates
        while self.running and not self._shutdown_event.is_set():
            try:
                # Collect and process latest data
                self.scheduler.run_once()
                pipeline_status["last_run"] = datetime.now().isoformat()

                # Wait before next update (configurable)
                interval = int(os.environ.get("REALTIME_INTERVAL", "300"))  # 5 minutes default
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Realtime update failed: {e}")
                pipeline_status["errors"].append(str(e))
                time.sleep(60)  # Wait before retry

        logger.info("Realtime mode stopped")


def main():
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
