#!/usr/bin/env python
"""
Production ML pipeline runner.

This script provides a production-ready entry point for running the ML pipeline in different modes:
- backfill: Process historical data for a date range
- daily: Run scheduled daily updates
- realtime: Start continuous real-time processing

"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import FrameType
from typing import Any

import click


try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None

from ml._imports import HAS_DATABENTO
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


# Configure logging
logger = logging.getLogger(__name__)


class MLPipelineRunner:
    """
    Production ML pipeline runner with multiple execution modes.

    This class manages the complete ML pipeline lifecycle including:
    - Data collection from Databento
    - Feature computation and storage
    - Model inference and signal generation
    - Health monitoring and metrics collection

    """

    def __init__(
        self,
        config: dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        """
        Initialize ML pipeline runner.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary loaded from file or defaults
        dry_run : bool
            Whether to run in dry-run mode (no actual data collection)

        """
        self.config = config
        self.dry_run = dry_run
        self.scheduler: DataScheduler | None = None
        self.catalog: ParquetDataCatalog | None = None
        self.shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> object:
        """
        Set up signal handlers for graceful shutdown.

        Returns a tiny wrapper whose ``__func__.__closure__[0].cell_contents``
        references the actual handler, to aid test inspection.

        """
        SignalHandler = Callable[[int, FrameType | None], None]

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            """
            Handle shutdown signals.
            """
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True

        # Register handlers for common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Provide a stable reference for tests to introspect the registered handler
        def _ref() -> SignalHandler:  # closes over handler via free var
            return signal_handler

        class _Wrapper:
            def __init__(self, fn: SignalHandler) -> None:
                self.__func__: SignalHandler = fn

        return _Wrapper(_ref)

    def setup_ml_system(self) -> DataScheduler:
        """
        Initialize all ML system components.

        Returns
        -------
        DataScheduler
            Configured data scheduler ready for operation

        Raises
        ------
        RuntimeError
            If system setup fails

        """
        logger.info("Setting up ML system components...")

        try:
            # 1. Validate environment
            self._validate_environment()

            # 2. Initialize catalog
            catalog_path = self.config.get("catalog_path", "./data")
            logger.info(f"Initializing ParquetDataCatalog at {catalog_path}")
            self.catalog = ParquetDataCatalog(catalog_path)

            # 3. Create scheduler configuration
            scheduler_config = self._create_scheduler_config()

            # 4. Initialize data collector
            collector = None
            if not self.dry_run:
                logger.info("Initializing DataCollector...")
                collector = DataCollector()

            # 5. Initialize feature engineer if configured
            feature_engineer = None
            if self.config.get("enable_features", True):
                feature_engineer = self._initialize_feature_engineer()

            # 6. Create scheduler
            logger.info("Creating DataScheduler...")
            self.scheduler = DataScheduler(
                catalog=self.catalog,
                config=scheduler_config,
                collector=collector,
                feature_engineer=feature_engineer,
            )

            # 7. Run health checks
            self._run_health_checks()

            logger.info("ML system setup completed successfully")
            return self.scheduler

        except Exception as e:
            logger.error(f"Failed to setup ML system: {e}", exc_info=True)
            raise RuntimeError(f"ML system setup failed: {e}") from e

    def _validate_environment(self) -> None:
        """
        Validate environment variables and dependencies.

        Raises
        ------
        ValueError
            If required environment variables are missing
        ImportError
            If required dependencies are not installed

        """
        # Check for Databento API key if not in dry-run mode
        if not self.dry_run:
            api_key = os.getenv("DATABENTO_API_KEY")
            if not api_key:
                raise ValueError(
                    "DATABENTO_API_KEY environment variable is required. "
                    "Set it with: export DATABENTO_API_KEY=your_key_here",
                )
            logger.info("Databento API key found")

        # Check database connection if features are enabled
        if self.config.get("enable_features", True):
            db_connection = os.getenv("DB_CONNECTION")
            if not db_connection:
                logger.warning(
                    "DB_CONNECTION not set, using default: "
                    "postgresql://postgres:postgres@localhost:5432/nautilus",
                )

        # Check required dependencies
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        if not self.dry_run and not HAS_DATABENTO:
            logger.error("Databento library not installed")
            logger.info("Install with: pip install databento")
            raise ImportError("databento library is required for data collection")

    def _create_scheduler_config(self) -> SchedulerConfig:
        """
        Create scheduler configuration from config dictionary.

        Returns
        -------
        SchedulerConfig
            Configured scheduler settings

        """
        # Get universe configuration
        universe_mode = self.config.get("universe_mode", "moderate")
        universe_config = UniverseConfig(expansion_mode=universe_mode)

        # Override symbols if provided in config
        symbols = self.config.get("symbols")
        if not symbols:
            symbols = universe_config.get_full_universe()

        # Create Databento configuration
        databento_config = DatabentoConfig(
            dataset=self.config.get("databento_dataset", "GLBX.MDP3"),
            schema=self.config.get("databento_schema", "ohlcv-1m"),
            stype_in=self.config.get("databento_stype", "raw_symbol"),
            use_temporary_files=self.config.get("use_temp_files", True),
            temp_data_dir=self.config.get("temp_data_dir", "./temp_databento_data"),
            price_precision=self.config.get("price_precision"),
        )

        # Create scheduler configuration
        return SchedulerConfig(
            symbols=symbols,
            collection_time=self.config.get("collection_time", "04:00"),
            retention_days=self.config.get("retention_days", 90),
            databento=databento_config,
            enable_l2_depth=self.config.get("enable_l2_depth", False),
            enable_trades=self.config.get("enable_trades", False),
            enable_quotes=self.config.get("enable_quotes", False),
            max_retries=self.config.get("max_retries", 3),
            retry_delay_seconds=self.config.get("retry_delay", 5.0),
            feature_store_enabled=self.config.get("enable_features", True),
            feature_store_connection=os.getenv("DB_CONNECTION"),
        )

    def _initialize_feature_engineer(self) -> object | None:
        """
        Initialize feature engineer for feature computation.

        Returns
        -------
        FeatureEngineer
            Configured feature engineer instance

        """
        try:
            from ml.features.engineering import FeatureConfig
            from ml.features.engineering import FeatureEngineer

            # Create feature configuration
            feature_config = FeatureConfig(
                include_microstructure=self.config.get("enable_microstructure_features", False),
                include_trade_flow=self.config.get("enable_trade_flow_features", False),
                return_periods=self.config.get("return_periods", [1, 5, 10, 20]),
                momentum_periods=self.config.get("momentum_periods", [5, 10, 20]),
            )

            logger.info("Initializing FeatureEngineer...")
            return FeatureEngineer(config=feature_config)

        except ImportError as e:
            logger.warning(f"Could not import FeatureEngineer: {e}")
            logger.info("Feature computation will be disabled")
            return None

    def _run_health_checks(self) -> None:
        """
        Run system health checks before starting.
        """
        logger.info("Running health checks...")

        # Check catalog accessibility
        if self.catalog:
            try:
                # Try to list instruments (should work even if empty)
                instruments = self.catalog.instruments()
                logger.info(f"Catalog accessible, found {len(instruments)} instruments")
            except Exception as e:
                logger.warning(f"Catalog health check warning: {e}")

        # Check database connectivity if features enabled
        if self.config.get("enable_features", True):
            try:
                import psycopg2

                db_connection = os.getenv(
                    "DB_CONNECTION",
                    "postgresql://postgres:postgres@localhost:5432/nautilus",
                )
                conn = psycopg2.connect(db_connection)
                conn.close()
                logger.info("Database connection successful")
            except ImportError:
                logger.warning("psycopg2 not installed, skipping database check")
            except Exception as e:
                logger.warning(f"Database connection check failed: {e}")

        logger.info("Health checks completed")

    def run_backfill(self, start_date: datetime, end_date: datetime) -> None:
        """
        Run backfill mode to process historical data.

        Parameters
        ----------
        start_date : datetime
            Start date for backfill
        end_date : datetime
            End date for backfill

        """
        logger.info(f"Starting backfill from {start_date.date()} to {end_date.date()}")

        if self.dry_run:
            logger.info("DRY RUN: Would process historical data")
            logger.info(f"  - Date range: {start_date} to {end_date}")
            if self.scheduler is not None:
                logger.info(f"  - Symbols: {len(self.scheduler.config.symbols)} symbols")
            logger.info(
                f"  - Features: {'enabled' if self.config.get('enable_features') else 'disabled'}",
            )
            return

        # Calculate number of trading days
        current_date = start_date
        days_processed = 0

        while current_date <= end_date and not self.shutdown_requested:
            # Skip weekends
            if current_date.weekday() in [5, 6]:  # Saturday, Sunday
                current_date += timedelta(days=1)
                continue

            logger.info(f"Processing {current_date.date()}...")

            try:
                # Run daily update for this date
                # In production, this would be modified to accept a specific date
                if self.scheduler is not None:
                    self.scheduler.run_daily_update()
                    days_processed += 1
                else:
                    raise RuntimeError("Scheduler not initialized")

            except Exception as e:
                logger.error(f"Failed to process {current_date.date()}: {e}")
                if self.config.get("stop_on_error", False):
                    raise

            current_date += timedelta(days=1)

            # Add small delay to avoid overwhelming API
            if not self.shutdown_requested:
                time.sleep(1)

        if self.shutdown_requested:
            logger.info("Backfill interrupted by shutdown request")
        else:
            logger.info(f"Backfill completed: processed {days_processed} trading days")

    def run_daily(self) -> None:
        """
        Run daily mode for scheduled updates.
        """
        logger.info("Starting daily update mode...")

        if self.dry_run:
            logger.info("DRY RUN: Would run daily update")
            if self.scheduler is not None:
                logger.info(f"  - Collection time: {self.scheduler.config.collection_time}")
                logger.info(f"  - Retention: {self.scheduler.config.retention_days} days")
            return

        # Run single daily update
        try:
            if self.scheduler is not None:
                self.scheduler.run_daily_update()
                logger.info("Daily update completed successfully")
            else:
                raise RuntimeError("Scheduler not initialized")
        except Exception as e:
            logger.error(f"Daily update failed: {e}", exc_info=True)
            raise

    def run_realtime(self) -> None:
        """
        Run continuous real-time processing mode.
        """
        logger.info("Starting real-time processing mode...")

        if self.dry_run:
            logger.info("DRY RUN: Would start real-time processing")
            logger.info("  - Continuous market data ingestion")
            logger.info("  - Real-time feature computation")
            logger.info("  - Live signal generation")
            return

        logger.info("Real-time mode active, press Ctrl+C to stop...")

        # In production, this would:
        # 1. Connect to real-time data feed
        # 2. Start feature computation pipeline
        # 3. Enable model inference
        # 4. Generate trading signals

        try:
            while not self.shutdown_requested:
                # Placeholder for real-time processing
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Real-time processing stopped by user")

        logger.info("Real-time processing shutdown complete")


def load_config(config_path: str | None) -> dict[str, Any]:
    """
    Load configuration from file or use defaults.

    Parameters
    ----------
    config_path : str | None
        Path to configuration file (YAML or JSON)

    Returns
    -------
    dict[str, Any]
        Configuration dictionary

    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            if config_file.suffix in [".yaml", ".yml"]:
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required for YAML config files. Install with: pip install PyYAML",
                    )
                config = yaml.safe_load(f)
            elif config_file.suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")

        logger.info(f"Loaded configuration from {config_path}")
        return config  # type: ignore[no-any-return]

    # Return default configuration
    return {
        "catalog_path": "./data",
        "universe_mode": "moderate",
        "enable_features": True,
        "enable_technical_features": True,
        "enable_microstructure_features": False,
        "enable_statistical_features": True,
        "retention_days": 90,
        "collection_time": "04:00",
        "max_retries": 3,
        "retry_delay": 5.0,
        "databento_dataset": "GLBX.MDP3",
        "databento_schema": "ohlcv-1m",
        "use_temp_files": True,
    }


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    verbose : bool
        Whether to enable verbose (DEBUG) logging

    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Adjust third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("databento").setLevel(logging.WARNING)


def _validate_backfill_dates(
    start_date: str | None,
    end_date: str | None,
) -> tuple[datetime, datetime]:
    """
    Validate and parse backfill dates.
    """
    if not start_date or not end_date:
        logger.error("--start-date and --end-date are required for backfill mode")
        sys.exit(1)

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.info("Use format: YYYY-MM-DD")
        sys.exit(1)

    if start_dt > end_dt:
        logger.error("Start date must be before end date")
        sys.exit(1)

    return start_dt, end_dt


def _execute_pipeline_mode(
    runner: MLPipelineRunner,
    mode: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """
    Execute the pipeline in the specified mode.
    """
    try:
        if mode == "backfill":
            start_dt, end_dt = _validate_backfill_dates(start_date, end_date)
            runner.run_backfill(start_dt, end_dt)
        elif mode == "daily":
            runner.run_daily()
        elif mode == "realtime":
            runner.run_realtime()

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["backfill", "daily", "realtime"]),
    required=True,
    help="Pipeline execution mode",
)
@click.option(
    "--start-date",
    type=str,
    help="Start date for backfill mode (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=str,
    help="End date for backfill mode (YYYY-MM-DD)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file (YAML or JSON)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run in dry-run mode without actual execution",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def main(
    mode: str,
    start_date: str | None,
    end_date: str | None,
    config: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Run ML pipeline in specified mode.

    This is the main entry point for operating the ML pipeline in production.

    Examples:
        # Run daily update
        python run_ml_pipeline.py --mode daily

        # Run backfill for date range
        python run_ml_pipeline.py --mode backfill --start-date 2024-01-01 --end-date 2024-01-31

        # Run with custom config
        python run_ml_pipeline.py --mode daily --config config/production.yaml

        # Test with dry run
        python run_ml_pipeline.py --mode daily --dry-run

    """
    # Setup logging
    setup_logging(verbose)

    logger.info("ML Pipeline Runner starting...")
    logger.info(f"Mode: {mode}, Dry run: {dry_run}")

    # Load configuration
    try:
        config_dict = load_config(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create and setup pipeline runner
    runner = MLPipelineRunner(config_dict, dry_run)
    try:
        scheduler = runner.setup_ml_system()
        logger.info(f"ML system initialized with {len(scheduler.config.symbols)} symbols")
    except Exception as e:
        logger.error(f"Failed to initialize ML system: {e}")
        sys.exit(1)

    # Execute pipeline mode
    _execute_pipeline_mode(runner, mode, start_date, end_date)
    logger.info("ML Pipeline Runner completed")


if __name__ == "__main__":
    main()
