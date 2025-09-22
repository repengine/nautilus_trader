"""High-level ML pipeline runner tasks reused by CLI entry points."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_DATABENTO
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml.common.logging_config import configure_logging
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class MLPipelineRunner:
    """Encapsulates the ML pipeline lifecycle (cold path)."""

    def __init__(self, config: dict[str, Any], dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.scheduler: DataScheduler | None = None
        self.catalog: ParquetDataCatalog | None = None
        self.shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> object:
        SignalHandler = Callable[[int, FrameType | None], None]

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            _ = frame
            logger.info("Received signal %s; requesting shutdown", signum)
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        def _ref() -> SignalHandler:
            return signal_handler

        class _Wrapper:
            def __init__(self, fn: Callable[[], SignalHandler]) -> None:
                self.__func__: SignalHandler = fn()

        return _Wrapper(_ref)

    def setup_ml_system(self) -> DataScheduler:
        logger.info("Setting up ML system components...")
        self._validate_environment()
        catalog_path = self.config.get("catalog_path", "./data")
        logger.info("Initializing ParquetDataCatalog at %s", catalog_path)
        self.catalog = ParquetDataCatalog(catalog_path)
        scheduler_config = self._create_scheduler_config()
        collector = None if self.dry_run else DataCollector()
        feature_engineer = None
        if self.config.get("enable_features", True):
            feature_engineer = self._initialize_feature_engineer()
        self.scheduler = DataScheduler(
            catalog=self.catalog,
            config=scheduler_config,
            collector=collector,
            feature_engineer=feature_engineer,
        )
        self._run_health_checks()
        return self.scheduler

    def _validate_environment(self) -> None:
        if not self.dry_run:
            api_key = os.getenv("DATABENTO_API_KEY")
            if not api_key:
                raise ValueError(
                    "DATABENTO_API_KEY environment variable is required. "
                    "Set it with: export DATABENTO_API_KEY=your_key_here",
                )
            logger.info("Databento API key found")
        if self.config.get("enable_features", True) and not os.getenv("DB_CONNECTION"):
            logger.warning(
                "DB_CONNECTION not set; defaulting to postgresql://postgres:postgres@localhost:5432/nautilus",
            )
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])
        if not self.dry_run and not HAS_DATABENTO:
            raise ImportError("databento package is required for data collection")

    def _create_scheduler_config(self) -> SchedulerConfig:
        universe_mode = self.config.get("universe_mode", "moderate")
        universe_config = UniverseConfig(expansion_mode=universe_mode)
        symbols = self.config.get("symbols") or universe_config.get_full_universe()
        databento_config = DatabentoConfig(
            dataset=self.config.get("databento_dataset", "GLBX.MDP3"),
            schema=self.config.get("databento_schema", "ohlcv-1m"),
            stype_in=self.config.get("databento_stype", "raw_symbol"),
            use_temporary_files=self.config.get("use_temp_files", True),
            temp_data_dir=self.config.get("temp_data_dir", "./temp_databento_data"),
            price_precision=self.config.get("price_precision"),
        )
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

    def _initialize_feature_engineer(self) -> FeatureEngineer | None:
        try:
            from ml.features.engineering import FeatureConfig
            from ml.features.engineering import FeatureEngineer

            feature_config = FeatureConfig(
                include_microstructure=self.config.get("enable_microstructure_features", False),
                include_trade_flow=self.config.get("enable_trade_flow_features", False),
                return_periods=self.config.get("return_periods", [1, 5, 10, 20]),
                momentum_periods=self.config.get("momentum_periods", [5, 10, 20]),
            )
            logger.info("Initializing FeatureEngineer")
            return FeatureEngineer(config=feature_config)
        except ImportError as exc:
            logger.warning("FeatureEngineer unavailable: %s", exc)
            return None

    def _run_health_checks(self) -> None:
        logger.info("Running health checks")
        if self.catalog:
            try:
                instruments = self.catalog.instruments()
                logger.info("Catalog accessible (instruments=%s)", len(instruments))
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Catalog health check warning: %s", exc)
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
                logger.warning("psycopg2 not installed; skipping database check")
            except Exception as exc:
                logger.warning("Database connection check failed: %s", exc)
        logger.info("Health checks completed")

    def run_backfill(self, start_date: datetime, end_date: datetime) -> None:
        logger.info("Starting backfill from %s to %s", start_date.date(), end_date.date())
        if self.dry_run:
            logger.info("DRY RUN: would process historical data")
            return
        current_date = start_date
        days_processed = 0
        while current_date <= end_date and not self.shutdown_requested:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            logger.info("Processing %s", current_date.date())
            if self.scheduler is None:
                raise RuntimeError("Scheduler not initialized")
            try:
                self.scheduler.run_daily_update()
                days_processed += 1
            except Exception as exc:
                logger.error("Failed to process %s: %s", current_date.date(), exc)
                if self.config.get("stop_on_error", False):
                    raise
            current_date += timedelta(days=1)
            if not self.shutdown_requested:
                time.sleep(1)
        if self.shutdown_requested:
            logger.info("Backfill interrupted by shutdown request")
        else:
            logger.info("Backfill completed: processed %s trading days", days_processed)

    def run_daily(self) -> None:
        logger.info("Starting daily update mode")
        if self.dry_run:
            logger.info("DRY RUN: would run daily update")
            return
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized")
        self.scheduler.run_daily_update()
        logger.info("Daily update completed successfully")

    def run_realtime(self) -> None:
        logger.info("Starting real-time processing mode")
        if self.dry_run:
            logger.info("DRY RUN: would start real-time processing")
            return
        logger.info("Real-time mode active; press Ctrl+C to stop")
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Real-time processing stopped by user")
        logger.info("Real-time processing shutdown complete")


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
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
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_file.open("r", encoding="utf-8") as handle:
        if config_file.suffix in {".yaml", ".yml"}:
            import importlib

            yaml_mod = importlib.import_module("yaml")
            config = yaml_mod.safe_load(handle)
        elif config_file.suffix == ".json":
            config = json.load(handle)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def setup_logging(verbose: bool) -> None:
    if verbose:
        configure_logging(level="DEBUG")
    else:
        configure_logging()
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("databento").setLevel(logging.WARNING)


def _validate_backfill_dates(start_date: str | None, end_date: str | None) -> tuple[datetime, datetime]:
    if not start_date or not end_date:
        raise ValueError("Backfill mode requires --start-date and --end-date")
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    if start_dt > end_dt:
        raise ValueError("Start date must be before end date")
    return start_dt, end_dt


def _execute_pipeline_mode(
    runner: MLPipelineRunner,
    mode: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    if mode == "backfill":
        start_dt, end_dt = _validate_backfill_dates(start_date, end_date)
        runner.run_backfill(start_dt, end_dt)
    elif mode == "daily":
        runner.run_daily()
    elif mode == "realtime":
        runner.run_realtime()
    else:
        raise ValueError(f"Unsupported pipeline mode: {mode}")


@dataclass(slots=True, frozen=True)
class PipelineRunConfig:
    mode: str
    start_date: str | None = None
    end_date: str | None = None
    config_path: str | None = None
    dry_run: bool = False
    verbose: bool = False


def run_pipeline(config: PipelineRunConfig) -> MLPipelineRunner:
    config_dict = load_config(config.config_path)
    runner = MLPipelineRunner(config_dict, config.dry_run)
    scheduler = runner.setup_ml_system()
    logger.info("ML system initialized with %s symbols", len(scheduler.config.symbols))
    _execute_pipeline_mode(runner, config.mode, config.start_date, config.end_date)
    logger.info("ML pipeline execution completed")
    return runner


__all__ = [
    "MLPipelineRunner",
    "PipelineRunConfig",
    "load_config",
    "run_pipeline",
    "setup_logging",
]
