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
import uuid as _uuid
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from flask import Flask
from flask import Response
from flask import jsonify

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.deployment.scheduling_utils import DailyTime
from ml.deployment.scheduling_utils import compute_next_utc_run
from ml.deployment.scheduling_utils import parse_bool_env
from ml.deployment.scheduling_utils import parse_daily_spec


# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TYPE_CHECKING
from typing import Any as _Any
from typing import cast as _cast

from ml._imports import check_ml_dependencies
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig
from ml.core.integration import MLIntegrationManager
from ml.data.scheduler import DataScheduler
from ml.observability.bootstrap import auto_start_if_configured
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore


# Provide a patchable symbol for tests; resolved lazily at runtime.
_ParquetDataCatalogRT: type[_Any] | None = None
# Public alias for tests to patch directly (e.g., via unittest.mock.patch)
# When None, the runtime will lazily import ParquetDataCatalog.
ParquetDataCatalog: type[_Any] | None = None

if TYPE_CHECKING:  # pragma: no cover - avoid heavy import at module import time
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _PDC_T


configure_logging()
_run_id: str = f"pipeline_{_uuid.uuid4().hex[:12]}"
bind_log_context(run_id=_run_id, component="ml.entrypoint_pipeline")
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


@app.route("/metrics")
def metrics() -> Response:  # pragma: no cover - simple pass-through
    """
    Prometheus metrics endpoint.
    """
    payload = generate_latest()
    return Response(payload, mimetype=CONTENT_TYPE_LATEST)


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

    def _signal_handler(self, signum: int, _frame: object | None) -> None:
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
                logger.debug("Scheduler.stop() failed during shutdown", exc_info=True)

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
                "DB_CONNECTION",
                os.environ.get("FEATURE_STORE_CONNECTION", os.environ.get("DATABASE_URL")),
            ),
        )

        return config

    def _initialize_stores(self, config: SchedulerConfig) -> tuple[FeatureStore, ModelStore]:
        """
        Initialize the feature and model stores.
        """
        # Initialize feature store
        fs_conn = config.feature_store_connection or os.environ.get(
            "DB_CONNECTION",
            os.environ.get(
                "FEATURE_STORE_CONNECTION",
                os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/nautilus"),
            ),
        )
        assert fs_conn is not None
        feature_store = FeatureStore(connection_string=fs_conn)

        # Initialize model store
        ms_conn = os.environ.get(
            "MODEL_STORE_CONNECTION",
            os.environ.get("DB_CONNECTION", os.environ.get("DATABASE_URL", fs_conn)),
        )
        model_store = ModelStore(connection_string=ms_conn)

        return feature_store, model_store

    def _initialize_catalog(self, config: SchedulerConfig) -> _PDC_T:
        """
        Initialize the data catalog.
        """
        catalog_path = Path(os.environ.get("CATALOG_PATH", "/app/data/catalog"))
        catalog_path.mkdir(parents=True, exist_ok=True)
        # Resolve catalog class: prefer patched module-level alias if present
        global _ParquetDataCatalogRT, ParquetDataCatalog
        ctor_any: type[_Any] | None = ParquetDataCatalog or _ParquetDataCatalogRT
        if ctor_any is None:
            # Lazy import to avoid heavy dependency at module import time
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _PDC

            _ParquetDataCatalogRT = _PDC
            ctor_any = _ParquetDataCatalogRT
        ctor = _cast(type["_PDC_T"], ctor_any)
        return ctor(str(catalog_path))

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
            _feature_store, _model_store = self._initialize_stores(config)
            catalog = self._initialize_catalog(config)

            # Create scheduler with unified ingestion flags (env truthy: 1,true,yes,on)
            use_orchestrator = parse_bool_env(os.environ.get("USE_ORCHESTRATOR"))
            dual_write = parse_bool_env(os.environ.get("DUAL_WRITE"))
            self.scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                use_orchestrator=use_orchestrator,
                dual_write=dual_write,
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
        assert self.scheduler is not None

        # Always perform an immediate daily update on entering daily mode
        try:
            self.scheduler.run_daily_update()
            pipeline_status["last_run"] = datetime.now(UTC).isoformat()
        except Exception as exc:
            logger.error("Initial daily update failed: %s", exc, exc_info=True)
            pipeline_status["errors"].append(str(exc))

        # Resolve schedule spec; allow HH:MM or crontab-like "M H * * *"
        schedule_raw = os.environ.get("PIPELINE_SCHEDULE")
        interval_seconds = int(os.environ.get("REALTIME_INTERVAL", "300"))

        while self.running and not self._shutdown_event.is_set():
            sleep_seconds: float
            if schedule_raw:
                try:
                    daily: DailyTime = parse_daily_spec(schedule_raw)
                    now = datetime.now(UTC)
                    next_run = compute_next_utc_run(now, daily)
                    sleep_seconds = max(0.0, (next_run - now).total_seconds())
                except ValueError:
                    # Bad spec; fall back to interval mode
                    sleep_seconds = float(interval_seconds)
            else:
                sleep_seconds = float(interval_seconds)

            # Sleep in short chunks to honor shutdown quickly
            end_time = time.monotonic() + sleep_seconds
            while time.monotonic() < end_time:
                if self._shutdown_event.wait(timeout=min(1.0, end_time - time.monotonic())):
                    break
            if self._shutdown_event.is_set() or not self.running:
                break

            try:
                self.scheduler.run_daily_update()
                pipeline_status["last_run"] = datetime.now(UTC).isoformat()
            except Exception as exc:
                logger.error("Scheduled daily update failed: %s", exc, exc_info=True)
                pipeline_status["errors"].append(str(exc))

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
                assert self.scheduler is not None

                # Use standardized retry/backoff for transient realtime update failures
                from ml.common.retry_utils import retry_with_backoff as _retry

                def _on_exc(attempt: int, exc: BaseException) -> None:
                    wait_time = min(60, 2 ** (attempt + 1))
                    logger.warning(
                        f"Realtime update attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait_time}s...",
                    )

                scheduler = self.scheduler
                assert scheduler is not None

                def _do_update() -> None:
                    scheduler.run_daily_update()

                _retry(
                    _do_update,
                    max_attempts=int(os.environ.get("REALTIME_MAX_RETRIES", "3")),
                    initial_delay=1.0,
                    multiplier=2.0,
                    max_delay=60.0,
                    on_exception=_on_exc,
                    sleep_fn=time.sleep,
                )

                pipeline_status["last_run"] = datetime.now().isoformat()

                # Wait before next update (configurable)
                interval = int(os.environ.get("REALTIME_INTERVAL", "300"))  # 5 minutes default
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Realtime update failed after retries: {e}")
                pipeline_status["errors"].append(str(e))
                # Cooldown before continuing loop
                time.sleep(60)

        logger.info("Realtime mode stopped")


def main() -> None:
    """
    Run main entry point.
    """
    # Start health check server in background
    health_host = os.environ.get("HEALTH_CHECK_HOST", "127.0.0.1")
    health_thread = threading.Thread(
        target=lambda: app.run(
            host=health_host,
            port=int(os.environ.get("HEALTH_CHECK_PORT", "8080")),
            debug=False,
        ),
    )
    health_thread.daemon = True
    health_thread.start()

    # Auto-start observability flushing if configured via env
    try:
        mgr: MLIntegrationManager = MLIntegrationManager.__new__(MLIntegrationManager)
        auto_start_if_configured(mgr)
    except Exception:
        logger.debug(
            "Observability auto-start skipped due to configuration or environment",
            exc_info=True,
        )

    # Run pipeline
    runner = PipelineRunner()
    runner.run()


if __name__ == "__main__":
    main()
