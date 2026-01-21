"""
Event Ingestion Component.

This module provides event ingestion and backfill operations extracted from
MLIntegrationManager as part of the god-class decomposition effort (Phase 3.6.7).
The component handles:

- Event ingestion pipeline execution
- Optional backfill on startup via CLI invocation
- Metric emission for ingestion status tracking

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.event_ingestion import EventIngestionComponent
>>> component = EventIngestionComponent(
...     db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
... )
>>> # Run event ingestion
>>> path = component.ingest_events(config)
>>> # Or trigger backfill on startup
>>> component.maybe_run_backfill_on_start()

"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml.common.metrics_bootstrap import get_counter
from ml.common.watermark_window import WatermarkRegistryProtocol
from ml.common.watermark_window import resolve_watermark_start_datetime
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.events import Source


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.preprocessing.event_ingestion import EventIngestionConfig
    from ml.stores.protocols import DataStoreFacadeProtocol


logger = logging.getLogger(__name__)

_EVENT_INGEST_COUNTER = get_counter(
    "ml_event_ingestion_total",
    "Event ingestion attempts",
    ["status"],
)


@dataclass
class EventIngestionComponent:
    """
    Manages event ingestion and backfill operations.

    This component implements the event ingestion responsibilities extracted from
    MLIntegrationManager. It provides:

    - Event ingestion pipeline execution via EventIngestionUtility
    - Optional backfill on startup controlled by environment variables
    - Metric emission for success/error tracking
    - Partition maintenance on backfill failure

    Attributes
    ----------
    db_connection : str | None
        Database connection string for CLI backfill invocations.
    partition_manager : object | None
        Partition manager for maintenance after backfill failures.
    init_partition_manager : Callable[[], PartitionManager | None]
        Callback to lazily initialize partition manager when needed.

    Example
    -------
    >>> component = EventIngestionComponent(
    ...     db_connection="postgresql://localhost:5432/nautilus",
    ... )
    >>> from ml.preprocessing.event_ingestion import EventIngestionConfig
    >>> config = EventIngestionConfig(
    ...     start=datetime(2024, 1, 1, tzinfo=UTC),
    ...     end=datetime(2024, 1, 31, tzinfo=UTC),
    ...     out_dir=Path("./data/features/events"),
    ... )
    >>> path = component.ingest_events(config)

    """

    # Database connection for CLI invocation
    db_connection: str | None = None

    # Partition manager (injected or initialized via callback)
    partition_manager: object | None = None

    # Callback to initialize partition manager lazily
    init_partition_manager: Callable[[], Any] = field(default=lambda: None)

    def ingest_events(self, config: EventIngestionConfig) -> Path:
        """
        Run the normalized event ingestion pipeline.

        Executes the EventIngestionUtility with the provided configuration to
        ingest events from the configured data sources. Emits metrics for
        success or failure tracking.

        Parameters
        ----------
        config : EventIngestionConfig
            Configuration describing the ingestion window, output directory, and
            optional data sources.

        Returns
        -------
        Path
            Location of the generated ``events.parquet`` artifact.

        Raises
        ------
        Exception
            Re-raises any exception from the ingestion utility after logging
            and emitting error metrics.

        Examples
        --------
        >>> from datetime import UTC, datetime
        >>> from pathlib import Path
        >>> from ml.preprocessing.event_ingestion import EventIngestionConfig
        >>> cfg = EventIngestionConfig(
        ...     start=datetime(2024, 1, 1, tzinfo=UTC),
        ...     end=datetime(2024, 1, 31, tzinfo=UTC),
        ...     out_dir=Path("./data/features/events"),
        ... )
        >>> component = EventIngestionComponent()
        >>> path = component.ingest_events(cfg)
        PosixPath('data/features/events/events.parquet')

        """
        logger.info(
            "Starting event ingestion (start=%s end=%s out_dir=%s)",
            config.start,
            config.end,
            config.out_dir,
        )
        try:
            from ml.preprocessing.event_ingestion import EventIngestionUtility
        except Exception as exc:  # pragma: no cover - import guard
            _EVENT_INGEST_COUNTER.labels(status="error").inc()
            logger.error(
                "Event ingestion utility unavailable: %s",
                exc,
                exc_info=True,
            )
            raise

        data_store = self._build_data_store(config)
        effective_config = config
        if data_store is not None:
            effective_config = replace(config, write_parquet=False)
            effective_config = self._apply_watermark_window(
                effective_config,
                data_store=data_store,
            )

        utility = EventIngestionUtility(
            effective_config,
            data_store=data_store,
            ingest_run_id="event_ingestion",
        )
        try:
            target = utility.ingest()
        except Exception as exc:  # pragma: no cover - runtime failure path
            _EVENT_INGEST_COUNTER.labels(status="error").inc()
            logger.error("Event ingestion failed: %s", exc, exc_info=True)
            raise

        _EVENT_INGEST_COUNTER.labels(status="success").inc()
        logger.info("Completed event ingestion: %s", target)
        return target

    def _apply_watermark_window(
        self,
        config: EventIngestionConfig,
        *,
        data_store: DataStoreFacadeProtocol,
    ) -> EventIngestionConfig:
        registry = getattr(data_store, "registry", None)
        if not isinstance(registry, WatermarkRegistryProtocol):
            return config
        if not config.watermark_config.use_watermark:
            return config
        result = resolve_watermark_start_datetime(
            registry=registry,
            dataset_id=EVENTS_CALENDAR_DATASET_ID,
            instrument_ids=[""],
            source=Source.HISTORICAL,
            end=config.end,
            config=config.watermark_config,
            start=config.start,
        )
        if result.start is None or result.start == config.start:
            return config
        logger.info(
            "Event ingestion watermark window applied",
            extra={
                "original_start": config.start,
                "adjusted_start": result.start,
                "reason": result.reason,
            },
        )
        return replace(config, start=result.start)

    def _build_data_store(
        self,
        config: EventIngestionConfig,
    ) -> DataStoreFacadeProtocol | None:
        if not self.db_connection:
            return None
        try:
            from ml.core.common.registry_initialization import RegistryInitializationComponent
            from ml.core.common.store_initialization import StoreInitializationComponent
            from ml.registry.base import DummyRegistry
            from ml.registry.dataclasses import DatasetType
            from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
            from ml.stores.io_raw import FilteredRawWriter
            from ml.stores.protocols import EarningsStoreProtocol
            from ml.stores.protocols import FeatureStoreProtocol
            from ml.stores.protocols import ModelStoreProtocol
            from ml.stores.protocols import StrategyStoreProtocol
        except Exception:
            logger.debug("Event ingestion store dependencies missing", exc_info=True)
            return None

        try:
            store_init = StoreInitializationComponent(db_connection=self.db_connection)
            store_init.init_stores()
            registry_init = RegistryInitializationComponent(db_connection=self.db_connection)
            registry_init.init_registries()
            if isinstance(registry_init.data_registry, DummyRegistry):
                logger.warning("Event ingestion data registry unavailable; skipping SQL ingest")
                return None
            registry_init.inject_data_registry_into_stores(
                store_init.feature_store,
                store_init.model_store,
            )
            events_path = Path(config.out_dir) / "events.parquet"
            raw_writer = FilteredRawWriter(
                FeatureDatasetParquetRawWriter(events_path=events_path),
                enabled={DatasetType.EVENTS_CALENDAR: True},
            )
            return registry_init.create_data_store(
                feature_store=cast(FeatureStoreProtocol, store_init.feature_store),
                feature_dataset_store=store_init.feature_dataset_store,
                model_store=cast(ModelStoreProtocol, store_init.model_store),
                strategy_store=cast(StrategyStoreProtocol, store_init.strategy_store),
                earnings_store=cast(EarningsStoreProtocol, store_init.earnings_store),
                raw_writer=raw_writer,
            )
        except Exception:
            logger.warning("Event ingestion SQL store init failed; using parquet-only mode", exc_info=True)
            return None

    def maybe_run_backfill_on_start(self) -> None:
        """
        Optionally run a gap backfill on startup using CLI, controlled by env.

        Checks environment variables to determine if backfill should run,
        builds the appropriate CLI command, and executes it via subprocess.
        On failure, attempts partition maintenance as a fallback.

        Environment Flags
        -----------------
        ML_BACKFILL_ON_START : str
            '1'|'true'|'yes' to enable backfill.
        BACKFILL_DATASET_ID : str
            Required dataset ID (e.g., 'EQUS.MINI').
        BACKFILL_INSTRUMENTS : str
            Required comma-separated list of instruments.
        BACKFILL_SCHEMA : str
            'bars'|'tbbo'|'trades' (default 'bars').
        COVERAGE_MODE : str
            'sql'|'catalog' (default 'sql').
        WRITE_MODE : str
            Write mode (default 'sql').
        INGEST_CLIENT_MODE : str
            'catalog'|'databento'|'noop' (default 'catalog').
        BACKFILL_LOOKBACK_DAYS : str
            Integer lookback days (default '7').
        TABLE_NAME : str
            Target table name (default 'market_data').
        CATALOG_PATH : str
            Required for coverage-mode 'catalog'.
        DATABENTO_API_KEY : str
            API key for client-mode 'databento'.
        ALSO_WRITE_CATALOG : str
            '1'|'true'|'yes' to also write to catalog.

        Raises
        ------
        RuntimeError
            If required environment variables are missing when backfill is enabled.

        Examples
        --------
        >>> import os
        >>> os.environ["ML_BACKFILL_ON_START"] = "1"
        >>> os.environ["BACKFILL_DATASET_ID"] = "EQUS.MINI"
        >>> os.environ["BACKFILL_INSTRUMENTS"] = "AAPL,MSFT"
        >>> component = EventIngestionComponent(
        ...     db_connection="postgresql://localhost:5432/nautilus",
        ... )
        >>> component.maybe_run_backfill_on_start()

        """
        enabled = os.getenv("ML_BACKFILL_ON_START", "").lower() in {"1", "true", "yes"}
        if not enabled:
            return

        dataset_id = os.getenv("BACKFILL_DATASET_ID")
        instruments = os.getenv("BACKFILL_INSTRUMENTS")
        if not dataset_id or not instruments:
            raise RuntimeError(
                "BACKFILL_DATASET_ID and BACKFILL_INSTRUMENTS are required for backfill bootstrap",
            )

        schema = os.getenv("BACKFILL_SCHEMA", "bars")
        coverage_mode = os.getenv("COVERAGE_MODE", "sql")
        write_mode = os.getenv("WRITE_MODE", "sql")
        client_mode = os.getenv("INGEST_CLIENT_MODE", "catalog")
        lookback = os.getenv("BACKFILL_LOOKBACK_DAYS", "7")
        table_name = os.getenv("TABLE_NAME", "market_data")
        catalog_path = os.getenv("CATALOG_PATH", "")
        api_key = os.getenv("DATABENTO_API_KEY", "")
        also_write_catalog = os.getenv("ALSO_WRITE_CATALOG", "").lower() in {
            "1",
            "true",
            "yes",
        }

        # Build CLI command
        db_conn = self.db_connection or ""
        cmd = [
            "python",
            "-m",
            "ml.cli.ingest_backfill",
            "--db",
            db_conn,
            "--dataset-id",
            dataset_id,
            "--schema",
            schema,
            "--instruments",
            instruments,
            "--lookback-days",
            lookback,
            "--coverage-mode",
            coverage_mode,
            "--write-mode",
            write_mode,
            "--table-name",
            table_name,
            "--client-mode",
            client_mode,
        ]
        if coverage_mode == "catalog" or client_mode == "catalog":
            if not catalog_path:
                raise RuntimeError("CATALOG_PATH required for catalog coverage/client")
            cmd += ["--catalog-path", catalog_path]
        if client_mode == "databento" and api_key:
            cmd += ["--api-key", api_key]
        if also_write_catalog:
            if not catalog_path:
                raise RuntimeError(
                    "ALSO_WRITE_CATALOG set but CATALOG_PATH is missing; provide CATALOG_PATH",
                )
            cmd += ["--also-write-catalog"]

        logger.info("Running backfill bootstrap: %s", shlex.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            logger.warning("Backfill CLI failed: %s", exc)
            self._run_partition_maintenance()

    def _run_partition_maintenance(self) -> None:
        """
        Run partition maintenance as fallback after backfill failure.

        Initializes the partition manager if not already available and runs
        maintenance to create/manage partitions.

        """
        try:
            if self.partition_manager is None:
                self.partition_manager = self.init_partition_manager()
            if self.partition_manager is not None:
                stats = self.partition_manager.run_maintenance()  # type: ignore[attr-defined]
                logger.info("Partition maintenance: %s", stats)
        except Exception as exc:
            logger.warning("Partition maintenance skipped: %s", exc)


__all__ = ["EventIngestionComponent"]
