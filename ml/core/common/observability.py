"""
Observability Component.

This module provides observability pipeline management extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6.5). The component handles:

- ObservabilityService initialization
- Background flush scheduling
- Sync and async persistence to file/database
- Store injection for stage boundary tracking
- Async worker management

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.observability import ObservabilityComponent
>>> component = ObservabilityComponent(stores=[feature_store, model_store])
>>> component.initialize_observability_pipeline()
>>> component.start_observability_flush(base_path=Path("./obs"), interval_seconds=60.0)
>>> # ... later ...
>>> component.stop_observability_flush()

"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Event
from threading import Thread
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame as PdDataFrame


logger = logging.getLogger(__name__)


@dataclass
class ObservabilityComponent:
    """
    Manages observability pipeline including metrics collection,
    flush scheduling, and async workers.

    This component implements the observability management responsibilities
    extracted from MLIntegrationManager. It provides initialization,
    periodic/immediate flushing, and async worker management for the
    observability service.

    Attributes
    ----------
    stores : list[object]
        List of stores to inject observability service into. Each store
        will receive ``_observability_service`` attribute for stage
        boundary tracking.

    observability_service : ObservabilityService | None
        The lazily initialized observability service instance.

    Example
    -------
    >>> component = ObservabilityComponent(stores=[feature_store, model_store])
    >>> component.initialize_observability_pipeline()
    >>> if component.observability_service is not None:
    ...     tables = component.collect_observability_dataframes()
    ...     print(f"Collected tables: {list(tables.keys())}")
    >>> component.inject_observability_service_into_stores()

    """

    # Stores for injection (optional)
    stores: list[object] = field(default_factory=list)

    # Service and workers (runtime state)
    observability_service: Any = field(default=None, init=False)
    _obs_flusher: Any = field(default=None, init=False)
    _obs_stop_event: Event | None = field(default=None, init=False)
    _obs_thread: Thread | None = field(default=None, init=False)
    _obs_async_worker: Any = field(default=None, init=False)

    def initialize_observability_pipeline(self) -> None:
        """
        Initialize a lightweight observability service (off hot-path).

        Creates an ObservabilityService instance if not already present.
        Safe to call multiple times; re-invocation is a no-op when service
        already exists.

        This method is non-fatal; import errors or exceptions are handled
        gracefully to avoid coupling in environments lacking optional deps.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.initialize_observability_pipeline()
        >>> assert component.observability_service is not None

        """
        try:
            from ml.observability.service import ObservabilityService

            # Attach service lazily; safe if re-called
            self.observability_service = getattr(self, "observability_service", None)
            if self.observability_service is None:
                self.observability_service = ObservabilityService()
        except Exception:  # pragma: no cover - defensive
            # Keep method non-fatal to avoid coupling in environments lacking optional deps
            try:
                # Ensure attribute exists for callers checking presence
                self.observability_service = None
            except Exception as inner_exc:
                logger.debug("Failed to set observability_service=None: %s", inner_exc)
            return None

    def start_end_to_end_tracking(self) -> None:
        """
        No-op start of E2E tracking (for tests).

        This is a stub method maintained for API compatibility with
        MLIntegrationManager. Future implementations may add actual
        E2E tracking functionality.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.start_end_to_end_tracking()  # No-op

        """
        return None

    def start_health_checks(self) -> None:
        """
        No-op start of health monitoring (for tests).

        This is a stub method maintained for API compatibility with
        MLIntegrationManager. Health checks are implemented via the
        HealthMonitoringComponent.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.start_health_checks()  # No-op

        """
        return None

    def collect_observability_dataframes(self) -> dict[str, PdDataFrame | None]:
        """
        Materialize observability DataFrames from the service, if available.

        Returns a mapping of table name to DataFrame. When the service is not
        initialized, returns empty DataFrames (None values).

        Returns
        -------
        dict[str, PdDataFrame | None]
            Mapping with keys:
            - ``latency``: Latency watermark data
            - ``metrics``: Metrics collection data
            - ``correlation``: Event correlation data
            - ``health``: Health scores data

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.initialize_observability_pipeline()
        >>> tables = component.collect_observability_dataframes()
        >>> print(list(tables.keys()))
        ['latency', 'metrics', 'correlation', 'health']

        """
        try:
            svc = getattr(self, "observability_service", None)
            if svc is None:
                return {
                    "latency": None,
                    "metrics": None,
                    "correlation": None,
                    "health": None,
                }
            return {
                "latency": svc.latency_watermarks_df(),
                "metrics": svc.metrics_collection_df(),
                "correlation": svc.event_correlation_df(),
                "health": svc.health_scores_df(),
            }
        except Exception:  # pragma: no cover - defensive
            # Keep integration resilient
            return {
                "latency": None,
                "metrics": None,
                "correlation": None,
                "health": None,
            }

    def flush_observability_to_path(
        self,
        *,
        base_path: Path,
        file_format: str = "jsonl",
    ) -> dict[str, Path]:
        """
        Persist current observability tables to disk (off hot-path).

        Writes non-empty tables under ``base_path`` using the specified format
        ("jsonl" or "csv"). Returns a mapping of table name to file path for
        written tables.

        Parameters
        ----------
        base_path : Path
            Directory to write observability files to.
        file_format : str
            Output format: "jsonl" or "csv". Defaults to "jsonl".

        Returns
        -------
        dict[str, Path]
            Mapping of table name to written file path. Empty dict on error.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.initialize_observability_pipeline()
        >>> paths = component.flush_observability_to_path(
        ...     base_path=Path("./observability"),
        ...     file_format="jsonl",
        ... )
        >>> for table, path in paths.items():
        ...     print(f"{table}: {path}")

        """
        try:
            from ml.observability.persistence import ObservabilityPersistor

            tables = self.collect_observability_dataframes()
            # Collect returns DataFrame | None; persist accepts Mapping[str, DataFrame | None]
            sink = ObservabilityPersistor(base_path=base_path, file_format=file_format)
            res = sink.persist(tables)
            return res
        except Exception:  # pragma: no cover - defensive
            return {}

    def flush_observability_to_db(self, *, connection_string: str) -> dict[str, int]:
        """
        Persist current observability tables to a SQL database (off hot-path).

        Uses ``ObservabilityDBPersistor`` to write non-empty tables to a relational
        store (e.g., SQLite/PostgreSQL) and returns a mapping of table name to
        number of rows written.

        Parameters
        ----------
        connection_string : str
            Database connection URL (e.g., "sqlite:///obs.db" or PostgreSQL URL).

        Returns
        -------
        dict[str, int]
            Mapping of table name to row count written. Empty dict on error.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.initialize_observability_pipeline()
        >>> counts = component.flush_observability_to_db(
        ...     connection_string="sqlite:///observability.db",
        ... )
        >>> for table, count in counts.items():
        ...     print(f"{table}: {count} rows")

        """
        try:
            from ml.observability.db_persistence import ObservabilityDBPersistor

            tables = self.collect_observability_dataframes()
            per = ObservabilityDBPersistor(connection_string=connection_string)
            res = per.persist(tables)
            return res
        except Exception:  # pragma: no cover - defensive
            return {}

    def start_observability_flush(
        self,
        *,
        base_path: Path,
        interval_seconds: float | None = 60.0,
        file_format: str = "jsonl",
        sink: str = "file",
        db_connection_string: str | None = None,
    ) -> dict[str, Path] | None:
        """
        Start periodic flush of observability tables.

        When ``interval_seconds`` is None or <= 0, performs a single flush and
        returns the written mapping. Otherwise, starts a background thread
        managed by the integration instance.

        Parameters
        ----------
        base_path : Path
            Directory to write observability files to.
        interval_seconds : float | None
            Flush interval in seconds. If None or <= 0, performs single flush.
        file_format : str
            Output format: "jsonl" or "csv". Defaults to "jsonl".
        sink : str
            Persistence sink: "file" or "db". Defaults to "file".
        db_connection_string : str | None
            Database URL for DB sink. Required when sink == "db".

        Returns
        -------
        dict[str, Path] | None
            For single flush (interval <= 0): mapping of table names to paths.
            For background flush: None (thread started).

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> # Single flush:
        >>> paths = component.start_observability_flush(
        ...     base_path=Path("./obs"),
        ...     interval_seconds=0,
        ... )
        >>> # Background flush:
        >>> component.start_observability_flush(
        ...     base_path=Path("./obs"),
        ...     interval_seconds=60.0,
        ... )
        >>> # ... later ...
        >>> component.stop_observability_flush()

        """
        # Ensure service exists
        self.initialize_observability_pipeline()

        # Inject observability service into stores for stage boundary tracking
        self.inject_observability_service_into_stores()

        if interval_seconds is None or interval_seconds <= 0:
            return self.flush_observability_to_path(base_path=base_path, file_format=file_format)

        # Background scheduler (off hot-path)
        from ml.observability.scheduler import ObservabilityFlusher

        svc = getattr(self, "observability_service", None)
        if svc is None:
            return None

        self._obs_stop_event = Event()
        self._obs_flusher = ObservabilityFlusher(
            service=svc,
            base_path=base_path,
            file_format=file_format,
            interval_seconds=float(interval_seconds),
            sink=sink,
            db_connection_string=db_connection_string,
        )
        self._obs_thread = self._obs_flusher.start_background(self._obs_stop_event)
        return None

    def stop_observability_flush(self) -> None:
        """
        Stop background flush if running (idempotent).

        Signals the stop event and joins the background thread with a
        1 second timeout. Safe to call multiple times.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> component.start_observability_flush(base_path=Path("./obs"), interval_seconds=60.0)
        >>> # ... later ...
        >>> component.stop_observability_flush()  # Stops background thread
        >>> component.stop_observability_flush()  # Safe to call again (no-op)

        """
        stop = getattr(self, "_obs_stop_event", None)
        thread = getattr(self, "_obs_thread", None)
        if stop is not None:
            try:
                stop.set()
            except Exception as exc:
                logger.debug("Stop event set() failed: %s", exc)
        if thread is not None:
            try:
                thread.join(timeout=1.0)
            except Exception as exc:
                logger.debug("Join on observability thread failed: %s", exc)

    def inject_observability_service_into_stores(self) -> None:
        """
        Inject the observability service into all stores for stage boundary tracking.

        This enables stores to record latency and metrics for cold path operations when
        observability is enabled via ML_OBSERVABILITY_ENABLED environment variable.

        Sets ``_observability_service`` attribute on all stores provided in the
        ``stores`` list.

        Example
        -------
        >>> component = ObservabilityComponent(stores=[feature_store, model_store])
        >>> component.initialize_observability_pipeline()
        >>> component.inject_observability_service_into_stores()
        >>> # Stores now have _observability_service attribute

        """
        try:
            obs_service = getattr(self, "observability_service", None)
            if obs_service is None:
                return

            # Inject observability service into all stores
            for store in self.stores:
                if store is not None:
                    # Set the observability service as a private attribute
                    setattr(store, "_observability_service", obs_service)

            logger.debug(
                "Injected observability service into %d stores",
                len([s for s in self.stores if s]),
            )

        except Exception as exc:
            # Keep non-fatal; add structured debug + metric for visibility (off hot-path)
            logger.debug("Observability injection failed: %s", exc, exc_info=True)
            try:
                from ml.common.metrics_manager import MetricsManager as _MM

                _MM.default().inc(
                    "ml_pipeline_errors_total",
                    "ML pipeline errors",
                    labels={
                        "component": "integration",
                        "op": "inject_observability_service",
                        "error_type": "exception",
                    },
                    labelnames=("component", "op", "error_type"),
                )
            except Exception:
                # Never raise from metrics
                logger.debug("Metric emit failed for observability injection error", exc_info=True)

    def start_observability_from_config(self, cfg: object) -> None:
        """
        Start observability flushing based on an ObservabilityConfig.

        Accepts any object with attributes matching ObservabilityConfig fields to avoid
        hard dependencies in call sites.

        Parameters
        ----------
        cfg : object
            Configuration object with attributes:
            - ``base_path``: Base directory for file output (default: "./observability")
            - ``sink``: Persistence sink "file" or "db" (default: "file")
            - ``file_format``: Output format "jsonl" or "csv" (default: "jsonl")
            - ``interval_seconds``: Flush interval in seconds (default: 60.0)
            - ``db_connection_string``: Database URL for DB sink (optional)
            - ``async_enabled``: Enable async worker mode (default: False)
            - ``async_queue_maxsize``: Async queue capacity (default: 4096)
            - ``async_component_label``: Component label for metrics (default: "obs_async_worker")

        Example
        -------
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     base_path: str = "./obs"
        ...     sink: str = "file"
        ...     interval_seconds: float = 30.0
        >>> component = ObservabilityComponent()
        >>> component.start_observability_from_config(Config())

        """
        base_path = Path(getattr(cfg, "base_path", "./observability"))
        sink = str(getattr(cfg, "sink", "file"))
        file_format = str(getattr(cfg, "file_format", "jsonl"))
        interval_seconds = float(getattr(cfg, "interval_seconds", 60.0))
        db_url = getattr(cfg, "db_connection_string", None)
        async_enabled = bool(getattr(cfg, "async_enabled", False))
        async_queue_max = int(getattr(cfg, "async_queue_maxsize", 4096))
        async_component = str(getattr(cfg, "async_component_label", "obs_async_worker"))

        if async_enabled:
            # Initialize service and async worker (off hot-path)
            self.initialize_observability_pipeline()
            svc = getattr(self, "observability_service", None)
            if svc is None:
                return None
            try:
                from ml.observability.async_worker import ObservabilityAsyncWorker

                self._obs_async_worker = ObservabilityAsyncWorker(
                    service=svc,
                    sink="db" if sink == "db" else "file",
                    base_path=base_path if sink != "db" else None,
                    db_connection_string=str(db_url) if sink == "db" else None,
                    flush_interval_seconds=interval_seconds,
                    queue_maxsize=async_queue_max,
                    component_label=async_component,
                )
                # Start background task
                self._obs_async_worker.start()

                # Inject observability service into stores for stage boundary tracking
                self.inject_observability_service_into_stores()
            except Exception:  # pragma: no cover - defensive
                return None
        else:
            self.start_observability_flush(
                base_path=base_path,
                interval_seconds=interval_seconds,
                file_format=file_format,
                sink=sink,
                db_connection_string=db_url,
            )

    def stop_observability_async(self) -> None:
        """
        Stop async observability worker if running (idempotent).

        Drains the async worker queue and stops the background task with a
        1 second timeout. Safe to call multiple times.

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> # ... async worker started via start_observability_from_config ...
        >>> component.stop_observability_async()  # Stops and drains queue
        >>> component.stop_observability_async()  # Safe to call again (no-op)

        """
        try:
            worker = getattr(self, "_obs_async_worker", None)
            if worker is not None:
                # Best-effort stop with small timeout
                asyncio.run(worker.stop(drain=True, timeout=1.0))
                self._obs_async_worker = None
        except Exception:
            return None

    def get_observability_async_status(self) -> dict[str, object]:
        """
        Return status of async observability worker if running.

        Returns
        -------
        dict[str, object]
            Mapping with keys:
            - ``running``: bool indicating if worker is active
            - ``queue_size``: int current queue size (0 when not running)

        Example
        -------
        >>> component = ObservabilityComponent()
        >>> status = component.get_observability_async_status()
        >>> print(f"Running: {status['running']}, Queue: {status['queue_size']}")

        """
        try:
            worker = getattr(self, "_obs_async_worker", None)
            if worker is None:
                return {"running": False, "queue_size": 0}
            # Typed at runtime to avoid hard dependency
            size = getattr(worker, "queue_size", lambda: 0)()
            return {"running": True, "queue_size": int(size)}
        except Exception:
            return {"running": False, "queue_size": 0}

    def start_observability_from_env(self) -> None:
        """
        Start observability flushing using environment-driven config.

        Loads configuration from environment variables via ObservabilityConfig.from_env()
        and starts observability flushing accordingly.

        Example
        -------
        >>> import os
        >>> os.environ["ML_OBSERVABILITY_ENABLED"] = "1"
        >>> os.environ["ML_OBSERVABILITY_SINK"] = "file"
        >>> component = ObservabilityComponent()
        >>> component.start_observability_from_env()

        """
        try:
            from ml.config.observability import ObservabilityConfig

            cfg = ObservabilityConfig.from_env()
            self.start_observability_from_config(cfg)
        except Exception:  # pragma: no cover - defensive
            return None


__all__ = ["ObservabilityComponent"]
