"""
MLIntegrationManager Facade - Decomposed Implementation.

This module provides a thin facade that preserves the exact public API of
MLIntegrationManager while internally delegating to the 7 decomposed components:

1. DatabaseLifecycleComponent - PostgreSQL probing, container startup, migrations
2. StoreInitializationComponent - Initialize 4 stores with progressive fallback
3. RegistryInitializationComponent - Initialize 4 registries with persistence config
4. HealthMonitoringComponent - Health checks, aggregation, protocol validation
5. ObservabilityComponent - Pipeline, flush scheduling, async workers
6. ActorFactoryComponent - Actor creation, shutdown, message publisher
7. EventIngestionComponent - Event ingestion pipeline, backfill operations

Feature Flag:
    Set ML_USE_LEGACY_INTEGRATION_MANAGER=1 to use the legacy implementation.

Example
-------
>>> from ml.core.integration_facade import MLIntegrationManagerFacade
>>> integration = MLIntegrationManagerFacade(
...     auto_start_postgres=True,
...     auto_migrate=True,
...     ensure_healthy=True,
... )
>>> integration.feature_store.write_features(...)
>>> integration.model_registry.register_model(...)

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.core.common import ActorFactoryComponent
from ml.core.common import DatabaseLifecycleComponent
from ml.core.common import EventIngestionComponent
from ml.core.common import HealthMonitoringComponent
from ml.core.common import ObservabilityComponent
from ml.core.common import RegistryInitializationComponent
from ml.core.common import StoreInitializationComponent


if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame as PdDataFrame

    from ml.preprocessing.event_ingestion import EventIngestionConfig
    from ml.stores.infrastructure import PartitionManager


logger = logging.getLogger(__name__)


def _use_legacy_integration_manager() -> bool:
    """
    Check if legacy mode is enabled via environment variable.

    Returns
    -------
    bool
        True if ML_USE_LEGACY_INTEGRATION_MANAGER is set to '1', False otherwise.

    """
    return os.getenv("ML_USE_LEGACY_INTEGRATION_MANAGER", "0") == "1"


@runtime_checkable
class HasDBConnection(Protocol):
    """
    Protocol for configs carrying an optional DB connection string.
    """

    db_connection: str | None


@dataclass(slots=True)
class ActorStoresRegistries:
    """
    Simple container for actor-attached stores and registries.

    This dataclass groups the primary store and registry instances provided to ML actors
    after applying progressive fallback (PRIMARY -> CACHED -> FILE -> DUMMY). It also
    carries persistence and connection information discovered during initialization.

    """

    feature_store: object
    model_store: object
    strategy_store: object
    data_store: object
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object
    persistence_config: object | None
    connection_string: str | None


class MLIntegrationManagerFacade:
    """
    Facade for MLIntegrationManager that delegates to decomposed components.

    This facade preserves the exact public API of the legacy MLIntegrationManager
    while internally using the 7 decomposed components for better maintainability.

    The facade implements the Universal ML Architecture Patterns:
    1. Mandatory 4-Store + 4-Registry Integration
    2. Protocol-First Interface Design
    3. Hot/Cold Path Separation
    4. Progressive Fallback Chains
    5. Centralized Metrics Bootstrap

    Feature Flag
    ------------
    Set ML_USE_LEGACY_INTEGRATION_MANAGER=1 to use legacy implementation instead.

    Attributes
    ----------
    feature_store : object
        Initialized FeatureStore (PostgreSQL, File, or Dummy).
    model_store : object
        Initialized ModelStore (PostgreSQL, File, or Dummy).
    strategy_store : object
        Initialized StrategyStore (PostgreSQL, File, or Dummy).
    data_store : object | None
        Initialized DataStore (wired with DataRegistry).
    feature_registry : object
        Initialized FeatureRegistry.
    model_registry : object
        Initialized ModelRegistry.
    strategy_registry : object
        Initialized StrategyRegistry.
    data_registry : object
        Initialized DataRegistry.
    partition_manager : PartitionManager | None
        Partition manager for PostgreSQL tables (None in fallback mode).
    db_connection : str
        The currently selected database connection string.
    auto_start_postgres : bool
        Whether to automatically start PostgreSQL via Docker.
    auto_migrate : bool
        Whether to automatically run database migrations.
    observability_service : ObservabilityService | None
        The lazily initialized observability service instance.

    Example
    -------
    >>> integration = MLIntegrationManagerFacade(
    ...     auto_start_postgres=True,
    ...     auto_migrate=True,
    ...     ensure_healthy=True,
    ... )
    >>> # All stores and registries are now available
    >>> integration.feature_store.write_features(...)
    >>> integration.model_registry.register_model(...)

    """

    # Public components (runtime-populated)
    feature_store: object
    model_store: object
    strategy_store: object
    data_store: object | None
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object
    partition_manager: PartitionManager | None

    if TYPE_CHECKING:  # pragma: no cover - typing only
        from threading import Event
        from threading import Thread

        from ml.observability.async_worker import ObservabilityAsyncWorker
        from ml.observability.scheduler import ObservabilityFlusher
        from ml.observability.service import ObservabilityService

        observability_service: ObservabilityService | None
        _obs_flusher: ObservabilityFlusher | None
        _obs_stop_event: Event | None
        _obs_thread: Thread | None
        _obs_async_worker: ObservabilityAsyncWorker | None

    def __init__(
        self,
        config: HasDBConnection | None = None,
        db_connection: str | None = None,
        auto_start_postgres: bool = False,
        auto_migrate: bool = False,
        ensure_healthy: bool = True,
        strict_protocol_validation: bool | None = None,
    ) -> None:
        """
        Initialize the ML integration manager facade.

        Parameters
        ----------
        config : HasDBConnection, optional
            ML system configuration with optional db_connection attribute.
        db_connection : str, optional
            Database connection string (overrides config.db_connection).
        auto_start_postgres : bool, default False
            Automatically start PostgreSQL container if not running.
        auto_migrate : bool, default False
            Automatically run database migrations on startup.
        ensure_healthy : bool, default True
            Block until all components are healthy.
        strict_protocol_validation : bool | None, default None
            Validate protocol compliance; None reads from env.

        Raises
        ------
        ValueError
            If no PostgreSQL connection candidates are found.

        """
        # Collect connection candidates
        candidate_source = db_connection or (config.db_connection if config else None)
        candidates = collect_postgres_candidates(
            ConnectionRole.PRIMARY,
            explicit=candidate_source,
        )
        if not candidates.urls:
            raise ValueError(
                "No PostgreSQL connection candidates found. Set NAUTILUS_DB or --db",
            )
        connection_candidates: tuple[str, ...] = candidates.urls

        # Read environment flags
        env_start = os.getenv("ML_AUTO_START_DB", "").lower() in {"1", "true", "yes"}
        env_migrate = os.getenv("ML_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
        allow_dummy = os.getenv("ML_ALLOW_DUMMY", "").lower() in {"1", "true", "yes"}
        self.auto_start_postgres = auto_start_postgres or env_start
        self.auto_migrate = auto_migrate or env_migrate

        # File store path for fallback
        file_store_path = Path(
            os.getenv(
                "ML_FILE_STORE_PATH",
                str(Path.home() / ".nautilus" / "ml" / "file_store"),
            )
        )

        # Initialize partition_manager (will be set later if PostgreSQL available)
        self.partition_manager = None

        # -------------------------------------------------------------------------
        # Component 1: Database Lifecycle
        # -------------------------------------------------------------------------
        self._db_lifecycle = DatabaseLifecycleComponent(
            connection_candidates=connection_candidates,
            auto_start_postgres=self.auto_start_postgres,
            auto_migrate=self.auto_migrate,
            allow_dummy=allow_dummy,
        )
        self.db_connection = self._db_lifecycle.db_connection

        # -------------------------------------------------------------------------
        # Component 2: Store Initialization
        # -------------------------------------------------------------------------
        self._store_init = StoreInitializationComponent(
            db_connection=self.db_connection,
            file_store_path=file_store_path,
        )

        # -------------------------------------------------------------------------
        # Component 3: Registry Initialization
        # -------------------------------------------------------------------------
        registry_path = Path("./ml_registry")
        self._registry_init = RegistryInitializationComponent(
            db_connection=self.db_connection,
            json_fallback=False,
            file_fallback=False,
            registry_path=registry_path,
        )

        # -------------------------------------------------------------------------
        # Initialization Flow (mirrors legacy)
        # -------------------------------------------------------------------------

        # Check PostgreSQL availability
        if not self._db_lifecycle.is_postgres_running():
            if self.auto_start_postgres:
                self._db_lifecycle.start_postgres_container()
            if not self._db_lifecycle.is_postgres_running():
                # Try file fallback, then JSON/dummy fallback
                if not self._store_init.enable_file_fallback():
                    self._store_init.json_fallback = True
                    self._registry_init.json_fallback = True
                    logger.warning(
                        "PostgreSQL unavailable - falling back to JSON registries and dummy stores",
                    )
                    try:
                        from ml.common.metrics_manager import MetricsManager as _MM

                        mm = _MM.default()
                        mm.inc(
                            "ml_fallback_activations_total",
                            "Fallback activations",
                            labels={
                                "component": "ml_integration_manager",
                                "level": "json",
                            },
                            labelnames=("component", "level"),
                        )
                    except Exception:
                        pass
                else:
                    self._registry_init.file_fallback = True

        # Update db_connection in case it changed during probing
        self.db_connection = self._db_lifecycle.db_connection
        self._store_init.db_connection = self.db_connection
        self._registry_init.db_connection = self.db_connection

        # Initialize according to selected mode
        is_fallback = self._store_init.json_fallback or self._store_init.file_fallback
        if not is_fallback:
            self._db_lifecycle.init_database()

        # Initialize stores
        self._store_init.init_stores()

        # Initialize registries
        self._registry_init.init_registries()

        # Wire DataStore if in PostgreSQL mode
        if not is_fallback:
            data_store = self._registry_init.create_data_store()
            self._store_init.set_data_store(data_store)
            # Inject DataRegistry into stores
            self._registry_init.inject_data_registry_into_stores(
                self._store_init.feature_store,
                self._store_init.model_store,
            )

        # Initialize partition manager if PostgreSQL available
        if not is_fallback:
            self._init_partition_manager()

        # -------------------------------------------------------------------------
        # Wire component attributes to facade
        # -------------------------------------------------------------------------
        self._wire_component_attributes()

        # -------------------------------------------------------------------------
        # Component 4: Health Monitoring
        # -------------------------------------------------------------------------
        self._health_monitoring = HealthMonitoringComponent(
            feature_store=self.feature_store,
            model_store=self.model_store,
            strategy_store=self.strategy_store,
            data_store=self.data_store,
            feature_registry=self.feature_registry,
            model_registry=self.model_registry,
            strategy_registry=self.strategy_registry,
            data_registry=self.data_registry,
            partition_manager=self.partition_manager,
            is_postgres_running=self._db_lifecycle.is_postgres_running,
        )

        # -------------------------------------------------------------------------
        # Component 5: Observability
        # -------------------------------------------------------------------------
        stores_list = [
            self.feature_store,
            self.model_store,
            self.strategy_store,
            self.data_store,
        ]
        self._observability = ObservabilityComponent(stores=[s for s in stores_list if s])
        # Initialize observability_service attribute
        self.observability_service = None

        # -------------------------------------------------------------------------
        # Component 6: Actor Factory
        # -------------------------------------------------------------------------
        self._actor_factory = ActorFactoryComponent(
            db_connection=self.db_connection,
            feature_store=self.feature_store,
            model_store=self.model_store,
            strategy_store=self.strategy_store,
            data_store=self.data_store,
        )

        # -------------------------------------------------------------------------
        # Component 7: Event Ingestion
        # -------------------------------------------------------------------------
        self._event_ingestion = EventIngestionComponent(
            db_connection=self.db_connection,
            partition_manager=self.partition_manager,
            init_partition_manager=self._init_partition_manager,
        )

        # -------------------------------------------------------------------------
        # Final initialization steps
        # -------------------------------------------------------------------------

        # Ensure all components are healthy
        if ensure_healthy:
            self.ensure_healthy()

        # Validate protocol compliance
        self._validate_protocol_compliance(strict=strict_protocol_validation)

        # Optional: auto-run backfill at startup when configured via env
        try:
            self._maybe_run_backfill_on_start()
        except Exception as exc:
            logger.warning("Backfill bootstrap skipped: %s", exc)

    def _wire_component_attributes(self) -> None:
        """Wire component attributes to facade for public access."""
        # Stores
        self.feature_store = self._store_init.feature_store
        self.model_store = self._store_init.model_store
        self.strategy_store = self._store_init.strategy_store
        self.data_store = self._store_init.data_store

        # Registries
        self.feature_registry = self._registry_init.feature_registry
        self.model_registry = self._registry_init.model_registry
        self.strategy_registry = self._registry_init.strategy_registry
        self.data_registry = self._registry_init.data_registry

    def _init_partition_manager(self) -> PartitionManager | None:
        """Initialize partition management for PostgreSQL tables."""
        try:
            from ml.stores.infrastructure import PartitionManager

            self.partition_manager = PartitionManager(
                connection_string=self.db_connection,
                tables=[
                    "ml_feature_values",
                    "ml_model_predictions",
                    "ml_strategy_signals",
                    "market_data",
                ],
            )
            return self.partition_manager
        except Exception:
            logger.debug("Partition manager initialization failed", exc_info=True)
            return None

    # =========================================================================
    # Public API - Event Ingestion (delegated to EventIngestionComponent)
    # =========================================================================

    def ingest_events(self, config: EventIngestionConfig) -> Path:
        """
        Run the normalized event ingestion pipeline.

        Parameters
        ----------
        config : EventIngestionConfig
            Configuration describing the ingestion window, output directory, and
            optional data sources.

        Returns
        -------
        Path
            Location of the generated ``events.parquet`` artifact.

        Examples
        --------
        >>> from datetime import UTC, datetime
        >>> from pathlib import Path
        >>> cfg = EventIngestionConfig(
        ...     start=datetime(2024, 1, 1, tzinfo=UTC),
        ...     end=datetime(2024, 1, 31, tzinfo=UTC),
        ...     out_dir=Path("./data/events"),
        ... )
        >>> integration = MLIntegrationManagerFacade(ensure_healthy=False)
        >>> integration.ingest_events(cfg)
        PosixPath('data/events/events.parquet')

        """
        return self._event_ingestion.ingest_events(config)

    def _maybe_run_backfill_on_start(self) -> None:
        """Optionally run a gap backfill on startup using CLI."""
        self._event_ingestion.maybe_run_backfill_on_start()

    # =========================================================================
    # Public API - Health Monitoring (delegated to HealthMonitoringComponent)
    # =========================================================================

    def ensure_healthy(self) -> None:
        """
        Ensure all components are healthy.

        Raises
        ------
        RuntimeError
            If one or more components are unhealthy.

        """
        self._health_monitoring.ensure_healthy()

    def _validate_protocol_compliance(self, strict: bool | None = None) -> None:
        """Validate MLComponentProtocol compliance for core components."""
        self._health_monitoring.validate_protocol_compliance(strict=strict)

    def aggregate_health(self) -> dict[str, object]:
        """
        Aggregate component health into domain and system summaries.

        Returns
        -------
        dict[str, object]
            A structured health summary with keys:
            - components: per-component health and metrics
            - domains: aggregated health per domain
            - system: overall status with list of unhealthy components

        """
        return self._health_monitoring.aggregate_health()

    def check_health(self) -> dict[str, bool]:
        """
        Check health of all components.

        Returns
        -------
        dict[str, bool]
            Health status of each component.

        """
        return self._health_monitoring.check_health()

    def _check_store_health(self, store: object) -> bool:
        """Check health of a store component."""
        return self._health_monitoring.check_store_health(store)

    def _check_registry_health(self, registry: object, method_name: str) -> bool:
        """Check health of a registry component."""
        return self._health_monitoring.check_registry_health(registry, method_name)

    def _check_data_store_health(self) -> bool:
        """Check health of DataStore component."""
        return self._health_monitoring.check_data_store_health()

    def _check_partition_health(self) -> bool:
        """Check health of partition manager."""
        return self._health_monitoring.check_partition_health()

    # =========================================================================
    # Public API - Actor Factory (delegated to ActorFactoryComponent)
    # =========================================================================

    def create_integrated_actor(
        self,
        actor_class: type[Any],
        config: object,
    ) -> object:
        """
        Create an actor with automatic integration.

        Parameters
        ----------
        actor_class : type
            The actor class to instantiate.
        config : Any
            Actor configuration (should include db_connection).

        Returns
        -------
        Any
            Instantiated actor with all stores automatically connected.

        """
        return self._actor_factory.create_integrated_actor(actor_class, config)

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        self._actor_factory.shutdown()
        # Also stop observability if running
        self._observability.stop_observability_flush()
        self._observability.stop_observability_async()

    def configure_message_bus(
        self,
        *,
        backend: str | None = None,
        topic_prefix: str | None = None,
        retention_hours: int | None = None,
        max_size_mb: int | None = None,
    ) -> None:
        """No-op configuration stub for message bus (for tests)."""
        return self._actor_factory.configure_message_bus(
            backend=backend,
            topic_prefix=topic_prefix,
            retention_hours=retention_hours,
            max_size_mb=max_size_mb,
        )

    def configure_event_emission(
        self,
        *,
        batching_enabled: bool | None = None,
        batch_size: int | None = None,
        flush_interval_ms: int | None = None,
        correlation_strategy: str | None = None,
    ) -> None:
        """No-op configuration stub for event emission (for tests)."""
        return self._actor_factory.configure_event_emission(
            batching_enabled=batching_enabled,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            correlation_strategy=correlation_strategy,
        )

    def configure_event_system(self, **_: object) -> None:
        """No-op aggregate configuration for event system (for tests)."""
        return self._actor_factory.configure_event_system(**_)

    def configure_domain_bookkeeping(self, _config: object) -> None:
        """No-op configuration stub for domain bookkeeping (for tests)."""
        return self._actor_factory.configure_domain_bookkeeping(_config)

    def emit_cross_domain_event(self, _event: dict[str, object]) -> None:
        """No-op cross-domain event emitter stub (for tests)."""
        return self._actor_factory.emit_cross_domain_event(_event)

    def emit_cascade(
        self,
        source_event: dict[str, object],
        target_domain: str,
        *,
        delay_ns: int | None = None,
    ) -> dict[str, object]:
        """
        Create a cascaded event preserving correlation and timestamp order.

        Parameters
        ----------
        source_event : dict[str, object]
            The source event with correlation_id and other metadata.
        target_domain : str
            The target domain for the cascaded event.
        delay_ns : int | None
            Optional delay in nanoseconds for the target event.

        Returns
        -------
        dict[str, object]
            The cascaded event with preserved correlation and updated domain.

        """
        return self._actor_factory.emit_cascade(
            source_event,
            target_domain,
            delay_ns=delay_ns,
        )

    def set_message_publisher(self, publisher: object) -> None:
        """Configure the message publisher for ML stores which support it."""
        self._actor_factory.set_message_publisher(publisher)

    # =========================================================================
    # Public API - Observability (delegated to ObservabilityComponent)
    # =========================================================================

    def initialize_observability_pipeline(self) -> None:
        """Initialize a lightweight observability service (off hot-path)."""
        self._observability.initialize_observability_pipeline()
        # Sync the service reference to facade
        self.observability_service = self._observability.observability_service

    def start_end_to_end_tracking(self) -> None:
        """No-op start of E2E tracking (for tests)."""
        return self._observability.start_end_to_end_tracking()

    def start_health_checks(self) -> None:
        """No-op start of health monitoring (for tests)."""
        return self._observability.start_health_checks()

    def collect_observability_dataframes(self) -> dict[str, PdDataFrame | None]:
        """
        Materialize observability DataFrames from the service, if available.

        Returns
        -------
        dict[str, PdDataFrame | None]
            Mapping with keys: latency, metrics, correlation, health.

        """
        return self._observability.collect_observability_dataframes()

    def flush_observability_to_path(
        self,
        *,
        base_path: Path,
        file_format: str = "jsonl",
    ) -> dict[str, Path]:
        """
        Persist current observability tables to disk (off hot-path).

        Parameters
        ----------
        base_path : Path
            Directory to write observability files to.
        file_format : str
            Output format: "jsonl" or "csv". Defaults to "jsonl".

        Returns
        -------
        dict[str, Path]
            Mapping of table name to written file path.

        """
        return self._observability.flush_observability_to_path(
            base_path=base_path,
            file_format=file_format,
        )

    def flush_observability_to_db(self, *, connection_string: str) -> dict[str, int]:
        """
        Persist current observability tables to a SQL database (off hot-path).

        Parameters
        ----------
        connection_string : str
            Database connection URL.

        Returns
        -------
        dict[str, int]
            Mapping of table name to row count written.

        """
        return self._observability.flush_observability_to_db(
            connection_string=connection_string,
        )

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

        Parameters
        ----------
        base_path : Path
            Directory to write observability files to.
        interval_seconds : float | None
            Flush interval. If None or <= 0, performs single flush.
        file_format : str
            Output format: "jsonl" or "csv".
        sink : str
            Persistence sink: "file" or "db".
        db_connection_string : str | None
            Database URL for DB sink.

        Returns
        -------
        dict[str, Path] | None
            For single flush: mapping of table names to paths.
            For background flush: None.

        """
        result = self._observability.start_observability_flush(
            base_path=base_path,
            interval_seconds=interval_seconds,
            file_format=file_format,
            sink=sink,
            db_connection_string=db_connection_string,
        )
        # Sync service reference
        self.observability_service = self._observability.observability_service
        # Sync private attributes for legacy compatibility
        self._obs_flusher = getattr(self._observability, "_obs_flusher", None)
        self._obs_stop_event = getattr(self._observability, "_obs_stop_event", None)
        self._obs_thread = getattr(self._observability, "_obs_thread", None)
        return result

    def stop_observability_flush(self) -> None:
        """Stop background flush if running (idempotent)."""
        self._observability.stop_observability_flush()

    def _inject_observability_service_into_stores(self) -> None:
        """Inject the observability service into all stores."""
        self._observability.inject_observability_service_into_stores()

    def start_observability_from_config(self, cfg: object) -> None:
        """
        Start observability flushing based on an ObservabilityConfig.

        Parameters
        ----------
        cfg : object
            Configuration object with observability settings.

        """
        self._observability.start_observability_from_config(cfg)
        # Sync service reference
        self.observability_service = self._observability.observability_service
        self._obs_async_worker = getattr(self._observability, "_obs_async_worker", None)

    def stop_observability_async(self) -> None:
        """Stop async observability worker if running (idempotent)."""
        self._observability.stop_observability_async()

    def get_observability_async_status(self) -> dict[str, object]:
        """
        Return status of async observability worker if running.

        Returns
        -------
        dict[str, object]
            Mapping with keys: running, queue_size.

        """
        return self._observability.get_observability_async_status()

    def start_observability_from_env(self) -> None:
        """Start observability flushing using environment-driven config."""
        self._observability.start_observability_from_env()
        # Sync service reference
        self.observability_service = self._observability.observability_service

    # =========================================================================
    # Legacy internal methods (for backward compatibility)
    # =========================================================================

    def _init_database(self) -> None:
        """Initialize database connection and run migrations."""
        self._db_lifecycle.init_database()

    def _enable_file_fallback(self) -> bool:
        """Attempt to enable file-backed fallback stores."""
        return self._store_init.enable_file_fallback()

    def _init_dummy_components(self) -> None:
        """Initialize in-memory dummy components for testing fallback."""
        self._store_init.init_dummy_components()

    def _init_stores(self) -> None:
        """Initialize all store components."""
        self._store_init.init_stores()
        self._wire_component_attributes()

    def _init_registries(self) -> None:
        """Initialize all registry components."""
        self._registry_init.init_registries()
        self._wire_component_attributes()

    def _is_postgres_running(self) -> bool:
        """Check whether any candidate PostgreSQL connection is reachable."""
        return self._db_lifecycle.is_postgres_running()

    def _can_connect(self, connection_string: str) -> bool:
        """Probe whether a database connection string is usable."""
        return self._db_lifecycle.can_connect(connection_string)

    def _start_postgres_container(self) -> None:
        """Start PostgreSQL using Docker Compose if available."""
        self._db_lifecycle.start_postgres_container()

    def _run_migrations(self) -> None:
        """Run database migrations using the shared CLI plan builder."""
        self._db_lifecycle.run_migrations()

    # =========================================================================
    # Internal attributes for legacy compatibility
    # =========================================================================

    @property
    def _json_fallback(self) -> bool:
        """Whether JSON/dummy fallback is active."""
        return self._store_init.json_fallback

    @property
    def _file_fallback(self) -> bool:
        """Whether file-backed fallback is active."""
        return self._store_init.file_fallback

    @property
    def _file_store_path(self) -> Path:
        """Base path for file-backed stores."""
        return self._store_init.file_store_path

    @property
    def _connection_candidates(self) -> tuple[str, ...]:
        """Ordered list of PostgreSQL connection URLs."""
        return self._db_lifecycle.connection_candidates

    @property
    def _allow_dummy(self) -> bool:
        """Whether to allow dummy mode when PostgreSQL is unavailable."""
        return self._db_lifecycle.allow_dummy


# =============================================================================
# Module-level functions (preserved from legacy for backward compatibility)
# =============================================================================

# Singleton instance for global access
_integration_manager: MLIntegrationManagerFacade | None = None


def get_integration_manager(
    config: HasDBConnection | None = None,
) -> MLIntegrationManagerFacade:
    """
    Get or create the global integration manager.

    Parameters
    ----------
    config : HasDBConnection, optional
        Configuration (only used on first call).

    Returns
    -------
    MLIntegrationManagerFacade
        The global integration manager instance.

    """
    global _integration_manager

    if _integration_manager is None:
        _integration_manager = MLIntegrationManagerFacade(config)

    return _integration_manager


def reset_integration_manager() -> None:
    """Reset the global integration manager."""
    global _integration_manager

    if _integration_manager is not None:
        _integration_manager.shutdown()
        _integration_manager = None


def init_ml_stores_and_registries(config: Any) -> ActorStoresRegistries:
    """
    Initialize ML stores and registries with progressive fallback chains.

    This function implements the Universal ML Architecture Pattern 1 by providing
    centralized initialization of all 4 stores (Feature, Model, Strategy, Data) and
    4 registries with automatic fallback handling.

    Parameters
    ----------
    config : Any
        Configuration object with optional attributes:
        - use_dummy_stores (bool): Use dummy stores for testing
        - db_connection (str | None): PostgreSQL connection string
        - allow_dummy_fallback (bool): Allow fallback to dummy stores

    Returns
    -------
    ActorStoresRegistries
        Dataclass containing all 4 stores and 4 registries.

    """
    # Delegate to legacy function for now to ensure compatibility
    from ml.core.integration import init_ml_stores_and_registries as _legacy_init

    result = _legacy_init(config)
    return ActorStoresRegistries(
        feature_store=result.feature_store,
        model_store=result.model_store,
        strategy_store=result.strategy_store,
        data_store=result.data_store,
        feature_registry=result.feature_registry,
        model_registry=result.model_registry,
        strategy_registry=result.strategy_registry,
        data_registry=result.data_registry,
        persistence_config=result.persistence_config,
        connection_string=result.connection_string,
    )


# Backward compatibility alias (deprecated)
init_actor_stores_and_registries = init_ml_stores_and_registries
"""Deprecated: Use init_ml_stores_and_registries instead."""


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.registry.data_registry import DataRegistry
    from ml.stores.io_raw import RawIngestionWriterProtocol
    from ml.stores.io_raw import RawReaderProtocol
    from ml.stores.protocols import DataStoreFacadeProtocol


def create_data_store(
    *,
    registry: DataRegistry,
    connection_string: str,
    raw_reader: RawReaderProtocol | None = None,
    raw_writer: RawIngestionWriterProtocol | None = None,
) -> DataStoreFacadeProtocol:
    """
    Create a DataStore instance and return it as a narrow facade protocol.

    Uses dynamic import to avoid mypy resolving the concrete class hierarchy.

    """
    from ml.core.integration import create_data_store as _legacy_create

    return _legacy_create(
        registry=registry,
        connection_string=connection_string,
        raw_reader=raw_reader,
        raw_writer=raw_writer,
    )

class MLIntegrationManager(MLIntegrationManagerFacade):
    """
    Backward-compatible alias to present the facade under the legacy class name.
    """



__all__ = [
    "ActorStoresRegistries",
    "HasDBConnection",
    "MLIntegrationManager",
    "MLIntegrationManagerFacade",
    "create_data_store",
    "get_integration_manager",
    "init_actor_stores_and_registries",
    "init_ml_stores_and_registries",
    "reset_integration_manager",
]
