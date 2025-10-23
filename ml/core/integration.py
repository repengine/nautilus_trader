"""
ML System Integration Manager.

This module provides automatic integration of all ML components including stores,
registries, and database connections. It ensures that all data flows are automatically
connected and persisted.

"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.common.metrics_bootstrap import get_counter
from ml.common.protocols import MLComponentProtocol
from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command
from ml.core.db_engine import EngineManager
from ml.tasks.db import MigrationSchema


if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame as PdDataFrame

    from ml.preprocessing.event_ingestion import EventIngestionConfig
    from ml.registry.data_registry import DataRegistry
    from ml.registry.feature_registry import FeatureRegistry
    from ml.registry.model_registry import ModelRegistry
    from ml.registry.persistence import PersistenceConfig
    from ml.registry.strategy_registry import StrategyRegistry
    from ml.stores.data_store import DataStore
    from ml.stores.feature_store import FeatureStore
    from ml.stores.infrastructure import PartitionManager
    from ml.stores.io_raw import ParquetCatalogRawWriter
    from ml.stores.model_store import ModelStore
    from ml.stores.strategy_store import StrategyStore


# Runtime imports for store components and adapters referenced below
from ml.registry.data_registry import DataRegistry
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.model_registry import ModelRegistry
from ml.registry.persistence import PersistenceConfig
from ml.registry.strategy_registry import StrategyRegistry
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.file_backed import FileDataStore
from ml.stores.file_backed import FileEarningsStore
from ml.stores.file_backed import FileFeatureStore
from ml.stores.file_backed import FileModelStore
from ml.stores.file_backed import FileStrategyStore
from ml.stores.infrastructure import PartitionManager
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.model_store import ModelStore
from ml.stores.providers import SqlMarketDataReader
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.stores.raw_protocols import RawReaderProtocol
from ml.stores.strategy_store import StrategyStore


# Type variable for generic actor return types
ActorT = TypeVar("ActorT")

logger = logging.getLogger(__name__)

_EVENT_INGEST_COUNTER = get_counter(
    "ml_event_ingestion_total",
    "Event ingestion attempts",
    ["status"],
)


def _decode_stream(data: str | bytes | None) -> str:
    """Decode subprocess output safely for logging and diagnostics."""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return data or ""


@runtime_checkable
class HasDBConnection(Protocol):
    """
    Protocol for configs carrying an optional DB connection string.
    """

    db_connection: str | None


class ComponentHealthStatus(TypedDict, total=False):
    """
    Health status for a single component.

    Fields
    ------
    healthy : bool
        Whether component is healthy
    health : dict[str, object]
        Component health status details
    metrics : dict[str, float]
        Component performance metrics
    """

    healthy: bool
    health: dict[str, object]
    metrics: dict[str, float]


class DomainHealth(TypedDict):
    """
    Health status for a domain (e.g., features, model, strategy).

    Fields
    ------
    components : list[str]
        List of component names in this domain
    healthy : bool
        Whether all components in domain are healthy
    """

    components: list[str]
    healthy: bool


class HealthDomains(TypedDict, total=False):
    """
    All domain health statuses.

    Fields
    ------
    data : DomainHealth
        Data domain health (optional)
    features : DomainHealth
        Features domain health (optional)
    model : DomainHealth
        Model domain health (optional)
    strategy : DomainHealth
        Strategy domain health (optional)
    """

    data: DomainHealth
    features: DomainHealth
    model: DomainHealth
    strategy: DomainHealth


class SystemHealth(TypedDict):
    """
    Overall system health status.

    Fields
    ------
    healthy : bool
        Whether entire system is healthy
    unhealthy : list[str]
        List of unhealthy component names
    """

    healthy: bool
    unhealthy: list[str]


class HealthSummary(TypedDict):
    """
    Complete health summary for the ML integration system.

    Fields
    ------
    components : dict[str, ComponentHealthStatus]
        Per-component health status
    domains : HealthDomains
        Health status aggregated by domain
    system : SystemHealth
        Overall system health status

    Example
    -------
    >>> mgr = MLIntegrationManager()
    >>> health = mgr.aggregate_health()
    >>> assert health["system"]["healthy"]  # IDE knows this key exists!
    >>> for domain, status in health["domains"].items():
    ...     if not status["healthy"]:
    ...         print(f"Unhealthy domain: {domain}")
    """

    components: dict[str, ComponentHealthStatus]
    domains: HealthDomains
    system: SystemHealth


class MLIntegrationManager:
    """
    Automatically wires all ML components together.

    This manager ensures that:
    1. PostgreSQL is running (or starts it)
    2. All migrations are applied
    3. All stores are initialized
    4. All registries are connected
    5. Data flows are automatic

    All store and registry attributes are fully typed for IDE autocomplete and
    static type checking:
    - feature_store: FeatureStore
    - model_store: ModelStore
    - strategy_store: StrategyStore
    - data_store: DataStore | None
    - feature_registry: FeatureRegistry
    - model_registry: ModelRegistry
    - strategy_registry: StrategyRegistry
    - data_registry: DataRegistry

    Usage
    -----
    >>> from ml.core.integration import MLIntegrationManager
    >>> from ml.config.base import MLConfig
    >>>
    >>> config = MLConfig()
    >>> integration = MLIntegrationManager(config)
    >>>
    >>> # Everything is now wired and ready!
    >>> integration.feature_store.write_features(...)
    >>> integration.model_registry.register_model(...)

    """

    # Public components (runtime-populated)
    feature_store: FeatureStore
    model_store: ModelStore
    strategy_store: StrategyStore
    data_store: DataStore | None
    feature_registry: FeatureRegistry
    model_registry: ModelRegistry
    strategy_registry: StrategyRegistry
    data_registry: DataRegistry
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
        Initialize the ML integration manager.

        Parameters
        ----------
        config : HasDBConnection, optional
            ML system configuration
        db_connection : str, optional
            Database connection string (overrides config)
        auto_start_postgres : bool, default True
            Automatically start PostgreSQL container if not running
        auto_migrate : bool, default True
            Automatically run database migrations
        ensure_healthy : bool, default True
            Block until all components are healthy

        """
        candidate_source = db_connection or (config.db_connection if config else None)
        self.partition_manager: PartitionManager | None = None
        candidates = collect_postgres_candidates(
            ConnectionRole.PRIMARY,
            explicit=candidate_source,
        )
        if not candidates.urls:
            raise ValueError(
                "No PostgreSQL connection candidates found. Set NAUTILUS_DB or --db",
            )
        self._connection_candidates: tuple[str, ...] = candidates.urls
        self.db_connection = self._connection_candidates[0]

        env_start = os.getenv("ML_AUTO_START_DB", "").lower() in {"1", "true", "yes"}
        env_migrate = os.getenv("ML_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
        self._allow_dummy = os.getenv("ML_ALLOW_DUMMY", "").lower() in {"1", "true", "yes"}
        self.auto_start_postgres = auto_start_postgres or env_start
        self.auto_migrate = auto_migrate or env_migrate

        # Initialize components with progressive fallback when enabled
        self._json_fallback: bool = False
        self._file_fallback: bool = False
        self._file_store_path = Path(
            os.getenv("ML_FILE_STORE_PATH", Path.home() / ".nautilus" / "ml" / "file_store"),
        )

        if not self._is_postgres_running():
            if self.auto_start_postgres:
                self._start_postgres_container()
            if not self._is_postgres_running():
                if not self._enable_file_fallback():
                    self._json_fallback = True
                    logger.warning(
                        "PostgreSQL unavailable — falling back to JSON registries and dummy stores",
                    )
                    try:
                        from ml.common.metrics_manager import MetricsManager as _MM

                        mm = _MM.default()
                        mm.inc(
                            "ml_fallback_activations_total",
                            "Fallback activations",
                            labels={"component": "ml_integration_manager", "level": "json"},
                            labelnames=("component", "level"),
                        )
                    except Exception:
                        logger.debug(
                            "MetricsManager fallback increment failed",
                            extra={
                                "component": "MLIntegrationManager",
                                "metric": "ml_fallback_activations_total",
                            },
                            exc_info=True,
                        )

        # Initialize according to selected mode
        if not (self._json_fallback or self._file_fallback):
            self._init_database()
        self._init_stores()
        self._init_registries()
        if not (self._json_fallback or self._file_fallback):
            self._init_partition_manager()

        # Ensure everything is healthy
        if ensure_healthy:
            self.ensure_healthy()

        # Validate protocol compliance (warn by default)
        self._validate_protocol_compliance(strict=strict_protocol_validation)

        # Message bus is configured explicitly by callers when required.

        # Optional: auto-run backfill at startup when configured via env
        try:
            self._maybe_run_backfill_on_start()
        except Exception as exc:
            logger.warning("Backfill bootstrap skipped: %s", exc)

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
        >>> integration = MLIntegrationManager(ensure_healthy=False)
        >>> integration.ingest_events(cfg)
        PosixPath('data/events/events.parquet')

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
            logger.error("Event ingestion utility unavailable: %s", exc, exc_info=True)
            raise

        utility = EventIngestionUtility(config)
        try:
            target = utility.ingest()
        except Exception as exc:  # pragma: no cover - runtime failure path
            _EVENT_INGEST_COUNTER.labels(status="error").inc()
            logger.error("Event ingestion failed: %s", exc, exc_info=True)
            raise

        _EVENT_INGEST_COUNTER.labels(status="success").inc()
        logger.info("Completed event ingestion: %s", target)
        return target

    def _init_database(self) -> None:
        """
        Initialize database connection and run migrations.
        """
        # At this point either PostgreSQL is running or we are in dummy mode.
        if self._allow_dummy and not self._is_postgres_running():
            # Dummy mode: nothing to do
            return
        # Run migrations if needed
        if self.auto_migrate:
            self._run_migrations()

    def _enable_file_fallback(self) -> bool:
        """
        Attempt to enable file-backed fallback stores.

        Returns
        -------
        bool
            ``True`` when the file-backed stores were initialised successfully.

        """
        try:
            self._file_store_path.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("file_store_path_unavailable", exc_info=True)
            return False

        try:
            from ml.common.metrics_manager import MetricsManager as _MM

            _MM.default().inc(
                "ml_fallback_activations_total",
                "Fallback activations",
                labels={"component": "ml_integration_manager", "level": "file"},
                labelnames=("component", "level"),
            )
        except Exception:
            logger.debug("file_fallback_metric_failed", exc_info=True)

        self._file_fallback = True
        logger.warning(
            "PostgreSQL unavailable — using file-backed ML stores at %s",
            self._file_store_path,
        )
        return True

    def _init_dummy_components(self) -> None:
        """
        Initialize in-memory dummy components for testing fallback.

        This mode provides protocol-compatible components without persistence.

        """
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        # Stores
        self.feature_store = cast(FeatureStore, DummyStore())
        self.model_store = cast(ModelStore, DummyStore())
        self.strategy_store = cast(StrategyStore, DummyStore())
        self.data_store = cast(DataStore, DummyStore())

        # Registries
        self.feature_registry = cast(FeatureRegistry, DummyRegistry())
        self.model_registry = cast(ModelRegistry, DummyRegistry())
        self.strategy_registry = cast(StrategyRegistry, DummyRegistry())
        self.data_registry = cast(DataRegistry, DummyRegistry())

        # Partition manager is not applicable in dummy mode
        self.partition_manager = None

    def _init_stores(self) -> None:
        """
        Initialize all store components.
        """
        # Import persistence types lazily to avoid import-time cycles
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        if self._file_fallback:
            file_root = self._file_store_path
            self.feature_store = cast(
                FeatureStore,
                FileFeatureStore(base_path=file_root / "features"),
            )
            self.model_store = cast(
                ModelStore,
                FileModelStore(base_path=file_root / "models"),
            )
            self.strategy_store = cast(
                StrategyStore,
                FileStrategyStore(base_path=file_root / "strategies"),
            )
            earnings_store = FileEarningsStore(base_path=file_root / "earnings")
            logger.info("FileEarningsStore initialized at %s", file_root / "earnings")
            self.data_store = cast(
                DataStore,
                FileDataStore(
                    base_path=file_root / "datastore",
                    earnings_store=earnings_store,
                ),
            )
        elif self._json_fallback:
            from ml.stores.base import DummyStore

            self.feature_store = cast(FeatureStore, DummyStore())
            self.model_store = cast(ModelStore, DummyStore())
            self.strategy_store = cast(StrategyStore, DummyStore())
            self.data_store = cast(DataStore, DummyStore())
        else:
            # Create persistence config (DB-first)
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=self.db_connection,
            )

            # Initialize stores with automatic persistence
            self.feature_store = FeatureStore(
                connection_string=self.db_connection,
                batch_size=1000,
                enable_batching=True,
            )

            self.model_store = ModelStore(
                persistence_config=persistence_config,
                batch_size=1000,
            )

            self.strategy_store = StrategyStore(
                persistence_config=persistence_config,
                batch_size=1000,
            )

            # Initialize DataStore after registries are available (will be set in _init_registries)
            self.data_store = None

    def _init_registries(self) -> None:
        """
        Initialize all registry components.
        """
        # Import registry components lazily to avoid import-time cycles
        from ml.registry import DataRegistry
        from ml.registry import FeatureRegistry
        from ml.registry import ModelRegistry
        from ml.registry import StrategyRegistry
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        # Create persistence config for registries (DB-first; fallback to JSON)
        if self._file_fallback or self._json_fallback:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=Path("./ml_registry"),
            )
        else:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=self.db_connection,
            )

        # Create a registry path (for file storage)
        registry_path = Path("./ml_registry")
        registry_path.mkdir(parents=True, exist_ok=True)

        # Initialize registries
        self.feature_registry = FeatureRegistry(
            registry_path=registry_path / "features",
            persistence_config=persistence_config,
        )
        self.model_registry = ModelRegistry(
            registry_path=registry_path / "models",
            persistence_config=persistence_config,
        )
        self.strategy_registry = StrategyRegistry(
            base_path=registry_path / "strategies",
            persistence_config=persistence_config,
        )

        # Initialize DataRegistry
        self.data_registry = DataRegistry(
            registry_path=registry_path / "datasets",
            persistence_config=persistence_config,
        )

        if self._file_fallback:
            return

        if self._json_fallback:
            # Stores are dummy in fallback; nothing more to wire
            return

        # Now initialize DataStore with the registry (DB path)
        # Attach SQL reader by default and optionally mirror to Parquet catalog
        table_name = os.getenv("TABLE_NAME", "market_data")
        raw_reader = SqlMarketDataReader(
            connection_string=self.db_connection,
            table_name=table_name,
        )
        raw_writer = None
        try:  # best-effort; keep init resilient
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

            catalog_path = os.getenv("CATALOG_PATH", "").strip()
            if catalog_path:
                catalog = ParquetDataCatalog(catalog_path)
                raw_writer = ParquetCatalogRawWriter(catalog)
        except Exception:
            logger.debug("Parquet catalog adapters not attached", exc_info=True)

        self.data_store = create_data_store(
            registry=self.data_registry,
            connection_string=self.db_connection,
            raw_reader=raw_reader,
            raw_writer=raw_writer,
        )
        # Ensure FeatureStore/ModelStore publish into the same DataRegistry instance
        try:
            setter = getattr(self.feature_store, "set_data_registry", None)
            if callable(setter):
                setter(self.data_registry)
            setter2 = getattr(self.model_store, "set_data_registry", None)
            if callable(setter2):
                setter2(self.data_registry)
        except Exception:
            logger.debug("Failed to inject shared DataRegistry into stores", exc_info=True)

    def _init_partition_manager(self) -> None:
        """
        Initialize partition management.
        """
        self.partition_manager = PartitionManager(
            connection_string=self.db_connection,
            tables=[
                "ml_feature_values",
                "ml_model_predictions",
                "ml_strategy_signals",
                "market_data",
            ],
        )

    def _is_postgres_running(self) -> bool:
        """
        Check whether any candidate PostgreSQL connection is reachable.
        """
        for candidate in self._connection_candidates:
            if self._can_connect(candidate):
                if candidate != self.db_connection:
                    try:  # pragma: no cover - structlog guard
                        alt_url = make_url(candidate)
                        host = alt_url.host or "localhost"
                        port = alt_url.port or "?"
                    except Exception:  # pragma: no cover - defensive guard
                        host = "localhost"
                        port = "?"
                    logger.info(
                        "PostgreSQL reachable — updating integration connection (host=%s port=%s)",
                        host,
                        port,
                    )
                    EngineManager.dispose_engine(self.db_connection)
                    self.db_connection = candidate
                return True

        logger.debug(
            "postgres_unreachable candidates=%s",
            list(self._connection_candidates),
        )
        return False

    def _can_connect(self, connection_string: str) -> bool:
        """
        Probe whether a database connection string is usable.
        """
        try:
            engine = EngineManager.get_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except OperationalError:
            EngineManager.dispose_engine(connection_string)
            return False
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("postgres_connect_probe_failed", exc_info=True)
            EngineManager.dispose_engine(connection_string)
            return False

    def _start_postgres_container(self) -> None:
        """
        Start PostgreSQL using Docker Compose if available, else docker run.
        """
        logger.info("Starting PostgreSQL (preferring Docker Compose if available)...")

        import shutil

        compose_file = None
        docker_path = shutil.which("docker")
        if docker_path is None:
            raise RuntimeError("docker executable not found in PATH")
        # Prefer explicit env override, then deployment compose, then dev compose, then root
        import os

        candidates: list[object] = []
        env_compose = os.getenv("ML_COMPOSE_FILE")
        if env_compose:
            candidates.append(Path(env_compose))
        candidates.extend(
            [
                Path("ml/deployment/docker-compose.yml"),
                Path("ml/docker-compose.dev.yml"),
                Path("docker-compose.yml"),
            ],
        )
        for candidate in candidates:
            try:
                if isinstance(candidate, Path) and candidate.exists():
                    compose_file = candidate
                    break
            except Exception:
                logger.debug(
                    "Failed to evaluate docker compose candidate",
                    extra={"candidate": str(candidate)},
                    exc_info=True,
                )
                continue

        timeout_env = os.getenv("ML_DOCKER_TIMEOUT")
        docker_timeout: float | None = float(timeout_env) if timeout_env else None

        if compose_file is not None:
            try:
                run_command(
                    [docker_path, "compose", "-f", str(compose_file), "up", "-d", "postgres"],
                    timeout=docker_timeout,
                    log=logger,
                )
            except SubprocessExecutionError as exc:
                logger.warning(
                    "docker_compose_up_failed compose_file=%s returncode=%s",
                    compose_file,
                    exc.returncode,
                    exc_info=True,
                )
                compose_file = None

        if compose_file is None:
            result_stdout = ""
            try:
                result = run_command(
                    [
                        docker_path,
                        "ps",
                        "-a",
                        "--filter",
                        "name=nautilus-postgres",
                        "--format",
                        "{{.Names}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=docker_timeout,
                    log=logger,
                )
                result_stdout = _decode_stream(result.stdout)
            except SubprocessExecutionError as exc:
                result_stdout = _decode_stream(exc.stdout)
                logger.warning(
                    "docker_ps_failed returncode=%s stdout_tail=%s",
                    exc.returncode,
                    result_stdout[-200:],
                    exc_info=True,
                )

            if "nautilus-postgres" in result_stdout:
                run_command(
                    [docker_path, "start", "nautilus-postgres"],
                    timeout=docker_timeout,
                    log=logger,
                )
            else:
                run_command(
                    [
                        docker_path,
                        "run",
                        "-d",
                        "--name",
                        "nautilus-postgres",
                        "-e",
                        "POSTGRES_PASSWORD=postgres",
                        "-e",
                        "POSTGRES_DB=nautilus",
                        "-p",
                        "5432:5432",
                        "postgres:15",
                    ],
                    timeout=docker_timeout,
                    log=logger,
                )

        # Wait for PostgreSQL to be ready
        for _ in range(30):
            if self._is_postgres_running():
                logger.info("PostgreSQL is ready!")
                return
            time.sleep(1)

        raise RuntimeError("PostgreSQL failed to start within 30 seconds")

    def _run_migrations(self) -> None:
        """
        Run database migrations using the shared CLI plan builder.
        """
        logger.info("Running database migrations...")

        # Decide plan from environment
        import os

        env_full = os.getenv("ML_MIGRATIONS_FULL", "").lower() in {"1", "true", "yes"}
        # Prefer full migrations in production-like environments
        env_mode = os.getenv("ML_ENV", "").lower()
        full = env_full or env_mode in {"prod", "production"}
        schema_env = os.getenv("ML_MIGRATIONS_SCHEMA", MigrationSchema.BOTH.value)
        try:
            schema_enum = MigrationSchema(schema_env)
        except ValueError:
            schema_enum = MigrationSchema.BOTH

        engine = EngineManager.get_engine(self.db_connection)

        # Prefer the CLI helpers to keep in sync
        try:
            from ml.cli.apply_migrations import apply_files as _apply
            from ml.cli.apply_migrations import build_plan as _build

            plan = _build(include_optional=full, schema=schema_enum)
            result = _apply(engine, plan, dry_run=False)
            logger.info(
                "Migrations applied=%d skipped=%d warnings=%d errors=%d",
                result.applied,
                result.skipped,
                result.warnings,
                result.errors,
            )
        except Exception as exc:
            # Fallback to the former inlined list with simple splitting
            logger.warning("CLI migration helpers unavailable (%s); using fallback plan", exc)
            migrations = [
                "ml/registry/migrations/001_initial_schema.sql",
                "ml/registry/migrations/002_add_cold_path_fields.sql",
                "ml/registry/migrations/003_add_artifact_digest.sql",
                "ml/stores/migrations/001_bootstrap_schema.sql",
            ]

            # Use the same splitter as the CLI when available
            try:
                from ml.cli.apply_migrations import split_statements as _splitter
            except Exception:

                def _splitter(sql: str) -> Iterable[str]:
                    return [s for s in sql.split(";") if s.strip()]

            for migration_path in migrations:
                migration_file = Path(migration_path)
                if not migration_file.exists():
                    continue

                logger.info("Applying migration: %s", migration_path)
                sql = migration_file.read_text(encoding="utf-8")
                try:
                    with engine.begin() as conn:
                        for statement in _splitter(sql):
                            try:
                                conn.execute(text(statement))
                            except Exception as e:
                                msg = str(e).lower()
                                if (
                                    "already exists" in msg
                                    or "does not exist" in msg
                                    or "duplicate" in msg
                                ):
                                    logger.debug("Migration notice (%s): %s", migration_path, e)
                                else:
                                    logger.warning("Warning in migration %s: %s", migration_path, e)
                except Exception as exc2:
                    logger.error("Migration failed for %s: %s", migration_path, exc2)
        # Best-effort: proactively create current/future partitions via PartitionManager
        try:
            if self.partition_manager is None:
                self._init_partition_manager()
            if self.partition_manager is not None:
                stats = self.partition_manager.run_maintenance()
                logger.info("Partition maintenance: %s", stats)
        except Exception as exc:
            logger.warning("Partition maintenance skipped: %s", exc)

    def _maybe_run_backfill_on_start(self) -> None:
        """
        Optionally run a gap backfill on startup using CLI or orchestrator, controlled
        by env.

        Environment flags:
        - ML_BACKFILL_ON_START: '1'|'true'|'yes' → enable
        - COVERAGE_MODE: 'sql'|'catalog' (default 'sql')
        - WRITE_MODE: 'sql' (default 'sql')
        - CATALOG_PATH: required for coverage-mode 'catalog'
        - DATABENTO_API_KEY: for client-mode 'databento' (optional)
        - INGEST_CLIENT_MODE: 'catalog'|'databento'|'noop' (default 'catalog')
        - BACKFILL_LOOKBACK_DAYS: integer (default 7)
        - BACKFILL_DATASET_ID: dataset id (e.g., 'EQUS.MINI')
        - BACKFILL_SCHEMA: 'bars'|'tbbo'|'trades' (default 'bars')
        - BACKFILL_INSTRUMENTS: comma-separated list

        """
        import os
        import shlex

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
        also_write_catalog = os.getenv("ALSO_WRITE_CATALOG", "").lower() in {"1", "true", "yes"}

        # Prefer invoking CLI for simplicity and isolation
        cmd = [
            "python",
            "-m",
            "ml.cli.ingest_backfill",
            "--db",
            self.db_connection,
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
            run_command(cmd, timeout=None, log=logger)
        except SubprocessExecutionError as exc:
            stdout_tail = _decode_stream(exc.stdout)[-200:]
            logger.warning(
                "Backfill CLI failed returncode=%s stdout_tail=%s",
                exc.returncode,
                stdout_tail,
                exc_info=True,
            )

            if self.partition_manager is None:
                self._init_partition_manager()
            if self.partition_manager is not None:
                try:
                    stats = self.partition_manager.run_maintenance()
                    logger.info("Partition maintenance: %s", stats)
                except Exception as maintenance_exc:
                    logger.warning("Partition maintenance skipped: %s", maintenance_exc)

    def ensure_healthy(self) -> None:
        """
        Ensure all components are healthy.
        """
        health = self.check_health()

        unhealthy = [k for k, v in health.items() if not v]
        if unhealthy:
            raise RuntimeError(f"Unhealthy components: {unhealthy}")

        logger.info("All ML components are healthy!")

    def _validate_protocol_compliance(self, strict: bool | None = None) -> None:
        """
        Validate MLComponentProtocol compliance for core components.

        Parameters
        ----------
        strict : bool | None
            If True, raise on violations. If None, read from env
            `ML_STRICT_PROTOCOL_VALIDATION` (defaults to False).

        """
        import os

        if strict is None:
            strict = os.getenv("ML_STRICT_PROTOCOL_VALIDATION", "").lower() in {"1", "true", "yes"}

        components: dict[str, Any] = {
            "feature_store": self.feature_store,
            "model_store": self.model_store,
            "strategy_store": self.strategy_store,
            "data_store": self.data_store,
            "feature_registry": self.feature_registry,
            "model_registry": self.model_registry,
            "strategy_registry": self.strategy_registry,
            "data_registry": self.data_registry,
        }

        violations: dict[str, list[str]] = {}

        for name, comp in components.items():
            issues: list[str] = []
            if comp is None or not isinstance(comp, MLComponentProtocol):
                issues.append("does_not_implement_protocol")
            else:
                try:
                    _ = comp.get_health_status()
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"health_status_error:{e}")
                try:
                    _ = comp.get_performance_metrics()
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"performance_metrics_error:{e}")
                try:
                    config_issues = comp.validate_configuration()
                    if config_issues:
                        issues.extend([f"config:{i}" for i in config_issues])
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"validate_configuration_error:{e}")

            if issues:
                violations[name] = issues

        if violations:
            msg = f"Protocol compliance issues: {violations}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    def aggregate_health(self) -> HealthSummary:
        """
        Aggregate component health into domain and system summaries.

        Collects health status from all stores and registries, computes domain-level
        aggregates, and provides overall system status. Components implementing
        MLComponentProtocol report detailed metrics; others return basic status.

        Returns
        -------
        HealthSummary
            Typed dictionary with keys:
            - components: dict[str, ComponentHealthStatus] - per-component health
            - domains: HealthDomains - health by domain (data, features, model, strategy)
            - system: SystemHealth - overall system health and unhealthy component list

        Example
        -------
        >>> integration = MLIntegrationManager()
        >>> health = integration.aggregate_health()
        >>> assert health["system"]["healthy"]  # Type-safe access

        """

        def _comp_health(comp: object) -> ComponentHealthStatus:
            healthy = True
            health: dict[str, object] | None = None
            metrics: dict[str, float] | None = None
            if isinstance(comp, MLComponentProtocol):
                try:
                    health = comp.get_health_status()
                    # Check if health status indicates unhealthy
                    if health and isinstance(health, dict):
                        status_value = health.get("status")
                        if status_value == "unhealthy":
                            healthy = False
                except Exception:
                    healthy = False
                try:
                    metrics = comp.get_performance_metrics()
                except Exception:
                    metrics = None
            return ComponentHealthStatus(
                healthy=healthy,
                health=health or {},
                metrics=metrics or {},
            )

        components: dict[str, ComponentHealthStatus] = {}
        comp_map: dict[str, object] = {
            "feature_store": getattr(self, "feature_store", None),
            "model_store": getattr(self, "model_store", None),
            "strategy_store": getattr(self, "strategy_store", None),
            "data_store": getattr(self, "data_store", None),
            "feature_registry": getattr(self, "feature_registry", None),
            "model_registry": getattr(self, "model_registry", None),
            "strategy_registry": getattr(self, "strategy_registry", None),
            "data_registry": getattr(self, "data_registry", None),
        }

        for name, comp in comp_map.items():
            components[name] = (
                _comp_health(comp)
                if comp is not None
                else ComponentHealthStatus(
                    healthy=False,
                    health={},
                    metrics={},
                )
            )

        def _domain_healthy(keys: list[str]) -> bool:
            return all(components[k]["healthy"] for k in keys if k in components)

        domains: HealthDomains = HealthDomains(
            data=DomainHealth(
                components=["data_store", "data_registry"],
                healthy=_domain_healthy(["data_store", "data_registry"]),
            ),
            features=DomainHealth(
                components=["feature_store", "feature_registry"],
                healthy=_domain_healthy(["feature_store", "feature_registry"]),
            ),
            model=DomainHealth(
                components=["model_store", "model_registry"],
                healthy=_domain_healthy(["model_store", "model_registry"]),
            ),
            strategy=DomainHealth(
                components=["strategy_store", "strategy_registry"],
                healthy=_domain_healthy(["strategy_store", "strategy_registry"]),
            ),
        )

        unhealthy_components = [name for name, info in components.items() if not info["healthy"]]
        system = SystemHealth(
            healthy=len(unhealthy_components) == 0,
            unhealthy=unhealthy_components,
        )

        return HealthSummary(
            components=components,
            domains=domains,
            system=system,
        )

    def check_health(self) -> dict[str, bool]:
        """
        Check health of all components.

        Returns
        -------
        dict[str, bool]
            Health status of each component

        """
        health = {}

        # Check database
        health["postgres"] = self._is_postgres_running()

        # Check stores
        health["feature_store"] = self._check_store_health(self.feature_store)
        health["model_store"] = self._check_store_health(self.model_store)
        health["strategy_store"] = self._check_store_health(self.strategy_store)

        # Check registries
        health["feature_registry"] = self._check_registry_health(
            self.feature_registry,
            "list_features",
        )
        health["model_registry"] = self._check_registry_health(self.model_registry, "list_models")
        health["strategy_registry"] = self._check_registry_health(
            self.strategy_registry,
            "list_strategies",
        )
        health["data_registry"] = self._check_registry_health(self.data_registry, "list_datasets")

        # Check DataStore
        health["data_store"] = self._check_data_store_health()

        # Check partitions
        health["partitions"] = self._check_partition_health()

        return health

    def _check_store_health(self, store: object) -> bool:
        """
        Check health of a store component.
        """
        try:
            # Prefer get_statistics() if available, else try is_healthy()
            if hasattr(store, "get_statistics") and callable(store.get_statistics):
                store.get_statistics()
                return True
            return bool(getattr(store, "is_healthy", lambda: False)())
        except Exception:
            return False

    def _check_registry_health(self, registry: object, method_name: str) -> bool:
        """
        Check health of a registry component.
        """
        try:
            method = getattr(registry, method_name, None)
            if method_name == "list_datasets":
                return bool(method and callable(method))
            return bool(method and callable(method) and method())
        except Exception:
            return False

    def _check_data_store_health(self) -> bool:
        """
        Check health of DataStore component.
        """
        try:
            return bool(self.data_store and hasattr(self.data_store, "registry"))
        except Exception:
            return False

    def _check_partition_health(self) -> bool:
        """
        Check health of partition manager.
        """
        try:
            if self.partition_manager is None:
                return False
            stats = self.partition_manager.get_partition_stats()
            return len(stats) > 0
        except Exception:
            return False

    def create_integrated_actor(self, actor_class: type[ActorT], config: object) -> ActorT:
        """
        Create an actor with automatic integration.

        Creates an instance of the given actor class with automatic initialization
        of all ML stores and registries via the base class.

        Parameters
        ----------
        actor_class : type[ActorT]
            The actor class to instantiate (should extend BaseMLInferenceActor)
        config : object
            Actor configuration object (should include db_connection and other
            configuration attributes matching actor_class expectations)

        Returns
        -------
        ActorT
            Instance of actor_class with all stores automatically connected

        Example
        -------
        >>> from ml.config.actors import MyActorConfig
        >>> from ml.actors.signal import MLSignalActor
        >>> integration = MLIntegrationManager()
        >>> config = MyActorConfig(db_connection="...")
        >>> actor = integration.create_integrated_actor(MLSignalActor, config)
        >>> # mypy knows actor is MLSignalActor, not just 'object'!
        >>> actor.predict(bar)  # IDE autocomplete works!

        """
        # Ensure config has the database connection
        if not hasattr(config, "db_connection"):
            # Best-effort attach db_connection for consumers expecting it
            import logging

            try:
                setattr(config, "db_connection", self.db_connection)
            except Exception:
                logging.exception("Failed to attach db_connection to config")

        # Create actor - stores are automatically initialized by the base class
        # mypy can't verify generic constructor signature, but all actors accept config
        actor = actor_class(config=config)  # type: ignore[call-arg]

        return actor  # Type inferred as ActorT from actor_class parameter

    def shutdown(self) -> None:
        """
        Gracefully shutdown all components.
        """
        # Flush all pending writes
        if hasattr(self.feature_store, "flush"):
            self.feature_store.flush()
        if hasattr(self.model_store, "flush"):
            self.model_store.flush()
        if hasattr(self.strategy_store, "flush"):
            self.strategy_store.flush()
        if self.data_store is not None and hasattr(self.data_store, "flush"):
            self.data_store.flush()

        logger.info("ML integration manager shutdown complete")

    # ---------------------------------------------------------------------
    # TDD prototype convenience hooks (no-op stubs)
    # ---------------------------------------------------------------------

    def configure_message_bus(
        self,
        *,
        backend: str | None = None,
        topic_prefix: str | None = None,
        retention_hours: int | None = None,
        max_size_mb: int | None = None,
    ) -> None:
        """
        No-op configuration stub for message bus (for tests).
        """
        _ = (backend, topic_prefix, retention_hours, max_size_mb)
        return None

    def configure_event_emission(
        self,
        *,
        batching_enabled: bool | None = None,
        batch_size: int | None = None,
        flush_interval_ms: int | None = None,
        correlation_strategy: str | None = None,
    ) -> None:
        """
        No-op configuration stub for event emission (for tests).
        """
        _ = (batching_enabled, batch_size, flush_interval_ms, correlation_strategy)
        return None

    def configure_event_system(self, **_: object) -> None:
        """
        No-op aggregate configuration for event system (for tests).
        """
        return None

    def configure_domain_bookkeeping(self, _config: object) -> None:
        """
        No-op configuration stub for domain bookkeeping (for tests).
        """
        return None

    def initialize_observability_pipeline(self) -> None:
        """
        Initialize a lightweight observability service (off hot-path).
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
        """
        return None

    def start_health_checks(self) -> None:
        """
        No-op start of health monitoring (for tests).
        """
        return None

    def collect_observability_dataframes(self) -> dict[str, PdDataFrame | None]:
        """
        Materialize observability DataFrames from the service, if available.

        Returns a mapping of table name -> DataFrame. When the service is not
        initialized, returns empty DataFrames.

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

        Writes non-empty tables under `base_path` using the specified format
        ("jsonl" or "csv"). Returns a mapping of table name to file path for
        written tables.

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

        Uses `ObservabilityDBPersistor` to write non-empty tables to a relational
        store (e.g., SQLite/PostgreSQL) and returns a mapping of table name to
        number of rows written.

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

        When ``interval_seconds`` is
        None or <= 0, performs a single flush and returns the written mapping.
        Otherwise, starts a background thread managed by the integration instance.

        """
        # Ensure service exists
        self.initialize_observability_pipeline()

        # Inject observability service into stores for stage boundary tracking
        self._inject_observability_service_into_stores()

        if interval_seconds is None or interval_seconds <= 0:
            return self.flush_observability_to_path(base_path=base_path, file_format=file_format)

        # Background scheduler (off hot-path)
        from threading import Event

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

    def _inject_observability_service_into_stores(self) -> None:
        """
        Inject the observability service into all stores for stage boundary tracking.

        This enables stores to record latency and metrics for cold path operations when
        observability is enabled via ML_OBSERVABILITY_ENABLED environment variable.

        """
        try:
            obs_service = getattr(self, "observability_service", None)
            if obs_service is None:
                return

            # Inject observability service into all stores
            stores = [
                getattr(self, "feature_store", None),
                getattr(self, "model_store", None),
                getattr(self, "strategy_store", None),
                getattr(self, "data_store", None),
            ]

            for store in stores:
                if store is not None:
                    # Set the observability service as a private attribute
                    setattr(store, "_observability_service", obs_service)

            logger.debug(
                "Injected observability service into %d stores",
                len([s for s in stores if s]),
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
                self._inject_observability_service_into_stores()
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
        """
        try:
            worker = getattr(self, "_obs_async_worker", None)
            if worker is not None:
                import asyncio

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
            - running: bool
            - queue_size: int (0 when not running)

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
        """
        try:
            from ml.config.observability import ObservabilityConfig

            cfg = ObservabilityConfig.from_env()
            self.start_observability_from_config(cfg)
        except Exception:  # pragma: no cover - defensive
            return None

    def emit_cross_domain_event(self, _event: dict[str, object]) -> None:
        """
        No-op cross-domain event emitter stub (for tests).
        """
        return None

    def emit_cascade(
        self,
        source_event: dict[str, object],
        target_domain: str,
        *,
        delay_ns: int | None = None,
    ) -> dict[str, object]:
        """
        Create a cascaded event preserving correlation and timestamp order.

        This adapter delegates to a light helper in ``ml.common.cascade`` to
        avoid deep coupling and keep hot paths unaffected.

        """
        from typing import Any, cast

        from ml.common.cascade import EventDict  # Local import to avoid cycles
        from ml.common.cascade import emit_cascade as _emit_cascade  # Local import to avoid cycles

        ev: EventDict = EventDict(
            domain=cast(str, source_event.get("domain", "")),
            event_type=cast(str, source_event.get("event_type", "")),
            correlation_id=cast(str, source_event.get("correlation_id", "")),
            instrument_id=cast(str, source_event.get("instrument_id", "")),
            ts_event=int(cast(Any, source_event.get("ts_event", 0))),
            source_event_id=cast(
                str,
                source_event.get("event_id", source_event.get("source_event_id", "unknown")),
            ),
            payload=cast(dict[str, Any], source_event.get("payload", {}) or {}),
        )
        out = _emit_cascade(ev, target_domain, delay_ns)
        return dict(out)

    def set_message_publisher(self, publisher: object) -> None:
        """
        Configure the message publisher for ML stores which support it.

        Currently applies to ``DataStore`` only. Safe to call at any time; if
        the store is not initialized yet, this method is a no-op.

        """
        # Avoid strict typing dependency; use duck-typing and assign when attribute exists
        if hasattr(self, "data_store") and hasattr(self.data_store, "publisher"):
            try:
                cast(Any, self.data_store).publisher = publisher
            except Exception:
                logger.debug("Failed to attach publisher to data_store", exc_info=True)


# Singleton instance for global access
_integration_manager: MLIntegrationManager | None = None


def get_integration_manager(config: HasDBConnection | None = None) -> MLIntegrationManager:
    """
    Get or create the global integration manager.

    Parameters
    ----------
    config : HasDBConnection, optional
        Configuration (only used on first call)

    Returns
    -------
    MLIntegrationManager
        The global integration manager instance

    """
    global _integration_manager

    if _integration_manager is None:
        _integration_manager = MLIntegrationManager(config)

    return _integration_manager


def reset_integration_manager() -> None:
    """
    Reset the global integration manager.
    """
    global _integration_manager

    if _integration_manager is not None:
        _integration_manager.shutdown()
        _integration_manager = None


# ======================================================================================
# Lightweight initializer for actors (centralizes progressive fallback + wiring)
# ======================================================================================


@dataclass(slots=True)
class ActorStoresRegistries:
    """
    Simple container for actor-attached stores and registries.

    This dataclass groups the primary store and registry instances provided to ML actors
    after applying progressive fallback (PRIMARY → CACHED → FILE → DUMMY). It also
    carries persistence and connection information discovered during initialization.

    All store and registry fields are properly typed for maximum type safety and IDE
    autocomplete support. This enables type-safe dependency injection for ML components.

    Example
    -------
    >>> from ml.core.integration import init_ml_stores_and_registries
    >>> config = MyMLConfig()
    >>> stores = init_ml_stores_and_registries(config)
    >>> assert isinstance(stores, ActorStoresRegistries)
    >>> assert isinstance(stores.feature_store, FeatureStore)  # Type-safe!
    >>> stores.feature_store.write_features(...)  # Full IDE autocomplete works!

    """

    feature_store: FeatureStore
    model_store: ModelStore
    strategy_store: StrategyStore
    data_store: DataStore
    feature_registry: FeatureRegistry
    model_registry: ModelRegistry
    strategy_registry: StrategyRegistry
    data_registry: DataRegistry
    persistence_config: PersistenceConfig | None
    connection_string: str | None


def init_ml_stores_and_registries(config: Any) -> ActorStoresRegistries:
    """
    Initialize ML stores and registries with progressive fallback chains.

    This function implements the Universal ML Architecture Pattern 1 by providing
    centralized initialization of all 4 stores (Feature, Model, Strategy, Data) and
    4 registries with automatic fallback handling.

    The function supports dependency injection for any ML component that needs
    access to stores and registries, not just actors. This enables clean separation
    of concerns without forcing inheritance hierarchies.

    Parameters
    ----------
    config : Any
        Configuration object with the following optional attributes:
        - use_dummy_stores (bool): Use dummy stores for testing (fast path)
        - db_connection (str | None): PostgreSQL connection string
        - allow_dummy_fallback (bool): Allow fallback to dummy stores on connection failure

    Returns
    -------
    ActorStoresRegistries
        Dataclass containing all 4 stores and 4 registries, along with
        persistence configuration and connection information.

    Progressive Fallback Chain
    --------------------------
    1. PRIMARY: PostgreSQL with full persistence
    2. CACHED: Local cache with periodic sync (future)
    3. FILE: File-based storage (future)
    4. DUMMY: In-memory stores for testing/development

    Examples
    --------
    >>> # Direct usage in any component
    >>> stores = init_ml_stores_and_registries(config)
    >>> feature_store = stores.feature_store

    >>> # Dependency injection in FeatureEngineer
    >>> class FeatureEngineer:
    ...     def __init__(self, config, stores=None):
    ...         self.stores = stores or init_ml_stores_and_registries(config)

    Notes
    -----
    This function was renamed from init_actor_stores_and_registries to better
    reflect its general-purpose nature for all ML components, not just actors.

    """
    # Local imports to avoid import-time cycles
    from ml.registry import DataRegistry
    from ml.registry import FeatureRegistry
    from ml.registry import ModelRegistry
    from ml.registry import StrategyRegistry
    from ml.registry.persistence import BackendType
    from ml.registry.persistence import PersistenceConfig

    # Fast-path for tests
    if bool(getattr(config, "use_dummy_stores", False)):
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        return ActorStoresRegistries(
            feature_store=cast(FeatureStore, DummyStore()),
            model_store=cast(ModelStore, DummyStore()),
            strategy_store=cast(StrategyStore, DummyStore()),
            data_store=cast(DataStore, DummyStore()),
            feature_registry=cast(FeatureRegistry, DummyRegistry()),
            model_registry=cast(ModelRegistry, DummyRegistry()),
            strategy_registry=cast(StrategyRegistry, DummyRegistry()),
            data_registry=cast(DataRegistry, DummyRegistry()),
            persistence_config=None,
            connection_string=None,
        )

    # Progressive fallback
    db_connection = cast(str | None, getattr(config, "db_connection", None))
    backend = BackendType.POSTGRES
    if not db_connection:
        try:
            test = EngineManager.get_engine(
                "postgresql://postgres:postgres@localhost:5432/nautilus",
            )
            with test.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connection = "postgresql://postgres:postgres@localhost:5432/nautilus"
        except Exception:
            backend = BackendType.JSON
            db_connection = ""
            try:
                # Emit fallback activation metric (no-op if metrics backend missing)
                from ml.common.metrics_bootstrap import get_counter

                get_counter(
                    "ml_fallback_activations_total",
                    "Fallback activations",
                    labelnames=("component", "level"),
                ).labels(component="actor_stores", level="dummy").inc()
            except Exception as metric_exc:
                # Metrics must not affect control flow — debug only
                logger.debug(
                    "Fallback metric emit failed (initial probe): %s",
                    metric_exc,
                    exc_info=True,
                )

    # If provided, probe reachability
    if db_connection and backend == BackendType.POSTGRES:
        try:
            eng = EngineManager.get_engine(str(db_connection))
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            if getattr(config, "allow_dummy_fallback", True):
                backend = BackendType.JSON
                try:
                    from ml.common.metrics_bootstrap import get_counter

                    get_counter(
                        "ml_fallback_activations_total",
                        "Fallback activations",
                        labelnames=("component", "level"),
                    ).labels(component="actor_stores", level="dummy").inc()
                except Exception as metric_exc:
                    logger.debug(
                        "Fallback metric emit failed (dummy backend activation): %s",
                        metric_exc,
                        exc_info=True,
                    )
            else:
                raise

    if backend == BackendType.JSON:
        file_store_path = Path(
            getattr(
                config,
                "file_store_path",
                os.getenv("ML_FILE_STORE_PATH", Path.home() / ".nautilus" / "ml" / "file_store"),
            ),
        )
        try:
            file_store_path.mkdir(parents=True, exist_ok=True)
            from ml.registry.base import DummyRegistry
            from ml.stores.file_backed import FileDataStore
            from ml.stores.file_backed import FileFeatureStore
            from ml.stores.file_backed import FileModelStore
            from ml.stores.file_backed import FileStrategyStore

            registry_root = file_store_path / "registry"
            registry_root.mkdir(parents=True, exist_ok=True)
            json_persistence = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_root,
            )
            freg = FeatureRegistry(registry_root / "features", persistence_config=json_persistence)
            mreg = ModelRegistry(registry_root / "models", persistence_config=json_persistence)
            sreg = StrategyRegistry(registry_root, persistence_config=json_persistence)
            dreg = DataRegistry(registry_root / "datasets", persistence_config=json_persistence)
            feature_store_file = FileFeatureStore(base_path=file_store_path / "features")
            model_store_file = FileModelStore(base_path=file_store_path / "models")
            strategy_store_file = FileStrategyStore(base_path=file_store_path / "strategies")
            data_store_file = FileDataStore(base_path=file_store_path / "datastore")
            try:
                from ml.common.metrics_bootstrap import get_counter

                get_counter(
                    "ml_fallback_activations_total",
                    "Fallback activations",
                    labelnames=("component", "level"),
                ).labels(component="actor_stores", level="file").inc()
            except Exception:
                logger.debug("File fallback metric emit failed", exc_info=True)

            return ActorStoresRegistries(
                feature_store=cast(FeatureStore, feature_store_file),
                model_store=cast(ModelStore, model_store_file),
                strategy_store=cast(StrategyStore, strategy_store_file),
                data_store=cast(DataStore, data_store_file),
                feature_registry=freg,
                model_registry=mreg,
                strategy_registry=sreg,
                data_registry=dreg,
                persistence_config=None,
                connection_string=db_connection,
            )
        except Exception:
            logger.debug("File-backed fallback unavailable for actor stores", exc_info=True)

        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        return ActorStoresRegistries(
            feature_store=cast(FeatureStore, DummyStore()),
            model_store=cast(ModelStore, DummyStore()),
            strategy_store=cast(StrategyStore, DummyStore()),
            data_store=cast(DataStore, DummyStore()),
            feature_registry=cast(FeatureRegistry, DummyRegistry()),
            model_registry=cast(ModelRegistry, DummyRegistry()),
            strategy_registry=cast(StrategyRegistry, DummyRegistry()),
            data_registry=cast(DataRegistry, DummyRegistry()),
            persistence_config=None,
            connection_string=db_connection,
        )

    # Production wiring with PostgreSQL
    persistence_config = PersistenceConfig(
        backend=BackendType.POSTGRES,
        connection_string=db_connection,
    )
    # Stores
    fs = FeatureStore(connection_string=db_connection)
    ms = ModelStore(persistence_config=persistence_config)
    ss = StrategyStore(persistence_config=persistence_config)

    # Registries under a local path
    registry_path = Path(".nautilus/ml/registry")
    registry_path.mkdir(parents=True, exist_ok=True)
    freg = FeatureRegistry(registry_path, persistence_config=persistence_config)
    mreg = ModelRegistry(registry_path, persistence_config=persistence_config)
    sreg = StrategyRegistry(registry_path)
    dreg = DataRegistry(registry_path / "datasets", persistence_config=persistence_config)

    # Inject shared DataRegistry into stores if supported
    try:
        setter = getattr(fs, "set_data_registry", None)
        if callable(setter):
            setter(dreg)
        setter2 = getattr(ms, "set_data_registry", None)
        if callable(setter2):
            setter2(dreg)
    except Exception:
        logger.debug("Failed to inject shared DataRegistry into stores", exc_info=True)

    dstore = create_data_store(registry=dreg, connection_string=db_connection)

    return ActorStoresRegistries(
        feature_store=fs,
        model_store=ms,
        strategy_store=ss,
        data_store=dstore,
        feature_registry=freg,
        model_registry=mreg,
        strategy_registry=sreg,
        data_registry=dreg,
        persistence_config=persistence_config,
        connection_string=db_connection,
    )


# Backward compatibility alias (deprecated)
init_actor_stores_and_registries = init_ml_stores_and_registries
"""Deprecated: Use init_ml_stores_and_registries instead."""


# ----------------------------------------------------------------------------
# Factory to create DataStore without tripping mypy abstract instantiation checks
# ----------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def create_data_store(
    *,
    registry: DataRegistry,
    connection_string: str,
    raw_reader: RawReaderProtocol | None = None,
    raw_writer: RawIngestionWriterProtocol | None = None,
) -> DataStore:
    """
    Create a DataStore instance with proper type safety.

    This factory function initializes a DataStore with automatic integration
    of all registry and data reader/writer components. It returns the instance
    with full type information for IDE autocomplete and mypy verification.

    Parameters
    ----------
    registry : DataRegistry
        The data registry for dataset manifest and lineage tracking
    connection_string : str
        PostgreSQL connection string for raw market data queries
    raw_reader : RawReaderProtocol | None, optional
        Reader for raw market data (e.g., SQL or Parquet catalog)
    raw_writer : RawIngestionWriterProtocol | None, optional
        Writer for market data (used for backfill and sync operations)

    Returns
    -------
    DataStore
        Initialized data store with full type information

    Example
    -------
    >>> from ml.core.integration import create_data_store
    >>> from ml.registry import DataRegistry
    >>> from ml.stores.io_raw import ParquetCatalogRawReader
    >>>
    >>> registry = DataRegistry(...)
    >>> reader = ParquetCatalogRawReader(...)
    >>> store = create_data_store(
    ...     registry=registry,
    ...     connection_string="postgresql://...",
    ...     raw_reader=reader,
    ... )
    >>> store.read_range(...)  # Proper IDE autocomplete!
    """
    from ml.stores.data_store import DataStore

    return DataStore(
        connection_string=connection_string,
        registry=registry,
        raw_reader=raw_reader,
        raw_writer=raw_writer,
    )
