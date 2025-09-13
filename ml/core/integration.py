"""
ML System Integration Manager.

This module provides automatic integration of all ML components including stores,
registries, and database connections. It ensures that all data flows are automatically
connected and persisted.

"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ml.common.protocols import MLComponentProtocol
from ml.core.db_engine import EngineManager
from ml.registry import DataRegistry
from ml.registry import FeatureRegistry
from ml.registry import ModelRegistry
from ml.registry import StrategyRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.partition_manager import PartitionManager
from ml.stores.raw_io_parquet import ParquetCatalogRawReader
from ml.stores.raw_io_parquet import ParquetCatalogRawWriter
from ml.stores.strategy_store import StrategyStore


logger = logging.getLogger(__name__)


@runtime_checkable
class HasDBConnection(Protocol):
    """
    Protocol for configs carrying an optional DB connection string.
    """

    db_connection: str | None


class MLIntegrationManager:
    """
    Automatically wires all ML components together.

    This manager ensures that:
    1. PostgreSQL is running (or starts it)
    2. All migrations are applied
    3. All stores are initialized
    4. All registries are connected
    5. Data flows are automatic

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
        # Use provided connection or default
        self.db_connection = (
            db_connection
            or (config.db_connection if config else None)
            or "postgresql://postgres:postgres@localhost:5432/nautilus"
        )

        # Allow environment variables to opt-in
        import os

        env_start = os.getenv("ML_AUTO_START_DB", "").lower() in {"1", "true", "yes"}
        env_migrate = os.getenv("ML_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
        self._allow_dummy = os.getenv("ML_ALLOW_DUMMY", "").lower() in {"1", "true", "yes"}
        self.auto_start_postgres = auto_start_postgres or env_start
        self.auto_migrate = auto_migrate or env_migrate

        # Initialize components with progressive fallback when enabled
        if not self._is_postgres_running():
            if self.auto_start_postgres:
                self._start_postgres_container()
            elif self._allow_dummy:
                logger.warning(
                    "PostgreSQL unavailable; ML_ALLOW_DUMMY enabled — using Dummy stores/registries (no persistence)",
                )
                self._init_dummy_components()
                if ensure_healthy:
                    self.ensure_healthy()
                self._validate_protocol_compliance(strict=strict_protocol_validation)
                return
            else:
                raise RuntimeError(
                    "PostgreSQL is not running. Start it or set ML_ALLOW_DUMMY=1 for in-memory fallback",
                )

        # Normal path (PostgreSQL available)
        self._init_database()
        self._init_stores()
        self._init_registries()
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

    def _init_dummy_components(self) -> None:
        """
        Initialize in-memory dummy components for testing fallback.

        This mode provides protocol-compatible components without persistence.

        """
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        # Stores
        self.feature_store = DummyStore()
        self.model_store = DummyStore()
        self.strategy_store = DummyStore()
        self.data_store = DummyStore()

        # Registries
        self.feature_registry = DummyRegistry()
        self.model_registry = DummyRegistry()
        self.strategy_registry = DummyRegistry()
        self.data_registry = DummyRegistry()

        # Partition manager is not applicable in dummy mode
        self.partition_manager = None

    def _init_stores(self) -> None:
        """
        Initialize all store components.
        """
        # Create persistence config
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
        # Create persistence config for registries
        persistence_config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=self.db_connection,
        )

        # Create a registry path (for file storage)
        registry_path = Path("./ml_registry")
        registry_path.mkdir(parents=True, exist_ok=True)

        # Initialize registries with PostgreSQL backend
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

        # Now initialize DataStore with the registry
        # Optionally attach raw adapters when a catalog path is provided
        raw_reader = None
        raw_writer = None
        try:  # best-effort; keep init resilient
            import os

            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

            catalog_path = os.getenv("CATALOG_PATH", "").strip()
            if catalog_path:
                catalog = ParquetDataCatalog(catalog_path)
                raw_reader = ParquetCatalogRawReader(catalog)
                raw_writer = ParquetCatalogRawWriter(catalog)
        except Exception:
            logger.debug("Parquet catalog adapters not attached", exc_info=True)

        self.data_store = DataStore(
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
        Check if PostgreSQL is accessible.
        """
        try:
            engine = EngineManager.get_engine(self.db_connection)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except OperationalError:
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
                # Fallback silently to next candidate
                continue

        if compose_file is not None:
            try:
                subprocess.run(
                    [docker_path, "compose", "-f", str(compose_file), "up", "-d", "postgres"],
                    check=True,
                )
            except Exception:
                compose_file = None

        if compose_file is None:
            result = subprocess.run(
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
            )

            if "nautilus-postgres" in result.stdout:
                subprocess.run([docker_path, "start", "nautilus-postgres"], check=True)
            else:
                subprocess.run(
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
                    check=True,
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
        Run all database migrations using a robust SQL splitter.
        """
        logger.info("Running database migrations...")

        migrations = [
            "ml/registry/migrations/001_initial_schema.sql",
            "ml/stores/migrations/001_stores_schema.sql",
            "ml/stores/migrations/002_auto_partitioning.sql",
            "ml/stores/migrations/003_market_data.sql",
            "ml/stores/migrations/004_data_registry.sql",
            "ml/stores/migrations/007_add_event_metadata.sql",
        ]

        engine = EngineManager.get_engine(self.db_connection)

        # Use the same splitter as the CLI migration runner to respect dollar-quoted bodies
        try:
            from ml.scripts.apply_migrations import _split_statements as _splitter
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
            except Exception as exc:
                logger.error("Migration failed for %s: %s", migration_path, exc)

        # Best-effort: proactively create current/future partitions so stores operate immediately
        try:
            # Call SQL helper if present
            with engine.begin() as conn:
                try:
                    conn.execute(text("SELECT auto_create_partitions()"))
                except Exception:
                    # Ignore if function not installed; PartitionManager handles below
                    pass
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

        logger.info("Running backfill bootstrap: %s", shlex.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            logger.warning("Backfill CLI failed: %s", exc)

            if self.partition_manager is None:
                self._init_partition_manager()
            if self.partition_manager is not None:
                stats = self.partition_manager.run_maintenance()
                logger.info("Partition maintenance: %s", stats)
        except Exception as exc:
            logger.warning("Partition maintenance skipped: %s", exc)

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

    def aggregate_health(self) -> dict[str, object]:
        """
        Aggregate component health into domain and system summaries.

        Returns
        -------
        dict[str, object]
            A structured health summary with keys:
            - components: per-component health and metrics (when available)
            - domains: aggregated health per domain (data, features, model, strategy)
            - system: overall status with list of unhealthy components

        """

        def _comp_health(comp: object) -> dict[str, object]:
            healthy = True
            health: dict[str, object] | None = None
            metrics: dict[str, float] | None = None
            if isinstance(comp, MLComponentProtocol):
                try:
                    health = comp.get_health_status()
                except Exception:
                    healthy = False
                try:
                    metrics = comp.get_performance_metrics()
                except Exception:
                    metrics = None
            return {"healthy": healthy, "health": health or {}, "metrics": metrics or {}}

        components: dict[str, dict[str, object]] = {}
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
                else {
                    "healthy": False,
                    "health": {},
                    "metrics": {},
                }
            )

        def _domain_healthy(keys: list[str]) -> bool:
            return all(components[k]["healthy"] for k in keys if k in components)

        domains = {
            "data": {
                "components": ["data_store", "data_registry"],
                "healthy": _domain_healthy(["data_store", "data_registry"]),
            },
            "features": {
                "components": ["feature_store", "feature_registry"],
                "healthy": _domain_healthy(["feature_store", "feature_registry"]),
            },
            "model": {
                "components": ["model_store", "model_registry"],
                "healthy": _domain_healthy(["model_store", "model_registry"]),
            },
            "strategy": {
                "components": ["strategy_store", "strategy_registry"],
                "healthy": _domain_healthy(["strategy_store", "strategy_registry"]),
            },
        }

        unhealthy_components = [name for name, info in components.items() if not info["healthy"]]
        system = {"healthy": len(unhealthy_components) == 0, "unhealthy": unhealthy_components}

        return {"components": components, "domains": domains, "system": system}

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

    def create_integrated_actor(self, actor_class: type[Any], config: object) -> object:
        """
        Create an actor with automatic integration.

        Parameters
        ----------
        actor_class : type
            The actor class to instantiate
        config : Any
            Actor configuration (should include db_connection)

        Returns
        -------
        Any
            Instantiated actor with all stores automatically connected

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
        actor = actor_class(config=config)

        return actor

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

    def collect_observability_dataframes(self) -> dict[str, object]:
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
            from typing import Any as _Any
            from typing import cast as _cast

            from ml.observability.persistence import ObservabilityPersistor

            tables = _cast(dict[str, _Any], self.collect_observability_dataframes())
            # Collect returns DataFrame | None; persist accepts Mapping[str, DataFrame | None]
            sink = ObservabilityPersistor(base_path=base_path, file_format=file_format)
            return sink.persist(tables)
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
            from typing import Any as _Any
            from typing import cast as _cast

            from ml.observability.db_persistence import ObservabilityDBPersistor

            tables = _cast(dict[str, _Any], self.collect_observability_dataframes())
            per = ObservabilityDBPersistor(connection_string=connection_string)
            return per.persist(tables)
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
        if hasattr(self, "data_store") and isinstance(self.data_store, DataStore):
            # Avoid strict typing dependency here; DataStore expects a compatible publisher.
            from typing import Any as _Any

            cast(_Any, self.data_store).publisher = publisher


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
    feature_store: object
    model_store: object
    strategy_store: object
    data_store: object
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object
    persistence_config: PersistenceConfig | None
    connection_string: str | None


def init_actor_stores_and_registries(config: Any) -> ActorStoresRegistries:
    """
    Initialize stores and registries for an actor with progressive fallback.

    Honors `use_dummy_stores` (fast path for tests) and `allow_dummy_fallback`.
    Attempts to probe PostgreSQL if no connection string is provided.

    """
    # Fast-path for tests
    if bool(getattr(config, "use_dummy_stores", False)):
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        return ActorStoresRegistries(
            feature_store=DummyStore(),
            model_store=DummyStore(),
            strategy_store=DummyStore(),
            data_store=DummyStore(),
            feature_registry=DummyRegistry(),
            model_registry=DummyRegistry(),
            strategy_registry=DummyRegistry(),
            data_registry=DummyRegistry(),
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

    # If provided, probe reachability
    if db_connection and backend == BackendType.POSTGRES:
        try:
            eng = EngineManager.get_engine(str(db_connection))
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            if getattr(config, "allow_dummy_fallback", True):
                backend = BackendType.JSON
            else:
                raise

    if backend == BackendType.JSON:
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        return ActorStoresRegistries(
            feature_store=DummyStore(),
            model_store=DummyStore(),
            strategy_store=DummyStore(),
            data_store=DummyStore(),
            feature_registry=DummyRegistry(),
            model_registry=DummyRegistry(),
            strategy_registry=DummyRegistry(),
            data_registry=DummyRegistry(),
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

    dstore = DataStore(registry=dreg, connection_string=db_connection)

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
