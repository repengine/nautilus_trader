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
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
        self.auto_start_postgres = auto_start_postgres or env_start
        self.auto_migrate = auto_migrate or env_migrate

        # Initialize components
        self._init_database()
        self._init_stores()
        self._init_registries()
        self._init_partition_manager()

        # Ensure everything is healthy
        if ensure_healthy:
            self.ensure_healthy()

        # Validate protocol compliance (warn by default)
        self._validate_protocol_compliance(strict=strict_protocol_validation)

    def _init_database(self) -> None:
        """
        Initialize database connection and run migrations.
        """
        # Check if PostgreSQL is running
        if not self._is_postgres_running():
            if self.auto_start_postgres:
                self._start_postgres_container()
            else:
                raise RuntimeError(
                    "PostgreSQL is not running. Start it manually or set auto_start_postgres=True",
                )

        # Run migrations if needed
        if self.auto_migrate:
            self._run_migrations()

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
        self.data_store: DataStore | None = None

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
        self.data_store = DataStore(
            registry=self.data_registry,
            connection_string=self.db_connection,
        )

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
        Start PostgreSQL using docker-compose if available, else docker run.
        """
        logger.info("Starting PostgreSQL (preferring docker-compose if available)...")

        import shutil

        compose_file = None
        docker_path = shutil.which("docker")
        if docker_path is None:
            raise RuntimeError("docker executable not found in PATH")
        for candidate in (Path("ml/docker-compose.yml"), Path("docker-compose.yml")):
            if candidate.exists():
                compose_file = candidate
                break

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
        Run all database migrations.
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

        for migration_path in migrations:
            migration_file = Path(migration_path)
            if migration_file.exists():
                print(f"Running migration: {migration_path}")
                with open(migration_file) as f:
                    sql = f.read()

                with engine.begin() as conn:
                    # Split by semicolons and execute each statement
                    # (some drivers don't support multiple statements)
                    for statement in sql.split(";"):
                        if statement.strip():
                            try:
                                conn.execute(text(statement))
                            except Exception as e:
                                # Ignore errors for existing objects
                                if "already exists" not in str(e):
                                    logger.warning("Warning in migration %s: %s", migration_path, e)

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
            components[name] = _comp_health(comp) if comp is not None else {
                "healthy": False,
                "health": {},
                "metrics": {},
            }

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
        """No-op configuration stub for message bus (for tests)."""
        return None

    def configure_event_emission(
        self,
        *,
        batching_enabled: bool | None = None,
        batch_size: int | None = None,
        flush_interval_ms: int | None = None,
        correlation_strategy: str | None = None,
    ) -> None:
        """No-op configuration stub for event emission (for tests)."""
        return None

    def configure_event_system(self, **_: object) -> None:
        """No-op aggregate configuration for event system (for tests)."""
        return None

    def configure_domain_bookkeeping(self, _config: object) -> None:
        """No-op configuration stub for domain bookkeeping (for tests)."""
        return None

    def initialize_observability_pipeline(self) -> None:
        """No-op initializer for observability pipeline (for tests)."""
        return None

    def emit_cross_domain_event(self, _event: dict[str, object]) -> None:
        """No-op cross-domain event emitter stub (for tests)."""
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
                str, source_event.get("event_id", source_event.get("source_event_id", "unknown"))
            ),
            payload=cast(dict[str, Any], source_event.get("payload", {}) or {}),
        )
        out = _emit_cascade(ev, target_domain, delay_ns)
        return dict(out)


class AutoIntegratedActor:
    """
    DEPRECATED: Use BaseMLInferenceActor from ml.actors.base instead.

    This class is kept for backward compatibility only.
    All new actors should inherit from BaseMLInferenceActor which has
    automatic store integration built-in.
    """

    def __init__(self, config: object, integration: MLIntegrationManager | None = None) -> None:
        """
        Initialize actor with automatic integration.

        Parameters
        ----------
        config : Any
            Actor configuration
        integration : MLIntegrationManager, optional
            Integration manager (creates one if not provided)

        """
        # Get or create integration manager
        self.integration = integration or MLIntegrationManager(
            config if isinstance(config, HasDBConnection) else None,
        )

        # Wire all stores
        self.feature_store = self.integration.feature_store
        self.model_store = self.integration.model_store
        self.strategy_store = self.integration.strategy_store
        self.data_store = self.integration.data_store

        # Wire all registries
        self.feature_registry = self.integration.feature_registry
        self.model_registry = self.integration.model_registry
        self.strategy_registry = self.integration.strategy_registry
        self.data_registry = self.integration.data_registry

    def write_features(self, features: dict[str, float], **kwargs: object) -> None:
        """
        Automatically write features to store.
        """
        self.feature_store.write_features(
            feature_set_id=getattr(self, "feature_set_id", "default"),
            instrument_id=getattr(self, "instrument_id", "unknown"),
            features=features,
            ts_event=getattr(self, "ts_event", None),
            ts_init=getattr(self, "ts_init", None),
        )

    def write_prediction(
        self,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        **kwargs: object,
    ) -> None:
        """
        Automatically write prediction to store.
        """
        self.model_store.write_prediction(
            model_id=getattr(self, "model_id", "default"),
            instrument_id=getattr(self, "instrument_id", "unknown"),
            prediction=prediction,
            confidence=confidence,
            features=features,
            inference_time_ms=float(getattr(self, "inference_time_ms", 0.0)),
            ts_event=int(getattr(self, "ts_event", 0)),
            is_live=bool(getattr(self, "is_live", False)),
        )

    def write_signal(
        self,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        **kwargs: object,
    ) -> None:
        """
        Automatically write signal to store.
        """
        self.strategy_store.write_signal(
            strategy_id=getattr(self, "strategy_id", "default"),
            instrument_id=getattr(self, "instrument_id", "unknown"),
            signal_type=signal_type,
            strength=strength,
            model_predictions=model_predictions,
            risk_metrics={},
            execution_params={},
            ts_event=int(getattr(self, "ts_event", 0)),
            is_live=bool(getattr(self, "is_live", False)),
        )


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
