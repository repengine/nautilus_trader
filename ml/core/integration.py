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
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ml.core.db_engine import EngineManager
from ml.registry import FeatureRegistry
from ml.registry import ModelRegistry
from ml.registry import StrategyRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.partition_manager import PartitionManager
from ml.stores.strategy_store import StrategyStore

logger = logging.getLogger(__name__)


class HasDBConnection(Protocol):
    """Protocol for configs carrying an optional DB connection string."""

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
        auto_start_postgres: bool = True,
        auto_migrate: bool = True,
        ensure_healthy: bool = True,
    ):
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

        self.auto_start_postgres = auto_start_postgres
        self.auto_migrate = auto_migrate

        # Initialize components
        self._init_database()
        self._init_stores()
        self._init_registries()
        self._init_partition_manager()

        # Ensure everything is healthy
        if ensure_healthy:
            self.ensure_healthy()

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
        """Start PostgreSQL using docker-compose if available, else docker run."""
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
        try:
            # FeatureStore exposes is_healthy()
            health["feature_store"] = bool(getattr(self.feature_store, "is_healthy", lambda: False)())
        except Exception:
            health["feature_store"] = False

        try:
            # Prefer get_statistics() if available, else try is_healthy()
            if hasattr(self.model_store, "get_statistics") and callable(self.model_store.get_statistics):
                self.model_store.get_statistics()
                health["model_store"] = True
            else:
                health["model_store"] = bool(getattr(self.model_store, "is_healthy", lambda: False)())
        except Exception:
            health["model_store"] = False

        try:
            if hasattr(self.strategy_store, "get_statistics") and callable(self.strategy_store.get_statistics):
                self.strategy_store.get_statistics()
                health["strategy_store"] = True
            else:
                health["strategy_store"] = bool(getattr(self.strategy_store, "is_healthy", lambda: False)())
        except Exception:
            health["strategy_store"] = False

        # Check registries
        try:
            lf = getattr(self.feature_registry, "list_features", None)
            health["feature_registry"] = bool(lf and callable(lf) and lf())
        except Exception:
            health["feature_registry"] = False

        try:
            lm = getattr(self.model_registry, "list_models", None)
            health["model_registry"] = bool(lm and callable(lm) and lm())
        except Exception:
            health["model_registry"] = False

        try:
            ls = getattr(self.strategy_registry, "list_strategies", None)
            health["strategy_registry"] = bool(ls and callable(ls) and ls())
        except Exception:
            health["strategy_registry"] = False

        # Check partitions
        try:
            stats = self.partition_manager.get_partition_stats()
            health["partitions"] = len(stats) > 0
        except Exception:
            health["partitions"] = False

        return health

    def create_integrated_actor(self, actor_class: type[Any], config: Any) -> Any:
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

        logger.info("ML integration manager shutdown complete")


class AutoIntegratedActor:
    """
    DEPRECATED: Use BaseMLInferenceActor from ml.actors.base instead.

    This class is kept for backward compatibility only.
    All new actors should inherit from BaseMLInferenceActor which has
    automatic store integration built-in.
    """

    def __init__(self, config: Any, integration: MLIntegrationManager | None = None):
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
        self.integration = integration or MLIntegrationManager(config)

        # Wire all stores
        self.feature_store = self.integration.feature_store
        self.model_store = self.integration.model_store
        self.strategy_store = self.integration.strategy_store

        # Wire all registries
        self.feature_registry = self.integration.feature_registry
        self.model_registry = self.integration.model_registry
        self.strategy_registry = self.integration.strategy_registry

    def write_features(self, features: dict[str, float], **kwargs: Any) -> None:
        """
        Automatically write features to store.
        """
        self.feature_store.write_features(
            feature_set_id=getattr(self, "feature_set_id", "default"),
            instrument_id=getattr(self, "instrument_id", "unknown"),
            features=features,
            **kwargs,
        )

    def write_prediction(
        self,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        **kwargs: Any,
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
            **kwargs,
        )

    def write_signal(
        self,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        **kwargs: Any,
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
            **kwargs,
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
