"""
Scheduler initialization component extracted from DataScheduler.

This component handles initialization logic for DataScheduler including:
- Connection string resolution from config or parameter
- DataRegistry initialization with Postgres/JSON fallback
- FeatureStore initialization with connection resolution

Extracted from legacy DataScheduler (lines 246-470):
- __init__() parameter handling (lines 277-287)
- _init_data_registry() (lines 375-420)
- _initialize_feature_store() (lines 422-470)

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig
    from ml.features.engineering import FeatureEngineer
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


class SchedulerInitProtocol(Protocol):
    """
    Protocol for scheduler initialization operations.

    This protocol defines the contract for scheduler initialization components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    resolve_connection
        Resolve feature store connection from config or parameter.
    init_data_registry
        Initialize DataRegistry with Postgres backend or JSON fallback.
    init_feature_store
        Initialize FeatureStore if enabled and configured.

    """

    def resolve_connection(
        self,
        config: SchedulerConfig,
        connection: str | None,
    ) -> str | None:
        """
        Resolve feature store connection from config or parameter.

        Args:
            config: Scheduler configuration with connection fields.
            connection: Explicit connection string parameter.

        Returns:
            Resolved connection string or None if unavailable.

        """
        ...

    def init_data_registry(
        self,
        connection: str | None,
    ) -> RegistryProtocol | None:
        """
        Initialize DataRegistry with Postgres backend or JSON fallback.

        Args:
            connection: Database connection string for Postgres backend.

        Returns:
            Initialized DataRegistry or None on failure.

        """
        ...

    def init_feature_store(
        self,
        config: SchedulerConfig,
        connection: str | None,
        feature_engineer: FeatureEngineer | None,
    ) -> Any | None:
        """
        Initialize FeatureStore if enabled and configured.

        Args:
            config: Scheduler configuration with feature store settings.
            connection: Database connection string.
            feature_engineer: Feature engineer with config.

        Returns:
            Initialized FeatureStore or None on failure/disabled.

        """
        ...


class SchedulerInitComponent:
    """
    Component for DataScheduler initialization logic.

    This component extracts initialization responsibilities from DataScheduler,
    providing focused methods for:
    - Connection string resolution with config fallback chain
    - DataRegistry initialization with Postgres/JSON backend selection
    - FeatureStore initialization with graceful fallback

    All methods are designed to handle errors gracefully and log appropriate
    warnings/errors without raising exceptions that would prevent scheduler
    initialization.

    Example:
        >>> from ml.config.scheduler_config import SchedulerConfig
        >>> component = SchedulerInitComponent()
        >>> config = SchedulerConfig()
        >>> connection = component.resolve_connection(config, None)
        >>> registry = component.init_data_registry(connection)

    """

    def resolve_connection(
        self,
        config: SchedulerConfig,
        connection: str | None,
    ) -> str | None:
        """
        Resolve feature store connection from config or parameter.

        Implements the frozen-config-friendly connection resolution logic
        extracted from DataScheduler.__init__. The resolution order is:

        1. Explicit `connection` parameter (highest priority)
        2. `config.feature_store_connection` attribute
        3. `config.connection_string` attribute (backward compatibility)
        4. None if all sources are unavailable

        Args:
            config: Scheduler configuration with connection fields.
            connection: Explicit connection string parameter.

        Returns:
            Resolved connection string or None if unavailable.

        Example:
            >>> from ml.config.scheduler_config import SchedulerConfig
            >>> component = SchedulerInitComponent()
            >>> config = SchedulerConfig(feature_store_connection="postgresql://...")
            >>> conn = component.resolve_connection(config, None)
            >>> assert conn == "postgresql://..."

        """
        # Priority 1: Explicit parameter
        conn_candidate = connection

        # Priority 2: feature_store_connection from config
        if conn_candidate is None:
            conn_candidate = getattr(config, "feature_store_connection", None)

        # Priority 3: connection_string from config (backward compat)
        if conn_candidate is None:
            conn_candidate = getattr(config, "connection_string", None)

        # Validate and return
        if isinstance(conn_candidate, str) and conn_candidate:
            return conn_candidate

        return None

    def init_data_registry(
        self,
        connection: str | None,
    ) -> RegistryProtocol | None:
        """
        Initialize DataRegistry with Postgres backend or JSON fallback.

        Implements the DataRegistry initialization logic extracted from
        DataScheduler._init_data_registry. Selects the appropriate backend:

        - Postgres backend when connection string is provided
        - JSON backend for development when no connection available

        This method never raises exceptions - it logs warnings and returns
        None on failure to allow the scheduler to continue without registry.

        Args:
            connection: Database connection string for Postgres backend.
                If None, uses JSON backend for development.

        Returns:
            Initialized DataRegistry or None on failure.

        Example:
            >>> component = SchedulerInitComponent()
            >>> registry = component.init_data_registry("postgresql://...")
            >>> if registry is not None:
            ...     print("Registry initialized with Postgres backend")

        """
        # Lazy imports to avoid circular dependencies
        from ml.registry.data_registry import DataRegistry
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        try:
            if connection:
                # Use PostgreSQL backend in production
                persistence_config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=connection,
                )
                registry_path = Path("/tmp/ml_registry")  # Path for JSON fallback
            else:
                # Use JSON backend for development (standardized location)
                registry_path = Path.home() / ".nautilus" / "ml" / "registry"
                try:
                    registry_path.mkdir(parents=True, exist_ok=True)
                except Exception:
                    logger.debug(
                        "Failed to create registry path %s, will continue with default location",
                        registry_path,
                        exc_info=True,
                    )
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=registry_path,
                )

            data_registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=persistence_config,
            )

            logger.info(
                "Initialized DataRegistry with backend=%s",
                persistence_config.backend.value,
            )

            return data_registry

        except Exception:
            logger.warning(
                "Failed to initialize DataRegistry. Events will not be tracked.",
                exc_info=True,
            )
            return None

    def init_feature_store(
        self,
        config: SchedulerConfig,
        connection: str | None,
        feature_engineer: FeatureEngineer | None,
    ) -> Any | None:
        """
        Initialize FeatureStore if enabled and configured.

        Implements the FeatureStore initialization logic extracted from
        DataScheduler._initialize_feature_store. Conditions for initialization:

        1. `config.feature_store_enabled` must be True
        2. `feature_engineer` must be provided (not None)

        Connection resolution order:
        1. Explicit `connection` parameter
        2. NAUTILUS_DB_CONNECTION environment variable
        3. Default local PostgreSQL connection

        This method never raises exceptions - it logs errors and returns
        None on failure to allow the scheduler to continue without feature store.

        Args:
            config: Scheduler configuration with feature store settings.
            connection: Database connection string.
            feature_engineer: Feature engineer with config.

        Returns:
            Initialized FeatureStore or None on failure/disabled.

        Example:
            >>> from ml.config.scheduler_config import SchedulerConfig
            >>> from ml.features.engineering import FeatureEngineer
            >>> component = SchedulerInitComponent()
            >>> config = SchedulerConfig(feature_store_enabled=True)
            >>> engineer = FeatureEngineer()
            >>> store = component.init_feature_store(config, None, engineer)

        """
        # Check if feature store should be initialized
        if not config.feature_store_enabled:
            logger.debug("Feature store disabled in config")
            return None

        if feature_engineer is None:
            logger.debug("Feature store not initialized: no feature engineer provided")
            return None

        # Lazy imports to check dependencies
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml.features.engineering import FeatureConfig
        from ml.features.engineering import FeatureConfigLike

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        try:
            # Get connection string from config, environment, or use default
            db_connection = (
                connection
                or os.getenv("NAUTILUS_DB_CONNECTION")
                or "postgresql://postgres:postgres@localhost:5432/nautilus"
            )

            # Get feature config from the feature engineer
            feature_config: FeatureConfigLike
            if hasattr(feature_engineer, "config"):
                feature_config = feature_engineer.config
            else:
                feature_config = FeatureConfig()

            # Instantiate via module to allow tests to patch ml.stores.feature_store.FeatureStore
            from ml.stores import feature_store as _fs

            feature_store = _fs.FeatureStore(
                connection_string=db_connection,
                feature_config=feature_config,
            )

            # Log connection info (hide password for security)
            safe_connection = (
                db_connection.split("@")[1]
                if "@" in db_connection
                else db_connection
            )
            logger.info(f"Initialized FeatureStore with connection to: {safe_connection}")

            return feature_store

        except Exception:
            logger.error(
                "Failed to initialize FeatureStore",
                exc_info=True,
            )
            return None


__all__ = [
    "SchedulerInitComponent",
    "SchedulerInitProtocol",
]
