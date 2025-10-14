"""
Initialization manager for DataScheduler.

This module handles initialization of external services including FeatureStore and
Prometheus metrics server.

"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from ml.features.engineering import FeatureConfig


class InitializationManagerProtocol(Protocol):
    """
    Protocol for initialization operations.
    """

    def initialize_feature_store(
        self,
        connection_string: str,
        feature_config: FeatureConfig,
    ) -> Any | None:
        """
        Initialize the FeatureStore.

        Parameters
        ----------
        connection_string : str
            Database connection string
        feature_config : FeatureConfig
            Feature configuration

        Returns
        -------
        Any | None
            FeatureStore instance or None if initialization failed

        """
        ...

    def start_metrics_server(
        self,
        port: int,
    ) -> Any | None:
        """
        Start the Prometheus metrics HTTP server.

        Parameters
        ----------
        port : int
            Port number for the metrics server

        Returns
        -------
        Any | None
            MetricsServer instance or None if start failed

        """
        ...


class InitializationManager:
    """
    Handle initialization of external services.

    Implements Pattern 2: Protocol-First Interface Design
    Implements Pattern 4: Progressive Fallback Chains

    This component is responsible ONLY for service initialization.

    """

    def __init__(
        self,
        feature_engineer: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize InitializationManager.

        Parameters
        ----------
        feature_engineer : Any | None
            Feature engineer instance (optional)
        logger : logging.Logger | None
            Logger for operations (default: creates module logger)

        """
        self._feature_engineer = feature_engineer
        self._logger = logger or logging.getLogger(__name__)

    def initialize_feature_store(
        self,
        connection_string: str,
        feature_config: FeatureConfig,
    ) -> Any | None:
        """
        Initialize the FeatureStore with proper configuration.

        This method sets up the FeatureStore for batch feature computation and storage,
        ensuring training/inference parity.

        Parameters
        ----------
        connection_string : str
            Database connection string for feature store
        feature_config : FeatureConfig
            Feature configuration from the feature engineer

        Returns
        -------
        Any | None
            FeatureStore instance or None if initialization failed

        """
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        try:
            # Get connection string (use parameter, fallback to env, fallback to default)
            db_connection = (
                connection_string
                or os.getenv("NAUTILUS_DB_CONNECTION")
                or "postgresql://postgres:postgres@localhost:5432/nautilus"
            )

            # Instantiate via module to allow tests to patch ml.stores.feature_store.FeatureStore
            from ml.stores import feature_store as _fs

            feature_store = _fs.FeatureStore(
                connection_string=db_connection,
                feature_config=feature_config,
            )

            # Log connection info (hide password for security)
            safe_connection = db_connection.split("@")[1] if "@" in db_connection else db_connection
            self._logger.info(f"Initialized FeatureStore with connection to: {safe_connection}")

            return feature_store

        except Exception:
            self._logger.error(
                "Failed to initialize FeatureStore",
                exc_info=True,
            )
            # Don't raise - allow scheduler to work without feature store
            return None

    def start_metrics_server(
        self,
        port: int,
    ) -> Any | None:
        """
        Start the HTTP server for Prometheus metrics.

        Parameters
        ----------
        port : int
            Port number for the metrics server

        Returns
        -------
        Any | None
            MetricsServer instance or None if start failed

        """
        try:
            from ml.monitoring._config import MonitoringConfig
            from ml.monitoring.server import MetricsServer

            # Create monitoring config with specified port
            monitoring_config = MonitoringConfig(
                enabled=True,
                metrics_port=port,
            )

            metrics_server = MetricsServer(config=monitoring_config)
            metrics_server.start()
            self._logger.info(f"Started metrics server on port {port}")

            return metrics_server

        except Exception:
            self._logger.warning(
                "Failed to start metrics server",
                exc_info=True,
            )
            return None
