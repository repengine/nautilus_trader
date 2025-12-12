#!/usr/bin/env python3

"""
Feature health component for FeatureStore.

Extracted from FeatureStore (Phase 3.7.6). Provides health check, feature clearing,
flush operations, and connection management utilities.

All methods are COLD path (infrastructure operations, no hot path constraints).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml._imports import HAS_PROMETHEUS
from ml.common.metrics_bootstrap import get_counter


if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
health_check_counter = get_counter(
    "ml_feature_health_checks_total",
    "Total number of feature store health checks",
    labelnames=["status"],
)
clear_features_counter = get_counter(
    "ml_feature_clear_operations_total",
    "Total number of feature clear operations",
    labelnames=["scope"],
)


# =========================================================================
# Protocols
# =========================================================================


@runtime_checkable
class FeatureHealthProtocol(Protocol):
    """
    Protocol for feature health operations.

    Defines the interface for health checking, feature clearing, flush operations,
    and database connection management.

    COLD PATH: All operations are infrastructure-level, no hot path constraints.

    """

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        COLD PATH: Health monitoring is infrastructure operation

        Returns
        -------
        bool
            True if store is healthy and database is accessible, False otherwise

        """
        ...

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features with optional filtering.

        COLD PATH: Feature clearing is administrative operation

        Args:
            instrument_id: Clear only for specific instrument (optional)
            feature_version: Clear only specific version (optional)

        """
        ...

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        COLD PATH: Flush is infrastructure operation

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.

        """
        ...


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class FeatureHealthConfig:
    """
    Configuration for FeatureHealthComponent.

    Attributes
    ----------
    health_check_timeout_seconds : float
        Timeout for health check query (default: 5.0 seconds)
    emit_metrics : bool
        Whether to emit Prometheus metrics (default: True)

    """

    health_check_timeout_seconds: float = 5.0
    emit_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.health_check_timeout_seconds <= 0:
            msg = "health_check_timeout_seconds must be positive"
            raise ValueError(msg)


# =========================================================================
# FeatureHealthComponent
# =========================================================================


class FeatureHealthComponent:
    """
    Health check, feature clearing, and flush operations for FeatureStore.

    Extracted from FeatureStore (Phase 3.7.6).
    All methods are COLD path (infrastructure operations, no hot path constraints).

    Provides:
    - Database health checking via simple SELECT query
    - Feature clearing with optional instrument/version filters
    - Flush operation (no-op for synchronous writes)
    - Connection management utilities

    Example
    -------
    >>> from ml.stores.common.feature_health import FeatureHealthComponent
    >>> health = FeatureHealthComponent(
    ...     engine=engine,
    ...     table=feature_values_table,
    ... )
    >>> if health.is_healthy():
    ...     print("Feature store is accessible")
    >>> health.clear_features(instrument_id="SPY.DATABENTO")
    >>> health.flush()

    """

    def __init__(
        self,
        engine: Engine,
        table: Table,
        *,
        config: FeatureHealthConfig | None = None,
    ) -> None:
        """
        Initialize feature health component.

        Args:
            engine: SQLAlchemy engine for database operations
            table: SQLAlchemy Table for feature values (ml_feature_values)
            config: Optional configuration (uses defaults if not provided)

        """
        self._engine = engine
        self._table = table
        self._config = config or FeatureHealthConfig()

    # =========================================================================
    # Public API - All COLD PATH
    # =========================================================================

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        COLD PATH: Health monitoring is infrastructure operation

        Performs a simple SELECT 1 query to verify database connectivity.
        Returns False on any connection or query errors.

        Returns
        -------
        bool
            True if store is healthy and database is accessible, False otherwise

        Example
        -------
        >>> if health.is_healthy():
        ...     print("Database connection is healthy")
        ... else:
        ...     print("Database connection failed")

        """
        try:
            # Try a simple query to verify connection
            with self._engine.connect() as conn:
                from sqlalchemy import text

                result = conn.execute(text("SELECT 1"))
                is_healthy = result is not None

            # Record metric
            if self._config.emit_metrics and HAS_PROMETHEUS:
                status = "healthy" if is_healthy else "unhealthy"
                health_check_counter.labels(status=status).inc()

            return is_healthy

        except Exception as exc:
            logger.warning(
                "Feature store health check failed: %s",
                exc,
                exc_info=True,
            )

            # Record unhealthy metric
            if self._config.emit_metrics and HAS_PROMETHEUS:
                health_check_counter.labels(status="unhealthy").inc()

            return False

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features with optional filtering.

        COLD PATH: Feature clearing is administrative operation

        Deletes feature rows from the database. When no filters are provided,
        all features are deleted. Filters can be combined for precise deletion.

        Args:
            instrument_id: Clear only for specific instrument (optional)
            feature_version: Clear only specific version (optional)

        Example
        -------
        >>> # Clear all features for SPY
        >>> health.clear_features(instrument_id="SPY.DATABENTO")
        >>>
        >>> # Clear all features for a specific version
        >>> health.clear_features(feature_version="v1.2.0")
        >>>
        >>> # Clear features matching both criteria
        >>> health.clear_features(
        ...     instrument_id="SPY.DATABENTO",
        ...     feature_version="v1.2.0",
        ... )
        >>>
        >>> # Clear all features (be careful!)
        >>> health.clear_features()

        """
        with self._engine.begin() as conn:
            delete_stmt = self._table.delete()

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    self._table.c.instrument_id == instrument_id,
                )

            if feature_version:
                delete_stmt = delete_stmt.where(
                    self._table.c.feature_version == feature_version,
                )

            conn.execute(delete_stmt)

        # Determine scope for metrics
        if instrument_id and feature_version:
            scope = "instrument_and_version"
        elif instrument_id:
            scope = "instrument"
        elif feature_version:
            scope = "version"
        else:
            scope = "all"

        # Record metric
        if self._config.emit_metrics and HAS_PROMETHEUS:
            clear_features_counter.labels(scope=scope).inc()

        logger.debug(
            "Cleared features: instrument_id=%s, feature_version=%s",
            instrument_id,
            feature_version,
        )

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        COLD PATH: Flush is infrastructure operation

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.

        Example
        -------
        >>> # Ensure all pending writes are committed
        >>> health.flush()

        """
        # Currently a no-op as writes are synchronous
        # Future: implement write buffering similar to ModelStore
        logger.debug("flush() called - no-op for synchronous FeatureStore writes")

    def _get_connection(self) -> Any:  # pragma: no cover (test hook for patching)
        """
        Return a connection context manager (patchable in tests).

        COLD PATH: Connection management is infrastructure operation

        This method exists primarily as a test hook to allow patching
        database connections in unit tests without requiring a real database.

        Returns
        -------
        Any
            Connection context manager from the engine

        """
        return self._engine.connect()

    @property
    def engine(self) -> Engine:
        """
        Return the SQLAlchemy engine.

        Returns
        -------
        Engine
            The SQLAlchemy engine used for database operations

        """
        return self._engine

    @property
    def table(self) -> Table:
        """
        Return the feature values table.

        Returns
        -------
        Table
            The SQLAlchemy Table for feature values

        """
        return self._table
