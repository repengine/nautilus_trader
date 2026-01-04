#!/usr/bin/env python3

"""
Store operations component for DataStore.

Extracted from DataStore (Phase 2.4.6). Provides store lifecycle management,
health monitoring, metrics collection, progressive fallback chains, and circuit
breaker logic for unstable dependencies.

ALL methods are COLD path (infrastructure operations, no hot path constraints).

"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.protocols import EarningsStoreProtocol
    from ml.stores.protocols import FeatureStoreStrictProtocol
    from ml.stores.protocols import ModelStoreStrictProtocol
    from ml.stores.protocols import StrategyStoreStrictProtocol

logger = logging.getLogger(__name__)


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
fallback_activation_counter = get_counter(
    "ml_fallback_activations_total",
    "Total number of fallback activations for store operations",
    labelnames=["component", "level"],
)
health_check_counter = get_counter(
    "ml_health_checks_total",
    "Total number of health checks performed",
    labelnames=["component", "status"],
)
operation_latency_histogram = get_histogram(
    "ml_store_operation_latency_seconds",
    "Latency of store operations in seconds",
    labelnames=["operation"],
)


# =========================================================================
# StoreOperationsComponent
# =========================================================================


class StoreOperationsComponent:
    """
    Store lifecycle, health monitoring, and metrics collection for DataStore.

    Extracted from DataStore (Phase 2.4.6).
    All methods are COLD path (infrastructure operations, no hot path constraints).

    Provides:
    - Health monitoring for all 4 stores + 4 registries
    - Performance metrics aggregation across components
    - Graceful shutdown and resource cleanup
    - Store initialization with progressive fallback chains
    - Fallback chain management (PostgreSQL → DummyStore)
    - Circuit breaker for unstable dependencies
    - Metrics emission for observability

    Example
    -------
    >>> from ml.stores.common.store_operations import StoreOperationsComponent
    >>> operations = StoreOperationsComponent(
    ...     connection_string="postgresql://...",
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     data_registry=registry,
    ... )
    >>> health = operations.health_check()
    >>> metrics = operations.get_metrics()
    >>> operations.close()

    """

    def __init__(
        self,
        connection_string: str,
        *,
        feature_store: FeatureStoreStrictProtocol | None = None,
        model_store: ModelStoreStrictProtocol | None = None,
        strategy_store: StrategyStoreStrictProtocol | None = None,
        earnings_store: EarningsStoreProtocol | None = None,
        data_registry: RegistryProtocol | None = None,
        feature_registry: Any | None = None,
        model_registry: Any | None = None,
        strategy_registry: Any | None = None,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        """
        Initialize store operations component with stores and configuration.

        Args:
            connection_string: Database connection string
            feature_store: Initialized feature store
            model_store: Initialized model store
            strategy_store: Initialized strategy store
            earnings_store: Initialized earnings store
            data_registry: Data registry for manifest/contract retrieval
            feature_registry: Feature registry for feature management
            model_registry: Model registry for model deployment
            strategy_registry: Strategy registry for strategy management
            enable_circuit_breaker: If True, enable circuit breaker for unstable stores
            circuit_breaker_threshold: Number of failures before opening circuit

        """
        self._connection_string = connection_string
        self._feature_store = feature_store
        self._model_store = model_store
        self._strategy_store = strategy_store
        self._earnings_store = earnings_store
        self._data_registry = data_registry
        self._feature_registry = feature_registry
        self._model_registry = model_registry
        self._strategy_registry = strategy_registry
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breaker_threshold = circuit_breaker_threshold

        # Fallback tracking
        self._fallback_active = False
        self._fallback_reason: str | None = None
        self._primary_connection_lost_at: int | None = None

        # Circuit breaker state
        self._circuit_breaker_failures: dict[str, int] = {}
        self._circuit_breaker_open: dict[str, bool] = {}

        # Performance tracking
        self._operation_counts: dict[str, int] = {}
        self._operation_latencies: dict[str, list[float]] = {}

    # =========================================================================
    # Public API - All COLD PATH
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check across all 4 stores + 4 registries.

        COLD PATH: Health monitoring is infrastructure operation

        Checks health status of all components and returns aggregated status.
        Individual component failures are logged but don't fail entire check.

        Returns
        -------
        dict[str, Any]
            Health status with component-level details
            - healthy: bool (True if all components healthy)
            - components: dict (status per component)
            - fallback_active: bool (True if fallback chain activated)
            - circuit_breakers_open: list (components with open circuit breakers)

        Examples
        --------
        >>> health = operations.health_check()
        >>> assert health["healthy"] is True
        >>> assert "feature_store" in health["components"]
        >>> assert health["components"]["feature_store"]["status"] == "healthy"

        """
        health_status: dict[str, Any] = {
            "healthy": True,
            "components": {},
            "fallback_active": self._fallback_active,
            "fallback_reason": self._fallback_reason,
            "circuit_breakers_open": [],
            "checked_at": time.time_ns(),
        }

        # Check all 4 stores
        store_components = [
            ("feature_store", self._feature_store),
            ("model_store", self._model_store),
            ("strategy_store", self._strategy_store),
            ("earnings_store", self._earnings_store),
        ]

        for component_name, store in store_components:
            if store is None:
                health_status["components"][component_name] = {
                    "status": "unavailable",
                    "reason": "not_initialized",
                }
                health_status["healthy"] = False
                continue

            try:
                # Check if store has health_check method
                if hasattr(store, "get_health_status"):
                    component_health_raw: object = store.get_health_status()

                    component_health: dict[str, Any]
                    if isinstance(component_health_raw, dict):
                        component_health = component_health_raw
                    else:
                        component_health = {"status": "unknown", "raw": component_health_raw}

                    healthy_flag: bool
                    if "healthy" in component_health:
                        healthy_flag = bool(component_health.get("healthy"))
                    else:
                        status_val = str(component_health.get("status", "")).lower()
                        healthy_flag = status_val in {"healthy", "ok"}

                    health_status["components"][component_name] = {
                        "status": "healthy" if healthy_flag else "degraded",
                        "details": component_health,
                    }
                    if not healthy_flag:
                        health_status["healthy"] = False
                else:
                    # Basic availability check
                    health_status["components"][component_name] = {
                        "status": "healthy",
                        "reason": "no_health_check_method",
                    }

            except Exception as exc:
                logger.warning(
                    "Health check failed for %s: %s",
                    component_name,
                    exc,
                    exc_info=True,
                )
                health_status["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(exc),
                }
                health_status["healthy"] = False

        # Check all 4 registries
        registry_components = [
            ("data_registry", self._data_registry),
            ("feature_registry", self._feature_registry),
            ("model_registry", self._model_registry),
            ("strategy_registry", self._strategy_registry),
        ]

        for component_name, registry in registry_components:
            if registry is None:
                health_status["components"][component_name] = {
                    "status": "unavailable",
                    "reason": "not_initialized",
                }
                # Registries are optional, don't fail health check
                continue

            try:
                # Check if registry has health_check method
                if hasattr(registry, "get_health_status"):
                    registry_health_raw: object = registry.get_health_status()
                    registry_health: dict[str, Any]
                    if isinstance(registry_health_raw, dict):
                        registry_health = registry_health_raw
                    else:
                        registry_health = {"status": "unknown", "raw": registry_health_raw}

                    registry_healthy_flag: bool
                    if "healthy" in registry_health:
                        registry_healthy_flag = bool(registry_health.get("healthy"))
                    else:
                        status_val = str(registry_health.get("status", "")).lower()
                        registry_healthy_flag = status_val in {"healthy", "ok"}

                    health_status["components"][component_name] = {
                        "status": "healthy" if registry_healthy_flag else "degraded",
                        "details": registry_health,
                    }
                    if not registry_healthy_flag:
                        # Registry failures are non-fatal
                        logger.warning("Registry %s is unhealthy", component_name)
                else:
                    health_status["components"][component_name] = {
                        "status": "healthy",
                        "reason": "no_health_check_method",
                    }

            except Exception as exc:
                logger.warning(
                    "Health check failed for %s: %s",
                    component_name,
                    exc,
                    exc_info=True,
                )
                health_status["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(exc),
                }
                # Registry failures are non-fatal

        # Add circuit breaker status
        for component, is_open in self._circuit_breaker_open.items():
            if is_open:
                health_status["circuit_breakers_open"].append(component)
                health_status["healthy"] = False

        # Record health check metric
        if HAS_PROMETHEUS:
            status = "healthy" if health_status["healthy"] else "unhealthy"
            health_check_counter.labels(
                component="store_operations",
                status=status,
            ).inc()

        return health_status

    def get_metrics(self) -> dict[str, float]:
        """
        Aggregate performance metrics from all components.

        COLD PATH: Metrics collection is infrastructure operation

        Collects and aggregates metrics from all stores and registries.
        Returns aggregated statistics for observability.

        Returns
        -------
        dict[str, float]
            Aggregated metrics
            - operation_count_total: Total operations across all stores
            - avg_operation_latency_ms: Average operation latency
            - p95_operation_latency_ms: 95th percentile latency
            - fallback_activations: Number of fallback activations
            - circuit_breakers_open: Number of open circuit breakers

        Examples
        --------
        >>> metrics = operations.get_metrics()
        >>> assert metrics["operation_count_total"] >= 0
        >>> assert metrics["avg_operation_latency_ms"] >= 0.0

        """
        metrics: dict[str, float] = {}

        # Aggregate operation counts
        total_operations = sum(self._operation_counts.values())
        metrics["operation_count_total"] = float(total_operations)

        # Calculate latency statistics
        all_latencies: list[float] = []
        for latencies in self._operation_latencies.values():
            all_latencies.extend(latencies)

        if all_latencies:
            metrics["avg_operation_latency_ms"] = sum(all_latencies) / len(all_latencies)
            sorted_latencies = sorted(all_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            metrics["p95_operation_latency_ms"] = sorted_latencies[p95_index] if sorted_latencies else 0.0
        else:
            metrics["avg_operation_latency_ms"] = 0.0
            metrics["p95_operation_latency_ms"] = 0.0

        # Fallback metrics
        metrics["fallback_active"] = 1.0 if self._fallback_active else 0.0
        metrics["circuit_breakers_open"] = float(sum(1 for is_open in self._circuit_breaker_open.values() if is_open))

        # Collect metrics from components (if available)
        store_components = [
            ("feature_store", self._feature_store),
            ("model_store", self._model_store),
            ("strategy_store", self._strategy_store),
            ("earnings_store", self._earnings_store),
        ]

        for component_name, store in store_components:
            if store is None:
                continue

            try:
                if hasattr(store, "get_performance_metrics"):
                    component_metrics = store.get_performance_metrics()
                    for metric_name, metric_value in component_metrics.items():
                        metrics[f"{component_name}_{metric_name}"] = float(metric_value)

            except Exception as exc:
                logger.debug(
                    "Failed to collect metrics from %s: %s",
                    component_name,
                    exc,
                    exc_info=True,
                )

        return metrics

    def close(self) -> None:
        """
        Gracefully shutdown all stores and clean up resources.

        COLD PATH: Shutdown is infrastructure operation

        Closes all database connections, flushes pending operations, and
        releases resources. Should be called before process termination.

        Examples
        --------
        >>> operations.close()
        >>> # All stores and connections closed

        """
        logger.info("Shutting down StoreOperationsComponent")

        # Close all stores
        store_components = [
            ("feature_store", self._feature_store),
            ("model_store", self._model_store),
            ("strategy_store", self._strategy_store),
            ("earnings_store", self._earnings_store),
        ]

        for component_name, store in store_components:
            if store is None:
                continue

            try:
                if hasattr(store, "close"):
                    logger.debug("Closing %s", component_name)
                    store.close()
                elif hasattr(store, "shutdown"):
                    logger.debug("Shutting down %s", component_name)
                    store.shutdown()

            except Exception as exc:
                logger.warning(
                    "Failed to close %s: %s",
                    component_name,
                    exc,
                    exc_info=True,
                )

        # Reset state
        self._operation_counts.clear()
        self._operation_latencies.clear()
        self._circuit_breaker_failures.clear()
        self._circuit_breaker_open.clear()

        logger.info("StoreOperationsComponent shutdown complete")

    # =========================================================================
    # Store Initialization & Fallback Management
    # =========================================================================

    def _initialize_stores(self) -> None:
        """
        Initialize all stores with progressive fallback chains.

        COLD PATH: Store initialization happens at startup

        Initializes stores with PRIMARY → CACHED → FILE → DUMMY fallback.
        Failures activate fallback chains automatically with metrics emission.

        Examples
        --------
        >>> operations._initialize_stores()
        >>> # All stores initialized with fallback chains

        """
        logger.info("Initializing stores with progressive fallback chains")

        # Initialize earnings store with fallback
        if self._earnings_store is None:
            try:
                from ml.stores.earnings_store import EarningsStore

                self._earnings_store = EarningsStore(self._connection_string)
                logger.info("Initialized primary EarningsStore (PostgreSQL)")

            except Exception as exc:
                logger.warning(
                    "Primary EarningsStore initialization failed: %s",
                    exc,
                    exc_info=True,
                )
                self._activate_fallback("earnings_store_init_failed")

                # Try file-backed fallback
                file_store = self._try_file_earnings_store()
                if file_store is not None:
                    self._earnings_store = file_store
                    logger.info("Activated file-backed EarningsStore fallback")
                else:
                    # Use dummy store as last resort
                    from ml.stores.earnings_store import DummyEarningsStore

                    self._earnings_store = DummyEarningsStore()
                    logger.warning("Activated DummyEarningsStore (no persistence)")

    def _initialize_fallback_chain(self) -> None:
        """
        Initialize progressive fallback chain: PRIMARY → CACHED → FILE → DUMMY.

        COLD PATH: Fallback chain initialization happens at startup

        Sets up fallback chain for store failures. Monitors primary connection
        and automatically activates fallbacks when failures are detected.

        Examples
        --------
        >>> operations._initialize_fallback_chain()
        >>> # Fallback chain configured and monitoring active

        """
        logger.info("Initializing progressive fallback chain")

        # Configure fallback chain for each store type
        fallback_configs = {
            "feature_store": ["postgresql", "cached", "dummy"],
            "model_store": ["postgresql", "cached", "dummy"],
            "strategy_store": ["postgresql", "cached", "dummy"],
            "earnings_store": ["postgresql", "file", "dummy"],
        }

        for store_name, chain in fallback_configs.items():
            logger.debug("Fallback chain for %s: %s", store_name, " → ".join(chain))

        # Initialize circuit breakers for each store
        for store_name in fallback_configs:
            self._circuit_breaker_failures[store_name] = 0
            self._circuit_breaker_open[store_name] = False

        logger.info("Fallback chain initialized successfully")

    def _activate_fallback(self, reason: str) -> None:
        """
        Activate fallback chain and emit metrics.

        COLD PATH: Fallback activation is rare (only on failures)

        Activates fallback chain when primary store fails. Records timestamp,
        reason, and emits metrics for observability.

        Args:
            reason: Reason for fallback activation (e.g., "connection_lost", "timeout")

        Examples
        --------
        >>> operations._activate_fallback("connection_lost")
        >>> # Fallback chain activated, metrics emitted

        """
        if not self._fallback_active:
            self._fallback_active = True
            self._fallback_reason = reason
            self._primary_connection_lost_at = time.time_ns()

            logger.warning(
                "Fallback chain activated: %s (at %d)",
                reason,
                self._primary_connection_lost_at,
            )

            # Emit fallback activation metric
            if HAS_PROMETHEUS:
                fallback_activation_counter.labels(
                    component="store_operations",
                    level="activated",
                ).inc()

    def _restore_primary(self) -> bool:
        """
        Attempt to restore primary store connection.

        COLD PATH: Primary restoration is rare (only after failures)

        Attempts to restore primary PostgreSQL connection after fallback
        activation. Returns True if restoration successful.

        Returns
        -------
        bool
            True if primary restored successfully, False otherwise

        Examples
        --------
        >>> if operations._restore_primary():
        ...     print("Primary connection restored")
        ... else:
        ...     print("Still using fallback")

        """
        if not self._fallback_active:
            logger.debug("Primary is already active, no restoration needed")
            return True

        logger.info("Attempting to restore primary store connection")

        try:
            # Test primary connection
            from sqlalchemy import text

            from ml.core.db_engine import EngineManager

            engine = EngineManager.get_engine(self._connection_string)
            with engine.connect() as conn:
                # Simple connectivity test
                conn.execute(text("SELECT 1"))

            # Primary connection restored
            self._fallback_active = False
            self._fallback_reason = None
            downtime_ns = time.time_ns() - (self._primary_connection_lost_at or 0)
            downtime_seconds = downtime_ns / 1e9

            logger.info(
                "Primary store connection restored after %.2f seconds",
                downtime_seconds,
            )

            # Emit restoration metric
            if HAS_PROMETHEUS:
                fallback_activation_counter.labels(
                    component="store_operations",
                    level="restored",
                ).inc()

            return True

        except Exception as exc:
            logger.debug(
                "Primary restoration failed: %s",
                exc,
                exc_info=True,
            )
            return False

    def _try_file_earnings_store(self) -> EarningsStoreProtocol | None:
        """
        Attempt to initialize file-backed earnings store fallback.

        COLD PATH: Fallback initialization is rare (only on failures)

        Tries to initialize FileEarningsStore as fallback when PostgreSQL
        is unavailable. Returns None if file store initialization fails.

        Returns
        -------
        EarningsStoreProtocol | None
            File-backed earnings store or None if initialization fails

        Examples
        --------
        >>> file_store = operations._try_file_earnings_store()
        >>> if file_store is not None:
        ...     print("File-backed fallback available")

        """
        if not HAS_POLARS:
            logger.debug("File earnings fallback unavailable (polars not installed)")
            return None

        try:
            from ml.stores.file_backed import FileEarningsStore

            file_root_str = os.getenv("ML_FILE_STORE_PATH")
            file_root = Path(file_root_str) if file_root_str else Path.home() / ".nautilus" / "ml" / "file_store"
            earnings_path = file_root / "earnings"

            store = FileEarningsStore(base_path=earnings_path)
            logger.info("Initialized FileEarningsStore fallback at %s", earnings_path)

            # Record fallback metric
            self._record_fallback_metric(level="file")

            return store

        except Exception as exc:
            logger.debug(
                "FileEarningsStore initialization failed: %s",
                exc,
                exc_info=True,
            )
            return None

    def _record_fallback_metric(self, level: str) -> None:
        """
        Record fallback activation metric.

        COLD PATH: Metric recording is infrastructure operation

        Records fallback activation to Prometheus for monitoring and alerting.
        Levels: primary, cached, file, dummy.

        Args:
            level: Fallback level (primary, cached, file, dummy)

        Examples
        --------
        >>> operations._record_fallback_metric(level="file")
        >>> # Metric recorded: ml_fallback_activations_total{level="file"}

        """
        if HAS_PROMETHEUS:
            fallback_activation_counter.labels(
                component="store_operations",
                level=level,
            ).inc()
            logger.debug("Recorded fallback metric: level=%s", level)

    def _emit_health_metric(self, status: str, component: str) -> None:
        """
        Emit health check metric for component.

        COLD PATH: Metric recording is infrastructure operation

        Records component health status to Prometheus for monitoring.
        Statuses: healthy, degraded, unhealthy.

        Args:
            status: Health status (healthy, degraded, unhealthy)
            component: Component name (feature_store, model_store, etc.)

        Examples
        --------
        >>> operations._emit_health_metric("healthy", "feature_store")
        >>> # Metric recorded: ml_health_checks_total{component="feature_store", status="healthy"}

        """
        if HAS_PROMETHEUS:
            health_check_counter.labels(
                component=component,
                status=status,
            ).inc()
            logger.debug("Recorded health metric: component=%s, status=%s", component, status)

    def _record_operation_latency(self, operation: str, duration_ms: float) -> None:
        """
        Record operation latency for performance tracking.

        COLD PATH: Metric recording is infrastructure operation

        Records operation latency to Prometheus histogram and internal tracking.
        Used for P95/P99 latency monitoring and alerting.

        Args:
            operation: Operation name (write_ingestion, read_features, etc.)
            duration_ms: Operation duration in milliseconds

        Examples
        --------
        >>> start = time.time()
        >>> # ... perform operation ...
        >>> duration_ms = (time.time() - start) * 1000
        >>> operations._record_operation_latency("write_ingestion", duration_ms)

        """
        # Record to Prometheus histogram
        if HAS_PROMETHEUS:
            duration_seconds = duration_ms / 1000.0
            operation_latency_histogram.labels(operation=operation).observe(duration_seconds)

        # Record to internal tracking
        if operation not in self._operation_latencies:
            self._operation_latencies[operation] = []

        self._operation_latencies[operation].append(duration_ms)

        # Keep only last 1000 latencies per operation (rolling window)
        if len(self._operation_latencies[operation]) > 1000:
            self._operation_latencies[operation] = self._operation_latencies[operation][-1000:]

        # Increment operation count
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1

        logger.debug(
            "Recorded operation latency: operation=%s, duration_ms=%.2f",
            operation,
            duration_ms,
        )

    # =========================================================================
    # Circuit Breaker Logic
    # =========================================================================

    def _check_circuit_breaker(self, component: str) -> bool:
        """
        Check if circuit breaker is open for component.

        COLD PATH: Circuit breaker checks are infrastructure operations

        Returns True if circuit breaker is open (component unavailable).
        Circuit breakers open after threshold consecutive failures.

        Args:
            component: Component name (feature_store, model_store, etc.)

        Returns
        -------
        bool
            True if circuit breaker is open (unavailable), False if closed

        """
        if not self._enable_circuit_breaker:
            return False

        return self._circuit_breaker_open.get(component, False)

    def _record_circuit_breaker_failure(self, component: str) -> None:
        """
        Record circuit breaker failure for component.

        COLD PATH: Circuit breaker management is infrastructure operation

        Increments failure count and opens circuit breaker if threshold exceeded.
        Open circuit breakers prevent further operations until manual reset.

        Args:
            component: Component name (feature_store, model_store, etc.)

        """
        if not self._enable_circuit_breaker:
            return

        failures = self._circuit_breaker_failures.get(component, 0) + 1
        self._circuit_breaker_failures[component] = failures

        if failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open[component] = True
            logger.error(
                "Circuit breaker OPENED for %s after %d failures",
                component,
                failures,
            )

    def _reset_circuit_breaker(self, component: str) -> None:
        """
        Reset circuit breaker for component.

        COLD PATH: Circuit breaker management is infrastructure operation

        Resets failure count and closes circuit breaker. Used when component
        recovers or after manual intervention.

        Args:
            component: Component name (feature_store, model_store, etc.)

        """
        self._circuit_breaker_failures[component] = 0
        self._circuit_breaker_open[component] = False
        logger.info("Circuit breaker RESET for %s", component)
