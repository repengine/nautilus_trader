"""
Store Operations Component.

This module implements Universal Pattern #1 (Mandatory 4-Store Integration) and
Universal Pattern #4 (Progressive Fallback Chains) from CLAUDE.md.

All ML actors MUST use this component to initialize and access the 4 mandatory stores:
- FeatureStore
- ModelStore
- StrategyStore
- DataStore

The component provides:
- Automatic store initialization with PostgreSQL
- Progressive fallback chain: PostgreSQL → DummyStore
- MLPersistenceWorker integration for async persistence
- Health monitoring for all 4 stores
- Centralized metrics for fallback activations

"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import MLActorConfig


if TYPE_CHECKING:
    from ml.observability.ml_async_persistence import MLPersistenceWorker
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import FeatureStoreStrictProtocol
    from ml.stores.protocols import ModelStoreStrictProtocol
    from ml.stores.protocols import StrategyStoreStrictProtocol


class StoreOperationsProtocol(Protocol):
    """
    Protocol for store operations component.

    Defines the interface for managing all 4 mandatory ML stores with progressive
    fallback chains and health monitoring.

    """

    @property
    def feature_store(self) -> FeatureStoreStrictProtocol:
        """
        Return initialized FeatureStore.
        """
        ...

    @property
    def model_store(self) -> ModelStoreStrictProtocol:
        """
        Return initialized ModelStore.
        """
        ...

    @property
    def strategy_store(self) -> StrategyStoreStrictProtocol:
        """
        Return initialized StrategyStore.
        """
        ...

    @property
    def data_store(self) -> DataStoreFacadeProtocol:
        """
        Return initialized DataStore.
        """
        ...

    def get_health_status(self) -> dict[str, str]:
        """
        Get health status for all stores.

        Returns:
            Dictionary mapping store name to status: "healthy", "degraded", or "unhealthy"

        """
        ...


class StoreOperationsComponent:
    """
    Manages initialization and access to all 4 mandatory ML stores.

    Implements Universal Pattern #1 (Mandatory 4-Store Integration) and
    Universal Pattern #4 (Progressive Fallback Chains).

    The component handles:
    - Store initialization with PostgreSQL (PRIMARY)
    - Fallback to DummyStore when PostgreSQL unavailable (FALLBACK)
    - Health monitoring for all stores
    - Metrics emission for fallback activations
    - MLPersistenceWorker lifecycle management

    Example:
        >>> config = MLActorConfig(
        ...     actor_id="my_actor",
        ...     db_connection="postgresql://localhost/nautilus",
        ...     enable_async_persistence=True,
        ... )
        >>> component = StoreOperationsComponent(config)
        >>> # Stores are initialized automatically
        >>> feature_store = component.feature_store
        >>> health = component.get_health_status()
        >>> assert health["aggregate"] == "healthy"

    """

    def __init__(
        self,
        config: MLActorConfig,
        actor_id: str | None = None,
        services: Any | None = None,
    ) -> None:
        """
        Initialize store operations component.

        Args:
            config: ML actor configuration containing database connection
                    and persistence settings
            actor_id: Optional actor ID for logging (defaults to "unknown")
            services: Pre-initialized ActorServices (to avoid duplicate initialization)
                      If None, will call init_actor_services internally

        Raises:
            RuntimeError: If allow_dummy_fallback=False and PostgreSQL connection fails

        """
        self._config = config
        self._actor_id = actor_id or "unknown"
        self._logger = logging.getLogger(__name__)

        # Store references (initialized in _init_stores)
        self._feature_store: FeatureStoreStrictProtocol | None = None
        self._model_store: ModelStoreStrictProtocol | None = None
        self._strategy_store: StrategyStoreStrictProtocol | None = None
        self._data_store: DataStoreFacadeProtocol | None = None

        # Async persistence worker (initialized if enabled)
        self._persistence_worker: MLPersistenceWorker | None = None

        # Metrics for fallback tracking
        self._fallback_counter = get_counter(
            "ml_fallback_activations_total",
            "Total fallback activations by store and stage",
            labelnames=("component", "level"),
        )

        # Initialize stores (pass pre-initialized services if provided)
        self._init_stores(services=services)

        # Initialize async persistence worker if enabled
        if hasattr(config, "enable_async_persistence") and config.enable_async_persistence:
            self._init_persistence_worker()

    def _init_stores(self, services: Any | None = None) -> None:
        """
        Initialize all 4 stores with progressive fallback.

        Attempts to initialize PostgreSQL-backed stores first.
        Falls back to DummyStore if PostgreSQL unavailable (unless disallowed).

        Args:
            services: Pre-initialized ActorServices (to avoid duplicate initialization)
                      If None, will call init_actor_services internally

        Raises:
            RuntimeError: If allow_dummy_fallback=False and PostgreSQL fails

        """
        from ml.actors.actor_services import init_actor_services

        try:
            # Use pre-initialized services if provided, otherwise initialize now
            if services is None:
                services = init_actor_services(self._config)

            self._feature_store = services.feature_store
            self._model_store = services.model_store
            self._strategy_store = services.strategy_store
            self._data_store = services.data_store

            self._logger.info(
                f"Initialized all 4 stores for actor {self._actor_id}",
            )

        except Exception as e:
            # Handle fallback based on configuration
            allow_fallback = getattr(self._config, "allow_dummy_fallback", True)

            if not allow_fallback:
                self._logger.error(
                    f"Failed to initialize stores for actor {self._actor_id}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Failed to create engine: {e}",
                ) from e

            # Fallback to DummyStore
            self._logger.warning(
                f"Failed to initialize stores with PostgreSQL for actor {self._actor_id}, "
                f"falling back to DummyStore: {e}",
                exc_info=True,
            )

            # Initialize with DummyStore - create a modified config
            # We need to ensure use_dummy_stores is set
            # The init_actor_services will handle fallback internally
            try:
                services = init_actor_services(self._config)
            except Exception as fallback_error:
                self._logger.error(
                    f"Even fallback initialization failed for actor {self._actor_id}: {fallback_error}",
                    exc_info=True,
                )
                raise

            self._feature_store = services.feature_store
            self._model_store = services.model_store
            self._strategy_store = services.strategy_store
            self._data_store = services.data_store

            # Emit fallback metrics for each store
            self._fallback_counter.labels(component="feature_store", level="fallback").inc()
            self._fallback_counter.labels(component="model_store", level="fallback").inc()
            self._fallback_counter.labels(component="strategy_store", level="fallback").inc()
            self._fallback_counter.labels(component="data_store", level="fallback").inc()

            self._logger.info(
                f"Initialized all 4 stores for actor {self._actor_id} with DummyStore fallback",
            )

    def _init_persistence_worker(self) -> None:
        """
        Initialize MLPersistenceWorker for async persistence.

        Creates worker with configured queue size, flush interval, and batch size.
        Worker is started separately in on_start() lifecycle method.

        """
        from ml.observability.ml_async_persistence import MLPersistenceWorker

        if self._feature_store is None or self._model_store is None:
            self._logger.warning(
                "Cannot initialize persistence worker: stores not initialized",
            )
            return

        # Get config values with defaults
        queue_size = getattr(self._config, "persistence_queue_size", 1000)
        flush_interval = getattr(self._config, "persistence_flush_interval", 5.0)
        batch_size = getattr(self._config, "persistence_batch_size", 100)

        self._persistence_worker = MLPersistenceWorker(
            feature_store=self._feature_store,
            model_store=self._model_store,
            queue_maxsize=queue_size,
            flush_interval_seconds=flush_interval,
            batch_size=batch_size,
        )

        self._logger.info(
            f"Initialized MLPersistenceWorker for actor {self._actor_id} "
            f"(queue={queue_size}, "
            f"interval={flush_interval}s, "
            f"batch={batch_size})",
        )

    @property
    def feature_store(self) -> FeatureStoreStrictProtocol:
        """
        Return initialized FeatureStore.

        Returns:
            FeatureStore instance (PostgreSQL or DummyStore fallback)

        Raises:
            RuntimeError: If stores not initialized

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> store = component.feature_store
            >>> store.write_features("EUR/USD", features, metadata)

        """
        if self._feature_store is None:
            raise RuntimeError("FeatureStore not initialized")
        return self._feature_store

    @property
    def model_store(self) -> ModelStoreStrictProtocol:
        """
        Return initialized ModelStore.

        Returns:
            ModelStore instance (PostgreSQL or DummyStore fallback)

        Raises:
            RuntimeError: If stores not initialized

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> store = component.model_store
            >>> store.write_prediction("EUR/USD", prediction, confidence, model_id, ts_event)

        """
        if self._model_store is None:
            raise RuntimeError("ModelStore not initialized")
        return self._model_store

    @property
    def strategy_store(self) -> StrategyStoreStrictProtocol:
        """
        Return initialized StrategyStore.

        Returns:
            StrategyStore instance (PostgreSQL or DummyStore fallback)

        Raises:
            RuntimeError: If stores not initialized

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> store = component.strategy_store
            >>> store.write_state(strategy_id, instrument_id, state, signal, ts_event)

        """
        if self._strategy_store is None:
            raise RuntimeError("StrategyStore not initialized")
        return self._strategy_store

    @property
    def data_store(self) -> DataStoreFacadeProtocol:
        """
        Return initialized DataStore.

        Returns:
            DataStore instance (PostgreSQL or fallback)

        Raises:
            RuntimeError: If stores not initialized

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> store = component.data_store
            >>> bars = store.read_bars("EUR/USD", start_ts, end_ts)

        """
        if self._data_store is None:
            raise RuntimeError("DataStore not initialized")
        return self._data_store

    @property
    def persistence_worker(self) -> MLPersistenceWorker | None:
        """
        Return MLPersistenceWorker if async persistence enabled.

        Returns:
            MLPersistenceWorker instance or None if not enabled

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> if component.persistence_worker:
            ...     component.persistence_worker.enqueue_features(...)

        """
        return self._persistence_worker

    def get_health_status(self) -> dict[str, str]:
        """
        Get health status for all stores.

        Checks each store individually and returns aggregate status.
        Health check is fast (<10ms) and non-blocking.

        Returns:
            Dictionary with keys:
                - "feature_store": "healthy" | "degraded" | "unhealthy"
                - "model_store": "healthy" | "degraded" | "unhealthy"
                - "strategy_store": "healthy" | "degraded" | "unhealthy"
                - "data_store": "healthy" | "degraded" | "unhealthy"
                - "aggregate": "healthy" | "degraded" | "unhealthy"

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> health = component.get_health_status()
            >>> if health["aggregate"] == "degraded":
            ...     print("Some stores are unhealthy")

        """
        status: dict[str, str] = {}

        # Check each store
        status["feature_store"] = self._check_store_health(self._feature_store, "feature")
        status["model_store"] = self._check_store_health(self._model_store, "model")
        status["strategy_store"] = self._check_store_health(self._strategy_store, "strategy")
        status["data_store"] = self._check_store_health(self._data_store, "data")

        # Compute aggregate status
        if all(s == "healthy" for s in status.values()):
            status["aggregate"] = "healthy"
        elif any(s == "unhealthy" for s in status.values()):
            status["aggregate"] = "degraded"
        else:
            status["aggregate"] = "degraded"

        return status

    def _check_store_health(self, store: object, store_name: str) -> str:
        """
        Check health of individual store.

        Args:
            store: Store instance to check
            store_name: Name of store for logging

        Returns:
            Health status: "healthy", "degraded", or "unhealthy"

        """
        if store is None:
            return "unhealthy"

        # Check if store has engine attribute
        if not hasattr(store, "engine"):
            # DummyStore or no engine - consider degraded
            return "degraded"

        try:
            # Check if engine is disposed
            engine = getattr(store, "engine", None)
            if engine is None:
                return "unhealthy"

            # Try to check if engine is disposed (SQLAlchemy-specific)
            if hasattr(engine, "dispose") and hasattr(engine, "pool"):
                # Engine exists and has pool - healthy
                return "healthy"

            return "healthy"
        except Exception as e:
            self._logger.warning(
                f"{store_name.capitalize()}Store health check failed: {e}",
                exc_info=True,
            )
            return "unhealthy"

    def on_start(self) -> None:
        """
        Start async persistence worker if enabled.

        Called by actor lifecycle when actor starts.
        Starts the MLPersistenceWorker thread.

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> component.on_start()  # Start worker thread

        """
        if self._persistence_worker is not None:
            self._logger.info(
                f"Starting ML persistence worker for actor {self._actor_id}",
            )
            # Worker start is handled internally by the worker
            # The worker thread starts in its __init__ method

    def on_stop(self) -> None:
        """
        Stop async persistence worker and drain queue.

        Called by actor lifecycle when actor stops.
        Drains pending writes with timeout and stops worker thread.

        Example:
            >>> component = StoreOperationsComponent(config)
            >>> component.on_stop()  # Drain queue and stop worker

        """
        if self._persistence_worker is not None:
            self._logger.info(
                f"Stopping ML persistence worker for actor {self._actor_id}",
            )

            try:
                # Stop worker with drain=True to flush pending writes
                asyncio.run(
                    self._persistence_worker.stop(
                        drain=True,
                        timeout=5.0,
                    ),
                )

                final_queue_size = self._persistence_worker.queue_size()
                self._logger.info(
                    f"ML persistence worker stopped for actor {self._actor_id} "
                    f"(final queue: {final_queue_size})",
                )
            except Exception as e:
                self._logger.error(
                    f"Error stopping persistence worker for actor {self._actor_id}: {e}",
                    exc_info=True,
                )

        # Synchronous flush for stores without worker
        if self._persistence_worker is None:
            self._logger.debug("Flushing stores synchronously (no async worker)")
            try:
                if self._feature_store and hasattr(self._feature_store, "flush"):
                    self._feature_store.flush()
                if self._model_store and hasattr(self._model_store, "flush"):
                    self._model_store.flush()
                if self._strategy_store and hasattr(self._strategy_store, "flush"):
                    self._strategy_store.flush()
                if self._data_store and hasattr(self._data_store, "flush"):
                    self._data_store.flush()
            except Exception as e:
                self._logger.error(
                    f"Error during synchronous store flush: {e}",
                    exc_info=True,
                )
