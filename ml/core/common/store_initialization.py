"""
Store Initialization Component.

This module provides store initialization extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6.2). The component handles:

- Initialize 4 stores (FeatureStore, ModelStore, StrategyStore, DataStore)
- Initialize EarningsStore for DataStore wiring
- Progressive fallback: PostgreSQL -> File -> Dummy
- File-backed store creation when PostgreSQL unavailable
- Dummy store creation for testing

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.store_initialization import StoreInitializationComponent
>>> from pathlib import Path
>>> component = StoreInitializationComponent(
...     db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
...     file_store_path=Path.home() / ".nautilus" / "ml" / "file_store",
... )
>>> component.init_stores()
>>> print(f"Feature store: {component.feature_store}")

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.metrics_bootstrap import get_counter


if TYPE_CHECKING:
    from ml.stores.feature_dataset_store import FeatureDatasetStore


logger = logging.getLogger(__name__)


# Metric for fallback activations
_FALLBACK_COUNTER = get_counter(
    "ml_fallback_activations_total",
    "Fallback activations",
    labelnames=("component", "level"),
)


@runtime_checkable
class StoreProtocol(Protocol):
    """Protocol for ML store components."""

    def flush(self) -> None:
        """Flush pending writes to storage."""
        ...

    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """Get storage statistics."""
        ...


@dataclass
class StoreInitializationComponent:
    """
    Manages store initialization with progressive fallback.

    Fallback chain: PostgreSQL -> File -> Dummy

    This component implements the store initialization responsibilities
    extracted from MLIntegrationManager. It follows the Progressive Fallback
    Chain pattern (PRIMARY -> CACHED -> FILE -> DUMMY) for resilient
    store initialization.

    Attributes
    ----------
    db_connection : str | None
        PostgreSQL connection string. When None, triggers fallback.
    file_store_path : Path
        Base path for file-backed stores when PostgreSQL unavailable.
    json_fallback : bool
        Whether JSON/dummy fallback is active (set during initialization).
    file_fallback : bool
        Whether file-backed fallback is active (set during initialization).
    feature_store : object
        Initialized FeatureStore (PostgreSQL, File, or Dummy).
    model_store : object
        Initialized ModelStore (PostgreSQL, File, or Dummy).
    strategy_store : object
        Initialized StrategyStore (PostgreSQL, File, or Dummy).
    data_store : object | None
        Initialized DataStore (set in registries init, or via set_data_store).
    earnings_store : object | None
        Initialized EarningsStore (PostgreSQL, File, or Dummy).
    feature_dataset_store : FeatureDatasetStore | None
        Initialized FeatureDatasetStore (PostgreSQL only).

    Example
    -------
    >>> component = StoreInitializationComponent(
    ...     db_connection="postgresql://localhost:5432/nautilus",
    ... )
    >>> component.init_stores()
    >>> component.feature_store.write_features(...)

    """

    db_connection: str | None = None
    file_store_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ML_FILE_STORE_PATH",
                str(Path.home() / ".nautilus" / "ml" / "file_store"),
            )
        )
    )

    # State flags (set during initialization)
    json_fallback: bool = field(default=False, init=False)
    file_fallback: bool = field(default=False, init=False)

    # Stores (initialized during init_stores)
    feature_store: object = field(default=None, init=False)
    feature_dataset_store: FeatureDatasetStore | None = field(default=None, init=False)
    model_store: object = field(default=None, init=False)
    strategy_store: object = field(default=None, init=False)
    data_store: object | None = field(default=None, init=False)
    earnings_store: object | None = field(default=None, init=False)

    def enable_file_fallback(self) -> bool:
        """
        Attempt to enable file-backed fallback stores.

        Creates the file store directory and sets the file_fallback flag.
        Emits a fallback activation metric on success.

        Returns
        -------
        bool
            ``True`` when file-backed stores were initialised successfully.

        Example
        -------
        >>> component = StoreInitializationComponent()
        >>> if not component.enable_file_fallback():
        ...     print("File fallback unavailable, using dummy stores")

        """
        try:
            self.file_store_path.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("file_store_path_unavailable", exc_info=True)
            return False

        try:
            _FALLBACK_COUNTER.labels(
                component="store_initialization",
                level="file",
            ).inc()
        except Exception:
            logger.debug("file_fallback_metric_failed", exc_info=True)

        self.file_fallback = True
        logger.warning(
            "PostgreSQL unavailable - using file-backed ML stores at %s",
            self.file_store_path,
        )
        return True

    def init_dummy_components(self) -> None:
        """
        Initialize in-memory dummy components for testing fallback.

        This mode provides protocol-compatible components without persistence.
        Creates 4 DummyStore instances plus a DummyEarningsStore instance,
        along with 4 DummyRegistry instances for registries (registries are set
        as attributes for consistency).

        Example
        -------
        >>> component = StoreInitializationComponent()
        >>> component.init_dummy_components()
        >>> assert component.feature_store is not None
        >>> assert component.model_store is not None

        """
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore

        # Stores
        self.feature_store = DummyStore()
        self.feature_dataset_store = None
        self.model_store = DummyStore()
        self.strategy_store = DummyStore()
        self.data_store = DummyStore()
        from ml.features.earnings.store import DummyEarningsStore

        self.earnings_store = DummyEarningsStore()

        # Registries (set as attributes for unified access)
        self.feature_registry = DummyRegistry()
        self.model_registry = DummyRegistry()
        self.strategy_registry = DummyRegistry()
        self.data_registry = DummyRegistry()

        logger.debug(
            "Initialized dummy components (4 stores + 4 registries) for fallback mode"
        )

    def init_stores(self) -> None:
        """
        Initialize all store components based on current fallback mode.

        Creates stores appropriate for the active mode:
        - PostgreSQL mode: Creates real FeatureStore, ModelStore, StrategyStore
        - File fallback: Creates FileFeatureStore, FileModelStore, FileStrategyStore, FileDataStore
        - JSON/Dummy fallback: Creates DummyStore instances

        Note: DataStore is initially None in PostgreSQL mode (set in registries init).

        Example
        -------
        >>> component = StoreInitializationComponent(
        ...     db_connection="postgresql://localhost:5432/nautilus",
        ... )
        >>> component.init_stores()
        >>> print(f"Feature store type: {type(component.feature_store).__name__}")

        """
        if self.file_fallback:
            self._init_file_stores()
        elif self.json_fallback:
            self._init_dummy_stores()
        else:
            self._init_postgres_stores()

    def _init_file_stores(self) -> None:
        """Initialize file-backed stores."""
        from ml.stores.file_backed import FileDataStore
        from ml.stores.file_backed import FileEarningsStore
        from ml.stores.file_backed import FileFeatureStore
        from ml.stores.file_backed import FileModelStore
        from ml.stores.file_backed import FileStrategyStore

        file_root = self.file_store_path
        self.feature_store = FileFeatureStore(base_path=file_root / "features")
        self.feature_dataset_store = None
        self.model_store = FileModelStore(base_path=file_root / "models")
        self.strategy_store = FileStrategyStore(base_path=file_root / "strategies")

        # Create earnings store for FileDataStore
        try:
            self.earnings_store = FileEarningsStore(base_path=file_root / "earnings")
            logger.info("FileEarningsStore initialized at %s", file_root / "earnings")
        except Exception:
            from ml.features.earnings.store import DummyEarningsStore

            logger.warning(
                "FileEarningsStore initialization failed; using DummyEarningsStore",
                exc_info=True,
            )
            try:
                _FALLBACK_COUNTER.labels(
                    component="earnings_store",
                    level="dummy",
                ).inc()
            except Exception:
                logger.debug("EarningsStore fallback metric emit failed", exc_info=True)
            self.earnings_store = DummyEarningsStore()

        self.data_store = FileDataStore(
            base_path=file_root / "datastore",
            earnings_store=self.earnings_store,
        )

        logger.debug(
            "Initialized file-backed stores at %s",
            file_root,
        )

    def _init_dummy_stores(self) -> None:
        """Initialize dummy stores for JSON fallback mode."""
        from ml.stores.base import DummyStore

        self.feature_store = DummyStore()
        self.model_store = DummyStore()
        self.strategy_store = DummyStore()
        self.data_store = DummyStore()
        from ml.features.earnings.store import DummyEarningsStore

        self.earnings_store = DummyEarningsStore()

        logger.debug("Initialized dummy stores for JSON fallback mode")

    def _init_postgres_stores(self) -> None:
        """Initialize PostgreSQL-backed stores."""
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig
        from ml.stores.feature_dataset_store import FeatureDatasetStore
        from ml.stores.feature_store import FeatureStore
        from ml.stores.model_store import ModelStore
        from ml.stores.strategy_store import StrategyStore

        if self.db_connection is None:
            raise ValueError(
                "db_connection required for PostgreSQL store initialization"
            )

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
        try:
            self.feature_dataset_store = FeatureDatasetStore(
                connection_string=self.db_connection,
            )
        except Exception:
            logger.warning(
                "FeatureDatasetStore initialization failed; continuing without SQL feature datasets",
                exc_info=True,
            )
            self.feature_dataset_store = None

        self.model_store = ModelStore(
            persistence_config=persistence_config,
            batch_size=1000,
        )

        self.strategy_store = StrategyStore(
            persistence_config=persistence_config,
            batch_size=1000,
        )

        try:
            from ml.features.earnings.store import EarningsStore

            self.earnings_store = EarningsStore(connection_string=self.db_connection)
            logger.info("Initialized EarningsStore (PostgreSQL)")
        except Exception:
            logger.warning(
                "EarningsStore initialization failed; using DummyEarningsStore",
                exc_info=True,
            )
            try:
                _FALLBACK_COUNTER.labels(
                    component="earnings_store",
                    level="dummy",
                ).inc()
            except Exception:
                logger.debug("EarningsStore fallback metric emit failed", exc_info=True)
            from ml.features.earnings.store import DummyEarningsStore

            self.earnings_store = DummyEarningsStore()

        # DataStore is initialized after registries are available
        # Will be set via set_data_store() from registries init
        self.data_store = None

        logger.debug(
            "Initialized PostgreSQL-backed stores with connection: %s",
            self.db_connection[:50] + "..." if len(self.db_connection) > 50 else self.db_connection,
        )

    def set_data_store(self, data_store: object) -> None:
        """
        Set the DataStore after registry initialization.

        In PostgreSQL mode, DataStore is created in the registry initialization
        phase because it requires the DataRegistry reference. This method
        allows setting it after initialization.

        Parameters
        ----------
        data_store : object
            The initialized DataStore instance.

        Example
        -------
        >>> component = StoreInitializationComponent(db_connection="...")
        >>> component.init_stores()
        >>> # Later, after registry init:
        >>> component.set_data_store(data_store)

        """
        self.data_store = data_store
        logger.debug("DataStore set via set_data_store()")

    def flush_all(self) -> None:
        """
        Flush all pending writes to all stores.

        Iterates through all stores and calls flush() if available.
        Logs any flush failures but continues to flush remaining stores.

        Example
        -------
        >>> component = StoreInitializationComponent()
        >>> component.init_stores()
        >>> # ... write operations ...
        >>> component.flush_all()

        """
        stores = [
            ("feature_store", self.feature_store),
            ("feature_dataset_store", self.feature_dataset_store),
            ("model_store", self.model_store),
            ("strategy_store", self.strategy_store),
            ("data_store", self.data_store),
            ("earnings_store", self.earnings_store),
        ]

        for name, store in stores:
            if store is not None and hasattr(store, "flush"):
                try:
                    store.flush()
                except Exception:
                    logger.debug("Flush failed for %s", name, exc_info=True)

        logger.debug("Flushed all stores")

    def get_store_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics from all initialized stores.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of store name to statistics dict.

        Example
        -------
        >>> component = StoreInitializationComponent()
        >>> component.init_stores()
        >>> stats = component.get_store_statistics()
        >>> print(stats["feature_store"])

        """
        result: dict[str, dict[str, Any]] = {}

        stores = [
            ("feature_store", self.feature_store),
            ("feature_dataset_store", self.feature_dataset_store),
            ("model_store", self.model_store),
            ("strategy_store", self.strategy_store),
            ("data_store", self.data_store),
            ("earnings_store", self.earnings_store),
        ]

        for name, store in stores:
            if store is not None and hasattr(store, "get_statistics"):
                try:
                    result[name] = store.get_statistics()
                except Exception:
                    logger.debug("get_statistics failed for %s", name, exc_info=True)
                    result[name] = {"error": "Failed to get statistics"}
            else:
                result[name] = {"status": "not_initialized" if store is None else "no_statistics_method"}

        return result


__all__ = ["StoreInitializationComponent", "StoreProtocol"]
