"""
Registry Initialization Component.

This module provides registry initialization extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6.3). The component handles:

- Initialize 4 registries (FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry)
- Create persistence config (PostgreSQL or JSON backend)
- Wire DataStore with DataRegistry
- Inject DataRegistry into FeatureStore and ModelStore

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.registry_initialization import RegistryInitializationComponent
>>> from pathlib import Path
>>> component = RegistryInitializationComponent(
...     db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
...     registry_path=Path("./ml_registry"),
... )
>>> component.init_registries()
>>> print(f"Feature registry: {component.feature_registry}")

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ml.common.metrics_bootstrap import get_counter
from ml.registry import DataRegistry
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.io_raw import RawReaderProtocol


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.stores.feature_dataset_store import FeatureDatasetStore
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import EarningsStoreProtocol
    from ml.stores.protocols import FeatureStoreProtocol
    from ml.stores.protocols import ModelStoreProtocol
    from ml.stores.protocols import StrategyStoreProtocol


logger = logging.getLogger(__name__)


# Metric for fallback activations
_FALLBACK_COUNTER = get_counter(
    "ml_registry_fallback_activations_total",
    "Registry fallback activations",
    labelnames=("component", "level"),
)


@runtime_checkable
class RegistryProtocol(Protocol):
    """
    Protocol for ML registry components.
    """

    def list_all(self) -> list[Any]:
        """
        List all registered items.
        """
        ...


@runtime_checkable
class StoreWithDataRegistryProtocol(Protocol):
    """
    Protocol for stores that can receive a DataRegistry injection.
    """

    def set_data_registry(self, registry: object) -> None:
        """
        Set the shared DataRegistry reference.
        """
        ...


@dataclass
class RegistryInitializationComponent:
    """
    Manages registry initialization with persistence configuration.

    Creates and configures:
    - FeatureRegistry: Feature schema validation and lifecycle management
    - ModelRegistry: Model deployment tracking and A/B testing
    - StrategyRegistry: Strategy compatibility and requirement validation
    - DataRegistry: Dataset manifest management and lineage tracking
    - DataStore wiring with registries

    This component implements the registry initialization responsibilities
    extracted from MLIntegrationManager. It follows the Progressive Fallback
    Chain pattern (PRIMARY -> CACHED -> FILE -> DUMMY) for resilient
    registry initialization.

    Attributes
    ----------
    db_connection : str | None
        PostgreSQL connection string. When None, triggers JSON fallback.
    json_fallback : bool
        Whether JSON fallback is active (no PostgreSQL).
    file_fallback : bool
        Whether file-backed fallback is active.
    registry_path : Path
        Base path for registry file storage (for JSON backend).
    feature_registry : object
        Initialized FeatureRegistry.
    model_registry : object
        Initialized ModelRegistry.
    strategy_registry : object
        Initialized StrategyRegistry.
    data_registry : object
        Initialized DataRegistry.
    persistence_config : object | None
        The persistence configuration used for registries.

    Example
    -------
    >>> component = RegistryInitializationComponent(
    ...     db_connection="postgresql://localhost:5432/nautilus",
    ... )
    >>> component.init_registries()
    >>> component.feature_registry.list_all()

    """

    db_connection: str | None = None
    json_fallback: bool = False
    file_fallback: bool = False
    registry_path: Path = field(default_factory=lambda: Path("./ml_registry"))

    # Registries (initialized during init_registries)
    feature_registry: object = field(default=None, init=False)
    model_registry: object = field(default=None, init=False)
    strategy_registry: object = field(default=None, init=False)
    data_registry: object = field(default=None, init=False)

    # Persistence configuration (set during init_registries)
    persistence_config: object | None = field(default=None, init=False)

    def init_registries(self) -> None:
        """
        Initialize all 4 registry components.

        Creates FeatureRegistry, ModelRegistry, StrategyRegistry, and DataRegistry
        with appropriate persistence configuration based on the current mode:
        - PostgreSQL mode: Uses BackendType.POSTGRES with connection string
        - JSON/File fallback: Uses BackendType.JSON with registry_path

        Creates the registry_path directory if it doesn't exist.

        Example
        -------
        >>> component = RegistryInitializationComponent(
        ...     db_connection="postgresql://localhost:5432/nautilus",
        ... )
        >>> component.init_registries()
        >>> assert component.feature_registry is not None

        """
        # Import registry components lazily to avoid import-time cycles
        from ml.registry import DataRegistry
        from ml.registry import FeatureRegistry
        from ml.registry import ModelRegistry
        from ml.registry import StrategyRegistry
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        # Create persistence config for registries (DB-first; fallback to JSON)
        if self.file_fallback or self.json_fallback:
            self.persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=self.registry_path,
            )
            try:
                _FALLBACK_COUNTER.labels(
                    component="registry_initialization",
                    level="json" if self.json_fallback else "file",
                ).inc()
            except Exception:
                logger.debug("Registry fallback metric emit failed", exc_info=True)
        else:
            self.persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=self.db_connection,
            )

        # Create a registry path (for file storage)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Initialize registries
        self.feature_registry = FeatureRegistry(
            registry_path=self.registry_path / "features",
            persistence_config=self.persistence_config,
        )
        self.model_registry = ModelRegistry(
            registry_path=self.registry_path / "models",
            persistence_config=self.persistence_config,
        )
        self.strategy_registry = StrategyRegistry(
            base_path=self.registry_path / "strategies",
            persistence_config=self.persistence_config,
        )

        # Initialize DataRegistry
        self.data_registry = DataRegistry(
            registry_path=self.registry_path / "datasets",
            persistence_config=self.persistence_config,
        )

        logger.debug(
            "Initialized 4 registries with persistence backend: %s",
            "JSON" if (self.file_fallback or self.json_fallback) else "POSTGRES",
        )

    def create_data_store(
        self,
        feature_store: FeatureStoreProtocol | None = None,
        feature_dataset_store: FeatureDatasetStore | None = None,
        model_store: ModelStoreProtocol | None = None,
        strategy_store: StrategyStoreProtocol | None = None,
        earnings_store: EarningsStoreProtocol | None = None,
        raw_reader: object | None = None,
        raw_writer: object | None = None,
    ) -> DataStoreFacadeProtocol:
        """
        Create DataStore with the initialized DataRegistry.

        In PostgreSQL mode, creates a DataStore with SQL reader and optional
        Parquet catalog writer. Uses the module-level create_data_store factory
        to avoid mypy abstract instantiation checks.

        Parameters
        ----------
        feature_store : FeatureStoreProtocol | None
            FeatureStore to attach to the DataStore.
        feature_dataset_store : object | None
            FeatureDatasetStore to attach to the DataStore.
        model_store : ModelStoreProtocol | None
            ModelStore to attach to the DataStore.
        strategy_store : StrategyStoreProtocol | None
            StrategyStore to attach to the DataStore.
        earnings_store : EarningsStoreProtocol | None
            EarningsStore to attach to the DataStore.
        raw_reader : object | None
            Optional raw data reader (e.g., SqlMarketDataReader).
            If None in PostgreSQL mode, creates SqlMarketDataReader automatically.
        raw_writer : object | None
            Optional raw data writer (e.g., ParquetCatalogRawWriter).

        Returns
        -------
        DataStoreFacadeProtocol
            The created DataStore instance.

        Raises
        ------
        RuntimeError
            If called before init_registries() or in fallback mode without
            proper setup.

        Example
        -------
        >>> component = RegistryInitializationComponent(
        ...     db_connection="postgresql://localhost:5432/nautilus",
        ... )
        >>> component.init_registries()
        >>> data_store = component.create_data_store()

        """
        if self.data_registry is None:
            raise RuntimeError(
                "Cannot create DataStore before init_registries() is called",
            )

        if self.file_fallback or self.json_fallback:
            # In fallback mode, DataStore is handled by StoreInitializationComponent
            raise RuntimeError(
                "DataStore creation not supported in fallback mode. "
                "Use StoreInitializationComponent for file/dummy stores.",
            )

        # Create SQL reader if not provided
        if raw_reader is None and self.db_connection:
            from ml.stores.providers import SqlMarketDataReader

            table_name = os.getenv("TABLE_NAME", "market_data")
            raw_reader = SqlMarketDataReader(
                connection_string=self.db_connection,
                table_name=table_name,
            )

        # Optionally create Parquet writer if not provided
        if raw_writer is None:
            raw_writer = self._build_raw_writer()

        # Use the module-level factory to avoid mypy issues
        from ml.core.integration import create_data_store as _create_data_store

        connection_string = self.db_connection or ""
        data_store = _create_data_store(
            registry=cast(DataRegistry, self.data_registry),
            connection_string=connection_string,
            feature_store=feature_store,
            feature_dataset_store=feature_dataset_store,
            model_store=model_store,
            strategy_store=strategy_store,
            earnings_store=earnings_store,
            raw_reader=cast(RawReaderProtocol | None, raw_reader),
            raw_writer=cast(RawIngestionWriterProtocol | None, raw_writer),
        )

        return data_store

    def _build_raw_writer(self) -> RawIngestionWriterProtocol | None:
        """
        Build a composite raw writer for market + feature dataset mirrors.

        Returns None when no parquet destinations are configured.
        """
        from ml.registry.dataclasses import DatasetType
        from ml.stores.feature_raw_writer import CompositeRawIngestionWriter
        from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
        from ml.stores.io_raw import FilteredRawWriter
        from ml.stores.io_raw import ParquetCatalogRawWriter

        writers: list[RawIngestionWriterProtocol] = []

        try:
            feature_writer = FilteredRawWriter(
                FeatureDatasetParquetRawWriter(),
                enabled={
                    DatasetType.EVENTS_CALENDAR: True,
                    DatasetType.MICRO_MINUTE_FEATURES: True,
                    DatasetType.L2_MINUTE_FEATURES: True,
                },
            )
            writers.append(feature_writer)
        except Exception:
            logger.debug("Feature raw writer not attached", exc_info=True)

        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

            catalog_path = os.getenv("CATALOG_PATH", "").strip()
            if catalog_path:
                catalog = ParquetDataCatalog(catalog_path)
                market_writer = FilteredRawWriter(
                    ParquetCatalogRawWriter(catalog),
                    enabled={
                        DatasetType.BARS: True,
                        DatasetType.TRADES: True,
                        DatasetType.TBBO: True,
                        DatasetType.MBP1: True,
                        DatasetType.QUOTES: True,
                    },
                )
                writers.append(market_writer)
        except Exception:
            logger.debug("Parquet catalog adapters not attached", exc_info=True)

        if not writers:
            return None
        if len(writers) == 1:
            return writers[0]
        return CompositeRawIngestionWriter(writers)

    def inject_data_registry_into_stores(
        self,
        feature_store: object | None,
        model_store: object | None,
    ) -> None:
        """
        Inject the shared DataRegistry into FeatureStore and ModelStore.

        Ensures FeatureStore and ModelStore publish into the same DataRegistry
        instance for unified data tracking and lineage.

        Parameters
        ----------
        feature_store : object | None
            The FeatureStore to inject DataRegistry into (if supported).
        model_store : object | None
            The ModelStore to inject DataRegistry into (if supported).

        Example
        -------
        >>> component = RegistryInitializationComponent(...)
        >>> component.init_registries()
        >>> component.inject_data_registry_into_stores(feature_store, model_store)

        """
        if self.data_registry is None:
            logger.debug(
                "Cannot inject DataRegistry - registries not initialized",
            )
            return

        try:
            # Inject into FeatureStore if supported
            if feature_store is not None:
                setter = getattr(feature_store, "set_data_registry", None)
                if callable(setter):
                    setter(self.data_registry)
                    logger.debug("Injected DataRegistry into FeatureStore")

            # Inject into ModelStore if supported
            if model_store is not None:
                setter2 = getattr(model_store, "set_data_registry", None)
                if callable(setter2):
                    setter2(self.data_registry)
                    logger.debug("Injected DataRegistry into ModelStore")

        except Exception:
            logger.debug(
                "Failed to inject shared DataRegistry into stores",
                exc_info=True,
            )

    def get_persistence_backend(self) -> str:
        """
        Get the current persistence backend type as a string.

        Returns
        -------
        str
            The backend type: "POSTGRES", "JSON", or "UNINITIALIZED".

        Example
        -------
        >>> component = RegistryInitializationComponent(db_connection="...")
        >>> component.init_registries()
        >>> print(component.get_persistence_backend())
        'POSTGRES'

        """
        if self.persistence_config is None:
            return "UNINITIALIZED"

        from ml.registry.persistence import BackendType

        backend = getattr(self.persistence_config, "backend", None)
        if backend == BackendType.POSTGRES:
            return "POSTGRES"
        elif backend == BackendType.JSON:
            return "JSON"
        else:
            return "UNKNOWN"

    def get_registry_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics from all initialized registries.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of registry name to statistics dict.

        Example
        -------
        >>> component = RegistryInitializationComponent(...)
        >>> component.init_registries()
        >>> stats = component.get_registry_statistics()
        >>> print(stats["feature_registry"])

        """
        result: dict[str, dict[str, Any]] = {}

        registries = [
            ("feature_registry", self.feature_registry),
            ("model_registry", self.model_registry),
            ("strategy_registry", self.strategy_registry),
            ("data_registry", self.data_registry),
        ]

        for name, registry in registries:
            if registry is None:
                result[name] = {"status": "not_initialized"}
            else:
                try:
                    # Try to get count via list_all or similar
                    list_method = getattr(registry, "list_all", None)
                    if callable(list_method):
                        items = list_method()
                        result[name] = {
                            "status": "initialized",
                            "count": len(items) if items else 0,
                        }
                    else:
                        result[name] = {"status": "initialized", "count": "unknown"}
                except Exception:
                    logger.debug("get_statistics failed for %s", name, exc_info=True)
                    result[name] = {"error": "Failed to get statistics"}

        result["persistence_backend"] = {"backend": self.get_persistence_backend()}

        return result


def build_data_store(
    *,
    db_connection: str,
    raw_reader: RawReaderProtocol | None = None,
    raw_writer: RawIngestionWriterProtocol | None = None,
) -> DataStoreFacadeProtocol:
    """
    Build a fully initialized DataStore with registries and stores.

    Args:
        db_connection: PostgreSQL connection string.
        raw_reader: Optional raw reader override (defaults to SQL reader).
        raw_writer: Optional raw writer override (defaults to composite parquet writers).

    Returns:
        Initialized DataStore facade.
    """
    from ml.core.common.store_initialization import StoreInitializationComponent
    from ml.registry.base import DummyRegistry
    from ml.stores.protocols import EarningsStoreProtocol
    from ml.stores.protocols import FeatureStoreProtocol
    from ml.stores.protocols import ModelStoreProtocol
    from ml.stores.protocols import StrategyStoreProtocol

    store_init = StoreInitializationComponent(db_connection=db_connection)
    store_init.init_stores()
    if store_init.file_fallback or store_init.json_fallback:
        msg = "DataStore requires PostgreSQL-backed stores"
        raise RuntimeError(msg)

    registry_init = RegistryInitializationComponent(db_connection=db_connection)
    registry_init.init_registries()
    if isinstance(registry_init.data_registry, DummyRegistry):
        msg = "DataStore requires an initialized DataRegistry"
        raise RuntimeError(msg)

    registry_init.inject_data_registry_into_stores(
        store_init.feature_store,
        store_init.model_store,
    )

    return registry_init.create_data_store(
        feature_store=cast(FeatureStoreProtocol, store_init.feature_store),
        feature_dataset_store=store_init.feature_dataset_store,
        model_store=cast(ModelStoreProtocol, store_init.model_store),
        strategy_store=cast(StrategyStoreProtocol, store_init.strategy_store),
        earnings_store=cast(EarningsStoreProtocol, store_init.earnings_store),
        raw_reader=raw_reader,
        raw_writer=raw_writer,
    )


__all__ = [
    "RegistryInitializationComponent",
    "RegistryProtocol",
    "StoreWithDataRegistryProtocol",
    "build_data_store",
]
