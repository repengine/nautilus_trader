"""
Registry integrator for DataScheduler.

This module handles DataRegistry initialization and dataset registration for lineage
tracking and event emission.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


class RegistryIntegratorProtocol(Protocol):
    """
    Protocol for registry integration operations.
    """

    def initialize_registry(
        self,
        connection: str | None,
    ) -> RegistryProtocol | None:
        """
        Initialize the DataRegistry.

        Parameters
        ----------
        connection : str | None
            Database connection string (optional)

        Returns
        -------
        RegistryProtocol | None
            DataRegistry instance or None if initialization failed

        """
        ...

    def ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type_label: str,
        location: str,
        retention_days: int,
    ) -> None:
        """
        Ensure a dataset manifest exists in the registry.

        Parameters
        ----------
        dataset_id : str
            The dataset identifier
        dataset_type_label : str
            High-level dataset type label
        location : str
            Storage location for the dataset
        retention_days : int
            Data retention period in days

        """
        ...


class RegistryIntegrator:
    """
    Integrate with DataRegistry for event tracking and manifests.

    Implements Pattern 2: Protocol-First Interface Design
    Implements Pattern 4: Progressive Fallback Chains

    This component is responsible ONLY for registry integration.

    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize RegistryIntegrator.

        Parameters
        ----------
        logger : logging.Logger | None
            Logger for operations (default: creates module logger)

        """
        self._logger = logger or logging.getLogger(__name__)
        self._registry: RegistryProtocol | None = None

    def initialize_registry(
        self,
        connection: str | None,
    ) -> RegistryProtocol | None:
        """
        Initialize the DataRegistry for event tracking.

        This method sets up the DataRegistry for emitting data processing events and
        tracking watermarks throughout the pipeline.

        Parameters
        ----------
        connection : str | None
            Database connection string for PostgreSQL backend.
            If None, uses JSON backend in ~/.nautilus/ml/registry

        Returns
        -------
        RegistryProtocol | None
            DataRegistry instance or None if initialization failed

        """
        from ml.registry.data_registry import DataRegistry
        from ml.registry.dataclasses import StorageKind  # noqa: F401
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
                    pass
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=registry_path,
                )

            self._registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=persistence_config,
            )

            self._logger.info(
                "Initialized DataRegistry with backend=%s",
                persistence_config.backend.value,
            )

            return self._registry

        except Exception:
            self._logger.warning(
                "Failed to initialize DataRegistry. Events will not be tracked.",
                exc_info=True,
            )
            return None

    def ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type_label: str,
        location: str,
        retention_days: int,
    ) -> None:
        """
        Ensure a dataset manifest exists in the registry (Postgres backend).

        Parameters
        ----------
        dataset_id : str
            The dataset identifier (e.g., "ohlcv_spy_xnas").
        dataset_type_label : str
            High-level dataset type label ("bars", "trades", "tbbo", "mbp1").
        location : str
            Storage location for the dataset (e.g., catalog path).
        retention_days : int
            Data retention period in days.

        """
        from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
        from ml.registry.dataclasses import DatasetType
        from ml.registry.dataclasses import StorageKind

        if self._registry is None:
            return

        # Map label to DatasetType enum
        dt_map = {
            "bars": DatasetType.BARS,
            "trades": DatasetType.TRADES,
            "tbbo": DatasetType.TBBO,
            "mbp1": DatasetType.MBP1,
        }
        dataset_type = dt_map.get(dataset_type_label, DatasetType.BARS)

        try:
            # If manifest exists, this will succeed
            self._registry.get_manifest(dataset_id)
            return
        except Exception:
            # Register a minimal manifest
            try:
                manifest = build_auto_dataset_manifest(
                    dataset_id=dataset_id,
                    dataset_type=dataset_type,
                    location=location,
                    storage_kind=StorageKind.PARQUET,
                    pipeline_signature="data_scheduler_v1",
                    retention_days=retention_days,
                    metadata={
                        "auto_registered": True,
                        "storage_path": str(Path(location).expanduser()),
                        "source": "data_scheduler",
                    },
                )
                self._registry.register_dataset(manifest)
            except Exception:
                self._logger.debug("Dataset registration skipped or failed", exc_info=True)

    @property
    def registry(self) -> RegistryProtocol | None:
        """
        Get the DataRegistry instance.

        Returns
        -------
        RegistryProtocol | None
            The registry instance or None

        """
        return self._registry
