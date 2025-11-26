"""
Dataset registration component extracted from DataScheduler.

This component handles dataset registration logic for DataScheduler including:
- Ensuring dataset manifests exist in the DataRegistry
- Mapping dataset type labels to DatasetType enums
- Building and registering auto-generated dataset manifests

Extracted from legacy DataScheduler (lines 320-373):
- _ensure_dataset_registered() (lines 320-373)

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


# Mapping from dataset type labels to DatasetType enum values
_DATASET_TYPE_MAP: dict[str, DatasetType] = {
    "bars": DatasetType.BARS,
    "trades": DatasetType.TRADES,
    "tbbo": DatasetType.TBBO,
    "mbp1": DatasetType.MBP1,
}


class DatasetRegistrationProtocol(Protocol):
    """
    Protocol for dataset registration operations.

    This protocol defines the contract for dataset registration components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    ensure_dataset_registered
        Ensure dataset manifest exists in registry, registering if needed.

    """

    def ensure_dataset_registered(
        self,
        registry: RegistryProtocol | None,
        dataset_id: str,
        dataset_type_label: str,
        location: str,
        retention_days: int,
    ) -> None:
        """
        Ensure dataset manifest exists in registry.

        Args:
            registry: DataRegistry instance or None if unavailable.
            dataset_id: The dataset identifier (e.g., "ohlcv_spy_xnas").
            dataset_type_label: High-level dataset type label
                ("bars", "trades", "tbbo", "mbp1").
            location: Storage location for the dataset (e.g., catalog path).
            retention_days: Number of days to retain data.

        """
        ...


class DatasetRegistrationComponent:
    """
    Component for dataset registration logic in DataScheduler.

    This component extracts dataset registration responsibilities from DataScheduler,
    providing focused methods for:
    - Mapping dataset type labels to DatasetType enums
    - Checking if dataset manifests already exist
    - Building and registering auto-generated manifests

    All methods are designed to handle errors gracefully and log appropriate
    debug messages without raising exceptions that would prevent scheduler
    operation.

    Example:
        >>> from ml.registry.data_registry import DataRegistry
        >>> component = DatasetRegistrationComponent()
        >>> registry = DataRegistry(registry_path=Path("/tmp/registry"))
        >>> component.ensure_dataset_registered(
        ...     registry=registry,
        ...     dataset_id="ohlcv_spy_xnas",
        ...     dataset_type_label="bars",
        ...     location="/data/catalogs/bars",
        ...     retention_days=90,
        ... )

    """

    def map_dataset_type(self, dataset_type_label: str) -> DatasetType:
        """
        Map a dataset type label string to a DatasetType enum.

        Args:
            dataset_type_label: High-level dataset type label
                ("bars", "trades", "tbbo", "mbp1").

        Returns:
            Corresponding DatasetType enum value. Defaults to BARS
            if label is not recognized.

        Example:
            >>> component = DatasetRegistrationComponent()
            >>> component.map_dataset_type("trades")
            <DatasetType.TRADES: 'trades'>

        """
        return _DATASET_TYPE_MAP.get(dataset_type_label, DatasetType.BARS)

    def ensure_dataset_registered(
        self,
        registry: RegistryProtocol | None,
        dataset_id: str,
        dataset_type_label: str,
        location: str,
        retention_days: int,
    ) -> None:
        """
        Ensure dataset manifest exists in registry, registering if needed.

        This method implements the dataset registration logic extracted from
        DataScheduler._ensure_dataset_registered. It:

        1. Returns early if registry is None
        2. Maps the dataset_type_label to a DatasetType enum
        3. Tries to get existing manifest first, returning if found
        4. Builds and registers a new manifest if not found
        5. Catches all exceptions silently (debug log only)

        This method never raises exceptions - it logs debug messages and returns
        silently on failure to allow the scheduler to continue without
        registration.

        Args:
            registry: DataRegistry instance or None if unavailable.
            dataset_id: The dataset identifier (e.g., "ohlcv_spy_xnas").
            dataset_type_label: High-level dataset type label
                ("bars", "trades", "tbbo", "mbp1").
            location: Storage location for the dataset (e.g., catalog path).
            retention_days: Number of days to retain data.

        Example:
            >>> from ml.registry.data_registry import DataRegistry
            >>> component = DatasetRegistrationComponent()
            >>> registry = DataRegistry(registry_path=Path("/tmp/registry"))
            >>> component.ensure_dataset_registered(
            ...     registry=registry,
            ...     dataset_id="ohlcv_spy_xnas",
            ...     dataset_type_label="bars",
            ...     location="/data/catalogs/bars",
            ...     retention_days=90,
            ... )

        """
        # Return early if no registry
        if registry is None:
            return

        # Map label to DatasetType enum
        dataset_type = self.map_dataset_type(dataset_type_label)

        try:
            # If manifest exists, this will succeed
            registry.get_manifest(dataset_id)
            return
        except Exception:
            # Manifest doesn't exist, need to register
            pass

        # Register a minimal manifest
        try:
            # Lazy import to avoid circular dependencies
            from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest

            manifest = build_auto_dataset_manifest(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                location=str(Path(location).expanduser()),
                storage_kind=StorageKind.PARQUET,
                pipeline_signature="data_scheduler_v1",
                retention_days=retention_days,
                metadata={
                    "auto_registered": True,
                    "storage_path": str(Path(location).expanduser()),
                    "source": "data_scheduler",
                },
            )
            registry.register_dataset(manifest)
            logger.debug(
                "Registered dataset manifest: %s (type=%s)",
                dataset_id,
                dataset_type.value,
            )
        except Exception:
            logger.debug(
                "Dataset registration skipped or failed for %s",
                dataset_id,
                exc_info=True,
            )


__all__ = [
    "DatasetRegistrationComponent",
    "DatasetRegistrationProtocol",
]
