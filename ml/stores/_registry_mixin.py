"""
Shared DataRegistry initialization mixin for stores.

Provides a common, lazy `_get_data_registry` implementation used by
multiple store classes to avoid duplicated logic.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


class DataRegistryMixin:
    """
    Mixin that provides lazy DataRegistry initialization.

    Expects the consumer class to define `connection_string` attribute.

    """

    _data_registry: RegistryProtocol | None = None
    connection_string: str | None

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Lazily initialize and return the DataRegistry instance.
        """
        if self._data_registry is not None:
            return self._data_registry

        try:
            from ml.registry.data_registry import DataRegistry
            from ml.registry.persistence import BackendType
            from ml.registry.persistence import PersistenceConfig

            registry_path = Path.home() / ".nautilus" / "ml" / "registry"

            # Determine backend based on connection string
            if self.connection_string and (
                "postgresql://" in self.connection_string or "postgres://" in self.connection_string
            ):
                persistence_config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=self.connection_string,
                )
            else:
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=registry_path,
                )

            self._data_registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=persistence_config,
            )
            logger.debug("Initialized DataRegistry for event emission")
        except Exception as e:
            logger.warning("Failed to initialize DataRegistry: %s", e)
            self._data_registry = None

        return self._data_registry
