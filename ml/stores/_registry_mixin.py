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
        Lazily initialize and return the DataRegistry instance with progressive
        fallback.

        Order: POSTGRES (if connection string indicates) → JSON file fallback.

        """
        if self._data_registry is not None:
            return self._data_registry

        from ml.registry.data_registry import DataRegistry
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        registry_path = Path.home() / ".nautilus" / "ml" / "registry"

        # Attempt PostgreSQL-backed registry when applicable
        tried_postgres = False
        if self.connection_string and (
            "postgresql://" in self.connection_string or "postgres://" in self.connection_string
        ):
            tried_postgres = True
            try:
                pg_cfg = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=self.connection_string,
                )
                self._data_registry = DataRegistry(
                    registry_path=registry_path,
                    persistence_config=pg_cfg,
                )
                logger.debug("Initialized DataRegistry (POSTGRES)")
                return self._data_registry
            except Exception as e:
                logger.warning("POSTGRES registry init failed, falling back to JSON: %s", e)

        # Fallback to JSON-backed registry (cached/file mode)
        try:
            json_cfg = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
            self._data_registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=json_cfg,
            )
            logger.debug(
                "Initialized DataRegistry (JSON%s)", " after PG fail" if tried_postgres else ""
            )
        except Exception as e:
            logger.warning("Failed to initialize JSON DataRegistry: %s", e)
            self._data_registry = None

        return self._data_registry
