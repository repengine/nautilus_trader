#!/usr/bin/env python3

"""
Abstract helpers for ML registries (Feature/Model/Strategy).

This module centralizes common, non-hot-path concerns shared by the concrete
registries to reduce duplication without changing their public APIs:

- RLock lifecycle and exposure via ``self._lock``
- Dual-backend setup via ``PersistenceManager`` (JSON/POSTGRES)
- JSON save/load helpers (delegating to ``PersistenceManager``)
- Audit logging passthrough (``log_audit``)
- Health summary implementation (count + last_modified hook)

DataRegistry remains separate due to distinct event/watermark/time-series
semantics.
"""

from __future__ import annotations

import threading
import time
from abc import ABC
from abc import abstractmethod
from typing import Any

from ml.common.protocols import MLComponentMixin
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceManager


class AbstractRegistry(MLComponentMixin, ABC):
    """
    Common base for Feature/Model/Strategy registries.

    Subclasses must maintain their own domain collections (e.g. ``self._features``,
    ``self._models``) and implement ``_health_snapshot`` to report entry counts and
    last-modified timestamps for health reporting.
    """

    def __init__(self, persistence: PersistenceManager) -> None:
        # Thread-safety for registry operations
        self._lock = threading.RLock()

        # Persistence wiring and backend flag
        self.persistence = persistence
        self.backend: BackendType = persistence.config.backend

    # --------------------------- JSON helper utilities ---------------------------
    def _json_load(self, filename: str) -> dict[str, Any] | None:
        """Load JSON via PersistenceManager (JSON backend only)."""
        if self.backend != BackendType.JSON:
            return None
        return self.persistence.load_json(filename)

    def _json_save(self, filename: str, data: dict[str, Any]) -> None:
        """Save JSON via PersistenceManager (JSON backend only)."""
        if self.backend != BackendType.JSON:
            return
        self.persistence.save_json(data, filename)

    # ------------------------------- Audit logging -------------------------------
    def log_audit(
        self,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        changes: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        """Passthrough to ``PersistenceManager.log_audit`` for consistency."""
        self.persistence.log_audit(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            changes=changes,
            user_id=user_id,
        )

    # --------------------------------- Health ----------------------------------
    @abstractmethod
    def _health_snapshot(self) -> tuple[int, float | None]:
        """
        Return (count, last_modified_epoch_seconds_or_None) for health reporting.

        Subclasses should compute:
        - count: number of entries (manifests/rows) tracked in memory
        - last_modified: the maximum ``last_modified`` timestamp across entries, or None
        """

    def get_health_status(self) -> dict[str, Any]:
        count, last_modified = self._health_snapshot()
        return {
            "component": getattr(self, "_component_name", None) or self.__class__.__name__,
            "status": "ok",
            "timestamp": time.time(),
            "backend": self.backend.value,
            "count": count,
            "last_modified": last_modified,
        }


__all__ = ["AbstractRegistry"]
