from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from ml.registry.dataclasses import DataContract
    from ml.registry.dataclasses import DatasetManifest


# Backward-compatible non-generic registry protocol.
# Phase 1: accept enums or strings to enable gradual migration.
class RegistryProtocol(Protocol):
    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None: ...

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None: ...

    # Read APIs used by DataStore
    def get_manifest(self, dataset_id: str) -> DatasetManifest: ...
    def get_contract(self, dataset_id: str) -> DataContract: ...
    def register_dataset(self, manifest: DatasetManifest) -> str: ...


# Phase 1 typed, generic protocol for registries. Adopt incrementally.
TManifest = TypeVar("TManifest")
TKey = TypeVar("TKey")


class TypedRegistryProtocol(Protocol, Generic[TManifest, TKey]):
    """
    Strictly-typed registry interface (adopt incrementally).

    This protocol uses enums for stage/source/status and generic types for
    manifests and keys. Implementations should persist enum ``.value`` where
    strings are required by storage schemas.

    """

    # Core lifecycle
    def get(self, key: TKey) -> TManifest: ...
    def save(self, manifest: TManifest) -> TKey: ...
    def delete(self, key: TKey) -> bool: ...
    def list_manifests(
        self,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[TManifest]: ...
    def batch_save(self, manifests: list[TManifest]) -> list[TKey]: ...

    # Events and watermarks (enum-typed)
    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None: ...

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None: ...


__all__ = [
    "RegistryProtocol",
    "TypedRegistryProtocol",
]
