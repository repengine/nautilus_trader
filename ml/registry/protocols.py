from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ml.registry.dataclasses import DataContract, DatasetManifest


class RegistryProtocol(Protocol):
    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str,
        error: str | None = None,
    ) -> None: ...

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None: ...

    # Read APIs used by DataStore
    def get_manifest(self, dataset_id: str) -> DatasetManifest: ...
    def get_contract(self, dataset_id: str) -> DataContract: ...
    def register_dataset(self, manifest: DatasetManifest) -> str: ...
