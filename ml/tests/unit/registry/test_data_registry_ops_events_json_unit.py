from __future__ import annotations

import json
from pathlib import Path

from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def _manifest(dataset_id: str, location: Path) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location=str(location),
        partitioning={},
        retention_days=1,
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
    )


class TestDataRegistryOpsEventsJson:
    def test_register_update_deprecate_emit_events(self, tmp_path: Path) -> None:
        reg_dir = tmp_path / "reg"
        reg = DataRegistry(
            registry_path=reg_dir,
            persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        )

        ds_id = "features.test"
        reg.register_dataset(_manifest(ds_id, tmp_path / "f.parquet"))
        reg.update_manifest(ds_id, {"version": "1.0.1"})
        reg.deprecate(ds_id)

        data = json.loads((reg_dir / "data_registry.json").read_text())
        events = data.get("events", [])

        # At least three ops-related events present
        assert any(e.get("dataset_id") == ds_id and e.get("stage") == Stage.CATALOG_WRITTEN.value for e in events)
        assert any(e.get("dataset_id") == ds_id and e.get("status") == "deprecated" for e in events)
        # Correlation id should be present on events
        assert all("correlation_id" in (e.get("metadata") or {}) for e in events if e.get("dataset_id") == ds_id)

