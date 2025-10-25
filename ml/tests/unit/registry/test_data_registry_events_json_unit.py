from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def _make_minimal_manifest(dataset_id: str, location: Path) -> DatasetManifest:
    schema: dict[str, str] = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location=str(location),
        partitioning={},
        retention_days=1,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",  # will be computed
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline",
        version="1.0.0",
    )


class TestDataRegistryJSON:
    def test_emit_event_json_backend_flushes_immediately(self, tmp_path: Path) -> None:
        registry_dir = tmp_path / "registry"
        registry = DataRegistry(
            registry_path=registry_dir,
            persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=registry_dir),
        )

        manifest = _make_minimal_manifest("features_dataset", tmp_path / "features.parquet")
        registry.register_dataset(manifest)

        # Emit one event and validate it is persisted to JSON immediately
        registry.emit_event(
            dataset_id=manifest.dataset_id,
            instrument_id="EURUSD.SIM",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="run_test",
            ts_min=1000,
            ts_max=2000,
            count=5,
            status=EventStatus.SUCCESS,
            metadata={"k": "v"},
        )

        # Explicit flush required for tests that verify persistence
        # (pytest detection skips automatic saves to avoid O(N²) serialization)
        registry.flush()

        # Read raw JSON and check event presence
        registry_file = registry_dir / "data_registry.json"
        assert registry_file.exists()
        data: dict[str, Any] = json.loads(registry_file.read_text())
        events = data.get("events", [])
        assert len(events) >= 1
        evt = events[-1]
        assert evt["dataset_id"] == manifest.dataset_id
        assert evt["instrument_id"] == "EURUSD.SIM"
        assert evt["stage"] == "FEATURE_COMPUTED"
        assert evt["status"] == "success"
        assert "created_at" in evt
        assert isinstance(evt.get("metadata", {}), dict)
