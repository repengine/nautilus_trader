"""
DataRegistry JSON: update_manifest and get_manifest functional test.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def _manifest(dataset_id: str) -> DatasetManifest:
    schema = {"instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "value": "float64"}
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location="/data",
        partitioning={"by": ["date"]},
        retention_days=7,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="pipe",
        version="1.0.0",
    )


def test_update_manifest_and_get_json(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    mid = reg.register_dataset(_manifest("ds1"))
    reg.update_manifest(mid, {"retention_days": 30, "version": "1.1.0"})
    m = reg.get_manifest(mid)
    assert m.retention_days == 30
    assert m.version == "1.1.0"
