from __future__ import annotations

from pathlib import Path

from ml.config.events import EventStatus, Source, Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind
from ml.registry.persistence import BackendType, PersistenceConfig


def test_json_backend_register_emit_watermark_roundtrip(tmp_path: Path) -> None:
    # Arrange JSON-backed registry in tmp home
    reg_dir = tmp_path / "registry"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        batch_save_interval=0.0,
    )

    manifest = DatasetManifest(
        dataset_id="features_eurusd",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location=str(tmp_path / "features"),
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=30,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "value": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={"nullability": {"instrument_id": False, "ts_event": False, "ts_init": False}},
        lineage=[],
        pipeline_signature="unit",
        version="1.0.0",
    )

    # Act: register dataset and materialize default contract
    reg.register_dataset(manifest)
    _ = reg.get_contract("features_eurusd")

    # Emit event and update watermark
    reg.emit_event(
        dataset_id="features_eurusd",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run-1",
        ts_min=100,
        ts_max=200,
        count=2,
        status=EventStatus.SUCCESS,
    )
    reg.update_watermark(
        dataset_id="features_eurusd",
        instrument_id="EUR/USD",
        source=Source.HISTORICAL,
        last_success_ns=200,
        count=2,
        completeness_pct=100.0,
    )
    reg.flush()

    # Reload and assert persistence
    reg2 = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        batch_save_interval=0.0,
    )

    m2 = reg2.get_manifest("features_eurusd")
    assert m2.dataset_id == "features_eurusd"
    c2 = reg2.get_contract("features_eurusd")
    assert c2.dataset_id == "features_eurusd"
    w = reg2.get_watermark("features_eurusd", "EUR/USD", Source.HISTORICAL)
    assert w is not None and w.last_success_ns == 200
