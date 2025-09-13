"""
Focused unit tests for DataRegistry JSON backend and schema/parity enforcement.

These tests avoid database dependencies by using the JSON backend and temporary
directories. They validate event emission, watermark tracking, lineage linking, and
model-feature schema enforcement via environment flags.

"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.base import DataRequirements
from ml.registry.base import ModelRole
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder


def _mk_dataset_manifest(dataset_id: str) -> DatasetManifest:
    schema = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "value": "float64",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location="/tmp",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=7,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline",
        version="1.0.0",
    )


def test_data_registry_json_emit_event_and_watermark(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    cfg = PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir)
    reg = DataRegistry(registry_path=reg_dir, persistence_config=cfg)

    # Register manifest and flush
    manifest = _mk_dataset_manifest("features_test")
    reg.register_dataset(manifest)
    reg.flush()

    # Emit event and ensure it persists
    reg.emit_event(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run_1",
        ts_min=1,
        ts_max=2,
        count=10,
        status=EventStatus.SUCCESS,
        metadata={"foo": "bar"},
    )
    # The JSON backend stores events in memory and flushes to file immediately
    assert len(reg._events) >= 1

    # Update watermark and verify retrieval
    reg.update_watermark(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        source=Source.LIVE,
        last_success_ns=2,
        count=10,
        completeness_pct=100.0,
    )
    wm = reg.get_watermark("features_test", "EUR/USD", Source.LIVE)
    assert wm is not None
    assert wm.last_success_ns == 2
    assert wm.last_count == 10

    # Link lineage entries and ensure they are recorded
    reg.link_lineage(
        child_dataset_id="features_test",
        parent_ids=["bars_eurusd_1m"],
        transform_id="feature_pipeline_v1",
        ts_range={"start_ns": 1, "end_ns": 2},
        params={"lookback": 20},
    )
    assert len(reg._lineage) >= 1


def test_model_feature_schema_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare feature registry JSON with a known schema_hash
    reg_dir = tmp_path / "registry"
    freg = FeatureRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    feature_names = ["f1", "f2"]
    feature_dtypes = ["float32", "float32"]
    pipeline_sig = "sig_v1"
    schema_hash = compute_schema_hash(feature_names, feature_dtypes, pipeline_sig)

    fmanifest = RegistryBuilder.feature_manifest(
        feature_set_id="feat_v1",
        name="test_features",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        schema_hash=schema_hash,
        pipeline_signature=pipeline_sig,
        pipeline_version="1",
        parity_tolerance=1e-10,
    )
    freg.register_feature_set(fmanifest)

    # Build a model manifest with a mismatched feature_schema_hash
    mmanifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="deadbeef",
        serveable=True,
        artifact_format="onnx",
        feature_set_id="feat_v1",
    )

    # Strict parity: mismatch should raise
    monkeypatch.setenv("ML_STRICT_FEATURE_PARITY", "1")

    # Prepare a dummy ONNX file (minimal, existence is validated only)
    model_path = reg_dir / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")

    # Register using the concrete ModelRegistry
    from ml.registry.model_registry import ModelRegistry

    mreg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    with pytest.raises(ValueError):
        mreg.register_model(model_path=model_path, manifest=mmanifest, auto_deploy=False)
