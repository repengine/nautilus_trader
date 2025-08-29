from __future__ import annotations

from pathlib import Path

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelRole
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureManifest as _FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRegistry as _FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.model_registry import ModelManifest
from ml.registry.model_registry import ModelRegistry


def _make_feature_registry(tmp_path: Path, feature_names: list[str]) -> tuple[Path, str, str]:
    reg_dir = tmp_path / "feature_registry"
    freg = FeatureRegistry(reg_dir)
    dtypes = ["float32"] * len(feature_names)
    schema = compute_schema_hash(feature_names, dtypes, pipeline_signature="sig_vX")
    manifest = FeatureManifest(
        feature_set_id="",
        name="features",
        version="0.0.1",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=schema,
        pipeline_signature="sig_vX",
        pipeline_version="1",
    )
    fid = freg.register_feature_set(manifest)
    return reg_dir, fid, schema


@pytest.mark.parallel_safe
def test_register_serveable_requires_onnx_and_schema(tmp_path: Path) -> None:
    model_reg_dir = tmp_path / "models"
    mreg = ModelRegistry(model_reg_dir)

    # Prepare a minimal FeatureRegistry entry for parity validation
    freg = _FeatureRegistry(model_reg_dir)
    f_schema = {"f1": "float32"}
    f_manifest = _FeatureManifest(
        feature_set_id="fs_dummy",
        name="dummy",
        version="1.0.0",
        role=_FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=list(f_schema.keys()),
        feature_dtypes=list(f_schema.values()),
        schema_hash="abc",
        pipeline_signature="sig",
        pipeline_version="1.0.0",
    )
    freg.register_feature_set(f_manifest)

    # Good path: ONNX suffix and schema hash
    onnx_path = model_reg_dir / "student.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"ONNX")
    manifest_ok = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32"},
        feature_schema_hash="abc",
        serveable=True,
        artifact_format="onnx",
        feature_set_id="fs_dummy",
    )
    _ = mreg.register_model(onnx_path, manifest_ok)

    # Bad path: wrong suffix
    bad_path = model_reg_dir / "student.pkl"
    bad_path.write_bytes(b"pkl")
    manifest_bad = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32"},
        feature_schema_hash="abc",
        serveable=True,
        artifact_format="pickle",
    )
    try:
        mreg.register_model(bad_path, manifest_bad)
        assert False, "Expected ValueError for non-ONNX serveable model"
    except ValueError as e:
        assert "Only ONNX models" in str(e)


def test_feature_set_id_linkage_and_hash_validation(tmp_path: Path) -> None:
    feature_names = ["f1", "f2"]
    f_reg_dir, feature_set_id, schema_hash = _make_feature_registry(tmp_path, feature_names)

    # Model registry shares the same root for JSON backend simplicity in tests
    mreg = ModelRegistry(f_reg_dir)

    onnx_path = f_reg_dir / "student2.onnx"
    onnx_path.write_bytes(b"ONNX")

    # Mismatch hash -> should raise
    manifest_mismatch = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="wrong_hash",
        feature_set_id=feature_set_id,
        serveable=True,
        artifact_format="onnx",
    )
    try:
        mreg.register_model(onnx_path, manifest_mismatch)
        assert False, "Expected hash mismatch error"
    except ValueError as e:
        assert "feature_schema_hash mismatch" in str(e)

    # Matching hash -> ok and pipeline fields backfilled
    manifest_ok = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash=schema_hash,
        feature_set_id=feature_set_id,
        serveable=True,
        artifact_format="onnx",
    )
    mid = mreg.register_model(onnx_path, manifest_ok)
    info = mreg.get_model(mid)
    assert info is not None
    assert info.manifest.pipeline_signature == "sig_vX"
    assert info.manifest.pipeline_version == "1"


def test_resolve_latest_and_list_compatible(tmp_path: Path) -> None:
    reg_dir = tmp_path / "models3"
    mreg = ModelRegistry(reg_dir)
    onnx_path = reg_dir / "m.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"ONNX")

    # Prepare FeatureRegistry for this model schema/hash
    freg = _FeatureRegistry(reg_dir)
    f_schema = {"f": "float32"}
    freg.register_feature_set(
        _FeatureManifest(
            feature_set_id="fs_dummy",
            name="dummy",
            version="1.0.0",
            role=_FeatureRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            feature_names=list(f_schema.keys()),
            feature_dtypes=list(f_schema.values()),
            schema_hash="hash1",
            pipeline_signature="sig",
            pipeline_version="1.0.0",
        ),
    )

    manifest_v1 = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f": "float32"},
        feature_schema_hash="hash1",
        serveable=True,
        artifact_format="onnx",
        version="1.0.0",
        feature_set_id="fs_dummy",
    )
    manifest_v2 = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f": "float32"},
        feature_schema_hash="hash1",
        serveable=True,
        artifact_format="onnx",
        version="1.0.1",
        feature_set_id="fs_dummy",
    )
    mreg.register_model(onnx_path, manifest_v1)
    mreg.register_model(onnx_path, manifest_v2)

    comps = mreg.list_compatible("hash1", role=ModelRole.STUDENT, architecture="LightGBM")
    assert len(comps) >= 2
    latest = mreg.resolve_latest(ModelRole.STUDENT, "LightGBM", "hash1")
    assert latest is not None
    assert latest.manifest.version == "1.0.1"
