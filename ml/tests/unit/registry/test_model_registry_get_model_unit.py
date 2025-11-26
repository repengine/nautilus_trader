"""
ModelRegistry JSON: get_model retrieval returns ModelInfo with manifest fields.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry import DataRequirements
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder


def test_get_model_returns_info(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    model_path = reg_dir / "m.onnx"
    model_path.write_bytes(b"onnx")

    manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f": "float32"},
        feature_schema_hash="hash",
        serveable=True,
        artifact_format="onnx",
        feature_set_id=None,
    )
    mid = reg.register_model(model_path=model_path, manifest=manifest, auto_deploy=False)
    info = reg.get_model(mid)
    assert info is not None
    assert info.manifest.model_id == mid
    assert info.model_path.name.endswith(".onnx")
