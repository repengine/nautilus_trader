"""
Strict parity mismatch raises in ModelRegistry when serveable and feature_set_id
missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.registry import DataRequirements
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder


def test_strict_parity_requires_feature_set_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_STRICT_FEATURE_PARITY", "1")
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    model_path = reg_dir / "m.onnx"
    model_path.write_bytes(b"onnx")

    manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f": "float32"},
        feature_schema_hash="hash",
        serveable=True,
        artifact_format="onnx",
        feature_set_id=None,  # Missing
    )
    with pytest.raises(ValueError):
        reg.register_model(model_path=model_path, manifest=manifest, auto_deploy=False)
