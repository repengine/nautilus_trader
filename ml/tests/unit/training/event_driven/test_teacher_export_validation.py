from __future__ import annotations

import json
from pathlib import Path

from ml.common.security import calculate_file_sha256
from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.feature_registry import compute_schema_hash
from ml.training.event_driven.teacher_export import validate_streaming_teacher_export


def _write_manifest(
    path: Path,
    *,
    registry_root: Path,
    model_id: str,
    artifact_rel_path: str,
    artifact_sha256: str,
) -> None:
    payload = {
        "cohort_run": {
            "model_registry": {
                "model_id": model_id,
                "registry_root": str(registry_root),
                "artifact_rel_path": artifact_rel_path,
                "artifact_sha256": artifact_sha256,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _register_dummy_model(registry_root: Path, artifact_path: Path) -> str:
    feature_schema = {"feature": "float32"}
    schema_hash = compute_schema_hash(
        list(feature_schema.keys()),
        list(feature_schema.values()),
        pipeline_signature="streaming",
    )
    manifest = ModelManifest(
        model_id="",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.STREAMING,
        architecture="tft_streaming_teacher",
        feature_schema=feature_schema,
        feature_schema_hash=schema_hash,
        serveable=False,
        artifact_format="npz",
        pipeline_signature="streaming",
        pipeline_version="1.0.0",
    )
    registry = ModelRegistry(registry_root)
    return registry.register_model(model_path=artifact_path, manifest=manifest, auto_deploy=False)


def test_validate_streaming_teacher_export_success(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_path = registry_root / "staging" / "teacher_logits.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"dummy")

    model_id = _register_dummy_model(registry_root, artifact_path)
    manifest_path = tmp_path / "manifest.json"
    digest = calculate_file_sha256(artifact_path)
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id=model_id,
        artifact_rel_path=str(artifact_path.relative_to(registry_root)),
        artifact_sha256=digest,
    )

    result = validate_streaming_teacher_export(manifest_path, require_registry=True)

    assert not result.errors
    assert result.model_id == model_id


def test_validate_streaming_teacher_export_missing_model(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_path = registry_root / "staging" / "teacher_logits.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"dummy")

    manifest_path = tmp_path / "manifest.json"
    digest = calculate_file_sha256(artifact_path)
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id="missing",
        artifact_rel_path=str(artifact_path.relative_to(registry_root)),
        artifact_sha256=digest,
    )

    result = validate_streaming_teacher_export(manifest_path, require_registry=True)

    assert result.errors
