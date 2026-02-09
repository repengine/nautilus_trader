from __future__ import annotations

import json
from pathlib import Path

import pytest

import ml.training.event_driven.teacher_export as teacher_export_module
from ml.common.security import calculate_file_sha256
from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.feature_registry import compute_schema_hash
from ml.training.event_driven.teacher_export import StreamingExportValidationResult
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


def test_streaming_export_validation_result_raises_for_errors() -> None:
    result = StreamingExportValidationResult(
        model_id=None,
        registry_root=None,
        artifact_path=None,
        errors=("broken",),
    )

    with pytest.raises(RuntimeError, match="broken"):
        result.raise_for_errors()


def test_validate_streaming_teacher_export_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        validate_streaming_teacher_export(tmp_path / "missing.json")


def test_validate_streaming_teacher_export_raises_for_non_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest payload must be a JSON object"):
        validate_streaming_teacher_export(manifest_path)


def test_validate_streaming_teacher_export_reports_missing_cohort(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert result.errors == ("manifest missing cohort_run",)


def test_validate_streaming_teacher_export_reports_missing_registry_section(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cohort_run": {}}), encoding="utf-8")

    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert result.errors == ("manifest missing cohort_run.model_registry",)


def test_validate_streaming_teacher_export_reports_invalid_registry_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "cohort_run": {
            "model_registry": {
                "model_id": "",
                "registry_root": "",
                "artifact_rel_path": "",
                "artifact_sha256": "",
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert "model_id missing or invalid" in result.errors
    assert "registry_root missing or invalid" in result.errors
    assert "artifact_rel_path missing or invalid" in result.errors


def test_validate_streaming_teacher_export_reports_missing_artifact(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id="model-1",
        artifact_rel_path="staging/missing.npz",
        artifact_sha256="",
    )

    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert any(error.startswith("artifact not found:") for error in result.errors)


def test_validate_streaming_teacher_export_reports_checksum_mismatch(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_path = registry_root / "staging" / "teacher_logits.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"dummy")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id="model-1",
        artifact_rel_path=str(artifact_path.relative_to(registry_root)),
        artifact_sha256="000000",
    )

    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert "artifact_sha256 mismatch" in result.errors


def test_validate_streaming_teacher_export_reports_checksum_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_path = registry_root / "staging" / "teacher_logits.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"dummy")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id="model-1",
        artifact_rel_path=str(artifact_path.relative_to(registry_root)),
        artifact_sha256="abc123",
    )

    def _raise_checksum(_path: Path) -> str:
        raise RuntimeError("checksum failed")

    monkeypatch.setattr(teacher_export_module, "calculate_file_sha256", _raise_checksum)
    result = validate_streaming_teacher_export(manifest_path, require_registry=False)

    assert any("artifact_sha256 check failed" in error for error in result.errors)


def test_validate_streaming_teacher_export_reports_registry_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_path = registry_root / "staging" / "teacher_logits.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"dummy")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        registry_root=registry_root,
        model_id="model-1",
        artifact_rel_path=str(artifact_path.relative_to(registry_root)),
        artifact_sha256=calculate_file_sha256(artifact_path),
    )

    class _ModelInfo:
        manifest = object()

    class _RegistryStub:
        def __init__(self, root: Path) -> None:
            self._root = root

        def get_model(self, model_id: str) -> _ModelInfo | None:
            assert model_id == "model-1"
            return _ModelInfo()

        def _validate_registration_inputs(self, artifact: Path, manifest: object) -> None:
            assert artifact.exists()
            assert manifest is not None
            raise RuntimeError("invalid registration")

    monkeypatch.setattr(teacher_export_module, "ModelRegistry", _RegistryStub)
    result = validate_streaming_teacher_export(manifest_path, require_registry=True)

    assert any("registry validation failed: invalid registration" in error for error in result.errors)
