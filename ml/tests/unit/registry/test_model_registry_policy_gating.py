from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.config.policy import RegistryCompatibilityPolicyConfig
from ml.config.registry import RegistryPolicyConfig
from ml.registry import DataRequirements
from ml.registry import FeatureRegistry
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.tests.builders import RegistryBuilder


pytestmark = pytest.mark.unit


def _build_registry_policy(
    *,
    strict_model_compatibility: bool,
    allow_migration_override: bool,
    allow_unsigned_artifacts: bool = False,
    require_output_semantics: bool = False,
) -> RegistryPolicyConfig:
    return RegistryPolicyConfig(
        compatibility_policy=RegistryCompatibilityPolicyConfig(
            strict_model_compatibility=strict_model_compatibility,
            allow_compatibility_migration_override=allow_migration_override,
            allow_unsigned_artifacts=allow_unsigned_artifacts,
            require_output_semantics=require_output_semantics,
        ),
    )


def _build_manifest(
    *,
    model_id: str,
    feature_schema_hash: str = "schema_hash",
    feature_set_id: str | None = None,
    serveable: bool = True,
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="XGBoost",
        feature_schema={"price": "float64"},
        feature_schema_hash=feature_schema_hash,
        feature_set_id=feature_set_id,
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        serveable=serveable,
        artifact_format="onnx",
    )


def _register_matching_feature_set(
    *,
    registry_path: Path,
    schema_hash: str,
) -> str:
    feature_registry = FeatureRegistry(registry_path)
    feature_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="",
        schema_hash=schema_hash,
    )
    return feature_registry.register_feature_set(feature_manifest)


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"policy-gating-onnx-content")
    return model_file


def test_register_model_permissive_mode_allows_missing_feature_set_id(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=False,
            allow_migration_override=True,
        ),
    )
    manifest = _build_manifest(model_id="model_permissive")

    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)

    assert model_id == "model_permissive"


def test_register_model_strict_mode_raises_without_migration_override(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=False,
        ),
    )
    manifest = _build_manifest(model_id="model_strict")

    with pytest.raises(ValueError, match="feature_set_id is required"):
        registry.register_model(model_path=sample_onnx_model, manifest=manifest)


def test_register_model_strict_mode_migration_override_bypasses_violation(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=True,
        ),
    )
    manifest = _build_manifest(model_id="model_override")
    manifest.output_schema = {
        "kind": "binary_proba",
        "shape": [None, 1],
        "classes": [0, 1],
        "positive_class_index": 1,
    }
    manifest.calibration = {"kind": "platt", "params": {"coef": 1.0}}

    with patch("ml.registry.model_registry_facade.registry_compatibility_migration_bypass_total") as metric:
        metric_labels = MagicMock()
        metric.labels.return_value = metric_labels

        model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)

    assert model_id == "model_override"
    metric_labels.inc.assert_called_once()


def test_register_model_from_env_defaults_to_strict_compatibility(
    tmp_path: Path,
    sample_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "ML_STRICT_MODEL_COMPATIBILITY",
        "ML_STRICT_FEATURE_PARITY",
        "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE",
        "ML_ALLOW_UNSIGNED_ARTIFACTS",
        "ML_REQUIRE_OUTPUT_SEMANTICS",
    ):
        monkeypatch.delenv(key, raising=False)

    registry = ModelRegistry(registry_path=tmp_path)
    manifest = _build_manifest(model_id="model_default_strict")

    with pytest.raises(ValueError, match="feature_set_id is required"):
        registry.register_model(model_path=sample_onnx_model, manifest=manifest)


def test_register_model_strict_mode_requires_output_semantics_by_default(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=False,
            require_output_semantics=False,
        ),
    )
    feature_set_id = _register_matching_feature_set(
        registry_path=tmp_path,
        schema_hash="schema_hash",
    )
    manifest = _build_manifest(
        model_id="model_strict_output_default",
        feature_set_id=feature_set_id,
    )

    with pytest.raises(ValueError, match="Output semantics validation failed"):
        registry.register_model(model_path=sample_onnx_model, manifest=manifest)


def test_register_model_strict_output_violation_ignores_migration_override(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=True,
            require_output_semantics=False,
        ),
    )
    feature_set_id = _register_matching_feature_set(
        registry_path=tmp_path,
        schema_hash="schema_hash",
    )
    manifest = _build_manifest(
        model_id="model_strict_output_override_blocked",
        feature_set_id=feature_set_id,
    )

    with patch("ml.registry.model_registry_facade.registry_compatibility_migration_bypass_total") as metric:
        with pytest.raises(ValueError, match="Output semantics validation failed"):
            registry.register_model(model_path=sample_onnx_model, manifest=manifest)
        metric.labels.assert_not_called()


def test_register_model_requires_output_semantics_when_policy_enabled(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=False,
            allow_migration_override=False,
            require_output_semantics=True,
        ),
    )
    manifest = _build_manifest(model_id="model_output_required")

    with pytest.raises(ValueError, match="Output semantics validation failed"):
        registry.register_model(model_path=sample_onnx_model, manifest=manifest)


def test_register_model_permissive_mode_accepts_invalid_output_semantics(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=False,
            allow_migration_override=True,
        ),
    )
    manifest = _build_manifest(model_id="model_invalid_output")
    manifest.output_schema = cast(Any, "invalid_schema")

    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)

    assert model_id == "model_invalid_output"


def test_load_model_missing_digest_raises_when_strict_and_unsigned_not_allowed(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=False,
            allow_unsigned_artifacts=False,
        ),
    )
    manifest = _build_manifest(model_id="model_digest_strict", serveable=False)
    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)

    model_info = registry._models[model_id]
    model_info.manifest.artifact_sha256_digest = None

    with pytest.raises(ValueError, match="No SHA-256 digest available"):
        registry.load_model(model_id)


def test_load_model_missing_digest_permissive_still_loads(
    tmp_path: Path,
    sample_onnx_model: Path,
    mock_onnx_runtime: Any,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=False,
            allow_migration_override=True,
        ),
    )
    manifest = _build_manifest(model_id="model_digest_permissive", serveable=False)
    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)
    registry._models[model_id].manifest.artifact_sha256_digest = None

    loaded_model = registry.load_model(model_id)

    assert loaded_model == mock_onnx_runtime.ort.InferenceSession.return_value


def test_load_model_missing_digest_strict_mode_ignores_migration_override(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=True,
            allow_unsigned_artifacts=False,
        ),
    )
    manifest = _build_manifest(model_id="model_digest_override_blocked", serveable=False)
    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)
    registry._models[model_id].manifest.artifact_sha256_digest = None

    with patch("ml.registry.model_registry_facade.registry_compatibility_migration_bypass_total") as metric:
        with pytest.raises(ValueError, match="No SHA-256 digest available"):
            registry.load_model(model_id)
        metric.labels.assert_not_called()


def test_load_model_missing_digest_allowed_by_unsigned_override(
    tmp_path: Path,
    sample_onnx_model: Path,
    mock_onnx_runtime: Any,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=False,
            allow_unsigned_artifacts=True,
        ),
    )
    manifest = _build_manifest(model_id="model_digest_unsigned", serveable=False)
    model_id = registry.register_model(model_path=sample_onnx_model, manifest=manifest)
    registry._models[model_id].manifest.artifact_sha256_digest = None

    with patch("ml.registry.model_registry_facade.registry_unsigned_artifact_override_total") as metric:
        metric_labels = MagicMock()
        metric.labels.return_value = metric_labels

        loaded_model = registry.load_model(model_id)

    assert loaded_model == mock_onnx_runtime.ort.InferenceSession.return_value
    metric_labels.inc.assert_called_once()


def test_hot_reload_strict_mode_blocks_schema_mismatch(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=False,
        ),
    )

    model_a = _build_manifest(
        model_id="model_a",
        feature_schema_hash=hashlib.sha256(b"schema_a").hexdigest(),
        serveable=False,
    )
    model_b = _build_manifest(
        model_id="model_b",
        feature_schema_hash=hashlib.sha256(b"schema_b").hexdigest(),
        serveable=False,
    )

    registry.register_model(model_path=sample_onnx_model, manifest=model_a)
    registry.register_model(model_path=sample_onnx_model, manifest=model_b)

    assert registry.deploy_model("model_a", "ml_signal_actor") is True

    with pytest.raises(ValueError, match="Feature schema mismatch during hot reload"):
        registry.hot_reload_model("ml_signal_actor", "model_b")


def test_hot_reload_strict_mode_ignores_migration_override(
    tmp_path: Path,
    sample_onnx_model: Path,
) -> None:
    registry = ModelRegistry(
        registry_path=tmp_path,
        policy_config=_build_registry_policy(
            strict_model_compatibility=True,
            allow_migration_override=True,
        ),
    )

    model_a = _build_manifest(
        model_id="model_a_override_blocked",
        feature_schema_hash=hashlib.sha256(b"schema_a").hexdigest(),
        serveable=False,
    )
    model_b = _build_manifest(
        model_id="model_b_override_blocked",
        feature_schema_hash=hashlib.sha256(b"schema_b").hexdigest(),
        serveable=False,
    )

    registry.register_model(model_path=sample_onnx_model, manifest=model_a)
    registry.register_model(model_path=sample_onnx_model, manifest=model_b)
    assert registry.deploy_model("model_a_override_blocked", "ml_signal_actor") is True

    with patch("ml.registry.model_registry_facade.registry_compatibility_migration_bypass_total") as metric:
        with pytest.raises(ValueError, match="Feature schema mismatch during hot reload"):
            registry.hot_reload_model("ml_signal_actor", "model_b_override_blocked")
        metric.labels.assert_not_called()
