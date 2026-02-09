#!/usr/bin/env python3

"""
Unit tests for ModelRegistryFacade.

These tests verify that the facade correctly wires all components and
provides the canonical ModelRegistry API.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.config.registry import RegistryPolicyConfig
from ml.registry.base import (
    DataRequirements,
    DeploymentStatus,
    ModelInfo,
    ModelManifest,
    ModelRole,
)
from ml.registry.dataclasses import CanaryConfig, QualityGate
from ml.registry.model_registry_facade import ModelRegistryFacade
from ml.registry.persistence import BackendType
from ml.tests.utils.model_artifacts import default_calibration
from ml.tests.utils.model_artifacts import default_output_schema
from ml.tests.utils.model_artifacts import register_feature_set_for_schema

pytestmark = pytest.mark.usefixtures("isolated_registry_policy_env")


def _build_strict_serveable_manifest(
    *,
    model_id: str,
    feature_schema: dict[str, str],
    feature_schema_hash: str,
    feature_set_id: str,
    architecture: str = "XGBoost",
    version: str = "1.0.0",
    performance_metrics: dict[str, float] | None = None,
) -> ModelManifest:
    """Build a strict-valid serveable manifest for non-policy behavior tests."""
    return ModelManifest(
        model_id=model_id,
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture=architecture,
        feature_schema=feature_schema,
        feature_schema_hash=feature_schema_hash,
        version=version,
        created_at=time.time(),
        last_modified=time.time(),
        serveable=True,
        artifact_format="onnx",
        feature_set_id=feature_set_id,
        output_schema=default_output_schema(),
        calibration=default_calibration(),
        performance_metrics=performance_metrics or {},
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    """Create temporary registry path."""
    registry_path = tmp_path / "model_registry"
    registry_path.mkdir(parents=True, exist_ok=True)
    return registry_path


@pytest.fixture
def sample_onnx_model(tmp_registry_path: Path) -> tuple[Path, str]:
    """Create a sample ONNX model file with known SHA-256 digest."""
    model_file = tmp_registry_path / "test_model.onnx"
    content = b"sample ONNX model content for testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


@pytest.fixture
def sample_manifest(
    tmp_registry_path: Path,
    sample_onnx_model: tuple[Path, str],
) -> ModelManifest:
    """Create sample ModelManifest for testing."""
    del sample_onnx_model
    schema_hash = "test_hash_abc123"
    feature_set_id = register_feature_set_for_schema(
        registry_path=tmp_registry_path,
        schema_hash=schema_hash,
    )
    return _build_strict_serveable_manifest(
        model_id="test_model_001",
        feature_schema={"price": "float64", "volume": "float64"},
        feature_schema_hash=schema_hash,
        feature_set_id=feature_set_id,
    )


@pytest.fixture
def facade(tmp_registry_path: Path) -> ModelRegistryFacade:
    """Create ModelRegistryFacade instance for testing."""
    return ModelRegistryFacade(registry_path=tmp_registry_path)


@pytest.fixture
def facade_with_registered_models(
    facade: ModelRegistryFacade,
    sample_onnx_model: tuple[Path, str],
) -> ModelRegistryFacade:
    """Create facade with 3 pre-registered models."""
    model_path, _ = sample_onnx_model
    schema_hash = "test_hash_123"
    feature_set_id = register_feature_set_for_schema(
        registry_path=facade.registry_path,
        schema_hash=schema_hash,
    )

    for i in range(3):
        manifest = _build_strict_serveable_manifest(
            model_id=f"model_{i}",
            feature_schema={"price": "float64"},
            feature_schema_hash=schema_hash,
            feature_set_id=feature_set_id,
            version=f"1.0.{i}",
            performance_metrics={"accuracy": 0.90 + i * 0.01},
        )
        facade.register_model(model_path, manifest)

    return facade


# =============================================================================
# Initialization Tests
# =============================================================================


class TestFacadeInitialization:
    """Tests for facade initialization."""

    def test_facade_initializes_with_defaults(self, tmp_registry_path: Path) -> None:
        """Verify facade initializes with default configuration."""
        facade = ModelRegistryFacade(registry_path=tmp_registry_path)

        assert facade.registry_path == tmp_registry_path
        assert facade.cache_size == 10
        assert facade.batch_save_interval == 0.1
        assert facade._persistence is not None
        assert facade._deployment is not None
        assert facade._ab_testing is not None
        assert facade._version is not None

    def test_facade_initializes_with_custom_config(self, tmp_registry_path: Path) -> None:
        """Verify facade initializes with custom configuration."""
        policy = RegistryPolicyConfig(
            max_inference_latency_ms=2,
            ab_models_required=2,
        )
        facade = ModelRegistryFacade(
            registry_path=tmp_registry_path,
            cache_size=20,
            batch_save_interval=0.5,
            policy_config=policy,
        )

        assert facade.cache_size == 20
        assert facade.batch_save_interval == 0.5
        assert facade._policy.max_inference_latency_ms == 2

    def test_facade_creates_registry_directory(self, tmp_path: Path) -> None:
        """Verify facade creates registry directory if it doesn't exist."""
        registry_path = tmp_path / "new_registry"
        assert not registry_path.exists()

        ModelRegistryFacade(registry_path=registry_path)

        assert registry_path.exists()


# =============================================================================
# Core Model Operations Tests
# =============================================================================


class TestCoreModelOperations:
    """Tests for core model operations."""

    def test_register_model_success(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """Verify successful model registration."""
        model_path, expected_digest = sample_onnx_model

        model_id = facade.register_model(model_path, sample_manifest)

        assert model_id == sample_manifest.model_id
        model_info = facade.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.artifact_sha256_digest == expected_digest

    def test_register_model_calculates_sha256(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """Verify SHA-256 digest is calculated during registration."""
        model_path, expected_digest = sample_onnx_model

        facade.register_model(model_path, sample_manifest)

        model_info = facade.get_model(sample_manifest.model_id)
        assert model_info is not None
        assert model_info.manifest.artifact_sha256_digest == expected_digest
        assert len(model_info.manifest.artifact_sha256_digest) == 64  # SHA-256 hex length

    def test_register_model_ingests_sidecar_metadata(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify registration ingests output schema + calibration from sidecar."""
        model_path, _ = sample_onnx_model
        sidecar_path = model_path.with_suffix(".meta.json")
        sidecar_payload = {
            "output_schema": {"kind": "binary_proba", "shape": [None, 1]},
            "calibrator_kind": "platt",
            "calibrator_params": {"coef": 1.1, "intercept": -0.2},
        }
        sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        schema_hash = "hash_sidecar"
        feature_set_id = register_feature_set_for_schema(
            registry_path=facade.registry_path,
            schema_hash=schema_hash,
        )

        manifest = ModelManifest(
            model_id="sidecar_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"price": "float64"},
            feature_schema_hash=schema_hash,
            version="1.0.0",
            created_at=time.time(),
            last_modified=time.time(),
            serveable=True,
            artifact_format="onnx",
            feature_set_id=feature_set_id,
            output_schema=None,
            calibration=None,
        )

        facade.register_model(model_path, manifest)
        model_info = facade.get_model(manifest.model_id)

        assert model_info is not None
        assert model_info.manifest.output_schema == {"kind": "binary_proba", "shape": [None, 1]}
        assert model_info.manifest.calibration == {
            "kind": "platt",
            "params": {"coef": 1.1, "intercept": -0.2},
        }

    def test_register_model_auto_generates_id(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify model_id is auto-generated when not provided."""
        model_path, _ = sample_onnx_model
        schema_hash = "test_hash"
        feature_set_id = register_feature_set_for_schema(
            registry_path=facade.registry_path,
            schema_hash=schema_hash,
        )
        manifest = _build_strict_serveable_manifest(
            model_id="",  # Empty - should be auto-generated
            feature_schema={"price": "float64"},
            feature_schema_hash=schema_hash,
            feature_set_id=feature_set_id,
        )

        model_id = facade.register_model(model_path, manifest)

        assert model_id.startswith("model_")
        assert facade.get_model(model_id) is not None

    def test_get_model_returns_none_for_nonexistent(self, facade: ModelRegistryFacade) -> None:
        """Verify get_model returns None for non-existent model."""
        result = facade.get_model("nonexistent_model")
        assert result is None

    def test_get_all_models(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify get_all_models returns all registered models."""
        models = facade_with_registered_models.get_all_models()
        assert len(models) == 3

    def test_get_active_models(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify get_active_models returns only deployed models."""
        facade = facade_with_registered_models

        # Initially no active models
        active = facade.get_active_models()
        assert len(active) == 0

        # Deploy one model
        facade.deploy_model("model_0", "test_target")
        active = facade.get_active_models()
        assert len(active) == 1
        assert active[0].manifest.model_id == "model_0"

    def test_get_models_by_role(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify get_models_by_role filters correctly."""
        models = facade_with_registered_models.get_models_by_role(ModelRole.INFERENCE)
        assert len(models) == 3

        # No STUDENT models
        students = facade_with_registered_models.get_models_by_role(ModelRole.STUDENT)
        assert len(students) == 0

    def test_get_models_by_data_requirements(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify get_models_by_data_requirements filters correctly."""
        models = facade_with_registered_models.get_models_by_data_requirements(
            DataRequirements.L1_ONLY
        )
        assert len(models) == 3


# =============================================================================
# Deployment Tests
# =============================================================================


class TestDeploymentOperations:
    """Tests for deployment operations."""

    def test_deploy_model_success(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify successful model deployment."""
        facade = facade_with_registered_models

        success = facade.deploy_model("model_0", "ml_signal_actor")

        assert success is True
        model_info = facade.get_model("model_0")
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.ACTIVE
        assert "ml_signal_actor" in model_info.deployed_to

    def test_deploy_model_not_found(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        """Verify deploy returns False for non-existent model."""
        success = facade.deploy_model("nonexistent", "target")
        assert success is False

    def test_rollback_success(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify successful rollback."""
        facade = facade_with_registered_models
        facade.deploy_model("model_1", "target")

        success = facade.rollback("target", "model_0")

        assert success is True
        model_0 = facade.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.ACTIVE

    def test_retire_model_success(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify successful model retirement."""
        facade = facade_with_registered_models
        facade.deploy_model("model_0", "target")

        success = facade.retire_model("model_0")

        assert success is True
        model_info = facade.get_model("model_0")
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.RETIRED
        assert len(model_info.deployed_to) == 0

    def test_hot_reload_model_success(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify successful hot reload."""
        facade = facade_with_registered_models
        facade.deploy_model("model_0", "target")

        success = facade.hot_reload_model("target", "model_1")

        assert success is True
        model_0 = facade.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.RETIRED

        model_1 = facade.get_model("model_1")
        assert model_1 is not None
        assert model_1.deployment_status == DeploymentStatus.ACTIVE


# =============================================================================
# A/B Testing Tests
# =============================================================================


class TestABTestingOperations:
    """Tests for A/B testing operations."""

    def test_configure_ab_test_success(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify successful A/B test configuration."""
        facade = facade_with_registered_models

        config = facade.configure_ab_test(
            models=["model_0", "model_1"],
            split_ratio=0.5,
            duration_hours=24,
            target="test_target",
        )

        assert config is not None
        assert config["model_a"] == "model_0"
        assert config["model_b"] == "model_1"
        assert config["split_ratio"] == 0.5
        assert config["status"] == "active"

    def test_configure_ab_test_wrong_model_count(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify A/B test requires exactly 2 models."""
        facade = facade_with_registered_models

        config = facade.configure_ab_test(
            models=["model_0"],
            split_ratio=0.5,
            duration_hours=24,
            target="test_target",
        )

        assert config is None

    def test_compare_models(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify model comparison by metric."""
        facade = facade_with_registered_models

        # Add performance history
        facade.track_performance("model_0", {"accuracy": 0.85})
        facade.track_performance("model_1", {"accuracy": 0.90})
        facade.track_performance("model_2", {"accuracy": 0.88})

        result = facade.compare_models(
            model_ids=["model_0", "model_1", "model_2"],
            metric="accuracy",
        )

        assert result is not None
        assert result["best_model"] == "model_1"
        assert result["rankings"][0]["model_id"] == "model_1"

    def test_compare_models_statistically(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify statistical model comparison."""
        facade = facade_with_registered_models

        # Add multiple performance samples
        for _ in range(10):
            facade.track_performance("model_0", {"accuracy": 0.85 + 0.02 * (0.5 - 0.5)})
            facade.track_performance("model_1", {"accuracy": 0.90 + 0.02 * (0.5 - 0.5)})

        result = facade.compare_models_statistically(
            model_ids=["model_0", "model_1"],
            metric="accuracy",
        )

        assert result is not None
        assert "p_value_approx" in result
        assert "statistically_significant" in result
        assert "relative_improvement" in result


# =============================================================================
# Version Management Tests
# =============================================================================


class TestVersionManagement:
    """Tests for version management operations."""

    def test_list_compatible(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify list_compatible filters by schema hash."""
        compatible = facade_with_registered_models.list_compatible(
            schema_hash="test_hash_123",
        )
        assert len(compatible) == 3

    def test_list_compatible_with_filters(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify list_compatible with role and architecture filters."""
        compatible = facade_with_registered_models.list_compatible(
            schema_hash="test_hash_123",
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
        )
        assert len(compatible) == 3

    def test_resolve_latest(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify resolve_latest returns highest version."""
        latest = facade_with_registered_models.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
            schema_hash="test_hash_123",
        )

        assert latest is not None
        assert latest.manifest.version == "1.0.2"

    def test_get_model_lineage(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage tracking for parent-child relationships."""
        model_path, _ = sample_onnx_model

        # Register parent
        parent_manifest = ModelManifest(
            model_id="parent_model",
            role=ModelRole.TEACHER,
            data_requirements=DataRequirements.L1_L2,
            architecture="TFT",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash",
            serveable=False,
            artifact_format="none",
        )
        facade.register_model(model_path, parent_manifest)

        child_schema_hash = "test_hash"
        child_feature_set_id = register_feature_set_for_schema(
            registry_path=facade.registry_path,
            schema_hash=child_schema_hash,
        )
        # Register child with parent reference
        child_manifest = ModelManifest(
            model_id="child_model",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash=child_schema_hash,
            parent_id="parent_model",
            feature_set_id=child_feature_set_id,
            output_schema=default_output_schema(),
            calibration=default_calibration(),
        )
        facade.register_model(model_path, child_manifest)

        # Get lineage
        lineage = facade.get_model_lineage("child_model")

        assert len(lineage) == 2
        assert lineage[0].manifest.model_id == "parent_model"
        assert lineage[1].manifest.model_id == "child_model"


# =============================================================================
# Quality Validation Tests
# =============================================================================


class TestQualityValidation:
    """Tests for quality validation operations."""

    def test_validate_model_quality_passes(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify quality validation passes when metrics meet gates."""
        model_path, _ = sample_onnx_model
        schema_hash = "test_hash"
        feature_set_id = register_feature_set_for_schema(
            registry_path=facade.registry_path,
            schema_hash=schema_hash,
        )
        manifest = _build_strict_serveable_manifest(
            model_id="quality_model",
            feature_schema={"price": "float64"},
            feature_schema_hash=schema_hash,
            feature_set_id=feature_set_id,
            performance_metrics={"accuracy": 0.95, "precision": 0.92},
        )
        facade.register_model(model_path, manifest)

        gates = [
            QualityGate(metric_name="accuracy", threshold=0.90, comparison="gte"),
            QualityGate(metric_name="precision", threshold=0.90, comparison="gte"),
        ]

        result = facade.validate_model_quality("quality_model", gates)

        assert result.overall_pass is True
        assert result.gates_passed == 2
        assert result.gates_failed == 0

    def test_validate_model_quality_fails(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify quality validation fails when metrics don't meet gates."""
        model_path, _ = sample_onnx_model
        schema_hash = "test_hash"
        feature_set_id = register_feature_set_for_schema(
            registry_path=facade.registry_path,
            schema_hash=schema_hash,
        )
        manifest = _build_strict_serveable_manifest(
            model_id="low_quality_model",
            feature_schema={"price": "float64"},
            feature_schema_hash=schema_hash,
            feature_set_id=feature_set_id,
            performance_metrics={"accuracy": 0.80},
        )
        facade.register_model(model_path, manifest)

        gates = [
            QualityGate(metric_name="accuracy", threshold=0.90, comparison="gte", required=True),
        ]

        result = facade.validate_model_quality("low_quality_model", gates)

        assert result.overall_pass is False
        assert result.gates_failed == 1


# =============================================================================
# Canary Deployment Tests
# =============================================================================


class TestCanaryDeployment:
    """Tests for canary deployment operations."""

    def test_start_canary_deployment(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify canary deployment creation."""
        facade = facade_with_registered_models

        config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            min_samples=100,
        )

        deployment_id = facade.start_canary_deployment(
            model_id="model_0",
            target="test_target",
            config=config,
        )

        assert deployment_id.startswith("canary_")
        canary = facade.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.model_id == "model_0"
        assert canary.status == "active"

    def test_update_canary_metrics(
        self,
        facade_with_registered_models: ModelRegistryFacade,
    ) -> None:
        """Verify canary metric tracking."""
        facade = facade_with_registered_models

        config = CanaryConfig(traffic_percentage=5.0, min_samples=10)
        deployment_id = facade.start_canary_deployment(
            model_id="model_0",
            target="test",
            config=config,
        )

        facade.update_canary_metrics(deployment_id, metric_value=0.95, latency_ms=2.0)

        canary = facade.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.metrics["sample_count"] == 1


# =============================================================================
# All Public Methods Presence Test
# =============================================================================


class TestAllPublicMethodsPresent:
    """Test that all 35 public methods exist on facade."""

    EXPECTED_METHODS = [
        # Core methods
        "register_model",
        "get_model",
        "get_all_models",
        "get_active_models",
        "load_model",
        # Deployment methods
        "deploy_model",
        "rollback",
        "retire_model",
        "hot_reload_model",
        # Canary methods
        "start_canary_deployment",
        "get_canary_deployment",
        "update_canary_metrics",
        "evaluate_canary",
        "evaluate_canary_for_rollback",
        "auto_promote_canary",
        # A/B testing methods
        "configure_ab_test",
        "compare_models",
        "compare_models_statistically",
        "run_ab_test",
        "track_ab_test_metric",
        "analyze_ab_test",
        # Version methods
        "list_compatible",
        "resolve_latest",
        "get_model_lineage",
        # Gradual rollout methods
        "start_gradual_rollout",
        "get_rollout_status",
        "advance_rollout_stage",
        # Performance/metadata methods
        "track_performance",
        "update_metadata",
        "get_performance_history",
        # Quality methods
        "validate_model_quality",
        "get_artifact_path",
        # Filtering methods
        "get_models_by_role",
        "get_models_by_data_requirements",
        # Utility
        "flush",
    ]

    def test_all_public_methods_present(self, facade: ModelRegistryFacade) -> None:
        """Verify all 35 legacy public methods exist on facade."""
        missing_methods = []
        for method_name in self.EXPECTED_METHODS:
            if not hasattr(facade, method_name):
                missing_methods.append(method_name)
            elif not callable(getattr(facade, method_name)):
                missing_methods.append(f"{method_name} (not callable)")

        assert len(missing_methods) == 0, f"Missing methods: {missing_methods}"

    def test_method_count(self, facade: ModelRegistryFacade) -> None:
        """Verify we have at least 35 public methods."""
        public_methods = [
            name for name in dir(facade)
            if not name.startswith("_") and callable(getattr(facade, name))
        ]
        assert len(public_methods) >= 35, f"Found only {len(public_methods)} public methods"


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for persistence behavior."""

    def test_flush_writes_pending_changes(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """Verify flush writes pending batch saves."""
        model_path, _ = sample_onnx_model
        facade.register_model(model_path, sample_manifest)

        # Flush should complete without error
        facade.flush()

        # Registry should persist the model
        assert facade.registry_path.exists()

    def test_backend_property(self, facade: ModelRegistryFacade) -> None:
        """Verify backend property returns correct type."""
        assert facade.backend == BackendType.JSON


class TestInternalHelpers:
    """Tests for helper and backward-compatibility paths."""

    def test_backward_compatibility_properties_proxy_persistence(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        facade._persistence.ab_tests["ab_test_1"] = {"status": "active"}
        facade._persistence.deployments["target"] = ["model_x"]

        assert facade._ab_tests["ab_test_1"]["status"] == "active"
        assert facade._deployments["target"] == ["model_x"]
        assert facade._lock is facade._persistence._lock

    def test_load_model_returns_cached_model_and_updates_access_time(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        cached_model = object()
        facade._model_cache["cached"] = cached_model
        facade._cache_access_times["cached"] = 1.0

        with patch("ml.registry.model_registry_facade.time.time", return_value=42.0):
            loaded = facade.load_model("cached")

        assert loaded is cached_model
        assert facade._cache_access_times["cached"] == 42.0

    def test_validate_model_quality_returns_model_not_found_result(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        result = facade.validate_model_quality(
            "missing_model",
            [QualityGate(metric_name="accuracy", threshold=0.9, comparison="gte", required=True)],
        )
        assert result.overall_pass is False
        assert result.gate_results["model_existence"]["reason"] == "model_not_found"

    def test_internal_persistence_delegate_helpers(self, facade: ModelRegistryFacade) -> None:
        with patch.object(facade._persistence, "save_registry") as save_registry_mock:
            facade._save_registry(immediate=True)
        save_registry_mock.assert_called_once_with(immediate=True)

        with patch.object(facade._persistence, "calculate_file_sha256", return_value="digest"):
            digest = facade._calculate_file_sha256(Path("/tmp/model.onnx"))
        assert digest == "digest"

    def test_digest_normalization_and_gate_evaluation_helpers(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        assert facade._normalize_expected_digest(None) is None
        assert facade._normalize_expected_digest("   ") is None
        assert facade._normalize_expected_digest(" digest ") == "digest"

        gt_result = facade._evaluate_gate(
            QualityGate(metric_name="metric_gt", threshold=1.0, comparison="gt", required=True),
            2.0,
        )
        lt_result = facade._evaluate_gate(
            QualityGate(metric_name="metric_lt", threshold=1.0, comparison="lt", required=True),
            0.5,
        )
        eq_result = facade._evaluate_gate(
            QualityGate(metric_name="metric_eq", threshold=1.0, comparison="eq", required=True),
            1.0,
        )
        missing_result = facade._evaluate_gate(
            QualityGate(metric_name="metric_missing", threshold=1.0, comparison="gte", required=True),
            None,
        )

        assert gt_result["passed"] is True
        assert lt_result["passed"] is True
        assert eq_result["passed"] is True
        assert missing_result["reason"] == "metric_not_found"

    def test_apply_quality_gates_raises_when_enforced(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        manifest = ModelManifest(
            model_id="quality_gate_failure",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="hash",
            performance_metrics={"accuracy": 0.5},
        )

        with pytest.raises(ValueError, match="Quality gates not met for model quality_gate_failure"):
            facade._apply_quality_gates(
                manifest,
                [QualityGate(metric_name="accuracy", threshold=0.9, comparison="gte", required=True)],
                enforce_quality=True,
            )

    def test_maybe_auto_deploy_routes_and_skips_by_constraints(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        student_invalid = ModelManifest(
            model_id="student_invalid",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_L2,
            architecture="student",
            feature_schema={"f": "float32"},
            feature_schema_hash="hash",
            parent_id=None,
            performance_metrics={"inference_latency_ms": 999.0},
        )
        with patch.object(facade, "deploy_model") as deploy_model_mock:
            facade._maybe_auto_deploy(student_invalid)
            deploy_model_mock.assert_not_called()

        student_valid = ModelManifest(
            model_id="student_valid",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="student",
            feature_schema={"f": "float32"},
            feature_schema_hash="hash",
            parent_id="teacher_model",
            performance_metrics={"inference_latency_ms": 0.5},
        )
        with patch.object(facade, "deploy_model") as deploy_model_mock:
            facade._maybe_auto_deploy(student_valid)
            deploy_model_mock.assert_called_once_with("student_valid", "ml_signal_actor")

        inference_model = ModelManifest(
            model_id="inference_valid",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="hash",
        )
        with patch.object(facade, "deploy_model") as deploy_model_mock:
            facade._maybe_auto_deploy(inference_model)
            deploy_model_mock.assert_called_once_with("inference_valid", "ml_signal_actor")

    def test_get_artifact_path_handles_invalid_path_guard(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        model_path, _ = sample_onnx_model
        facade.register_model(model_path, sample_manifest)
        assert facade.get_artifact_path(sample_manifest.model_id) == model_path

        with patch.object(facade, "_validate_model_path", return_value=False):
            assert facade.get_artifact_path(sample_manifest.model_id) is None

    def test_delegate_wrappers_forward_to_deployment_and_ab_testing_components(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        with patch.object(
            facade._deployment,
            "evaluate_canary",
            return_value=(True, "promote"),
        ) as evaluate_canary_mock:
            assert facade.evaluate_canary("canary_1") == (True, "promote")
        evaluate_canary_mock.assert_called_once_with("canary_1")

        with patch.object(
            facade._deployment,
            "evaluate_canary_for_rollback",
            return_value=(False, "stable"),
        ) as rollback_mock:
            assert facade.evaluate_canary_for_rollback("canary_2") == (False, "stable")
        rollback_mock.assert_called_once_with("canary_2")

        with patch.object(
            facade._deployment,
            "auto_promote_canary",
            return_value=True,
        ) as auto_promote_mock:
            assert facade.auto_promote_canary("canary_3") is True
        auto_promote_mock.assert_called_once_with("canary_3")

        with patch.object(facade._ab_testing, "run_ab_test", return_value="ab_1") as run_ab_mock:
            assert facade.run_ab_test("model_a", "model_b", 0.5, 12.0, "target") == "ab_1"
        run_ab_mock.assert_called_once_with("model_a", "model_b", 0.5, 12.0, "target")

        with patch.object(facade._ab_testing, "track_ab_test_metric") as track_ab_metric_mock:
            facade.track_ab_test_metric("ab_1", "model_a", 0.92)
        track_ab_metric_mock.assert_called_once_with("ab_1", "model_a", 0.92)

        with patch.object(
            facade._ab_testing,
            "analyze_ab_test",
            return_value={"winner": "model_a"},
        ) as analyze_ab_mock:
            assert facade.analyze_ab_test("ab_1") == {"winner": "model_a"}
        analyze_ab_mock.assert_called_once_with("ab_1")

    def test_load_model_guard_paths_for_missing_invalid_missing_file_and_non_onnx(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        assert facade.load_model("missing_model") is None

        outside_manifest = ModelManifest(
            model_id="outside_path",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="outside_hash",
            serveable=False,
        )
        outside_model_info = ModelInfo(
            manifest=outside_manifest,
            model_path=Path("/tmp/outside.onnx"),
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("outside_path", outside_model_info)
        assert facade.load_model("outside_path") is None

        missing_path = facade.registry_path / "missing_file.onnx"
        missing_manifest = ModelManifest(
            model_id="missing_file",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="missing_hash",
            serveable=False,
        )
        missing_file_info = ModelInfo(
            manifest=missing_manifest,
            model_path=missing_path,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("missing_file", missing_file_info)
        assert facade.load_model("missing_file") is None

        non_onnx_path = facade.registry_path / "model.bin"
        non_onnx_path.write_bytes(b"binary")
        non_onnx_manifest = ModelManifest(
            model_id="non_onnx",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="non_onnx_hash",
            serveable=False,
        )
        non_onnx_info = ModelInfo(
            manifest=non_onnx_manifest,
            model_path=non_onnx_path,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("non_onnx", non_onnx_info)
        assert facade.load_model("non_onnx") is None

    def test_load_model_returns_none_when_onnx_runtime_unavailable(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        model_path = facade.registry_path / "runtime_missing.onnx"
        model_content = b"onnx"
        model_path.write_bytes(model_content)
        manifest = ModelManifest(
            model_id="runtime_missing_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="runtime_hash",
            serveable=False,
            artifact_sha256_digest=hashlib.sha256(model_content).hexdigest(),
        )
        model_info = ModelInfo(
            manifest=manifest,
            model_path=model_path,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("runtime_missing_model", model_info)

        with (
            patch("ml.registry.model_registry_facade.HAS_ONNX", False),
            patch("ml.registry.model_registry_facade.ort", None),
            patch("ml.registry.model_registry_facade.check_ml_dependencies") as deps_mock,
        ):
            assert facade.load_model("runtime_missing_model") is None
        deps_mock.assert_called_once_with(["onnxruntime"])

    def test_metadata_and_lookup_helpers_cover_not_found_and_exception_paths(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        assert facade.get_performance_history("missing_model") == []
        assert facade.get_artifact_path("missing_model") is None

        model_manifest = ModelManifest(
            model_id="metadata_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="metadata_hash",
        )
        model_info = ModelInfo(
            manifest=model_manifest,
            model_path=facade.registry_path / "metadata_model.onnx",
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("metadata_model", model_info)

        with patch.object(facade._persistence, "save_registry", side_effect=RuntimeError("save")):
            facade.update_metadata("metadata_model", {"owner": "ml-team"})
        assert model_info.metadata["owner"] == "ml-team"

    def test_internal_validation_and_cleanup_error_paths(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        with patch.object(facade, "flush", side_effect=RuntimeError("flush failed")):
            facade.__del__()

        with patch.object(Path, "resolve", side_effect=RuntimeError("resolve failed")):
            assert facade._validate_model_path(facade.registry_path / "bad.onnx") is False

        valid_manifest = ModelManifest(
            model_id="invalid_path_manifest",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="valid_hash",
            serveable=False,
        )
        with (
            patch.object(facade, "_validate_model_path", return_value=False),
            pytest.raises(ValueError, match="Security: Invalid model path"),
        ):
            facade._validate_registration_inputs(
                facade.registry_path / "invalid_path.onnx",
                valid_manifest,
            )

        missing_hash_manifest = ModelManifest(
            model_id="missing_hash_manifest",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="",
            serveable=False,
        )
        with (
            patch.object(facade, "_validate_model_path", return_value=True),
            pytest.raises(ValueError, match="feature_schema_hash is required"),
        ):
            facade._validate_registration_inputs(
                facade.registry_path / "missing_hash.onnx",
                missing_hash_manifest,
            )

    def test_hot_reload_and_quality_gate_helper_edge_paths(
        self,
        facade: ModelRegistryFacade,
    ) -> None:
        facade._enforce_hot_reload_compatibility(target="ml_signal_actor", new_model_id="missing")

        new_model_manifest = ModelManifest(
            model_id="new_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="schema_new",
        )
        new_model_info = ModelInfo(
            manifest=new_model_manifest,
            model_path=facade.registry_path / "new_model.onnx",
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        inactive_manifest = ModelManifest(
            model_id="inactive_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="schema_old",
        )
        inactive_info = ModelInfo(
            manifest=inactive_manifest,
            model_path=facade.registry_path / "inactive_model.onnx",
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=["ml_signal_actor"],
            performance_history=[],
            metadata={},
        )
        wrong_target_manifest = ModelManifest(
            model_id="wrong_target_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="schema_other",
        )
        wrong_target_info = ModelInfo(
            manifest=wrong_target_manifest,
            model_path=facade.registry_path / "wrong_target_model.onnx",
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["other_target"],
            performance_history=[],
            metadata={},
        )
        facade._persistence.set_model("new_model", new_model_info)
        facade._persistence.set_model("inactive_model", inactive_info)
        facade._persistence.set_model("wrong_target_model", wrong_target_info)
        facade._enforce_hot_reload_compatibility(target="ml_signal_actor", new_model_id="new_model")

        quality_result = facade._validate_quality_gates(
            "optional_gate_model",
            {"latency_ms": 0.5},
            [
                QualityGate(metric_name="latency_ms", threshold=1.0, comparison="lte", required=False),
                QualityGate(metric_name="missing_metric", threshold=0.1, comparison="gte", required=False),
            ],
        )
        assert quality_result.gates_failed == 1
        assert quality_result.overall_pass is True

        no_hash_manifest = ModelManifest(
            model_id="auto_deploy_no_hash",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="xgboost",
            feature_schema={"f": "float32"},
            feature_schema_hash="",
        )
        with patch.object(facade, "deploy_model") as deploy_model_mock:
            facade._maybe_auto_deploy(no_hash_manifest)
            deploy_model_mock.assert_not_called()
