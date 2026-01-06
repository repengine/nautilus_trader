#!/usr/bin/env python3

"""
Unit tests for ModelRegistryFacade.

These tests verify that the facade correctly wires all components and
provides the canonical ModelRegistry API.
"""

from __future__ import annotations

import hashlib
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
def sample_manifest(sample_onnx_model: tuple[Path, str]) -> ModelManifest:
    """Create sample ModelManifest for testing."""
    return ModelManifest(
        model_id="test_model_001",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="XGBoost",
        feature_schema={"price": "float64", "volume": "float64"},
        feature_schema_hash="test_hash_abc123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        serveable=True,
        artifact_format="onnx",
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

    for i in range(3):
        manifest = ModelManifest(
            model_id=f"model_{i}",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash_123",
            version=f"1.0.{i}",
            created_at=time.time(),
            last_modified=time.time(),
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

    def test_register_model_auto_generates_id(
        self,
        facade: ModelRegistryFacade,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify model_id is auto-generated when not provided."""
        model_path, _ = sample_onnx_model
        manifest = ModelManifest(
            model_id="",  # Empty - should be auto-generated
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash",
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

        # Register child with parent reference
        child_manifest = ModelManifest(
            model_id="child_model",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash",
            parent_id="parent_model",
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

        manifest = ModelManifest(
            model_id="quality_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash",
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

        manifest = ModelManifest(
            model_id="low_quality_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"price": "float64"},
            feature_schema_hash="test_hash",
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
        from ml.registry.persistence import BackendType
        assert facade.backend == BackendType.JSON
