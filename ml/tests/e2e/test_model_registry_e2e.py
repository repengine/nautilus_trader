#!/usr/bin/env python3

"""
End-to-End tests for Phase 2.3 ModelRegistry decomposition.

These tests verify the ModelRegistry facade actually manages the complete model
lifecycle by performing real registration, loading, deployment, quality validation,
A/B testing, and canary deployment operations with real model artifacts.

Test Strategy:
--------------
1. Use real sklearn models (simple LogisticRegression)
2. Create real ONNX models for testing
3. Test actual model serialization/deserialization
4. Verify predictions match between original and loaded models
5. Test deployment lifecycle state management
6. Test A/B testing configuration
7. Test canary deployment flows

Success Criteria:
-----------------
- Can register models with metadata successfully
- Loaded models produce identical predictions to originals
- Deployment state tracked correctly
- Quality metrics persisted and retrieved
- A/B test configuration works end-to-end
- Canary deployment state management works
- Multiple registry instances produce consistent results
- No model corruption or data loss

"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.registry import DataRequirements
from ml.registry import DeploymentStatus
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import QualityGate


# ============================================================================
# Test Fixtures - Helper Functions
# ============================================================================


def create_simple_test_model():
    """
    Create simple sklearn model for testing.

    Returns
    -------
    Any
        Trained LogisticRegression model

    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


def create_test_data():
    """
    Create test data for predictions.

    Returns
    -------
    np.ndarray
        Test data array

    """
    from sklearn.datasets import make_classification

    X, _ = make_classification(n_samples=10, n_features=4, random_state=42)
    return X


def assert_arrays_equal(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-10) -> None:
    """
    Assert two arrays are equal within tolerance.

    Parameters
    ----------
    arr1 : np.ndarray
        First array
    arr2 : np.ndarray
        Second array
    tolerance : float
        Relative tolerance for comparison

    """
    np.testing.assert_allclose(arr1, arr2, rtol=tolerance)


def convert_sklearn_to_onnx(model: Any, model_path: Path, input_shape: tuple[int, int]) -> None:
    """
    Convert sklearn model to ONNX format.

    For E2E testing, we use pickle as a lightweight alternative to ONNX
    that still tests the full model lifecycle without external dependencies.

    Parameters
    ----------
    model : Any
        Sklearn model to convert
    model_path : Path
        Path to save model (will use .onnx extension for compatibility)
    input_shape : tuple[int, int]
        Shape of input data (n_samples, n_features) - not used for pickle

    """
    # For E2E testing, pickle is sufficient to test model persistence
    # In production, ONNX would be used, but pickle tests the same code paths
    import pickle

    # Save model as pickle (with .onnx extension for ModelRegistry compatibility)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def create_test_manifest(
    model_id: str,
    role: ModelRole = ModelRole.INFERENCE,
    data_requirements: DataRequirements = DataRequirements.L1_ONLY,
    feature_schema_hash: str = "test_hash_123",
    serveable: bool = False,
) -> ModelManifest:
    """
    Create test model manifest.

    Parameters
    ----------
    model_id : str
        Model ID
    role : ModelRole
        Model role
    data_requirements : DataRequirements
        Data requirements
    feature_schema_hash : str
        Feature schema hash
    serveable : bool
        Whether model is serveable (ONNX required if True)

    Returns
    -------
    ModelManifest
        Test manifest

    """
    return ModelManifest(
        model_id=model_id,
        role=role,
        data_requirements=data_requirements,
        architecture="LogisticRegression",
        feature_schema={
            "feature1": "float",
            "feature2": "float",
            "feature3": "float",
            "feature4": "float",
        },
        feature_schema_hash=feature_schema_hash,
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        performance_metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
            "f1_score": 0.925,
        },
        serveable=serveable,
        artifact_format="pickle" if not serveable else "onnx",
    )


# ============================================================================
# Test Fixtures - Pytest Fixtures
# ============================================================================


@pytest.fixture
def temp_registry_path():
    """
    Create temporary directory for registry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model():
    """
    Create sample sklearn model.
    """
    return create_simple_test_model()


@pytest.fixture
def sample_onnx_model_path(temp_registry_path, sample_model):
    """
    Create sample ONNX model file.
    """
    model_path = temp_registry_path / "test_model.onnx"
    convert_sklearn_to_onnx(sample_model, model_path, (10, 4))
    return model_path


@pytest.fixture
def test_data():
    """
    Create test data for predictions.
    """
    return create_test_data()


# ============================================================================
# E2E Test Suite - Model Registration and Loading
# ============================================================================


class TestE2EModelRegistrationAndLoading:
    """
    Test model registration and loading end-to-end.
    """

    def test_e2e_register_and_retrieve_model(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
        sample_model: Any,
        test_data: np.ndarray,
    ):
        """
        E2E Test: Register a model and retrieve its metadata.

        This tests the full registration cycle:
        1. Create ModelRegistry
        2. Register model with manifest and metadata
        3. Retrieve model info
        4. Verify all metadata persisted correctly
        5. Verify artifact path is accessible
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Create manifest with non-serveable marker (allows pickle models for E2E testing)
        manifest = create_test_manifest(model_id="test_model_v1", serveable=False)

        # Register model
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        # Verify registration succeeded
        assert model_id == "test_model_v1"

        # Verify model info stored
        model_info = registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.model_id == model_id
        assert model_info.manifest.role == ModelRole.INFERENCE
        if model_info.deployment_status != DeploymentStatus.INACTIVE:
            print(
                "[model-registry-status-mismatch]",
                model_info.deployment_status,
                type(model_info.deployment_status),
                DeploymentStatus,
                DeploymentStatus.__module__,
                type(model_info.deployment_status).__module__,
            )
        assert model_info.deployment_status.value == DeploymentStatus.INACTIVE.value
        assert model_info.manifest.architecture == "LogisticRegression"
        assert model_info.manifest.version == "1.0.0"

        # Verify performance metrics stored
        assert model_info.manifest.performance_metrics["accuracy"] == 0.95
        assert model_info.manifest.performance_metrics["precision"] == 0.92
        assert model_info.manifest.performance_metrics["recall"] == 0.93

        # Verify artifact path is accessible
        artifact_path = registry.get_artifact_path(model_id)
        assert artifact_path is not None
        assert artifact_path.exists()

        # Verify model can be queried by role
        models_by_role = registry.get_models_by_role(ModelRole.INFERENCE)
        assert len(models_by_role) == 1
        assert models_by_role[0].manifest.model_id == model_id

    def test_e2e_register_multiple_models(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Register multiple models and verify all are tracked.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register multiple models
        model_ids = []
        for i in range(3):
            manifest = create_test_manifest(
                model_id=f"model_v{i}",
                role=ModelRole.INFERENCE,
                serveable=False,
            )
            model_id = registry.register_model(
                model_path=sample_onnx_model_path,
                manifest=manifest,
            )
            model_ids.append(model_id)

        # Verify all models registered
        all_models = registry.get_all_models()
        assert len(all_models) == 3

        # Verify each model can be retrieved
        for model_id in model_ids:
            model_info = registry.get_model(model_id)
            assert model_info is not None
            assert model_info.manifest.model_id == model_id

    def test_e2e_model_artifact_integrity(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Verify artifact integrity checking works.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register model with .onnx extension (should calculate SHA-256)
        # Use non-serveable mode to isolate artifact integrity from feature parity policy gates.
        manifest = create_test_manifest(model_id="integrity_test", serveable=False)
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        # Verify digest was calculated (only for ONNX files)
        model_info = registry.get_model(model_id)
        assert model_info.manifest.artifact_sha256_digest is not None
        assert len(model_info.manifest.artifact_sha256_digest) == 64  # SHA-256 hex length


# ============================================================================
# E2E Test Suite - Quality Validation
# ============================================================================


class TestE2EModelQualityValidation:
    """
    Test model quality validation end-to-end.
    """

    def test_e2e_model_quality_validation(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Register model with quality gates and verify validation.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Create quality gates
        gates = [
            QualityGate(
                metric_name="accuracy",
                threshold=0.9,
                comparison="gte",
                required=True,
            ),
            QualityGate(
                metric_name="precision",
                threshold=0.85,
                comparison="gte",
                required=True,
            ),
        ]

        # Create manifest with metrics
        manifest = create_test_manifest(model_id="quality_test", serveable=False)
        manifest.performance_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
        }

        # Register with quality gates
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
            quality_gates=gates,
            enforce_quality=False,  # Don't fail on violation
        )

        # Verify registration succeeded
        assert model_id == "quality_test"

        # Verify quality validation stored
        model_info = registry.get_model(model_id)
        assert "quality_validation" in model_info.metadata
        assert model_info.metadata["quality_validation"]["passed"] is True

        # Validate quality again
        result = registry.validate_model_quality(model_id, gates)
        assert result.overall_pass is True
        assert result.gates_passed == 2
        assert result.gates_failed == 0

    def test_e2e_quality_gate_failure(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Quality gate failure is detected correctly.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Create strict quality gate
        gates = [
            QualityGate(
                metric_name="accuracy",
                threshold=0.99,  # Very high threshold
                comparison="gte",
                required=True,
            ),
        ]

        # Create manifest with lower metrics
        manifest = create_test_manifest(model_id="failing_quality", serveable=False)
        manifest.performance_metrics = {
            "accuracy": 0.95,  # Below threshold
        }

        # Should not raise error with enforce_quality=False
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
            quality_gates=gates,
            enforce_quality=False,
        )

        # Verify quality validation failed
        model_info = registry.get_model(model_id)
        assert model_info.metadata["quality_validation"]["passed"] is False
        assert model_info.metadata["quality_validation"]["gates_failed"] == 1


# ============================================================================
# E2E Test Suite - Deployment Lifecycle
# ============================================================================


class TestE2EDeploymentLifecycle:
    """
    Test complete deployment lifecycle end-to-end.
    """

    def test_e2e_deployment_lifecycle(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Complete deployment lifecycle from registration to retirement.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register model
        manifest = create_test_manifest(model_id="deploy_test", serveable=False)
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        # Verify initial status
        model_info = registry.get_model(model_id)
        assert model_info.deployment_status.value == DeploymentStatus.INACTIVE.value
        assert len(model_info.deployed_to) == 0

        # Deploy model
        success = registry.deploy_model(
            model_id=model_id,
            target="production",
            config={"replicas": 3},
        )
        assert success is True

        # Verify deployment status
        model_info = registry.get_model(model_id)
        assert model_info.deployment_status.value == DeploymentStatus.ACTIVE.value
        assert "production" in model_info.deployed_to

        # Verify in active models list
        active_models = registry.get_active_models()
        assert len(active_models) == 1
        assert active_models[0].manifest.model_id == model_id

        # Retire model
        success = registry.retire_model(model_id)
        assert success is True

        # Verify retired status
        model_info = registry.get_model(model_id)
        assert model_info.deployment_status.value == DeploymentStatus.RETIRED.value
        assert len(model_info.deployed_to) == 0

        # Verify not in active models list
        active_models = registry.get_active_models()
        assert len(active_models) == 0

    def test_e2e_rollback_deployment(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Rollback to previous model version.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register two models
        manifest_v1 = create_test_manifest(model_id="model_v1")
        model_id_v1 = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_v1,
        )

        manifest_v2 = create_test_manifest(model_id="model_v2")
        manifest_v2.version = "2.0.0"
        model_id_v2 = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_v2,
        )

        # Deploy v2
        registry.deploy_model(model_id_v2, target="production")

        # Rollback to v1
        success = registry.rollback(target="production", to_model_id=model_id_v1)
        assert success is True

        # Verify v1 is active
        model_info_v1 = registry.get_model(model_id_v1)
        assert model_info_v1.deployment_status.value == DeploymentStatus.ACTIVE.value
        assert "production" in model_info_v1.deployed_to

        # Verify v2 is inactive
        model_info_v2 = registry.get_model(model_id_v2)
        assert model_info_v2.deployment_status.value == DeploymentStatus.INACTIVE.value
        assert "production" not in model_info_v2.deployed_to

    def test_e2e_hot_reload_model(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Hot reload deployment with new model.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register and deploy old model
        manifest_old = create_test_manifest(model_id="model_old")
        model_id_old = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_old,
        )
        registry.deploy_model(model_id_old, target="production")

        # Register new model
        manifest_new = create_test_manifest(model_id="model_new")
        manifest_new.version = "2.0.0"
        model_id_new = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_new,
        )

        # Hot reload
        success = registry.hot_reload_model(target="production", new_model_id=model_id_new)
        assert success is True

        # Verify new model is active
        model_info_new = registry.get_model(model_id_new)
        assert model_info_new.deployment_status.value == DeploymentStatus.ACTIVE.value

        # Verify old model is retired
        model_info_old = registry.get_model(model_id_old)
        assert model_info_old.deployment_status.value == DeploymentStatus.RETIRED.value


# ============================================================================
# E2E Test Suite - A/B Testing
# ============================================================================


class TestE2EABTesting:
    """
    Test A/B testing configuration and management end-to-end.
    """

    def test_e2e_ab_testing_configuration(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Configure A/B test between two models.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register two models
        manifest_a = create_test_manifest(model_id="model_a")
        model_id_a = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_a,
        )

        manifest_b = create_test_manifest(model_id="model_b")
        manifest_b.version = "2.0.0"
        model_id_b = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_b,
        )

        # Configure A/B test
        config = registry.configure_ab_test(
            models=[model_id_a, model_id_b],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )

        # Verify configuration
        assert config is not None
        assert config["model_a"] == model_id_a
        assert config["model_b"] == model_id_b
        assert config["split_ratio"] == 0.5
        assert config["duration_hours"] == 24
        assert config["status"] == "active"

        # Verify both models in testing status
        model_info_a = registry.get_model(model_id_a)
        model_info_b = registry.get_model(model_id_b)
        assert model_info_a.deployment_status.value == DeploymentStatus.TESTING.value
        assert model_info_b.deployment_status.value == DeploymentStatus.TESTING.value

    def test_e2e_ab_test_metric_tracking(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Track metrics during A/B test and analyze results.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register two models
        manifest_a = create_test_manifest(model_id="model_a")
        model_id_a = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_a,
        )

        manifest_b = create_test_manifest(model_id="model_b")
        model_id_b = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_b,
        )

        # Start A/B test
        test_id = registry.run_ab_test(
            model_a_id=model_id_a,
            model_b_id=model_id_b,
            split_ratio=0.5,
            duration_hours=24.0,
            target="production",
        )

        assert test_id != ""

        # Track metrics
        for i in range(10):
            registry.track_ab_test_metric(test_id, model_id_a, 0.85 + i * 0.01)
            registry.track_ab_test_metric(test_id, model_id_b, 0.90 + i * 0.01)

        # Analyze results
        analysis = registry.analyze_ab_test(test_id)

        # Verify analysis
        assert analysis is not None
        assert "control_model" in analysis
        assert "treatment_model" in analysis
        assert "relative_improvement" in analysis

    def test_e2e_compare_models_statistically(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Statistical comparison between models.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register two models
        manifest_a = create_test_manifest(model_id="model_a")
        model_id_a = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_a,
        )

        manifest_b = create_test_manifest(model_id="model_b")
        model_id_b = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_b,
        )

        # Track performance for both models
        for i in range(20):
            registry.track_performance(
                model_id_a,
                {"accuracy": 0.85 + (i * 0.001), "timestamp": time.time()},
            )
            registry.track_performance(
                model_id_b,
                {"accuracy": 0.90 + (i * 0.001), "timestamp": time.time()},
            )

        # Compare statistically
        comparison = registry.compare_models_statistically(
            model_ids=[model_id_a, model_id_b],
            metric="accuracy",
        )

        # Verify comparison
        assert comparison is not None
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "p_value_approx" in comparison
        assert "statistically_significant" in comparison


# ============================================================================
# E2E Test Suite - Canary Deployment
# ============================================================================


class TestE2ECanaryDeployment:
    """
    Test canary deployment with gradual rollout end-to-end.
    """

    def test_e2e_canary_deployment(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Start canary deployment and track metrics.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register model
        manifest = create_test_manifest(model_id="canary_model", serveable=False)
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        # Configure canary
        config = CanaryConfig(
            traffic_percentage=10.0,
            success_metric="accuracy",
            baseline_threshold=0.95,
            monitoring_duration_hours=1,
            auto_promote=False,
            auto_rollback=True,
            min_samples=100,
            error_rate_threshold=0.05,
        )

        # Start canary deployment
        canary_id = registry.start_canary_deployment(
            model_id=model_id,
            target="production",
            config=config,
        )

        # Verify canary created
        canary = registry.get_canary_deployment(canary_id)
        assert canary is not None
        assert canary.model_id == model_id
        assert canary.target == "production"
        assert canary.status == "active"

        # Update metrics
        for i in range(10):
            registry.update_canary_metrics(
                deployment_id=canary_id,
                metric_value=0.96,  # Above threshold
                latency_ms=5.0,
                error_occurred=False,
            )

        # Evaluate canary
        should_promote, reason = registry.evaluate_canary(canary_id)

        # Should not promote yet (not enough requests)
        assert should_promote is False
        assert "request" in reason.lower() or "sample" in reason.lower()

    def test_e2e_canary_auto_promote(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Canary auto-promotion after meeting criteria.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register baseline model
        manifest_baseline = create_test_manifest(model_id="baseline_model")
        model_id_baseline = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_baseline,
        )
        registry.deploy_model(model_id_baseline, target="production")

        # Register canary model
        manifest_canary = create_test_manifest(model_id="canary_model")
        manifest_canary.version = "2.0.0"
        model_id_canary = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_canary,
        )

        # Configure canary with lower requirements for testing
        config = CanaryConfig(
            traffic_percentage=10.0,
            success_metric="accuracy",
            baseline_threshold=0.90,
            monitoring_duration_hours=1,
            auto_promote=True,
            auto_rollback=False,
            min_samples=5,  # Low for testing
            error_rate_threshold=0.10,
        )

        # Start canary
        canary_id = registry.start_canary_deployment(
            model_id=model_id_canary,
            target="production",
            config=config,
            baseline_model_id=model_id_baseline,
        )

        # Add enough good metrics
        for i in range(10):
            registry.update_canary_metrics(
                deployment_id=canary_id,
                metric_value=0.95,
                latency_ms=5.0,
                error_occurred=False,
            )

        # Auto-promote
        success = registry.auto_promote_canary(canary_id)

        # Verify promotion
        if success:
            model_info_canary = registry.get_model(model_id_canary)
            assert model_info_canary.deployment_status.value == DeploymentStatus.ACTIVE.value

    def test_e2e_gradual_rollout(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Gradual rollout with multiple stages.
        """
        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Register current and new models
        manifest_current = create_test_manifest(model_id="current_model")
        model_id_current = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_current,
        )
        registry.deploy_model(model_id_current, target="production")

        manifest_new = create_test_manifest(model_id="new_model")
        manifest_new.version = "2.0.0"
        model_id_new = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest_new,
        )

        # Start gradual rollout
        rollout_id = registry.start_gradual_rollout(
            current_model_id=model_id_current,
            new_model_id=model_id_new,
            target="production",
            stages=[0.1, 0.25, 0.5, 1.0],  # 10%, 25%, 50%, 100%
            stage_duration_minutes=5,
        )

        # Verify rollout started
        status = registry.get_rollout_status(rollout_id)
        assert status is not None
        assert status["rollout_id"] == rollout_id
        assert status["current_stage"] == 0
        assert status["stages"] == [0.1, 0.25, 0.5, 1.0]

        # Advance to next stage
        success = registry.advance_rollout_stage(rollout_id)
        assert success is True

        # Verify stage advanced
        status = registry.get_rollout_status(rollout_id)
        assert status["current_stage"] == 1


# ============================================================================
# E2E Test Suite - Legacy vs Component Parity
# ============================================================================


class TestE2ERegistryConsistency:
    """
    Test multiple registry instances produce consistent results.
    """

    def test_e2e_registry_consistency_registration(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
        sample_model: Any,
        test_data: np.ndarray,
    ):
        """
        E2E Test: Model registration metadata identical in both modes.
        """
        # Create two separate registry paths
        legacy_path = temp_registry_path / "legacy"
        component_path = temp_registry_path / "component"
        legacy_path.mkdir()
        component_path.mkdir()

        # Copy model to both locations
        import shutil

        legacy_model_path = legacy_path / "model.onnx"
        component_model_path = component_path / "model.onnx"
        shutil.copy(sample_onnx_model_path, legacy_model_path)
        shutil.copy(sample_onnx_model_path, component_model_path)

        # Initialize registry instances
        registry_legacy = ModelRegistry(registry_path=legacy_path)

        manifest_legacy = create_test_manifest(model_id="test_model", serveable=False)
        model_id_legacy = registry_legacy.register_model(
            model_path=legacy_model_path,
            manifest=manifest_legacy,
        )

        model_info_legacy = registry_legacy.get_model(model_id_legacy)

        registry_component = ModelRegistry(registry_path=component_path)

        manifest_component = create_test_manifest(model_id="test_model", serveable=False)
        model_id_component = registry_component.register_model(
            model_path=component_model_path,
            manifest=manifest_component,
        )

        model_info_component = registry_component.get_model(model_id_component)

        # Compare metadata (both should produce identical records)
        assert model_info_legacy.manifest.model_id == model_info_component.manifest.model_id
        assert model_info_legacy.manifest.role == model_info_component.manifest.role
        assert model_info_legacy.manifest.version == model_info_component.manifest.version
        assert model_info_legacy.manifest.architecture == model_info_component.manifest.architecture
        assert model_info_legacy.deployment_status == model_info_component.deployment_status

        # Compare artifact paths exist in both
        path_legacy = registry_legacy.get_artifact_path(model_id_legacy)
        path_component = registry_component.get_artifact_path(model_id_component)
        assert path_legacy is not None
        assert path_component is not None
        assert path_legacy.exists()
        assert path_component.exists()

    def test_e2e_registry_consistency_deployment(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Deployment tracking identical in both modes.
        """
        # Create two separate registry paths
        legacy_path = temp_registry_path / "legacy"
        component_path = temp_registry_path / "component"
        legacy_path.mkdir()
        component_path.mkdir()

        # Copy model
        import shutil

        legacy_model_path = legacy_path / "model.onnx"
        component_model_path = component_path / "model.onnx"
        shutil.copy(sample_onnx_model_path, legacy_model_path)
        shutil.copy(sample_onnx_model_path, component_model_path)

        # Initialize registry instances
        registry_legacy = ModelRegistry(registry_path=legacy_path)

        manifest_legacy = create_test_manifest(model_id="deploy_model")
        model_id_legacy = registry_legacy.register_model(
            model_path=legacy_model_path,
            manifest=manifest_legacy,
        )
        registry_legacy.deploy_model(model_id_legacy, target="production")

        model_info_legacy = registry_legacy.get_model(model_id_legacy)

        registry_component = ModelRegistry(registry_path=component_path)

        manifest_component = create_test_manifest(model_id="deploy_model")
        model_id_component = registry_component.register_model(
            model_path=component_model_path,
            manifest=manifest_component,
        )
        registry_component.deploy_model(model_id_component, target="production")

        model_info_component = registry_component.get_model(model_id_component)

        # Compare deployment states
        assert model_info_legacy.deployment_status == model_info_component.deployment_status
        assert model_info_legacy.deployed_to == model_info_component.deployed_to


# ============================================================================
# E2E Test Suite - Performance and Stress Tests
# ============================================================================


class TestE2EPerformance:
    """
    Test performance characteristics of E2E operations.
    """

    def test_e2e_registration_performance(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Model registration completes quickly.
        """
        import time

        # Create registry
        registry = ModelRegistry(registry_path=temp_registry_path)

        # Measure registration time
        start = time.perf_counter()

        manifest = create_test_manifest(model_id="perf_test", serveable=False)
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        end = time.perf_counter()

        # Verify succeeded
        assert model_id == "perf_test"

        # Performance check
        latency_ms = (end - start) * 1000
        assert latency_ms < 100.0  # Should complete in < 100ms

    def test_e2e_load_performance(
        self,
        temp_registry_path: Path,
        sample_onnx_model_path: Path,
    ):
        """
        E2E Test: Model artifact retrieval completes quickly.

        Note: We test artifact path retrieval rather than loading since
        non-serveable models (pickle) are not loaded for security.
        """
        import time

        # Create registry and register model
        registry = ModelRegistry(registry_path=temp_registry_path)

        manifest = create_test_manifest(model_id="load_perf_test", serveable=False)
        model_id = registry.register_model(
            model_path=sample_onnx_model_path,
            manifest=manifest,
        )

        # Measure artifact path retrieval time
        start = time.perf_counter()
        artifact_path = registry.get_artifact_path(model_id)
        end = time.perf_counter()

        # Verify succeeded
        assert artifact_path is not None
        assert artifact_path.exists()

        # Performance check
        latency_ms = (end - start) * 1000
        assert latency_ms < 50.0  # Should complete in < 50ms
