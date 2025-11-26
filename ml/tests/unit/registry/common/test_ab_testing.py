#!/usr/bin/env python3

"""
Unit tests for ABTestingComponent.

This module tests the ABTestingComponent which handles A/B testing
and statistical model comparison, extracted from the ModelRegistry god class.

Test categories:
- Happy path tests: Standard successful operations
- Error condition tests: Validation failures and error handling
- Edge case tests: Boundary conditions and special scenarios

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.config.registry import RegistryPolicyConfig
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.common.ab_testing import ABTestingComponent
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def persistence_component(tmp_path: Path) -> ModelPersistenceComponent:
    """
    ModelPersistenceComponent with JSON backend for unit tests.

    Uses temporary directory to avoid side effects.
    """
    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path,
    )
    return ModelPersistenceComponent(config, tmp_path)


@pytest.fixture
def ab_testing_component(
    persistence_component: ModelPersistenceComponent,
) -> ABTestingComponent:
    """
    ABTestingComponent for A/B test and statistical comparison tests.
    """
    return ABTestingComponent(persistence_component)


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> tuple[Path, str]:
    """
    Create a sample ONNX model file and return (path, sha256_digest).

    This is needed for integrity verification tests.
    """
    import hashlib

    model_file = tmp_path / "test_model.onnx"
    content = b"sample ONNX model content for testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


@pytest.fixture
def registered_models(
    persistence_component: ModelPersistenceComponent,
    sample_onnx_model: tuple[Path, str],
) -> dict[str, ModelInfo]:
    """
    Pre-registered models for comparison tests.

    Creates 3 models with different versions and performance metrics.
    """
    models: dict[str, ModelInfo] = {}
    for i, version in enumerate(["1.0.0", "1.0.1", "1.1.0"]):
        manifest = ModelManifest(
            model_id=f"model_{i}",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"input": "float32"},
            feature_schema_hash="test_hash_123",
            version=version,
            created_at=time.time(),
            last_modified=time.time(),
        )
        model_info = ModelInfo(
            manifest=manifest,
            model_path=sample_onnx_model[0],
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            # Add different accuracy values for comparison tests
            performance_history=[{"accuracy": 0.9 + i * 0.01}],
        )
        models[manifest.model_id] = model_info
        persistence_component.set_model(manifest.model_id, model_info)

    persistence_component.save_registry(immediate=True)
    return models


@pytest.fixture
def models_with_history(
    persistence_component: ModelPersistenceComponent,
    sample_onnx_model: tuple[Path, str],
) -> dict[str, ModelInfo]:
    """
    Models with performance history for statistical comparison tests.

    Creates 2 models with multiple performance data points.
    """
    models: dict[str, ModelInfo] = {}

    # Model A: Control with accuracy around 0.85
    manifest_a = ModelManifest(
        model_id="model_a",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="test_arch",
        feature_schema={"input": "float32"},
        feature_schema_hash="test_hash_123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )
    model_a = ModelInfo(
        manifest=manifest_a,
        model_path=sample_onnx_model[0],
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[
            {"accuracy": 0.84},
            {"accuracy": 0.85},
            {"accuracy": 0.86},
            {"accuracy": 0.85},
            {"accuracy": 0.84},
        ],
    )
    models["model_a"] = model_a
    persistence_component.set_model("model_a", model_a)

    # Model B: Treatment with accuracy around 0.90
    manifest_b = ModelManifest(
        model_id="model_b",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="test_arch",
        feature_schema={"input": "float32"},
        feature_schema_hash="test_hash_123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )
    model_b = ModelInfo(
        manifest=manifest_b,
        model_path=sample_onnx_model[0],
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[
            {"accuracy": 0.89},
            {"accuracy": 0.90},
            {"accuracy": 0.91},
            {"accuracy": 0.90},
            {"accuracy": 0.89},
        ],
    )
    models["model_b"] = model_b
    persistence_component.set_model("model_b", model_b)

    persistence_component.save_registry(immediate=True)
    return models


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestConfigureABTest:
    """Tests for configure_ab_test method."""

    def test_configure_ab_test_success(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """
        Verify A/B test configuration is created and models set to TESTING.

        Test that:
        - Config contains both model IDs
        - Config has correct split_ratio
        - Both models have deployment_status == DeploymentStatus.TESTING
        """
        result = ab_testing_component.configure_ab_test(
            models=["model_0", "model_1"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )

        assert result is not None
        assert result["model_a"] == "model_0"
        assert result["model_b"] == "model_1"
        assert result["split_ratio"] == 0.5
        assert result["duration_hours"] == 24
        assert result["target"] == "production"
        assert result["status"] == "active"

        # Verify both models are now in TESTING status
        model_0 = ab_testing_component._persistence.get_model("model_0")
        model_1 = ab_testing_component._persistence.get_model("model_1")
        assert model_0 is not None
        assert model_1 is not None
        assert model_0.deployment_status == DeploymentStatus.TESTING
        assert model_1.deployment_status == DeploymentStatus.TESTING

    def test_configure_ab_test_preserves_ab_tests_dict(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify A/B test config is stored in persistence ab_tests dict."""
        ab_testing_component.configure_ab_test(
            models=["model_0", "model_1"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )

        # Check that a test was stored
        assert len(ab_testing_component._persistence.ab_tests) > 0


class TestCompareModels:
    """Tests for compare_models method."""

    def test_compare_models_basic(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """
        Verify basic model comparison returns rankings sorted by metric.

        Test that:
        - result["best_model"] is highest accuracy model
        - result["rankings"] sorted descending by accuracy
        """
        result = ab_testing_component.compare_models(
            model_ids=["model_0", "model_1", "model_2"],
            metric="accuracy",
        )

        assert result is not None
        assert result["metric"] == "accuracy"
        assert result["best_model"] == "model_2"  # 0.92 accuracy
        assert len(result["rankings"]) == 3

        # Verify sorting (descending)
        rankings = result["rankings"]
        accuracies = [r["accuracy"] for r in rankings]
        assert accuracies == sorted(accuracies, reverse=True)


class TestCompareModelsStatistically:
    """Tests for compare_models_statistically method."""

    def test_compare_models_statistically(
        self,
        ab_testing_component: ABTestingComponent,
        models_with_history: dict[str, ModelInfo],
    ) -> None:
        """
        Verify statistical comparison using Welch's t-test returns results.

        Test that result contains:
        - p_value_approx
        - statistically_significant boolean
        - relative_improvement
        """
        result = ab_testing_component.compare_models_statistically(
            model_ids=["model_a", "model_b"],
            metric="accuracy",
        )

        assert result is not None
        assert "p_value_approx" in result
        assert "statistically_significant" in result
        assert "relative_improvement" in result
        assert result["model_a"] == "model_a"
        assert result["model_b"] == "model_b"
        assert result["metric"] == "accuracy"


class TestRunABTest:
    """Tests for run_ab_test method."""

    def test_run_ab_test(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """
        Verify A/B test execution returns test ID and initializes metrics tracking.

        Test that:
        - test_id is non-empty string
        - _ab_test_metrics[test_id] initialized for both models
        """
        test_id = ab_testing_component.run_ab_test(
            model_a_id="model_0",
            model_b_id="model_1",
            split_ratio=0.5,
            duration_hours=24.0,
            target="production",
        )

        assert test_id != ""
        assert test_id in ab_testing_component._ab_test_metrics
        assert "model_0" in ab_testing_component._ab_test_metrics[test_id]
        assert "model_1" in ab_testing_component._ab_test_metrics[test_id]
        assert ab_testing_component._ab_test_metrics[test_id]["model_0"] == []
        assert ab_testing_component._ab_test_metrics[test_id]["model_1"] == []


class TestTrackABTestMetric:
    """Tests for track_ab_test_metric method."""

    def test_track_ab_test_metric(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """
        Verify metric tracking appends values to model's list.

        Test that:
        - _ab_test_metrics[test_id][model_id] contains the tracked values
        """
        test_id = ab_testing_component.run_ab_test(
            model_a_id="model_0",
            model_b_id="model_1",
            split_ratio=0.5,
            duration_hours=24.0,
            target="production",
        )

        # Track metrics
        ab_testing_component.track_ab_test_metric(test_id, "model_0", 0.85)
        ab_testing_component.track_ab_test_metric(test_id, "model_0", 0.86)
        ab_testing_component.track_ab_test_metric(test_id, "model_1", 0.90)

        assert 0.85 in ab_testing_component._ab_test_metrics[test_id]["model_0"]
        assert 0.86 in ab_testing_component._ab_test_metrics[test_id]["model_0"]
        assert 0.90 in ab_testing_component._ab_test_metrics[test_id]["model_1"]


class TestAnalyzeABTest:
    """Tests for analyze_ab_test method."""

    def test_analyze_ab_test(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """
        Verify A/B test analysis returns correct means and significance.

        Test that result contains:
        - control_mean calculated correctly
        - treatment_mean calculated correctly
        - statistical_significance boolean present
        """
        test_id = ab_testing_component.run_ab_test(
            model_a_id="model_0",
            model_b_id="model_1",
            split_ratio=0.5,
            duration_hours=24.0,
            target="production",
        )

        # Track enough metrics for analysis
        for val in [0.84, 0.85, 0.86, 0.85, 0.84]:
            ab_testing_component.track_ab_test_metric(test_id, "model_0", val)
        for val in [0.89, 0.90, 0.91, 0.90, 0.89]:
            ab_testing_component.track_ab_test_metric(test_id, "model_1", val)

        result = ab_testing_component.analyze_ab_test(test_id)

        assert result is not None
        assert result["test_id"] == test_id
        assert "control_mean" in result
        assert "treatment_mean" in result
        assert "statistical_significance" in result
        assert np.isclose(result["control_mean"], 0.848, atol=0.01)
        assert np.isclose(result["treatment_mean"], 0.898, atol=0.01)


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestConfigureABTestErrors:
    """Tests for configure_ab_test error conditions."""

    def test_configure_ab_test_wrong_model_count_one(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify rejection when only 1 model provided."""
        result = ab_testing_component.configure_ab_test(
            models=["model_0"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )
        assert result is None

    def test_configure_ab_test_wrong_model_count_three(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify rejection when 3 models provided."""
        result = ab_testing_component.configure_ab_test(
            models=["model_0", "model_1", "model_2"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )
        assert result is None

    def test_configure_ab_test_model_not_found(
        self,
        ab_testing_component: ABTestingComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify returns None when model not in registry."""
        result = ab_testing_component.configure_ab_test(
            models=["model_0", "nonexistent_model"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )
        assert result is None


class TestCompareModelsStatisticallyErrors:
    """Tests for compare_models_statistically error conditions."""

    def test_compare_models_statistically_no_samples(
        self,
        ab_testing_component: ABTestingComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify returns None when models have no performance history."""
        # Create models with empty performance history
        for model_id in ["empty_a", "empty_b"]:
            manifest = ModelManifest(
                model_id=model_id,
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="test_arch",
                feature_schema={"input": "float32"},
                feature_schema_hash="test_hash_123",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )
            model_info = ModelInfo(
                manifest=manifest,
                model_path=sample_onnx_model[0],
                deployment_status=DeploymentStatus.INACTIVE,
                deployed_to=[],
                performance_history=[],  # Empty!
            )
            ab_testing_component._persistence.set_model(model_id, model_info)

        result = ab_testing_component.compare_models_statistically(
            model_ids=["empty_a", "empty_b"],
            metric="accuracy",
        )
        assert result is None


class TestAnalyzeABTestErrors:
    """Tests for analyze_ab_test error conditions."""

    def test_analyze_ab_test_not_found(
        self,
        ab_testing_component: ABTestingComponent,
    ) -> None:
        """Verify returns None for non-existent test."""
        result = ab_testing_component.analyze_ab_test("nonexistent_test_id")
        assert result is None


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestCompareModelsEdgeCases:
    """Tests for compare_models edge cases."""

    def test_compare_models_missing_metric(
        self,
        ab_testing_component: ABTestingComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """
        Verify model without metric is excluded from rankings.

        When a model doesn't have the requested metric in its performance
        history, it should not appear in the rankings.
        """
        # Create model with no accuracy metric
        manifest = ModelManifest(
            model_id="model_no_accuracy",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"input": "float32"},
            feature_schema_hash="test_hash_123",
            version="1.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )
        model_info = ModelInfo(
            manifest=manifest,
            model_path=sample_onnx_model[0],
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[{"loss": 0.1}],  # No accuracy!
        )
        ab_testing_component._persistence.set_model("model_no_accuracy", model_info)

        # Create model with accuracy
        manifest2 = ModelManifest(
            model_id="model_with_accuracy",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"input": "float32"},
            feature_schema_hash="test_hash_123",
            version="1.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )
        model_info2 = ModelInfo(
            manifest=manifest2,
            model_path=sample_onnx_model[0],
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[{"accuracy": 0.9}],
        )
        ab_testing_component._persistence.set_model("model_with_accuracy", model_info2)

        result = ab_testing_component.compare_models(
            model_ids=["model_no_accuracy", "model_with_accuracy"],
            metric="accuracy",
        )

        assert result is not None
        # Only model_with_accuracy should be in rankings
        assert len(result["rankings"]) == 1
        assert result["rankings"][0]["model_id"] == "model_with_accuracy"

    def test_compare_models_empty_list(
        self,
        ab_testing_component: ABTestingComponent,
    ) -> None:
        """Verify returns None when model list is empty."""
        result = ab_testing_component.compare_models(
            model_ids=[],
            metric="accuracy",
        )
        assert result is None


class TestWelchTTestEdgeCases:
    """Tests for Welch t-test edge cases."""

    def test_welch_t_test_equal_means(
        self,
        ab_testing_component: ABTestingComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """
        Verify t-test with identical means is not statistically significant.

        When two models have the same mean performance, the test should
        indicate no significant difference.
        """
        # Create two models with identical performance distributions
        for model_id in ["equal_a", "equal_b"]:
            manifest = ModelManifest(
                model_id=model_id,
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="test_arch",
                feature_schema={"input": "float32"},
                feature_schema_hash="test_hash_123",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )
            model_info = ModelInfo(
                manifest=manifest,
                model_path=sample_onnx_model[0],
                deployment_status=DeploymentStatus.INACTIVE,
                deployed_to=[],
                # Same values for both models
                performance_history=[
                    {"accuracy": 0.85},
                    {"accuracy": 0.85},
                    {"accuracy": 0.85},
                    {"accuracy": 0.85},
                    {"accuracy": 0.85},
                ],
            )
            ab_testing_component._persistence.set_model(model_id, model_info)

        result = ab_testing_component.compare_models_statistically(
            model_ids=["equal_a", "equal_b"],
            metric="accuracy",
        )

        assert result is not None
        # With equal means, should not be statistically significant
        assert result["statistically_significant"] is False
        assert np.isclose(result["relative_improvement"], 0.0, atol=0.1)


class TestTrackABTestMetricEdgeCases:
    """Tests for track_ab_test_metric edge cases."""

    def test_track_ab_test_metric_invalid_test(
        self,
        ab_testing_component: ABTestingComponent,
    ) -> None:
        """
        Verify tracking metric for invalid test ID does not raise error.

        Should silently skip when test_id doesn't exist.
        """
        # This should not raise an exception
        ab_testing_component.track_ab_test_metric(
            "nonexistent_test",
            "model_0",
            0.85,
        )
        # No assertion needed - just verify no exception raised


class TestCustomPolicy:
    """Tests for custom policy configuration."""

    def test_configure_ab_test_custom_policy(
        self,
        persistence_component: ModelPersistenceComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify custom policy configuration is respected."""
        # Create component with custom policy requiring 3 models
        custom_policy = RegistryPolicyConfig(ab_models_required=3)
        component = ABTestingComponent(
            persistence_component,
            policy_config=custom_policy,
        )

        # 2 models should now fail
        result = component.configure_ab_test(
            models=["model_0", "model_1"],
            split_ratio=0.5,
            duration_hours=24,
            target="production",
        )
        assert result is None
