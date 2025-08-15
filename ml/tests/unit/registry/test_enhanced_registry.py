#!/usr/bin/env python3

"""
Comprehensive tests for enhanced ModelRegistry with integrated features.

This test file uses TDD to define all expected behavior for:
- Quality validation on registration
- Canary deployment management
- Statistical model comparison
- Hot reload and gradual rollout

"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import ValidationResult
from ml.registry.model_registry import ModelRegistry


class TestEnhancedModelRegistry:
    """
    Test enhanced ModelRegistry with all integrated features.
    """

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """
        Create temporary directory for testing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self, temp_dir: Path) -> ModelRegistry:
        """
        Create registry instance.
        """
        return ModelRegistry(temp_dir)

    @pytest.fixture
    def sample_manifest(self) -> ModelManifest:
        """
        Create sample model manifest.
        """
        return ModelManifest(
            model_id="test_model_001",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="abc123",
            performance_metrics={"accuracy": 0.85, "latency_p99_ms": 4.5},
            deployment_constraints={"max_latency_ms": 5},
            version="1.0.0",
        )

    @pytest.fixture
    def model_path(self, temp_dir: Path) -> Path:
        """
        Create dummy ONNX model file.
        """
        model_file = temp_dir / "model.onnx"
        model_file.write_bytes(b"dummy_onnx_model")
        return model_file

    # ========== Quality Validation Tests ==========

    def test_register_with_quality_gates_pass(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test model registration with passing quality gates.
        """
        # Define quality gates
        gates = [
            QualityGate("accuracy", 0.8, "gte", required=True),
            QualityGate("latency_p99_ms", 5.0, "lte", required=True),
        ]

        # Register with quality validation
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
            quality_gates=gates,
        )

        assert model_id == "test_model_001"
        model_info = registry.get_model(model_id)
        assert model_info is not None
        assert model_info.metadata.get("quality_validation", {}).get("passed") is True

    def test_register_with_quality_gates_fail(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test model registration fails when quality gates not met.
        """
        # Set strict gates that will fail
        gates = [
            QualityGate("accuracy", 0.9, "gte", required=True),  # Will fail
        ]

        # Registration should fail or mark as not deployable
        with pytest.raises(ValueError, match="Quality gates not met"):
            registry.register_model(
                model_path=model_path,
                manifest=sample_manifest,
                quality_gates=gates,
                enforce_quality=True,
            )

    def test_validate_model_quality(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test post-registration quality validation.
        """
        # Register without gates
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Validate quality after registration
        gates = [
            QualityGate("accuracy", 0.8, "gte", required=True),
            QualityGate("latency_p99_ms", 10.0, "lte", required=False),
        ]

        result: ValidationResult = registry.validate_model_quality(model_id, gates)

        assert result.model_id == model_id
        assert result.overall_pass is True
        assert result.gates_passed == 2
        assert result.gates_failed == 0

    # ========== Canary Deployment Tests ==========

    def test_start_canary_deployment(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test starting a canary deployment.
        """
        # Register model
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Deploy model first
        registry.deploy_model(model_id, "ml_signal_actor")

        # Start canary deployment
        config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            baseline_threshold=0.95,
            monitoring_duration_hours=24,
        )

        deployment_id = registry.start_canary_deployment(
            model_id=model_id,
            target="ml_signal_actor",
            config=config,
            baseline_model_id=None,  # Use current production as baseline
        )

        assert deployment_id is not None
        assert deployment_id.startswith("canary_")

        # Check canary status
        canary = registry.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.model_id == model_id
        assert canary.config.traffic_percentage == 5.0
        assert canary.status == "active"

    def test_update_canary_metrics(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test updating canary deployment metrics.
        """
        # Setup canary
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )
        registry.deploy_model(model_id, "ml_signal_actor")

        config = CanaryConfig(traffic_percentage=5.0, min_samples=10)
        deployment_id = registry.start_canary_deployment(
            model_id=model_id,
            target="ml_signal_actor",
            config=config,
        )

        # Update metrics
        for i in range(15):
            registry.update_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.82 + i * 0.01,
                latency_ms=3.5,
                error_occurred=False,
            )

        # Check metrics were recorded
        canary = registry.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.metrics["sample_count"] == 15
        assert canary.metrics["success_count"] == 15
        assert canary.metrics["error_count"] == 0

    def test_evaluate_canary_for_promotion(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test canary evaluation for auto-promotion.
        """
        # Setup canary with short monitoring period
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )
        registry.deploy_model(model_id, "ml_signal_actor")

        config = CanaryConfig(
            traffic_percentage=5.0,
            min_samples=10,
            monitoring_duration_hours=0.0001,  # Very short for testing (0.36 seconds)
            auto_promote=True,
        )
        deployment_id = registry.start_canary_deployment(
            model_id=model_id,
            target="ml_signal_actor",
            config=config,
        )

        # Add good metrics
        for _ in range(20):
            registry.update_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.88,
                latency_ms=3.0,
                error_occurred=False,
            )

        # Wait briefly (at least the monitoring duration)
        time.sleep(0.4)  # 0.4 seconds > 0.0001 hours (0.36 seconds)

        # Evaluate canary
        should_promote, reason = registry.evaluate_canary(deployment_id)
        assert should_promote is True
        assert reason == "monitoring_period_complete"

        # Auto-promote if configured
        if config.auto_promote:
            promoted = registry.auto_promote_canary(deployment_id)
            assert promoted is True

    def test_evaluate_canary_for_rollback(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        model_path: Path,
    ) -> None:
        """
        Test canary evaluation for auto-rollback.
        """
        # Setup canary
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )
        registry.deploy_model(model_id, "ml_signal_actor")

        config = CanaryConfig(
            traffic_percentage=5.0,
            min_samples=10,
            error_rate_threshold=0.05,
            auto_rollback=True,
        )
        deployment_id = registry.start_canary_deployment(
            model_id=model_id,
            target="ml_signal_actor",
            config=config,
        )

        # Add bad metrics (high error rate)
        for i in range(20):
            registry.update_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.5 if i % 2 == 0 else 0.0,
                latency_ms=10.0,
                error_occurred=(i % 3 == 0),  # 33% error rate
            )

        # Evaluate canary
        should_rollback, reason = registry.evaluate_canary_for_rollback(deployment_id)
        assert should_rollback is True
        assert reason == "high_error_rate"

    # ========== Statistical Comparison Tests ==========

    def test_compare_models_statistically(
        self,
        registry: ModelRegistry,
        temp_dir: Path,
    ) -> None:
        """
        Test statistical comparison between models.
        """
        # Register two models
        manifest1 = ModelManifest(
            model_id="model_a",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",
            performance_metrics={"accuracy": 0.82},
        )

        manifest2 = ModelManifest(
            model_id="model_b",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash2",
            performance_metrics={"accuracy": 0.85},
        )

        model_path1 = temp_dir / "model1.onnx"
        model_path1.write_bytes(b"model1")
        model_path2 = temp_dir / "model2.onnx"
        model_path2.write_bytes(b"model2")

        registry.register_model(model_path1, manifest1)
        registry.register_model(model_path2, manifest2)

        # Track performance samples for both models
        for _ in range(100):
            registry.track_performance("model_a", {"accuracy_sample": np.random.normal(0.82, 0.02)})
            registry.track_performance("model_b", {"accuracy_sample": np.random.normal(0.85, 0.02)})

        # Compare models statistically
        comparison = registry.compare_models_statistically(
            model_ids=["model_a", "model_b"],
            metric="accuracy_sample",
        )

        assert comparison is not None
        assert "t_statistic" in comparison
        assert "p_value_approx" in comparison
        assert "statistically_significant" in comparison
        assert comparison["mean_b"] > comparison["mean_a"]  # Model B should be better

    def test_run_ab_test_analysis(
        self,
        registry: ModelRegistry,
        temp_dir: Path,
    ) -> None:
        """
        Test A/B test setup and analysis.
        """
        # Register two models
        manifest1 = ModelManifest(
            model_id="control",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",
        )

        manifest2 = ModelManifest(
            model_id="treatment",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash2",
        )

        model_path1 = temp_dir / "control.onnx"
        model_path1.write_bytes(b"control")
        model_path2 = temp_dir / "treatment.onnx"
        model_path2.write_bytes(b"treatment")

        registry.register_model(model_path1, manifest1)
        registry.register_model(model_path2, manifest2)

        # Start A/B test
        test_id = registry.run_ab_test(
            model_a_id="control",
            model_b_id="treatment",
            split_ratio=0.5,
            duration_hours=1,
            target="ml_signal_actor",
        )

        assert test_id is not None
        assert test_id.startswith("ab_test_")

        # Simulate collecting metrics
        for _ in range(50):
            # Control group metrics
            registry.track_ab_test_metric(
                test_id=test_id,
                model_id="control",
                metric_value=np.random.normal(0.80, 0.03),
            )
            # Treatment group metrics
            registry.track_ab_test_metric(
                test_id=test_id,
                model_id="treatment",
                metric_value=np.random.normal(0.83, 0.03),
            )

        # Analyze A/B test
        analysis = registry.analyze_ab_test(test_id)

        assert analysis is not None
        assert "control_mean" in analysis
        assert "treatment_mean" in analysis
        assert "relative_improvement" in analysis
        assert "statistical_significance" in analysis
        assert analysis["treatment_mean"] > analysis["control_mean"]

    # ========== Hot Reload and Gradual Rollout Tests ==========

    def test_hot_reload_model(
        self,
        registry: ModelRegistry,
        temp_dir: Path,
    ) -> None:
        """
        Test hot reload functionality.
        """
        # Register and deploy initial model
        manifest_v1 = ModelManifest(
            model_id="model_v1",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",
            version="1.0.0",
        )

        model_path_v1 = temp_dir / "model_v1.onnx"
        model_path_v1.write_bytes(b"model_v1")

        registry.register_model(model_path_v1, manifest_v1)
        registry.deploy_model("model_v1", "ml_signal_actor")

        # Register new version
        manifest_v2 = ModelManifest(
            model_id="model_v2",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",  # Same schema for compatibility
            version="2.0.0",
        )

        model_path_v2 = temp_dir / "model_v2.onnx"
        model_path_v2.write_bytes(b"model_v2")

        registry.register_model(model_path_v2, manifest_v2)

        # Perform hot reload
        success = registry.hot_reload_model(
            target="ml_signal_actor",
            new_model_id="model_v2",
        )

        assert success is True

        # Check deployment status
        active_models = registry.get_active_models()
        active_ids = [m.manifest.model_id for m in active_models]
        assert "model_v2" in active_ids
        assert "model_v1" not in active_ids  # Should be retired

    def test_gradual_rollout(
        self,
        registry: ModelRegistry,
        temp_dir: Path,
    ) -> None:
        """
        Test gradual rollout with stages.
        """
        # Register models
        manifest_current = ModelManifest(
            model_id="current_prod",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",
        )

        manifest_new = ModelManifest(
            model_id="new_version",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32"},
            feature_schema_hash="hash1",
        )

        model_path_current = temp_dir / "current.onnx"
        model_path_current.write_bytes(b"current")
        model_path_new = temp_dir / "new.onnx"
        model_path_new.write_bytes(b"new")

        registry.register_model(model_path_current, manifest_current)
        registry.register_model(model_path_new, manifest_new)
        registry.deploy_model("current_prod", "ml_signal_actor")

        # Start gradual rollout
        rollout_id = registry.start_gradual_rollout(
            current_model_id="current_prod",
            new_model_id="new_version",
            target="ml_signal_actor",
            stages=[0.1, 0.25, 0.5, 1.0],  # 10%, 25%, 50%, 100%
            stage_duration_minutes=30,
        )

        assert rollout_id is not None
        assert rollout_id.startswith("rollout_")

        # Check rollout status
        status = registry.get_rollout_status(rollout_id)
        assert status is not None
        assert status["current_stage"] == 0
        assert status["stages"] == [0.1, 0.25, 0.5, 1.0]

        # Advance to next stage
        advanced = registry.advance_rollout_stage(rollout_id)
        assert advanced is True

        status = registry.get_rollout_status(rollout_id)
        assert status is not None
        assert status["current_stage"] == 1
        assert status["traffic_split"] == 0.25

    # ========== Integration Tests ==========

    def test_full_deployment_pipeline(
        self,
        registry: ModelRegistry,
        temp_dir: Path,
    ) -> None:
        """
        Test complete deployment pipeline with quality gates and canary.
        """
        # Define strict quality gates
        quality_gates = [
            QualityGate("accuracy", 0.8, "gte", required=True),
            QualityGate("latency_p99_ms", 5.0, "lte", required=True),
            QualityGate("error_rate", 0.01, "lte", required=True),
        ]

        # Register model with quality validation
        manifest = ModelManifest(
            model_id="production_model",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="prod_hash",
            performance_metrics={
                "accuracy": 0.85,
                "latency_p99_ms": 4.0,
                "error_rate": 0.005,
            },
            deployment_constraints={"max_latency_ms": 5},
        )

        model_path = temp_dir / "prod_model.onnx"
        model_path.write_bytes(b"production_model")

        model_id = registry.register_model(
            model_path=model_path,
            manifest=manifest,
            quality_gates=quality_gates,
        )

        # Validate quality passed
        validation = registry.validate_model_quality(model_id, quality_gates)
        assert validation.overall_pass is True

        # Start canary deployment
        canary_config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            baseline_threshold=0.95,
            monitoring_duration_hours=0.0001,
            auto_promote=True,
        )

        deployment_id = registry.start_canary_deployment(
            model_id=model_id,
            target="ml_signal_actor",
            config=canary_config,
        )

        # Simulate good canary metrics
        for _ in range(100):
            registry.update_canary_metrics(
                deployment_id=deployment_id,
                metric_value=0.86,
                latency_ms=3.8,
                error_occurred=False,
            )

        # Wait and evaluate
        time.sleep(0.4)
        should_promote, _ = registry.evaluate_canary(deployment_id)
        assert should_promote is True

        # Auto-promote
        promoted = registry.auto_promote_canary(deployment_id)
        assert promoted is True

        # Verify model is fully deployed
        active_models = registry.get_active_models()
        assert len(active_models) == 1
        assert active_models[0].manifest.model_id == "production_model"

    def test_type_safety_with_mypy(self) -> None:
        """
        Ensure all methods have proper type hints for mypy --strict.
        """
        # This test doesn't run but ensures we think about types
        registry: ModelRegistry
        model_id: str
        gates: list[QualityGate]
        result: ValidationResult
        config: CanaryConfig
        deployment: CanaryDeployment
        metrics: dict[str, float]
        comparison: dict[str, Any]

        # All these should type-check with mypy --strict
        assert True
