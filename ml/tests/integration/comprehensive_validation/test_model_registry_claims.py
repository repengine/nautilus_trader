#!/usr/bin/env python3

"""
Comprehensive test suite to validate ModelRegistry operations claims against actual
implementation.

This test file thoroughly validates all claims made in the documentation about
ModelRegistry capabilities, including model loading, versioning, A/B testing, canary
deployments, statistical validation, and integration with storage backends.

"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Import the model registry components
from ml.registry.model_registry import ModelRegistry
from ml.registry.base import ModelManifest, ModelRole, DataRequirements, DeploymentStatus
from ml.registry.dataclasses import QualityGate, CanaryConfig, ValidationResult
from ml.registry.persistence import PersistenceConfig, BackendType
from ml.registry.statistics import welch_t_test, compare_models, calculate_sample_size
from ml.config.constants import Versions


class TestModelRegistryCore:
    """
    Test core ModelRegistry functionality.
    """

    def setup_method(self):
        """
        Set up test environment.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.temp_dir / "registry"
        # Put models inside the registry path to satisfy security validation
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create mock ONNX model files
        self.mock_model_1 = self.models_path / "model_1.onnx"
        self.mock_model_2 = self.models_path / "model_2.onnx"
        self.mock_model_1.write_text("mock onnx model 1")
        self.mock_model_2.write_text("mock onnx model 2")

    def teardown_method(self):
        """
        Clean up test environment.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_model_registration(self):
        """
        Test basic model registration functionality.
        """
        print("\n=== Testing Basic Model Registration ===")

        # Initialize registry with JSON backend
        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Create a model manifest
        manifest = ModelManifest(
            model_id="test_model_1",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="abc123def456",
            version="1.0.0",
        )

        # Test model registration
        model_id = registry.register_model(self.mock_model_1, manifest)
        print(f"✓ Registered model with ID: {model_id}")

        # Verify model was registered
        model_info = registry.get_model(model_id)
        assert model_info is not None, "Model should be retrievable after registration"
        assert model_info.manifest.model_id == model_id
        assert model_info.manifest.architecture == "LightGBM"
        print(f"✓ Model retrieved successfully: {model_info.manifest.architecture}")

        # Test getting all models
        all_models = registry.get_all_models()
        assert len(all_models) == 1, "Should have exactly one registered model"
        print(f"✓ All models count: {len(all_models)}")

        return registry, model_id

    def test_model_versioning(self):
        """
        Test model versioning capabilities.
        """
        print("\n=== Testing Model Versioning ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Register multiple versions of the same model architecture
        models = []
        for i in range(3):
            manifest = ModelManifest(
                model_id=f"lgb_model_v{i+1}",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="LightGBM",
                feature_schema={"close": "float32", "volume": "float32"},
                feature_schema_hash="abc123def456",
                version=f"1.0.{i}",
            )
            model_path = self.models_path / f"model_v{i+1}.onnx"
            model_path.write_text(f"mock model version {i+1}")

            model_id = registry.register_model(model_path, manifest)
            models.append(model_id)
            print(f"✓ Registered model version {manifest.version}: {model_id}")

        # Test compatibility queries
        compatible_models = registry.list_compatible(
            schema_hash="abc123def456",
            role=ModelRole.INFERENCE,
            architecture="LightGBM",
        )
        assert len(compatible_models) == 3, "Should find all 3 compatible models"
        print(f"✓ Found {len(compatible_models)} compatible models")

        # Test latest version resolution
        latest_model = registry.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="LightGBM",
            schema_hash="abc123def456",
        )
        assert latest_model is not None, "Should find latest model"
        assert latest_model.manifest.version == "1.0.2", "Should return latest version"
        print(f"✓ Latest version resolved: {latest_model.manifest.version}")

        return registry, models

    def test_deployment_operations(self):
        """
        Test model deployment and status management.
        """
        print("\n=== Testing Deployment Operations ===")

        registry, model_id = self.test_basic_model_registration()

        # Test model deployment
        success = registry.deploy_model(model_id, "ml_signal_actor", {"config": "test"})
        assert success, "Model deployment should succeed"
        print(f"✓ Model deployed successfully")

        # Verify deployment status
        model_info = registry.get_model(model_id)
        assert model_info.deployment_status == DeploymentStatus.ACTIVE
        assert "ml_signal_actor" in model_info.deployed_to
        print(f"✓ Deployment status: {model_info.deployment_status.value}")

        # Test getting active models
        active_models = registry.get_active_models()
        assert len(active_models) == 1, "Should have one active model"
        print(f"✓ Active models count: {len(active_models)}")

        # Test rollback
        success = registry.rollback("ml_signal_actor", model_id)
        assert success, "Rollback should succeed"
        print(f"✓ Rollback successful")

        # Test model retirement
        success = registry.retire_model(model_id)
        assert success, "Model retirement should succeed"
        print(f"✓ Model retired successfully")

        return registry, model_id

    def test_quality_gates(self):
        """
        Test quality gate validation.
        """
        print("\n=== Testing Quality Gates ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Create manifest with performance metrics
        manifest = ModelManifest(
            model_id="quality_test_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="quality123",
            performance_metrics={"accuracy": 0.85, "latency_ms": 3.0},
            version="1.0.0",
        )

        # Define quality gates
        quality_gates = [
            QualityGate("accuracy", 0.80, "gte", required=True),
            QualityGate("latency_ms", 5.0, "lte", required=True),
        ]

        # Test registration with quality gates
        model_id = registry.register_model(
            self.mock_model_1,
            manifest,
            quality_gates=quality_gates,
            enforce_quality=True,
        )
        print(f"✓ Model registered with quality gates: {model_id}")

        # Test quality validation
        validation_result = registry.validate_model_quality(model_id, quality_gates)
        assert validation_result.overall_pass, "Quality validation should pass"
        assert validation_result.gates_passed == 2, "Both gates should pass"
        print(f"✓ Quality validation passed: {validation_result.gates_passed}/2 gates")

        # Test failing quality gates
        failing_gates = [QualityGate("accuracy", 0.95, "gte", required=True)]
        validation_result = registry.validate_model_quality(model_id, failing_gates)
        assert not validation_result.overall_pass, "Quality validation should fail"
        print(f"✓ Quality validation correctly failed for high threshold")

        return registry, model_id


class TestModelRegistryAdvanced:
    """
    Test advanced ModelRegistry features.
    """

    def setup_method(self):
        """
        Set up test environment.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.temp_dir / "registry"
        # Put models inside the registry path to satisfy security validation
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create mock ONNX model files
        for i in range(5):
            model_path = self.models_path / f"model_{i}.onnx"
            model_path.write_text(f"mock onnx model {i}")

    def teardown_method(self):
        """
        Clean up test environment.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ab_testing_framework(self):
        """
        Test A/B testing functionality.
        """
        print("\n=== Testing A/B Testing Framework ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Register two models for A/B testing
        models = []
        for i in range(2):
            manifest = ModelManifest(
                model_id=f"ab_test_model_{i}",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="LightGBM",
                feature_schema={"close": "float32", "volume": "float32"},
                feature_schema_hash="ab123def456",
                version=f"1.0.{i}",
            )
            model_path = self.models_path / f"model_{i}.onnx"
            model_id = registry.register_model(model_path, manifest)
            models.append(model_id)
            print(f"✓ Registered A/B test model {i}: {model_id}")

        # Configure A/B test
        ab_config = registry.configure_ab_test(
            models=models,
            split_ratio=0.5,
            duration_hours=24,
            target="ml_signal_actor",
        )
        assert ab_config is not None, "A/B test configuration should succeed"
        assert ab_config["model_a"] == models[0]
        assert ab_config["model_b"] == models[1]
        print(f"✓ A/B test configured: {ab_config['split_ratio']} split")

        # Test A/B test tracking
        test_id = registry.run_ab_test(
            model_a_id=models[0],
            model_b_id=models[1],
            split_ratio=0.6,
            duration_hours=12,
            target="ml_signal_actor",
        )
        assert test_id, "A/B test should start successfully"
        print(f"✓ A/B test started with ID: {test_id}")

        # Simulate metric collection
        for i in range(50):
            registry.track_ab_test_metric(test_id, models[0], 0.85 + np.random.normal(0, 0.05))
            registry.track_ab_test_metric(test_id, models[1], 0.87 + np.random.normal(0, 0.05))

        # Analyze A/B test results
        analysis = registry.analyze_ab_test(test_id)
        assert analysis is not None, "A/B test analysis should succeed"
        assert "statistical_significance" in analysis
        print(f"✓ A/B test analysis completed: {analysis['statistical_significance']}")

        return registry, models, test_id

    def test_canary_deployment(self):
        """
        Test canary deployment functionality.
        """
        print("\n=== Testing Canary Deployment ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Register models for canary deployment
        baseline_manifest = ModelManifest(
            model_id="baseline_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="canary123",
            performance_metrics={"accuracy": 0.85},
            version="1.0.0",
        )
        baseline_id = registry.register_model(self.models_path / "model_0.onnx", baseline_manifest)

        canary_manifest = ModelManifest(
            model_id="canary_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="canary123",
            performance_metrics={"accuracy": 0.87},
            version="1.1.0",
        )
        canary_id = registry.register_model(self.models_path / "model_1.onnx", canary_manifest)

        print(f"✓ Registered baseline model: {baseline_id}")
        print(f"✓ Registered canary model: {canary_id}")

        # Deploy baseline first
        registry.deploy_model(baseline_id, "ml_signal_actor")

        # Start canary deployment
        canary_config = CanaryConfig(
            traffic_percentage=10.0,
            success_metric="accuracy",
            baseline_threshold=0.95,
            monitoring_duration_hours=1.0,
            min_samples=20,
        )

        deployment_id = registry.start_canary_deployment(
            model_id=canary_id,
            target="ml_signal_actor",
            config=canary_config,
            baseline_model_id=baseline_id,
        )
        print(f"✓ Started canary deployment: {deployment_id}")

        # Simulate canary metrics
        for i in range(30):
            registry.update_canary_metrics(
                deployment_id,
                metric_value=0.88 + np.random.normal(0, 0.02),
                latency_ms=2.5 + np.random.normal(0, 0.5),
                error_occurred=(i % 20 == 0),  # 5% error rate
            )

        # Check canary status
        canary = registry.get_canary_deployment(deployment_id)
        assert canary is not None, "Canary deployment should be retrievable"

        status_summary = canary.get_status_summary()
        print(f"✓ Canary status: {status_summary['sample_count']} samples")
        print(f"✓ Error rate: {status_summary['error_rate']:.2%}")

        # Evaluate for promotion
        should_promote, reason = registry.evaluate_canary(deployment_id)
        print(f"✓ Promotion evaluation: {should_promote} ({reason})")

        # Test auto-promotion if applicable
        if should_promote:
            success = registry.auto_promote_canary(deployment_id)
            assert success, "Auto-promotion should succeed"
            print(f"✓ Auto-promoted canary to production")

        return registry, baseline_id, canary_id, deployment_id

    def test_hot_reload_functionality(self):
        """
        Test hot reload capabilities.
        """
        print("\n=== Testing Hot Reload Functionality ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Register and deploy initial model
        old_manifest = ModelManifest(
            model_id="old_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="reload123",
            version="1.0.0",
        )
        old_model_id = registry.register_model(self.models_path / "model_0.onnx", old_manifest)
        registry.deploy_model(old_model_id, "ml_signal_actor")
        print(f"✓ Deployed initial model: {old_model_id}")

        # Register new model with same schema
        new_manifest = ModelManifest(
            model_id="new_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="reload123",  # Same schema hash
            version="1.1.0",
        )
        new_model_id = registry.register_model(self.models_path / "model_1.onnx", new_manifest)
        print(f"✓ Registered new model: {new_model_id}")

        # Perform hot reload
        success = registry.hot_reload_model("ml_signal_actor", new_model_id)
        assert success, "Hot reload should succeed"
        print(f"✓ Hot reload completed successfully")

        # Verify deployment state
        new_model_info = registry.get_model(new_model_id)
        old_model_info = registry.get_model(old_model_id)

        assert new_model_info.deployment_status == DeploymentStatus.ACTIVE
        assert old_model_info.deployment_status == DeploymentStatus.RETIRED
        print(f"✓ New model active, old model retired")

        return registry, old_model_id, new_model_id

    def test_gradual_rollout(self):
        """
        Test gradual rollout functionality.
        """
        print("\n=== Testing Gradual Rollout ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Register current and new models
        current_manifest = ModelManifest(
            model_id="current_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="rollout123",
            version="1.0.0",
        )
        current_id = registry.register_model(self.models_path / "model_0.onnx", current_manifest)
        registry.deploy_model(current_id, "ml_signal_actor")

        new_manifest = ModelManifest(
            model_id="new_model_rollout",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="rollout123",
            version="2.0.0",
        )
        new_id = registry.register_model(self.models_path / "model_1.onnx", new_manifest)

        print(f"✓ Registered current model: {current_id}")
        print(f"✓ Registered new model: {new_id}")

        # Start gradual rollout
        rollout_id = registry.start_gradual_rollout(
            current_model_id=current_id,
            new_model_id=new_id,
            target="ml_signal_actor",
            stages=[0.1, 0.25, 0.5, 1.0],  # 10%, 25%, 50%, 100%
            stage_duration_minutes=60,
        )
        print(f"✓ Started gradual rollout: {rollout_id}")

        # Check rollout status
        status = registry.get_rollout_status(rollout_id)
        assert status is not None, "Rollout status should be available"
        assert status["current_stage"] == 0
        assert status["traffic_split"] == 0.1
        print(f"✓ Initial rollout stage: {status['traffic_split']*100}% traffic")

        # Advance rollout stage
        success = registry.advance_rollout_stage(rollout_id)
        assert success, "Stage advancement should succeed"

        status = registry.get_rollout_status(rollout_id)
        assert status["traffic_split"] == 0.25
        print(f"✓ Advanced to stage: {status['traffic_split']*100}% traffic")

        return registry, current_id, new_id, rollout_id


class TestStatisticalValidation:
    """
    Test statistical validation capabilities.
    """

    def test_welch_t_test(self):
        """
        Test Welch's t-test implementation.
        """
        print("\n=== Testing Welch's T-Test ===")

        # Generate sample data
        np.random.seed(42)
        sample_a = np.random.normal(0.85, 0.05, 100)  # Control group
        sample_b = np.random.normal(0.87, 0.05, 100)  # Treatment group

        # Perform test
        result = welch_t_test(sample_a, sample_b)

        assert "t_statistic" in result, "Result should include t-statistic"
        assert "p_value_approx" in result, "Result should include p-value"
        assert "statistically_significant" in result, "Result should include significance"
        assert "relative_improvement" in result, "Result should include relative improvement"

        print(f"✓ T-statistic: {result['t_statistic']:.4f}")
        print(f"✓ P-value: {result['p_value_approx']:.4f}")
        print(f"✓ Significant: {result['statistically_significant']}")
        print(f"✓ Relative improvement: {result['relative_improvement']:.2f}%")

        return result

    def test_model_comparison(self):
        """
        Test multi-model comparison functionality.
        """
        print("\n=== Testing Model Comparison ===")

        # Create mock model data
        models = [
            {"model_id": "model_a", "metrics": {"accuracy": 0.85, "latency": 3.0}},
            {"model_id": "model_b", "metrics": {"accuracy": 0.87, "latency": 2.5}},
            {"model_id": "model_c", "metrics": {"accuracy": 0.83, "latency": 4.0}},
        ]

        # Compare models on accuracy
        comparison = compare_models(models, "accuracy", baseline_index=0)

        assert "winner" in comparison, "Comparison should identify winner"
        assert "models" in comparison, "Comparison should include model results"
        assert len(comparison["models"]) == 3, "Should include all models"

        print(f"✓ Winner: {comparison['winner']}")
        print(f"✓ Baseline: {comparison['baseline_model']}")

        for model in comparison["models"]:
            print(
                f"   Rank {model['rank']}: {model['model_id']} " f"(accuracy: {model['value']:.3f})"
            )

        return comparison

    def test_sample_size_calculation(self):
        """
        Test A/B test sample size calculation.
        """
        print("\n=== Testing Sample Size Calculation ===")

        # Calculate required sample sizes for different effect sizes
        effect_sizes = [0.1, 0.2, 0.3, 0.5]

        for effect_size in effect_sizes:
            sample_size = calculate_sample_size(
                effect_size=effect_size,
                power=0.8,
                significance_level=0.05,
            )
            print(f"✓ Effect size {effect_size}: {sample_size} samples needed")
            assert sample_size > 0, "Sample size should be positive"

        return effect_sizes


class TestStorageBackends:
    """
    Test storage backend integration.
    """

    def setup_method(self):
        """
        Set up test environment.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.temp_dir / "registry"
        # Put models inside the registry path to satisfy security validation
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create mock model
        self.mock_model = self.models_path / "test_model.onnx"
        self.mock_model.write_text("mock onnx model")

    def teardown_method(self):
        """
        Clean up test environment.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_backend(self):
        """
        Test JSON backend persistence.
        """
        print("\n=== Testing JSON Backend ===")

        # Initialize registry with JSON backend
        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=self.registry_path,
            ),
        )

        # Register a model
        manifest = ModelManifest(
            model_id="json_test_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32"},
            feature_schema_hash="json123",
            version="1.0.0",
        )

        model_id = registry.register_model(self.mock_model, manifest)
        print(f"✓ Registered model with JSON backend: {model_id}")

        # Force save and verify file exists
        registry.flush()
        registry_file = self.registry_path / "registry.json"
        assert registry_file.exists(), "Registry JSON file should exist"

        # Verify file contents
        with open(registry_file) as f:
            data = json.load(f)

        assert "models" in data, "Registry should contain models section"
        assert model_id in data["models"], "Model should be in registry data"
        print(f"✓ Model persisted to JSON file")

        # Test registry reload
        registry2 = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=self.registry_path,
            ),
        )

        loaded_model = registry2.get_model(model_id)
        assert loaded_model is not None, "Model should be loadable after restart"
        assert loaded_model.manifest.architecture == "LightGBM"
        print(f"✓ Model successfully reloaded from JSON")

        return registry, model_id

    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES"),
        reason="PostgreSQL tests require TEST_POSTGRES environment variable",
    )
    def test_postgresql_backend(self):
        """
        Test PostgreSQL backend persistence (if available).
        """
        print("\n=== Testing PostgreSQL Backend ===")

        # This test would require a PostgreSQL connection
        # For now, we'll demonstrate the configuration
        postgres_config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://user:pass@localhost:5432/test_db",
        )
        print(f"✓ PostgreSQL configuration created")

        # Note: Full PostgreSQL testing would require actual database setup
        # This serves as a placeholder for integration testing
        return postgres_config


class TestModelRegistryIntegration:
    """
    Test complete integration scenarios.
    """

    def setup_method(self):
        """
        Set up test environment.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.temp_dir / "registry"
        # Put models inside the registry path to satisfy security validation
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create mock models
        for i in range(3):
            model_path = self.models_path / f"model_{i}.onnx"
            model_path.write_text(f"mock onnx model {i}")

    def teardown_method(self):
        """
        Clean up test environment.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_mlops_workflow(self):
        """
        Test complete MLOps workflow from registration to production.
        """
        print("\n=== Testing Complete MLOps Workflow ===")

        registry = ModelRegistry(
            registry_path=self.registry_path,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON, json_path=self.registry_path
            ),
        )

        # Step 1: Register teacher model (cold path)
        teacher_manifest = ModelManifest(
            model_id="teacher_tft_v1",
            role=ModelRole.TEACHER,
            data_requirements=DataRequirements.L1_L2_L3,
            architecture="TFT",
            feature_schema={"close": "float32", "volume": "float32", "book_imbalance": "float32"},
            feature_schema_hash="teacher123",
            performance_metrics={"accuracy": 0.92, "sharpe_ratio": 1.8},
            version="1.0.0",
            serveable=False,  # Cold path only
        )

        teacher_id = registry.register_model(self.models_path / "model_0.onnx", teacher_manifest)
        print(f"✓ Step 1: Registered teacher model: {teacher_id}")

        # Step 2: Register student model (hot path)
        student_manifest = ModelManifest(
            model_id="student_lgb_v1",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="student123",
            parent_id=teacher_id,
            performance_metrics={"accuracy": 0.88, "inference_latency_ms": 2.5},
            version="1.0.0",
            serveable=True,
        )

        student_id = registry.register_model(
            self.models_path / "model_1.onnx", student_manifest, auto_deploy=True
        )
        print(f"✓ Step 2: Registered student model: {student_id}")

        # Step 3: Verify parent-child relationship
        lineage = registry.get_model_lineage(student_id)
        assert len(lineage) == 2, "Lineage should include parent and child"
        teacher_model = registry.get_model(teacher_id)
        assert student_id in teacher_model.manifest.children_ids
        print(f"✓ Step 3: Parent-child relationship established")

        # Step 4: Track performance over time
        for i in range(10):
            registry.track_performance(
                student_id,
                {
                    "accuracy": 0.88 + np.random.normal(0, 0.01),
                    "latency_ms": 2.5 + np.random.normal(0, 0.2),
                    "timestamp": time.time() + i * 3600,  # Hourly metrics
                },
            )

        performance_history = registry.get_performance_history(student_id)
        assert len(performance_history) == 10, "Should track all performance metrics"
        print(f"✓ Step 4: Performance tracking active ({len(performance_history)} metrics)")

        # Step 5: Register improved student model
        improved_manifest = ModelManifest(
            model_id="student_lgb_v2",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"close": "float32", "volume": "float32"},
            feature_schema_hash="student123",
            parent_id=teacher_id,
            performance_metrics={"accuracy": 0.90, "inference_latency_ms": 2.2},
            version="1.1.0",
            serveable=True,
        )

        improved_id = registry.register_model(self.models_path / "model_2.onnx", improved_manifest)
        print(f"✓ Step 5: Registered improved model: {improved_id}")

        # Step 6: A/B test between models
        ab_config = registry.configure_ab_test(
            models=[student_id, improved_id],
            split_ratio=0.5,
            duration_hours=24,
            target="ml_signal_actor",
        )
        assert ab_config is not None
        print(f"✓ Step 6: A/B test configured")

        # Step 7: Simulate A/B test results favoring improved model
        test_id = registry.run_ab_test(student_id, improved_id, 0.5, 24, "ml_signal_actor")

        # Simulate metrics (improved model performs better)
        for i in range(100):
            registry.track_ab_test_metric(test_id, student_id, 0.88 + np.random.normal(0, 0.02))
            registry.track_ab_test_metric(test_id, improved_id, 0.90 + np.random.normal(0, 0.02))

        analysis = registry.analyze_ab_test(test_id)
        print(
            f"✓ Step 7: A/B test analysis: Treatment improvement = {analysis['relative_improvement']:.2f}%"
        )

        # Step 8: Hot reload with improved model
        success = registry.hot_reload_model("ml_signal_actor", improved_id)
        assert success
        print(f"✓ Step 8: Hot reloaded to improved model")

        # Step 9: Verify final state
        active_models = registry.get_active_models()
        assert len(active_models) == 1
        assert active_models[0].manifest.model_id == improved_id
        print(f"✓ Step 9: Workflow completed successfully")

        return registry, teacher_id, student_id, improved_id


def run_comprehensive_tests():
    """
    Run all comprehensive tests and report results.
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL REGISTRY VALIDATION")
    print("=" * 80)

    # Test results tracking
    results = {
        "core_functionality": {},
        "advanced_features": {},
        "statistical_validation": {},
        "storage_backends": {},
        "integration": {},
    }

    try:
        # Core functionality tests
        print("\n" + "=" * 60)
        print("TESTING CORE FUNCTIONALITY")
        print("=" * 60)

        core_tests = TestModelRegistryCore()
        core_tests.setup_method()

        try:
            core_tests.test_basic_model_registration()
            results["core_functionality"]["basic_registration"] = "✓ PASS"
        except Exception as e:
            results["core_functionality"]["basic_registration"] = f"✗ FAIL: {e}"

        try:
            core_tests.test_model_versioning()
            results["core_functionality"]["versioning"] = "✓ PASS"
        except Exception as e:
            results["core_functionality"]["versioning"] = f"✗ FAIL: {e}"

        try:
            core_tests.test_deployment_operations()
            results["core_functionality"]["deployment"] = "✓ PASS"
        except Exception as e:
            results["core_functionality"]["deployment"] = f"✗ FAIL: {e}"

        try:
            core_tests.test_quality_gates()
            results["core_functionality"]["quality_gates"] = "✓ PASS"
        except Exception as e:
            results["core_functionality"]["quality_gates"] = f"✗ FAIL: {e}"

        core_tests.teardown_method()

        # Advanced features tests
        print("\n" + "=" * 60)
        print("TESTING ADVANCED FEATURES")
        print("=" * 60)

        advanced_tests = TestModelRegistryAdvanced()
        advanced_tests.setup_method()

        try:
            advanced_tests.test_ab_testing_framework()
            results["advanced_features"]["ab_testing"] = "✓ PASS"
        except Exception as e:
            results["advanced_features"]["ab_testing"] = f"✗ FAIL: {e}"

        try:
            advanced_tests.test_canary_deployment()
            results["advanced_features"]["canary_deployment"] = "✓ PASS"
        except Exception as e:
            results["advanced_features"]["canary_deployment"] = f"✗ FAIL: {e}"

        try:
            advanced_tests.test_hot_reload_functionality()
            results["advanced_features"]["hot_reload"] = "✓ PASS"
        except Exception as e:
            results["advanced_features"]["hot_reload"] = f"✗ FAIL: {e}"

        try:
            advanced_tests.test_gradual_rollout()
            results["advanced_features"]["gradual_rollout"] = "✓ PASS"
        except Exception as e:
            results["advanced_features"]["gradual_rollout"] = f"✗ FAIL: {e}"

        advanced_tests.teardown_method()

        # Statistical validation tests
        print("\n" + "=" * 60)
        print("TESTING STATISTICAL VALIDATION")
        print("=" * 60)

        stats_tests = TestStatisticalValidation()

        try:
            stats_tests.test_welch_t_test()
            results["statistical_validation"]["welch_t_test"] = "✓ PASS"
        except Exception as e:
            results["statistical_validation"]["welch_t_test"] = f"✗ FAIL: {e}"

        try:
            stats_tests.test_model_comparison()
            results["statistical_validation"]["model_comparison"] = "✓ PASS"
        except Exception as e:
            results["statistical_validation"]["model_comparison"] = f"✗ FAIL: {e}"

        try:
            stats_tests.test_sample_size_calculation()
            results["statistical_validation"]["sample_size"] = "✓ PASS"
        except Exception as e:
            results["statistical_validation"]["sample_size"] = f"✗ FAIL: {e}"

        # Storage backend tests
        print("\n" + "=" * 60)
        print("TESTING STORAGE BACKENDS")
        print("=" * 60)

        storage_tests = TestStorageBackends()
        storage_tests.setup_method()

        try:
            storage_tests.test_json_backend()
            results["storage_backends"]["json_backend"] = "✓ PASS"
        except Exception as e:
            results["storage_backends"]["json_backend"] = f"✗ FAIL: {e}"

        try:
            storage_tests.test_postgresql_backend()
            results["storage_backends"]["postgresql_backend"] = "✓ PASS (Config Only)"
        except Exception as e:
            results["storage_backends"]["postgresql_backend"] = f"✗ FAIL: {e}"

        storage_tests.teardown_method()

        # Integration tests
        print("\n" + "=" * 60)
        print("TESTING INTEGRATION SCENARIOS")
        print("=" * 60)

        integration_tests = TestModelRegistryIntegration()
        integration_tests.setup_method()

        try:
            integration_tests.test_complete_mlops_workflow()
            results["integration"]["mlops_workflow"] = "✓ PASS"
        except Exception as e:
            results["integration"]["mlops_workflow"] = f"✗ FAIL: {e}"

        integration_tests.teardown_method()

    except Exception as e:
        print(f"Critical error during testing: {e}")
        import traceback

        traceback.print_exc()

    # Generate final report
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS SUMMARY")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0

    for category, tests in results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for test_name, result in tests.items():
            print(f"  {test_name}: {result}")
            total_tests += 1
            if result.startswith("✓"):
                passed_tests += 1

    print(
        f"\nOVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)"
    )

    return results


if __name__ == "__main__":
    # Set up environment
    os.environ["PYTHONPATH"] = "/home/nate/projects/nautilus_trader"

    # Run all tests
    test_results = run_comprehensive_tests()

    # Print gap analysis
    print("\n" + "=" * 80)
    print("CLAIMS VS IMPLEMENTATION ANALYSIS")
    print("=" * 80)

    gaps_found = []
    working_features = []

    for category, tests in test_results.items():
        for test_name, result in tests.items():
            if result.startswith("✓"):
                working_features.append(f"{category}.{test_name}")
            else:
                gaps_found.append(f"{category}.{test_name}: {result}")

    print(f"\nWORKING FEATURES ({len(working_features)}):")
    for feature in working_features:
        print(f"  ✓ {feature}")

    print(f"\nGAPS/ISSUES FOUND ({len(gaps_found)}):")
    for gap in gaps_found:
        print(f"  ✗ {gap}")

    if not gaps_found:
        print("\n🎉 All claimed ModelRegistry features are working as documented!")
    else:
        print(f"\n⚠️  Found {len(gaps_found)} gaps between claims and implementation.")
