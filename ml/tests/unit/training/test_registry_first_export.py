#!/usr/bin/env python3

"""
Unit tests for registry-first training export and promotion.

Tests the integration between training outputs and the ModelRegistry/FeatureRegistry
to ensure proper model registration with feature parity validation.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.registry import FeatureRegistry, ModelRegistry
from ml.registry.base import DataRequirements, ModelManifest, ModelRole
from ml.registry.feature_registry import FeatureManifest, FeatureRole, FeatureStage, compute_schema_hash
from ml.training.export import (
    create_model_manifest_stub,
    detect_model_type,
    register_model_with_registry,
    save_model_with_metadata,
)


class MockMLTrainer:
    """Mock trainer for testing registry integration."""

    def __init__(self, model_role: str = "inference"):
        self.model = MagicMock()
        self.model.__class__.__name__ = "MockXGBClassifier"
        self.feature_names = ["feature_1", "feature_2", "feature_3", "rsi_14", "ema_20"]
        self.training_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "training_time": 120.5,
            "feature_count": 5,
            "non_numeric_field": "should_be_filtered",
        }
        self.config = MagicMock()
        self.config.model_role = model_role
        self.config.feature_set_id = "test_feature_set_001"
        self.config.pipeline_signature = "test_pipeline_v1_hash"
        self.config.pipeline_version = "1.2.0"


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for registry testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def feature_registry(temp_registry_dir):
    """Create a FeatureRegistry for testing."""
    return FeatureRegistry(temp_registry_dir)


@pytest.fixture
def model_registry(temp_registry_dir):
    """Create a ModelRegistry for testing."""
    return ModelRegistry(temp_registry_dir)


@pytest.fixture
def sample_feature_manifest():
    """Create a sample FeatureManifest for testing."""
    feature_names = ["feature_1", "feature_2", "feature_3", "rsi_14", "ema_20"]
    feature_dtypes = ["float32"] * len(feature_names)
    pipeline_signature = "test_pipeline_v1_hash"

    return FeatureManifest(
        feature_set_id="test_feature_set_001",
        name="Test Feature Set",
        version="1.0.0",
        role=FeatureRole.INFERENCE_SUPPORT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        schema_hash=compute_schema_hash(feature_names, feature_dtypes, pipeline_signature),
        pipeline_signature=pipeline_signature,
        pipeline_version="1.2.0",
        stage=FeatureStage.PROD,
        created_at=time.time(),
        last_modified=time.time(),
    )


class TestCreateModelManifestStub:
    """Tests for create_model_manifest_stub function."""

    def test_create_basic_manifest_stub(self):
        """Test creating a basic model manifest stub."""
        mock_trainer = MockMLTrainer()

        manifest_data = create_model_manifest_stub(
            model=mock_trainer.model,
            feature_names=mock_trainer.feature_names,
            training_metrics=mock_trainer.training_metrics,
            model_role="inference",
            data_requirements="l1_only",
            architecture="XGBoost",
        )

        # Verify basic structure
        assert manifest_data["role"] == "inference"
        assert manifest_data["data_requirements"] == "l1_only"
        assert manifest_data["architecture"] == "XGBoost"
        # Note: serveable is determined by model interface (has .run method)
        assert manifest_data["artifact_format"] == "onnx" if manifest_data["serveable"] else "native"

        # Verify feature schema
        expected_schema = dict.fromkeys(mock_trainer.feature_names, "float32")
        assert manifest_data["feature_schema"] == expected_schema

        # Verify performance metrics (only numeric)
        expected_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "training_time": 120.5,
            "feature_count": 5.0,
        }
        assert manifest_data["performance_metrics"] == expected_metrics

        # Verify schema hash is computed
        assert manifest_data["feature_schema_hash"] != ""
        assert len(manifest_data["feature_schema_hash"]) == 64  # SHA-256

    def test_create_manifest_with_onnx_model(self):
        """Test creating manifest for ONNX model (serveable)."""
        mock_model = MagicMock()
        mock_model.run = MagicMock()  # ONNX-like interface

        manifest_data = create_model_manifest_stub(
            model=mock_model,
            feature_names=["feature_1", "feature_2"],
            architecture="onnx",
        )

        assert manifest_data["serveable"] is True
        assert manifest_data["artifact_format"] == "onnx"

    def test_create_manifest_with_pipeline_info(self):
        """Test creating manifest with pipeline signature and version."""
        mock_trainer = MockMLTrainer()

        manifest_data = create_model_manifest_stub(
            model=mock_trainer.model,
            feature_names=mock_trainer.feature_names,
            feature_set_id="test_feature_set_001",
            pipeline_signature="abc123def456",
            pipeline_version="2.1.0",
        )

        assert manifest_data["feature_set_id"] == "test_feature_set_001"
        assert manifest_data["pipeline_signature"] == "abc123def456"
        assert manifest_data["pipeline_version"] == "2.1.0"

    def test_auto_detect_architecture(self):
        """Test automatic architecture detection."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"

        with patch("ml.training.export.detect_model_type") as mock_detect:
            mock_detect.return_value.value = "xgboost"

            manifest_data = create_model_manifest_stub(
                model=mock_model,
                feature_names=["feature_1"],
                architecture=None,  # Should auto-detect
            )

            assert manifest_data["architecture"] == "xgboost"
            mock_detect.assert_called_once_with(mock_model)


class TestRegisterModelWithRegistry:
    """Tests for register_model_with_registry function."""

    def test_register_model_basic(self, temp_registry_dir):
        """Test basic model registration."""
        # Create a dummy model file
        model_path = temp_registry_dir / "test_model.onnx"
        model_path.write_text("dummy onnx content")

        # Create manifest data
        manifest_data = {
            "model_id": "",
            "role": "inference",
            "data_requirements": "l1_only",
            "architecture": "XGBoost",
            "feature_schema": {"feature_1": "float32", "feature_2": "float32"},
            "feature_schema_hash": "abc123",
            "parent_id": None,
            "children_ids": [],
            "training_config": {},
            "performance_metrics": {"accuracy": 0.85},
            "deployment_constraints": {"max_inference_latency_ms": 50.0},
            "version": "1.0.0",
            "created_at": time.time(),
            "last_modified": time.time(),
            "serveable": True,
            "artifact_format": "onnx",
            "feature_set_id": None,
            "pipeline_signature": None,
            "pipeline_version": None,
            "decision_policy": None,
            "decision_config": {},
            "artifact_sha256_digest": None,
        }

        # Register model
        model_id = register_model_with_registry(
            model_path=model_path,
            manifest_data=manifest_data,
            registry_path=temp_registry_dir,
        )

        # Verify registration
        assert model_id != ""
        assert model_id.startswith("model_")

        # Verify model is in registry
        registry = ModelRegistry(temp_registry_dir)
        model_info = registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.architecture == "XGBoost"
        assert model_info.manifest.performance_metrics["accuracy"] == 0.85

    def test_register_with_auto_deploy(self, temp_registry_dir):
        """Test model registration with auto-deploy enabled."""
        model_path = temp_registry_dir / "test_model.onnx"
        model_path.write_text("dummy onnx content")

        manifest_data = {
            "model_id": "",
            "role": "student",  # Student models can auto-deploy
            "data_requirements": "l1_only",
            "architecture": "LightGBM",
            "feature_schema": {"feature_1": "float32"},
            "feature_schema_hash": "abc123",
            "parent_id": "teacher_model_123",
            "children_ids": [],
            "training_config": {},
            "performance_metrics": {"inference_latency_ms": 25.0},  # Within limits
            "deployment_constraints": {"max_inference_latency_ms": 50.0},
            "version": "1.0.0",
            "created_at": time.time(),
            "last_modified": time.time(),
            "serveable": True,
            "artifact_format": "onnx",
            "feature_set_id": None,
            "pipeline_signature": None,
            "pipeline_version": None,
            "decision_policy": None,
            "decision_config": {},
            "artifact_sha256_digest": None,
        }

        # Register with auto-deploy
        model_id = register_model_with_registry(
            model_path=model_path,
            manifest_data=manifest_data,
            registry_path=temp_registry_dir,
            auto_deploy=True,
        )

        # Verify model was registered and potentially deployed
        registry = ModelRegistry(temp_registry_dir)
        model_info = registry.get_model(model_id)
        assert model_info is not None
        # Note: Auto-deployment success depends on validation logic


class TestFeatureParity:
    """Tests for feature parity validation between ModelRegistry and FeatureRegistry."""

    def test_feature_parity_validation_success(self, temp_registry_dir, sample_feature_manifest):
        """Test successful feature parity validation."""
        # Register feature set first
        feature_registry = FeatureRegistry(temp_registry_dir)
        feature_set_id = feature_registry.register_feature_set(sample_feature_manifest)

        # Create model manifest with matching feature schema
        model_manifest = ModelManifest(
            model_id="",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema=dict.fromkeys(sample_feature_manifest.feature_names, "float32"),
            feature_schema_hash=sample_feature_manifest.schema_hash,  # Matching hash
            feature_set_id=feature_set_id,  # Link to feature set
            pipeline_signature=sample_feature_manifest.pipeline_signature,
            pipeline_version=sample_feature_manifest.pipeline_version,
            serveable=True,
            artifact_format="onnx",
        )

        # Create dummy model file
        model_path = temp_registry_dir / "test_model.onnx"
        model_path.write_text("dummy onnx content")

        # Register model - should succeed with parity validation
        model_registry = ModelRegistry(temp_registry_dir)
        model_id = model_registry.register_model(
            model_path=model_path,
            manifest=model_manifest,
        )

        # Verify registration succeeded
        assert model_id != ""
        model_info = model_registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.feature_set_id == feature_set_id

    def test_feature_parity_validation_hash_mismatch(self, temp_registry_dir, sample_feature_manifest):
        """Test feature parity validation with hash mismatch."""
        # Register feature set first
        feature_registry = FeatureRegistry(temp_registry_dir)
        feature_set_id = feature_registry.register_feature_set(sample_feature_manifest)

        # Create model manifest with mismatched feature schema hash
        model_manifest = ModelManifest(
            model_id="",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema=dict.fromkeys(sample_feature_manifest.feature_names, "float32"),
            feature_schema_hash="mismatched_hash_123",  # Wrong hash
            feature_set_id=feature_set_id,
            serveable=True,
            artifact_format="onnx",
        )

        model_path = temp_registry_dir / "test_model.onnx"
        model_path.write_text("dummy onnx content")

        # Registration should succeed with warning (unless strict mode enabled)
        model_registry = ModelRegistry(temp_registry_dir)
        model_id = model_registry.register_model(
            model_path=model_path,
            manifest=model_manifest,
        )

        # Should still register but with warnings
        assert model_id != ""

    def test_feature_parity_strict_mode_failure(self, temp_registry_dir, sample_feature_manifest):
        """Test feature parity validation failure in strict mode."""
        # Enable strict parity mode
        with patch.dict("os.environ", {"ML_STRICT_FEATURE_PARITY": "1"}):
            # Register feature set first
            feature_registry = FeatureRegistry(temp_registry_dir)
            feature_set_id = feature_registry.register_feature_set(sample_feature_manifest)

            # Create model manifest with mismatched feature schema hash
            model_manifest = ModelManifest(
                model_id="",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="XGBoost",
                feature_schema=dict.fromkeys(sample_feature_manifest.feature_names, "float32"),
                feature_schema_hash="mismatched_hash_123",  # Wrong hash
                feature_set_id=feature_set_id,
                serveable=True,
                artifact_format="onnx",
            )

            model_path = temp_registry_dir / "test_model.onnx"
            model_path.write_text("dummy onnx content")

            # Registration should fail in strict mode
            model_registry = ModelRegistry(temp_registry_dir)
            with pytest.raises(ValueError, match="feature_schema_hash mismatch"):
                model_registry.register_model(
                    model_path=model_path,
                    manifest=model_manifest,
                )


class TestEndToEndTrainingIntegration:
    """End-to-end integration tests."""

    def test_complete_training_to_registry_flow(self, temp_registry_dir):
        """Test complete flow from training to registry registration."""
        # 1. Setup feature registry with feature set
        feature_registry = FeatureRegistry(temp_registry_dir)
        feature_manifest = FeatureManifest(
            feature_set_id="",
            name="E2E Test Features",
            version="1.0.0",
            role=FeatureRole.INFERENCE_SUPPORT,
            data_requirements=DataRequirements.L1_ONLY,
            feature_names=["close", "volume", "rsi_14", "ema_20", "bb_upper"],
            feature_dtypes=["float32"] * 5,
            schema_hash="",  # Will be computed
            pipeline_signature="e2e_test_pipeline_v1",
            pipeline_version="1.0.0",
            stage=FeatureStage.PROD,
        )

        # Compute and set schema hash
        feature_manifest.schema_hash = compute_schema_hash(
            feature_manifest.feature_names,
            feature_manifest.feature_dtypes,
            feature_manifest.pipeline_signature,
        )

        feature_set_id = feature_registry.register_feature_set(feature_manifest)

        # 2. Create mock training results
        mock_trainer = MockMLTrainer("student")
        mock_trainer.feature_names = feature_manifest.feature_names
        mock_trainer.config.feature_set_id = feature_set_id
        mock_trainer.config.pipeline_signature = feature_manifest.pipeline_signature
        mock_trainer.config.pipeline_version = feature_manifest.pipeline_version
        mock_trainer.config.parent_model_id = "teacher_model_xyz"

        # 3. Save model artifact
        model_path = temp_registry_dir / "e2e_test_model.onnx"
        with patch("ml.training.export.save_model_with_metadata") as mock_save:
            mock_save.return_value = model_path
            model_path.write_text("dummy onnx content")

            # 4. Create manifest and register
            manifest_data = create_model_manifest_stub(
                model=mock_trainer.model,
                feature_names=mock_trainer.feature_names,
                training_metrics=mock_trainer.training_metrics,
                model_role="student",
                data_requirements="l1_only",
                architecture="LightGBM",
                feature_set_id=feature_set_id,
                pipeline_signature=feature_manifest.pipeline_signature,
                pipeline_version=feature_manifest.pipeline_version,
            )

            model_id = register_model_with_registry(
                model_path=model_path,
                manifest_data=manifest_data,
                registry_path=temp_registry_dir,
                auto_deploy=True,
            )

        # 5. Verify complete registration
        model_registry = ModelRegistry(temp_registry_dir)
        model_info = model_registry.get_model(model_id)
        assert model_info is not None

        # Verify feature parity
        assert model_info.manifest.feature_set_id == feature_set_id
        assert model_info.manifest.feature_schema_hash == feature_manifest.schema_hash
        assert model_info.manifest.pipeline_signature == feature_manifest.pipeline_signature

        # Verify model properties
        assert model_info.manifest.role == ModelRole.STUDENT
        assert model_info.manifest.serveable is True
        assert model_info.manifest.parent_id == "teacher_model_xyz"

        # 6. Verify feature registry still has the feature set
        retrieved_feature_info = feature_registry.get_feature_set(feature_set_id)
        assert retrieved_feature_info is not None
        assert retrieved_feature_info.manifest.schema_hash == model_info.manifest.feature_schema_hash

    def test_onnx_model_registration_with_validation(self, temp_registry_dir):
        """Test ONNX model registration with proper validation."""
        # Create a minimal ONNX-like content
        model_path = temp_registry_dir / "validated_model.onnx"
        model_path.write_text("mock onnx model content")

        # Create manifest with ONNX-specific properties
        manifest_data = create_model_manifest_stub(
            model=MagicMock(),
            feature_names=["feature_1", "feature_2", "feature_3"],
            training_metrics={"accuracy": 0.92, "inference_latency_ms": 15.0},
            model_role="inference",
            data_requirements="l1_only",
            architecture="onnx",
        )

        # Ensure it's marked as serveable
        assert manifest_data["serveable"] is True
        assert manifest_data["artifact_format"] == "onnx"

        # Register model
        model_id = register_model_with_registry(
            model_path=model_path,
            manifest_data=manifest_data,
            registry_path=temp_registry_dir,
        )

        # Verify ONNX model registration
        model_registry = ModelRegistry(temp_registry_dir)
        model_info = model_registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.serveable is True
        assert model_info.manifest.artifact_format == "onnx"
        assert model_info.model_path == model_path


if __name__ == "__main__":
    pytest.main([__file__])
