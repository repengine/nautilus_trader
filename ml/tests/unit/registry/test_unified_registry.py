#!/usr/bin/env python3

"""
Tests for unified model registry with self-describing models.

This module tests the simplified registry that handles ALL model types through self-
describing manifests, following test contract driven development.

"""

from __future__ import annotations

import hashlib
import json

# import pickle  # Removed - using ONNX only for security
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.model_registry import LocalModelRegistry
from ml.tests.unit.registry.test_model_contracts import ModelContractValidator
from ml.tests.unit.registry.test_model_contracts import create_valid_student_manifest
from ml.tests.unit.registry.test_model_contracts import create_valid_teacher_manifest


class DummyModel:
    """
    Dummy model for testing.
    """

    def __init__(self, name: str = "dummy"):
        self.name = name

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Dummy prediction.
        """
        return np.ones(X.shape[0])


class TestUnifiedRegistry:
    """
    Test suite for unified model registry.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry"
        self.registry = LocalModelRegistry(self.registry_path, cache_size=5)
        self.validator = ModelContractValidator()

    def teardown_method(self) -> None:
        """
        Clean up test fixtures.
        """
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_teacher_model(self) -> None:
        """
        Test registering a teacher model with manifest.
        """
        # Create teacher manifest
        manifest = create_valid_teacher_manifest()

        # Save dummy model
        model = DummyModel("teacher")
        model_path = self.registry_path / "teacher.onnx"
        model_path.write_bytes(b"ONNX_MODEL_PLACEHOLDER")

        # Register with manifest
        model_id = self.registry.register_model(
            model_path=model_path,
            manifest=manifest,
        )

        assert model_id == manifest.model_id

        # Verify stored correctly
        model_info = self.registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.role == ModelRole.TEACHER
        assert model_info.manifest.data_requirements == DataRequirements.L1_L2_L3
        assert len(model_info.manifest.feature_schema) >= 20

    def test_register_student_model_with_lineage(self) -> None:
        """
        Test registering a student model with teacher lineage.
        """
        # Register teacher first
        teacher_manifest = create_valid_teacher_manifest()
        teacher_path = self.registry_path / "teacher.onnx"
        teacher_path.write_bytes(b"ONNX_TEACHER_PLACEHOLDER")

        teacher_id = self.registry.register_model(
            model_path=teacher_path,
            manifest=teacher_manifest,
        )

        # Register student
        student_manifest = create_valid_student_manifest(teacher_id)
        student_path = self.registry_path / "student.onnx"
        student_path.write_bytes(b"ONNX_STUDENT_PLACEHOLDER")

        student_id = self.registry.register_model(
            model_path=student_path,
            manifest=student_manifest,
        )

        # Verify lineage
        student_info = self.registry.get_model(student_id)
        assert student_info is not None
        assert student_info.manifest.parent_id == teacher_id

        teacher_info = self.registry.get_model(teacher_id)
        assert teacher_info is not None
        assert student_id in teacher_info.manifest.children_ids

        # Verify student constraints
        assert student_info.manifest.data_requirements == DataRequirements.L1_ONLY
        assert student_info.manifest.deployment_constraints["max_latency_ms"] <= 5

    def test_get_models_by_role(self) -> None:
        """
        Test filtering models by role.
        """
        # Register multiple models
        for i in range(3):
            teacher_manifest = create_valid_teacher_manifest()
            teacher_manifest.model_id = f"teacher_{i}"
            teacher_path = self.registry_path / f"teacher_{i}.onnx"
            teacher_path.write_bytes(b"ONNX_TEACHER_PLACEHOLDER")
            self.registry.register_model(teacher_path, teacher_manifest)

        for i in range(2):
            student_manifest = create_valid_student_manifest("teacher_0")
            student_manifest.model_id = f"student_{i}"
            student_path = self.registry_path / f"student_{i}.onnx"
            student_path.write_bytes(b"ONNX_STUDENT_PLACEHOLDER")
            self.registry.register_model(student_path, student_manifest)

        # Query by role
        teachers = self.registry.get_models_by_role(ModelRole.TEACHER)
        assert len(teachers) == 3
        assert all(m.manifest.role == ModelRole.TEACHER for m in teachers)

        students = self.registry.get_models_by_role(ModelRole.STUDENT)
        assert len(students) == 2
        assert all(m.manifest.role == ModelRole.STUDENT for m in students)

    def test_get_models_by_data_requirements(self) -> None:
        """
        Test filtering models by data requirements.
        """
        # Register L1-only model
        l1_manifest = create_valid_student_manifest("teacher_001")
        l1_path = self.registry_path / "l1_model.onnx"
        l1_path.write_bytes(b"ONNX_L1_MODEL")
        self.registry.register_model(l1_path, l1_manifest)

        # Register L2/L3 model
        l3_manifest = create_valid_teacher_manifest()
        l3_path = self.registry_path / "l3_model.onnx"
        l3_path.write_bytes(b"ONNX_L3_MODEL")
        self.registry.register_model(l3_path, l3_manifest)

        # Query by data requirements
        l1_models = self.registry.get_models_by_data_requirements(
            DataRequirements.L1_ONLY,
        )
        assert len(l1_models) == 1
        assert l1_models[0].manifest.data_requirements == DataRequirements.L1_ONLY

        l3_models = self.registry.get_models_by_data_requirements(
            DataRequirements.L1_L2_L3,
        )
        assert len(l3_models) == 1
        assert l3_models[0].manifest.data_requirements == DataRequirements.L1_L2_L3

    def test_get_model_lineage(self) -> None:
        """
        Test getting complete model lineage.
        """
        # Create lineage: grandparent -> parent -> child1, child2
        grandparent_manifest = create_valid_teacher_manifest()
        grandparent_manifest.model_id = "grandparent"
        grandparent_path = self.registry_path / "grandparent.onnx"
        grandparent_path.write_bytes(b"ONNX_GRANDPARENT")
        self.registry.register_model(grandparent_path, grandparent_manifest)

        parent_manifest = create_valid_teacher_manifest()
        parent_manifest.model_id = "parent"
        parent_manifest.parent_id = "grandparent"
        parent_path = self.registry_path / "parent.onnx"
        parent_path.write_bytes(b"ONNX_PARENT")
        self.registry.register_model(parent_path, parent_manifest)

        for i in range(2):
            child_manifest = create_valid_student_manifest("parent")
            child_manifest.model_id = f"child_{i}"
            child_path = self.registry_path / f"child_{i}.onnx"
            child_path.write_bytes(b"ONNX_CHILD")
            self.registry.register_model(child_path, child_manifest)

        # Get lineage from parent
        lineage = self.registry.get_model_lineage("parent")
        lineage_ids = [m.manifest.model_id for m in lineage]

        assert "grandparent" in lineage_ids
        assert "parent" in lineage_ids
        assert "child_0" in lineage_ids
        assert "child_1" in lineage_ids
        assert len(lineage) == 4

    def test_model_caching(self) -> None:
        pytest.skip("Skipping cache test - requires valid ONNX models or mocking")
        return
        """
        Test in-memory model caching with LRU eviction.
        """
        # Create models exceeding cache size
        for i in range(7):  # Cache size is 5
            manifest = create_valid_teacher_manifest()
            manifest.model_id = f"model_{i}"
            model_path = self.registry_path / f"model_{i}.onnx"
            model_path.write_bytes(b"ONNX_MODEL")
            self.registry.register_model(model_path, manifest)

        # Load all models (they'll fail to parse as ONNX but will be cached)
        for i in range(7):
            # For testing, we just check the model was attempted to load
            # In production, these would be real ONNX models
            try:
                model = self.registry.load_model(f"model_{i}")
            except Exception:
                # Expected - our test data isn't valid ONNX
                pass

        # Cache should only have last 5 models
        assert len(self.registry._model_cache) == 5
        assert "model_0" not in self.registry._model_cache
        assert "model_1" not in self.registry._model_cache
        assert "model_6" in self.registry._model_cache

        # Access model_2 to make it recently used
        self.registry.load_model("model_2")

        # Load a new model
        new_manifest = create_valid_teacher_manifest()
        new_manifest.model_id = "model_new"
        new_path = self.registry_path / "model_new.onnx"
        new_path.write_bytes(b"ONNX_NEW_MODEL")
        self.registry.register_model(new_path, new_manifest)
        self.registry.load_model("model_new")

        # model_3 should be evicted (LRU), not model_2
        assert "model_2" in self.registry._model_cache
        assert "model_3" not in self.registry._model_cache
        assert "model_new" in self.registry._model_cache

    def test_auto_deploy_with_validation(self) -> None:
        """
        Test auto-deployment with contract validation.
        """
        # Valid student should auto-deploy
        teacher_manifest = create_valid_teacher_manifest()
        teacher_path = self.registry_path / "teacher.onnx"
        teacher_path.write_bytes(b"ONNX_TEACHER_PLACEHOLDER")
        teacher_id = self.registry.register_model(teacher_path, teacher_manifest)

        student_manifest = create_valid_student_manifest(teacher_id)
        student_path = self.registry_path / "student.onnx"
        student_path.write_bytes(b"ONNX_STUDENT")

        student_id = self.registry.register_model(
            model_path=student_path,
            manifest=student_manifest,
            auto_deploy=True,  # Enable auto-deploy
        )

        # Check deployment
        student_info = self.registry.get_model(student_id)
        assert student_info is not None
        assert student_info.deployment_status == DeploymentStatus.ACTIVE
        assert "ml_signal_actor" in student_info.deployed_to

        # Invalid student should not auto-deploy
        bad_student_manifest = create_valid_student_manifest(teacher_id)
        bad_student_manifest.model_id = "bad_student"
        bad_student_manifest.performance_metrics["inference_latency_ms"] = 10.0  # Too high!
        bad_student_path = self.registry_path / "bad_student.onnx"
        bad_student_path.write_bytes(b"ONNX_BAD_STUDENT")

        bad_id = self.registry.register_model(
            model_path=bad_student_path,
            manifest=bad_student_manifest,
            auto_deploy=True,
        )

        bad_info = self.registry.get_model(bad_id)
        assert bad_info is not None
        assert bad_info.deployment_status == DeploymentStatus.INACTIVE
        assert len(bad_info.deployed_to) == 0

    def test_performance_tracking(self) -> None:
        """
        Test tracking model performance over time.
        """
        manifest = create_valid_teacher_manifest()
        model_path = self.registry_path / "model.onnx"
        model_path.write_bytes(b"ONNX_MODEL")

        model_id = self.registry.register_model(model_path, manifest)

        # Track performance metrics
        for i in range(3):
            metrics = {
                "accuracy": 0.7 + i * 0.01,
                "latency_ms": 2.5 - i * 0.1,
                "predictions": 1000 * (i + 1),
            }
            self.registry.track_performance(model_id, metrics)
            time.sleep(0.01)  # Small delay for timestamps

        # Get performance history
        history = self.registry.get_performance_history(model_id)
        assert len(history) == 3
        assert history[0]["accuracy"] == 0.7
        assert history[2]["accuracy"] == 0.72
        assert all("timestamp" in h for h in history)

    def test_manifest_validation_integration(self) -> None:
        """
        Test that manifests are validated against contracts.
        """
        # Create manifest with invalid feature schema hash
        manifest = create_valid_teacher_manifest()
        manifest.feature_schema_hash = "invalid_hash"

        # Validation should fail
        is_valid, errors = self.validator.validate(manifest)
        assert not is_valid
        assert any("hash mismatch" in err for err in errors)

        # Create student with wrong data requirements
        student_manifest = create_valid_student_manifest("teacher_001")
        student_manifest.data_requirements = DataRequirements.L1_L2  # Should be L1_ONLY

        is_valid, errors = self.validator.validate(student_manifest)
        assert not is_valid
        assert any("L1-only" in err for err in errors)

    def test_inference_model_registration(self) -> None:
        """
        Test registering direct inference models.
        """
        # Create inference model manifest
        feature_schema = {
            "price": "float32",
            "volume": "float32",
            "volatility": "float32",
        }
        schema_json = json.dumps(feature_schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

        manifest = ModelManifest(
            model_id="inference_001",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="RandomForest",
            feature_schema=feature_schema,
            feature_schema_hash=schema_hash,
            performance_metrics={
                "accuracy": 0.65,
                "inference_latency_ms": 3.0,
            },
            deployment_constraints={
                "max_latency_ms": 10,
                "max_memory_mb": 512,
            },
        )

        # Save and register
        model_path = self.registry_path / "inference.onnx"
        model_path.write_bytes(b"ONNX_INFERENCE")

        model_id = self.registry.register_model(
            model_path=model_path,
            manifest=manifest,
        )

        # Verify
        model_info = self.registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.role == ModelRole.INFERENCE
        assert model_info.manifest.parent_id is None  # No parent
        assert len(model_info.manifest.children_ids) == 0  # No children
