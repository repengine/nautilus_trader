#!/usr/bin/env python3
"""
Test contracts for model registry - defines expected behaviors for all model types.

This module implements test contract driven development, defining the
expected behaviors and invariants that ALL models must satisfy.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC
from abc import abstractmethod

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelRole
from ml.registry.model_registry import ModelManifest


class ModelContract(ABC):
    """
    Abstract base class defining the contract all models must satisfy.

    This implements test contract driven development - each model type
    must pass these invariant tests to be considered valid.

    """

    @pytest.mark.parallel_safe
    @pytest.mark.unit
    @abstractmethod
    def test_manifest_completeness(self, manifest: ModelManifest) -> None:
        """
        Test that manifest contains all required fields.
        """
        assert manifest.model_id is not None
        assert manifest.role in ModelRole
        assert manifest.data_requirements in DataRequirements
        assert manifest.architecture is not None
        assert manifest.feature_schema is not None
        assert manifest.feature_schema_hash is not None
        assert manifest.version is not None

    @abstractmethod
    def test_feature_schema_consistency(self, manifest: ModelManifest) -> None:
        """
        Test that feature schema hash matches actual schema.
        """
        schema_json = json.dumps(manifest.feature_schema, sort_keys=True)
        expected_hash = hashlib.sha256(schema_json.encode()).hexdigest()
        assert (
            manifest.feature_schema_hash == expected_hash
        ), f"Feature schema hash mismatch: {manifest.feature_schema_hash} != {expected_hash}"

    @abstractmethod
    def test_performance_metrics_validity(self, manifest: ModelManifest) -> None:
        """
        Test that performance metrics are within expected bounds.
        """

    @abstractmethod
    def test_deployment_constraints(self, manifest: ModelManifest) -> None:
        """
        Test that deployment constraints are satisfied.
        """


class TeacherModelContract(ModelContract):
    """
    Contract for teacher models using L2/L3 data.
    """

    def test_manifest_completeness(self, manifest: ModelManifest) -> None:
        """
        Teacher models must have complete manifests.
        """
        super().test_manifest_completeness(manifest)
        assert manifest.role == ModelRole.TEACHER
        assert manifest.data_requirements in [
            DataRequirements.L1_L2,
            DataRequirements.L1_L2_L3,
        ], "Teachers must use L2 or L3 data"

    def test_feature_schema_consistency(self, manifest: ModelManifest) -> None:
        """
        Teacher feature schemas must include rich features.
        """
        super().test_feature_schema_consistency(manifest)

        # Teachers should have rich feature sets
        assert (
            len(manifest.feature_schema) >= 20
        ), "Teachers should have at least 20 features from L2/L3 data"

        # Check for expected L2/L3 features
        feature_names = set(manifest.feature_schema.keys())
        l2_features = {"bid_ask_spread", "order_book_imbalance", "depth"}
        assert l2_features.issubset(feature_names) or any(
            "l2" in name.lower() for name in feature_names
        ), "Teacher should have L2 features"

    def test_performance_metrics_validity(self, manifest: ModelManifest) -> None:
        """
        Teacher models focus on accuracy over latency.
        """
        metrics = manifest.performance_metrics

        if "accuracy" in metrics:
            assert metrics["accuracy"] >= 0.6, "Teacher accuracy should be at least 60%"

        if "sharpe_ratio" in metrics:
            assert metrics["sharpe_ratio"] >= 0.5, "Teacher Sharpe ratio should be positive"

        # Teachers can have higher latency since they're offline
        if "inference_latency_ms" in metrics:
            assert (
                metrics["inference_latency_ms"] < 1000
            ), "Teacher inference should be under 1 second"

    def test_deployment_constraints(self, manifest: ModelManifest) -> None:
        """
        Teachers have relaxed deployment constraints.
        """
        constraints = manifest.deployment_constraints

        # Teachers run offline, so latency constraints are relaxed
        if "max_latency_ms" in constraints:
            assert constraints["max_latency_ms"] >= 100, "Teachers can have higher latency"

        # Teachers can use more memory
        if "max_memory_mb" in constraints:
            assert constraints["max_memory_mb"] >= 1000, "Teachers can use more memory"


class StudentModelContract(ModelContract):
    """
    Contract for student models using L1-only data.
    """

    def test_manifest_completeness(self, manifest: ModelManifest) -> None:
        """
        Student models must reference their teacher.
        """
        super().test_manifest_completeness(manifest)
        assert manifest.role == ModelRole.STUDENT
        assert (
            manifest.data_requirements == DataRequirements.L1_ONLY
        ), "Students must use L1-only data for live trading"
        assert manifest.parent_id is not None, "Students must have a teacher (parent_id)"

    def test_feature_schema_consistency(self, manifest: ModelManifest) -> None:
        """
        Student feature schemas must be L1-compatible.
        """
        super().test_feature_schema_consistency(manifest)

        # Students should have efficient feature sets
        assert (
            len(manifest.feature_schema) <= 50
        ), "Students should have efficient feature sets (<=50)"

        # Check all features are L1-derivable
        feature_names = set(manifest.feature_schema.keys())
        l2_l3_indicators = {"order_book", "l2", "l3", "depth", "flow"}
        for indicator in l2_l3_indicators:
            assert not any(
                indicator in name.lower() for name in feature_names
            ), f"Student features cannot contain {indicator} (L2/L3 data)"

    def test_performance_metrics_validity(self, manifest: ModelManifest) -> None:
        """
        Student models must maintain teacher accuracy with low latency.
        """
        metrics = manifest.performance_metrics

        # Students should maintain reasonable accuracy
        if "accuracy" in metrics:
            assert metrics["accuracy"] >= 0.55, "Student accuracy should be at least 55%"

        # Critical: Students must have low latency for live trading
        if "inference_latency_ms" in metrics:
            assert (
                metrics["inference_latency_ms"] < 5
            ), "Student inference MUST be under 5ms for live trading"

        # Check distillation quality
        if "distillation_loss" in metrics:
            assert metrics["distillation_loss"] < 0.1, "Distillation loss should be low"

        if "feature_parity_error" in metrics:
            assert (
                metrics["feature_parity_error"] < 1e-10
            ), "Feature parity error must be extremely small"

    def test_deployment_constraints(self, manifest: ModelManifest) -> None:
        """
        Students have strict deployment constraints for live trading.
        """
        constraints = manifest.deployment_constraints

        # Students must have low latency for live trading
        if "max_latency_ms" in constraints:
            assert constraints["max_latency_ms"] <= 5, "Students must guarantee <5ms latency"

        # Students must be memory efficient
        if "max_memory_mb" in constraints:
            assert constraints["max_memory_mb"] <= 500, "Students must be memory efficient"

    def test_teacher_student_relationship(
        self,
        student_manifest: ModelManifest,
        teacher_manifest: ModelManifest,
    ) -> None:
        """
        Test the relationship between student and teacher.
        """
        assert student_manifest.parent_id == teacher_manifest.model_id
        assert student_manifest.model_id in teacher_manifest.children_ids

        # Student should have subset of teacher's feature types
        for feature_name, feature_type in student_manifest.feature_schema.items():
            if feature_name in teacher_manifest.feature_schema:
                assert (
                    feature_type == teacher_manifest.feature_schema[feature_name]
                ), f"Feature type mismatch for {feature_name}"


class InferenceModelContract(ModelContract):
    """
    Contract for direct inference models (no distillation).
    """

    def test_manifest_completeness(self, manifest: ModelManifest) -> None:
        """
        Inference models are standalone.
        """
        super().test_manifest_completeness(manifest)
        assert manifest.role == ModelRole.INFERENCE
        # Inference models can use any data level
        assert manifest.data_requirements in DataRequirements

    def test_feature_schema_consistency(self, manifest: ModelManifest) -> None:
        """
        Inference models have flexible schemas.
        """
        super().test_feature_schema_consistency(manifest)
        # No specific requirements - depends on use case

    def test_performance_metrics_validity(self, manifest: ModelManifest) -> None:
        """
        Inference models balance accuracy and latency.
        """
        metrics = manifest.performance_metrics

        # Requirements depend on data level
        if manifest.data_requirements == DataRequirements.L1_ONLY:
            # L1-only models need low latency
            if "inference_latency_ms" in metrics:
                assert (
                    metrics["inference_latency_ms"] < 10
                ), "L1-only inference should be under 10ms"
        else:
            # L2/L3 models can have higher latency
            if "inference_latency_ms" in metrics:
                assert (
                    metrics["inference_latency_ms"] < 100
                ), "L2/L3 inference should be under 100ms"

    def test_deployment_constraints(self, manifest: ModelManifest) -> None:
        """
        Inference models have flexible constraints.
        """
        # Constraints depend on specific use case


class ModelContractValidator:
    """
    Validates models against their contracts.

    This is the core of test contract driven development - ensuring
    all models satisfy their behavioral contracts.

    """

    def __init__(self) -> None:
        """
        Initialize the validator with contract mappings.
        """
        self.contracts = {
            ModelRole.TEACHER: TeacherModelContract(),
            ModelRole.STUDENT: StudentModelContract(),
            ModelRole.INFERENCE: InferenceModelContract(),
        }

    def validate(self, manifest: ModelManifest) -> tuple[bool, list[str]]:
        """
        Validate a model against its contract.

        Parameters
        ----------
        manifest : ModelManifest
            Model manifest to validate

        Returns
        -------
        tuple[bool, list[str]]
            Validation result and list of errors

        """
        errors = []
        contract = self.contracts.get(manifest.role)

        if not contract:
            errors.append(f"No contract defined for role: {manifest.role}")
            return False, errors

        # Run all contract tests
        try:
            contract.test_manifest_completeness(manifest)
        except AssertionError as e:
            errors.append(f"Manifest completeness: {e!s}")

        try:
            contract.test_feature_schema_consistency(manifest)
        except AssertionError as e:
            errors.append(f"Feature schema: {e!s}")

        try:
            contract.test_performance_metrics_validity(manifest)
        except AssertionError as e:
            errors.append(f"Performance metrics: {e!s}")

        try:
            contract.test_deployment_constraints(manifest)
        except AssertionError as e:
            errors.append(f"Deployment constraints: {e!s}")

        return len(errors) == 0, errors

    def validate_relationship(
        self,
        child_manifest: ModelManifest,
        parent_manifest: ModelManifest,
    ) -> tuple[bool, list[str]]:
        """
        Validate parent-child relationship between models.

        Parameters
        ----------
        child_manifest : ModelManifest
            Child model manifest
        parent_manifest : ModelManifest
            Parent model manifest

        Returns
        -------
        tuple[bool, list[str]]
            Validation result and list of errors

        """
        errors = []

        if child_manifest.role == ModelRole.STUDENT and parent_manifest.role == ModelRole.TEACHER:
            contract = StudentModelContract()
            try:
                contract.test_teacher_student_relationship(
                    child_manifest,
                    parent_manifest,
                )
            except AssertionError as e:
                errors.append(f"Teacher-student relationship: {e!s}")

        return len(errors) == 0, errors


# Test fixtures for contract validation
def create_valid_teacher_manifest() -> ModelManifest:
    """
    Create a valid teacher model manifest.
    """
    feature_schema = {
        "close": "float32",
        "volume": "float32",
        "bid_ask_spread": "float32",
        "order_book_imbalance": "float32",
        "depth_10": "float32",
        "l2_pressure": "float32",
        # ... 20+ features total
        **{f"feature_{i}": "float32" for i in range(14)},
    }

    schema_json = json.dumps(feature_schema, sort_keys=True)
    schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

    return ModelManifest(
        model_id="teacher_001",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_L2_L3,
        architecture="TFT",
        feature_schema=feature_schema,
        feature_schema_hash=schema_hash,
        performance_metrics={
            "accuracy": 0.72,
            "sharpe_ratio": 1.2,
            "inference_latency_ms": 250,
        },
        deployment_constraints={
            "max_latency_ms": 1000,
            "max_memory_mb": 2000,
        },
        created_at=time.time(),
        last_modified=time.time(),
    )


def create_valid_student_manifest(teacher_id: str) -> ModelManifest:
    """
    Create a valid student model manifest.
    """
    feature_schema = {
        "close": "float32",
        "volume": "float32",
        "rsi": "float32",
        "sma_20": "float32",
        "ema_10": "float32",
        # L1-only features
        **{f"ta_feature_{i}": "float32" for i in range(10)},
    }

    schema_json = json.dumps(feature_schema, sort_keys=True)
    schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

    return ModelManifest(
        model_id="student_001",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema=feature_schema,
        feature_schema_hash=schema_hash,
        parent_id=teacher_id,
        performance_metrics={
            "accuracy": 0.68,
            "inference_latency_ms": 2.5,
            "distillation_loss": 0.05,
            "feature_parity_error": 1e-11,
        },
        deployment_constraints={
            "max_latency_ms": 5,
            "max_memory_mb": 256,
        },
        created_at=time.time(),
        last_modified=time.time(),
    )


class TestModelContracts:
    """
    Test suite for model contracts.
    """

    def test_valid_teacher_contract(self) -> None:
        """
        Test that valid teacher passes contract.
        """
        manifest = create_valid_teacher_manifest()
        validator = ModelContractValidator()

        is_valid, errors = validator.validate(manifest)
        assert is_valid, f"Valid teacher failed: {errors}"
        assert len(errors) == 0

    def test_valid_student_contract(self) -> None:
        """
        Test that valid student passes contract.
        """
        teacher = create_valid_teacher_manifest()
        student = create_valid_student_manifest(teacher.model_id)
        teacher.children_ids.append(student.model_id)

        validator = ModelContractValidator()

        # Validate student alone
        is_valid, errors = validator.validate(student)
        assert is_valid, f"Valid student failed: {errors}"

        # Validate relationship
        is_valid, errors = validator.validate_relationship(student, teacher)
        assert is_valid, f"Valid relationship failed: {errors}"

    def test_invalid_student_with_l2_data(self) -> None:
        """
        Test that student with L2 data fails contract.
        """
        student = create_valid_student_manifest("teacher_001")
        student.data_requirements = DataRequirements.L1_L2  # Invalid!

        validator = ModelContractValidator()
        is_valid, errors = validator.validate(student)

        assert not is_valid
        assert any("L1-only" in err for err in errors)

    def test_invalid_student_high_latency(self) -> None:
        """
        Test that student with high latency fails contract.
        """
        student = create_valid_student_manifest("teacher_001")
        student.performance_metrics["inference_latency_ms"] = 10.0  # Too high!

        validator = ModelContractValidator()
        is_valid, errors = validator.validate(student)

        assert not is_valid
        assert any("under 5ms" in err for err in errors)

    def test_invalid_feature_schema_hash(self) -> None:
        """
        Test that incorrect schema hash fails contract.
        """
        manifest = create_valid_teacher_manifest()
        manifest.feature_schema_hash = "invalid_hash"

        validator = ModelContractValidator()
        is_valid, errors = validator.validate(manifest)

        assert not is_valid
        assert any("hash mismatch" in err for err in errors)

    def test_teacher_without_l2_features(self) -> None:
        """
        Test that teacher without L2 features fails contract.
        """
        manifest = create_valid_teacher_manifest()
        # Remove L2/L3 features
        manifest.feature_schema = {
            "close": "float32",
            "volume": "float32",
            "rsi": "float32",
        }
        # Update hash to match new schema
        schema_json = json.dumps(manifest.feature_schema, sort_keys=True)
        manifest.feature_schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

        validator = ModelContractValidator()
        is_valid, errors = validator.validate(manifest)

        assert not is_valid
        assert any("20 features" in err for err in errors)
