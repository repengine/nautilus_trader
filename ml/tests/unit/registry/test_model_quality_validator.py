#!/usr/bin/env python3

"""
Unit tests for ModelQualityValidator component.

Tests cover:
- Quality gate validation
- All comparison operators (gte, lte, gt, lt, eq)
- Required vs optional gates
- Missing metrics handling
- Margin calculations
- Overall pass/fail logic

"""

import pytest

from ml.registry.dataclasses import QualityGate, ValidationResult
from ml.registry.model_quality_validator import ModelQualityValidator


@pytest.fixture
def validator():
    """Create ModelQualityValidator instance."""
    return ModelQualityValidator()


# ========== Comparison Operator Tests ==========


def test_evaluate_gate_gte_pass(validator):
    """Test gte comparison operator - passing case."""
    gate = QualityGate(
        metric_name="accuracy",
        threshold=0.8,
        comparison="gte",
        required=True,
    )

    result = validator.evaluate_gate(gate, 0.85)

    assert result["passed"] is True
    assert result["actual"] == 0.85
    assert result["threshold"] == 0.8
    assert result["margin"] == pytest.approx(0.05)
    assert result["comparison"] == "gte"


def test_evaluate_gate_gte_fail(validator):
    """Test gte comparison operator - failing case."""
    gate = QualityGate(
        metric_name="accuracy",
        threshold=0.8,
        comparison="gte",
        required=True,
    )

    result = validator.evaluate_gate(gate, 0.75)

    assert result["passed"] is False
    assert result["actual"] == 0.75
    assert result["margin"] == pytest.approx(-0.05)


def test_evaluate_gate_lte_pass(validator):
    """Test lte comparison operator - passing case."""
    gate = QualityGate(
        metric_name="latency_ms",
        threshold=100.0,
        comparison="lte",
        required=True,
    )

    result = validator.evaluate_gate(gate, 90.0)

    assert result["passed"] is True
    assert result["margin"] == 10.0


def test_evaluate_gate_lte_fail(validator):
    """Test lte comparison operator - failing case."""
    gate = QualityGate(
        metric_name="latency_ms",
        threshold=100.0,
        comparison="lte",
        required=True,
    )

    result = validator.evaluate_gate(gate, 110.0)

    assert result["passed"] is False
    assert result["margin"] == -10.0


def test_evaluate_gate_gt_pass(validator):
    """Test gt comparison operator - passing case."""
    gate = QualityGate(
        metric_name="sharpe_ratio",
        threshold=1.5,
        comparison="gt",
        required=True,
    )

    result = validator.evaluate_gate(gate, 2.0)

    assert result["passed"] is True
    assert result["margin"] == 0.5


def test_evaluate_gate_gt_fail_equal(validator):
    """Test gt comparison operator - fails when equal."""
    gate = QualityGate(
        metric_name="sharpe_ratio",
        threshold=1.5,
        comparison="gt",
        required=True,
    )

    result = validator.evaluate_gate(gate, 1.5)

    assert result["passed"] is False


def test_evaluate_gate_lt_pass(validator):
    """Test lt comparison operator - passing case."""
    gate = QualityGate(
        metric_name="error_rate",
        threshold=0.05,
        comparison="lt",
        required=True,
    )

    result = validator.evaluate_gate(gate, 0.03)

    assert result["passed"] is True


def test_evaluate_gate_lt_fail_equal(validator):
    """Test lt comparison operator - fails when equal."""
    gate = QualityGate(
        metric_name="error_rate",
        threshold=0.05,
        comparison="lt",
        required=True,
    )

    result = validator.evaluate_gate(gate, 0.05)

    assert result["passed"] is False


def test_evaluate_gate_eq_pass(validator):
    """Test eq comparison operator - passing case."""
    gate = QualityGate(
        metric_name="target_value",
        threshold=1.0,
        comparison="eq",
        required=True,
    )

    result = validator.evaluate_gate(gate, 1.0)

    assert result["passed"] is True


def test_evaluate_gate_eq_fail(validator):
    """Test eq comparison operator - failing case."""
    gate = QualityGate(
        metric_name="target_value",
        threshold=1.0,
        comparison="eq",
        required=True,
    )

    result = validator.evaluate_gate(gate, 1.1)

    assert result["passed"] is False


def test_evaluate_gate_eq_float_precision(validator):
    """Test eq comparison with floating point precision."""
    gate = QualityGate(
        metric_name="precision",
        threshold=0.5,
        comparison="eq",
        required=True,
    )

    # Very close value (within tolerance)
    result = validator.evaluate_gate(gate, 0.5 + 1e-11)

    assert result["passed"] is True


# ========== Missing Metric Tests ==========


def test_evaluate_gate_missing_metric(validator):
    """Test evaluation when metric is missing."""
    gate = QualityGate(
        metric_name="missing_metric",
        threshold=0.8,
        comparison="gte",
        required=True,
    )

    result = validator.evaluate_gate(gate, None)

    assert result["passed"] is False
    assert result["actual"] is None
    assert result["reason"] == "metric_not_found"


# ========== Required vs Optional Gates Tests ==========


def test_validate_quality_gates_all_pass(validator):
    """Test validation when all gates pass."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
        QualityGate("precision", 0.75, "gte", required=True),
        QualityGate("recall", 0.7, "gte", required=False),
    ]

    metrics = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is True
    assert result.gates_passed == 3
    assert result.gates_failed == 0


def test_validate_quality_gates_optional_fail(validator):
    """Test validation when optional gate fails but required pass."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
        QualityGate("recall", 0.9, "gte", required=False),
    ]

    metrics = {
        "accuracy": 0.85,
        "recall": 0.75,  # Fails optional gate
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is True  # Still passes because optional
    assert result.gates_passed == 1
    assert result.gates_failed == 1


def test_validate_quality_gates_required_fail(validator):
    """Test validation when required gate fails."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
        QualityGate("precision", 0.9, "gte", required=True),
    ]

    metrics = {
        "accuracy": 0.85,
        "precision": 0.75,  # Fails required gate
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is False
    assert result.gates_passed == 1
    assert result.gates_failed == 1


def test_validate_quality_gates_missing_metric(validator):
    """Test validation when metric is missing."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
        QualityGate("missing_metric", 0.9, "gte", required=True),
    ]

    metrics = {
        "accuracy": 0.85,
        # missing_metric not provided
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is False
    assert result.gates_passed == 1
    assert result.gates_failed == 1
    assert result.gate_results["missing_metric"]["reason"] == "metric_not_found"


# ========== Gate Results Structure Tests ==========


def test_validate_quality_gates_result_structure(validator):
    """Test that validation result has correct structure."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
    ]

    metrics = {"accuracy": 0.85}

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.model_id == "model_001"
    assert "accuracy" in result.gate_results
    assert "threshold" in result.gate_results["accuracy"]
    assert "actual" in result.gate_results["accuracy"]
    assert "passed" in result.gate_results["accuracy"]
    assert "required" in result.gate_results["accuracy"]


# ========== Edge Cases ==========


def test_validate_quality_gates_empty_gates(validator):
    """Test validation with no gates."""
    gates = []
    metrics = {"accuracy": 0.85}

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is True
    assert result.gates_passed == 0
    assert result.gates_failed == 0


def test_validate_quality_gates_empty_metrics(validator):
    """Test validation with no metrics."""
    gates = [
        QualityGate("accuracy", 0.8, "gte", required=True),
    ]
    metrics = {}

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is False
    assert result.gates_passed == 0
    assert result.gates_failed == 1


def test_validate_quality_gates_all_optional_fail(validator):
    """Test validation when all gates are optional and fail."""
    gates = [
        QualityGate("accuracy", 0.9, "gte", required=False),
        QualityGate("precision", 0.9, "gte", required=False),
    ]

    metrics = {
        "accuracy": 0.7,
        "precision": 0.7,
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is True  # Passes because all optional
    assert result.gates_passed == 0
    assert result.gates_failed == 2


# ========== Margin Calculation Tests ==========


def test_margin_calculation_gte(validator):
    """Test margin calculation for gte comparison."""
    gate = QualityGate("metric", 0.8, "gte", required=True)

    # Passing case
    result = validator.evaluate_gate(gate, 0.9)
    assert result["margin"] == pytest.approx(0.1)

    # Failing case
    result = validator.evaluate_gate(gate, 0.7)
    assert result["margin"] == pytest.approx(-0.1)


def test_margin_calculation_lte(validator):
    """Test margin calculation for lte comparison."""
    gate = QualityGate("metric", 100.0, "lte", required=True)

    # Passing case
    result = validator.evaluate_gate(gate, 90.0)
    assert result["margin"] == pytest.approx(10.0)

    # Failing case
    result = validator.evaluate_gate(gate, 110.0)
    assert result["margin"] == pytest.approx(-10.0)


# ========== Multiple Models Tests ==========


def test_validate_different_models(validator):
    """Test validating multiple different models."""
    gates = [QualityGate("accuracy", 0.8, "gte", required=True)]

    # Model 1 - passes
    result1 = validator.validate_quality_gates("model_001", {"accuracy": 0.85}, gates)
    assert result1.overall_pass is True

    # Model 2 - fails
    result2 = validator.validate_quality_gates("model_002", {"accuracy": 0.75}, gates)
    assert result2.overall_pass is False

    # Results should be independent
    assert result1.model_id == "model_001"
    assert result2.model_id == "model_002"


# ========== Complex Scenarios ==========


def test_validate_complex_scenario(validator):
    """Test validation with complex mix of gates and metrics."""
    gates = [
        QualityGate("accuracy", 0.85, "gte", required=True),
        QualityGate("precision", 0.80, "gte", required=True),
        QualityGate("recall", 0.75, "gte", required=True),
        QualityGate("f1_score", 0.80, "gte", required=False),
        QualityGate("latency_ms", 100.0, "lte", required=True),
        QualityGate("memory_mb", 512.0, "lte", required=False),
    ]

    metrics = {
        "accuracy": 0.88,  # Pass
        "precision": 0.82,  # Pass
        "recall": 0.77,  # Pass
        "f1_score": 0.75,  # Fail (optional)
        "latency_ms": 95.0,  # Pass
        "memory_mb": 600.0,  # Fail (optional)
    }

    result = validator.validate_quality_gates("model_001", metrics, gates)

    assert result.overall_pass is True  # All required pass
    assert result.gates_passed == 4
    assert result.gates_failed == 2


# ========== Boundary Tests ==========


def test_evaluate_gate_boundary_gte(validator):
    """Test gte at boundary (equal value)."""
    gate = QualityGate("metric", 0.8, "gte", required=True)
    result = validator.evaluate_gate(gate, 0.8)
    assert result["passed"] is True


def test_evaluate_gate_boundary_lte(validator):
    """Test lte at boundary (equal value)."""
    gate = QualityGate("metric", 0.8, "lte", required=True)
    result = validator.evaluate_gate(gate, 0.8)
    assert result["passed"] is True


def test_evaluate_gate_zero_values(validator):
    """Test with zero values."""
    gate = QualityGate("metric", 0.0, "gte", required=True)
    result = validator.evaluate_gate(gate, 0.0)
    assert result["passed"] is True


def test_evaluate_gate_negative_values(validator):
    """Test with negative values."""
    gate = QualityGate("metric", -1.0, "gte", required=True)
    result = validator.evaluate_gate(gate, -0.5)
    assert result["passed"] is True


# ========== Validation Result Object Tests ==========


def test_validation_result_initialization():
    """Test ValidationResult initialization."""
    result = ValidationResult(model_id="test_model")

    assert result.model_id == "test_model"
    assert result.overall_pass is True
    assert result.gates_passed == 0
    assert result.gates_failed == 0
    assert result.gate_results == {}
    assert result.timestamp > 0
