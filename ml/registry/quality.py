#!/usr/bin/env python3

"""
Model quality validation and gates for deployment readiness.

This module provides quality gate validation to ensure models meet
minimum standards before deployment to production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class QualityGate:
    """
    Defines a quality threshold that must be met.

    Attributes
    ----------
    metric_name : str
        Name of the metric to check
    threshold : float
        Minimum acceptable value
    comparison : str
        Comparison operator ('gte', 'lte', 'eq', 'gt', 'lt')
    required : bool
        Whether this gate must pass for overall validation
    """

    metric_name: str
    threshold: float
    comparison: str = "gte"  # greater than or equal
    required: bool = True


@dataclass
class ValidationResult:
    """
    Results from quality gate validation.

    Attributes
    ----------
    model_id : str
        Model being validated
    timestamp : float
        When validation occurred
    overall_pass : bool
        Whether all required gates passed
    gates_passed : int
        Number of gates that passed
    gates_failed : int
        Number of gates that failed
    gate_results : dict[str, dict[str, Any]]
        Detailed results for each gate
    """

    model_id: str
    timestamp: float = field(default_factory=time.time)
    overall_pass: bool = True
    gates_passed: int = 0
    gates_failed: int = 0
    gate_results: dict[str, dict[str, Any]] = field(default_factory=dict)


class ModelQualityValidator:
    """
    Validates models against defined quality gates.

    This validator ensures models meet minimum quality standards
    before being deployed to production environments.
    """

    def __init__(self, default_gates: list[QualityGate] | None = None) -> None:
        """
        Initialize quality validator.

        Parameters
        ----------
        default_gates : Optional[list[QualityGate]]
            Default quality gates to apply to all models
        """
        self.default_gates = default_gates or [
            QualityGate("accuracy", 0.8, "gte", required=True),
            QualityGate("precision", 0.75, "gte", required=True),
            QualityGate("recall", 0.75, "gte", required=True),
            QualityGate("latency_p99_ms", 100, "lte", required=True),
            QualityGate("memory_mb", 1000, "lte", required=False),
        ]

    def validate(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate] | None = None,
    ) -> ValidationResult:
        """
        Validate model against quality gates.

        Parameters
        ----------
        model_id : str
            Model identifier
        metrics : dict[str, float]
            Model metrics to validate
        gates : Optional[list[QualityGate]]
            Quality gates to use (defaults to default_gates)

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status
        """
        gates = gates or self.default_gates
        result = ValidationResult(model_id=model_id)

        for gate in gates:
            gate_result = self._evaluate_gate(gate, metrics.get(gate.metric_name))

            if gate_result["passed"]:
                result.gates_passed += 1
            else:
                result.gates_failed += 1
                if gate.required:
                    result.overall_pass = False

            result.gate_results[gate.metric_name] = gate_result

        return result

    def _evaluate_gate(
        self,
        gate: QualityGate,
        actual_value: float | None,
    ) -> dict[str, Any]:
        """
        Evaluate a single quality gate.

        Parameters
        ----------
        gate : QualityGate
            Gate to evaluate
        actual_value : Optional[float]
            Actual metric value

        Returns
        -------
        dict[str, Any]
            Gate evaluation result
        """
        if actual_value is None:
            return {
                "threshold": gate.threshold,
                "actual": None,
                "passed": False,
                "required": gate.required,
                "reason": "metric_not_found",
            }

        # Perform comparison
        passed = False
        if gate.comparison == "gte":
            passed = actual_value >= gate.threshold
        elif gate.comparison == "lte":
            passed = actual_value <= gate.threshold
        elif gate.comparison == "gt":
            passed = actual_value > gate.threshold
        elif gate.comparison == "lt":
            passed = actual_value < gate.threshold
        elif gate.comparison == "eq":
            passed = abs(actual_value - gate.threshold) < 1e-10

        return {
            "threshold": gate.threshold,
            "actual": actual_value,
            "passed": passed,
            "required": gate.required,
            "comparison": gate.comparison,
            "margin": actual_value - gate.threshold if gate.comparison in ["gte", "gt"] else gate.threshold - actual_value,
        }

    def create_deployment_gates(self) -> list[QualityGate]:
        """
        Create standard gates for production deployment.

        Returns
        -------
        list[QualityGate]
            Production-ready quality gates
        """
        return [
            # Performance gates
            QualityGate("accuracy", 0.85, "gte", required=True),
            QualityGate("precision", 0.80, "gte", required=True),
            QualityGate("recall", 0.80, "gte", required=True),
            QualityGate("f1_score", 0.80, "gte", required=True),

            # Latency gates
            QualityGate("latency_p50_ms", 10, "lte", required=True),
            QualityGate("latency_p99_ms", 50, "lte", required=True),
            QualityGate("latency_p999_ms", 100, "lte", required=False),

            # Resource gates
            QualityGate("memory_mb", 500, "lte", required=True),
            QualityGate("model_size_mb", 100, "lte", required=False),

            # Stability gates
            QualityGate("error_rate", 0.01, "lte", required=True),
            QualityGate("feature_drift_score", 0.1, "lte", required=False),
        ]

    def create_canary_gates(self) -> list[QualityGate]:
        """
        Create relaxed gates for canary deployments.

        Returns
        -------
        list[QualityGate]
            Canary deployment quality gates
        """
        return [
            # Relaxed performance gates for canary
            QualityGate("accuracy", 0.75, "gte", required=True),
            QualityGate("error_rate", 0.05, "lte", required=True),
            QualityGate("latency_p99_ms", 200, "lte", required=True),
        ]
