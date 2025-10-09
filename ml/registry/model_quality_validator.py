#!/usr/bin/env python3

"""
Model quality validation and gate evaluation.

This module provides comprehensive model quality validation including quality gate
evaluation with support for multiple comparison operators, required vs optional gates,
and detailed result reporting.

Extracted from ModelRegistry god class as part of Phase 2.3 refactoring.

"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import ValidationResult


logger = logging.getLogger(__name__)


class ModelQualityValidatorProtocol(Protocol):
    """Protocol for model quality validation operations."""

    def validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult: ...

    def evaluate_gate(
        self,
        gate: QualityGate,
        actual_value: float | None,
    ) -> dict[str, Any]: ...


class ModelQualityValidator:
    """
    Validates models against quality gates.

    Performs quality gate evaluation with support for multiple comparison
    operators (gte, lte, gt, lt, eq), required vs optional gates, and detailed
    result reporting.

    This component is extracted from ModelRegistry god class to provide focused,
    testable quality validation functionality.

    """

    def __init__(self) -> None:
        """Initialize quality validator."""
        logger.debug("Initialized ModelQualityValidator")

    def validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult:
        """
        Validate model metrics against quality gates.

        Parameters
        ----------
        model_id : str
            Model identifier
        metrics : dict[str, float]
            Model metrics to validate
        gates : list[QualityGate]
            Quality gates to check

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status

        """
        result = ValidationResult(model_id=model_id)

        for gate in gates:
            gate_result = self.evaluate_gate(gate, metrics.get(gate.metric_name))

            if gate_result["passed"]:
                result.gates_passed += 1
            else:
                result.gates_failed += 1
                if gate.required:
                    result.overall_pass = False

            result.gate_results[gate.metric_name] = gate_result

        logger.debug(
            "Quality validation for %s: %d/%d gates passed",
            model_id,
            result.gates_passed,
            len(gates),
        )

        return result

    def evaluate_gate(
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
        actual_value : float | None
            Actual metric value

        Returns
        -------
        dict[str, Any]
            Gate evaluation result with keys: threshold, actual, passed, required,
            comparison, margin, reason (if failed)

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

        # Calculate margin
        if gate.comparison in ["gte", "gt"]:
            margin = actual_value - gate.threshold
        else:
            margin = gate.threshold - actual_value

        return {
            "threshold": gate.threshold,
            "actual": actual_value,
            "passed": passed,
            "required": gate.required,
            "comparison": gate.comparison,
            "margin": margin,
        }
