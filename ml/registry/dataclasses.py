#!/usr/bin/env python3

"""
Shared data structures for the ML registry.

This module contains dataclasses used across the registry system for quality validation,
canary deployments, and statistical analysis.

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
        Minimum or maximum acceptable value
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


@dataclass(frozen=True)
class CanaryConfig:
    """
    Configuration for canary deployment.

    Attributes
    ----------
    traffic_percentage : float
        Percentage of traffic to route to canary (0.0 to 100.0)
    success_metric : str
        Metric to monitor for deployment success
    baseline_threshold : float
        Minimum acceptable performance relative to baseline (0.0 to 1.0)
    monitoring_duration_hours : int
        Hours to monitor before auto-promotion
    auto_promote : bool
        Whether to automatically promote if metrics are good
    auto_rollback : bool
        Whether to automatically rollback if metrics are bad
    min_samples : int
        Minimum samples before making decisions
    error_rate_threshold : float
        Maximum acceptable error rate

    """

    traffic_percentage: float = 5.0
    success_metric: str = "accuracy"
    baseline_threshold: float = 0.95
    monitoring_duration_hours: float = 24.0
    auto_promote: bool = True
    auto_rollback: bool = True
    min_samples: int = 100
    error_rate_threshold: float = 0.05


@dataclass
class CanaryDeployment:
    """
    Tracks state of a canary deployment.

    Attributes
    ----------
    deployment_id : str
        Unique deployment identifier
    model_id : str
        Model being deployed
    target : str
        Deployment target
    config : CanaryConfig
        Deployment configuration
    baseline_model_id : Optional[str]
        ID of baseline model for comparison
    baseline_performance : Optional[float]
        Performance of current production model
    created_at : float
        Unix timestamp of deployment start
    status : str
        Current status (active, promoted, rolled_back)
    metrics : dict[str, Any]
        Collected performance metrics

    """

    deployment_id: str
    model_id: str
    target: str
    config: CanaryConfig
    baseline_model_id: str | None = None
    baseline_performance: float | None = None
    created_at: float = field(default_factory=time.time)
    status: str = "active"
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "sample_count": 0,
            "success_count": 0,
            "error_count": 0,
            "metric_sum": 0.0,
            "metric_values": [],
            "latency_values": [],
        },
    )

    def record_metric(
        self,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> None:
        """
        Record a metric observation for the canary.

        Parameters
        ----------
        metric_value : float
            Value of the success metric
        latency_ms : Optional[float]
            Response latency in milliseconds
        error_occurred : bool
            Whether an error occurred

        """
        self.metrics["sample_count"] += 1

        if error_occurred:
            self.metrics["error_count"] += 1
        else:
            self.metrics["success_count"] += 1
            self.metrics["metric_sum"] += metric_value
            self.metrics["metric_values"].append(metric_value)

        if latency_ms is not None:
            self.metrics["latency_values"].append(latency_ms)

    def should_promote(self) -> tuple[bool, str]:
        """
        Check if canary should be promoted to production.

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason)

        """
        if self.status != "active":
            return False, "not_active"

        duration_hours = (time.time() - self.created_at) / 3600
        sample_count = self.metrics["sample_count"]
        success_count = self.metrics["success_count"]
        error_count = self.metrics["error_count"]

        if sample_count < self.config.min_samples:
            return False, "insufficient_samples"

        error_rate = error_count / sample_count if sample_count > 0 else 0.0
        if error_rate > self.config.error_rate_threshold:
            return False, "high_error_rate"

        current_performance = (
            self.metrics["metric_sum"] / success_count if success_count > 0 else 0.0
        )

        if self.baseline_performance is not None:
            relative_performance = (
                current_performance / self.baseline_performance
                if self.baseline_performance > 0
                else 1.0
            )
            if relative_performance < self.config.baseline_threshold:
                return False, "performance_below_baseline"

        if duration_hours >= self.config.monitoring_duration_hours:
            return True, "monitoring_period_complete"

        return False, "monitoring_in_progress"

    def should_rollback(self) -> tuple[bool, str]:
        """
        Check if canary should be rolled back.

        Returns
        -------
        tuple[bool, str]
            (should_rollback, reason)

        """
        if self.status != "active":
            return False, "not_active"

        sample_count = self.metrics["sample_count"]

        # Need minimum samples to make decision
        if sample_count < min(self.config.min_samples, 30):
            return False, "insufficient_samples"

        error_count = self.metrics["error_count"]
        success_count = self.metrics["success_count"]

        error_rate = error_count / sample_count if sample_count > 0 else 0.0
        if error_rate > self.config.error_rate_threshold:
            return True, "high_error_rate"

        if success_count > 0:
            current_performance = self.metrics["metric_sum"] / success_count

            if self.baseline_performance is not None and self.baseline_performance > 0:
                relative_performance = current_performance / self.baseline_performance
                if relative_performance < self.config.baseline_threshold:
                    return True, "performance_degradation"

        return False, "metrics_acceptable"

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get summary of canary deployment status.

        Returns
        -------
        dict[str, Any]
            Status summary including metrics and decisions

        """
        sample_count = self.metrics["sample_count"]
        success_count = self.metrics["success_count"]
        error_count = self.metrics["error_count"]

        current_performance = (
            self.metrics["metric_sum"] / success_count if success_count > 0 else 0.0
        )

        error_rate = error_count / sample_count if sample_count > 0 else 0.0

        avg_latency = (
            sum(self.metrics["latency_values"]) / len(self.metrics["latency_values"])
            if self.metrics["latency_values"]
            else 0.0
        )

        duration_hours = (time.time() - self.created_at) / 3600

        relative_performance = 1.0
        if self.baseline_performance is not None and self.baseline_performance > 0:
            relative_performance = current_performance / self.baseline_performance

        should_promote, promote_reason = self.should_promote()
        should_rollback, rollback_reason = self.should_rollback()

        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "status": self.status,
            "duration_hours": duration_hours,
            "sample_count": sample_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "current_performance": current_performance,
            "baseline_performance": self.baseline_performance,
            "relative_performance": relative_performance,
            "average_latency_ms": avg_latency,
            "traffic_percentage": self.config.traffic_percentage,
            "should_promote": should_promote,
            "promote_reason": promote_reason,
            "should_rollback": should_rollback,
            "rollback_reason": rollback_reason,
        }


@dataclass
class RolloutPlan:
    """
    Tracks gradual rollout plan and progress.

    Attributes
    ----------
    rollout_id : str
        Unique rollout identifier
    current_model_id : str
        Currently deployed model
    new_model_id : str
        New model being rolled out
    target : str
        Deployment target
    stages : list[float]
        Traffic percentages for each stage
    stage_duration_minutes : int
        Duration of each stage
    current_stage : int
        Current stage index
    started_at : float
        When rollout started
    status : str
        Rollout status
    stage_results : list[dict[str, Any]]
        Results from each completed stage

    """

    rollout_id: str
    current_model_id: str
    new_model_id: str
    target: str
    stages: list[float]
    stage_duration_minutes: int
    current_stage: int = 0
    started_at: float = field(default_factory=time.time)
    status: str = "active"
    stage_results: list[dict[str, Any]] = field(default_factory=list)

    def get_current_traffic_split(self) -> float:
        """
        Get current traffic percentage for new model.
        """
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return 1.0  # Full deployment

    def advance_stage(self) -> bool:
        """
        Advance to next rollout stage.

        Returns
        -------
        bool
            True if advanced, False if already at final stage

        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def is_complete(self) -> bool:
        """
        Check if rollout is complete.
        """
        return self.current_stage >= len(self.stages) - 1 and self.stages[-1] == 1.0
