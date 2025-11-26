#!/usr/bin/env python3

"""
Shared data structures for the ML registry.

This module contains dataclasses used across the registry system for quality validation,
canary deployments, statistical analysis, and data registry management.

"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


# ========================================================================
# Data Registry Types
# ========================================================================


class DatasetType(Enum):
    """
    Types of datasets tracked in the data registry.

    Attributes
    ----------
    BARS : str
        OHLCV bar data
    TRADES : str
        Individual trade ticks
    QUOTES : str
        Bid/ask quote ticks
    MBP1 : str
        Market by price depth 1 (best bid/ask)
    TBBO : str
        Top of book best bid/offer
    FEATURES : str
        Computed feature values
    PREDICTIONS : str
        Model predictions
    SIGNALS : str
        Strategy signals
    EARNINGS_ACTUALS : str
        Reported corporate earnings fundamentals
    EARNINGS_ESTIMATES : str
        Analyst consensus earnings estimates

    """

    BARS = "bars"
    TRADES = "trades"
    QUOTES = "quotes"
    MBP1 = "mbp1"
    TBBO = "tbbo"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    SIGNALS = "signals"
    EARNINGS_ACTUALS = "earnings_actuals"
    EARNINGS_ESTIMATES = "earnings_estimates"


class StorageKind(Enum):
    """
    Storage backend types for datasets.

    Attributes
    ----------
    PARQUET : str
        Apache Parquet file storage
    POSTGRES : str
        PostgreSQL database storage

    """

    PARQUET = "parquet"
    POSTGRES = "postgres"


class ValidationRuleType(Enum):
    """
    Types of validation rules for data contracts.

    Attributes
    ----------
    TYPE_CHECK : str
        Data type validation
    RANGE : str
        Value range validation
    UNIQUENESS : str
        Uniqueness constraint validation
    MONOTONICITY : str
        Monotonic sequence validation
    NULLABILITY : str
        Null value validation
    LATENESS : str
        Data freshness/lateness validation

    """

    TYPE_CHECK = "type_check"
    RANGE = "range"
    UNIQUENESS = "uniqueness"
    MONOTONICITY = "monotonicity"
    NULLABILITY = "nullability"
    LATENESS = "lateness"
    REGEX = "regex"


class QualityFlag(Enum):
    """
    Quality flags for data validation results.

    Attributes
    ----------
    PASS : str
        Validation passed
    WARN : str
        Validation passed with warnings
    FAIL : str
        Validation failed
    SKIP : str
        Validation skipped

    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class ValidationRule:
    """
    A single validation rule for data contracts.

    Attributes
    ----------
    rule_type : ValidationRuleType
        Type of validation rule
    field_name : str
        Field to validate (or "*" for all fields)
    parameters : dict[str, Any]
        Rule-specific parameters (e.g., min/max for range)
    severity : QualityFlag
        Severity if rule fails (WARN or FAIL)
    description : str
        Human-readable description of the rule

    Examples
    --------
    >>> # Range validation rule
    >>> rule = ValidationRule(
    ...     rule_type=ValidationRuleType.RANGE,
    ...     field_name="price",
    ...     parameters={"min": 0.0, "max": 1000000.0},
    ...     severity=QualityFlag.FAIL,
    ...     description="Price must be between 0 and 1M"
    ... )

    >>> # Monotonicity rule for timestamps
    >>> ts_rule = ValidationRule(
    ...     rule_type=ValidationRuleType.MONOTONICITY,
    ...     field_name="ts_event",
    ...     parameters={"direction": "increasing", "strict": True},
    ...     severity=QualityFlag.FAIL,
    ...     description="Timestamps must be strictly increasing"
    ... )

    """

    rule_type: ValidationRuleType
    field_name: str
    parameters: dict[str, Any]
    severity: QualityFlag
    description: str

    def __post_init__(self) -> None:
        """
        Validate rule configuration.
        """
        if self.severity not in (QualityFlag.WARN, QualityFlag.FAIL):
            raise ValueError(f"Invalid severity: {self.severity}. Must be WARN or FAIL")

        # Validate parameters based on rule type
        if self.rule_type == ValidationRuleType.RANGE:
            if "min" not in self.parameters and "max" not in self.parameters:
                raise ValueError("Range rule requires 'min' and/or 'max' parameters")
        elif self.rule_type == ValidationRuleType.MONOTONICITY:
            if "direction" not in self.parameters:
                raise ValueError("Monotonicity rule requires 'direction' parameter")
            if self.parameters["direction"] not in ("increasing", "decreasing"):
                raise ValueError("Monotonicity direction must be 'increasing' or 'decreasing'")
        elif self.rule_type == ValidationRuleType.LATENESS:
            if "max_lateness_ns" not in self.parameters:
                raise ValueError("Lateness rule requires 'max_lateness_ns' parameter")
        elif self.rule_type == ValidationRuleType.REGEX:
            if "pattern" not in self.parameters or not isinstance(self.parameters["pattern"], str):
                raise ValueError("Regex rule requires 'pattern' string parameter")


@dataclass(frozen=True)
class DataContract:
    """
    Data quality contract with validation rules.

    Attributes
    ----------
    contract_id : str
        Unique identifier for the contract
    dataset_id : str
        Associated dataset ID
    version : str
        Contract version (semantic versioning)
    validation_rules : list[ValidationRule]
        List of validation rules to apply
    quality_thresholds : dict[str, float]
        Quality metric thresholds (e.g., {"null_rate": 0.01})
    enforcement_mode : str
        How to enforce contract ("strict", "lenient", "monitor_only")
    created_at : int
        Creation timestamp in nanoseconds
    last_modified : int
        Last modification timestamp in nanoseconds
    metadata : dict[str, Any]
        Additional metadata

    Examples
    --------
    >>> contract = DataContract(
    ...     contract_id="bars_contract_v1",
    ...     dataset_id="bars_eurusd_1m",
    ...     version="1.0.0",
    ...     validation_rules=[
    ...         ValidationRule(
    ...             rule_type=ValidationRuleType.RANGE,
    ...             field_name="close",
    ...             parameters={"min": 0.0},
    ...             severity=QualityFlag.FAIL,
    ...             description="Close price must be positive"
    ...         )
    ...     ],
    ...     quality_thresholds={"null_rate": 0.001, "duplicate_rate": 0.0},
    ...     enforcement_mode="strict",
    ...     created_at=time.time_ns(),
    ...     last_modified=time.time_ns()
    ... )

    """

    contract_id: str
    dataset_id: str
    version: str
    validation_rules: list[ValidationRule]
    quality_thresholds: dict[str, float] = field(default_factory=dict)
    enforcement_mode: str = "strict"
    created_at: int = field(default_factory=time.time_ns)
    last_modified: int = field(default_factory=time.time_ns)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate contract configuration.
        """
        if self.enforcement_mode not in ("strict", "lenient", "monitor_only"):
            raise ValueError(
                f"Invalid enforcement_mode: {self.enforcement_mode}. "
                "Must be 'strict', 'lenient', or 'monitor_only'",
            )

        if not self.validation_rules:
            raise ValueError("DataContract must have at least one validation rule")

        # Validate quality thresholds are in valid range [0, 1]
        for metric, threshold in self.quality_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Quality threshold '{metric}' value {threshold} must be between 0 and 1",
                )


@dataclass(frozen=True)
class DatasetManifest:
    """
    Self-describing manifest for a dataset in the data registry.

    This manifest contains all metadata needed to understand, validate, and
    process a dataset within the Nautilus ML pipeline.

    Attributes
    ----------
    dataset_id : str
        Unique identifier for the dataset
    dataset_type : DatasetType
        Type of data (BARS, TRADES, QUOTES, etc.)
    storage_kind : StorageKind
        Storage backend (parquet or postgres)
    location : str
        Storage location (file path or table name)
    partitioning : dict[str, Any]
        Partitioning strategy (e.g., {"by": "ts_event", "interval": "monthly"})
    retention_days : int
        Data retention period in days
    schema : dict[str, str]
        Column names and data types
    ts_field : str
        Name of timestamp field (in nanoseconds)
    seq_field : str | None
        Optional sequence number field name
    primary_keys : list[str]
        Primary key columns
    schema_hash : str
        SHA256 hash of the schema for validation
    constraints : dict[str, Any]
        Validation constraints (ranges, nullability, etc.)
    lineage : list[str]
        Parent dataset IDs for lineage tracking
    pipeline_signature : str
        Signature of the pipeline that created this dataset
    version : str
        Dataset version (semantic versioning)
    created_at : int
        Creation timestamp in nanoseconds
    last_modified : int
        Last modification timestamp in nanoseconds
    metadata : dict[str, Any]
        Additional metadata

    Examples
    --------
    >>> manifest = DatasetManifest(
    ...     dataset_id="bars_eurusd_1m",
    ...     dataset_type=DatasetType.BARS,
    ...     storage_kind=StorageKind.PARQUET,
    ...     location="/data/bars/eurusd/1m/",
    ...     partitioning={"by": "ts_event", "interval": "daily"},
    ...     retention_days=365,
    ...     schema={
    ...         "instrument_id": "str",
    ...         "ts_event": "int64",
    ...         "ts_init": "int64",
    ...         "open": "float64",
    ...         "high": "float64",
    ...         "low": "float64",
    ...         "close": "float64",
    ...         "volume": "float64"
    ...     },
    ...     ts_field="ts_event",
    ...     seq_field=None,
    ...     primary_keys=["instrument_id", "ts_event"],
    ...     schema_hash="abc123...",
    ...     constraints={
    ...         "ranges": {
    ...             "open": {"min": 0.0},
    ...             "high": {"min": 0.0},
    ...             "low": {"min": 0.0},
    ...             "close": {"min": 0.0},
    ...             "volume": {"min": 0.0}
    ...         },
    ...         "nullability": {
    ...             "instrument_id": False,
    ...             "ts_event": False,
    ...             "ts_init": False
    ...         }
    ...     },
    ...     lineage=[],
    ...     pipeline_signature="data_scheduler_v1",
    ...     version="1.0.0",
    ...     created_at=time.time_ns(),
    ...     last_modified=time.time_ns()
    ... )

    """

    # Identity
    dataset_id: str
    dataset_type: DatasetType

    # Storage
    storage_kind: StorageKind
    location: str
    partitioning: dict[str, Any]
    retention_days: int

    # Schema
    schema: dict[str, str]
    ts_field: str
    seq_field: str | None
    primary_keys: list[str]
    schema_hash: str

    # Validation
    constraints: dict[str, Any]

    # Lineage
    lineage: list[str]
    pipeline_signature: str

    # Versioning
    version: str
    created_at: int = field(default_factory=time.time_ns)
    last_modified: int = field(default_factory=time.time_ns)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate manifest configuration.
        """
        # Validate required timestamp field is in schema
        if self.ts_field not in self.schema:
            raise ValueError(f"Timestamp field '{self.ts_field}' not found in schema")

        # Validate sequence field if provided
        if self.seq_field and self.seq_field not in self.schema:
            raise ValueError(f"Sequence field '{self.seq_field}' not found in schema")

        # Validate primary keys exist in schema
        for key in self.primary_keys:
            if key not in self.schema:
                raise ValueError(f"Primary key '{key}' not found in schema")

        # Validate Nautilus required fields for certain dataset types
        nautilus_required = ["instrument_id", "ts_event", "ts_init"]
        if self.dataset_type in (
            DatasetType.BARS,
            DatasetType.TRADES,
            DatasetType.QUOTES,
            DatasetType.MBP1,
            DatasetType.TBBO,
            DatasetType.FEATURES,
            DatasetType.PREDICTIONS,
            DatasetType.SIGNALS,
        ):
            for field in nautilus_required:
                if field not in self.schema:
                    raise ValueError(
                        f"Nautilus required field '{field}' not found in schema "
                        f"for dataset type {self.dataset_type.value}",
                    )

        # Validate retention days is positive
        if self.retention_days <= 0:
            raise ValueError(f"retention_days must be positive, got {self.retention_days}")

        # Compute schema hash if not provided
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", self.compute_schema_hash())

    def compute_schema_hash(self) -> str:
        """
        Compute SHA256 hash of the schema for validation.

        The hash includes column names, types, and primary keys to ensure
        schema compatibility across different versions.

        Returns
        -------
        str
            SHA256 hash of the schema

        """
        h = hashlib.sha256()

        # Hash schema fields in sorted order for stability
        for column_name in sorted(self.schema.keys()):
            dtype = self.schema[column_name]
            h.update(column_name.encode("utf-8"))
            h.update(b"::")
            h.update(dtype.encode("utf-8"))
            h.update(b"\n")

        # Include primary keys in hash
        h.update(b"|keys|")
        for key in sorted(self.primary_keys):
            h.update(key.encode("utf-8"))
            h.update(b",")

        # Include timestamp field
        h.update(b"|ts|")
        h.update(self.ts_field.encode("utf-8"))

        # Include sequence field if present
        if self.seq_field:
            h.update(b"|seq|")
            h.update(self.seq_field.encode("utf-8"))

        # Include pipeline signature for lineage
        h.update(b"|pipeline|")
        h.update(self.pipeline_signature.encode("utf-8"))

        return h.hexdigest()

    def is_compatible_with(self, other: DatasetManifest) -> bool:
        """
        Check if this manifest is schema-compatible with another.

        Parameters
        ----------
        other : DatasetManifest
            Another dataset manifest to compare

        Returns
        -------
        bool
            True if schemas are compatible

        """
        return self.schema_hash == other.schema_hash


# ========================================================================
# Derived Registry Records
# ========================================================================


@dataclass(slots=True, frozen=True)
class DatasetLineageRecord:
    """
    Immutable representation of a dataset lineage link.

    Attributes
    ----------
    transform_id : str
        Identifier for the transform establishing the lineage edge.
    child_dataset_id : str
        Identifier of the downstream (child) dataset.
    parent_dataset_id : str
        Identifier of the upstream (parent) dataset.
    ts_range : dict[str, int]
        Nanosecond epoch window describing the source data interval.
    parameters : dict[str, Any]
        Parameters applied during the transform.
    created_at : float
        Unix timestamp (seconds) at which the lineage entry was captured.

    """

    transform_id: str
    child_dataset_id: str
    parent_dataset_id: str
    ts_range: dict[str, int]
    parameters: dict[str, Any]
    created_at: float


# ========================================================================
# Existing Registry Types
# ========================================================================


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
