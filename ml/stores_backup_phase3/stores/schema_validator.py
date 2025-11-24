#!/usr/bin/env python3

"""
Schema validation for ML data contracts.

This module provides comprehensive validation of data against schema contracts,
including type checking, range validation, regex patterns, nullability, uniqueness,
monotonicity, and lateness checks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, cast

from ml._imports import HAS_PROMETHEUS
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.validation_types import QualityReport
from ml.stores.validation_types import ValidationViolation


logger = logging.getLogger(__name__)


# ========================================================================
# Prometheus Metrics (using centralized bootstrap pattern)
# ========================================================================

class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike: ...
    def inc(self, *args: object, **kwargs: object) -> None: ...


class _HistogramLike(Protocol):
    def labels(self, **kwargs: object) -> _HistogramLike: ...
    def observe(self, *args: object, **kwargs: object) -> None: ...


class _NoOpMetric:
    def labels(self, **_: object) -> _NoOpMetric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


# Declare metric variables
validation_violations_counter: Any = _NoOpMetric()
validation_duration_histogram: Any = _NoOpMetric()
quality_score_histogram: Any = _NoOpMetric()

try:
    from ml.common.metrics import quality_score_histogram as _qsh
    from ml.common.metrics import validation_duration_histogram as _vd
    from ml.common.metrics import validation_violations_counter as _vvc

    quality_score_histogram = _qsh
    validation_duration_histogram = _vd
    validation_violations_counter = _vvc
except Exception:
    logger.debug("Metrics import failed; using no-op counters/histograms", exc_info=True)


# ========================================================================
# Protocol Definition
# ========================================================================


class SchemaValidatorProtocol(Protocol):
    """Protocol for schema validation operations."""

    def validate_batch(
        self,
        data: DataFrameLike,
        manifest: DatasetManifest,
        contract: DataContract,
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against contract.

        Parameters
        ----------
        data : DataFrameLike
            Data to validate
        manifest : DatasetManifest
            Dataset manifest with schema
        contract : DataContract
            Data contract with validation rules
        strict_mode : bool
            If True, treat warnings as failures

        Returns
        -------
        QualityReport
            Validation results with quality score and violations
        """
        ...

    def apply_validation_rule(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Apply a single validation rule to data.

        Parameters
        ----------
        rule : ValidationRule
            Validation rule to apply
        data_frame : object
            Data to validate
        manifest : DatasetManifest
            Dataset manifest

        Returns
        -------
        ValidationViolation | None
            Violation details if rule failed, None otherwise
        """
        ...


# ========================================================================
# SchemaValidator Implementation
# ========================================================================


class SchemaValidator:
    """
    Validates data against schema contracts.

    Performs comprehensive validation including type checking, range validation,
    regex patterns, nullability, uniqueness, monotonicity, and lateness checks.

    This component is extracted from the DataStore god class to provide focused,
    testable schema validation functionality.
    """

    def __init__(self) -> None:
        """Initialize schema validator."""
        logger.debug("Initialized SchemaValidator")

    def validate_batch(
        self,
        data: DataFrameLike,
        manifest: DatasetManifest,
        contract: DataContract,
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against contract.

        Parameters
        ----------
        data : DataFrameLike
            Data to validate (pandas/polars DataFrame or list of dicts)
        manifest : DatasetManifest
            Dataset manifest with schema
        contract : DataContract
            Data contract with validation rules
        strict_mode : bool
            If True, treat warnings as failures

        Returns
        -------
        QualityReport
            Validation results with quality score and violations
        """
        start_time = time.perf_counter()

        # Import locally to avoid circular dependency
        data_frame = data

        data_frame_any = cast(Any, data_frame)

        # Get total record count
        if hasattr(data_frame_any, "__len__"):
            total_records = len(data_frame_any)
        else:
            total_records = 0

        # Apply all validation rules
        violations: list[ValidationViolation] = []
        for rule in contract.validation_rules:
            violation = self.apply_validation_rule(rule, data_frame_any, manifest)
            if violation:
                violations.append(violation)

        # Calculate quality metrics
        failed_record_estimate = sum(
            v.violation_count
            for v in violations
            if v.severity == QualityFlag.FAIL
            or (strict_mode and v.severity == QualityFlag.WARN)
        )
        failed_records = min(total_records, failed_record_estimate)
        passed_records = max(0, total_records - failed_records)

        # Calculate quality score (simple ratio)
        if total_records > 0:
            quality_score = passed_records / total_records
        else:
            quality_score = 1.0

        validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Evaluate quality thresholds (null rate, etc.)
        if contract.quality_thresholds:
            try:
                from ml.common.dataframe_utils import total_nulls as _total_nulls

                columns_obj = getattr(data_frame_any, "columns", None)
                column_count = len(columns_obj) if columns_obj is not None else 0
                null_count_total: int = _total_nulls(data_frame_any)
                base_count = total_records * column_count if total_records > 0 else 0
                null_rate = float(null_count_total) / float(base_count) if base_count else 0.0

                threshold = contract.quality_thresholds.get("null_rate")
                if threshold is not None and null_rate > threshold:
                    violations.append(
                        ValidationViolation(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name="*",
                            severity=QualityFlag.WARN,
                            violation_count=max(1, int(null_rate * total_records)) if total_records else 1,
                            sample_values=[],
                            description=f"Null rate {null_rate:.2%} exceeds threshold",
                        ),
                    )
            except Exception:  # pragma: no cover - defensive
                logger.debug("quality_threshold_evaluation_failed", exc_info=True)

        # Record metrics
        if HAS_PROMETHEUS:
            logger.info(
                "recording_schema_validation_metrics",
                extra={
                    "dataset_id": manifest.dataset_id,
                    "quality_metric": type(quality_score_histogram).__name__,
                    "duration_metric": type(validation_duration_histogram).__name__,
                    "violation_count": len(violations),
                },
            )
            quality_score_histogram.labels(dataset_id=manifest.dataset_id).observe(quality_score)
            validation_duration_histogram.labels(dataset_id=manifest.dataset_id).observe(
                validation_time_ms / 1000
            )

            for violation in violations:
                validation_violations_counter.labels(
                    dataset_id=manifest.dataset_id,
                    rule_type=violation.rule_type.value,
                    severity=violation.severity.value,
                ).inc()
        else:
            logger.info(
                "schema_validation_metrics_not_recorded",
                extra={
                    "dataset_id": manifest.dataset_id,
                    "has_prometheus": HAS_PROMETHEUS,
                },
            )

        report_metadata = {
            "contract_version": contract.version,
            "enforcement_mode": contract.enforcement_mode,
            "strict_mode": strict_mode,
            "rules_evaluated": len(contract.validation_rules),
        }

        return QualityReport(
            dataset_id=manifest.dataset_id,
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            quality_score=quality_score,
            violations=violations,
            validation_time_ms=validation_time_ms,
            metadata=report_metadata,
        )

    def apply_validation_rule(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Apply a single validation rule to data.

        Parameters
        ----------
        rule : ValidationRule
            Validation rule to apply
        data_frame : object
            Data to validate
        manifest : DatasetManifest
            Dataset manifest

        Returns
        -------
        ValidationViolation | None
            Violation details if rule failed, None otherwise
        """
        try:
            if rule.rule_type == ValidationRuleType.TYPE_CHECK:
                return self._validate_types(rule, data_frame, manifest)
            elif rule.rule_type == ValidationRuleType.RANGE:
                return self._validate_range(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.UNIQUENESS:
                return self._validate_uniqueness(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.MONOTONICITY:
                return self._validate_monotonicity(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.NULLABILITY:
                return self._validate_nullability(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.LATENESS:
                return self._validate_lateness(rule, data_frame, manifest)
            elif rule.rule_type == ValidationRuleType.REGEX:
                return self._validate_regex(rule, data_frame)
            else:
                logger.warning("Unknown validation rule type: %s", rule.rule_type)
                return None
        except Exception as exc:
            logger.error(
                "Error applying validation rule %s",
                rule.rule_type,
                exc_info=True,
            )
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=QualityFlag.WARN,
                violation_count=1,
                sample_values=[],
                description=f"Validation error: {exc}",
            )

    def _validate_types(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """Validate data types match schema."""
        data_frame_any = cast(Any, data_frame)
        violations = 0
        sample_values = []

        # Check each column in schema
        for col_name, expected_type in manifest.schema.items():
            if hasattr(data_frame_any, "columns") and col_name in data_frame_any.columns:
                # Get actual type
                actual_type = str(data_frame_any[col_name].dtype)

                # Simple type checking
                if not self._types_compatible(actual_type, expected_type):
                    violations += 1
                    sample_values.append(f"{col_name}: {actual_type} != {expected_type}")

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"Type mismatches found in {violations} columns",
            )

        return None

    def _validate_regex(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """Validate a column against a regex pattern."""
        import re as _re


        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters
        pattern = str(params.get("pattern", ""))

        if not pattern or not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        try:
            regex = _re.compile(pattern)
        except Exception:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=QualityFlag.WARN,
                violation_count=1,
                sample_values=[pattern],
                description="Invalid regex pattern",
            )

        col = data_frame_any[field_name]
        violations = 0
        samples: list[str] = []

        # Polars series path
        if hasattr(col, "__module__") and "polars" in str(col.__module__):
            try:
                mismatches = [not bool(regex.match(str(v))) for v in col.to_list()]
                violations = int(sum(1 for b in mismatches if b))
                if violations:
                    for v, bad in zip(col.to_list(), mismatches):
                        if bad:
                            samples.append(str(v))
                        if len(samples) >= 5:
                            break
            except Exception:
                violations = 0
        else:
            # pandas/iterable path
            try:
                if hasattr(col, "tolist"):
                    values_iter = cast(Any, col).tolist()
                else:
                    values_iter = list(cast(Any, col))
                for v in values_iter:
                    if not bool(regex.match(str(v))):
                        violations += 1
                        if len(samples) < 5:
                            samples.append(str(v))
            except Exception:
                violations = 0

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=samples,
                description=f"{field_name} failed regex pattern",
            )
        return None

    def _validate_range(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """Validate values are within specified range."""
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        violations = 0
        sample_values = []
        col = data_frame_any[field_name]

        # Check min
        if "min" in params:
            min_val = params["min"]
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                below_min = col < min_val
                violations += below_min.sum()
                if violations > 0:
                    violating = col.filter(below_min)
                    sample_values.extend(violating.head(5).to_list())
            else:
                try:
                    below_min = col < min_val
                    if hasattr(below_min, "sum"):
                        violations += below_min.sum()
                        if violations > 0:
                            violating = col[below_min]
                            if hasattr(violating, "head"):
                                sample_values.extend(violating.head(5).to_list())
                except Exception:
                    logger.debug("Failed to compute min-bound violations sample", exc_info=True)

        # Check max
        if "max" in params:
            max_val = params["max"]
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                above_max = col > max_val
                violations += above_max.sum()
                if violations > 0 and len(sample_values) < 5:
                    violating = col.filter(above_max)
                    sample_values.extend(violating.head(5 - len(sample_values)).to_list())
            else:
                try:
                    above_max = col > max_val
                    if hasattr(above_max, "sum"):
                        violations += above_max.sum()
                        if violations > 0 and len(sample_values) < 5:
                            violating = col[above_max]
                            if hasattr(violating, "head"):
                                sample_values.extend(violating.head(5 - len(sample_values)).to_list())
                except Exception:
                    logger.debug("Failed to compute max-bound violations sample", exc_info=True)

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"{field_name} values out of range [{params.get('min', '-inf')}, {params.get('max', 'inf')}]",
            )

        return None

    def _validate_uniqueness(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """Validate uniqueness constraints."""
        field_name = rule.field_name
        data_frame_any = cast(Any, data_frame)

        if not hasattr(data_frame_any, "columns"):
            return None

        # Handle composite keys
        if "," in field_name:
            key_fields = [f.strip() for f in field_name.split(",")]
            if all(f in data_frame_any.columns for f in key_fields):
                if hasattr(data_frame_any, "is_duplicated"):
                    duplicates = data_frame_any.select(key_fields).is_duplicated()
                    duplicate_count = duplicates.sum()
                elif hasattr(data_frame_any, "duplicated"):
                    duplicates = data_frame_any.duplicated(subset=key_fields)
                    duplicate_count = duplicates.sum()
                else:
                    duplicate_count = 0

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=[],
                        description=f"Duplicate values found for composite key {field_name}",
                    )
        else:
            # Single field uniqueness
            if field_name in data_frame_any.columns:
                if hasattr(data_frame_any, "is_duplicated"):
                    duplicates = data_frame_any.select([field_name]).is_duplicated()
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        duplicate_vals = data_frame_any[field_name].filter(duplicates)
                        sample_values = duplicate_vals.head(5).to_list()
                elif hasattr(data_frame_any, "duplicated"):
                    duplicates = data_frame_any.duplicated(subset=[field_name])
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        duplicate_vals = data_frame_any[field_name][duplicates]
                        if hasattr(duplicate_vals, "head"):
                            sample_values = duplicate_vals.head(5).to_list()
                else:
                    duplicate_count = 0
                    sample_values = []

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=sample_values,
                        description=f"Duplicate values found in {field_name}",
                    )

        return None

    def _validate_monotonicity(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """Validate monotonic sequences (e.g., timestamps)."""
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        col = data_frame_any[field_name]
        direction = params.get("direction", "increasing")
        strict = params.get("strict", True)
        violations = 0

        # Handle both Polars and pandas
        if hasattr(col, "__module__") and "polars" in str(col.__module__):
            diffs = col.diff()
            if direction == "increasing":
                violations = (diffs.drop_nulls() <= 0).sum() if strict else (diffs.drop_nulls() < 0).sum()
            else:  # decreasing
                violations = (diffs.drop_nulls() >= 0).sum() if strict else (diffs.drop_nulls() > 0).sum()
        else:
            if hasattr(col, "diff"):
                diffs = col.diff()
                if direction == "increasing":
                    violations = (cast(Any, diffs.dropna()) <= 0).sum() if strict else (cast(Any, diffs.dropna()) < 0).sum()
                else:
                    violations = (cast(Any, diffs.dropna()) >= 0).sum() if strict else (cast(Any, diffs.dropna()) > 0).sum()

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=[],
                description=f"{field_name} is not {'strictly ' if strict else ''}{direction}",
            )

        return None

    def _validate_nullability(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """Validate null value constraints."""
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns"):
            return None

        nullable = params.get("nullable", True)

        if not nullable:
            if field_name == "*":
                from ml.common.dataframe_utils import column_nulls as _col_nulls
                from ml.common.dataframe_utils import total_nulls as _total

                total_nulls = _total(data_frame_any)
                fields_with_nulls = [col for col in data_frame_any.columns if _col_nulls(data_frame_any, col) > 0]

                if total_nulls > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=total_nulls,
                        sample_values=fields_with_nulls[:5],
                        description=f"Null values found in {len(fields_with_nulls)} fields",
                    )
            else:
                if field_name in data_frame_any.columns:
                    from ml.common.dataframe_utils import column_nulls as _col_nulls

                    null_count = _col_nulls(data_frame_any, field_name)

                    if null_count > 0:
                        return ValidationViolation(
                            rule_type=rule.rule_type,
                            field_name=field_name,
                            severity=rule.severity,
                            violation_count=null_count,
                            sample_values=[],
                            description=f"{field_name} contains {null_count} null values",
                        )

        return None

    def _validate_lateness(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """Validate data freshness/lateness."""
        data_frame_any = cast(Any, data_frame)
        params = rule.parameters
        max_lateness_ns = params.get("max_lateness_ns", 300_000_000_000)  # Default 5 minutes

        ts_field = manifest.ts_field
        if not hasattr(data_frame_any, "columns") or ts_field not in data_frame_any.columns:
            return None

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        current_ns = _sanitize(int(time.time_ns()), context="schema_validator._validate_lateness:now")
        latest_ts = _sanitize(
            int(cast(Any, data_frame)[ts_field].max()),
            context="schema_validator._validate_lateness:latest",
        )

        lateness_ns = current_ns - latest_ts

        if lateness_ns > max_lateness_ns:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=ts_field,
                severity=rule.severity,
                violation_count=1,
                sample_values=[f"Lateness: {lateness_ns / 1e9:.1f} seconds"],
                description=f"Data is {lateness_ns / 1e9:.1f} seconds late",
            )

        return None

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_map = {
            "int64": ["int", "int64", "i8", "Int64"],
            "float64": ["float", "float64", "f8", "Float64"],
            "str": ["str", "string", "object", "Utf8"],
            "bool": ["bool", "boolean", "Boolean"],
        }

        for expected_base, compatible_types in type_map.items():
            if expected in compatible_types:
                return any(t in actual.lower() for t in compatible_types)

        return actual.lower() == expected.lower()

    @staticmethod
    def format_violations(violations: list[ValidationViolation]) -> str:
        """Format violations for logging."""
        if not violations:
            return "None"

        parts = []
        for v in violations[:3]:
            parts.append(f"{v.field_name}: {v.description} ({v.violation_count} records)")

        if len(violations) > 3:
            parts.append(f"... and {len(violations) - 3} more")

        return "; ".join(parts)

    def enforce_quality_report(
        self,
        *,
        dataset_id: str,
        contract: DataContract,
        quality_report: QualityReport,
        fail_on_validation_error: bool = True,
    ) -> None:
        """
        Apply contract enforcement logic for point-in-time writes.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        contract : DataContract
            Data contract with enforcement mode
        quality_report : QualityReport
            Quality report from validation
        fail_on_validation_error : bool
            If True, raise on validation errors

        Raises
        ------
        ValueError
            If validation fails and enforcement mode requires failure
        """
        from ml._imports import HAS_PROMETHEUS

        if quality_report.quality_score >= 1.0:
            return

        violations_str = self.format_violations(quality_report.violations)
        critical = [v for v in quality_report.violations if v.severity == QualityFlag.FAIL]

        if critical and contract.enforcement_mode != "monitor_only":
            if HAS_PROMETHEUS:
                try:
                    from ml.common.metrics import write_rejection_counter

                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="validation_failed",
                    ).inc()
                except Exception as metric_exc:
                    logger.debug(
                        "schema_validator.metric_increment_failed dataset_id=%s reason=%s",
                        dataset_id,
                        "validation_failed",
                        exc_info=True,
                        extra={"error": repr(metric_exc)},
                    )
            raise ValueError(
                f"Data validation failed for {dataset_id} (fail-closed). "
                f"Quality score: {quality_report.quality_score:.2f}. "
                f"Critical violations: {len(critical)}. "
                f"Details: {violations_str}",
            )

        if fail_on_validation_error and contract.enforcement_mode == "strict":
            if HAS_PROMETHEUS:
                try:
                    from ml.common.metrics import write_rejection_counter

                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="strict_mode_violation",
                    ).inc()
                except Exception as metric_exc:
                    logger.debug(
                        "schema_validator.metric_increment_failed dataset_id=%s reason=%s",
                        dataset_id,
                        "strict_mode_violation",
                        exc_info=True,
                        extra={"error": repr(metric_exc)},
                    )
            raise ValueError(
                f"Data validation failed for {dataset_id} (strict mode). "
                f"Quality score: {quality_report.quality_score:.2f}. "
                f"Violations: {violations_str}",
            )

        if contract.enforcement_mode == "lenient":
            logger.warning(
                "Data validation warnings for %s (lenient mode): %s",
                dataset_id,
                violations_str,
            )
        else:  # monitor_only or other advisory modes
            logger.info(
                "Data validation issues for %s (monitor-only): %s",
                dataset_id,
                violations_str,
            )
