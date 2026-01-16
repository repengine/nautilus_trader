#!/usr/bin/env python3

"""
Schema validation component for DataStore.

Extracted from DataStore (Phase 2.4.1). Provides preflight checks, batch validation, and
individual validation rules.

All methods are COLD path (not performance-critical).

"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, cast

from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.utils import compute_dataset_schema_hash
from ml.stores.validation_types import QualityReport
from ml.stores.validation_types import ValidationViolation


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol

logger = logging.getLogger(__name__)


__all__ = [
    "QualityReport",
    "SchemaValidatorComponent",
    "ValidationViolation",
    "quality_score_histogram",
]


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
validation_violations_counter = get_counter(
    "ml_validation_violations_total",
    "Total number of validation violations",
    labelnames=["dataset_id", "rule_type", "severity"],
)
schema_mismatch_counter = get_counter(
    "ml_schema_mismatch_total",
    "Total number of schema mismatches detected",
    labelnames=["dataset", "mismatch_type"],
)
quality_score_histogram = get_histogram(
    "ml_quality_score",
    "Distribution of data quality scores",
    labelnames=["dataset_id"],
)


# =========================================================================
# SchemaValidatorComponent
# =========================================================================


class SchemaValidatorComponent:
    """
    Schema validation, preflight checks, and contract enforcement.

    Extracted from DataStore (Phase 2.4.1).
    All methods are COLD path (not performance-critical).

    Provides:
    - Pre-write schema validation (preflight checks)
    - Batch validation against data contracts
    - Individual validation rules (types, ranges, uniqueness, etc.)

    Example
    -------
    >>> from ml.stores.common.schema_validator import SchemaValidatorComponent
    >>> validator = SchemaValidatorComponent(data_registry=registry)
    >>> success, error, details = validator.preflight_check(
    ...     dataset_id="bars_eurusd_1m",
    ...     data=df,
    ... )
    >>> if not success:
    ...     print(f"Validation failed: {error}")

    """

    def __init__(
        self,
        data_registry: RegistryProtocol,
        *,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
        migration_state: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize schema validator with registry dependency.

        Args:
            data_registry: Data registry for manifest/contract lookup
            allow_schema_migration: Whether to allow schema migration windows
            schema_migration_window_hours: Duration of migration window in hours
            migration_state: Optional shared migration state across components

        """
        self._registry = data_registry
        self._allow_schema_migration = allow_schema_migration
        self._schema_migration_window_hours = schema_migration_window_hours

        # Caches for manifests and contracts
        self._manifest_cache: dict[str, DatasetManifest] = {}
        self._contract_cache: dict[str, DataContract] = {}
        self._schema_migration_state = migration_state if migration_state is not None else {}

    # =========================================================================
    # Public API
    # =========================================================================

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        EXTRACTED FROM: ml/stores/data_store_facade.py:971

        This method checks that the data conforms to the expected schema
        without actually writing anything. It validates column names,
        data types, required fields, and schema hash compatibility.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict: If True, require exact schema match. If False, allow subset.

        Returns:
            (success, error_message, validation_details)

        Example:
            >>> success, error, details = validator.preflight_check(
            ...     "bars_eurusd_1m",
            ...     data_frame,
            ... )
            >>> if not success:
            ...     print(f"Preflight check failed: {error}")
            ...     print(f"Details: {details}")

        """
        try:
            # Get manifest and contract
            manifest = self._get_manifest(dataset_id)

            # Convert to DataFrame for validation
            data_frame_obj = self._to_dataframe(data)
            data_frame = cast(DataFrameLike, data_frame_obj)
            data_frame_any = cast(Any, data_frame)

            validation_details: dict[str, Any] = {
                "dataset_id": dataset_id,
                "expected_schema_hash": manifest.schema_hash,
                "checks_performed": [],
                "warnings": [],
            }

            # Empty data shortcut: accept empty DataFrames as valid shape
            try:
                if hasattr(data_frame_any, "__len__") and len(data_frame_any) == 0:
                    validation_details["empty_data"] = True
                    validation_details["preflight_passed"] = True
                    return True, None, validation_details
            except Exception as exc:
                # Continue with normal checks if len() is not supported
                logger.debug(
                    "len() check failed during preflight (ignored): %s",
                    exc,
                    exc_info=True,
                )

            # Check 1: Required columns present
            if hasattr(data_frame, "columns"):
                actual_columns = set(data_frame.columns)
                expected_columns = set(manifest.schema.keys())

                missing_columns = expected_columns - actual_columns
                extra_columns = actual_columns - expected_columns

                validation_details["actual_columns"] = list(actual_columns)
                validation_details["expected_columns"] = list(expected_columns)
                validation_details["checks_performed"].append("column_presence")

                if missing_columns:
                    error_msg = f"Missing required columns: {missing_columns}"
                    validation_details["missing_columns"] = list(missing_columns)
                    if strict or any(col in manifest.primary_keys for col in missing_columns):
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

                if extra_columns and strict:
                    error_msg = f"Unexpected columns: {extra_columns}"
                    validation_details["extra_columns"] = list(extra_columns)
                    # Treat as schema hash mismatch for metrics purposes
                    if HAS_PROMETHEUS:
                        schema_mismatch_counter.labels(
                            dataset=dataset_id,
                            mismatch_type="hash_mismatch",
                        ).inc()
                    return False, error_msg, validation_details
                elif extra_columns:
                    validation_details["warnings"].append(
                        f"Extra columns will be ignored: {extra_columns}",
                    )

            # Check 2: Data types compatibility
            type_mismatches: list[dict[str, str]] = []
            if hasattr(data_frame, "columns"):
                validation_details["checks_performed"].append("type_compatibility")
                for col_name, expected_type in manifest.schema.items():
                    if col_name in data_frame.columns:
                        actual_type = str(data_frame[col_name].dtype)
                        if not self._types_compatible(actual_type, expected_type):
                            type_mismatches.append(
                                {
                                    "column": col_name,
                                    "expected": expected_type,
                                    "actual": actual_type,
                                },
                            )

            if type_mismatches:
                validation_details["type_mismatches"] = type_mismatches
                if strict:
                    error_msg = f"Type mismatches found: {type_mismatches}"
                    return False, error_msg, validation_details
                else:
                    validation_details["warnings"].append(
                        f"Type coercion will be attempted for {len(type_mismatches)} columns",
                    )

            # Check 3: Schema hash compatibility
            actual_schema_hash = self._compute_schema_hash(data_frame, manifest)
            validation_details["actual_schema_hash"] = actual_schema_hash
            validation_details["checks_performed"].append("schema_hash")

            if actual_schema_hash != manifest.schema_hash:
                # Check if we're in a migration window
                if self._is_in_migration_window(dataset_id):
                    validation_details["migration_mode"] = True
                    validation_details["warnings"].append(
                        "Schema migration in progress - dual-write enabled",
                    )
                else:
                    error_msg = (
                        f"Schema hash mismatch. Expected: {manifest.schema_hash}, "
                        f"Got: {actual_schema_hash}. Version bump required."
                    )
                    # Record schema mismatch metric
                    if HAS_PROMETHEUS:
                        schema_mismatch_counter.labels(
                            dataset=dataset_id,
                            mismatch_type="hash_mismatch",
                        ).inc()

                    if strict:
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

            # Check 4: Primary key fields present and not null
            if manifest.primary_keys:
                validation_details["checks_performed"].append("primary_keys")
                for pk_field in manifest.primary_keys:
                    if hasattr(data_frame_any, "columns") and pk_field in data_frame_any.columns:
                        # Handle both Polars and pandas
                        if hasattr(data_frame_any[pk_field], "is_null"):
                            # Polars
                            null_count = data_frame_any[pk_field].is_null().sum()
                        elif hasattr(data_frame_any[pk_field], "isna"):
                            # pandas
                            null_count = data_frame_any[pk_field].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Primary key field '{pk_field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            # Check 5: Required fields (from constraints)
            if manifest.constraints and "nullability" in manifest.constraints:
                validation_details["checks_performed"].append("required_fields")
                for field_name, nullable in manifest.constraints["nullability"].items():
                    if (
                        not nullable
                        and hasattr(data_frame, "columns")
                        and field_name in data_frame.columns
                    ):
                        # Handle both Polars and pandas
                        if hasattr(data_frame[field_name], "is_null"):
                            # Polars
                            null_count = data_frame[field_name].is_null().sum()
                        elif hasattr(data_frame[field_name], "isna"):
                            # pandas
                            null_count = data_frame[field_name].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Required field '{field_name}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            validation_details["preflight_passed"] = True
            return True, None, validation_details

        except Exception as exc:
            logger.error(
                "Preflight check failed for %s: %s",
                dataset_id,
                exc,
                exc_info=True,
            )
            return False, f"Preflight check error: {exc}", {"error": str(exc)}

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against the dataset's contract.

        EXTRACTED FROM: ml/stores/data_store_facade.py:2413

        Performs comprehensive contract validation including type checking,
        null validation, range validation, uniqueness constraints,
        monotonicity checks, and lateness validation.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict_mode: If True, apply stricter validation rules

        Returns:
            QualityReport with quality score and violations

        Example:
            >>> report = validator.validate_batch("bars_eurusd_1m", data_frame)
            >>> print(f"Quality score: {report.quality_score:.2%}")
            >>> if report.violations:
            ...     for violation in report.violations:
            ...         print(f"  - {violation.description}")

        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self._get_manifest(dataset_id)
        contract = self._get_contract(dataset_id)

        # Convert to DataFrame for validation
        data_frame_obj = self._to_dataframe(data)
        data_frame = cast(DataFrameLike, data_frame_obj)

        # Track violations
        violations: list[ValidationViolation] = []
        failed_record_indices = set()  # Track unique failed record indices
        validation_metadata = {
            "rules_evaluated": len(contract.validation_rules),
            "strict_mode": strict_mode,
        }

        # Apply each validation rule with enhanced checks
        for rule in contract.validation_rules:
            # Skip WARN-only rules in strict mode if configured
            if strict_mode and rule.severity == QualityFlag.WARN:
                # In strict mode, treat warnings as failures
                rule = ValidationRule(
                    rule_type=rule.rule_type,
                    field_name=rule.field_name,
                    parameters=rule.parameters,
                    severity=QualityFlag.FAIL,
                    description=rule.description,
                )

            violation = self._apply_validation_rule(rule, data_frame, manifest)
            if violation:
                violations.append(violation)
                # For simplicity, assume each violation affects unique records up to the total
                # In a real implementation, we'd track actual record indices
                if violation.severity == QualityFlag.FAIL:
                    # Add violation count but cap at total records to avoid overcounting
                    for i in range(min(violation.violation_count, len(data_frame))):
                        failed_record_indices.add(i)

                # Record violation metric
                if HAS_PROMETHEUS:
                    validation_violations_counter.labels(
                        dataset_id=dataset_id,
                        rule_type=str(violation.rule_type.value),
                        severity=str(violation.severity.value),
                    ).inc(violation.violation_count)

        # Calculate quality score
        total_records = len(data_frame)
        failed_records = len(failed_record_indices)
        passed_records = total_records - failed_records
        quality_score = passed_records / total_records if total_records > 0 else 0.0

        # Evaluate quality thresholds (null rate, etc.)
        if contract.quality_thresholds:
            try:
                from ml.common.dataframe_utils import total_nulls as _total_nulls

                columns_obj = getattr(data_frame, "columns", None)
                column_count = len(columns_obj) if columns_obj is not None else 0
                null_count_total = _total_nulls(data_frame)
                base_count = total_records * column_count if total_records > 0 else 0
                null_rate = float(null_count_total) / float(base_count) if base_count else 0.0

                threshold = contract.quality_thresholds.get("null_rate")
                if threshold is not None and null_rate > threshold:
                    violations.append(
                        ValidationViolation(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name="*",
                            severity=QualityFlag.WARN,
                            violation_count=max(1, int(null_rate * total_records))
                            if total_records
                            else 1,
                            sample_values=[],
                            description=f"Null rate {null_rate:.2%} exceeds threshold",
                        ),
                    )
            except Exception:
                logger.debug("quality_threshold_evaluation_failed", exc_info=True)

        # Calculate validation time
        validation_time_ms = (time.perf_counter() - start_time) * 1000.0

        if HAS_PROMETHEUS:
            quality_score_histogram.labels(dataset_id=dataset_id).observe(quality_score)

        # Build quality report
        report = QualityReport(
            dataset_id=dataset_id,
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            quality_score=quality_score,
            violations=violations,
            validation_time_ms=validation_time_ms,
            metadata={
                "contract_version": contract.version,
                "enforcement_mode": contract.enforcement_mode,
                "strict_mode": strict_mode,
                **validation_metadata,
            },
        )

        logger.debug(
            "Validated %d records for %s: quality=%.2f, violations=%d",
            total_records,
            dataset_id,
            quality_score,
            len(violations),
        )

        return report

    # =========================================================================
    # Individual Validation Rules
    # =========================================================================

    def _validate_types(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Validate data types match schema.

        EXTRACTED FROM: ml/stores/data_store_facade.py:2845

        """
        data_frame_any = cast(Any, data_frame)
        violations = 0
        sample_values = []

        # Check each column in schema
        for col_name, expected_type in manifest.schema.items():
            if hasattr(data_frame_any, "columns") and col_name in data_frame_any.columns:
                # Get actual type
                actual_type = str(data_frame_any[col_name].dtype)

                # Simple type checking (would be more sophisticated in production)
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
        """
        Validate a column against a regex pattern.

        EXTRACTED FROM: ml/stores/data_store_facade.py:2881

        """
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters
        pattern = str(params.get("pattern", ""))
        if (
            not pattern
            or not hasattr(data_frame_any, "columns")
            or field_name not in data_frame_any.columns
        ):
            return None

        try:
            regex = re.compile(pattern)
        except Exception:
            # Treat invalid regex as configuration issue (warn)
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
                    # collect first 5 mismatches
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
                # Convert to iterable of python values
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
        """
        Validate values are within specified range.

        EXTRACTED FROM: ml/stores/data_store_facade.py:2956

        """
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        violations = 0
        sample_values = []

        # Get column values
        col = data_frame_any[field_name]

        # Check min
        if "min" in params:
            min_val = params["min"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                below_min = col < min_val
                violations += below_min.sum()
                if violations > 0:
                    # Get sample of violating values
                    violating = col.filter(below_min)
                    sample_values.extend(violating.head(5).to_list())
            else:
                # pandas or raw data
                try:
                    below_min = col < min_val
                    if hasattr(below_min, "sum"):
                        violations += below_min.sum()
                        if violations > 0:
                            # Get sample of violating values
                            violating = col[below_min]
                            if hasattr(violating, "head"):
                                sample_values.extend(violating.head(5).to_list())
                except Exception:
                    logger.debug("Failed to compute min-bound violations sample", exc_info=True)

        # Check max
        if "max" in params:
            max_val = params["max"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                above_max = col > max_val
                violations += above_max.sum()
                if violations > 0 and len(sample_values) < 5:
                    # Get sample of violating values
                    violating = col.filter(above_max)
                    sample_values.extend(violating.head(5 - len(sample_values)).to_list())
            else:
                # pandas or raw data
                try:
                    above_max = col > max_val
                    if hasattr(above_max, "sum"):
                        violations += above_max.sum()
                        if violations > 0 and len(sample_values) < 5:
                            # Get sample of violating values
                            violating = col[above_max]
                            if hasattr(violating, "head"):
                                sample_values.extend(
                                    violating.head(5 - len(sample_values)).to_list(),
                                )
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
        """
        Validate uniqueness constraints.

        EXTRACTED FROM: ml/stores/data_store_facade.py:3041

        """
        field_name = rule.field_name
        data_frame_any = cast(Any, data_frame)

        if not hasattr(data_frame_any, "columns"):
            return None

        # Handle composite keys
        if "," in field_name:
            key_fields = [f.strip() for f in field_name.split(",")]
            if all(f in data_frame_any.columns for f in key_fields):
                # Check for duplicates on composite key
                if hasattr(data_frame_any, "is_duplicated"):
                    # Polars
                    duplicates = data_frame_any.select(key_fields).is_duplicated()
                    duplicate_count = duplicates.sum()
                elif hasattr(data_frame_any, "duplicated"):
                    # pandas
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
                    # Polars
                    duplicates = data_frame_any.select([field_name]).is_duplicated()
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
                        duplicate_vals = data_frame_any[field_name].filter(duplicates)
                        sample_values = duplicate_vals.head(5).to_list()
                elif hasattr(data_frame_any, "duplicated"):
                    # pandas
                    duplicates = data_frame_any.duplicated(subset=[field_name])
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
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
        """
        Validate monotonic sequences (e.g., timestamps).

        EXTRACTED FROM: ml/stores/data_store_facade.py:3118

        """
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
            # Polars Series
            diffs = col.diff()

            if direction == "increasing":
                if strict:
                    # Check for strictly increasing (diff must be > 0)
                    # First element of diff is null, so we drop it
                    violations = (diffs.drop_nulls() <= 0).sum()
                else:
                    # Check for non-decreasing (diff must be >= 0)
                    violations = (diffs.drop_nulls() < 0).sum()
            else:  # decreasing
                if strict:
                    # Check for strictly decreasing (diff must be < 0)
                    violations = (diffs.drop_nulls() >= 0).sum()
                else:
                    # Check for non-increasing (diff must be <= 0)
                    violations = (diffs.drop_nulls() > 0).sum()
        else:
            # pandas or raw data
            if hasattr(col, "diff"):
                diffs = col.diff()

                if direction == "increasing":
                    if strict:
                        # Check for strictly increasing
                        # dropna() to remove the first NaN
                        violations = (cast(Any, diffs.dropna()) <= 0).sum()
                    else:
                        # Check for non-decreasing
                        violations = (cast(Any, diffs.dropna()) < 0).sum()
                else:  # decreasing
                    if strict:
                        # Check for strictly decreasing
                        violations = (cast(Any, diffs.dropna()) >= 0).sum()
                    else:
                        # Check for non-increasing
                        violations = (cast(Any, diffs.dropna()) > 0).sum()

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
        """
        Validate null value constraints.

        EXTRACTED FROM: ml/stores/data_store_facade.py:3193

        """
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns"):
            return None

        nullable = params.get("nullable", True)

        if not nullable:
            if field_name == "*":
                # Check all fields
                from ml.common.dataframe_utils import column_nulls as _col_nulls
                from ml.common.dataframe_utils import total_nulls as _total

                total_nulls = _total(data_frame_any)
                fields_with_nulls = [
                    col for col in data_frame_any.columns if _col_nulls(data_frame_any, col) > 0
                ]

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
                # Check specific field
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
        """
        Validate data freshness/lateness.

        EXTRACTED FROM: ml/stores/data_store_facade.py:3247

        """
        data_frame_any = cast(Any, data_frame)
        params = rule.parameters
        max_lateness_ns = params.get("max_lateness_ns", 300_000_000_000)  # Default 5 minutes

        ts_field = manifest.ts_field
        if not hasattr(data_frame_any, "columns") or ts_field not in data_frame_any.columns:
            return None

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        current_ns = _sanitize(
            int(time.time_ns()), context="schema_validator._validate_lateness:now"
        )
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

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _apply_validation_rule(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Apply a single validation rule to data.
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

    def _get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get dataset manifest with caching and version check.
        """
        if dataset_id not in self._manifest_cache:
            manifest = self._registry.get_manifest(dataset_id)
            self._manifest_cache[dataset_id] = manifest
        return self._manifest_cache[dataset_id]

    def _get_contract(self, dataset_id: str) -> DataContract:
        """
        Get data contract with caching and version check.
        """
        if dataset_id not in self._contract_cache:
            contract = self._registry.get_contract(dataset_id)
            self._contract_cache[dataset_id] = contract
            logger.debug(
                "Loaded contract for %s: version=%s, mode=%s, rules=%d",
                dataset_id,
                contract.version,
                contract.enforcement_mode,
                len(contract.validation_rules),
            )
        return self._contract_cache[dataset_id]

    def _to_dataframe(
        self,
        data: DataFrameLike | list[dict[str, Any]],
    ) -> DataFrameLike | list[dict[str, Any]]:
        """
        Convert various data formats to DataFrame-like or pass-through list.
        """
        # Import here to avoid circular dependency
        from ml._imports import pl

        if not HAS_POLARS:
            # If Polars not available, work with raw data
            if isinstance(data, list):
                return data
            return data

        # If already a DataFrame, return as is
        if hasattr(data, "columns"):
            return data

        # Convert list of dicts to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Cast to DataFrameLike for strict typing compliance
            if HAS_POLARS and pl is not None:
                return cast(DataFrameLike, pl.DataFrame(data))
            return data

        # Return as is for other formats
        return data

    def _compute_schema_hash(self, data_frame: DataFrameLike, manifest: DatasetManifest) -> str:
        """
        Compute schema hash for the actual data.
        """
        data_frame_any = cast(Any, data_frame)
        if not hasattr(data_frame_any, "columns"):
            # For non-DataFrame data, use manifest hash
            return manifest.schema_hash

        # Build schema dict from actual data
        actual_schema: dict[str, str] = {}
        for col in data_frame_any.columns:
            if col in manifest.schema:
                actual_schema[col] = manifest.schema[col]
            else:
                dtype = str(data_frame_any[col].dtype)
                lower = dtype.lower()
                if "int" in lower:
                    actual_schema[col] = "int64"
                elif "float" in lower:
                    actual_schema[col] = "float64"
                elif "bool" in lower:
                    actual_schema[col] = "bool"
                else:
                    actual_schema[col] = "str"

        for manifest_column, manifest_dtype in manifest.schema.items():
            actual_schema.setdefault(manifest_column, manifest_dtype)

        return compute_dataset_schema_hash(
            schema=actual_schema,
            primary_keys=manifest.primary_keys,
            ts_field=manifest.ts_field,
            seq_field=manifest.seq_field,
            pipeline_signature=manifest.pipeline_signature,
        )

    def _is_in_migration_window(self, dataset_id: str) -> bool:
        """
        Check if dataset is in schema migration window.
        """
        if not self._allow_schema_migration:
            return False

        state = self._schema_migration_state.get(dataset_id, {})
        window_start = state.get("start_time") or state.get("window_start_ns", 0)
        if not window_start:
            return False

        window_duration_ns = self._schema_migration_window_hours * 3_600_000_000_000
        current_ns = time.time_ns()
        return bool((current_ns - window_start) < window_duration_ns)

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if actual type is compatible with expected type.
        """
        # Simple type compatibility check
        type_map = {
            "int64": ["int", "int64", "i8", "Int64"],
            "float64": ["float", "float64", "f8", "Float64"],
            "str": ["str", "string", "object", "Utf8"],
            "bool": ["bool", "boolean", "Boolean"],
        }

        for expected_base, compatible_types in type_map.items():
            if expected in compatible_types:
                return any(t in actual.lower() for t in compatible_types)

        # Default to string comparison
        return actual.lower() == expected.lower()
