#!/usr/bin/env python3

"""
Shared validation dataclasses for ML data operations.

This module contains dataclasses used across multiple store components to avoid
circular dependencies. These types are used by SchemaValidator, DataStore, and
other store components for validation and quality reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRuleType


__all__ = [
    "DataEvent",
    "QualityReport",
    "ValidationViolation",
]


# ========================================================================
# Data Operation Event Tracking
# ========================================================================


@dataclass(frozen=True)
class DataEvent:
    """
    Event tracking data operations in the store.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    operation : str
        Operation type (write_ingestion, write_features, etc.)
    source : str
        Data source (live, historical, backfill)
    run_id : str
        Processing run identifier
    ts_min : int
        Minimum timestamp in nanoseconds
    ts_max : int
        Maximum timestamp in nanoseconds
    record_count : int
        Number of records processed
    status : str
        Operation status (success, failed, partial)
    error_message : str | None
        Error message if failed
    created_at : int
        Event creation timestamp in nanoseconds
    metadata : dict[str, Any]
        Additional event metadata

    """

    event_id: str
    dataset_id: str
    instrument_id: str
    operation: str
    source: str
    run_id: str
    ts_min: int
    ts_max: int
    record_count: int
    status: str
    error_message: str | None = None
    created_at: int = field(default_factory=time.time_ns)
    metadata: dict[str, Any] = field(default_factory=dict)


# ========================================================================
# Validation Results
# ========================================================================


@dataclass(frozen=True)
class ValidationViolation:
    """
    Details of a validation rule violation.

    Attributes
    ----------
    rule_type : ValidationRuleType
        Type of validation rule violated
    field_name : str
        Field that failed validation
    severity : QualityFlag
        Severity of the violation
    violation_count : int
        Number of records with this violation
    sample_values : list[Any]
        Sample of violating values (max 5)
    description : str
        Human-readable description

    """

    rule_type: ValidationRuleType
    field_name: str
    severity: QualityFlag
    violation_count: int
    sample_values: list[Any]
    description: str


@dataclass(frozen=True)
class QualityReport:
    """
    Quality validation report for a batch of data.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier
    total_records : int
        Total number of records validated
    passed_records : int
        Number of records that passed validation
    failed_records : int
        Number of records that failed validation
    quality_score : float
        Overall quality score (0-1)
    violations : list[ValidationViolation]
        List of validation rule violations
    validation_time_ms : float
        Time taken for validation in milliseconds
    metadata : dict[str, Any]
        Additional metadata

    """

    dataset_id: str
    total_records: int
    passed_records: int
    failed_records: int
    quality_score: float
    violations: list[ValidationViolation]
    validation_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
