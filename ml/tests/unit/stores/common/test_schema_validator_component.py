#!/usr/bin/env python3

"""
Unit tests for SchemaValidatorComponent (Phase 2.4.1).

Tests schema validation, preflight checks, and contract enforcement. All 15 tests from
phase_2_4_datastore_test_design.md.

"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.stores.common.schema_validator import (
    QualityReport,
    SchemaValidatorComponent,
    ValidationViolation,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_registry() -> MagicMock:
    """
    Create a mock DataRegistry for testing.
    """
    registry = MagicMock()

    # Default manifest for "bars_eurusd_1m"
    registry.get_manifest.return_value = DatasetManifest(
        dataset_id="bars_eurusd_1m",
        version="1.0.0",
        dataset_type=DatasetType.BARS,
        location="ml_bars",
        storage_kind=StorageKind.POSTGRES,
        partitioning={},
        retention_days=365,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="c04c04da8863b89636f89544935641613908654c92edd3d8bc582a113fa36b51",  # Computed from schema
        constraints={"nullability": {"instrument_id": False, "ts_event": False}},
        lineage=[],
        pipeline_signature="test_pipeline",
        metadata={},
    )

    # Default contract for "bars_eurusd_1m"
    registry.get_contract.return_value = DataContract(
        contract_id="bars_eurusd_1m_contract",
        dataset_id="bars_eurusd_1m",
        version="1.0.0",
        enforcement_mode="strict",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                parameters={},
                severity=QualityFlag.FAIL,
                description="Type compatibility check",
            ),
            ValidationRule(
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="instrument_id",
                parameters={"nullable": False},
                severity=QualityFlag.FAIL,
                description="Instrument ID cannot be null",
            ),
        ],
        quality_thresholds={"min_completeness": 0.95, "max_error_rate": 0.05},
        metadata={},
    )

    return registry


@pytest.fixture
def schema_validator(mock_registry: MagicMock) -> SchemaValidatorComponent:
    """
    Create SchemaValidatorComponent for testing.
    """
    return SchemaValidatorComponent(data_registry=mock_registry)


@pytest.fixture
def valid_ohlcv_data() -> pl.DataFrame:
    """
    Generate valid OHLCV data for testing.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": list(range(1000000000, 1000000010)),
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [1000.0] * 10,
        }
    )


@pytest.fixture
def data_with_missing_columns() -> pl.DataFrame:
    """
    Generate data missing required columns.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": list(range(1000000000, 1000000010)),
            # Missing: open, high, low, close, volume
        }
    )


@pytest.fixture
def data_with_type_mismatch() -> pl.DataFrame:
    """
    Generate data with incorrect types.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": ["not_an_int"] * 10,  # Wrong type (should be int64)
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [1000.0] * 10,
        }
    )


@pytest.fixture
def data_with_nulls() -> pl.DataFrame:
    """
    Generate data with null values in required fields.
    """
    return pl.DataFrame(
        {
            "instrument_id": [None, "EURUSD.SIM", "EURUSD.SIM", None, "EURUSD.SIM"] * 2,
            "ts_event": list(range(1000000000, 1000000010)),
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [1000.0] * 10,
        }
    )


@pytest.fixture
def data_with_duplicates() -> pl.DataFrame:
    """
    Generate data with duplicate primary keys.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": [1000000000] * 5 + list(range(1000000001, 1000000006)),  # 5 duplicates
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [1000.0] * 10,
        }
    )


@pytest.fixture
def data_with_monotonicity_violations() -> pl.DataFrame:
    """
    Generate data with non-monotonic timestamps.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": [
                1000000000,
                1000000005,
                1000000003,
                1000000007,
                1000000006,
                1000000009,
                1000000008,
                1000000011,
                1000000010,
                1000000012,
            ],  # Not strictly increasing
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [1000.0] * 10,
        }
    )


@pytest.fixture
def data_with_range_violations() -> pl.DataFrame:
    """
    Generate data with values outside valid ranges.
    """
    return pl.DataFrame(
        {
            "instrument_id": ["EURUSD.SIM"] * 10,
            "ts_event": list(range(1000000000, 1000000010)),
            "ts_init": list(range(1000000000, 1000000010)),
            "open": [1.08] * 10,
            "high": [1.09] * 10,
            "low": [1.07] * 10,
            "close": [1.085] * 10,
            "volume": [
                -100.0,
                1000.0,
                1000.0,
                -50.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ],  # Negative volumes
        }
    )


# =========================================================================
# Tests
# =========================================================================


class TestSchemaValidatorComponent:
    """
    Unit tests for SchemaValidatorComponent.
    """

    # =====================================================================
    # Test 1: test_preflight_check_valid_data
    # =====================================================================

    def test_preflight_check_valid_data(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify preflight validation passes for schema-compliant data.
        """
        success, error, details = schema_validator.preflight_check(
            dataset_id="bars_eurusd_1m",
            data=valid_ohlcv_data,
        )

        # Debug output
        if not success:
            print(f"\nERROR: {error}")
            print(f"DETAILS: {details}")

        assert success is True, f"Preflight check failed: {error}, details: {details}"
        assert error is None
        assert details["preflight_passed"] is True
        assert "column_presence" in details["checks_performed"]
        assert "type_compatibility" in details["checks_performed"]
        assert "schema_hash" in details["checks_performed"]
        assert "primary_keys" in details["checks_performed"]

    # =====================================================================
    # Test 2: test_preflight_check_missing_columns
    # =====================================================================

    def test_preflight_check_missing_columns(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_missing_columns: pl.DataFrame,
    ) -> None:
        """
        Verify preflight validation fails when required columns are missing.
        """
        success, error, details = schema_validator.preflight_check(
            dataset_id="bars_eurusd_1m",
            data=data_with_missing_columns,
            strict=True,
        )

        assert success is False
        assert error is not None
        assert "Missing required columns" in error
        assert "missing_columns" in details
        assert len(details["missing_columns"]) > 0

    # =====================================================================
    # Test 3: test_preflight_check_type_mismatch
    # =====================================================================

    def test_preflight_check_type_mismatch(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_type_mismatch: pl.DataFrame,
    ) -> None:
        """
        Verify preflight validation detects type incompatibility.
        """
        success, error, details = schema_validator.preflight_check(
            dataset_id="bars_eurusd_1m",
            data=data_with_type_mismatch,
            strict=True,
        )

        assert success is False
        assert error is not None
        assert "Type mismatches found" in error
        assert "type_mismatches" in details
        assert len(details["type_mismatches"]) > 0

    # =====================================================================
    # Test 4: test_preflight_check_schema_hash_mismatch
    # =====================================================================

    def test_preflight_check_schema_hash_mismatch(
        self,
        schema_validator: SchemaValidatorComponent,
        mock_registry: MagicMock,
    ) -> None:
        """
        Verify preflight validation detects schema drift via hash mismatch.
        """
        # Create data with extra column (causes hash mismatch)
        data_with_extra_col = pl.DataFrame(
            {
                "instrument_id": ["EURUSD.SIM"] * 10,
                "ts_event": list(range(1000000000, 1000000010)),
                "ts_init": list(range(1000000000, 1000000010)),
                "open": [1.08] * 10,
                "high": [1.09] * 10,
                "low": [1.07] * 10,
                "close": [1.085] * 10,
                "volume": [1000.0] * 10,
                "extra_field": [999] * 10,  # Extra field not in schema
            }
        )

        success, error, details = schema_validator.preflight_check(
            dataset_id="bars_eurusd_1m",
            data=data_with_extra_col,
            strict=True,
        )

        # Should fail due to extra columns in strict mode
        assert success is False
        assert error is not None
        assert "Unexpected columns" in error or "extra_field" in str(details)

    # =====================================================================
    # Test 5: test_validate_types_all_compatible
    # =====================================================================

    def test_validate_types_all_compatible(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
        mock_registry: MagicMock,
    ) -> None:
        """
        Verify type validation passes when all types are compatible.
        """
        manifest = mock_registry.get_manifest("bars_eurusd_1m")

        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type check",
        )

        violation = schema_validator._validate_types(rule, valid_ohlcv_data, manifest)

        assert violation is None

    # =====================================================================
    # Test 6: test_validate_types_incompatible
    # =====================================================================

    def test_validate_types_incompatible(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_type_mismatch: pl.DataFrame,
        mock_registry: MagicMock,
    ) -> None:
        """
        Verify type validation fails when types are incompatible.
        """
        manifest = mock_registry.get_manifest("bars_eurusd_1m")

        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type check",
        )

        violation = schema_validator._validate_types(rule, data_with_type_mismatch, manifest)

        assert violation is not None
        assert violation.violation_count > 0
        assert violation.severity == QualityFlag.FAIL

    # =====================================================================
    # Test 7: test_validate_range_within_bounds
    # =====================================================================

    def test_validate_range_within_bounds(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify range validation passes when values are within bounds.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="volume",
            parameters={"min": 0.0, "max": 10000.0},
            severity=QualityFlag.FAIL,
            description="Volume must be positive",
        )

        violation = schema_validator._validate_range(rule, valid_ohlcv_data)

        assert violation is None

    # =====================================================================
    # Test 8: test_validate_range_outside_bounds
    # =====================================================================

    def test_validate_range_outside_bounds(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_range_violations: pl.DataFrame,
    ) -> None:
        """
        Verify range validation fails when values are out of bounds.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="volume",
            parameters={"min": 0.0},
            severity=QualityFlag.FAIL,
            description="Volume must be non-negative",
        )

        violation = schema_validator._validate_range(rule, data_with_range_violations)

        assert violation is not None
        assert violation.violation_count == 2  # Two negative volumes
        assert violation.severity == QualityFlag.FAIL
        assert "out of range" in violation.description

    # =====================================================================
    # Test 9: test_validate_uniqueness_no_duplicates
    # =====================================================================

    def test_validate_uniqueness_no_duplicates(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify uniqueness validation passes when no duplicates exist.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="instrument_id,ts_event",  # Composite key
            parameters={},
            severity=QualityFlag.FAIL,
            description="Primary key must be unique",
        )

        violation = schema_validator._validate_uniqueness(rule, valid_ohlcv_data)

        assert violation is None

    # =====================================================================
    # Test 10: test_validate_uniqueness_with_duplicates
    # =====================================================================

    def test_validate_uniqueness_with_duplicates(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_duplicates: pl.DataFrame,
    ) -> None:
        """
        Verify uniqueness validation fails when duplicates exist.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="instrument_id,ts_event",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Primary key must be unique",
        )

        violation = schema_validator._validate_uniqueness(rule, data_with_duplicates)

        assert violation is not None
        assert violation.violation_count > 0
        assert violation.severity == QualityFlag.FAIL
        assert "Duplicate values found" in violation.description

    # =====================================================================
    # Test 11: test_validate_monotonicity_increasing
    # =====================================================================

    def test_validate_monotonicity_increasing(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify monotonicity validation passes for strictly increasing sequence.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="ts_event",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Timestamps must be strictly increasing",
        )

        violation = schema_validator._validate_monotonicity(rule, valid_ohlcv_data)

        assert violation is None

    # =====================================================================
    # Test 12: test_validate_monotonicity_violations
    # =====================================================================

    def test_validate_monotonicity_violations(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_monotonicity_violations: pl.DataFrame,
    ) -> None:
        """
        Verify monotonicity validation fails for non-monotonic sequence.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="ts_event",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Timestamps must be strictly increasing",
        )

        violation = schema_validator._validate_monotonicity(rule, data_with_monotonicity_violations)

        assert violation is not None
        assert violation.violation_count > 0
        assert violation.severity == QualityFlag.FAIL
        assert "not strictly increasing" in violation.description

    # =====================================================================
    # Test 13: test_validate_nullability_no_nulls
    # =====================================================================

    def test_validate_nullability_no_nulls(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify nullability validation passes when no nulls exist.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="instrument_id",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="Instrument ID cannot be null",
        )

        violation = schema_validator._validate_nullability(rule, valid_ohlcv_data)

        assert violation is None

    # =====================================================================
    # Test 14: test_validate_nullability_with_nulls
    # =====================================================================

    def test_validate_nullability_with_nulls(
        self,
        schema_validator: SchemaValidatorComponent,
        data_with_nulls: pl.DataFrame,
    ) -> None:
        """
        Verify nullability validation fails when nulls exist in non-nullable field.
        """
        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="instrument_id",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="Instrument ID cannot be null",
        )

        violation = schema_validator._validate_nullability(rule, data_with_nulls)

        assert violation is not None
        assert violation.violation_count == 4  # 4 null values
        assert violation.severity == QualityFlag.FAIL
        assert "null values" in violation.description

    # =====================================================================
    # Test 15: test_validate_lateness_fresh_data
    # =====================================================================

    def test_validate_lateness_fresh_data(
        self,
        schema_validator: SchemaValidatorComponent,
        mock_registry: MagicMock,
    ) -> None:
        """
        Verify lateness validation passes for fresh data.
        """
        manifest = mock_registry.get_manifest("bars_eurusd_1m")

        # Create data with current timestamps (fresh data)
        current_ns = time.time_ns()
        fresh_data = pl.DataFrame(
            {
                "instrument_id": ["EURUSD.SIM"] * 10,
                "ts_event": list(range(current_ns - 10, current_ns)),  # Very recent
                "ts_init": list(range(current_ns - 10, current_ns)),
                "open": [1.08] * 10,
                "high": [1.09] * 10,
                "low": [1.07] * 10,
                "close": [1.085] * 10,
                "volume": [1000.0] * 10,
            }
        )

        rule = ValidationRule(
            rule_type=ValidationRuleType.LATENESS,
            field_name="ts_event",
            parameters={"max_lateness_ns": 300_000_000_000},  # 5 minutes
            severity=QualityFlag.WARN,
            description="Data must be fresh",
        )

        violation = schema_validator._validate_lateness(rule, fresh_data, manifest)

        assert violation is None

    # =====================================================================
    # Additional Test: test_validate_batch_full_integration
    # =====================================================================

    def test_validate_batch_full_integration(
        self,
        schema_validator: SchemaValidatorComponent,
        valid_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        Verify batch validation integrates all validation rules.
        """
        report = schema_validator.validate_batch(
            dataset_id="bars_eurusd_1m",
            data=valid_ohlcv_data,
            strict_mode=False,
        )

        assert isinstance(report, QualityReport)
        assert report.total_records == 10
        assert report.quality_score >= 0.0
        assert report.quality_score <= 1.0
        assert report.validation_time_ms > 0
        assert report.dataset_id == "bars_eurusd_1m"
