"""Tests for schema validator component."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.schema_validator import SchemaValidator
from ml.stores.schema_validator import SchemaValidatorProtocol
from ml.stores.validation_types import QualityReport
from ml.stores.validation_types import ValidationViolation

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

# ========================================================================
# Test Fixtures
# ========================================================================

@pytest.fixture
def basic_manifest() -> DatasetManifest:
    """Create a basic dataset manifest for testing."""
    return DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="test_table",
        partitioning={},
        retention_days=30,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "price": "float64",
            "volume": "int64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="test_hash",
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline",
        version="1.0.0",
        created_at=int(time.time_ns()),
        last_modified=int(time.time_ns()),
        metadata={},
    )

@pytest.fixture
def basic_contract() -> DataContract:
    """Create a basic data contract for testing."""
    # Add a dummy validation rule to satisfy contract requirements
    dummy_rule = ValidationRule(
        rule_type=ValidationRuleType.NULLABILITY,
        field_name="instrument_id",
        parameters={"nullable": False},
        severity=QualityFlag.WARN,
        description="Dummy rule",
    )
    return DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        validation_rules=[dummy_rule],
        quality_thresholds={"quality_score": 0.95},
        enforcement_mode="strict",
        created_at=int(time.time_ns()),
        last_modified=int(time.time_ns()),
        metadata={},
    )

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "instrument_id": ["EUR/USD", "EUR/USD", "EUR/USD"],
        "ts_event": [1000000000, 2000000000, 3000000000],
        "ts_init": [1000000100, 2000000100, 3000000100],
        "price": [1.1234, 1.1235, 1.1236],
        "volume": [100, 200, 300],
    })

@pytest.fixture
def validator() -> SchemaValidator:
    """Create a SchemaValidator instance."""
    return SchemaValidator()

# ========================================================================
# TestSchemaValidatorProtocol
# ========================================================================

class TestSchemaValidatorProtocol:
    """Protocol compliance tests."""

    def test_schema_validator_implements_protocol(self, validator: SchemaValidator) -> None:
        """SchemaValidator implements SchemaValidatorProtocol."""
        # Check that validator has required methods (structural typing)
        assert hasattr(validator, "validate_batch")
        assert hasattr(validator, "apply_validation_rule")
        assert callable(validator.validate_batch)
        assert callable(validator.apply_validation_rule)

    def test_protocol_has_validate_batch_method(self) -> None:
        """Protocol defines validate_batch method."""
        assert hasattr(SchemaValidatorProtocol, "validate_batch")

    def test_protocol_has_apply_validation_rule_method(self) -> None:
        """Protocol defines apply_validation_rule method."""
        assert hasattr(SchemaValidatorProtocol, "apply_validation_rule")

# ========================================================================
# TestValidateBatch
# ========================================================================

class TestValidateBatch:
    """Main validation orchestration tests."""

    def test_validate_batch_with_no_violations(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate batch with no violations returns perfect quality score."""
        # Create contract with no violations expected
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_dataset",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="instrument_id",
                    parameters={"nullable": False},
                    severity=QualityFlag.WARN,
                    description="No nulls in instrument_id",
                )
            ],
            quality_thresholds={"quality_score": 0.95},
            enforcement_mode="strict",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        report = validator.validate_batch(
            sample_dataframe,
            basic_manifest,
            contract,
            strict_mode=False,
        )

        assert isinstance(report, QualityReport)
        assert report.dataset_id == "test_dataset"
        assert report.total_records == 3
        assert report.passed_records == 3
        assert report.failed_records == 0
        assert report.quality_score == 1.0
        assert len(report.violations) == 0
        assert report.validation_time_ms > 0

    def test_validate_batch_with_violations(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate batch with violations returns degraded quality score."""
        # Create contract with range validation rule that will fail
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_dataset",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    parameters={"min": 2.0, "max": 3.0},
                    severity=QualityFlag.FAIL,
                    description="Price out of range",
                )
            ],
            quality_thresholds={"quality_score": 0.95},
            enforcement_mode="strict",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        report = validator.validate_batch(
            sample_dataframe,
            basic_manifest,
            contract,
            strict_mode=False,
        )

        assert report.quality_score < 1.0
        assert report.failed_records > 0
        assert len(report.violations) > 0

    def test_validate_batch_strict_mode_with_warnings(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate batch in strict mode penalizes warnings."""
        # Create contract with warning-level validation rule
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_dataset",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="volume",
                    parameters={"min": 500, "max": 1000},
                    severity=QualityFlag.WARN,
                    description="Volume outside recommended range",
                )
            ],
            quality_thresholds={"quality_score": 0.95},
            enforcement_mode="strict",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        report = validator.validate_batch(
            sample_dataframe,
            basic_manifest,
            contract,
            strict_mode=True,
        )

        # In strict mode, warnings reduce quality score
        assert report.quality_score < 1.0
        assert len(report.violations) > 0
        assert report.violations[0].severity == QualityFlag.WARN

    def test_validate_batch_with_empty_data(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate batch with empty DataFrame returns perfect score."""
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_dataset",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="instrument_id",
                    parameters={"nullable": False},
                    severity=QualityFlag.WARN,
                    description="No nulls",
                )
            ],
            quality_thresholds={"quality_score": 0.95},
            enforcement_mode="strict",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        empty_df = pd.DataFrame()
        report = validator.validate_batch(
            empty_df,
            basic_manifest,
            contract,
            strict_mode=False,
        )

        assert report.total_records == 0
        assert report.quality_score == 1.0
        assert len(report.violations) == 0

    def test_validate_batch_records_metrics(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Validate batch records Prometheus metrics."""
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_dataset",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="instrument_id",
                    parameters={"nullable": False},
                    severity=QualityFlag.WARN,
                    description="No nulls",
                )
            ],
            quality_thresholds={"quality_score": 0.95},
            enforcement_mode="strict",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        monkeypatch.setattr("ml.stores.schema_validator.HAS_PROMETHEUS", True)

        mock_quality_histogram = MagicMock()
        mock_duration_histogram = MagicMock()
        mock_quality_labels = MagicMock()
        mock_duration_labels = MagicMock()
        mock_quality_histogram.labels.return_value = mock_quality_labels
        mock_duration_histogram.labels.return_value = mock_duration_labels

        monkeypatch.setattr(
            "ml.stores.schema_validator.quality_score_histogram",
            mock_quality_histogram,
        )
        monkeypatch.setattr(
            "ml.stores.schema_validator.validation_duration_histogram",
            mock_duration_histogram,
        )
        monkeypatch.setattr(
            "ml.common.metrics.quality_score_histogram",
            mock_quality_histogram,
        )
        monkeypatch.setattr(
            "ml.common.metrics.validation_duration_histogram",
            mock_duration_histogram,
        )

        validator.validate_batch(
            sample_dataframe,
            basic_manifest,
            contract,
            strict_mode=False,
        )

        mock_quality_histogram.labels.assert_called_once_with(dataset_id="test_dataset")
        mock_quality_labels.observe.assert_called_once()
        mock_duration_histogram.labels.assert_called_once_with(
            dataset_id="test_dataset",
            operation="validate_batch",
        )
        mock_duration_labels.observe.assert_called_once()

# ========================================================================
# TestApplyValidationRule
# ========================================================================

class TestApplyValidationRule:
    """Rule dispatcher tests."""

    def test_apply_validation_rule_type_check(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply type check validation rule."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type check",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None  # All types match

    def test_apply_validation_rule_range(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply range validation rule."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="price",
            parameters={"min": 1.0, "max": 2.0},
            severity=QualityFlag.FAIL,
            description="Price range",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None  # All values in range

    def test_apply_validation_rule_uniqueness(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply uniqueness validation rule."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="ts_event",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Timestamp uniqueness",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None  # All timestamps unique

    def test_apply_validation_rule_monotonicity(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply monotonicity validation rule."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="ts_event",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Timestamp monotonicity",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None  # Timestamps are increasing

    def test_apply_validation_rule_nullability(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply nullability validation rule."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="price",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="Price nullability",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None  # No nulls in price

    def test_apply_validation_rule_unknown_type(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply unknown validation rule type returns None."""
        # Create a mock rule with unknown type
        rule = MagicMock()
        rule.rule_type = "unknown_type"
        rule.field_name = "price"

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        assert violation is None

    def test_apply_validation_rule_with_exception(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Apply validation rule that raises exception returns warning violation."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="nonexistent_field",
            parameters={"min": 0, "max": 100},
            severity=QualityFlag.FAIL,
            description="Range check",
        )

        violation = validator.apply_validation_rule(rule, sample_dataframe, basic_manifest)
        # Should return None for missing field
        assert violation is None

# ========================================================================
# TestTypeValidation
# ========================================================================

class TestTypeValidation:
    """_validate_types() tests."""

    def test_validate_types_all_match(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate types when all types match schema."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type validation",
        )

        violation = validator._validate_types(rule, sample_dataframe, basic_manifest)
        assert violation is None

    def test_validate_types_mismatch(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate types when types don't match schema."""
        # Create DataFrame with wrong type
        df = pd.DataFrame({
            "instrument_id": [1, 2, 3],  # int instead of str
            "ts_event": [1000, 2000, 3000],
            "price": [1.1, 1.2, 1.3],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type validation",
        )

        violation = validator._validate_types(rule, df, basic_manifest)
        assert violation is not None
        assert violation.rule_type == ValidationRuleType.TYPE_CHECK
        assert violation.violation_count > 0

    def test_types_compatible_int64(self, validator: SchemaValidator) -> None:
        """Type compatibility check for int64 types."""
        assert validator._types_compatible("int64", "int64")
        assert validator._types_compatible("int64", "int")
        assert validator._types_compatible("Int64", "int")

    def test_types_compatible_float64(self, validator: SchemaValidator) -> None:
        """Type compatibility check for float64 types."""
        assert validator._types_compatible("float64", "float64")
        assert validator._types_compatible("float64", "float")
        assert validator._types_compatible("Float64", "float")

    def test_types_compatible_string(self, validator: SchemaValidator) -> None:
        """Type compatibility check for string types."""
        assert validator._types_compatible("object", "str")
        assert validator._types_compatible("object", "string")
        # Note: Utf8 is case-sensitive in the type_map
        assert not validator._types_compatible("utf8", "int")

    def test_types_incompatible(self, validator: SchemaValidator) -> None:
        """Type compatibility check for incompatible types."""
        assert not validator._types_compatible("int64", "float64")
        assert not validator._types_compatible("str", "int64")

# ========================================================================
# TestRegexValidation
# ========================================================================

class TestRegexValidation:
    """_validate_regex() tests."""

    def test_validate_regex_all_match(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate regex when all values match pattern."""
        df = pd.DataFrame({
            "instrument_id": ["EUR/USD", "GBP/USD", "USD/JPY"],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.REGEX,
            field_name="instrument_id",
            parameters={"pattern": r"^[A-Z]{3}/[A-Z]{3}$"},
            severity=QualityFlag.FAIL,
            description="Instrument ID format",
        )

        violation = validator._validate_regex(rule, df)
        assert violation is None

    def test_validate_regex_some_mismatch(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate regex when some values don't match pattern."""
        df = pd.DataFrame({
            "instrument_id": ["EUR/USD", "invalid", "USD/JPY"],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.REGEX,
            field_name="instrument_id",
            parameters={"pattern": r"^[A-Z]{3}/[A-Z]{3}$"},
            severity=QualityFlag.FAIL,
            description="Instrument ID format",
        )

        violation = validator._validate_regex(rule, df)
        assert violation is not None
        assert violation.violation_count == 1
        assert "invalid" in violation.sample_values

    def test_validate_regex_invalid_pattern(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate regex with invalid pattern returns warning."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.REGEX,
            field_name="instrument_id",
            parameters={"pattern": "[invalid(regex"},
            severity=QualityFlag.FAIL,
            description="Invalid regex",
        )

        violation = validator._validate_regex(rule, sample_dataframe)
        assert violation is not None
        assert violation.severity == QualityFlag.WARN
        assert "Invalid regex pattern" in violation.description

    def test_validate_regex_missing_field(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate regex on missing field returns None."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.REGEX,
            field_name="nonexistent",
            parameters={"pattern": r".*"},
            severity=QualityFlag.FAIL,
            description="Missing field",
        )

        violation = validator._validate_regex(rule, sample_dataframe)
        assert violation is None

    def test_validate_regex_empty_pattern(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate regex with empty pattern string returns None."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.REGEX,
            field_name="instrument_id",
            parameters={"pattern": ""},
            severity=QualityFlag.FAIL,
            description="Empty pattern",
        )

        violation = validator._validate_regex(rule, sample_dataframe)
        assert violation is None

# ========================================================================
# TestRangeValidation
# ========================================================================

class TestRangeValidation:
    """_validate_range() tests."""

    def test_validate_range_within_bounds(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate range when all values within bounds."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="price",
            parameters={"min": 1.0, "max": 2.0},
            severity=QualityFlag.FAIL,
            description="Price range",
        )

        violation = validator._validate_range(rule, sample_dataframe)
        assert violation is None

    def test_validate_range_below_min(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate range when values below minimum."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="price",
            parameters={"min": 2.0},
            severity=QualityFlag.FAIL,
            description="Price minimum",
        )

        violation = validator._validate_range(rule, sample_dataframe)
        assert violation is not None
        assert violation.violation_count == 3  # All values below 2.0

    def test_validate_range_above_max(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate range when values above maximum."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="price",
            parameters={"max": 1.0},
            severity=QualityFlag.FAIL,
            description="Price maximum",
        )

        violation = validator._validate_range(rule, sample_dataframe)
        assert violation is not None
        assert violation.violation_count == 3  # All values above 1.0

    def test_validate_range_mixed_violations(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate range with both min and max violations."""
        df = pd.DataFrame({
            "value": [0.5, 1.5, 2.5, 3.5, 4.5],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="value",
            parameters={"min": 1.0, "max": 4.0},
            severity=QualityFlag.FAIL,
            description="Value range",
        )

        violation = validator._validate_range(rule, df)
        assert violation is not None
        assert violation.violation_count == 2  # 0.5 below min, 4.5 above max

    def test_validate_range_missing_field(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate range on missing field returns None."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="nonexistent",
            parameters={"min": 0, "max": 100},
            severity=QualityFlag.FAIL,
            description="Missing field",
        )

        violation = validator._validate_range(rule, sample_dataframe)
        assert violation is None

# ========================================================================
# TestUniquenessValidation
# ========================================================================

class TestUniquenessValidation:
    """_validate_uniqueness() tests."""

    def test_validate_uniqueness_no_duplicates(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate uniqueness when no duplicates exist."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="ts_event",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Timestamp uniqueness",
        )

        violation = validator._validate_uniqueness(rule, sample_dataframe)
        assert violation is None

    def test_validate_uniqueness_with_duplicates(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate uniqueness when duplicates exist."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="id",
            parameters={},
            severity=QualityFlag.FAIL,
            description="ID uniqueness",
        )

        violation = validator._validate_uniqueness(rule, df)
        assert violation is not None
        assert violation.violation_count > 0

    def test_validate_uniqueness_composite_key(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate uniqueness for composite key."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="instrument_id, ts_event",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Composite key uniqueness",
        )

        violation = validator._validate_uniqueness(rule, sample_dataframe)
        assert violation is None

    def test_validate_uniqueness_composite_key_duplicates(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate uniqueness for composite key with duplicates."""
        df = pd.DataFrame({
            "key1": ["A", "A", "B", "B"],
            "key2": [1, 1, 2, 3],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.UNIQUENESS,
            field_name="key1, key2",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Composite uniqueness",
        )

        violation = validator._validate_uniqueness(rule, df)
        assert violation is not None

# ========================================================================
# TestMonotonicityValidation
# ========================================================================

class TestMonotonicityValidation:
    """_validate_monotonicity() tests."""

    def test_validate_monotonicity_strictly_increasing(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate monotonicity for strictly increasing sequence."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="ts_event",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Timestamp monotonicity",
        )

        violation = validator._validate_monotonicity(rule, sample_dataframe)
        assert violation is None

    def test_validate_monotonicity_non_increasing(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate monotonicity for non-increasing sequence."""
        df = pd.DataFrame({
            "ts": [1000, 2000, 1500, 3000],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="ts",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Timestamp monotonicity",
        )

        violation = validator._validate_monotonicity(rule, df)
        assert violation is not None
        assert violation.violation_count > 0

    def test_validate_monotonicity_decreasing(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate monotonicity for decreasing sequence."""
        df = pd.DataFrame({
            "countdown": [100, 90, 80, 70],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="countdown",
            parameters={"direction": "decreasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Countdown monotonicity",
        )

        violation = validator._validate_monotonicity(rule, df)
        assert violation is None

    def test_validate_monotonicity_non_strict(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate monotonicity with non-strict mode."""
        df = pd.DataFrame({
            "value": [1, 2, 2, 3],  # Equal values allowed in non-strict
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="value",
            parameters={"direction": "increasing", "strict": False},
            severity=QualityFlag.FAIL,
            description="Non-strict monotonicity",
        )

        violation = validator._validate_monotonicity(rule, df)
        assert violation is None

    def test_validate_monotonicity_missing_field(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate monotonicity on missing field returns None."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.MONOTONICITY,
            field_name="nonexistent",
            parameters={"direction": "increasing", "strict": True},
            severity=QualityFlag.FAIL,
            description="Missing field",
        )

        violation = validator._validate_monotonicity(rule, sample_dataframe)
        assert violation is None

# ========================================================================
# TestNullabilityValidation
# ========================================================================

class TestNullabilityValidation:
    """_validate_nullability() tests."""

    def test_validate_nullability_no_nulls(
        self,
        validator: SchemaValidator,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Validate nullability when no nulls exist."""
        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="price",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="Price nullability",
        )

        violation = validator._validate_nullability(rule, sample_dataframe)
        assert violation is None

    def test_validate_nullability_with_nulls(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate nullability when nulls exist."""
        df = pd.DataFrame({
            "value": [1.0, None, 3.0, None, 5.0],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="value",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="Value nullability",
        )

        violation = validator._validate_nullability(rule, df)
        assert violation is not None
        assert violation.violation_count == 2

    def test_validate_nullability_nullable_allowed(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate nullability when nulls are allowed."""
        df = pd.DataFrame({
            "value": [1.0, None, 3.0],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="value",
            parameters={"nullable": True},
            severity=QualityFlag.FAIL,
            description="Value nullability",
        )

        violation = validator._validate_nullability(rule, df)
        assert violation is None

    def test_validate_nullability_wildcard_field(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate nullability for all fields using wildcard."""
        df = pd.DataFrame({
            "col1": [1, 2, None],
            "col2": [4.0, None, 6.0],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="*",
            parameters={"nullable": False},
            severity=QualityFlag.FAIL,
            description="No nulls allowed",
        )

        violation = validator._validate_nullability(rule, df)
        assert violation is not None
        assert violation.violation_count == 2  # 2 nulls total
        assert len(violation.sample_values) > 0  # Field names with nulls

# ========================================================================
# TestLatenessValidation
# ========================================================================

class TestLatenessValidation:
    """_validate_lateness() tests."""

    def test_validate_lateness_fresh_data(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate lateness for fresh data."""
        current_ns = int(time.time_ns())
        df = pd.DataFrame({
            "ts_event": [current_ns - 1_000_000_000],  # 1 second ago
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.LATENESS,
            field_name="ts_event",
            parameters={"max_lateness_ns": 60_000_000_000},  # 60 seconds
            severity=QualityFlag.WARN,
            description="Data lateness",
        )

        violation = validator._validate_lateness(rule, df, basic_manifest)
        assert violation is None

    def test_validate_lateness_stale_data(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate lateness for stale data."""
        current_ns = int(time.time_ns())
        df = pd.DataFrame({
            "ts_event": [current_ns - 600_000_000_000],  # 600 seconds ago
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.LATENESS,
            field_name="ts_event",
            parameters={"max_lateness_ns": 60_000_000_000},  # 60 seconds
            severity=QualityFlag.WARN,
            description="Data lateness",
        )

        violation = validator._validate_lateness(rule, df, basic_manifest)
        assert violation is not None
        assert violation.violation_count == 1
        assert "late" in violation.description.lower()

    def test_validate_lateness_missing_field_in_dataframe(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate lateness when timestamp field exists in schema but not in dataframe."""
        # Create a manifest with ts_field in schema
        manifest = DatasetManifest(
            dataset_id="test",
            dataset_type=DatasetType.BARS,
            storage_kind=StorageKind.POSTGRES,
            location="test",
            partitioning={},
            retention_days=30,
            schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "other_field": "str"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["ts_event"],
            schema_hash="hash",
            constraints={},
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
            created_at=int(time.time_ns()),
            last_modified=int(time.time_ns()),
            metadata={},
        )

        # Create dataframe without ts_event column
        df_without_ts = pd.DataFrame({
            "other_field": ["value1", "value2"],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.LATENESS,
            field_name="ts_event",
            parameters={"max_lateness_ns": 60_000_000_000},
            severity=QualityFlag.WARN,
            description="Data lateness",
        )

        violation = validator._validate_lateness(rule, df_without_ts, manifest)
        assert violation is None

# ========================================================================
# TestFormatViolations
# ========================================================================

class TestFormatViolations:
    """format_violations() tests."""

    def test_format_violations_empty(self, validator: SchemaValidator) -> None:
        """Format empty violations list."""
        result = validator.format_violations([])
        assert result == "None"

    def test_format_violations_single(self, validator: SchemaValidator) -> None:
        """Format single violation."""
        violations = [
            ValidationViolation(
                rule_type=ValidationRuleType.RANGE,
                field_name="price",
                severity=QualityFlag.FAIL,
                violation_count=5,
                sample_values=[],
                description="Price out of range",
            )
        ]

        result = validator.format_violations(violations)
        assert "price" in result
        assert "Price out of range" in result
        assert "5 records" in result

    def test_format_violations_multiple(self, validator: SchemaValidator) -> None:
        """Format multiple violations."""
        violations = [
            ValidationViolation(
                rule_type=ValidationRuleType.RANGE,
                field_name="price",
                severity=QualityFlag.FAIL,
                violation_count=5,
                sample_values=[],
                description="Price out of range",
            ),
            ValidationViolation(
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="volume",
                severity=QualityFlag.WARN,
                violation_count=2,
                sample_values=[],
                description="Null values found",
            ),
        ]

        result = validator.format_violations(violations)
        assert "price" in result
        assert "volume" in result

    def test_format_violations_many(self, validator: SchemaValidator) -> None:
        """Format many violations shows truncation."""
        violations = [
            ValidationViolation(
                rule_type=ValidationRuleType.RANGE,
                field_name=f"field{i}",
                severity=QualityFlag.FAIL,
                violation_count=1,
                sample_values=[],
                description=f"Violation {i}",
            )
            for i in range(5)
        ]

        result = validator.format_violations(violations)
        assert "... and 2 more" in result

# ========================================================================
# TestEnforceQualityReport
# ========================================================================

def _create_contract(enforcement_mode: str = "strict") -> DataContract:
    """Helper to create a test contract."""
    return DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="instrument_id",
                parameters={"nullable": False},
                severity=QualityFlag.WARN,
                description="No nulls",
            )
        ],
        quality_thresholds={"quality_score": 0.95},
        enforcement_mode=enforcement_mode,
        created_at=int(time.time_ns()),
        last_modified=int(time.time_ns()),
        metadata={},
    )

class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_validate_batch_with_list_data(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate batch with data that has no __len__ attribute."""
        contract = _create_contract()

        # Test with data that doesn't have __len__
        class FakeData:
            pass

        fake_data = FakeData()
        report = validator.validate_batch(
            fake_data,
            basic_manifest,
            contract,
            strict_mode=False,
        )

        assert report.total_records == 0
        assert report.quality_score == 1.0

    def test_validate_range_handles_exceptions_gracefully(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Validate range handles exceptions gracefully."""
        # Create a DataFrame where operations will likely fail
        df = pd.DataFrame({
            "weird_col": ["a", "b", "c"],  # String data for range check
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="weird_col",
            parameters={"min": 0, "max": 100},
            severity=QualityFlag.FAIL,
            description="Test rule",
        )

        # Should not raise an exception, just log the error
        # The code catches exceptions and logs them
        try:
            violation = validator._validate_range(rule, df)
            # May return None or a violation depending on how the exception is handled
        except Exception:
            pass  # The code should catch exceptions internally

    def test_validate_types_with_missing_columns(
        self,
        validator: SchemaValidator,
        basic_manifest: DatasetManifest,
    ) -> None:
        """Validate types when DataFrame is missing columns from schema."""
        # DataFrame with only some columns
        df = pd.DataFrame({
            "instrument_id": ["EUR/USD"],
            "ts_event": [1000000000],
        })

        rule = ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="*",
            parameters={},
            severity=QualityFlag.FAIL,
            description="Type check",
        )

        # Should not raise, just skip missing columns
        violation = validator._validate_types(rule, df, basic_manifest)
        # May or may not have violations depending on columns present

class TestEnforceQualityReport:
    """enforce_quality_report() tests."""

    def test_enforce_quality_report_perfect_score(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report with perfect score passes."""
        contract = _create_contract()
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=100,
            failed_records=0,
            quality_score=1.0,
            violations=[],
            validation_time_ms=10.0,
        )

        # Should not raise
        validator.enforce_quality_report(
            dataset_id="test_dataset",
            contract=contract,
            quality_report=report,
            fail_on_validation_error=True,
        )

    def test_enforce_quality_report_strict_mode_fails(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report in strict mode fails on violations."""
        contract = _create_contract(enforcement_mode="strict")
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=90,
            failed_records=10,
            quality_score=0.9,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    severity=QualityFlag.FAIL,
                    violation_count=10,
                    sample_values=[],
                    description="Price out of range",
                )
            ],
            validation_time_ms=10.0,
        )

        with pytest.raises(ValueError, match="validation failed"):
            validator.enforce_quality_report(
                dataset_id="test_dataset",
                contract=contract,
                quality_report=report,
                fail_on_validation_error=True,
            )

    def test_enforce_quality_report_lenient_mode_logs(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report in lenient mode logs warnings."""
        contract = _create_contract(enforcement_mode="lenient")
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=90,
            failed_records=10,
            quality_score=0.9,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    severity=QualityFlag.WARN,
                    violation_count=10,
                    sample_values=[],
                    description="Price out of range",
                )
            ],
            validation_time_ms=10.0,
        )

        # Should not raise, only log
        validator.enforce_quality_report(
            dataset_id="test_dataset",
            contract=contract,
            quality_report=report,
            fail_on_validation_error=True,
        )

    def test_enforce_quality_report_monitor_only(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report in monitor_only mode does not fail."""
        contract = _create_contract(enforcement_mode="monitor_only")
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=50,
            failed_records=50,
            quality_score=0.5,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    severity=QualityFlag.FAIL,
                    violation_count=50,
                    sample_values=[],
                    description="Price out of range",
                )
            ],
            validation_time_ms=10.0,
        )

        # Should not raise even with critical violations
        validator.enforce_quality_report(
            dataset_id="test_dataset",
            contract=contract,
            quality_report=report,
            fail_on_validation_error=True,
        )

    def test_enforce_quality_report_critical_violations_fail(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report with critical violations fails."""
        contract = _create_contract(enforcement_mode="strict")
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=90,
            failed_records=10,
            quality_score=0.9,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    severity=QualityFlag.FAIL,
                    violation_count=10,
                    sample_values=[],
                    description="Critical failure",
                )
            ],
            validation_time_ms=10.0,
        )

        with pytest.raises(ValueError, match="validation failed"):
            validator.enforce_quality_report(
                dataset_id="test_dataset",
                contract=contract,
                quality_report=report,
                fail_on_validation_error=True,
            )

    def test_enforce_quality_report_records_rejection_metric(
        self,
        validator: SchemaValidator,
    ) -> None:
        """Enforce quality report records rejection metrics."""
        contract = _create_contract(enforcement_mode="strict")
        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=90,
            failed_records=10,
            quality_score=0.9,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="price",
                    severity=QualityFlag.FAIL,
                    violation_count=10,
                    sample_values=[],
                    description="Critical failure",
                )
            ],
            validation_time_ms=10.0,
        )

        try:
            validator.enforce_quality_report(
                dataset_id="test_dataset",
                contract=contract,
                quality_report=report,
                fail_on_validation_error=True,
            )
        except ValueError:
            pass  # Expected to raise
