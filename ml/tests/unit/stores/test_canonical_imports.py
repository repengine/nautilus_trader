#!/usr/bin/env python3

"""
Import verification tests for DataStore cleanup.

These tests verify that canonical types can be imported correctly from
validation_types.py and that components use these canonical imports
instead of local definitions.

Part of CLAUDE.md Pattern 6 compliance.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


# Mark all tests as cleanup verification
pytestmark = [pytest.mark.unit, pytest.mark.cleanup]


# ========================================================================
# Canonical Type Import Tests
# ========================================================================


class TestValidationTypesImports:
    """Verify canonical dataclasses import correctly from validation_types.py."""

    def test_dataevent_imports_correctly(self) -> None:
        """Verify DataEvent can be imported from validation_types."""
        from ml.stores.validation_types import DataEvent

        # Verify it's a dataclass
        dataclass_fields = fields(DataEvent)
        assert len(dataclass_fields) > 0, "DataEvent should have fields"

        # Verify key attributes
        field_names = [f.name for f in dataclass_fields]
        assert "event_id" in field_names
        assert "dataset_id" in field_names
        assert "instrument_id" in field_names
        assert "status" in field_names
        assert "record_count" in field_names

    def test_validationviolation_imports_correctly(self) -> None:
        """Verify ValidationViolation can be imported from validation_types."""
        from ml.stores.validation_types import ValidationViolation

        # Verify it's a dataclass
        dataclass_fields = fields(ValidationViolation)
        assert len(dataclass_fields) > 0, "ValidationViolation should have fields"

        # Verify key attributes
        field_names = [f.name for f in dataclass_fields]
        assert "rule_type" in field_names
        assert "field_name" in field_names
        assert "severity" in field_names
        assert "violation_count" in field_names

    def test_qualityreport_imports_correctly(self) -> None:
        """Verify QualityReport can be imported from validation_types."""
        from ml.stores.validation_types import QualityReport

        # Verify it's a dataclass
        dataclass_fields = fields(QualityReport)
        assert len(dataclass_fields) > 0, "QualityReport should have fields"

        # Verify key attributes
        field_names = [f.name for f in dataclass_fields]
        assert "dataset_id" in field_names
        assert "total_records" in field_names
        assert "quality_score" in field_names
        assert "violations" in field_names

    def test_all_types_import_in_single_statement(self) -> None:
        """Verify all types can be imported together."""
        from ml.stores.validation_types import (
            DataEvent,
            QualityReport,
            ValidationViolation,
        )

        # All should be classes
        assert isinstance(DataEvent, type)
        assert isinstance(ValidationViolation, type)
        assert isinstance(QualityReport, type)


# ========================================================================
# Dataclass Behavior Tests
# ========================================================================


class TestDataEventBehavior:
    """Verify DataEvent dataclass behaves correctly."""

    def test_dataevent_is_frozen(self) -> None:
        """Verify DataEvent is immutable (frozen)."""
        from ml.stores.validation_types import DataEvent

        event = DataEvent(
            event_id="test-123",
            dataset_id="test_dataset",
            instrument_id="EURUSD.SIM",
            operation="write_ingestion",
            source="test",
            run_id="run-1",
            ts_min=0,
            ts_max=100,
            record_count=10,
            status="success",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            event.status = "failed"  # type: ignore[misc]

    def test_dataevent_optional_fields(self) -> None:
        """Verify DataEvent optional fields have defaults."""
        from ml.stores.validation_types import DataEvent

        event = DataEvent(
            event_id="test-123",
            dataset_id="test_dataset",
            instrument_id="EURUSD.SIM",
            operation="write_ingestion",
            source="test",
            run_id="run-1",
            ts_min=0,
            ts_max=100,
            record_count=10,
            status="success",
        )

        # Optional fields should have defaults
        assert event.error_message is None
        assert isinstance(event.created_at, int)
        assert isinstance(event.metadata, dict)


class TestValidationViolationBehavior:
    """Verify ValidationViolation dataclass behaves correctly."""

    def test_validationviolation_is_frozen(self) -> None:
        """Verify ValidationViolation is immutable (frozen)."""
        from ml.registry.dataclasses import QualityFlag, ValidationRuleType
        from ml.stores.validation_types import ValidationViolation

        violation = ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name="price",
            severity=QualityFlag.FAIL,
            violation_count=5,
            sample_values=[1.0, 2.0],
            description="Price out of range",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            violation.violation_count = 10  # type: ignore[misc]


class TestQualityReportBehavior:
    """Verify QualityReport dataclass behaves correctly."""

    def test_qualityreport_is_frozen(self) -> None:
        """Verify QualityReport is immutable (frozen)."""
        from ml.stores.validation_types import QualityReport

        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=95,
            failed_records=5,
            quality_score=0.95,
            violations=[],
            validation_time_ms=10.5,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            report.quality_score = 0.5  # type: ignore[misc]

    def test_qualityreport_default_metadata(self) -> None:
        """Verify QualityReport has empty dict default for metadata."""
        from ml.stores.validation_types import QualityReport

        report = QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=95,
            failed_records=5,
            quality_score=0.95,
            violations=[],
            validation_time_ms=10.5,
        )

        assert isinstance(report.metadata, dict)
        assert report.metadata == {}


# ========================================================================
# Component Import Verification
# ========================================================================


class TestComponentsUseCanonicalTypes:
    """Verify components import types from validation_types.py."""

    def test_common_data_writer_imports_canonical(self) -> None:
        """Verify common/data_writer.py imports from validation_types."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "data_writer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        # Should import from validation_types
        has_canonical_import = (
            "from ml.stores.validation_types import" in content or
            "from ml.stores.validation_types import DataEvent" in content
        )

        assert has_canonical_import, (
            f"{target_file} should import DataEvent from ml.stores.validation_types"
        )

    def test_common_schema_validator_imports_canonical(self) -> None:
        """Verify common/schema_validator.py imports from validation_types."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "schema_validator.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        # Should import from validation_types
        has_canonical_import = (
            "from ml.stores.validation_types import" in content
        )

        assert has_canonical_import, (
            f"{target_file} should import types from ml.stores.validation_types"
        )


# ========================================================================
# Circular Import Prevention Tests
# ========================================================================


class TestNoCircularImports:
    """Verify no circular imports after cleanup."""

    def test_import_validation_types_first(self) -> None:
        """Verify validation_types can be imported independently."""
        import importlib
        import sys

        # Clear any cached modules
        modules_to_clear = [
            "ml.stores.validation_types",
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Should not raise
        try:
            import ml.stores.validation_types
            importlib.reload(ml.stores.validation_types)
        except ImportError as e:
            pytest.fail(f"Failed to import validation_types: {e}")

    def test_import_facade_after_types(self) -> None:
        """Verify facade can be imported after validation_types."""
        # Import in expected order
        try:
            from ml.stores.validation_types import DataEvent, QualityReport
            from ml.stores.data_store_facade import DataStoreFacade
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_import_order_does_not_matter(self) -> None:
        """Verify import order doesn't cause errors."""
        import importlib
        import sys

        # Clear cached modules
        modules = [
            "ml.stores.validation_types",
            "ml.stores.data_store_facade",
        ]
        for mod in modules:
            if mod in sys.modules:
                del sys.modules[mod]

        # Try reverse order - should also work
        try:
            import ml.stores.data_store_facade
            import ml.stores.validation_types
        except ImportError as e:
            pytest.fail(f"Import order caused circular import: {e}")
