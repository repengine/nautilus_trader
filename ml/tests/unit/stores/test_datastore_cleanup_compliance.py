#!/usr/bin/env python3

"""
Static analysis tests for DataStore cleanup compliance.

These tests verify that duplicate code has been removed and CLAUDE.md
patterns 5 and 6 are followed:
- Pattern 5: No _NoOpMetric definitions (use metrics_bootstrap)
- Pattern 6: No duplicate dataclasses (use validation_types.py)

This is a DELETION task - tests should pass after cleanup is complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Mark all tests as cleanup verification
pytestmark = [pytest.mark.unit, pytest.mark.cleanup]


# ========================================================================
# Pattern 5 Compliance: No _NoOpMetric Definitions
# ========================================================================


class TestNoOpMetricRemoval:
    """Verify all _NoOpMetric class definitions have been removed."""

    def test_no_noopmetric_in_common_data_writer(self) -> None:
        """Verify _NoOpMetric removed from common/data_writer.py."""
        # Navigate from ml/tests/unit/stores/test_*.py to ml/stores
        ml_path = Path(__file__).parents[3]  # Go up to ml/
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "data_writer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_common_schema_validator(self) -> None:
        """Verify _NoOpMetric removed from common/schema_validator.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "schema_validator.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_common_data_reader(self) -> None:
        """Verify _NoOpMetric removed from common/data_reader.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "data_reader.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_common_event_emitter(self) -> None:
        """Verify _NoOpMetric removed from common/event_emitter.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "event_emitter.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_common_contract_enforcer(self) -> None:
        """Verify _NoOpMetric removed from common/contract_enforcer.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "contract_enforcer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_common_store_operations(self) -> None:
        """Verify _NoOpMetric removed from common/store_operations.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "store_operations.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_data_store_facade(self) -> None:
        """Verify _NoOpMetric removed from data_store_facade.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_store_facade.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_root_data_writer(self) -> None:
        """Verify _NoOpMetric removed from root data_writer.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_writer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_root_schema_validator(self) -> None:
        """Verify _NoOpMetric removed from root schema_validator.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "schema_validator.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_in_root_contract_enforcer(self) -> None:
        """Verify _NoOpMetric removed from root contract_enforcer.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "contract_enforcer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        assert "class _NoOpMetric" not in content, (
            f"_NoOpMetric still defined in {target_file}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )

    def test_no_noopmetric_definitions_anywhere(self) -> None:
        """Comprehensive check: No _NoOpMetric in any stores file."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        violations: list[str] = []

        for py_file in stores_path.rglob("*.py"):
            # Skip test files
            if "test" in str(py_file):
                continue

            content = py_file.read_text()
            if "class _NoOpMetric" in content:
                violations.append(str(py_file))

        assert not violations, (
            f"_NoOpMetric still defined in {len(violations)} files: {violations}. "
            "Use ml.common.metrics_bootstrap instead (CLAUDE.md Pattern 5)"
        )


# ========================================================================
# Pattern 6 Compliance: No Duplicate Dataclass Definitions
# ========================================================================


class TestDataEventDuplicateRemoval:
    """Verify DataEvent is only defined in validation_types.py."""

    def test_no_dataevent_in_common_data_writer(self) -> None:
        """Verify DataEvent definition removed from common/data_writer.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "data_writer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        # Check for class definition, not import
        has_definition = (
            "class DataEvent:" in content or
            "class DataEvent(" in content
        )

        assert not has_definition, (
            f"DataEvent still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )

    def test_no_dataevent_in_data_store_facade(self) -> None:
        """Verify DataEvent definition removed from data_store_facade.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_store_facade.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class DataEvent:" in content or
            "class DataEvent(" in content
        )

        assert not has_definition, (
            f"DataEvent still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )

    def test_no_dataevent_in_root_data_writer(self) -> None:
        """Verify DataEvent definition removed from root data_writer.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_writer.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class DataEvent:" in content or
            "class DataEvent(" in content
        )

        assert not has_definition, (
            f"DataEvent still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )


class TestValidationViolationDuplicateRemoval:
    """Verify ValidationViolation is only defined in validation_types.py."""

    def test_no_validationviolation_in_common_schema_validator(self) -> None:
        """Verify ValidationViolation removed from common/schema_validator.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "schema_validator.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class ValidationViolation:" in content or
            "class ValidationViolation(" in content
        )

        assert not has_definition, (
            f"ValidationViolation still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )

    def test_no_validationviolation_in_data_store_facade(self) -> None:
        """Verify ValidationViolation removed from data_store_facade.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_store_facade.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class ValidationViolation:" in content or
            "class ValidationViolation(" in content
        )

        assert not has_definition, (
            f"ValidationViolation still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )


class TestQualityReportDuplicateRemoval:
    """Verify QualityReport is only defined in validation_types.py."""

    def test_no_qualityreport_in_common_schema_validator(self) -> None:
        """Verify QualityReport removed from common/schema_validator.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "common" / "schema_validator.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class QualityReport:" in content or
            "class QualityReport(" in content
        )

        assert not has_definition, (
            f"QualityReport still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )

    def test_no_qualityreport_in_data_store_facade(self) -> None:
        """Verify QualityReport removed from data_store_facade.py."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        target_file = stores_path / "data_store_facade.py"

        if not target_file.exists():
            pytest.skip(f"{target_file} does not exist")

        content = target_file.read_text()

        has_definition = (
            "class QualityReport:" in content or
            "class QualityReport(" in content
        )

        assert not has_definition, (
            f"QualityReport still defined in {target_file}. "
            "Import from ml.stores.validation_types instead (CLAUDE.md Pattern 6)"
        )


# ========================================================================
# Legacy File Cleanup Verification
# ========================================================================


class TestLegacyFileCleanup:
    """Verify legacy root-level duplicate files are removed or deprecated."""

    def test_legacy_schema_validator_deprecated(self) -> None:
        """Verify root schema_validator.py is deprecated or removed."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        legacy_file = stores_path / "schema_validator.py"

        if not legacy_file.exists():
            # File removed - test passes
            return

        content = legacy_file.read_text()

        # File should have deprecation warning if it still exists
        has_deprecation = (
            "DeprecationWarning" in content or
            "warnings.warn" in content or
            "deprecated" in content.lower()
        )

        assert has_deprecation, (
            f"{legacy_file} exists but has no deprecation warning. "
            "Either remove the file or add a deprecation warning."
        )

    def test_legacy_contract_enforcer_deprecated(self) -> None:
        """Verify root contract_enforcer.py is deprecated or removed."""
        ml_path = Path(__file__).parents[3]
        stores_path = ml_path / "stores"
        legacy_file = stores_path / "contract_enforcer.py"

        if not legacy_file.exists():
            # File removed - test passes
            return

        content = legacy_file.read_text()

        # File should have deprecation warning if it still exists
        has_deprecation = (
            "DeprecationWarning" in content or
            "warnings.warn" in content or
            "deprecated" in content.lower()
        )

        assert has_deprecation, (
            f"{legacy_file} exists but has no deprecation warning. "
            "Either remove the file or add a deprecation warning."
        )
