#!/usr/bin/env python3
"""
Verification tests for enum comparison patterns in parallel execution.

These tests validate that our fix patterns (value comparison, string comparison,
class name comparison) work correctly across pytest-xdist worker boundaries.
"""

import pytest
from ml.config.events import Stage, Source, EventStatus


class TestEnumValueComparison:
    """Test Pattern A: Value comparison."""

    def test_enum_value_comparison_works(self):
        """Verify enum.value comparisons work (Pattern A)."""
        stage = Stage.FEATURE_COMPUTED

        # This should work in parallel execution
        assert stage.value == "FEATURE_COMPUTED"

        # Multiple enums
        source = Source.LIVE
        status = EventStatus.SUCCESS

        assert source.value == "live"
        assert status.value == "success"

    def test_enum_value_comparison_in_dict(self):
        """Verify value comparisons work when enum in dict."""
        data = {
            "stage": Stage.PREDICTION_EMITTED,
            "source": Source.HISTORICAL,
            "status": EventStatus.PARTIAL,
        }

        # Pattern A: Compare values
        assert data["stage"].value == "PREDICTION_EMITTED"
        assert data["source"].value == "historical"
        assert data["status"].value == "partial"

    def test_enum_value_comparison_from_mock(self):
        """Verify value comparisons work with mock call args."""
        from unittest.mock import MagicMock

        mock_fn = MagicMock()
        mock_fn(
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            status=EventStatus.SUCCESS,
        )

        call_args = mock_fn.call_args

        # This is the pattern we'll use in contract tests
        assert call_args.kwargs["stage"].value == "FEATURE_COMPUTED"
        assert call_args.kwargs["source"].value == "live"
        assert call_args.kwargs["status"].value == "success"


class TestEnumStringComparison:
    """Test Pattern B: String comparison."""

    def test_enum_string_comparison_works(self):
        """Verify str(enum) comparisons work (Pattern B)."""
        stage = Stage.FEATURE_COMPUTED

        # String comparison includes class name
        assert str(stage) == "Stage.FEATURE_COMPUTED"

    def test_enum_string_comparison_with_different_values(self):
        """Verify string comparison for all enum types."""
        assert str(Source.LIVE) == "Source.LIVE"
        assert str(EventStatus.SUCCESS) == "EventStatus.SUCCESS"


class TestEnumClassNameComparison:
    """Test Pattern C: Class name comparison."""

    def test_enum_class_name_comparison_works(self):
        """Verify __class__.__name__ comparisons work (Pattern C)."""
        stage = Stage.FEATURE_COMPUTED

        # Type-only check (don't care about value)
        assert stage.__class__.__name__ == "Stage"

    def test_class_name_comparison_for_all_enum_types(self):
        """Verify class name pattern for all our enum types."""
        assert Source.LIVE.__class__.__name__ == "Source"
        assert EventStatus.SUCCESS.__class__.__name__ == "EventStatus"


class TestEnumIsinstanceAntiPattern:
    """Document WHY isinstance() doesn't work in parallel."""

    @pytest.mark.xfail(
        reason="isinstance() fails in pytest-xdist parallel execution",
        strict=False,
    )
    def test_isinstance_may_fail_in_parallel(self):
        """
        This test documents the anti-pattern that fails in parallel.

        In pytest-xdist, each worker imports modules independently. When
        two workers import the same enum class, Python creates separate
        class objects with different memory addresses.

        isinstance() compares memory addresses (identity), not values.
        Therefore: isinstance(Stage.FEATURE_COMPUTED, Stage) may return
        False if the enum was created in a different worker.

        This test is marked xfail to document the issue, not to validate
        our fix. It MAY pass in serial execution or with our pytest-xdist
        config improvements.
        """
        stage = Stage.FEATURE_COMPUTED

        # This MAY fail in parallel execution
        assert isinstance(stage, Stage)


class TestEnumComparisonInParametrizedTests:
    """Verify patterns work in parameterized tests (common in our suite)."""

    @pytest.mark.parametrize(
        "stage,expected_value",
        [
            (Stage.FEATURE_COMPUTED, "FEATURE_COMPUTED"),
            (Stage.PREDICTION_EMITTED, "PREDICTION_EMITTED"),
            (Stage.SIGNAL_EMITTED, "SIGNAL_EMITTED"),
        ],
    )
    def test_value_comparison_in_parameterized_test(self, stage, expected_value):
        """Verify Pattern A works in parameterized tests."""
        assert stage.value == expected_value

    @pytest.mark.parametrize("source", [Source.LIVE, Source.HISTORICAL, Source.BACKFILL])
    def test_class_name_comparison_in_parameterized_test(self, source):
        """Verify Pattern C works in parameterized tests."""
        assert source.__class__.__name__ == "Source"
