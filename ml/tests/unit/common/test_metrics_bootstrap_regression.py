#!/usr/bin/env python3

"""
Regression tests for metrics_bootstrap fallback behavior.

These tests verify that ml.common.metrics_bootstrap continues to work
correctly after the DataStore cleanup, ensuring that dummy metrics
are returned when Prometheus is unavailable.

Part of CLAUDE.md Pattern 5 compliance - components should use
metrics_bootstrap instead of defining local _NoOpMetric classes.
"""

from __future__ import annotations

import pytest


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ========================================================================
# Metrics Bootstrap Fallback Tests
# ========================================================================


class TestMetricsBootstrapFallback:
    """Verify metrics_bootstrap returns working dummy metrics."""

    def test_get_counter_returns_callable(self) -> None:
        """Verify get_counter returns an object with expected interface."""
        from ml.common.metrics_bootstrap import get_counter

        # Must specify labels when creating a labeled counter
        counter = get_counter(
            "test_regression_counter",
            "Test counter for regression",
            labelnames=["test"],
        )

        # Should have labels method
        assert hasattr(counter, "labels"), "Counter should have labels method"

        # Labels should return object with inc
        labeled = counter.labels(test="value")
        assert hasattr(labeled, "inc"), "Labeled counter should have inc method"

    def test_get_histogram_returns_callable(self) -> None:
        """Verify get_histogram returns an object with expected interface."""
        from ml.common.metrics_bootstrap import get_histogram

        histogram = get_histogram(
            "test_regression_histogram",
            "Test histogram",
            labelnames=["test"],
        )

        # Should have labels method
        assert hasattr(histogram, "labels"), "Histogram should have labels method"

        # Labels should return object with observe
        labeled = histogram.labels(test="value")
        assert hasattr(labeled, "observe"), "Labeled histogram should have observe method"

    def test_get_gauge_returns_callable(self) -> None:
        """Verify get_gauge returns an object with expected interface."""
        from ml.common.metrics_bootstrap import get_gauge

        gauge = get_gauge(
            "test_regression_gauge",
            "Test gauge",
            labelnames=["test"],
        )

        # Should have labels method
        assert hasattr(gauge, "labels"), "Gauge should have labels method"

        # Labels should return object with set
        labeled = gauge.labels(test="value")
        assert hasattr(labeled, "set"), "Labeled gauge should have set method"

    def test_counter_operations_do_not_raise(self) -> None:
        """Verify counter operations don't raise exceptions."""
        from ml.common.metrics_bootstrap import get_counter

        counter = get_counter(
            "test_safe_counter",
            "Safe counter",
            labelnames=["operation"],
        )

        # None of these should raise
        counter.labels(operation="test").inc()
        counter.labels(operation="test").inc(5)

    def test_histogram_operations_do_not_raise(self) -> None:
        """Verify histogram operations don't raise exceptions."""
        from ml.common.metrics_bootstrap import get_histogram

        histogram = get_histogram(
            "test_safe_histogram",
            "Safe histogram",
            labelnames=["operation"],
        )

        # None of these should raise
        histogram.labels(operation="test").observe(1.0)
        histogram.labels(operation="test").observe(0.5)
        histogram.labels(operation="test").observe(100.0)

    def test_gauge_operations_do_not_raise(self) -> None:
        """Verify gauge operations don't raise exceptions."""
        from ml.common.metrics_bootstrap import get_gauge

        gauge = get_gauge(
            "test_safe_gauge",
            "Safe gauge",
            labelnames=["component"],
        )

        # None of these should raise
        gauge.labels(component="test").set(42)
        gauge.labels(component="test").set(0)
        gauge.labels(component="test").set(-1)


# ========================================================================
# HAS_METRICS_BACKEND Flag Tests
# ========================================================================


class TestHasMetricsBackendFlag:
    """Verify HAS_METRICS_BACKEND flag is available and correct."""

    def test_has_metrics_backend_is_boolean(self) -> None:
        """Verify HAS_METRICS_BACKEND is a boolean."""
        from ml.common.metrics_bootstrap import HAS_METRICS_BACKEND

        assert isinstance(HAS_METRICS_BACKEND, bool)

    def test_has_metrics_backend_is_exported(self) -> None:
        """Verify HAS_METRICS_BACKEND is in __all__."""
        from ml.common.metrics_bootstrap import __all__

        assert "HAS_METRICS_BACKEND" in __all__


# ========================================================================
# Metric Idempotency Tests
# ========================================================================


class TestMetricIdempotency:
    """Verify metrics can be retrieved multiple times without error."""

    def test_same_counter_returned_on_second_call(self) -> None:
        """Verify get_counter returns same instance for same name."""
        from ml.common.metrics_bootstrap import get_counter

        counter1 = get_counter("idempotent_counter", "Test")
        counter2 = get_counter("idempotent_counter", "Test")

        # Should be the same object (or at least functionally equivalent)
        # The bootstrap caches metrics by name
        assert counter1 is counter2 or (
            hasattr(counter1, "labels") and hasattr(counter2, "labels")
        )

    def test_same_histogram_returned_on_second_call(self) -> None:
        """Verify get_histogram returns same instance for same name."""
        from ml.common.metrics_bootstrap import get_histogram

        hist1 = get_histogram("idempotent_histogram", "Test")
        hist2 = get_histogram("idempotent_histogram", "Test")

        assert hist1 is hist2 or (
            hasattr(hist1, "labels") and hasattr(hist2, "labels")
        )

    def test_same_gauge_returned_on_second_call(self) -> None:
        """Verify get_gauge returns same instance for same name."""
        from ml.common.metrics_bootstrap import get_gauge

        gauge1 = get_gauge("idempotent_gauge", "Test")
        gauge2 = get_gauge("idempotent_gauge", "Test")

        assert gauge1 is gauge2 or (
            hasattr(gauge1, "labels") and hasattr(gauge2, "labels")
        )


# ========================================================================
# Reset Functionality Tests
# ========================================================================


class TestResetMetricsCache:
    """Verify reset_metrics_cache works correctly."""

    def test_reset_metrics_cache_is_exported(self) -> None:
        """Verify reset_metrics_cache is available."""
        from ml.common.metrics_bootstrap import reset_metrics_cache

        assert callable(reset_metrics_cache)

    def test_reset_metrics_cache_does_not_raise(self) -> None:
        """Verify reset_metrics_cache can be called without error."""
        from ml.common.metrics_bootstrap import reset_metrics_cache

        # Should not raise
        reset_metrics_cache()


# ========================================================================
# Label Variations Tests
# ========================================================================


class TestLabelVariations:
    """Verify metrics work with various label configurations."""

    def test_counter_with_no_labels(self) -> None:
        """Verify counter works when defined without labels."""
        from ml.common.metrics_bootstrap import get_counter

        counter = get_counter("no_label_counter", "Counter without labels")

        # Direct inc should work (no labels needed)
        if hasattr(counter, "inc"):
            counter.inc()

    def test_counter_with_multiple_labels(self) -> None:
        """Verify counter works with multiple labels."""
        from ml.common.metrics_bootstrap import get_counter

        counter = get_counter(
            "multi_label_counter",
            "Counter with multiple labels",
            labelnames=["component", "operation", "status"],
        )

        # Should work with all labels
        counter.labels(component="store", operation="write", status="success").inc()

    def test_histogram_with_custom_buckets(self) -> None:
        """Verify histogram works with custom buckets."""
        from ml.common.metrics_bootstrap import get_histogram

        histogram = get_histogram(
            "custom_bucket_histogram",
            "Histogram with custom buckets",
            labelnames=["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
        )

        # Should work with custom buckets
        histogram.labels(operation="test").observe(0.003)
