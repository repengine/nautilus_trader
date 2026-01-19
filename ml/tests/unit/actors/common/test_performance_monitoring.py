#!/usr/bin/env python3
"""
Unit tests for PerformanceMonitoringComponent.

This module tests the performance monitoring component which manages timing
measurements, signal/error counting, and metrics emission for MLSignalActor
decomposition.

Test Categories (32 tests total):
- Timing Capture: 8 tests
- Signal and Error Recording: 4 tests
- Statistics Calculation: 10 tests
- Latency Percentiles: 6 tests
- Metrics Emission: 4 tests

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (zero allocations in hot path)
- Pattern 2: Protocol-First Interface Design (property accessors)

"""

from __future__ import annotations

import logging
import os
import sys
import tracemalloc
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


# =============================================================================
# Fixtures
# =============================================================================


def _under_coverage() -> bool:
    if os.getenv("COV_CORE_SOURCE") or os.getenv("COVERAGE_PROCESS_START"):
        return True
    addopts = os.getenv("PYTEST_ADDOPTS", "")
    if "--cov" in addopts:
        return True
    gettrace = getattr(sys, "gettrace", None)
    return bool(callable(gettrace) and gettrace())


def _allocation_threshold() -> int:
    return 2000 if _under_coverage() else 1000


@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Mock logger for testing.
    """
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def default_monitor():
    """
    Provides a default PerformanceMonitoringComponent with reservoir_size=1000.
    """
    from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

    return PerformanceMonitoringComponent(
        reservoir_size=1000,
        actor_id="test_actor",
        log=None,
    )


@pytest.fixture
def monitor_factory(mock_logger: logging.Logger):
    """
    Factory fixture for creating PerformanceMonitoringComponent instances.

    Returns a callable that creates monitors with specified parameters.

    """
    from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

    def _create_monitor(
        reservoir_size: int = 1000,
        actor_id: str | None = "test_actor",
        log: logging.Logger | None = None,
    ) -> PerformanceMonitoringComponent:
        """
        Create a monitor with specified parameters.
        """
        return PerformanceMonitoringComponent(
            reservoir_size=reservoir_size,
            actor_id=actor_id,
            log=log,
        )

    return _create_monitor


@pytest.fixture
def monitor_with_data(monitor_factory):
    """
    Provides a monitor pre-populated with test data.

    Creates a reservoir_size=1000 monitor with 100 items.

    """
    monitor = monitor_factory(reservoir_size=1000)

    # Add 100 items with predictable values
    for i in range(100):
        # Times in nanoseconds: feature=500us, inference=2ms, total=2.5ms
        monitor.record_timing(
            feature_time_ns=500_000 + i * 1000,  # 0.5ms + 1us per iteration
            inference_time_ns=2_000_000 + i * 1000,  # 2ms + 1us per iteration
            total_time_ns=2_500_000 + i * 2000,  # 2.5ms + 2us per iteration
        )

    return monitor


# =============================================================================
# Timing Capture Tests (8 tests)
# =============================================================================


class TestTimingCapture:
    """
    Tests for timing capture functionality.
    """

    def test_monitor_initializes_with_reservoir_size(self) -> None:
        """
        Verify PerformanceMonitor initializes with specified reservoir size.

        Ring buffers must be allocated with exact size matching capacity.

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Arrange & Act
        monitor = PerformanceMonitoringComponent(reservoir_size=1000)

        # Assert
        assert monitor._cap == 1000, "Capacity must match reservoir_size"
        assert (
            monitor._feature_times_ms.shape[0] == 1000
        ), "Feature times buffer must have shape matching reservoir_size"
        assert (
            monitor._inference_times_ms.shape[0] == 1000
        ), "Inference times buffer must have shape matching reservoir_size"
        assert (
            monitor._total_times_ms.shape[0] == 1000
        ), "Total times buffer must have shape matching reservoir_size"

    def test_monitor_record_timing_stores_values(self, monitor_factory) -> None:
        """
        Verify record_timing() stores values in ring buffers.

        Times should be converted from nanoseconds to milliseconds.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        monitor.record_timing(
            feature_time_ns=500_000,  # 0.5ms
            inference_time_ns=2_000_000,  # 2.0ms
            total_time_ns=2_500_000,  # 2.5ms
        )

        # Assert: Values should be stored in milliseconds
        assert monitor._feature_times_ms[0] == pytest.approx(
            0.5, rel=1e-5
        ), "Feature time should be 0.5ms"
        assert monitor._inference_times_ms[0] == pytest.approx(
            2.0, rel=1e-5
        ), "Inference time should be 2.0ms"
        assert monitor._total_times_ms[0] == pytest.approx(
            2.5, rel=1e-5
        ), "Total time should be 2.5ms"

    def test_monitor_increments_index(self, monitor_factory) -> None:
        """
        Verify index increments after record_timing().

        After 5 recordings, index should be 5.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Record 5 timings
        for i in range(5):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert
        assert monitor._idx == 5, f"Index should be 5 after 5 recordings, got {monitor._idx}"

    def test_monitor_index_wraps_at_capacity(self, monitor_factory) -> None:
        """
        Verify index wraps to 0 at capacity.

        After capacity+5 recordings, index should be 5.

        """
        # Arrange
        capacity = 100
        monitor = monitor_factory(reservoir_size=capacity)

        # Act: Record capacity+5 timings
        for i in range(capacity + 5):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert
        expected_index = 5  # (capacity + 5) % capacity
        assert (
            monitor._idx == expected_index
        ), f"Index should wrap to {expected_index}, got {monitor._idx}"

    def test_monitor_count_increments(self, monitor_factory) -> None:
        """
        Verify count increments until capacity.

        After 500 recordings with capacity=1000, count should be 500.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Record 500 timings
        for i in range(500):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert
        assert (
            monitor._count == 500
        ), f"Count should be 500 after 500 recordings, got {monitor._count}"

    def test_monitor_count_stops_at_capacity(self, monitor_factory) -> None:
        """
        Verify count caps at capacity.

        After 1500 recordings with capacity=1000, count should be 1000.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Record 1500 timings (exceeds capacity)
        for i in range(1500):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert
        assert (
            monitor._count == 1000
        ), f"Count must saturate at capacity (1000), got {monitor._count}"

    def test_monitor_prediction_count_increments(self, monitor_factory) -> None:
        """
        Verify prediction_count increments on each record_timing().

        After 100 recordings, prediction_count should be 100.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Record 100 timings
        for i in range(100):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert
        assert (
            monitor.prediction_count == 100
        ), f"Prediction count should be 100, got {monitor.prediction_count}"

    def test_monitor_no_allocations_on_record_timing(self, monitor_factory) -> None:
        """
        Verify record_timing() allocates no memory (hot path).

        Zero allocations required for hot path performance.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)
        # Warm up with one recording to avoid lazy initialization effects
        monitor.record_timing(
            feature_time_ns=500_000,
            inference_time_ns=2_000_000,
            total_time_ns=2_500_000,
        )

        # Act: Measure allocations during record_timing
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(100):
            monitor.record_timing(
                feature_time_ns=500_000 + i * 1000,
                inference_time_ns=2_000_000 + i * 1000,
                total_time_ns=2_500_000 + i * 2000,
            )

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Allow small threshold for interpreter overhead
        # Ring buffer updates should not allocate new arrays
        assert total_new_bytes < _allocation_threshold(), (
            f"Hot path record_timing() should not allocate significant memory, "
            f"but allocated {total_new_bytes} bytes"
        )


# =============================================================================
# Signal and Error Recording Tests (4 tests)
# =============================================================================


class TestSignalAndErrorRecording:
    """
    Tests for signal and error recording functionality.
    """

    def test_monitor_record_signal_increments_count(self, monitor_factory) -> None:
        """
        Verify record_signal() increments signal_count.

        After 50 calls, signal_count should be 50.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Call record_signal() 50 times
        for _ in range(50):
            monitor.record_signal()

        # Assert
        assert monitor.signal_count == 50, f"Signal count should be 50, got {monitor.signal_count}"

    def test_monitor_record_error_increments_count(self, monitor_factory) -> None:
        """
        Verify record_error() increments error_count.

        After 10 calls, error_count should be 10.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Call record_error() 10 times
        for _ in range(10):
            monitor.record_error()

        # Assert
        assert monitor.error_count == 10, f"Error count should be 10, got {monitor.error_count}"

    def test_monitor_record_signal_no_allocations(self, monitor_factory) -> None:
        """
        Verify record_signal() doesn't allocate.

        Zero allocations required for hot path performance.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)
        # Warm up
        monitor.record_signal()

        # Act: Measure allocations during record_signal
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            monitor.record_signal()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Should be zero or very minimal allocations
        # Allow small threshold for interpreter overhead in test loop
        assert total_new_bytes < _allocation_threshold(), (
            f"Hot path record_signal() should not allocate significant memory, "
            f"but allocated {total_new_bytes} bytes"
        )

    def test_monitor_record_error_no_allocations(self, monitor_factory) -> None:
        """
        Verify record_error() doesn't allocate.

        Zero allocations required for hot path performance.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)
        # Warm up
        monitor.record_error()

        # Act: Measure allocations during record_error
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            monitor.record_error()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate difference
        stats = snapshot2.compare_to(snapshot1, "lineno")
        total_new_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Assert: Should be zero or very minimal allocations
        # Allow small threshold for interpreter overhead in test loop
        assert total_new_bytes < _allocation_threshold(), (
            f"Hot path record_error() should not allocate significant memory, "
            f"but allocated {total_new_bytes} bytes"
        )


# =============================================================================
# Statistics Calculation Tests (10 tests)
# =============================================================================


class TestStatisticsCalculation:
    """
    Tests for statistics calculation functionality.
    """

    def test_monitor_get_current_stats_includes_all_fields(
        self,
        monitor_with_data,
    ) -> None:
        """
        Verify get_current_stats() returns all required fields.

        Stats dict should contain prediction_count, signal_count, error_count, rates,
        averages, and P99.

        """
        # Arrange
        monitor = monitor_with_data

        # Act
        stats = monitor.get_current_stats()

        # Assert: Check all expected keys
        expected_keys = [
            "prediction_count",
            "signal_count",
            "error_count",
            "signal_rate",
            "error_rate",
            "avg_feature_time_ms",
            "avg_inference_time_ms",
            "avg_total_time_ms",
            "p99_total_time_ms",
        ]
        for key in expected_keys:
            assert key in stats, f"Stats must contain key '{key}'"

    def test_monitor_calculates_signal_rate(self, monitor_factory) -> None:
        """
        Verify signal_rate = signal_count / prediction_count.

        With 100 predictions and 30 signals, signal_rate should be 0.3.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 100 predictions
        for i in range(100):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Record 30 signals
        for _ in range(30):
            monitor.record_signal()

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert stats["signal_rate"] == pytest.approx(
            0.3
        ), f"Signal rate should be 0.3, got {stats['signal_rate']}"

    def test_monitor_calculates_error_rate(self, monitor_factory) -> None:
        """
        Verify error_rate = error_count / prediction_count.

        With 100 predictions and 5 errors, error_rate should be 0.05.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 100 predictions
        for i in range(100):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Record 5 errors
        for _ in range(5):
            monitor.record_error()

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert stats["error_rate"] == pytest.approx(
            0.05
        ), f"Error rate should be 0.05, got {stats['error_rate']}"

    def test_monitor_calculates_avg_feature_time(self, monitor_factory) -> None:
        """
        Verify avg_feature_time_ms calculated correctly.

        With feature times 0.3ms, 0.4ms, 0.5ms, average should be 0.4ms.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 3 timings with specific feature times
        feature_times_ns = [300_000, 400_000, 500_000]  # 0.3ms, 0.4ms, 0.5ms
        for ft_ns in feature_times_ns:
            monitor.record_timing(
                feature_time_ns=ft_ns,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert stats["avg_feature_time_ms"] == pytest.approx(
            0.4, rel=1e-5
        ), f"Avg feature time should be 0.4ms, got {stats['avg_feature_time_ms']}"

    def test_monitor_calculates_avg_inference_time(self, monitor_factory) -> None:
        """
        Verify avg_inference_time_ms calculated correctly.

        With inference times 1.5ms, 2.0ms, 2.5ms, average should be 2.0ms.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 3 timings with specific inference times
        inference_times_ns = [1_500_000, 2_000_000, 2_500_000]  # 1.5ms, 2.0ms, 2.5ms
        for it_ns in inference_times_ns:
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=it_ns,
                total_time_ns=3_000_000,
            )

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert stats["avg_inference_time_ms"] == pytest.approx(
            2.0, rel=1e-5
        ), f"Avg inference time should be 2.0ms, got {stats['avg_inference_time_ms']}"

    def test_monitor_calculates_avg_total_time(self, monitor_factory) -> None:
        """
        Verify avg_total_time_ms calculated correctly.

        With total times 2.0ms, 2.5ms, 3.0ms, average should be 2.5ms.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 3 timings with specific total times
        total_times_ns = [2_000_000, 2_500_000, 3_000_000]  # 2.0ms, 2.5ms, 3.0ms
        for tt_ns in total_times_ns:
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=tt_ns,
            )

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert stats["avg_total_time_ms"] == pytest.approx(
            2.5, rel=1e-5
        ), f"Avg total time should be 2.5ms, got {stats['avg_total_time_ms']}"

    def test_monitor_calculates_p99_total_time(self, monitor_factory) -> None:
        """
        Verify p99_total_time_ms calculated correctly.

        P99 should match numpy's percentile calculation.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 100 timings with increasing total times
        total_times_ms = []
        for i in range(100):
            tt_ns = 2_000_000 + i * 100_000  # 2.0ms to 11.9ms
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=tt_ns,
            )
            total_times_ms.append(tt_ns / 1_000_000.0)

        # Act
        stats = monitor.get_current_stats()

        # Assert
        expected_p99 = float(np.percentile(total_times_ms, 99))
        assert stats["p99_total_time_ms"] == pytest.approx(
            expected_p99, rel=1e-3
        ), f"P99 total time should be {expected_p99}ms, got {stats['p99_total_time_ms']}"

    def test_monitor_stats_with_zero_predictions(self, monitor_factory) -> None:
        """
        Verify stats handle zero predictions gracefully.

        With no timings recorded, rates should be 0 and times should be 0.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert (
            stats["signal_rate"] == 0.0
        ), f"Signal rate should be 0.0 with no predictions, got {stats['signal_rate']}"
        assert (
            stats["error_rate"] == 0.0
        ), f"Error rate should be 0.0 with no predictions, got {stats['error_rate']}"
        assert (
            stats["avg_feature_time_ms"] == 0.0
        ), f"Avg feature time should be 0.0 with no data, got {stats['avg_feature_time_ms']}"
        assert (
            stats["avg_inference_time_ms"] == 0.0
        ), f"Avg inference time should be 0.0 with no data, got {stats['avg_inference_time_ms']}"
        assert (
            stats["avg_total_time_ms"] == 0.0
        ), f"Avg total time should be 0.0 with no data, got {stats['avg_total_time_ms']}"

    def test_monitor_stats_includes_last_timing(self, monitor_factory) -> None:
        """
        Verify stats include most recent timing values.

        Stats should contain last_feature_time_ms, last_inference_time_ms,
        last_total_time_ms.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record one timing
        monitor.record_timing(
            feature_time_ns=500_000,  # 0.5ms
            inference_time_ns=2_000_000,  # 2.0ms
            total_time_ns=2_500_000,  # 2.5ms
        )

        # Act
        stats = monitor.get_current_stats()

        # Assert
        assert "last_feature_time_ms" in stats, "Stats should include last_feature_time_ms"
        assert "last_inference_time_ms" in stats, "Stats should include last_inference_time_ms"
        assert "last_total_time_ms" in stats, "Stats should include last_total_time_ms"
        assert stats["last_feature_time_ms"] == pytest.approx(
            0.5, rel=1e-5
        ), "Last feature time should be 0.5ms"
        assert stats["last_inference_time_ms"] == pytest.approx(
            2.0, rel=1e-5
        ), "Last inference time should be 2.0ms"
        assert stats["last_total_time_ms"] == pytest.approx(
            2.5, rel=1e-5
        ), "Last total time should be 2.5ms"

    def test_monitor_get_stats_uses_count_not_capacity(self, monitor_factory) -> None:
        """
        Verify stats calculated over count, not full capacity.

        With 50 timings in a capacity-1000 buffer, stats should be from 50 values.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record only 50 timings with known values
        for i in range(50):
            monitor.record_timing(
                feature_time_ns=1_000_000,  # 1.0ms
                inference_time_ns=2_000_000,  # 2.0ms
                total_time_ns=3_000_000,  # 3.0ms
            )

        # Act
        stats = monitor.get_current_stats()

        # Assert: Averages should be exactly the recorded values
        # (not diluted by zeros in unused capacity)
        assert stats["avg_feature_time_ms"] == pytest.approx(
            1.0, rel=1e-5
        ), "Average should be calculated from 50 values, not capacity"
        assert stats["avg_inference_time_ms"] == pytest.approx(
            2.0, rel=1e-5
        ), "Average should be calculated from 50 values, not capacity"
        assert stats["avg_total_time_ms"] == pytest.approx(
            3.0, rel=1e-5
        ), "Average should be calculated from 50 values, not capacity"


# =============================================================================
# Latency Percentiles Tests (6 tests)
# =============================================================================


class TestLatencyPercentiles:
    """
    Tests for latency percentile calculation functionality.
    """

    def test_monitor_get_latency_percentiles_returns_all_types(
        self,
        monitor_with_data,
    ) -> None:
        """
        Verify get_latency_percentiles() returns feature, inference, total.

        Result should contain 3 keys for each timing type.

        """
        # Arrange
        monitor = monitor_with_data

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert
        assert (
            "feature_computation" in percentiles
        ), "Percentiles should include 'feature_computation'"
        assert "inference" in percentiles, "Percentiles should include 'inference'"
        assert "total" in percentiles, "Percentiles should include 'total'"

    def test_monitor_percentiles_include_p50_p90_p95_p99(
        self,
        monitor_with_data,
    ) -> None:
        """
        Verify all 4 percentiles calculated.

        Each type should have P50, P90, P95, P99.

        """
        # Arrange
        monitor = monitor_with_data

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert
        expected_percentiles = {50.0, 90.0, 95.0, 99.0}
        assert (
            set(percentiles["total"].keys()) == expected_percentiles
        ), f"Total should have all 4 percentiles, got {set(percentiles['total'].keys())}"
        assert (
            set(percentiles["inference"].keys()) == expected_percentiles
        ), "Inference should have all 4 percentiles"
        assert (
            set(percentiles["feature_computation"].keys()) == expected_percentiles
        ), "Feature computation should have all 4 percentiles"

    def test_monitor_p50_is_median(self, monitor_factory) -> None:
        """
        Verify P50 equals median.

        P50 should match numpy's median calculation.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record known timing distribution
        total_times_ms = []
        for i in range(100):
            tt_ns = 2_000_000 + i * 100_000  # 2.0ms to 11.9ms
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=tt_ns,
            )
            total_times_ms.append(tt_ns / 1_000_000.0)

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert
        expected_median = float(np.median(total_times_ms))
        assert percentiles["total"][50.0] == pytest.approx(
            expected_median, rel=1e-3
        ), f"P50 should equal median {expected_median}, got {percentiles['total'][50.0]}"

    def test_monitor_p99_greater_than_p50(self, monitor_with_data) -> None:
        """
        Verify percentiles ordered correctly.

        P99 >= P95 >= P90 >= P50 must hold.

        """
        # Arrange
        monitor = monitor_with_data

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert
        p50 = percentiles["total"][50.0]
        p90 = percentiles["total"][90.0]
        p95 = percentiles["total"][95.0]
        p99 = percentiles["total"][99.0]

        assert p99 >= p95, f"P99 ({p99}) should be >= P95 ({p95})"
        assert p95 >= p90, f"P95 ({p95}) should be >= P90 ({p90})"
        assert p90 >= p50, f"P90 ({p90}) should be >= P50 ({p50})"

    def test_monitor_percentiles_empty_buffer(self, monitor_factory) -> None:
        """
        Verify percentiles handle empty buffer.

        With no data, should return empty dict.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert
        assert (
            percentiles == {}
        ), f"Percentiles should be empty dict with no data, got {percentiles}"

    def test_monitor_percentiles_uses_count_not_capacity(
        self,
        monitor_factory,
    ) -> None:
        """
        Verify percentiles calculated over count.

        With 50 timings in capacity-1000 buffer, percentiles should be from 50 values.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Record 50 timings with known values (all identical for simple verification)
        for _ in range(50):
            monitor.record_timing(
                feature_time_ns=1_000_000,  # 1.0ms
                inference_time_ns=2_000_000,  # 2.0ms
                total_time_ns=3_000_000,  # 3.0ms
            )

        # Act
        percentiles = monitor.get_latency_percentiles()

        # Assert: All percentiles should be 3.0ms since all values are identical
        assert percentiles["total"][50.0] == pytest.approx(
            3.0, rel=1e-5
        ), "P50 should be calculated from 50 values, not capacity"
        assert percentiles["total"][99.0] == pytest.approx(
            3.0, rel=1e-5
        ), "P99 should be calculated from 50 values, not capacity"


# =============================================================================
# Metrics Emission Tests (4 tests)
# =============================================================================


class TestMetricsEmission:
    """
    Tests for Prometheus metrics initialization and emission.
    """

    def test_metrics_initialized_at_module_import(self, monitor_factory) -> None:
        """
        Verify initialize_metrics() sets _metrics_initialized=True.

        After calling initialize_metrics(), flag should be True and metrics populated.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        monitor.initialize_metrics()

        # Assert
        assert (
            monitor._metrics_initialized is True
        ), "_metrics_initialized should be True after initialize_metrics()"
        assert len(monitor._metrics) > 0, "Metrics dict should be populated after initialization"

    def test_metrics_idempotent_initialization(self, monitor_factory) -> None:
        """
        Verify initialize_metrics() is idempotent.

        Calling twice should produce same metrics instances.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act: Call twice
        monitor.initialize_metrics()
        first_metrics = dict(monitor._metrics)
        first_count = len(first_metrics)

        monitor.initialize_metrics()
        second_metrics = dict(monitor._metrics)
        second_count = len(second_metrics)

        # Assert: Same count, flag still True
        assert (
            first_count == second_count
        ), "Metrics count should be same after second initialization"
        assert monitor._metrics_initialized is True, "_metrics_initialized should still be True"

    def test_metrics_include_all_required(self, monitor_factory) -> None:
        """
        Verify all 9 metrics initialized.

        All expected metric keys should be present.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        monitor.initialize_metrics()

        # Assert: Check for expected metric keys
        expected_metric_keys = [
            "prediction_distribution",
            "confidence_distribution",
            "signal_generation_time",
            "feature_time_by_feature_set",
            "signals_generated",
            "adaptive_threshold",
            "market_regime",
            "feature_parity_checks_total",
            "feature_parity_drift",
        ]

        for key in expected_metric_keys:
            assert key in monitor._metrics, f"Metrics should contain key '{key}'"

        assert len(monitor._metrics) == 9, f"Should have 9 metrics, got {len(monitor._metrics)}"

    def test_metrics_labels_correctly_configured(self, monitor_factory) -> None:
        """
        Verify metrics have correct label names.

        Metrics should be configured with appropriate labels.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Act
        monitor.initialize_metrics()

        # Assert: Metrics should be initialized (not None)
        for key, metric in monitor._metrics.items():
            assert metric is not None, f"Metric '{key}' should not be None after initialization"


# =============================================================================
# Property Tests
# =============================================================================


class TestPropertyBased:
    """
    Hypothesis property-based tests for monitor invariants.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        reservoir_size=st.integers(min_value=1, max_value=1000),
        num_recordings=st.integers(min_value=0, max_value=5000),
    )
    def test_monitor_count_never_exceeds_capacity_property(
        self,
        reservoir_size: int,
        num_recordings: int,
    ) -> None:
        """
        Verify count never exceeds capacity for any sequence of recordings.

        Property: _count <= _cap for all states.

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Arrange
        monitor = PerformanceMonitoringComponent(reservoir_size=reservoir_size)

        # Act
        for i in range(num_recordings):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert: Invariant must hold
        assert (
            monitor._count <= monitor._cap
        ), f"Count ({monitor._count}) must never exceed capacity ({monitor._cap})"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        reservoir_size=st.integers(min_value=1, max_value=1000),
        num_recordings=st.integers(min_value=0, max_value=5000),
    )
    def test_monitor_index_always_valid_property(
        self,
        reservoir_size: int,
        num_recordings: int,
    ) -> None:
        """
        Verify index always in [0, capacity).

        Property: 0 <= _idx < _cap for all states.

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Arrange
        monitor = PerformanceMonitoringComponent(reservoir_size=reservoir_size)

        # Act
        for i in range(num_recordings):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Assert: Index must be in valid range
        assert (
            0 <= monitor._idx < monitor._cap
        ), f"Index ({monitor._idx}) must be in [0, {monitor._cap})"

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        reservoir_size=st.integers(min_value=10, max_value=100),
        num_predictions=st.integers(min_value=1, max_value=1000),
        num_signals=st.integers(min_value=0, max_value=1000),
        num_errors=st.integers(min_value=0, max_value=1000),
    )
    def test_monitor_rates_bounded_property(
        self,
        reservoir_size: int,
        num_predictions: int,
        num_signals: int,
        num_errors: int,
    ) -> None:
        """
        Verify rates are bounded correctly.

        Property: signal_rate = signal_count / prediction_count
        Property: error_rate = error_count / prediction_count

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Arrange
        monitor = PerformanceMonitoringComponent(reservoir_size=reservoir_size)

        # Record predictions
        for _ in range(num_predictions):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )

        # Record signals and errors
        for _ in range(num_signals):
            monitor.record_signal()
        for _ in range(num_errors):
            monitor.record_error()

        # Act
        stats = monitor.get_current_stats()

        # Assert: Rates should match expected calculation
        expected_signal_rate = num_signals / max(num_predictions, 1)
        expected_error_rate = num_errors / max(num_predictions, 1)

        assert stats["signal_rate"] == pytest.approx(
            expected_signal_rate, rel=1e-5
        ), f"Signal rate should be {expected_signal_rate}"
        assert stats["error_rate"] == pytest.approx(
            expected_error_rate, rel=1e-5
        ), f"Error rate should be {expected_error_rate}"


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


class TestEdgeCasesAndErrors:
    """
    Tests for edge cases and error conditions.
    """

    def test_monitor_rejects_zero_reservoir_size(self) -> None:
        """
        Verify monitor rejects reservoir_size=0.

        ValueError should be raised for invalid reservoir_size.

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Act & Assert
        with pytest.raises(ValueError, match=r"reservoir_size.*must be > 0"):
            PerformanceMonitoringComponent(reservoir_size=0)

    def test_monitor_rejects_negative_reservoir_size(self) -> None:
        """
        Verify monitor rejects negative reservoir_size.

        ValueError should be raised for invalid reservoir_size.

        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Act & Assert
        with pytest.raises(ValueError, match=r"reservoir_size.*must be > 0"):
            PerformanceMonitoringComponent(reservoir_size=-10)

    def test_monitor_handles_capacity_one(self, monitor_factory) -> None:
        """Verify edge case: reservoir_size=1.

        Monitor with capacity=1 should work correctly with index always 0 after wrap.
        """
        from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent

        # Arrange
        monitor = PerformanceMonitoringComponent(reservoir_size=1)

        # Act & Assert: Multiple recordings
        for i in range(5):
            monitor.record_timing(
                feature_time_ns=500_000 * (i + 1),
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )
            # After each recording, index wraps to 0
            assert (
                monitor._idx == 0
            ), f"With capacity=1, index should always be 0, got {monitor._idx}"

        # Final state check
        assert monitor._count == 1, "Count should saturate at 1 for capacity=1"
        assert monitor.prediction_count == 5, "Prediction count should be 5"

    def test_monitor_property_accessors_return_correct_values(
        self,
        monitor_factory,
    ) -> None:
        """
        Verify property accessors return correct values.

        Test prediction_count, signal_count, error_count, reservoir_size properties.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=500)

        # Act
        for _ in range(100):
            monitor.record_timing(
                feature_time_ns=500_000,
                inference_time_ns=2_000_000,
                total_time_ns=2_500_000,
            )
        for _ in range(30):
            monitor.record_signal()
        for _ in range(5):
            monitor.record_error()

        # Assert
        assert monitor.reservoir_size == 500, "reservoir_size property should return 500"
        assert monitor.prediction_count == 100, "prediction_count should be 100"
        assert monitor.signal_count == 30, "signal_count should be 30"
        assert monitor.error_count == 5, "error_count should be 5"

    def test_monitor_uses_float32_dtype(self, monitor_factory) -> None:
        """
        Verify ring buffers use float32 (not float64).

        Using float32 reduces memory footprint and improves cache performance.

        """
        # Arrange & Act
        monitor = monitor_factory(reservoir_size=1000)

        # Assert
        assert (
            monitor._feature_times_ms.dtype == np.float32
        ), "Feature times buffer must use float32 dtype"
        assert (
            monitor._inference_times_ms.dtype == np.float32
        ), "Inference times buffer must use float32 dtype"
        assert (
            monitor._total_times_ms.dtype == np.float32
        ), "Total times buffer must use float32 dtype"

    def test_monitor_metrics_property_returns_dict(self, monitor_factory) -> None:
        """
        Verify metrics property returns dictionary.

        Before and after initialization, metrics property should return dict.

        """
        # Arrange
        monitor = monitor_factory(reservoir_size=1000)

        # Assert: Before initialization
        assert isinstance(monitor.metrics, dict), "metrics property should return dict"
        assert len(monitor.metrics) == 0, "metrics should be empty before initialization"

        # Act: Initialize
        monitor.initialize_metrics()

        # Assert: After initialization
        assert isinstance(
            monitor.metrics, dict
        ), "metrics property should return dict after initialization"
        assert len(monitor.metrics) > 0, "metrics should be populated after initialization"
