"""
Unit tests for MetricsCollectorComponent.

Tests metrics snapshot creation, success criteria evaluation, and edge cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.dashboard.common.metrics_collector import MetricsCollectorComponent
from ml.dashboard.common.metrics_collector import MetricsCollectorProtocol
from ml.dashboard.config import DashboardConfig
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport


@pytest.fixture
def mock_config() -> DashboardConfig:
    """Create mock DashboardConfig for testing."""
    return DashboardConfig(
        actor_port=8000,
        strategy_port=8001,
        pipeline_port=8002,
        prometheus_url="http://localhost:9090",
        grafana_url="http://localhost:3000",
    )


@pytest.fixture
def mock_counters() -> dict[str, Any]:
    """Create mock Prometheus counters."""
    return {
        "registry_cache_hits": MagicMock(),
        "registry_cache_misses": MagicMock(),
        "event_cache_hits": MagicMock(),
        "event_cache_misses": MagicMock(),
        "request_counter": MagicMock(),
    }


@pytest.fixture
def mock_histograms() -> dict[str, Any]:
    """Create mock Prometheus histograms."""
    return {
        "registry_histogram": MagicMock(),
        "store_histogram": MagicMock(),
    }


@pytest.fixture
def collector(
    mock_config: DashboardConfig,
    mock_counters: dict[str, Any],
    mock_histograms: dict[str, Any],
) -> MetricsCollectorComponent:
    """Create MetricsCollectorComponent with mock dependencies."""
    return MetricsCollectorComponent(
        config=mock_config,
        registry_cache_hits=mock_counters["registry_cache_hits"],
        registry_cache_misses=mock_counters["registry_cache_misses"],
        registry_histogram=mock_histograms["registry_histogram"],
        event_cache_hits=mock_counters["event_cache_hits"],
        event_cache_misses=mock_counters["event_cache_misses"],
        request_counter=mock_counters["request_counter"],
        store_histogram=mock_histograms["store_histogram"],
    )


# ============================================================
# Protocol conformance tests
# ============================================================


def test_conforms_to_protocol(collector: MetricsCollectorComponent) -> None:
    """Verify component conforms to MetricsCollectorProtocol."""
    assert isinstance(collector, MetricsCollectorProtocol)


def test_protocol_has_required_methods() -> None:
    """Verify protocol defines required methods."""
    protocol = MetricsCollectorProtocol
    assert hasattr(protocol, "get_metrics_snapshot")
    assert hasattr(protocol, "evaluate_success_criteria")


# ============================================================
# get_metrics_snapshot tests
# ============================================================


def test_get_metrics_snapshot_returns_snapshot(
    collector: MetricsCollectorComponent,
) -> None:
    """Test get_metrics_snapshot returns DashboardMetricsSnapshot."""
    snapshot = collector.get_metrics_snapshot()
    assert isinstance(snapshot, DashboardMetricsSnapshot)


def test_get_metrics_snapshot_contains_cache_stats(
    collector: MetricsCollectorComponent,
) -> None:
    """Test snapshot contains registry and event cache stats."""
    snapshot = collector.get_metrics_snapshot()
    assert hasattr(snapshot, "registry_cache")
    assert hasattr(snapshot, "event_cache")
    assert hasattr(snapshot.registry_cache, "hits")
    assert hasattr(snapshot.registry_cache, "misses")
    assert hasattr(snapshot.event_cache, "hits")
    assert hasattr(snapshot.event_cache, "misses")


def test_get_metrics_snapshot_contains_request_stats(
    collector: MetricsCollectorComponent,
) -> None:
    """Test snapshot contains Grafana provisioning request stats."""
    snapshot = collector.get_metrics_snapshot()
    assert hasattr(snapshot, "grafana_provisioning")
    assert hasattr(snapshot.grafana_provisioning, "successes")
    assert hasattr(snapshot.grafana_provisioning, "total")


def test_get_metrics_snapshot_contains_latency_percentiles(
    collector: MetricsCollectorComponent,
) -> None:
    """Test snapshot contains P95 latency metrics."""
    snapshot = collector.get_metrics_snapshot()
    assert hasattr(snapshot, "store_summary_p95_seconds")
    assert hasattr(snapshot, "registry_latency_p95_seconds")


def test_get_metrics_snapshot_handles_empty_metrics(
    mock_config: DashboardConfig,
) -> None:
    """Test snapshot creation with empty metrics counters."""
    # Create mocks that return no samples
    empty_counter = MagicMock()
    empty_counter.collect.return_value = []
    empty_histogram = MagicMock()
    empty_histogram.collect.return_value = []

    collector = MetricsCollectorComponent(
        config=mock_config,
        registry_cache_hits=empty_counter,
        registry_cache_misses=empty_counter,
        registry_histogram=empty_histogram,
        event_cache_hits=empty_counter,
        event_cache_misses=empty_counter,
        request_counter=empty_counter,
        store_histogram=empty_histogram,
    )

    snapshot = collector.get_metrics_snapshot()
    assert isinstance(snapshot, DashboardMetricsSnapshot)
    assert snapshot.registry_cache.hits == 0.0
    assert snapshot.registry_cache.misses == 0.0
    assert snapshot.event_cache.hits == 0.0
    assert snapshot.event_cache.misses == 0.0


def test_get_metrics_snapshot_is_deterministic(
    collector: MetricsCollectorComponent,
) -> None:
    """Test multiple calls return consistent snapshots."""
    snapshot1 = collector.get_metrics_snapshot()
    snapshot2 = collector.get_metrics_snapshot()

    # Snapshots should have same structure (values may differ if metrics change)
    assert type(snapshot1) is type(snapshot2)
    assert type(snapshot1.registry_cache) is type(snapshot2.registry_cache)
    assert type(snapshot1.event_cache) is type(snapshot2.event_cache)


# ============================================================
# evaluate_success_criteria tests
# ============================================================


def test_evaluate_success_criteria_returns_report(
    collector: MetricsCollectorComponent,
) -> None:
    """Test evaluate_success_criteria returns DashboardSuccessReport."""
    report = collector.evaluate_success_criteria()
    assert isinstance(report, DashboardSuccessReport)


def test_evaluate_success_criteria_contains_all_fields(
    collector: MetricsCollectorComponent,
) -> None:
    """Test success report contains all required fields."""
    report = collector.evaluate_success_criteria()
    assert hasattr(report, "registry_latency_p95_seconds")
    assert hasattr(report, "event_cache_hit_ratio")
    assert hasattr(report, "grafana_success_rate")
    assert hasattr(report, "store_summary_p95_seconds")


def test_evaluate_success_criteria_has_status_properties(
    collector: MetricsCollectorComponent,
) -> None:
    """Test success report has boolean status properties."""
    report = collector.evaluate_success_criteria()
    assert hasattr(report, "registry_latency_ok")
    assert hasattr(report, "event_cache_hit_ok")
    assert hasattr(report, "grafana_success_ok")
    assert hasattr(report, "store_summary_latency_ok")
    assert hasattr(report, "all_ok")


def test_evaluate_success_criteria_all_ok_is_boolean(
    collector: MetricsCollectorComponent,
) -> None:
    """Test all_ok property returns boolean."""
    report = collector.evaluate_success_criteria()
    assert isinstance(report.all_ok, bool)


def test_evaluate_success_criteria_individual_flags_are_boolean(
    collector: MetricsCollectorComponent,
) -> None:
    """Test individual status flags are boolean."""
    report = collector.evaluate_success_criteria()
    assert isinstance(report.registry_latency_ok, bool)
    assert isinstance(report.event_cache_hit_ok, bool)
    assert isinstance(report.grafana_success_ok, bool)
    assert isinstance(report.store_summary_latency_ok, bool)


def test_evaluate_success_criteria_with_empty_metrics(
    mock_config: DashboardConfig,
) -> None:
    """Test success criteria evaluation with no metrics data."""
    empty_counter = MagicMock()
    empty_counter.collect.return_value = []
    empty_histogram = MagicMock()
    empty_histogram.collect.return_value = []

    collector = MetricsCollectorComponent(
        config=mock_config,
        registry_cache_hits=empty_counter,
        registry_cache_misses=empty_counter,
        registry_histogram=empty_histogram,
        event_cache_hits=empty_counter,
        event_cache_misses=empty_counter,
        request_counter=empty_counter,
        store_histogram=empty_histogram,
    )

    report = collector.evaluate_success_criteria()
    assert isinstance(report, DashboardSuccessReport)
    # With no metrics, most criteria will fail (None values)
    assert report.registry_latency_p95_seconds is None
    assert report.event_cache_hit_ratio is None
    assert report.grafana_success_rate is None
    assert report.store_summary_p95_seconds is None


def test_evaluate_success_criteria_is_deterministic(
    collector: MetricsCollectorComponent,
) -> None:
    """Test multiple evaluations return consistent reports."""
    report1 = collector.evaluate_success_criteria()
    report2 = collector.evaluate_success_criteria()

    # Reports should have same structure and values (if metrics unchanged)
    assert type(report1) is type(report2)
    assert report1.registry_latency_p95_seconds == report2.registry_latency_p95_seconds
    assert report1.event_cache_hit_ratio == report2.event_cache_hit_ratio
    assert report1.grafana_success_rate == report2.grafana_success_rate
    assert report1.store_summary_p95_seconds == report2.store_summary_p95_seconds


# ============================================================
# Edge case and error handling tests
# ============================================================


def test_collector_creation_with_minimal_config() -> None:
    """Test collector can be created with minimal configuration."""
    minimal_config = DashboardConfig(
        actor_port=8000,
        strategy_port=8001,
        pipeline_port=8002,
    )

    mock_counter = MagicMock()
    mock_counter.collect.return_value = []
    mock_histogram = MagicMock()
    mock_histogram.collect.return_value = []

    collector = MetricsCollectorComponent(
        config=minimal_config,
        registry_cache_hits=mock_counter,
        registry_cache_misses=mock_counter,
        registry_histogram=mock_histogram,
        event_cache_hits=mock_counter,
        event_cache_misses=mock_counter,
        request_counter=mock_counter,
        store_histogram=mock_histogram,
    )

    assert collector is not None
    assert isinstance(collector.config, DashboardConfig)


def test_collector_stores_all_dependencies(
    collector: MetricsCollectorComponent,
    mock_counters: dict[str, Any],
    mock_histograms: dict[str, Any],
) -> None:
    """Test collector stores references to all metric dependencies."""
    assert collector.registry_cache_hits is mock_counters["registry_cache_hits"]
    assert collector.registry_cache_misses is mock_counters["registry_cache_misses"]
    assert collector.event_cache_hits is mock_counters["event_cache_hits"]
    assert collector.event_cache_misses is mock_counters["event_cache_misses"]
    assert collector.request_counter is mock_counters["request_counter"]
    assert collector.registry_histogram is mock_histograms["registry_histogram"]
    assert collector.store_histogram is mock_histograms["store_histogram"]


def test_collector_config_is_accessible(
    collector: MetricsCollectorComponent,
    mock_config: DashboardConfig,
) -> None:
    """Test collector provides access to configuration."""
    assert collector.config is mock_config
    assert collector.config.actor_port == 8000
    assert collector.config.strategy_port == 8001


def test_snapshot_cache_hit_ratio_calculation() -> None:
    """Test cache hit ratio is correctly calculated in snapshot."""
    from ml.dashboard.metrics_snapshot import CacheStats

    # Test with hits and misses
    stats = CacheStats(hits=70.0, misses=30.0)
    assert stats.hit_ratio == 0.7

    # Test with only hits
    stats_only_hits = CacheStats(hits=100.0, misses=0.0)
    assert stats_only_hits.hit_ratio == 1.0

    # Test with only misses
    stats_only_misses = CacheStats(hits=0.0, misses=100.0)
    assert stats_only_misses.hit_ratio == 0.0

    # Test with no data
    stats_empty = CacheStats(hits=0.0, misses=0.0)
    assert stats_empty.hit_ratio is None


def test_snapshot_success_rate_calculation() -> None:
    """Test success rate is correctly calculated in snapshot."""
    from ml.dashboard.metrics_snapshot import RequestStats

    # Test with successes and total
    stats = RequestStats(successes=95.0, total=100.0)
    assert stats.success_rate == 0.95

    # Test with all successes
    stats_all_success = RequestStats(successes=100.0, total=100.0)
    assert stats_all_success.success_rate == 1.0

    # Test with no successes
    stats_no_success = RequestStats(successes=0.0, total=100.0)
    assert stats_no_success.success_rate == 0.0

    # Test with no requests
    stats_empty = RequestStats(successes=0.0, total=0.0)
    assert stats_empty.success_rate is None


# ============================================================
# Integration-style tests
# ============================================================


def test_full_workflow_snapshot_to_evaluation(
    collector: MetricsCollectorComponent,
) -> None:
    """Test complete workflow from snapshot creation to evaluation."""
    # Get snapshot
    snapshot = collector.get_metrics_snapshot()
    assert isinstance(snapshot, DashboardMetricsSnapshot)

    # Evaluate success criteria
    report = collector.evaluate_success_criteria()
    assert isinstance(report, DashboardSuccessReport)

    # Verify report reflects snapshot data
    assert report.registry_latency_p95_seconds == snapshot.registry_latency_p95_seconds
    assert report.event_cache_hit_ratio == snapshot.event_cache.hit_ratio
    assert report.grafana_success_rate == snapshot.grafana_provisioning.success_rate
    assert report.store_summary_p95_seconds == snapshot.store_summary_p95_seconds
