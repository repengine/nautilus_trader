from __future__ import annotations

import pytest

from typing import Any

from ml.dashboard.config import DashboardConfig
from ml.dashboard.metrics_snapshot import CacheStats
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import RequestStats
from ml.dashboard.metrics_snapshot import evaluate_success_criteria
from ml.dashboard.service import DashboardService

pytestmark = pytest.mark.usefixtures("mock_tracing_backend")


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


@pytest.fixture(autouse=True)
def _isolated_prom_registry(isolated_prometheus_registry: Any) -> None:
    """Ensure Prometheus collectors are isolated between tests."""
    del isolated_prometheus_registry


def test_cache_and_request_stats_ratio() -> None:
    cache_stats = CacheStats(hits=7.0, misses=3.0)
    assert cache_stats.hit_ratio == pytest.approx(0.7)
    zero_stats = CacheStats(hits=0.0, misses=0.0)
    assert zero_stats.hit_ratio is None

    request_stats = RequestStats(successes=19.0, total=20.0)
    assert request_stats.success_rate == pytest.approx(0.95)
    empty_requests = RequestStats(successes=0.0, total=0.0)
    assert empty_requests.success_rate is None


def test_dashboard_metrics_snapshot_reports_deltas() -> None:
    svc = DashboardService.from_config(DashboardConfig())

    before = svc.get_metrics_snapshot()

    # Simulate cache hits/misses and Grafana provisioning outcomes.
    from ml.dashboard.service import _EVENT_CACHE_HITS as EVENT_HITS
    from ml.dashboard.service import _EVENT_CACHE_MISSES as EVENT_MISSES
    from ml.dashboard.service import _REGISTRY_CACHE_HITS as REGISTRY_HITS
    from ml.dashboard.service import _REGISTRY_CACHE_MISSES as REGISTRY_MISSES
    from ml.dashboard.service import _REGISTRY_LATENCY_SECONDS as REGISTRY_HISTOGRAM
    from ml.dashboard.service import _REQS_TOTAL as REQUESTS
    from ml.dashboard.service import _STORE_SUMMARY_SECONDS as STORE_HISTOGRAM

    svc._cached_registry_call(key="snapshot-test", fetch=lambda: ["value"])
    svc._cached_registry_call(key="snapshot-test", fetch=lambda: ["value"])
    REGISTRY_HITS.labels(entry="snapshot-test").inc(4.0)
    REGISTRY_MISSES.labels(entry="snapshot-test").inc(2.0)
    EVENT_HITS.inc(5.0)
    EVENT_MISSES.inc(2.0)
    REQUESTS.labels(route="/api/observability/grafana/provision", method="POST", status="success").inc(5.0)
    REQUESTS.labels(route="/api/observability/grafana/provision", method="POST", status="error").inc(1.0)
    STORE_HISTOGRAM.labels(operation="collect").observe(0.2)
    REGISTRY_HISTOGRAM.labels(entry="snapshot-test").observe(0.01)

    after = svc.get_metrics_snapshot()

    registry_hits_delta = _delta(before.registry_cache.hits, after.registry_cache.hits)
    assert registry_hits_delta is not None and registry_hits_delta >= 1.0

    event_hit_delta = _delta(before.event_cache.hits, after.event_cache.hits)
    event_miss_delta = _delta(before.event_cache.misses, after.event_cache.misses)
    assert event_hit_delta is not None and event_hit_delta >= 1.0
    assert event_miss_delta is not None and event_miss_delta >= 1.0

    success_delta = _delta(before.grafana_provisioning.successes, after.grafana_provisioning.successes)
    total_delta = _delta(before.grafana_provisioning.total, after.grafana_provisioning.total)
    assert success_delta is not None and success_delta >= 1.0
    assert total_delta is not None and total_delta >= 2.0

    assert after.store_summary_p95_seconds is None or after.store_summary_p95_seconds >= 0.0
    assert after.registry_latency_p95_seconds is None or after.registry_latency_p95_seconds >= 0.0


def test_evaluate_success_criteria() -> None:
    snapshot = DashboardMetricsSnapshot(
        registry_cache=CacheStats(hits=80.0, misses=10.0),
        event_cache=CacheStats(hits=75.0, misses=15.0),
        grafana_provisioning=RequestStats(successes=95.0, total=100.0),
        store_summary_p95_seconds=0.5,
        registry_latency_p95_seconds=0.1,
    )
    report = evaluate_success_criteria(snapshot)
    assert report.all_ok
    assert report.registry_latency_ok
    assert report.event_cache_hit_ok
    assert report.grafana_success_ok
    assert report.store_summary_latency_ok

    degraded_snapshot = DashboardMetricsSnapshot(
        registry_cache=CacheStats(hits=1.0, misses=9.0),
        event_cache=CacheStats(hits=1.0, misses=9.0),
        grafana_provisioning=RequestStats(successes=1.0, total=10.0),
        store_summary_p95_seconds=1.0,
        registry_latency_p95_seconds=0.5,
    )
    degraded_report = evaluate_success_criteria(degraded_snapshot)
    assert not degraded_report.all_ok
    assert not degraded_report.registry_latency_ok
    assert not degraded_report.event_cache_hit_ok
    assert not degraded_report.grafana_success_ok
    assert not degraded_report.store_summary_latency_ok


def test_service_evaluate_success_criteria_reports_pass() -> None:
    svc = DashboardService.from_config(DashboardConfig())

    from ml.dashboard.service import _EVENT_CACHE_HITS as EVENT_HITS
    from ml.dashboard.service import _EVENT_CACHE_MISSES as EVENT_MISSES
    from ml.dashboard.service import _REGISTRY_CACHE_HITS as REGISTRY_HITS
    from ml.dashboard.service import _REGISTRY_LATENCY_SECONDS as REGISTRY_LATENCY
    from ml.dashboard.service import _REQS_TOTAL as REQUESTS
    from ml.dashboard.service import _STORE_SUMMARY_SECONDS as STORE_HISTOGRAM

    REGISTRY_HITS.labels(entry="criteria").inc(1_000_000.0)
    REGISTRY_LATENCY.labels(entry="criteria").observe(0.05)

    EVENT_HITS.inc(1_000_000.0)
    EVENT_MISSES.inc(1_000.0)

    REQUESTS.labels(route="/api/observability/grafana/provision", method="POST", status="success").inc(1_000_000.0)
    REQUESTS.labels(route="/api/observability/grafana/provision", method="POST", status="error").inc(1_000.0)

    STORE_HISTOGRAM.labels(operation="collect").observe(0.5)

    report = svc.evaluate_success_criteria()
    assert report.registry_latency_ok
    assert report.event_cache_hit_ok
    assert report.grafana_success_ok
    assert report.store_summary_latency_ok
    assert report.all_ok
