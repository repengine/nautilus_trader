"""Dashboard metrics snapshot helpers for success criteria validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ml.common.metrics_bootstrap import HAS_METRICS_BACKEND


def _matches_labels(sample: Any, labels: Mapping[str, str] | None) -> bool:
    if labels is None:
        return True
    sample_labels = getattr(sample, "labels", {})
    for key, value in labels.items():
        if sample_labels.get(key) != value:
            return False
    return True


def _counter_total(counter: Any, *, labels: Mapping[str, str] | None = None) -> float:
    if not HAS_METRICS_BACKEND:
        return 0.0
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            name: str = getattr(sample, "name", "")
            if name.endswith("_created"):
                continue
            if not _matches_labels(sample, labels):
                continue
            total += float(sample.value)
    return total


def _histogram_quantile(
    histogram: Any,
    *,
    quantile: float,
    labels: Mapping[str, str] | None = None,
) -> float | None:
    if not HAS_METRICS_BACKEND:
        return None
    buckets: list[tuple[float, float]] = []
    total_count: float | None = None
    for metric in histogram.collect():
        for sample in metric.samples:
            if not _matches_labels(sample, labels):
                continue
            name: str = getattr(sample, "name", "")
            if name.endswith("_bucket"):
                upper_raw = getattr(sample, "labels", {}).get("le")
                if upper_raw is None:
                    continue
                if upper_raw == "+Inf":
                    upper = float("inf")
                else:
                    try:
                        upper = float(upper_raw)
                    except Exception:
                        continue
                buckets.append((upper, float(sample.value)))
            elif name.endswith("_count"):
                total_count = float(sample.value)
    if total_count is None or total_count <= 0.0 or not buckets:
        return None
    buckets.sort(key=lambda item: item[0])
    target = quantile * total_count
    cumulative_prev = 0.0
    lower_bound = 0.0
    for upper_bound, cumulative in buckets:
        if cumulative >= target:
            if upper_bound == float("inf"):
                return None
            bucket_count = cumulative - cumulative_prev
            if bucket_count <= 0.0:
                return upper_bound
            fraction = (target - cumulative_prev) / bucket_count
            width = upper_bound - lower_bound
            return lower_bound + fraction * width
        cumulative_prev = cumulative
        lower_bound = upper_bound
    # If we did not return inside the loop but have finite buckets, use the last bound.
    last_upper = buckets[-1][0]
    return None if last_upper == float("inf") else last_upper


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


@dataclass(slots=True, frozen=True)
class CacheStats:
    hits: float
    misses: float

    @property
    def hit_ratio(self) -> float | None:
        return _safe_ratio(self.hits, self.hits + self.misses)


@dataclass(slots=True, frozen=True)
class RequestStats:
    successes: float
    total: float

    @property
    def success_rate(self) -> float | None:
        return _safe_ratio(self.successes, self.total)


@dataclass(slots=True, frozen=True)
class DashboardMetricsSnapshot:
    registry_cache: CacheStats
    event_cache: CacheStats
    grafana_provisioning: RequestStats
    store_summary_p95_seconds: float | None
    registry_latency_p95_seconds: float | None


def build_dashboard_snapshot(
    *,
    registry_cache_hits: Any,
    registry_cache_misses: Any,
    registry_histogram: Any,
    event_cache_hits: Any,
    event_cache_misses: Any,
    request_counter: Any,
    store_histogram: Any,
) -> DashboardMetricsSnapshot:
    registry_hits_total = _counter_total(registry_cache_hits)
    registry_misses_total = _counter_total(registry_cache_misses)
    event_hits_total = _counter_total(event_cache_hits)
    event_misses_total = _counter_total(event_cache_misses)

    success_total = _counter_total(
        request_counter,
        labels={
            "route": "/api/observability/grafana/provision",
            "method": "POST",
            "status": "success",
        },
    )
    total_requests = success_total + _counter_total(
        request_counter,
        labels={
            "route": "/api/observability/grafana/provision",
            "method": "POST",
            "status": "error",
        },
    ) + _counter_total(
        request_counter,
        labels={
            "route": "/api/observability/grafana/provision",
            "method": "POST",
            "status": "exception",
        },
    )
    store_p95 = _histogram_quantile(
        store_histogram,
        quantile=0.95,
        labels={"operation": "collect"},
    )
    registry_p95 = _histogram_quantile(
        registry_histogram,
        quantile=0.95,
    )
    return DashboardMetricsSnapshot(
        registry_cache=CacheStats(hits=registry_hits_total, misses=registry_misses_total),
        event_cache=CacheStats(hits=event_hits_total, misses=event_misses_total),
        grafana_provisioning=RequestStats(successes=success_total, total=total_requests),
        store_summary_p95_seconds=store_p95,
        registry_latency_p95_seconds=registry_p95,
    )


_REGISTRY_P95_TARGET_SECONDS = 0.2
_EVENT_CACHE_HIT_TARGET = 0.70
_GRAFANA_SUCCESS_TARGET = 0.95
_STORE_P95_TARGET_SECONDS = 0.75


@dataclass(slots=True, frozen=True)
class DashboardSuccessReport:
    registry_latency_p95_seconds: float | None
    event_cache_hit_ratio: float | None
    grafana_success_rate: float | None
    store_summary_p95_seconds: float | None

    @property
    def registry_latency_ok(self) -> bool:
        return (
            self.registry_latency_p95_seconds is not None
            and self.registry_latency_p95_seconds <= _REGISTRY_P95_TARGET_SECONDS
        )

    @property
    def event_cache_hit_ok(self) -> bool:
        return (
            self.event_cache_hit_ratio is not None
            and self.event_cache_hit_ratio >= _EVENT_CACHE_HIT_TARGET
        )

    @property
    def grafana_success_ok(self) -> bool:
        return (
            self.grafana_success_rate is not None
            and self.grafana_success_rate >= _GRAFANA_SUCCESS_TARGET
        )

    @property
    def store_summary_latency_ok(self) -> bool:
        return (
            self.store_summary_p95_seconds is not None
            and self.store_summary_p95_seconds <= _STORE_P95_TARGET_SECONDS
        )

    @property
    def all_ok(self) -> bool:
        return (
            self.registry_latency_ok
            and self.event_cache_hit_ok
            and self.grafana_success_ok
            and self.store_summary_latency_ok
        )


def evaluate_success_criteria(snapshot: DashboardMetricsSnapshot) -> DashboardSuccessReport:
    return DashboardSuccessReport(
        registry_latency_p95_seconds=snapshot.registry_latency_p95_seconds,
        event_cache_hit_ratio=snapshot.event_cache.hit_ratio,
        grafana_success_rate=snapshot.grafana_provisioning.success_rate,
        store_summary_p95_seconds=snapshot.store_summary_p95_seconds,
    )


__all__ = [
    "CacheStats",
    "DashboardMetricsSnapshot",
    "DashboardSuccessReport",
    "RequestStats",
    "build_dashboard_snapshot",
    "evaluate_success_criteria",
]
