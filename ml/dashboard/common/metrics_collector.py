"""
Metrics collector component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Aggregates dashboard metrics and evaluates success criteria against thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport
from ml.dashboard.metrics_snapshot import build_dashboard_snapshot
from ml.dashboard.metrics_snapshot import evaluate_success_criteria


if TYPE_CHECKING:
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection operations."""

    def get_metrics_snapshot(self) -> DashboardMetricsSnapshot:
        """Get aggregated metrics snapshot for dashboard operations."""
        ...

    def evaluate_success_criteria(self) -> DashboardSuccessReport:
        """Evaluate success criteria against observed metrics."""
        ...


@dataclass
class MetricsCollectorComponent:
    """
    Component for collecting and evaluating dashboard metrics.

    Extracted from DashboardService to follow single-responsibility principle.
    Aggregates metrics from Prometheus counters/histograms and evaluates against
    predefined success thresholds.
    """

    config: DashboardConfig
    registry_cache_hits: Any
    registry_cache_misses: Any
    registry_histogram: Any
    event_cache_hits: Any
    event_cache_misses: Any
    request_counter: Any
    store_histogram: Any

    def get_metrics_snapshot(self) -> DashboardMetricsSnapshot:
        """
        Return aggregated dashboard metrics useful for success criteria validation.

        Collects metrics from Prometheus counters and histograms to build a snapshot
        containing cache statistics, request statistics, and latency percentiles.

        Returns:
            DashboardMetricsSnapshot containing aggregated metrics.

        Example:
            >>> collector = MetricsCollectorComponent(config, ...)
            >>> snapshot = collector.get_metrics_snapshot()
            >>> assert snapshot.registry_cache.hits >= 0.0
            >>> assert snapshot.event_cache.misses >= 0.0
        """
        return build_dashboard_snapshot(
            registry_cache_hits=self.registry_cache_hits,
            registry_cache_misses=self.registry_cache_misses,
            registry_histogram=self.registry_histogram,
            event_cache_hits=self.event_cache_hits,
            event_cache_misses=self.event_cache_misses,
            request_counter=self.request_counter,
            store_histogram=self.store_histogram,
        )

    def evaluate_success_criteria(self) -> DashboardSuccessReport:
        """
        Evaluate dashboard success criteria using observed metrics.

        Builds a metrics snapshot and evaluates it against predefined thresholds:
        - Registry latency P95 <= 0.2s
        - Event cache hit ratio >= 70%
        - Grafana provisioning success rate >= 95%
        - Store summary latency P95 <= 0.75s

        Returns:
            DashboardSuccessReport containing evaluation results and status flags.

        Example:
            >>> collector = MetricsCollectorComponent(config, ...)
            >>> report = collector.evaluate_success_criteria()
            >>> assert hasattr(report, "all_ok")
            >>> assert hasattr(report, "registry_latency_ok")
            >>> assert hasattr(report, "event_cache_hit_ok")
        """
        snapshot = self.get_metrics_snapshot()
        return evaluate_success_criteria(snapshot)


__all__ = [
    "MetricsCollectorComponent",
    "MetricsCollectorProtocol",
]
