"""
Shared scheduler metrics for data pipeline components.

This module centralizes scheduler-level metrics used by the facade components
to avoid circular imports and keep metric instances consistent.
"""

from __future__ import annotations

import logging
from typing import Any

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


logger = logging.getLogger(__name__)


class _NoOpMetric:
    def labels(self, **_: object) -> _NoOpMetric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


feature_store_operations_total: Any = _NoOpMetric()
try:
    from ml.common.metrics import feature_store_operations_total as _feature_store_ops

    feature_store_operations_total = _feature_store_ops
except Exception:
    logger.debug("Prometheus metrics unavailable; using no-op metrics", exc_info=True)


pipeline_stage_latency = get_histogram(
    "nautilus_ml_pipeline_stage_latency_seconds",
    "Pipeline stage execution latency in seconds",
    ["stage"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

pipeline_runs_total = get_counter(
    "nautilus_ml_pipeline_runs_total",
    "Total pipeline runs",
    ["status"],
)

active_feature_tasks = get_gauge(
    "nautilus_ml_active_feature_tasks",
    "Number of active feature computation tasks",
)

data_retention_cleanup_total = get_counter(
    "nautilus_ml_data_retention_cleanup_total",
    "Total data retention cleanup operations",
    ["status"],
)

feature_store_latency = get_histogram(
    "nautilus_ml_feature_store_latency_seconds",
    "Feature store operation latency",
    ["operation"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

feature_computation_errors_total = get_counter(
    "nautilus_ml_feature_computation_errors_total",
    "Total errors during feature computation",
    ["instrument", "error_type"],
)


__all__ = [
    "active_feature_tasks",
    "data_retention_cleanup_total",
    "feature_computation_errors_total",
    "feature_store_latency",
    "feature_store_operations_total",
    "pipeline_runs_total",
    "pipeline_stage_latency",
]
