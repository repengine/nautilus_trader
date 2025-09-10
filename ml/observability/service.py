"""
Thin Observability service façade (off hot-path).

This module provides a minimal, typed service to collect observability rows
and materialize Pandas DataFrames using the DTO builders in
``ml.observability.pipeline``. All heavy work remains off the hot path; hot
loops should only record cheap counters/histograms via metrics bootstrap.

"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.observability.pipeline import build_event_correlation
from ml.observability.pipeline import build_health_scores
from ml.observability.pipeline import build_latency_watermarks
from ml.observability.pipeline import build_metrics_collection


class ObservabilityService:
    """
    Collects observability rows and builds contract-compliant DataFrames.

    Notes
    -----
    - Keep this service off the hot path; use it in background tasks or
      batch workflows. In hot loops, prefer direct metric observation via
      ``ml.common.metrics_bootstrap``.
    - The service does not persist; callers own storage concerns.

    """

    def __init__(self) -> None:
        self._latency_rows: list[dict[str, Any]] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._corr_rows: list[dict[str, Any]] = []
        self._health_rows: list[dict[str, Any]] = []

    # ----------------------- Row collection (cheap) -----------------------

    def add_latency_stage(
        self,
        *,
        correlation_id: str,
        instrument_id: str,
        pipeline_stage: str,
        ts_stage_start: int,
        ts_stage_end: int,
    ) -> None:
        self._latency_rows.append(
            {
                "correlation_id": correlation_id,
                "instrument_id": instrument_id,
                "pipeline_stage": pipeline_stage,
                "ts_stage_start": int(ts_stage_start),
                "ts_stage_end": int(ts_stage_end),
            },
        )

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, Any] | str | None = None,
    ) -> None:
        self._metric_rows.append(
            {
                "metric_name": metric_name,
                "metric_type": metric_type,
                "value": float(value),
                "timestamp": int(timestamp),
                "labels": labels or {},
            },
        )

    def add_correlation(
        self,
        *,
        correlation_id: str,
        event_id: str,
        parent_event_id: str | None,
        instrument_id: str,
        domain: str,
        lineage_depth: int,
        ts_event: int,
        propagation_path: list[str] | str,
    ) -> None:
        self._corr_rows.append(
            {
                "correlation_id": correlation_id,
                "event_id": event_id,
                "parent_event_id": parent_event_id,
                "instrument_id": instrument_id,
                "domain": domain,
                "lineage_depth": int(lineage_depth),
                "ts_event": int(ts_event),
                "propagation_path": propagation_path,
            },
        )

    def add_health(
        self,
        *,
        component_id: str,
        health_score: float,
        subsystem_scores: dict[str, float] | str,
        timestamp: int,
        measurement_window_ms: int,
    ) -> None:
        self._health_rows.append(
            {
                "component_id": component_id,
                "health_score": float(health_score),
                "subsystem_scores": subsystem_scores,
                "timestamp": int(timestamp),
                "measurement_window_ms": int(measurement_window_ms),
            },
        )

    # --------------------- DataFrame materialization ----------------------

    def latency_watermarks_df(self) -> pd.DataFrame:
        """
        Return latency watermark DataFrame (may be empty).
        """
        return build_latency_watermarks(self._latency_rows)

    def metrics_collection_df(self) -> pd.DataFrame:
        """
        Return metrics collection DataFrame (may be empty).
        """
        return build_metrics_collection(self._metric_rows)

    def event_correlation_df(self) -> pd.DataFrame:
        """
        Return event correlation/lineage DataFrame (may be empty).
        """
        return build_event_correlation(self._corr_rows)

    def health_scores_df(self) -> pd.DataFrame:
        """
        Return health scores DataFrame (may be empty).
        """
        return build_health_scores(self._health_rows)


__all__ = ["ObservabilityService"]
