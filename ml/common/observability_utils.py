"""
Observability helpers (cold-path only).

Provide small, typed utilities to record stage boundaries via an injected
observability-like service. Designed to avoid import-time heavy deps and keep
stores thin by delegating common logic here.
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class ObservabilityLike(Protocol):
    def add_latency_stage(
        self,
        *,
        correlation_id: str,
        instrument_id: str,
        pipeline_stage: str,
        ts_stage_start: int,
        ts_stage_end: int,
    ) -> None: ...

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, Any] | str | None = None,
    ) -> None: ...


def is_observability_enabled() -> bool:
    """
    Return True if observability helpers should record data.

    Controlled by the environment variable `ML_OBSERVABILITY_ENABLED` in
    {"1","true","yes"} (case-insensitive).
    """
    return os.getenv("ML_OBSERVABILITY_ENABLED", "").lower() in {"1", "true", "yes"}


def record_stage_boundary(
    obs_service: ObservabilityLike | None,
    *,
    component: str,
    instrument_id: str,
    stage: str,
    ts_stage_start: int,
    ts_stage_end: int,
    row_count: int = 1,
) -> None:
    """
    Record a latency stage and a histogram metric for a component.

    Parameters
    ----------
    obs_service : ObservabilityLike | None
        Observability service instance; if None, no-op.
    component : str
        Component name used for correlation prefix and metric naming.
    instrument_id : str
        Instrument identifier for labels and correlation.
    stage : str
        Pipeline stage name.
    ts_stage_start : int
        Stage start timestamp (ns).
    ts_stage_end : int
        Stage end timestamp (ns).
    row_count : int
        Optional row count for labels.
    """
    try:
        if obs_service is None or not is_observability_enabled():
            return

        correlation_id = f"{component}_{hash((instrument_id, ts_stage_start)) % 1_000_000}"
        obs_service.add_latency_stage(
            correlation_id=correlation_id,
            instrument_id=instrument_id,
            pipeline_stage=stage,
            ts_stage_start=int(ts_stage_start),
            ts_stage_end=int(ts_stage_end),
        )

        latency_ms = (int(ts_stage_end) - int(ts_stage_start)) / 1_000_000
        obs_service.add_metric(
            metric_name=f"{component}_latency_ms",
            metric_type="histogram",
            value=float(latency_ms),
            timestamp=int(ts_stage_end),
            labels={
                "stage": stage,
                "instrument_id": instrument_id,
                "row_count": str(int(row_count)),
            },
        )
    except Exception as exc:
        # Non-blocking by design
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "record_stage_boundary failed (ignored): %s", exc, exc_info=True
        )


__all__ = [
    "ObservabilityLike",
    "is_observability_enabled",
    "record_stage_boundary",
]
