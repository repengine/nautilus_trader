"""
Shared event emission utilities for stores and actors.

Centralizes consistent usage of Stage/Source/EventStatus, watermark updates, and
optional metrics for dataset-level events.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.registry.protocols import RegistryProtocol


def emit_dataset_event_and_watermark(
    registry: RegistryProtocol,
    *,
    dataset_id: str,
    instrument_id: str,
    stage: Stage,
    source: Source,
    run_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
    status: EventStatus,
    dataset_type: str | None = None,
    component: str | None = None,
) -> None:
    """
    Emit a dataset event and update its watermark atomically, with optional metrics.

    This helper enforces consistent enum usage and label application across stores.

    """
    # Emit the event via registry
    registry.emit_event(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        stage=stage,
        source=source,
        run_id=run_id,
        ts_min=ts_min,
        ts_max=ts_max,
        count=count,
        status=status,
    )

    # Update watermark to reflect progress
    registry.update_watermark(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        source=source,
        last_success_ns=ts_max,
        count=count,
        completeness_pct=100.0,
    )

    # Optional metrics: best-effort, no hard dependency
    try:  # pragma: no cover - metrics optional
        from ml.common.metrics_manager import MetricsManager

        mm = MetricsManager.default()
        mm.inc(
            "nautilus_ml_data_events_total",
            "Total data events processed by stage",
            labels={
                "dataset_type": (dataset_type or dataset_id),
                "component": (component or ""),
                "stage": stage.value,
                "source": source.value,
                "status": status.value,
            },
            labelnames=("dataset_type", "component", "stage", "source", "status"),
        )
    except Exception:
        # Metrics are best-effort; ignore in hot paths
        pass


def emit_dataset_event(
    registry: RegistryProtocol,
    *,
    dataset_id: str,
    instrument_id: str,
    stage: Stage,
    source: Source,
    run_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
    status: EventStatus,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
    dataset_type: str | None = None,
    component: str | None = None,
) -> None:
    """
    Emit a dataset event only (no watermark), with optional metrics.

    Centralizes enum-safe emission and consistent metric labeling.

    """
    registry.emit_event(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        stage=stage,
        source=source,
        run_id=run_id,
        ts_min=ts_min,
        ts_max=ts_max,
        count=count,
        status=status,
        error=error,
        metadata=metadata,
    )

    # Optional metrics: best-effort, no hard dependency
    try:  # pragma: no cover - metrics optional
        from ml.common.metrics_manager import MetricsManager

        mm = MetricsManager.default()
        mm.inc(
            "nautilus_ml_data_events_total",
            "Total data events processed by stage",
            labels={
                "dataset_type": (dataset_type or dataset_id),
                "component": (component or ""),
                "stage": stage.value,
                "source": source.value,
                "status": status.value,
            },
            labelnames=("dataset_type", "component", "stage", "source", "status"),
        )
    except Exception:
        # Metrics are best-effort; ignore in hot paths
        pass


__all__ = ["emit_dataset_event", "emit_dataset_event_and_watermark"]
