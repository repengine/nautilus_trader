"""
Shared event emission utilities for stores and actors.

Centralizes consistent usage of Stage/Source/EventStatus, watermark updates, and
optional metrics for dataset-level events.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml.common.correlation import make_correlation_id
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
    metadata: dict[str, object] | None = None,
) -> None:
    """
    Emit a dataset event and update its watermark atomically, with optional metrics.

    This helper enforces consistent enum usage, correlation_id attachment, and label
    application across stores.

    """
    # Ensure deterministic correlation_id is attached
    event_metadata = _ensure_correlation_id(
        metadata=metadata,
        run_id=run_id,
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        ts_min=ts_min,
        ts_max=ts_max,
        count=count,
    )

    # Emit the event via registry
    try:
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
            metadata=event_metadata,
        )
    except TypeError:
        # Backwards-compatible registries may not accept metadata
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
    except Exception as exc:
        # Metrics are best-effort; ignore in hot paths — debug for visibility
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Data event metric emit failed (event+watermark): %s",
            exc,
            exc_info=True,
        )


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

    Centralizes enum-safe emission, correlation_id attachment, and consistent metric
    labeling.

    """
    # Ensure deterministic correlation_id is attached
    event_metadata = _ensure_correlation_id(
        metadata=metadata,
        run_id=run_id,
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        ts_min=ts_min,
        ts_max=ts_max,
        count=count,
    )

    try:
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
            metadata=event_metadata,
        )
    except TypeError:
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
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Data event metric emit failed: %s",
            exc,
            exc_info=True,
        )


def _ensure_correlation_id(
    *,
    metadata: dict[str, object] | None,
    run_id: str,
    dataset_id: str,
    instrument_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
) -> dict[str, object]:
    """
    Ensure a deterministic correlation_id and optional trace context are attached.

    If metadata already contains correlation_id, it is preserved. Otherwise, a new one
    is generated using make_correlation_id.

    Automatically injects W3C trace context when distributed tracing is enabled.

    """
    event_metadata: dict[str, object] = {}
    if metadata:
        event_metadata.update(metadata)

    # Only generate correlation_id if not already provided
    if "correlation_id" not in event_metadata:
        correlation_id = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )
        event_metadata["correlation_id"] = correlation_id

    # Inject trace context if tracing enabled and not already present
    if "trace_context" not in event_metadata:
        try:
            # Lazy import to avoid circular dependencies
            from ml.observability.tracing import inject_trace_context

            event_metadata = inject_trace_context(event_metadata)

            # Best-effort healing: if a test temporarily replaced the tracing module
            # with a minimal stub, restore the real implementation to sys.modules
            # to avoid breaking later imports in the same worker process.
            try:  # pragma: no cover - test environment quirk
                import importlib.util as _ilu
                import os as _os
                import sys as _sys

                _mod = _sys.modules.get("ml.observability.tracing")
                if _mod is not None and not hasattr(_mod, "extract_and_link_trace_context"):
                    _base = _os.path.dirname(__file__)  # ml/common
                    _tracing_path = _os.path.abspath(
                        _os.path.join(_base, "..", "observability", "tracing.py"),
                    )
                    _spec = _ilu.spec_from_file_location(
                        "ml.observability._tracing_real", _tracing_path,
                    )
                    if _spec and _spec.loader:
                        _real = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_real)
                        _sys.modules["ml.observability.tracing"] = _real
            except Exception:
                # Never impact normal operation
                pass
        except ImportError:
            # Graceful fallback when tracing not available
            ...
        except Exception as exc:
            # Graceful fallback on any tracing error — debug
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Trace context injection failed: %s",
                exc,
                exc_info=True,
            )

    return event_metadata


__all__ = ["emit_dataset_event", "emit_dataset_event_and_watermark"]
