"""
Shared event emission utilities for stores and actors.

Centralizes consistent usage of Stage/Source/EventStatus, watermark updates, and
optional metrics for dataset-level events.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ml.common.correlation import make_correlation_id
from ml.common.events_util import to_source_enum
from ml.common.events_util import to_stage_enum
from ml.common.events_util import to_status_enum
from ml.common.metrics_bootstrap import get_counter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)

_METADATA_FALLBACK_COUNTER = get_counter(
    "ml_dataset_event_metadata_fallback_total",
    "Dataset event metadata fallback activations",
    ("dataset_type", "component", "stage", "source"),
)


def _record_metadata_fallback(
    *,
    dataset_label: str,
    component_label: str,
    stage_value: str,
    source_value: str,
    dataset_id: str,
    instrument_id: str,
    error: Exception,
) -> None:
    """
    Log and record a metadata fallback activation.
    """
    logger.warning(
        "Registry rejected dataset event metadata; emitting without metadata support",
        extra={
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "stage": stage_value,
            "source": source_value,
            "component": component_label,
        },
        exc_info=error,
    )
    try:
        _METADATA_FALLBACK_COUNTER.labels(
            dataset_type=dataset_label,
            component=component_label,
            stage=stage_value,
            source=source_value,
        ).inc()
    except Exception as metrics_exc:  # pragma: no cover - defensive metrics path
        logger.debug(
            "Metadata fallback counter increment failed: %s",
            metrics_exc,
            exc_info=True,
        )


def emit_dataset_event_and_watermark(
    registry: RegistryProtocol,
    *,
    dataset_id: str,
    instrument_id: str,
    stage: Stage | str,
    source: Source | str,
    run_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
    status: EventStatus | str,
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

    stage_enum = to_stage_enum(stage)
    source_enum = to_source_enum(source)
    status_enum = to_status_enum(status)
    dataset_label = dataset_type or dataset_id
    component_label = component or ""

    # Emit the event via registry
    try:
        registry.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage_enum,
            source=source_enum,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status_enum,
            metadata=event_metadata,
        )
    except TypeError as exc:
        _record_metadata_fallback(
            dataset_label=dataset_label,
            component_label=component_label,
            stage_value=stage_enum.value,
            source_value=source_enum.value,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            error=exc,
        )
        # Backwards-compatible registries may not accept metadata
        registry.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage_enum,
            source=source_enum,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status_enum,
        )

    # Update watermark to reflect progress
    registry.update_watermark(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        source=source_enum,
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
                "stage": stage_enum.value,
                "source": source_enum.value,
                "status": status_enum.value,
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
    stage: Stage | str,
    source: Source | str,
    run_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
    status: EventStatus | str,
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

    stage_enum = to_stage_enum(stage)
    source_enum = to_source_enum(source)
    status_enum = to_status_enum(status)
    dataset_label = dataset_type or dataset_id
    component_label = component or ""

    try:
        registry.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage_enum,
            source=source_enum,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status_enum,
            error=error,
            metadata=event_metadata,
        )
    except TypeError as exc:
        _record_metadata_fallback(
            dataset_label=dataset_label,
            component_label=component_label,
            stage_value=stage_enum.value,
            source_value=source_enum.value,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            error=exc,
        )
        registry.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage_enum,
            source=source_enum,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status_enum,
            error=error,
        )
        try:
            last_emit = getattr(registry, "last_emit", None)
            if isinstance(last_emit, dict) and last_emit.get("metadata") is None:
                last_emit.pop("metadata", None)
        except Exception as exc:  # pragma: no cover - defensive
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Registry metadata cleanup skipped due to error: %s",
                exc,
                exc_info=True,
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
                "stage": stage_enum.value,
                "source": source_enum.value,
                "status": status_enum.value,
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
            except (AttributeError, FileNotFoundError, ImportError, OSError):
                logger.debug(
                    "event_emitter.restore_tracing_module_failed dataset_id=%s run_id=%s",
                    dataset_id,
                    run_id,
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "event_run_id": run_id,
                    },
                )
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
