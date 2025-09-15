"""
Trace context utilities for event consumers.

Provides helper functions for event consumers to extract and link trace context
from event metadata, enabling parent-child span relationships across components.
"""

from __future__ import annotations

from typing import Any


def extract_and_link_from_event(event_metadata: dict[str, Any]) -> None:
    """
    Extract trace context from event metadata and link to current span.

    This is a convenience wrapper around the observability tracing module
    for event consumers. It safely extracts W3C trace context and establishes
    parent-child relationships with the current tracing context.

    Parameters
    ----------
    event_metadata : dict[str, Any]
        Event metadata potentially containing trace_context

    Examples
    --------
    >>> # In an event consumer
    >>> extract_and_link_from_event(event.metadata)
    >>> with trace_cold_path("process_features") as span:
    ...     # This span will be linked to the original feature computation trace
    ...     features = process_features(event.data)
    """
    try:
        # Lazy import to avoid circular dependencies
        from ml.observability.tracing import extract_and_link_trace_context
        extract_and_link_trace_context(event_metadata)
    except ImportError:
        # Graceful fallback when tracing not available
        pass
    except Exception:
        # Graceful fallback on any tracing error
        pass


def get_correlation_and_trace_context(
    *,
    run_id: str,
    dataset_id: str,
    instrument_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
) -> dict[str, Any]:
    """
    Generate correlation_id and extract trace context for event metadata.

    Creates a complete metadata dict containing both correlation_id and
    trace context, suitable for event emission or message publishing.

    Parameters
    ----------
    run_id : str
        Run identifier
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    ts_min : int
        Minimum timestamp
    ts_max : int
        Maximum timestamp
    count : int
        Record count

    Returns
    -------
    dict[str, Any]
        Metadata dict with correlation_id and optional trace_context

    Examples
    --------
    >>> metadata = get_correlation_and_trace_context(
    ...     run_id="train_123",
    ...     dataset_id="features",
    ...     instrument_id="EUR/USD",
    ...     ts_min=start_ns,
    ...     ts_max=end_ns,
    ...     count=1000
    ... )
    >>> emit_dataset_event(..., metadata=metadata)
    """
    # Generate correlation_id
    try:
        from ml.common.correlation import make_correlation_id
        correlation_id = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )
    except ImportError:
        # Fallback correlation_id if module not available
        correlation_id = f"{run_id}:{dataset_id}:{instrument_id}:{ts_min}:{ts_max}:{count}"

    metadata = {"correlation_id": correlation_id}

    # Inject trace context if available
    try:
        from ml.observability.tracing import inject_trace_context
        metadata = inject_trace_context(metadata)
    except ImportError:
        # Graceful fallback when tracing not available
        pass
    except Exception:
        # Graceful fallback on any tracing error
        pass

    return metadata


__all__ = [
    "extract_and_link_from_event",
    "get_correlation_and_trace_context",
]
