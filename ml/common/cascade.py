"""
Helpers for cross-domain event cascades with correlation preservation.

These utilities are designed for testability and light integration in
`MLIntegrationManager.emit_cascade` without affecting hot paths.

"""

from __future__ import annotations

from typing import Any, TypedDict


class EventDict(TypedDict, total=False):
    domain: str
    event_type: str
    correlation_id: str
    instrument_id: str
    ts_event: int
    source_event_id: str
    parent_event_id: str
    payload: dict[str, Any]


def emit_cascade(
    source_event: EventDict,
    target_domain: str,
    delay_ns: int | None = None,
) -> EventDict:
    """
    Create a cascaded event for a target domain preserving correlation.

    Parameters
    ----------
    source_event : EventDict
        The originating event with at least `correlation_id` and `ts_event`.
    target_domain : str
        The target domain for the cascaded event (e.g., 'features').
    delay_ns : int | None
        Optional nanosecond delay to add to the timestamp.

    Returns
    -------
    EventDict
        A new event dictionary with updated `domain`, `ts_event`, and
        `source_event_id`.

    """
    new_ts = int(source_event.get("ts_event", 0)) + int(delay_ns or 0)
    cascaded: EventDict = EventDict(
        domain=target_domain,
        event_type=source_event.get("event_type", "cascade"),
        correlation_id=source_event.get("correlation_id", ""),
        instrument_id=source_event.get("instrument_id", ""),
        ts_event=new_ts,
        source_event_id=str(
            source_event.get("event_id", source_event.get("source_event_id", "unknown")),
        ),
        payload=dict(source_event.get("payload", {}) or {}),
    )
    return cascaded


__all__ = ["EventDict", "emit_cascade"]
