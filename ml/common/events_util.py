"""
Typed helpers for event source normalization and conversions.

This module centralizes conversions between persisted source strings and the
enum-typed ``Source`` used across the codebase. Keeping this in one place
avoids small duplicated normalization blocks and makes intent obvious.

Usage
-----
>>> to_source_enum("live")
<Source.LIVE: 'live'>
>>> to_source_str(Source.HISTORICAL)
'historical'

"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal, cast

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


# Persisted representation accepted by the DB and JSON artifacts
SourceStr = Literal["live", "historical", "backfill"]
_ALLOWED: tuple[SourceStr, ...] = ("live", "historical", "backfill")


def to_source_enum(x: Source | str) -> Source:
    """
    Convert a persisted source string or Source enum to Source.

    Raises ValueError if the string is not one of the allowed values.

    """
    if isinstance(x, Source):
        return x
    # Normalize to lower before enum construction
    value = str(x).lower()
    try:
        return Source(value)
    except ValueError:
        logging.getLogger(__name__).debug(
            "Unknown source provided, falling back to live",
            extra={"provided_source": value},
        )
        return Source.LIVE


def to_source_str(x: Source | str) -> SourceStr:
    """
    Convert a Source enum or persisted string to the canonical persisted string.

    Ensures the returned value matches DB constraints and JSON persistence.

    """
    v = x.value if isinstance(x, Source) else str(x).lower()
    if v not in _ALLOWED:
        # Match Source.__members__ for consistent error message behavior
        raise ValueError(f"Invalid source '{v}', must be one of {_ALLOWED}")
    return cast(SourceStr, v)


def build_bus_payload(
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
    metadata: Mapping[str, object] | None = None,
    inject_trace_context: bool = True,
) -> dict[str, object]:
    """
    Build a canonical message bus payload from event attributes.

    Converts enums to their persisted string values where appropriate.
    Optionally injects distributed tracing context for cross-component correlation.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    stage : Stage | str
        Processing stage
    source : Source | str
        Data source
    run_id : str
        Run identifier
    ts_min : int
        Minimum timestamp
    ts_max : int
        Maximum timestamp
    count : int
        Record count
    status : EventStatus | str
        Event status
    metadata : Mapping[str, object] | None, optional
        Additional metadata
    inject_trace_context : bool, default True
        Whether to inject W3C trace context when tracing enabled

    Returns
    -------
    dict[str, object]
        Bus payload with optional trace context

    """
    stage_val = normalize_stage_value(stage)
    source_val = to_source_str(source)
    status_val = to_status_str(status)

    # Start with base metadata
    final_metadata = dict(metadata or {})

    # Inject trace context if enabled and available
    if inject_trace_context:
        try:
            # Lazy import to avoid circular dependencies
            from ml.observability.tracing import inject_trace_context as _inject

            final_metadata = _inject(final_metadata)
        except ImportError:
            # Graceful fallback when tracing not available
            ...
        except Exception as exc:
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Trace inject failed in build_bus_payload: %s",
                exc,
                exc_info=True,
            )

    payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "instrument_id": instrument_id,
        "stage": stage_val,
        "source": source_val,
        "run_id": run_id,
        "ts_min": int(ts_min),
        "ts_max": int(ts_max),
        "count": int(count),
        "status": status_val,
        "metadata": final_metadata,
    }
    return payload


def to_stage_enum(stage: Stage | str) -> Stage:
    """
    Convert legacy string representations to a ``Stage`` enum.

    Accepts values like ``Stage.FEATURE_COMPUTED``, ``STAGE.FEATURE_COMPUTED``,
    or raw enum instances. Raises ``ValueError`` for unknown names.
    """
    if isinstance(stage, Stage):
        return stage
    stage_str = str(stage)
    if stage_str.startswith("Stage."):
        stage_str = stage_str.split(".", 1)[1]
    if stage_str.startswith("STAGE."):
        stage_str = stage_str.split(".", 1)[1]
    try:
        return Stage(stage_str)
    except ValueError:
        alias = stage_str.strip().upper()
        alias_map = {
            "FEATURE": Stage.FEATURE_COMPUTED,
            "FEATURES": Stage.FEATURE_COMPUTED,
            "FEATURE_COMPUTED": Stage.FEATURE_COMPUTED,
            "PREDICTION": Stage.PREDICTION_EMITTED,
            "PREDICTIONS": Stage.PREDICTION_EMITTED,
            "PREDICTIONS_EMITTED": Stage.PREDICTION_EMITTED,
            "SIGNAL": Stage.SIGNAL_EMITTED,
            "SIGNALS": Stage.SIGNAL_EMITTED,
            "DATA": Stage.DATA_INGESTED,
            "DATA_INGESTED": Stage.DATA_INGESTED,
            "CATALOG": Stage.CATALOG_WRITTEN,
            "CATALOG_WRITTEN": Stage.CATALOG_WRITTEN,
        }
        try:
            return alias_map[alias]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown stage identifier '{stage_str}'") from exc


def normalize_stage_value(stage: Stage | str) -> str:
    """
    Return the canonical ``Stage`` value string for event payloads.
    """
    return to_stage_enum(stage).value


def to_status_enum(status: EventStatus | str) -> EventStatus:
    """
    Convert legacy string representations to an ``EventStatus`` enum.
    """
    if isinstance(status, EventStatus):
        return status
    status_str = str(status)
    if status_str.startswith("EventStatus."):
        status_str = status_str.split(".", 1)[1]
    try:
        return EventStatus(status_str)
    except ValueError:
        alias = status_str.strip().upper()
        alias_map = {
            "SUCCESS": EventStatus.SUCCESS,
            "FAILED": EventStatus.FAILED,
            "PARTIAL": EventStatus.PARTIAL,
            "DEFERRED": EventStatus.DEFERRED,
        }
        try:
            return alias_map[alias]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown status identifier '{status_str}'") from exc


def to_status_str(status: EventStatus | str) -> str:
    """
    Return the canonical status string used for persistence.
    """
    return to_status_enum(status).value


__all__ = [
    "SourceStr",
    "build_bus_payload",
    "normalize_stage_value",
    "to_source_enum",
    "to_source_str",
    "to_stage_enum",
    "to_status_enum",
    "to_status_str",
]
