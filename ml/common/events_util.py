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

from ml.config.events import CANONICAL_STAGE_EQUIVALENTS
from ml.config.events import LEGACY_STAGE_ALIAS_MAP
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.dataclasses import DatasetType


logger = logging.getLogger(__name__)


# Persisted representation accepted by the DB and JSON artifacts
SourceStr = Literal["live", "historical", "backfill", "batch"]
_ALLOWED: tuple[SourceStr, ...] = ("live", "historical", "backfill", "batch")


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


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
        logger.debug(
            "Unknown source provided, falling back to live",
            extra={"provided_source": value},
            exc_info=True,
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
        return CANONICAL_STAGE_EQUIVALENTS.get(stage, stage)

    stage_str = str(stage).strip()
    if stage_str.startswith("Stage."):
        stage_str = stage_str.split(".", 1)[1]
    if stage_str.startswith("STAGE."):
        stage_str = stage_str.split(".", 1)[1]

    def _canonical(resolved: Stage) -> Stage:
        return CANONICAL_STAGE_EQUIVALENTS.get(resolved, resolved)

    # Direct value lookups
    try:
        resolved = Stage(stage_str)
        return _canonical(resolved)
    except ValueError:
        logger.debug(
            "Stage enum resolution failed, attempting fallback normalization",
            extra={"provided_stage": stage_str},
            exc_info=True,
        )

    upper_value = stage_str.upper()
    if upper_value != stage_str:
        try:
            resolved_upper = Stage(upper_value)
            return _canonical(resolved_upper)
        except ValueError:
            ...

    normalized = stage_str.replace(".", "_").replace("-", "_").replace(" ", "_").upper()

    if normalized in Stage.__members__:
        member = Stage[normalized]
        return _canonical(member)

    alias_map: dict[str, Stage] = {
        **LEGACY_STAGE_ALIAS_MAP,
        "FEATURE": Stage.FEATURE_COMPUTED,
        "FEATURES": Stage.FEATURE_COMPUTED,
        "FEATURE_ENGINEERING": Stage.FEATURE_COMPUTED,
        "FEATURES_ENGINEERED": Stage.FEATURE_COMPUTED,
        "FEATURES_COMPUTED": Stage.FEATURE_COMPUTED,
        "FEATURE_COMPUTED": Stage.FEATURE_COMPUTED,
        "PREDICTION": Stage.PREDICTION_EMITTED,
        "PREDICTIONS": Stage.PREDICTION_EMITTED,
        "PREDICTIONS_EMITTED": Stage.PREDICTION_EMITTED,
        "MODEL_INFERENCE": Stage.PREDICTION_EMITTED,
        "SIGNAL": Stage.SIGNAL_EMITTED,
        "SIGNALS": Stage.SIGNAL_EMITTED,
        "SIGNAL_GENERATION": Stage.SIGNAL_EMITTED,
        "SIGNAL_GENERATED": Stage.SIGNAL_EMITTED,
        "EMIT_SIGNAL": Stage.SIGNAL_EMITTED,
        "DATA": Stage.DATA_INGESTED,
        "INGEST": Stage.DATA_INGESTED,
        "INGESTED": Stage.DATA_INGESTED,
        "DATA_INGEST": Stage.DATA_INGESTED,
        "CATALOG": Stage.CATALOG_WRITTEN,
        "CATALOG_WRITE": Stage.CATALOG_WRITTEN,
        "CATALOG_WRITTEN": Stage.CATALOG_WRITTEN,
    }

    try:
        mapped = alias_map[normalized]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown stage identifier '{stage_str}'") from exc
    return _canonical(mapped)


def normalize_stage_value(stage: Stage | str) -> str:
    """
    Return the canonical ``Stage`` value string for event payloads.
    """
    return to_stage_enum(stage).value


def stage_for_dataset_type(dataset_type: DatasetType) -> Stage:
    """
    Map a dataset type to the canonical processing stage.

    Args:
        dataset_type: Dataset type from the registry manifest.

    Returns:
        Stage enum representing the expected pipeline stage.

    """
    stage_map: dict[DatasetType, Stage] = {
        DatasetType.FEATURES: Stage.FEATURE_COMPUTED,
        DatasetType.PREDICTIONS: Stage.PREDICTION_EMITTED,
        DatasetType.SIGNALS: Stage.SIGNAL_EMITTED,
        DatasetType.ORDER_EVENTS: Stage.ORDER_EVENT_EMITTED,
        DatasetType.EARNINGS_ACTUALS: Stage.DATA_INGESTED,
        DatasetType.EARNINGS_ESTIMATES: Stage.DATA_INGESTED,
        DatasetType.MACRO_RELEASES: Stage.DATA_INGESTED,
        DatasetType.MACRO_OBSERVATIONS: Stage.DATA_INGESTED,
        DatasetType.EVENTS_CALENDAR: Stage.DATA_INGESTED,
        DatasetType.MICRO_MINUTE_FEATURES: Stage.FEATURE_COMPUTED,
        DatasetType.L2_MINUTE_FEATURES: Stage.FEATURE_COMPUTED,
    }
    return stage_map.get(dataset_type, Stage.DATA_INGESTED)


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


def validate_bus_payload(payload: Mapping[str, object]) -> tuple[bool, list[str]]:
    """
    Validate a canonical message bus payload.

    Returns a tuple of (is_valid, errors). This is a best-effort validator for
    the canonical bus payload shape used by build_bus_payload.
    """
    errors: list[str] = []
    required = (
        "dataset_id",
        "instrument_id",
        "stage",
        "source",
        "run_id",
        "ts_min",
        "ts_max",
        "count",
        "status",
        "metadata",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing {key}")

    if errors:
        return False, errors

    dataset_id = payload.get("dataset_id")
    instrument_id = payload.get("instrument_id")
    run_id = payload.get("run_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        errors.append("dataset_id must be a non-empty string")
    if not isinstance(instrument_id, str) or not instrument_id.strip():
        errors.append("instrument_id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("run_id must be a non-empty string")

    try:
        _ = to_stage_enum(cast(Stage | str, payload.get("stage")))
    except Exception as exc:
        errors.append(f"invalid stage: {exc}")
    try:
        _ = to_source_str(cast(Source | str, payload.get("source")))
    except Exception as exc:
        errors.append(f"invalid source: {exc}")
    try:
        _ = to_status_enum(cast(EventStatus | str, payload.get("status")))
    except Exception as exc:
        errors.append(f"invalid status: {exc}")

    ts_min = payload.get("ts_min")
    ts_max = payload.get("ts_max")
    ts_min_val = _coerce_int(ts_min)
    ts_max_val = _coerce_int(ts_max)
    if ts_min_val is None or ts_max_val is None:
        errors.append("ts_min and ts_max must be integers")
    else:
        if ts_min_val < 0 or ts_max_val < 0:
            errors.append("timestamps must be non-negative")
        if ts_min_val > ts_max_val:
            errors.append("ts_min must be <= ts_max")

    count = payload.get("count")
    count_val = _coerce_int(count)
    if count_val is None:
        errors.append("count must be an integer")
    elif count_val < 0:
        errors.append("count must be >= 0")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping")

    return len(errors) == 0, errors


__all__ = [
    "SourceStr",
    "build_bus_payload",
    "normalize_stage_value",
    "stage_for_dataset_type",
    "to_source_enum",
    "to_source_str",
    "to_stage_enum",
    "to_status_enum",
    "to_status_str",
    "validate_bus_payload",
]
