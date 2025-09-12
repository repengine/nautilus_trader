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
    return Source(str(x).lower())


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
) -> dict[str, object]:
    """
    Build a canonical message bus payload from event attributes.

    Converts enums to their persisted string values where appropriate.
    """
    stage_val = stage.value if isinstance(stage, Stage) else str(stage)
    source_val = to_source_str(source)
    status_val = status.value if isinstance(status, EventStatus) else str(status)
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
        "metadata": dict(metadata or {}),
    }
    return payload


__all__ = ["SourceStr", "build_bus_payload", "to_source_enum", "to_source_str"]
