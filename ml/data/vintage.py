"""Vintage policy definitions for macro feature ingestion."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import Enum


class VintagePolicy(str, Enum):
    """How macro inputs should treat data revisions."""

    REAL_TIME = "real_time"
    FINAL = "final"

def format_dt(dt: datetime | None) -> str | None:
    """Format datetimes for metadata output."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        aware = dt.replace(tzinfo=UTC)
    else:
        aware = dt.astimezone(UTC)
    return aware.replace(microsecond=0).isoformat()


def parse_dt(value: str | None) -> datetime | None:
    """Parse ISO8601 strings back to timezone-aware datetimes."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
