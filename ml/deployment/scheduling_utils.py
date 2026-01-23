"""
Lightweight scheduling helpers for deployment entrypoints (cold path).

Provides deterministic utilities to compute the next UTC run from a daily
time specification and to parse environment schedule values. This module is
kept import‑light for use in unit tests and entrypoints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Final

from ml.registry.dataclasses import DatasetType


_TRUTHY: Final = {"1", "true", "yes", "on"}
_LOGGER = logging.getLogger(__name__)


def parse_bool_env(value: str | None) -> bool:
    """
    Parse a boolean environment value.

    Accepts common truthy variants (case-insensitive): "1", "true", "yes", "on".
    Any other non-empty value returns False.
    """
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def parse_template_map_env(raw: str | None) -> dict[str, str]:
    """
    Parse a template map from an environment value.

    Supports JSON maps (``{"key": "value"}``) and comma-separated ``key=value`` pairs.
    Keys are normalized to lowercase for case-insensitive lookups.
    """
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            _LOGGER.warning("Failed to parse template map (JSON decode)")
            return {}
        if not isinstance(parsed, Mapping):
            return {}
        return {
            str(key).strip().lower(): str(value)
            for key, value in parsed.items()
            if str(key).strip() and str(value)
        }
    templates: dict[str, str] = {}
    for token in text.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if "=" in candidate:
            key, value = candidate.split("=", 1)
        elif ":" in candidate:
            key, value = candidate.split(":", 1)
        else:
            continue
        key_normalized = key.strip().lower()
        value_normalized = value.strip()
        if not key_normalized or not value_normalized:
            continue
        templates[key_normalized] = value_normalized
    return templates


def parse_dataset_template_map_env(raw: str | None) -> dict[DatasetType, str]:
    """
    Parse a dataset-type→template mapping from environment payloads.
    """
    parsed = parse_template_map_env(raw)
    resolved: dict[DatasetType, str] = {}
    for key, value in parsed.items():
        try:
            dataset_type = DatasetType(key)
        except ValueError:
            _LOGGER.warning(
                "Ignoring unknown dataset type in identifier template map",
                extra={"dataset_type": key},
            )
            continue
        resolved[dataset_type] = value
    return resolved


@dataclass(slots=True, frozen=True)
class DailyTime:
    """A daily time specification (UTC) with hour and minute components."""

    hour: int
    minute: int


def parse_daily_spec(spec: str) -> DailyTime:
    """
    Parse a daily time specification.

    Supported formats (UTC):
    - "HH:MM" (e.g., "17:00")
    - Crontab-like: "M H * * *" (e.g., "0 17 * * *"). Only first two fields used.

    Raises ValueError if the spec is invalid.
    """
    s = spec.strip()

    # Try HH:MM
    if ":" in s and " " not in s:
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid daily time spec: {spec!r}")
        hour_str, minute_str = parts[0].strip(), parts[1].strip()
        if not (hour_str.isdigit() and minute_str.isdigit()):
            raise ValueError(f"Invalid daily time spec: {spec!r}")
        hour, minute = int(hour_str), int(minute_str)
        _validate_hm(hour, minute, spec)
        return DailyTime(hour=hour, minute=minute)

    # Try crontab-like: M H * * *
    fields = s.split()
    if len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit():
        minute, hour = int(fields[0]), int(fields[1])
        _validate_hm(hour, minute, spec)
        return DailyTime(hour=hour, minute=minute)

    raise ValueError(f"Invalid daily schedule format: {spec!r}")


def _validate_hm(hour: int, minute: int, raw: str) -> None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Out of range hour/minute in spec {raw!r}")


def compute_next_utc_run(now: datetime, daily: DailyTime) -> datetime:
    """
    Compute the next run datetime in UTC from a daily time.

    Assumes ``now`` is timezone-aware in UTC or naive treated as UTC.
    Returns a timezone-aware UTC ``datetime``.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        # Normalize to UTC for comparison
        now = now.astimezone(UTC)

    run_today = now.replace(hour=daily.hour, minute=daily.minute, second=0, microsecond=0)
    if now < run_today:
        return run_today
    return (run_today + timedelta(days=1)).replace(tzinfo=UTC)


__all__ = [
    "DailyTime",
    "compute_next_utc_run",
    "parse_bool_env",
    "parse_daily_spec",
    "parse_dataset_template_map_env",
    "parse_template_map_env",
]
