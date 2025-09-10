"""
Timestamp normalization and sanitization utilities.

Centralizes policy for UNIX timestamp units and provides a single helper for normalizing
to nanoseconds with optional logging and modes.

"""

from __future__ import annotations

import logging
import os


def normalize_timestamp_ns(ts_value: int) -> tuple[int, bool]:
    """
    Normalize a UNIX timestamp to nanoseconds.

    Returns (normalized_value, was_normalized).

    Heuristic based on magnitude:
    - < 1e11   -> seconds
    - < 1e14   -> milliseconds
    - < 1e17   -> microseconds
    - otherwise -> nanoseconds

    """
    try:
        ts = int(ts_value)
    except Exception:
        return int(ts_value), False

    if ts < 100_000_000_000:  # seconds
        return ts * 1_000_000_000, True
    if ts < 100_000_000_000_000:  # milliseconds
        return ts * 1_000_000, True
    if ts < 100_000_000_000_000_000:  # microseconds
        return ts * 1_000, True
    return ts, False


def _get_mode(env_default: str | None = None) -> str:
    mode = (env_default or os.getenv("ML_TS_NORMALIZATION_MODE") or "warn").lower()
    if mode not in {"warn", "normalize", "reject"}:
        mode = "warn"
    return mode


def sanitize_timestamp_ns(
    ts_value: int,
    *,
    mode: str | None = None,
    logger: logging.Logger | None = None,
    context: str = "",
) -> int:
    """
    Sanitize a timestamp to nanoseconds per policy.

    - warn (default): normalize and log a warning if unit was < ns
    - normalize: normalize silently
    - reject: raise ValueError if normalization would be required

    """
    norm, changed = normalize_timestamp_ns(ts_value)
    effective_mode = _get_mode(mode)
    if changed:
        if effective_mode == "reject":
            raise ValueError(f"Non-ns ts encountered ({ts_value}) in {context}")
        if effective_mode == "warn" and logger is not None:
            logger.warning(
                "Normalized timestamp to ns in %s: %s -> %s",
                context or "write",
                ts_value,
                norm,
            )
    return norm
