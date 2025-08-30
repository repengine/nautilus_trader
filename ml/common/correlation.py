"""
Correlation utilities for event tracing.

Provides helpers to generate deterministic correlation IDs that can be used to
trace a single flow across Data → Features → Predictions → Signals.
"""

from __future__ import annotations

import hashlib


def make_correlation_id(
    *,
    run_id: str,
    dataset_id: str,
    instrument_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
) -> str:
    """
    Generate a deterministic correlation ID from event attributes.

    Parameters
    ----------
    run_id : str
        Unique identifier for the pipeline run.
    dataset_id : str
        Dataset identifier (e.g., "features", "predictions", "signals").
    instrument_id : str
        Instrument identifier.
    ts_min : int
        Minimum timestamp (ns) covered by the event.
    ts_max : int
        Maximum timestamp (ns) covered by the event.
    count : int
        Record count in the event.

    Returns
    -------
    str
        Hex-encoded SHA256 digest as correlation ID.
    """
    h = hashlib.sha256()
    h.update(run_id.encode("utf-8"))
    h.update(b"|")
    h.update(dataset_id.encode("utf-8"))
    h.update(b"|")
    h.update(instrument_id.encode("utf-8"))
    h.update(b"|")
    h.update(str(ts_min).encode("utf-8"))
    h.update(b"|")
    h.update(str(ts_max).encode("utf-8"))
    h.update(b"|")
    h.update(str(count).encode("utf-8"))
    return h.hexdigest()


__all__ = ["make_correlation_id"]

