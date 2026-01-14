#!/usr/bin/env python3

"""
Shared Watermark dataclass for registry components.

This module centralizes the Watermark definition to avoid circular imports between
DataRegistry and the watermark management components.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Watermark:
    """
    Watermark tracking data processing progress for a dataset.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    source : str
        Data source ('live', 'historical', 'backfill')
    last_success_ns : int
        Last successful processing timestamp in nanoseconds
    last_attempt_ns : int
        Last attempted processing timestamp in nanoseconds
    last_count : int
        Count from last successful processing
    completeness_pct : float
        Percentage of expected data received (0-100)
    updated_at : float
        Unix timestamp of last update

    """

    dataset_id: str
    instrument_id: str
    source: str
    last_success_ns: int
    last_attempt_ns: int
    last_count: int
    completeness_pct: float
    updated_at: float


__all__ = ["Watermark"]
