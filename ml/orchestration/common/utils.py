"""
Shared utility functions for orchestration components.

This module provides common utility functions used across orchestration
components to avoid duplication and ensure consistent behavior.

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from ml.registry.dataclasses import DatasetType
from ml.schema import map_schema_to_dataset_type as _map_schema_to_dataset_type_central


def ns_to_datetime(value: int) -> datetime:
    """
    Convert nanoseconds since epoch to datetime.

    Parameters
    ----------
    value : int
        Nanoseconds since Unix epoch

    Returns
    -------
    datetime
        UTC datetime object

    Examples
    --------
    >>> ts = 1701432000000000000  # 2023-12-01 12:00:00 UTC in nanoseconds
    >>> dt = ns_to_datetime(ts)
    >>> dt.year
    2023

    """
    return datetime.fromtimestamp(value / 1e9, tz=UTC)


def map_schema_to_dataset_type(schema: str) -> DatasetType:
    """Alias to centralized mapping to avoid drift."""
    return _map_schema_to_dataset_type_central(schema)


def parse_symbols(symbols_str: str) -> list[str]:
    """
    Parse comma-separated symbols into a list.

    Parameters
    ----------
    symbols_str : str
        Comma-separated symbol string (e.g., "AAPL,GOOGL,MSFT")

    Returns
    -------
    list[str]
        List of trimmed symbol strings

    Examples
    --------
    >>> parse_symbols("AAPL")
    ['AAPL']
    >>> parse_symbols("AAPL,GOOGL,MSFT")
    ['AAPL', 'GOOGL', 'MSFT']
    >>> parse_symbols(" AAPL , GOOGL ")
    ['AAPL', 'GOOGL']

    """
    return [s.strip() for s in symbols_str.split(",") if s.strip()]
