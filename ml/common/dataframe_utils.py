"""
Lightweight, typed DataFrame helpers used across ML components.

These functions provide small, central utilities that work with either pandas or
polars objects without introducing heavy dependencies or import-time costs.
"""

from __future__ import annotations

from typing import Any


def total_nulls(df: Any) -> int:
    """
    Return total null count across all columns for pandas or polars DataFrame-like objects.

    Falls back to 0 if the object does not provide null-counting methods.
    """
    try:
        if hasattr(df, "null_count"):
            # polars: DataFrame.null_count() -> DataFrame; sum across columns
            nc = df.null_count()
            # sum_horizontal() exists on polars DataFrame; guard generically
            if hasattr(nc, "sum_horizontal"):
                return int(nc.sum_horizontal().sum())
            # Fallback: to_dicts then sum
            dicts = nc.to_dicts()
            return int(sum(dicts[0].values()) if dicts else 0)
        if hasattr(df, "isnull"):
            # pandas: DataFrame.isnull().sum().sum()
            return int(df.isnull().sum().sum())
    except Exception:
        return 0
    return 0


def column_nulls(df: Any, column: str) -> int:
    """
    Return null count for a specific column for pandas or polars Series/Frame.

    Uses Series.isnull()/is_null() where available. Returns 0 if unsupported.
    """
    try:
        if hasattr(df, "__getitem__"):
            col = df[column]
            if hasattr(col, "is_null"):
                # polars Series
                return int(col.is_null().sum())
            if hasattr(col, "isnull"):
                # pandas Series
                return int(col.isnull().sum())
    except Exception:
        return 0
    return 0

