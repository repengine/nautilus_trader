"""
Utility functions for data providers.

Pure functions for common calculations used across providers.
Following functional programming principles - no side effects.

"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl


if TYPE_CHECKING:
    import polars as pl


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """
    Encode a cyclic value as sin/cos pair for neural networks.

    This encoding preserves the cyclic nature of features like time of day,
    day of week, etc. The sin/cos encoding ensures smooth transitions and
    allows models to learn cyclic patterns.

    Parameters
    ----------
    value : float
        The value to encode (e.g., hour of day: 0-23)
    period : float
        The period of the cycle (e.g., 24 for hours in a day)

    Returns
    -------
    tuple[float, float]
        (sin, cos) encoding on the unit circle

    Examples
    --------
    >>> cyclic_encode(0, 24)  # Midnight
    (0.0, 1.0)
    >>> cyclic_encode(6, 24)  # 6 AM
    (1.0, 0.0)
    >>> cyclic_encode(12, 24)  # Noon
    (0.0, -1.0)
    >>> cyclic_encode(18, 24)  # 6 PM
    (-1.0, 0.0)

    Notes
    -----
    The encoding maps the cyclic value to a point on the unit circle:
    - value=0 maps to angle=0 (top of circle)
    - value=period/4 maps to angle=π/2 (right of circle)
    - value=period/2 maps to angle=π (bottom of circle)
    - value=3*period/4 maps to angle=3π/2 (left of circle)

    """
    # Convert value to angle in radians
    angle = 2 * np.pi * value / period

    # Return sin and cos of angle
    return (float(np.sin(angle)), float(np.cos(angle)))


def time_to_event(
    current: datetime,
    event: datetime,
    unit: str = "hours",
) -> float:
    """
    Calculate time until (or since) an event.

    Useful for calculating features like "hours to market close",
    "days to earnings", etc.

    Parameters
    ----------
    current : datetime
        Current timestamp
    event : datetime
        Event timestamp
    unit : str, default "hours"
        Time unit for result: "hours", "days", or "minutes"

    Returns
    -------
    float
        Time to event in specified units
        Positive if event is in future, negative if in past

    Examples
    --------
    >>> from datetime import datetime
    >>> current = datetime(2024, 1, 1, 12, 0)
    >>> event = datetime(2024, 1, 1, 15, 30)
    >>> time_to_event(current, event, "hours")
    3.5
    >>> time_to_event(current, event, "minutes")
    210.0

    Raises
    ------
    ValueError
        If unit is not one of "hours", "days", "minutes"

    """
    # Calculate time difference
    delta = event - current

    # Convert to requested unit
    total_seconds = delta.total_seconds()

    if unit == "hours":
        return total_seconds / 3600
    elif unit == "days":
        return total_seconds / 86400
    elif unit == "minutes":
        return total_seconds / 60
    else:
        raise ValueError(f"Unknown unit: {unit}. Use 'hours', 'days', or 'minutes'")


def validate_timestamps(series: pl.Series) -> bool:
    """
    Validate a series of timestamps.

    Checks that timestamps are:
    - Non-null
    - Monotonically increasing (sorted)
    - Within reasonable range (1970 to 2100)

    Parameters
    ----------
    series : pl.Series
        Series of timestamps to validate

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> import polars as pl
    >>> valid_ts = pl.Series([100, 200, 300])
    >>> validate_timestamps(valid_ts)
    True
    >>> invalid_ts = pl.Series([300, 100, 200])  # Not sorted
    >>> validate_timestamps(invalid_ts)
    False

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Check for nulls
    if series.null_count() > 0:
        return False

    # Check if sorted
    if not series.is_sorted():
        return False

    # Check reasonable range
    # Min: Unix epoch (0)
    # Max: Year 2100 in nanoseconds
    min_ts = series.min()
    max_ts = series.max()

    if min_ts is None or max_ts is None:
        return False

    # Cast to int for comparison (polars timestamps are ints)
    if isinstance(min_ts, int | float):
        if min_ts < 0:
            return False
    else:
        return False  # Unexpected type

    # Year 2100 in nanoseconds since epoch
    max_reasonable = 4102444800000000000
    if isinstance(max_ts, int | float):
        if max_ts > max_reasonable:
            return False
    else:
        return False  # Unexpected type

    return True


def align_timeseries(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    on: str = "timestamp",
    how: str = "inner",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Align two timeseries dataframes on a common column.

    Useful for aligning data from different sources that may have
    different timestamps or missing data.

    Parameters
    ----------
    df1, df2 : pl.DataFrame
        DataFrames to align
    on : str, default "timestamp"
        Column name to align on
    how : str, default "inner"
        Join type: "inner", "left", or "outer"
        - "inner": Only timestamps present in both
        - "left": All timestamps from df1
        - "outer": All timestamps from either

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Aligned dataframes with matching timestamps

    Examples
    --------
    >>> import polars as pl
    >>> df1 = pl.DataFrame({"timestamp": [100, 200, 300], "value1": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"timestamp": [200, 300, 400], "value2": [4, 5, 6]})
    >>> aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "inner")
    >>> aligned1["timestamp"].to_list()
    [200, 300]
    >>> aligned2["timestamp"].to_list()
    [200, 300]

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Get the alignment column from each dataframe
    ts1 = df1[on]
    ts2 = df2[on]

    if how == "inner":
        # Only common timestamps
        common_ts = ts1.filter(ts1.is_in(ts2)).unique().sort()
        df1_aligned = df1.filter(df1[on].is_in(common_ts))
        df2_aligned = df2.filter(df2[on].is_in(common_ts))

    elif how == "left":
        # All timestamps from df1
        df1_aligned = df1
        df2_aligned = df2.filter(df2[on].is_in(ts1))

    elif how == "outer":
        # Union of all timestamps
        all_ts = pl.concat([ts1, ts2]).unique().sort()
        df1_aligned = df1.filter(df1[on].is_in(all_ts))
        df2_aligned = df2.filter(df2[on].is_in(all_ts))

    else:
        raise ValueError(f"Unknown join type: {how}. Use 'inner', 'left', or 'outer'")

    return df1_aligned, df2_aligned
