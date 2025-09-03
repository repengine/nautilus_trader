"""
Point-in-time join utilities for ML data preparation.

This module provides utilities for performing point-in-time correct joins to avoid
lookahead bias in ML model training and evaluation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


DirectionType = Literal["backward", "forward", "nearest"]


def asof_join(
    left: pl.DataFrame | pd.DataFrame,
    right: pl.DataFrame | pd.DataFrame,
    on: str | list[str],
    by: str | list[str] | None = None,
    tolerance: str | None = None,
    direction: DirectionType = "backward",
) -> pl.DataFrame | pd.DataFrame:
    """
    Perform point-in-time correct as-of join between two dataframes.

    This ensures that we only use information available at or before the timestamp
    in the left dataframe, preventing lookahead bias.

    Parameters
    ----------
    left : DataFrame
        Left dataframe with timestamps to join on
    right : DataFrame
        Right dataframe with reference data
    on : str or list[str]
        Column(s) to perform temporal join on (typically timestamp columns)
    by : str or list[str], optional
        Column(s) to group by before joining (e.g., instrument_id)
    tolerance : str, optional
        Maximum time tolerance for matches (e.g., "1h", "5m")
    direction : str, default "backward"
        Join direction: "backward" (past data only), "forward", or "nearest"

    Returns
    -------
    DataFrame
        Joined dataframe with point-in-time correct data

    Examples
    --------
    >>> # Join market data with corporate events
    >>> market_df = pl.DataFrame({
    ...     "timestamp": [100, 200, 300],
    ...     "instrument_id": ["SPY", "SPY", "SPY"],
    ...     "price": [400.0, 401.0, 402.0]
    ... })
    >>> events_df = pl.DataFrame({
    ...     "timestamp": [150, 250],
    ...     "instrument_id": ["SPY", "SPY"],
    ...     "event": ["earnings", "fed_meeting"]
    ... })
    >>> joined = asof_join(
    ...     market_df, events_df,
    ...     on="timestamp",
    ...     by="instrument_id"
    ... )

    """
    # Check dependencies
    if isinstance(left, pl.DataFrame):
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])
        right_pl = cast(pl.DataFrame, right)
        return _asof_join_polars(left, right_pl, on, by, tolerance, direction)
    else:
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])
        right_pd = cast(pd.DataFrame, right)
        return _asof_join_pandas(left, right_pd, on, by, tolerance, direction)


def _asof_join_polars(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str | list[str],
    by: str | list[str] | None,
    tolerance: str | None,
    direction: DirectionType,
) -> pl.DataFrame:
    """Polars implementation of as-of join."""
    # Ensure on is a string for Polars
    if isinstance(on, list):
        if len(on) != 1:
            msg = "Polars asof join only supports single column"
            raise ValueError(msg)
        on = on[0]

    # Sort both dataframes
    left = left.sort(on)
    right = right.sort(on)

    # Perform as-of join
    kwargs: dict[str, Any] = {
        "other": right,
        "on": on,
        "strategy": direction,
    }

    if by is not None:
        kwargs["by"] = by

    if tolerance is not None:
        kwargs["tolerance"] = tolerance

    return left.join_asof(**kwargs)


def _asof_join_pandas(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    by: str | list[str] | None,
    tolerance: str | pd.Timedelta | None,
    direction: DirectionType,
) -> pd.DataFrame:
    """Pandas implementation of as-of join."""
    # Convert direction naming
    pd_direction = {
        "backward": "backward",
        "forward": "forward",
        "nearest": "nearest",
    }.get(direction, "backward")

    # Ensure on is a string
    if isinstance(on, list):
        if len(on) != 1:
            msg = "Pandas asof join only supports single column"
            raise ValueError(msg)
        on = on[0]

    # Sort both dataframes
    left = left.sort_values(on)
    right = right.sort_values(on)

    # Convert tolerance if provided as string
    if tolerance is not None and isinstance(tolerance, str):
        tolerance = pd.Timedelta(tolerance)

    # Perform merge_asof
    return pd.merge_asof(
        left,
        right,
        on=on,
        by=by,
        tolerance=tolerance,
        direction=cast(Literal["backward", "forward", "nearest"], pd_direction),
    )


def embargo_window(
    df: pl.DataFrame | pd.DataFrame,
    event_timestamps: list[int] | npt.NDArray[np.int64],
    embargo_before_ns: int = 3600_000_000_000,  # 1 hour default
    embargo_after_ns: int = 3600_000_000_000,
    timestamp_col: str = "ts_event",
) -> pl.DataFrame | pd.DataFrame:
    """
    Apply embargo windows around significant events to prevent information leakage.

    This function marks data points that fall within embargo windows around events
    like earnings releases, economic data, or other market-moving announcements.

    Parameters
    ----------
    df : DataFrame
        Dataframe with timestamp column
    event_timestamps : list[int] or array
        Timestamps of events requiring embargo (in nanoseconds)
    embargo_before_ns : int, default 3600_000_000_000
        Embargo window before event in nanoseconds (default 1 hour)
    embargo_after_ns : int, default 3600_000_000_000
        Embargo window after event in nanoseconds (default 1 hour)
    timestamp_col : str, default "ts_event"
        Name of timestamp column

    Returns
    -------
    DataFrame
        Original dataframe with added 'embargo' boolean column

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "ts_event": [100, 200, 300, 400],
    ...     "price": [100.0, 101.0, 102.0, 103.0]
    ... })
    >>> # Embargo around event at timestamp 250
    >>> df_embargo = embargo_window(
    ...     df,
    ...     event_timestamps=[250],
    ...     embargo_before_ns=100,
    ...     embargo_after_ns=100
    ... )
    >>> # Rows at timestamps 200 and 300 will be marked as embargoed

    """
    if isinstance(df, pl.DataFrame):
        return _embargo_window_polars(
            df, event_timestamps, embargo_before_ns, embargo_after_ns, timestamp_col
        )
    else:
        return _embargo_window_pandas(
            df, event_timestamps, embargo_before_ns, embargo_after_ns, timestamp_col
        )


def _embargo_window_polars(
    df: pl.DataFrame,
    event_timestamps: list[int] | npt.NDArray[np.int64],
    embargo_before_ns: int,
    embargo_after_ns: int,
    timestamp_col: str,
) -> pl.DataFrame:
    """Polars implementation of embargo window."""
    # Initialize embargo column as False
    embargo_mask = pl.lit(False)

    # Check each event
    for event_ts in event_timestamps:
        start = event_ts - embargo_before_ns
        end = event_ts + embargo_after_ns

        # Mark rows within embargo window
        event_embargo = (pl.col(timestamp_col) >= start) & (pl.col(timestamp_col) <= end)
        embargo_mask = embargo_mask | event_embargo

    # Add embargo column
    return df.with_columns(embargo_mask.alias("embargo"))


def _embargo_window_pandas(
    df: pd.DataFrame,
    event_timestamps: list[int] | npt.NDArray[np.int64],
    embargo_before_ns: int,
    embargo_after_ns: int,
    timestamp_col: str,
) -> pd.DataFrame:
    """Pandas implementation of embargo window."""
    # Initialize embargo column
    df = df.copy()
    df["embargo"] = False

    # Check each event
    # Normalize to numpy array of int64 to avoid ExtensionArray typing issues
    _ts_values = df[timestamp_col].to_numpy()
    timestamps = np.asarray(_ts_values, dtype=np.int64)
    embargo_mask = np.zeros(len(df), dtype=bool)

    for event_ts in event_timestamps:
        start = event_ts - embargo_before_ns
        end = event_ts + embargo_after_ns

        # Vectorized check for embargo window
        event_embargo = (timestamps >= start) & (timestamps <= end)
        embargo_mask |= event_embargo

    df["embargo"] = embargo_mask
    return df


def validate_no_lookahead(
    features_df: pl.DataFrame | pd.DataFrame,
    targets_df: pl.DataFrame | pd.DataFrame,
    feature_timestamp_col: str = "ts_event",
    target_timestamp_col: str = "ts_event",
) -> bool:
    """
    Validate that features don't contain future information relative to targets.

    Parameters
    ----------
    features_df : DataFrame
        Features dataframe with timestamps
    targets_df : DataFrame
        Targets dataframe with timestamps
    feature_timestamp_col : str
        Timestamp column in features
    target_timestamp_col : str
        Timestamp column in targets

    Returns
    -------
    bool
        True if no lookahead bias detected, False otherwise

    Raises
    ------
    ValueError
        If lookahead bias is detected

    """
    # Get max feature timestamp and min target timestamp
    if isinstance(features_df, pl.DataFrame):
        max_feat_val = features_df[feature_timestamp_col].max()
        min_tgt_val = targets_df[target_timestamp_col].min()
    else:
        max_feat_val = features_df[feature_timestamp_col].max()
        min_tgt_val = targets_df[target_timestamp_col].min()

    # Normalize to numeric for comparison where possible
    max_feature_ts = cast(Any, max_feat_val)
    min_target_ts = cast(Any, min_tgt_val)

    # Handle empty dataframes
    if max_feature_ts is None or min_target_ts is None:
        return True

    # Check for lookahead
    if cast(Any, max_feature_ts) > cast(Any, min_target_ts):
        msg = (
            f"Lookahead bias detected! "
            f"Max feature timestamp {max_feature_ts} > "
            f"Min target timestamp {min_target_ts}"
        )
        raise ValueError(msg)

    return True


def create_lag_features(
    df: pl.DataFrame | pd.DataFrame,
    columns: list[str],
    lags: list[int],
    group_by: str | list[str] | None = None,
    timestamp_col: str = "ts_event",
) -> pl.DataFrame | pd.DataFrame:
    """
    Create lagged features ensuring point-in-time correctness.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    columns : list[str]
        Columns to create lags for
    lags : list[int]
        Number of periods to lag (positive = past values)
    group_by : str or list[str], optional
        Columns to group by (e.g., instrument_id)
    timestamp_col : str
        Timestamp column for ordering

    Returns
    -------
    DataFrame
        Original dataframe with added lag features

    """
    if isinstance(df, pl.DataFrame):
        return _create_lag_features_polars(df, columns, lags, group_by, timestamp_col)
    else:
        return _create_lag_features_pandas(df, columns, lags, group_by, timestamp_col)


def _create_lag_features_polars(
    df: pl.DataFrame,
    columns: list[str],
    lags: list[int],
    group_by: str | list[str] | None,
    timestamp_col: str,
) -> pl.DataFrame:
    """Polars implementation of lag features."""
    # Sort by timestamp
    df = df.sort(timestamp_col)

    # Create lag features
    lag_exprs = []
    for col in columns:
        for lag in lags:
            if group_by:
                lag_expr = pl.col(col).shift(lag).over(group_by).alias(f"{col}_lag_{lag}")
            else:
                lag_expr = pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            lag_exprs.append(lag_expr)

    return df.with_columns(lag_exprs)


def _create_lag_features_pandas(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    group_by: str | list[str] | None,
    timestamp_col: str,
) -> pd.DataFrame:
    """Pandas implementation of lag features."""
    # Sort by timestamp
    df = df.sort_values(timestamp_col).copy()

    # Create lag features
    for col in columns:
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"
            if group_by:
                df[lag_col_name] = df.groupby(group_by)[col].shift(lag)
            else:
                df[lag_col_name] = df[col].shift(lag)

    return df
