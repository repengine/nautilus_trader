#!/usr/bin/env python3

"""
Vintage policy enforcement for temporal data filtering.

This module provides vintage policy enforcement to ensure temporal boundaries
in training data. It filters datasets based on data age constraints, ensuring
that only recent data within a specified window is used for training.

Universal ML Architecture Patterns Compliance:
- Pattern 2: Protocol-first interface design
- Pattern 3: Cold-path only - no hot-path operations
- Pattern 4: Progressive fallback chains for missing data
- Pattern 5: Uses centralized metrics bootstrap

Notes
-----
- Vintage policies are applied during dataset construction
- All filtering is based on data timestamps (cold-path only)
- Metadata tracking enables audit trails for temporal filtering
- Cutoff dates are computed from reference dates (typically current date)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VintagePolicy:
    """
    Vintage policy for enforcing temporal boundaries in training data.

    This policy filters datasets to include only data within a specified
    age window. Data older than max_age_days is excluded from the dataset.

    Attributes
    ----------
    max_age_days : int
        Maximum age of data in days (must be > 0)

    Examples
    --------
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> policy = VintagePolicy(max_age_days=30)
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.date_range('2024-01-01', '2024-12-31', freq='D'),
    ...     'value': range(365)
    ... })
    >>> current_date = datetime(2024, 12, 15)
    >>> filtered = policy.filter_by_vintage(df, current_date)
    >>> # Only data from 2024-11-15 to 2024-12-15
    >>> assert len(filtered) == 31  # 30 days + current day

    """

    max_age_days: int

    def __post_init__(self) -> None:
        """
        Validate vintage policy parameters.

        Raises
        ------
        ValueError
            If max_age_days <= 0

        """
        if self.max_age_days <= 0:
            msg = f"max_age_days must be > 0, got {self.max_age_days}"
            raise ValueError(msg)

    def filter_by_vintage(
        self,
        df: pd.DataFrame,
        current_date: datetime,
        *,
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Filter dataframe by vintage policy.

        Applies temporal filtering to exclude data older than the vintage window.
        The cutoff date is computed as current_date - max_age_days.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with timestamp column
        current_date : datetime
            Reference date for vintage calculation (typically today's date)
        timestamp_column : str, default="timestamp"
            Name of timestamp column to use for filtering

        Returns
        -------
        pd.DataFrame
            Filtered dataframe with data within vintage window

        Raises
        ------
        ValueError
            If timestamp_column not found in dataframe

        Examples
        --------
        >>> policy = VintagePolicy(max_age_days=30)
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', '2024-12-31', freq='D'),
        ...     'close': range(365)
        ... })
        >>> filtered = policy.filter_by_vintage(df, datetime(2024, 12, 15))
        >>> assert all(filtered['timestamp'] >= pd.Timestamp('2024-11-15'))
        >>> assert all(filtered['timestamp'] <= pd.Timestamp('2024-12-15'))

        """
        if timestamp_column not in df.columns:
            msg = (
                f"Timestamp column '{timestamp_column}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
            raise ValueError(msg)

        # Compute cutoff date
        cutoff = current_date - timedelta(days=self.max_age_days)

        # Convert to pandas Timestamp for comparison
        cutoff_ts = pd.Timestamp(cutoff)

        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            logger.debug(
                f"Converting {timestamp_column} to datetime for vintage filtering",
                extra={"column_dtype": str(df[timestamp_column].dtype)},
            )
            try:
                df_copy = df.copy()
                df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
            except Exception as exc:
                logger.error(
                    f"Failed to convert {timestamp_column} to datetime",
                    exc_info=True,
                    extra={"error": str(exc)},
                )
                raise ValueError(
                    f"Cannot convert {timestamp_column} to datetime: {exc}"
                ) from exc
        else:
            df_copy = df

        # Convert current_date to pandas Timestamp for comparison
        current_ts = pd.Timestamp(current_date)

        # Filter rows within vintage window (cutoff <= timestamp <= current_date)
        mask = (df_copy[timestamp_column] >= cutoff_ts) & (
            df_copy[timestamp_column] <= current_ts
        )
        filtered_df = df_copy[mask].copy()

        # Log filtering statistics
        original_count = len(df)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count

        logger.info(
            f"Vintage filtering applied: {original_count} rows → {filtered_count} rows",
            extra={
                "cutoff_date": cutoff.date().isoformat(),
                "max_age_days": self.max_age_days,
                "current_date": current_date.date().isoformat(),
                "original_count": original_count,
                "filtered_count": filtered_count,
                "removed_count": removed_count,
                "removal_pct": (
                    round(100.0 * removed_count / original_count, 2)
                    if original_count > 0
                    else 0.0
                ),
            },
        )

        return filtered_df

    def compute_vintage_metadata(
        self,
        current_date: datetime,
        original_count: int,
        filtered_count: int,
    ) -> dict[str, Any]:
        """
        Compute vintage metadata for storage and audit trails.

        Parameters
        ----------
        current_date : datetime
            Reference date used for vintage calculation
        original_count : int
            Number of rows before filtering
        filtered_count : int
            Number of rows after filtering

        Returns
        -------
        dict[str, Any]
            Vintage metadata with cutoff dates, row counts, and policy settings

        Examples
        --------
        >>> policy = VintagePolicy(max_age_days=30)
        >>> metadata = policy.compute_vintage_metadata(
        ...     current_date=datetime(2024, 12, 15),
        ...     original_count=365,
        ...     filtered_count=31,
        ... )
        >>> assert metadata['vintage_policy']['max_age_days'] == 30
        >>> assert metadata['vintage_policy']['cutoff_date'] == '2024-11-15'
        >>> assert metadata['vintage_policy']['rows_removed'] == 334

        """
        cutoff = current_date - timedelta(days=self.max_age_days)

        return {
            "vintage_policy": {
                "max_age_days": self.max_age_days,
                "cutoff_date": cutoff.date().isoformat(),
                "current_date": current_date.date().isoformat(),
                "original_count": original_count,
                "filtered_count": filtered_count,
                "rows_removed": original_count - filtered_count,
                "removal_pct": (
                    round(100.0 * (original_count - filtered_count) / original_count, 2)
                    if original_count > 0
                    else 0.0
                ),
            }
        }


__all__ = [
    "VintagePolicy",
]
