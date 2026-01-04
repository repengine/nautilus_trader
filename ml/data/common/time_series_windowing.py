"""
Time series windowing component for TFT dataset building.

This component extracts and handles time bounds, windowing operations, and multi-symbol
timestamp alignment from the legacy TFTDatasetBuilder.

"""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pl = Any


# Runtime alias
pl: Any = cast(Any, pl_runtime)


logger = logging.getLogger(__name__)


class TimeSeriesWindowingComponent:
    """
    Component for time series windowing and timestamp operations.

    This component provides methods for:
    - Extracting time bounds from DataFrames
    - Converting various types to nanosecond timestamps
    - Windowing data by time ranges
    - Creating sliding windows for sequence models
    - Aligning timestamps across multiple symbols

    All methods preserve timestamp monotonicity and handle edge cases
    gracefully (empty DataFrames, missing columns, etc.).

    Example:
        >>> component = TimeSeriesWindowingComponent()
        >>> bounds = component.frame_time_bounds(df)
        >>> windowed = component.window_by_time_range(df, start_dt, end_dt)
        >>> windows = component.create_sliding_windows(df, window_size=20, stride=5)

    """

    @staticmethod
    def frame_time_bounds(frame: _pl.DataFrame) -> tuple[int | None, int | None]:
        """
        Extract minimum and maximum timestamp bounds from a DataFrame.

        Looks for 'timestamp' column first, then falls back to 'ts_event'.
        Returns nanosecond timestamps or (None, None) for empty/invalid frames.

        Args:
            frame: Polars DataFrame with timestamp or ts_event column

        Returns:
            Tuple of (min_timestamp_ns, max_timestamp_ns) or (None, None)
            if the frame is empty or has no valid timestamp column.

        Example:
            >>> df = pl.DataFrame({"timestamp": [1000, 2000, 3000]})
            >>> bounds = TimeSeriesWindowingComponent.frame_time_bounds(df)
            >>> assert bounds == (1000, 3000)

        """
        if frame.is_empty():
            return (None, None)

        series = None
        if "timestamp" in frame.columns:
            series = frame.get_column("timestamp")
        elif "ts_event" in frame.columns:
            series = frame.get_column("ts_event")

        if series is None:
            return (None, None)

        try:
            ts_min = series.min()
            ts_max = series.max()
        except Exception:
            return (None, None)

        return (
            TimeSeriesWindowingComponent.coerce_to_ns(ts_min),
            TimeSeriesWindowingComponent.coerce_to_ns(ts_max),
        )

    @staticmethod
    def coerce_to_ns(value: object) -> int | None:
        """
        Coerce a value to nanoseconds since epoch.

        Handles various input types:
        - None -> None
        - int/float -> int
        - numpy generic types -> int
        - datetime objects -> nanoseconds (assumes UTC if naive)
        - Other types -> None

        Args:
            value: Value to convert to nanoseconds

        Returns:
            Integer nanoseconds or None if conversion not possible.

        Example:
            >>> TimeSeriesWindowingComponent.coerce_to_ns(1000000000)
            1000000000
            >>> TimeSeriesWindowingComponent.coerce_to_ns(datetime(2024, 1, 1, tzinfo=UTC))
            1704067200000000000

        """
        if value is None:
            return None
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, np.generic):
            return int(value)
        if isinstance(value, datetime):
            dt_value = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            dt_utc = dt_value.astimezone(UTC)
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            delta = dt_utc - epoch
            day_ns = 86_400 * 1_000_000_000
            return delta.days * day_ns + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
        return None

    @staticmethod
    def datetime_to_ns(value: datetime | None, *, fallback: int) -> int:
        """
        Convert a datetime to nanoseconds with fallback for None.

        Handles timezone-naive datetimes by assuming UTC.

        Args:
            value: Datetime to convert, or None
            fallback: Value to return if value is None

        Returns:
            Nanoseconds since epoch, or fallback if value is None.

        Example:
            >>> TimeSeriesWindowingComponent.datetime_to_ns(None, fallback=0)
            0
            >>> dt = datetime(2024, 1, 1, tzinfo=UTC)
            >>> ns = TimeSeriesWindowingComponent.datetime_to_ns(dt, fallback=0)
            >>> assert ns == 1704067200000000000

        """
        if value is None:
            return fallback
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        delta = value - epoch
        day_ns = 86_400 * 1_000_000_000
        return delta.days * day_ns + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000

    def window_by_time_range(
        self,
        df: _pl.DataFrame,
        start: datetime | None,
        end: datetime | None,
    ) -> _pl.DataFrame:
        """
        Filter DataFrame to rows within a time range.

        Args:
            df: Polars DataFrame with timestamp column
            start: Start datetime (inclusive), or None for no lower bound
            end: End datetime (exclusive), or None for no upper bound

        Returns:
            Filtered DataFrame with only rows in the time range.

        Raises:
            ValueError: If start > end (invalid range)

        Example:
            >>> start = datetime(2024, 1, 15, tzinfo=UTC)
            >>> end = datetime(2024, 1, 20, tzinfo=UTC)
            >>> windowed = component.window_by_time_range(df, start, end)

        """
        if start is not None and end is not None and start > end:
            raise ValueError(
                f"Invalid time range: start ({start}) must be <= end ({end})",
            )

        if df.is_empty():
            return df

        # Determine timestamp column
        ts_col = None
        if "timestamp" in df.columns:
            ts_col = "timestamp"
        elif "ts_event" in df.columns:
            ts_col = "ts_event"

        if ts_col is None:
            logger.warning("No timestamp column found for windowing")
            return df

        # Build filter condition using datetime comparison
        # This handles both datetime columns and integer nanosecond columns
        result = df

        if start is not None:
            # Ensure start has timezone
            start_utc = start if start.tzinfo is not None else start.replace(tzinfo=UTC)
            result = result.filter(pl.col(ts_col) >= start_utc)

        if end is not None:
            # Ensure end has timezone
            end_utc = end if end.tzinfo is not None else end.replace(tzinfo=UTC)
            result = result.filter(pl.col(ts_col) < end_utc)

        return result

    def create_sliding_windows(
        self,
        df: _pl.DataFrame,
        window_size: int,
        stride: int = 1,
    ) -> list[_pl.DataFrame]:
        """
        Create overlapping sliding windows from a DataFrame.

        Each window contains exactly window_size rows, with windows
        starting stride rows apart.

        Args:
            df: Input DataFrame to window
            window_size: Number of rows in each window
            stride: Number of rows between window starts (default: 1)

        Returns:
            List of DataFrames, each with window_size rows.
            Returns empty list if data has fewer rows than window_size.

        Raises:
            ValueError: If window_size <= 0 or stride <= 0

        Example:
            >>> # 100 rows, window=20, stride=5 -> 17 windows
            >>> windows = component.create_sliding_windows(df, window_size=20, stride=5)
            >>> assert len(windows) == 17
            >>> assert all(len(w) == 20 for w in windows)

        """
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        n_rows = len(df)

        if n_rows < window_size:
            return []

        windows: list[_pl.DataFrame] = []
        start_idx = 0

        while start_idx + window_size <= n_rows:
            window = df.slice(start_idx, window_size)
            windows.append(window)
            start_idx += stride

        return windows

    def align_multi_symbol_timestamps(
        self,
        frames: dict[str, _pl.DataFrame],
    ) -> dict[str, _pl.DataFrame]:
        """
        Align multiple symbol DataFrames to a common timestamp grid.

        Returns only rows where timestamps are present in ALL input DataFrames.
        This ensures synchronization across symbols for multi-asset analysis.

        Args:
            frames: Dictionary mapping symbol names to their DataFrames

        Returns:
            Dictionary with same keys, but DataFrames filtered to only
            include timestamps present in all inputs. Empty DataFrames
            are returned if there's no timestamp overlap.

        Example:
            >>> frames = {"SPY": spy_df, "QQQ": qqq_df}
            >>> aligned = component.align_multi_symbol_timestamps(frames)
            >>> # All aligned frames now have the same timestamp set

        """
        if not frames:
            return {}

        # Handle single frame case
        if len(frames) == 1:
            return frames.copy()

        # Collect timestamp sets from each frame
        ts_sets: list[set[Any]] = []

        for symbol, frame in frames.items():
            if frame.is_empty():
                # Empty frame means no common timestamps possible
                return {sym: frame.head(0) for sym, frame in frames.items()}

            ts_col = None
            if "timestamp" in frame.columns:
                ts_col = "timestamp"
            elif "ts_event" in frame.columns:
                ts_col = "ts_event"

            if ts_col is None:
                logger.warning(f"No timestamp column in {symbol} frame")
                return {sym: frame.head(0) for sym, frame in frames.items()}

            # Get unique timestamps as a set
            ts_values = set(frame.get_column(ts_col).to_list())
            ts_sets.append(ts_values)

        # Find intersection of all timestamp sets
        common_ts = ts_sets[0]
        for ts_set in ts_sets[1:]:
            common_ts = common_ts.intersection(ts_set)

        if not common_ts:
            # No common timestamps - return empty frames with schema preserved
            return {sym: frame.head(0) for sym, frame in frames.items()}

        # Filter each frame to common timestamps
        aligned: dict[str, _pl.DataFrame] = {}

        for symbol, frame in frames.items():
            ts_col = "timestamp" if "timestamp" in frame.columns else "ts_event"
            aligned[symbol] = frame.filter(pl.col(ts_col).is_in(list(common_ts)))

        return aligned
