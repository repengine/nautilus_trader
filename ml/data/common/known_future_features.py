"""
Known-future feature component for TFT dataset building.

This component extracts and handles calendar and time-based features
that are known in advance (suitable for TFT known-future inputs).

Extracted methods:
- _add_known_future_features_polars() (lines 2076-2151 in legacy)
- _add_known_future_features_pandas() (lines 2153-2208 in legacy)

Features Generated:
1. Time extraction: hour, minute from time_index
2. Cyclical time-of-day: tod_sin, tod_cos
3. Cyclical day-of-week: dow_sin, dow_cos
4. Market session flags: is_market_open, is_premarket, is_aftermarket
5. Optional: MarketCalendarProvider features (when include_calendar=True)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.data.providers.utils import cyclic_encode


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


logger = logging.getLogger(__name__)


class KnownFutureFeatureComponent:
    """
    Component for generating known-future features for TFT models.

    This component computes time-based and calendar features that are
    known in advance, making them suitable for the TFT's "known future"
    input category.

    Features Generated:
        - hour, minute: Time components extracted from time_index
        - tod_sin, tod_cos: Cyclical time-of-day encoding
        - dow, dow_sin, dow_cos: Day-of-week and its cyclical encoding
        - is_market_open: 1 if 9:00-16:00, else 0
        - is_premarket: 1 if 4:00-9:00, else 0
        - is_aftermarket: 1 if 16:00-20:00, else 0

    Optional (when include_calendar=True):
        - Calendar provider features from MarketCalendarProvider

    Example:
        >>> component = KnownFutureFeatureComponent()
        >>> df_with_features = component.add_known_future_features_polars(df)
        >>> assert "tod_sin" in df_with_features.columns
        >>> assert "is_market_open" in df_with_features.columns

    """

    def __init__(self, *, include_calendar: bool = False) -> None:
        """
        Initialize KnownFutureFeatureComponent.

        Args:
            include_calendar: Whether to include precise calendar features
                from MarketCalendarProvider. Default False for performance.

        """
        self.include_calendar = include_calendar

    def add_known_future_features_polars(
        self,
        df: _pl.DataFrame,
    ) -> _pl.DataFrame:
        """
        Add known-future time and calendar features using Polars.

        Requires 'time_index' column in the DataFrame representing
        minutes since epoch or similar time unit.

        Args:
            df: Polars DataFrame with 'time_index' column.

        Returns:
            DataFrame with additional known-future feature columns:
            - hour, minute: Time components
            - tod_sin, tod_cos: Time-of-day cyclical encoding
            - dow, dow_sin, dow_cos: Day-of-week features
            - is_market_open, is_premarket, is_aftermarket: Session flags

        Raises:
            KeyError: If 'time_index' column is missing.

        Example:
            >>> component = KnownFutureFeatureComponent()
            >>> df = pl.DataFrame({"time_index": [540, 600, 660]})  # 9:00, 10:00, 11:00
            >>> result = component.add_known_future_features_polars(df)
            >>> assert result["is_market_open"].to_list() == [1, 1, 1]

        """
        if "time_index" not in df.columns:
            raise KeyError("Missing required 'time_index' column for known-future features")

        # Handle empty DataFrame
        if df.is_empty():
            return df

        # Create hour and minute from time_index (assuming minute bars)
        df = df.with_columns(
            [
                ((pl.col("time_index") // 60) % 24).alias("hour"),
                (pl.col("time_index") % 60).alias("minute"),
            ],
        )

        # Time of day features (cyclical encoding)
        df = df.with_columns(
            [
                (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60))
                .sin()
                .alias("tod_sin"),
                (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60))
                .cos()
                .alias("tod_cos"),
            ],
        )

        # Day of week (simplified - assuming continuous trading for now)
        df = df.with_columns(
            [
                ((pl.col("time_index") // (24 * 60)) % 7).alias("dow"),
            ],
        )

        df = df.with_columns(
            [
                (2 * np.pi * pl.col("dow") / 7).sin().alias("dow_sin"),
                (2 * np.pi * pl.col("dow") / 7).cos().alias("dow_cos"),
            ],
        )

        # Market session flags
        df = df.with_columns(
            [
                ((pl.col("hour") >= 9) & (pl.col("hour") < 16))
                .cast(pl.Int32)
                .alias("is_market_open"),
                ((pl.col("hour") >= 4) & (pl.col("hour") < 9))
                .cast(pl.Int32)
                .alias("is_premarket"),
                ((pl.col("hour") >= 16) & (pl.col("hour") < 20))
                .cast(pl.Int32)
                .alias("is_aftermarket"),
            ],
        )

        # Optional: precise market calendar features (known-future)
        if self.include_calendar:
            df = self._join_calendar_features_polars(df)

        return df

    def add_known_future_features_pandas(
        self,
        df: _pd.DataFrame,
    ) -> _pd.DataFrame:
        """
        Add known-future time and calendar features using Pandas.

        Identical logic to add_known_future_features_polars but using
        Pandas operations. Both implementations produce identical outputs.

        Args:
            df: Pandas DataFrame with 'time_index' column.

        Returns:
            DataFrame with additional known-future feature columns.

        Raises:
            KeyError: If 'time_index' column is missing.

        Example:
            >>> component = KnownFutureFeatureComponent()
            >>> df = pd.DataFrame({"time_index": [540, 600, 660]})
            >>> result = component.add_known_future_features_pandas(df)
            >>> assert result["is_market_open"].tolist() == [1, 1, 1]

        """
        if "time_index" not in df.columns:
            raise KeyError("Missing required 'time_index' column for known-future features")

        # Handle empty DataFrame
        if len(df) == 0:
            return df

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Create hour and minute from time_index (assuming minute bars)
        df["hour"] = (df["time_index"] // 60) % 24
        df["minute"] = df["time_index"] % 60

        # Time of day features (cyclical encoding)
        time_in_minutes = df["hour"] * 60 + df["minute"]
        # Use centralized cyclic_encode for clarity and DRY
        sincos = time_in_minutes.apply(lambda v: cyclic_encode(float(v), 24 * 60))
        df["tod_sin"] = sincos.apply(lambda t: t[0])
        df["tod_cos"] = sincos.apply(lambda t: t[1])

        # Day of week (simplified - assuming continuous trading for now)
        df["dow"] = (df["time_index"] // (24 * 60)) % 7
        # Day-of-week cyclic encoding via centralized helper
        dowsc = df["dow"].apply(lambda d: cyclic_encode(float(d), 7))
        df["dow_sin"] = dowsc.apply(lambda t: t[0])
        df["dow_cos"] = dowsc.apply(lambda t: t[1])

        # Market session flags
        df["is_market_open"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
        df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
        df["is_aftermarket"] = ((df["hour"] >= 16) & (df["hour"] < 20)).astype(int)

        # Optional: precise market calendar features via provider
        if self.include_calendar:
            df = self._join_calendar_features_pandas(df)

        return df

    def _join_calendar_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Join MarketCalendarProvider features to a Polars DataFrame.

        Args:
            df: DataFrame with timestamp and optionally instrument_id columns.

        Returns:
            DataFrame with calendar features joined.

        """
        try:
            from ml.data.providers.calendar import MarketCalendarProvider
            from ml.data.sources.calendar import PandasCalendarSource

            # Determine instrument(s) for this frame
            instruments = (
                df.select(pl.col("instrument_id")).unique()["instrument_id"].to_list()
                if "instrument_id" in df.columns
                else ["GLOBAL"]
            )

            provider = MarketCalendarProvider(PandasCalendarSource())
            ts_series = df.select(pl.col("timestamp").cast(pl.Int64))["timestamp"]
            cal = provider.load_timeseries(instruments, ts_series)

            if not cal.is_empty():
                cal = cal.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                join_keys: list[str] = ["timestamp"]
                if "instrument_id" in df.columns and "instrument_id" in cal.columns:
                    join_keys.append("instrument_id")
                df = df.join(cal, on=join_keys, how="left")

        except Exception as exc:  # pragma: no cover
            logger.debug(f"Calendar feature join skipped: {exc}", exc_info=True)

        return df

    def _join_calendar_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Join MarketCalendarProvider features to a Pandas DataFrame.

        Args:
            df: DataFrame with timestamp and optionally instrument_id columns.

        Returns:
            DataFrame with calendar features joined.

        """
        try:
            from ml.data.providers.calendar import MarketCalendarProvider
            from ml.data.sources.calendar import PandasCalendarSource

            provider = MarketCalendarProvider(PandasCalendarSource())
            ts_series = pl.Series(df["timestamp"].astype("int64").to_numpy())
            instruments = (
                list({str(v) for v in df["instrument_id"].astype(str).tolist()})
                if "instrument_id" in df.columns
                else ["GLOBAL"]
            )

            cal_pl = provider.load_timeseries(instruments, ts_series)

            if cal_pl.shape[0] > 0:
                cal_pl = cal_pl.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                )
                cal_pd = cal_pl.to_pandas()
                join_cols = ["timestamp"] + (
                    ["instrument_id"] if "instrument_id" in df.columns else []
                )
                df = df.merge(cal_pd, on=join_cols, how="left")

        except Exception as exc:  # pragma: no cover
            logger.debug(f"Calendar feature join skipped: {exc}", exc_info=True)

        return df
