"""
Market calendar provider for time-based features.

This module provides calendar features including trading hours, holidays, and cyclic
time encodings for ML models.

"""

from __future__ import annotations

import calendar
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast as _cast

from ml._imports import check_ml_dependencies
from ml._imports import pl as pl_runtime
from ml.data.providers.base import BaseTimeSeriesProvider
from ml.data.providers.utils import cyclic_encode


if TYPE_CHECKING:
    from ml.data.sources.calendar import CalendarSource
    import polars as _pl

# Local runtime alias
PL = _cast(Any, pl_runtime)


logger = logging.getLogger(__name__)


class MarketCalendarProvider(BaseTimeSeriesProvider):
    """
    Provider for market calendar features.

    Provides time-based features like trading hours, holidays,
    time-of-day encodings, and other calendar-related features
    that are known in advance (suitable for TFT known-future inputs).

    Attributes
    ----------
    calendar_source : CalendarSource
        Source for market calendar data

    """

    def __init__(self, calendar_source: CalendarSource) -> None:
        """
        Initialize calendar provider.

        Parameters
        ----------
        calendar_source : CalendarSource
            Source for market calendar data

        """
        super().__init__()
        self.calendar = calendar_source
        logger.info(f"Initialized MarketCalendarProvider with {calendar_source.__class__.__name__}")

    def compute_features(
        self,
        timestamps: "_pl.Series",
        exchange: str = "NYSE",
    ) -> "_pl.DataFrame":
        """
        Compute calendar features for timestamps.

        Parameters
        ----------
        timestamps : pl.Series
            Series of timestamps in nanoseconds since epoch
        exchange : str, default "NYSE"
            Exchange identifier for market hours

        Returns
        -------
        pl.DataFrame
            DataFrame with calendar features:
            - timestamp: int
            - is_trading_day: bool
            - is_pre_market: bool
            - is_after_hours: bool
            - minutes_to_close: int
            - hour_sin: float
            - hour_cos: float
            - dow_sin: float (day of week)
            - dow_cos: float
            - month_sin: float
            - month_cos: float
            - is_weekend: bool
            - is_month_start: bool
            - is_month_end: bool
            - days_to_month_end: int

        """
        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensures helpful error if missing

        features: list[dict[str, Any]] = []

        for ts in timestamps:
            # Convert nanoseconds to datetime
            dt = self._to_datetime(ts)

            try:
                # Get market schedule
                schedule = self.calendar.get_schedule(dt, exchange)

                # Time encodings
                hour_sin, hour_cos = cyclic_encode(dt.hour + dt.minute / 60, 24)
                dow_sin, dow_cos = cyclic_encode(dt.weekday(), 7)
                month_sin, month_cos = cyclic_encode(dt.month - 1, 12)

                # Month boundaries
                is_month_start = dt.day <= 3
                _, last_day = calendar.monthrange(dt.year, dt.month)
                is_month_end = dt.day >= (last_day - 2)
                days_to_month_end = last_day - dt.day
                days_from_month_start = dt.day - 1

                # Quarter boundaries
                is_quarter_start = dt.month in [1, 4, 7, 10] and dt.day <= 3
                is_quarter_end = dt.month in [3, 6, 9, 12] and dt.day >= (last_day - 2)

                features.append(
                    {
                        "timestamp": ts,
                        "is_trading_day": schedule.is_trading_day,
                        "is_pre_market": schedule.is_pre_market,
                        "is_after_hours": schedule.is_after_hours,
                        "minutes_to_close": schedule.minutes_to_close,
                        "hour_sin": hour_sin,
                        "hour_cos": hour_cos,
                        "dow_sin": dow_sin,
                        "dow_cos": dow_cos,
                        "month_sin": month_sin,
                        "month_cos": month_cos,
                        "is_weekend": dt.weekday() >= 5,
                        "is_month_start": is_month_start,
                        "is_month_end": is_month_end,
                        "is_quarter_start": is_quarter_start,
                        "is_quarter_end": is_quarter_end,
                        "days_to_month_end": days_to_month_end,
                        "days_from_month_start": days_from_month_start,
                    },
                )

            except Exception as e:
                logger.warning(f"Failed to compute features for {dt}: {e}")
                # Return default features
                features.append(self._default_features(ts, dt))

        from typing import cast as __cast
        return __cast("_pl.DataFrame", PL.DataFrame(features))

    def _to_datetime(self, timestamp_ns: int) -> datetime:
        """
        Convert nanosecond timestamp to datetime.

        Parameters
        ----------
        timestamp_ns : int
            Timestamp in nanoseconds since epoch

        Returns
        -------
        datetime
            Converted datetime

        """
        # Convert nanoseconds to seconds
        timestamp_s = timestamp_ns / 1e9
        return datetime.fromtimestamp(timestamp_s)

    def _default_features(self, timestamp: int, dt: datetime) -> dict[str, Any]:
        """
        Get default features when calendar source fails.

        Parameters
        ----------
        timestamp : int
            Original timestamp
        dt : datetime
            Converted datetime

        Returns
        -------
        dict
            Default feature values

        """
        # Time encodings (always available)
        hour_sin, hour_cos = cyclic_encode(dt.hour + dt.minute / 60, 24)
        dow_sin, dow_cos = cyclic_encode(dt.weekday(), 7)
        month_sin, month_cos = cyclic_encode(dt.month - 1, 12)

        # Month boundaries
        is_month_start = dt.day <= 3
        _, last_day = calendar.monthrange(dt.year, dt.month)
        is_month_end = dt.day >= (last_day - 2)
        days_to_month_end = last_day - dt.day
        days_from_month_start = dt.day - 1

        # Quarter boundaries
        is_quarter_start = dt.month in [1, 4, 7, 10] and dt.day <= 3
        is_quarter_end = dt.month in [3, 6, 9, 12] and dt.day >= (last_day - 2)

        return {
            "timestamp": timestamp,
            "is_trading_day": dt.weekday() < 5,  # Assume weekdays are trading days
            "is_pre_market": False,
            "is_after_hours": False,
            "minutes_to_close": 0,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "is_weekend": dt.weekday() >= 5,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "is_quarter_start": is_quarter_start,
            "is_quarter_end": is_quarter_end,
            "days_to_month_end": days_to_month_end,
            "days_from_month_start": days_from_month_start,
        }

    def load_timeseries(
        self,
        instruments: list[str],
        timestamps: "_pl.Series",
    ) -> "_pl.DataFrame":
        """
        Load time series calendar features.

        This method provides compatibility with the TimeSeriesProvider protocol.
        Since calendar features are not instrument-specific, we replicate
        the features for each instrument.

        Parameters
        ----------
        instruments : list[str]
            List of instruments (not used for calendar features)
        timestamps : pl.Series
            Timestamps to compute features for

        Returns
        -------
        pl.DataFrame
            Calendar features, replicated for each instrument

        """
        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensures helpful error if missing

        # Compute calendar features once
        calendar_df = self.compute_features(timestamps)

        # If multiple instruments, replicate features
        if len(instruments) > 1:
            # Create instrument column
            instrument_dfs: list["_pl.DataFrame"] = []
            for instrument in instruments:
                inst_df = calendar_df.with_columns(
                    PL.lit(instrument).alias("instrument_id"),
                )
                instrument_dfs.append(inst_df)

            from typing import cast as __cast
            return __cast("_pl.DataFrame", PL.concat(instrument_dfs))
        elif len(instruments) == 1:
            # Single instrument
            return calendar_df.with_columns(
                PL.lit(instruments[0]).alias("instrument_id"),
            )
        else:
            # No instruments specified
            return calendar_df

    def _load_timeseries_impl(
        self,
        instruments: list[str],
        timestamps: "_pl.Series",
    ) -> "_pl.DataFrame":
        """
        Implement time series loading.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers.
        timestamps : pl.Series
            Series of timestamps to load data for.

        Returns
        -------
        pl.DataFrame
            Time series data with features.

        """
        return self.load_timeseries(instruments, timestamps)
