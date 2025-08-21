"""
Unit tests for calendar provider and sources.

Tests market calendar features, trading hours, and time encodings.

"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.strategies import datetimes

from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.sources.calendar import CalendarSource
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import SimpleCalendarSource


class TestMockCalendarSource:
    """
    Test mock calendar source.
    """

    def test_mock_source_generates_schedule(self) -> None:
        """
        Test that mock source generates valid schedule.
        """
        source = MockCalendarSource()

        # Test single date
        dt = datetime(2024, 1, 15, 10, 30)  # Monday 10:30 AM
        schedule = source.get_schedule(dt, "NYSE")

        assert schedule.is_trading_day is True  # Monday is trading day
        assert schedule.market_open == datetime(2024, 1, 15, 9, 30)
        assert schedule.market_close == datetime(2024, 1, 15, 16, 0)

    def test_mock_source_weekend_detection(self) -> None:
        """
        Test that mock source correctly identifies weekends.
        """
        source = MockCalendarSource()

        # Saturday
        saturday = datetime(2024, 1, 13, 12, 0)
        schedule_sat = source.get_schedule(saturday, "NYSE")
        assert schedule_sat.is_trading_day is False

        # Sunday
        sunday = datetime(2024, 1, 14, 12, 0)
        schedule_sun = source.get_schedule(sunday, "NYSE")
        assert schedule_sun.is_trading_day is False

        # Monday
        monday = datetime(2024, 1, 15, 12, 0)
        schedule_mon = source.get_schedule(monday, "NYSE")
        assert schedule_mon.is_trading_day is True

    def test_mock_source_pre_after_market(self) -> None:
        """
        Test pre-market and after-hours detection.
        """
        source = MockCalendarSource()

        # Pre-market (7 AM)
        pre_market = datetime(2024, 1, 15, 7, 0)
        schedule_pre = source.get_schedule(pre_market, "NYSE")
        assert schedule_pre.is_pre_market is True
        assert schedule_pre.is_after_hours is False
        assert schedule_pre.is_market_hours is False

        # Market hours (10 AM)
        market_hours = datetime(2024, 1, 15, 10, 0)
        schedule_market = source.get_schedule(market_hours, "NYSE")
        assert schedule_market.is_pre_market is False
        assert schedule_market.is_after_hours is False
        assert schedule_market.is_market_hours is True

        # After hours (5 PM)
        after_hours = datetime(2024, 1, 15, 17, 0)
        schedule_after = source.get_schedule(after_hours, "NYSE")
        assert schedule_after.is_pre_market is False
        assert schedule_after.is_after_hours is True
        assert schedule_after.is_market_hours is False

    def test_mock_source_holidays(self) -> None:
        """
        Test that mock source handles holidays.
        """
        source = MockCalendarSource()

        # New Year's Day
        new_year = datetime(2024, 1, 1, 12, 0)
        schedule = source.get_schedule(new_year, "NYSE")
        assert schedule.is_trading_day is False
        assert schedule.is_holiday is True

    @given(
        dt=datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31),
        ),
    )
    @settings(max_examples=20)
    def test_mock_source_handles_any_date(self, dt: datetime) -> None:
        """Property test: mock source handles any date."""
        source = MockCalendarSource()
        schedule = source.get_schedule(dt, "NYSE")

        # Should always return valid schedule
        assert isinstance(schedule.is_trading_day, bool)
        assert isinstance(schedule.market_open, datetime)
        assert isinstance(schedule.market_close, datetime)

        # Market close should be after market open
        assert schedule.market_close > schedule.market_open


class TestSimpleCalendarSource:
    """
    Test simple calendar source.
    """

    def test_simple_source_basic_schedule(self) -> None:
        """
        Test simple source returns basic NYSE schedule.
        """
        source = SimpleCalendarSource()

        # Regular trading day
        dt = datetime(2024, 1, 15, 12, 0)  # Monday noon
        schedule = source.get_schedule(dt, "NYSE")

        assert schedule.is_trading_day is True
        assert schedule.market_open.hour == 9
        assert schedule.market_open.minute == 30
        assert schedule.market_close.hour == 16
        assert schedule.market_close.minute == 0

    def test_simple_source_minutes_to_close(self) -> None:
        """
        Test calculation of minutes to market close.
        """
        source = SimpleCalendarSource()

        # 2 hours before close (2 PM)
        dt = datetime(2024, 1, 15, 14, 0)
        schedule = source.get_schedule(dt, "NYSE")
        assert schedule.minutes_to_close == 120  # 2 hours = 120 minutes

        # 30 minutes before close (3:30 PM)
        dt = datetime(2024, 1, 15, 15, 30)
        schedule = source.get_schedule(dt, "NYSE")
        assert schedule.minutes_to_close == 30

        # After market close
        dt = datetime(2024, 1, 15, 17, 0)
        schedule = source.get_schedule(dt, "NYSE")
        assert schedule.minutes_to_close == 0  # Market closed

    def test_simple_source_different_exchanges(self) -> None:
        """
        Test simple source handles different exchanges.
        """
        source = SimpleCalendarSource()

        dt = datetime(2024, 1, 15, 12, 0)

        # NYSE
        nyse_schedule = source.get_schedule(dt, "NYSE")
        assert nyse_schedule.exchange == "NYSE"

        # NASDAQ (should have same hours as NYSE in simple implementation)
        nasdaq_schedule = source.get_schedule(dt, "NASDAQ")
        assert nasdaq_schedule.exchange == "NASDAQ"
        assert nasdaq_schedule.market_open == nyse_schedule.market_open

        # Unknown exchange (should use default)
        unknown_schedule = source.get_schedule(dt, "UNKNOWN")
        assert unknown_schedule.exchange == "UNKNOWN"


class TestMarketCalendarProvider:
    """
    Test the main calendar provider.
    """

    def test_provider_computes_features(self) -> None:
        """
        Test provider computes calendar features.
        """
        mock_source = MockCalendarSource()
        provider = MarketCalendarProvider(mock_source)

        # Create timestamps
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),  # Monday 10 AM
                int(datetime(2024, 1, 15, 15, 30).timestamp() * 1e9),  # Monday 3:30 PM
                int(datetime(2024, 1, 13, 12, 0).timestamp() * 1e9),  # Saturday noon
            ],
        )

        df = provider.compute_features(timestamps, exchange="NYSE")

        # Check columns exist
        expected_cols = {
            "timestamp",
            "is_trading_day",
            "is_pre_market",
            "is_after_hours",
            "minutes_to_close",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        }
        assert expected_cols.issubset(df.columns)

        # Check data
        assert len(df) == 3

        # Monday 10 AM should be trading day
        monday_row = df[0]
        assert monday_row["is_trading_day"][0] is True
        assert monday_row["is_weekend"][0] is False

        # Saturday should be weekend
        saturday_row = df[2]
        assert saturday_row["is_trading_day"][0] is False
        assert saturday_row["is_weekend"][0] is True

    def test_provider_cyclic_encoding(self) -> None:
        """
        Test that cyclic encodings are valid.
        """
        provider = MarketCalendarProvider(MockCalendarSource())

        # Test various times
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 0, 0).timestamp() * 1e9),  # Midnight
                int(datetime(2024, 1, 15, 6, 0).timestamp() * 1e9),  # 6 AM
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),  # Noon
                int(datetime(2024, 1, 15, 18, 0).timestamp() * 1e9),  # 6 PM
            ],
        )

        df = provider.compute_features(timestamps)

        # Check that sin^2 + cos^2 = 1 for all cyclic features
        for i in range(len(df)):
            hour_sin = df["hour_sin"][i]
            hour_cos = df["hour_cos"][i]
            assert abs(hour_sin**2 + hour_cos**2 - 1.0) < 1e-10

            dow_sin = df["dow_sin"][i]
            dow_cos = df["dow_cos"][i]
            assert abs(dow_sin**2 + dow_cos**2 - 1.0) < 1e-10

    def test_provider_month_boundaries(self) -> None:
        """
        Test month start/end detection.
        """
        provider = MarketCalendarProvider(MockCalendarSource())

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 1, 12, 0).timestamp() * 1e9),  # Jan 1
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),  # Jan 15
                int(datetime(2024, 1, 31, 12, 0).timestamp() * 1e9),  # Jan 31
                int(datetime(2024, 2, 1, 12, 0).timestamp() * 1e9),  # Feb 1
            ],
        )

        df = provider.compute_features(timestamps)

        # Jan 1 should be month start
        assert df["is_month_start"][0] is True
        assert df["is_month_end"][0] is False

        # Jan 15 should be neither
        assert df["is_month_start"][1] is False
        assert df["is_month_end"][1] is False

        # Jan 31 should be month end
        assert df["is_month_start"][2] is False
        assert df["is_month_end"][2] is True

        # Feb 1 should be month start
        assert df["is_month_start"][3] is True
        assert df["is_month_end"][3] is False

    def test_provider_days_to_month_end(self) -> None:
        """
        Test calculation of days to month end.
        """
        provider = MarketCalendarProvider(MockCalendarSource())

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 1, 12, 0).timestamp() * 1e9),  # Jan 1 -> 30 days
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),  # Jan 15 -> 16 days
                int(datetime(2024, 1, 31, 12, 0).timestamp() * 1e9),  # Jan 31 -> 0 days
                int(datetime(2024, 2, 28, 12, 0).timestamp() * 1e9),  # Feb 28 -> 1 day (leap year)
            ],
        )

        df = provider.compute_features(timestamps)

        assert df["days_to_month_end"][0] == 30
        assert df["days_to_month_end"][1] == 16
        assert df["days_to_month_end"][2] == 0
        assert df["days_to_month_end"][3] == 1  # 2024 is leap year

    @given(
        timestamps=st.lists(
            st.integers(
                min_value=int(datetime(2020, 1, 1).timestamp() * 1e9),
                max_value=int(datetime(2030, 12, 31).timestamp() * 1e9),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=10)
    def test_provider_handles_any_timestamps(self, timestamps: list[int]) -> None:
        """Property test: provider handles any valid timestamps."""
        provider = MarketCalendarProvider(MockCalendarSource())

        ts_series = pl.Series("timestamp", timestamps)
        df = provider.compute_features(ts_series)

        # Should return data for all timestamps
        assert len(df) == len(timestamps)

        # All cyclic features should be valid
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert (df[col] >= -1.0).all()
            assert (df[col] <= 1.0).all()

        # Boolean columns should be boolean
        for col in ["is_trading_day", "is_weekend", "is_month_start", "is_month_end"]:
            assert df[col].dtype == pl.Boolean

    def test_provider_handles_source_errors(self) -> None:
        """
        Test provider handles source errors gracefully.
        """
        mock_source = MagicMock(spec=CalendarSource)
        mock_source.get_schedule.side_effect = Exception("API Error")

        provider = MarketCalendarProvider(mock_source)

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
            ],
        )

        # Should return default features
        df = provider.compute_features(timestamps)
        assert len(df) == 1

        # Should have all columns with default values
        assert "is_trading_day" in df.columns
        assert "hour_sin" in df.columns
