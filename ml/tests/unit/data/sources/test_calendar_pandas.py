"""
Unit tests for PandasCalendarSource.

Tests the real market calendar implementation with pandas_market_calendars.

"""

from __future__ import annotations

from datetime import datetime
from datetime import time
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_PANDAS_MARKET_CALENDARS
from ml.data.sources.calendar import CalendarSource
from ml.data.sources.calendar import MarketSchedule
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import PandasCalendarSource
from ml.data.sources.calendar import SimpleCalendarSource


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestPandasCalendarSource:
    """
    Test suite for PandasCalendarSource.
    """

    def test_init_with_pandas_market_calendars_available(self) -> None:
        """
        Test initialization when pandas_market_calendars is available.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()
            assert source._use_fallback is False
            assert isinstance(source._fallback, SimpleCalendarSource)
            assert source._cache_ttl == timedelta(hours=24)

    def test_init_without_pandas_market_calendars(self) -> None:
        """
        Test initialization when pandas_market_calendars is unavailable.
        """
        # No patching needed - use force_fallback parameter
        source = PandasCalendarSource(force_fallback=True)
        assert source._use_fallback is True
        assert isinstance(source._fallback, SimpleCalendarSource)

    def test_init_with_custom_fallback(self) -> None:
        """
        Test initialization with custom fallback source.
        """
        mock_fallback = MockCalendarSource()
        # No patching needed - use force_fallback parameter
        source = PandasCalendarSource(fallback_source=mock_fallback, force_fallback=True)
        assert source._use_fallback is True
        assert source._fallback is mock_fallback

    def test_init_with_custom_cache_ttl(self) -> None:
        """
        Test initialization with custom cache TTL.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource(cache_ttl_hours=48)
            assert source._cache_ttl == timedelta(hours=48)

    def test_get_schedule_uses_fallback_when_disabled(self) -> None:
        """
        Test that get_schedule uses fallback when pandas_market_calendars is disabled.
        """
        # No patching needed - use force_fallback parameter
        source = PandasCalendarSource(force_fallback=True)
        dt = datetime(2024, 1, 15, 10, 30)  # Monday
        schedule = source.get_schedule(dt, "NYSE")

        # Should return SimpleCalendarSource results
        assert isinstance(schedule, MarketSchedule)
        assert schedule.is_trading_day is True
        assert schedule.is_holiday is False

    def test_exchange_mapping(self) -> None:
        """
        Test exchange code mapping.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()

            # US exchanges
            assert "NYSE" in source._exchange_mapping
            assert "XNYS" in source._exchange_mapping
            assert "NASDAQ" in source._exchange_mapping
            assert "XNAS" in source._exchange_mapping
            assert "CME" in source._exchange_mapping

            # European exchanges
            assert "LSE" in source._exchange_mapping
            assert "XLON" in source._exchange_mapping
            assert "EUREX" in source._exchange_mapping

            # Asian exchanges
            assert "JPX" in source._exchange_mapping
            assert "HKEX" in source._exchange_mapping
            assert "ASX" in source._exchange_mapping

            # Crypto exchanges
            assert "CRYPTO" in source._exchange_mapping
            assert "BINANCE" in source._exchange_mapping
            assert source._exchange_mapping["CRYPTO"] == "24/7"

    def test_get_24_7_schedule(self) -> None:
        """
        Test 24/7 market schedule for crypto exchanges.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()
            dt = datetime(2024, 1, 15, 10, 30)

            # Direct call to _get_24_7_schedule
            schedule = source._get_24_7_schedule(dt, "CRYPTO")

            assert schedule.is_trading_day is True
            assert schedule.is_holiday is False
            assert schedule.is_market_hours is True
            assert schedule.is_pre_market is False
            assert schedule.is_after_hours is False
            assert schedule.market_open == datetime(2024, 1, 15, 0, 0)
            assert schedule.market_close == datetime(2024, 1, 15, 23, 59, 59)

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_get_schedule_with_real_calendar(self) -> None:
        """
        Test get_schedule with real pandas_market_calendars.
        """
        source = PandasCalendarSource()

        # Test regular trading day
        dt = datetime(2024, 1, 16, 10, 30)  # Tuesday during market hours
        schedule = source.get_schedule(dt, "NYSE")

        assert isinstance(schedule, MarketSchedule)
        assert schedule.exchange == "NYSE"
        # Note: Actual values depend on real calendar data

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_cache_functionality(self) -> None:
        """
        Test that schedules are properly cached.
        """
        source = PandasCalendarSource()
        dt = datetime(2024, 1, 16, 10, 30)

        # First call should populate cache
        schedule1 = source.get_schedule(dt, "NYSE")
        cache_key = ("NYSE", dt.date().isoformat())
        assert cache_key in source._schedule_cache

        # Second call should use cache
        with patch.object(source, "_get_or_create_calendar") as mock_get_calendar:
            schedule2 = source.get_schedule(dt, "NYSE")
            # Should not call _get_or_create_calendar since it uses cache
            mock_get_calendar.assert_not_called()

        assert schedule1.is_trading_day == schedule2.is_trading_day
        assert schedule1.market_open == schedule2.market_open

    def test_clear_cache(self) -> None:
        """
        Test cache clearing functionality.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()

            # Add some dummy cache entries
            source._schedule_cache[("NYSE", "2024-01-16")] = MagicMock()
            source._cache_timestamps[("NYSE", "2024-01-16")] = datetime.now()

            assert len(source._schedule_cache) > 0
            assert len(source._cache_timestamps) > 0

            # Clear cache
            source.clear_cache()

            assert len(source._schedule_cache) == 0
            assert len(source._cache_timestamps) == 0

    def test_get_supported_exchanges(self) -> None:
        """
        Test getting list of supported exchanges.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()
            exchanges = source.get_supported_exchanges()

            assert isinstance(exchanges, list)
            assert "NYSE" in exchanges
            assert "NASDAQ" in exchanges
            assert "LSE" in exchanges
            assert "CRYPTO" in exchanges
            assert len(exchanges) > 20  # Should have many exchanges

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_get_holidays(self) -> None:
        """
        Test getting holidays for an exchange.
        """
        source = PandasCalendarSource()

        # Get holidays for NYSE in January 2024
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        holidays = source.get_holidays("NYSE", start_date, end_date)

        assert isinstance(holidays, list)
        # New Year's Day 2024 was on Monday, should be a holiday
        # MLK Day is typically third Monday of January

    def test_get_holidays_with_fallback(self) -> None:
        """
        Test getting holidays when using fallback source.
        """
        # No patching needed - use force_fallback parameter
        source = PandasCalendarSource(force_fallback=True)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        holidays = source.get_holidays("NYSE", start_date, end_date)

        # Fallback doesn't support holiday lists
        assert holidays == []

    def test_extended_hours_settings(self) -> None:
        """
        Test extended hours configuration.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()

            # Check NYSE extended hours
            nyse_extended = source._extended_hours["NYSE"]
            assert nyse_extended["pre_market_start"] == time(4, 0)
            assert nyse_extended["pre_market_end"] == time(9, 30)
            assert nyse_extended["after_hours_start"] == time(16, 0)
            assert nyse_extended["after_hours_end"] == time(20, 0)

            # Check NASDAQ has same settings
            nasdaq_extended = source._extended_hours["NASDAQ"]
            assert nasdaq_extended == nyse_extended

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_build_schedule_pre_market(self) -> None:
        """
        Test schedule building during pre-market hours.
        """
        source = PandasCalendarSource()

        # Pre-market time (5 AM on a weekday)
        dt = datetime(2024, 1, 16, 5, 0)  # Tuesday
        schedule = source.get_schedule(dt, "NYSE")

        if schedule.is_trading_day:  # Only check if it's a trading day
            assert schedule.is_pre_market or schedule.is_market_hours or schedule.is_after_hours

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_build_schedule_after_hours(self) -> None:
        """
        Test schedule building during after-hours trading.
        """
        source = PandasCalendarSource()

        # After-hours time (5 PM on a weekday)
        dt = datetime(2024, 1, 16, 17, 0)  # Tuesday
        schedule = source.get_schedule(dt, "NYSE")

        if schedule.is_trading_day:  # Only check if it's a trading day
            # Could be after-hours or market could be closed
            assert isinstance(schedule.is_after_hours, bool)

    def test_error_handling_in_get_schedule(self) -> None:
        """
        Test error handling in get_schedule method.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()

            # Mock _get_or_create_calendar to raise an exception
            with patch.object(
                source,
                "_get_or_create_calendar",
                side_effect=ValueError("Test error"),
            ):
                dt = datetime(2024, 1, 16, 10, 30)
                schedule = source.get_schedule(dt, "INVALID_EXCHANGE")

                # Should fall back to fallback source
                assert isinstance(schedule, MarketSchedule)
                # Fallback (SimpleCalendarSource) behavior
                assert schedule.is_trading_day is True  # Tuesday is a weekday

    def test_cache_expiration(self) -> None:
        """
        Test that cache respects TTL.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource(cache_ttl_hours=0)  # Immediate expiration

            dt = datetime(2024, 1, 16, 10, 30)
            cache_key = ("NYSE", dt.date().isoformat())

            # Add to cache with old timestamp
            source._schedule_cache[cache_key] = MagicMock()
            source._cache_timestamps[cache_key] = datetime.now() - timedelta(hours=1)

            with patch.object(
                source,
                "_get_or_create_calendar",
                return_value=MagicMock(),
            ) as mock_get:
                with patch.object(source, "_build_schedule", return_value=MagicMock()):
                    # Should not use cache due to expiration
                    source.get_schedule(dt, "NYSE")
                    mock_get.assert_called_once()


class TestMarketScheduleIntegration:
    """
    Integration tests for MarketSchedule with calendar sources.
    """

    def test_market_schedule_dataclass(self) -> None:
        """
        Test MarketSchedule dataclass properties.
        """
        dt = datetime(2024, 1, 16, 10, 30)
        schedule = MarketSchedule(
            date=dt,
            exchange="NYSE",
            is_trading_day=True,
            is_holiday=False,
            market_open=datetime(2024, 1, 16, 9, 30),
            market_close=datetime(2024, 1, 16, 16, 0),
            is_pre_market=False,
            is_after_hours=False,
            is_market_hours=True,
            minutes_to_close=330,
        )

        assert schedule.date == dt
        assert schedule.exchange == "NYSE"
        assert schedule.is_trading_day is True
        assert schedule.is_holiday is False
        assert schedule.is_market_hours is True
        assert schedule.minutes_to_close == 330

    def test_calendar_source_interface(self) -> None:
        """
        Test that PandasCalendarSource implements CalendarSource interface.
        """
        with patch("ml.data.sources.calendar.HAS_PANDAS_MARKET_CALENDARS", True):
            source = PandasCalendarSource()
            assert isinstance(source, CalendarSource)
            assert hasattr(source, "get_schedule")

            # Test that get_schedule returns MarketSchedule
            dt = datetime(2024, 1, 16, 10, 30)
            with patch.object(source, "_use_fallback", True):
                schedule = source.get_schedule(dt, "NYSE")
                assert isinstance(schedule, MarketSchedule)
