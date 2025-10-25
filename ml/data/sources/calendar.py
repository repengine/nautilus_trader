"""
Calendar sources for market schedules and trading hours.

This module provides various sources for market calendar data, including trading hours,
holidays, and market sessions.

"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from datetime import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from ml._imports import HAS_PANDAS_MARKET_CALENDARS
from ml._imports import mcal as mcal_runtime
from ml._imports import pd as pd_runtime


if TYPE_CHECKING:
    pass

# Local runtime aliases to avoid Optional[Module] union typing at use sites
PD: Any = cast(Any, pd_runtime)
MCAL: Any = cast(Any, mcal_runtime)


logger = logging.getLogger(__name__)


@dataclass
class MarketSchedule:
    """
    Market schedule for a specific date and exchange.

    Attributes
    ----------
    date : datetime
        The date of the schedule
    exchange : str
        Exchange identifier
    is_trading_day : bool
        Whether this is a trading day
    is_holiday : bool
        Whether this is a market holiday
    market_open : datetime
        Market open time
    market_close : datetime
        Market close time
    is_pre_market : bool
        Whether current time is pre-market
    is_after_hours : bool
        Whether current time is after-hours
    is_market_hours : bool
        Whether current time is regular market hours
    minutes_to_close : int
        Minutes until market close (0 if closed)

    """

    date: datetime
    exchange: str
    is_trading_day: bool
    is_holiday: bool
    market_open: datetime
    market_close: datetime
    is_pre_market: bool
    is_after_hours: bool
    is_market_hours: bool
    minutes_to_close: int


class CalendarSource(ABC):
    """
    Abstract base class for calendar sources.
    """

    @abstractmethod
    def get_schedule(self, dt: datetime, exchange: str) -> MarketSchedule:
        """
        Get market schedule for a specific datetime and exchange.

        Parameters
        ----------
        dt : datetime
            The datetime to get schedule for
        exchange : str
            Exchange identifier (e.g., 'NYSE', 'NASDAQ')

        Returns
        -------
        MarketSchedule
            Market schedule information

        """
        ...


class MockCalendarSource(CalendarSource):
    """
    Mock calendar source for testing.

    Provides simplified but realistic market schedules.

    """

    def __init__(self) -> None:
        """
        Initialize mock calendar source.
        """
        # US market holidays (simplified)
        self.holidays = {
            (1, 1),  # New Year's Day
            (7, 4),  # Independence Day
            (12, 25),  # Christmas
        }

        # Market hours by exchange
        self.market_hours = {
            "NYSE": (time(9, 30), time(16, 0)),
            "NASDAQ": (time(9, 30), time(16, 0)),
            "CME": (time(8, 30), time(15, 15)),
            "DEFAULT": (time(9, 30), time(16, 0)),
        }

    def get_schedule(self, dt: datetime, exchange: str) -> MarketSchedule:
        """
        Get mock market schedule.

        Parameters
        ----------
        dt : datetime
            The datetime to get schedule for
        exchange : str
            Exchange identifier

        Returns
        -------
        MarketSchedule
            Mock market schedule

        """
        # Get market hours for exchange
        open_time, close_time = self.market_hours.get(
            exchange,
            self.market_hours["DEFAULT"],
        )

        # Create market open/close datetimes
        market_open = datetime.combine(dt.date(), open_time)
        market_close = datetime.combine(dt.date(), close_time)

        # Check if weekend
        is_weekend = dt.weekday() >= 5  # Saturday = 5, Sunday = 6

        # Check if holiday
        is_holiday = (dt.month, dt.day) in self.holidays

        # Trading day if not weekend and not holiday
        is_trading_day = not is_weekend and not is_holiday

        # Pre-market: 4 AM - 9:30 AM
        is_pre_market = is_trading_day and dt.time() >= time(4, 0) and dt.time() < open_time

        # Market hours: 9:30 AM - 4 PM
        is_market_hours = is_trading_day and dt.time() >= open_time and dt.time() < close_time

        # After hours: 4 PM - 8 PM
        is_after_hours = is_trading_day and dt.time() >= close_time and dt.time() < time(20, 0)

        # Minutes to close
        minutes_to_close = 0
        if is_market_hours:
            time_to_close = market_close - dt
            minutes_to_close = int(time_to_close.total_seconds() / 60)
            minutes_to_close = max(0, minutes_to_close)

        return MarketSchedule(
            date=dt,
            exchange=exchange,
            is_trading_day=is_trading_day,
            is_holiday=is_holiday,
            market_open=market_open,
            market_close=market_close,
            is_pre_market=is_pre_market,
            is_after_hours=is_after_hours,
            is_market_hours=is_market_hours,
            minutes_to_close=minutes_to_close,
        )


class SimpleCalendarSource(CalendarSource):
    """
    Simple calendar source with basic NYSE schedule.

    Uses fixed market hours and simple weekend detection.
    No holiday calendar - all weekdays are trading days.

    """

    def __init__(self) -> None:
        """
        Initialize simple calendar source.
        """
        self.default_open = time(9, 30)
        self.default_close = time(16, 0)

    def get_schedule(self, dt: datetime, exchange: str) -> MarketSchedule:
        """
        Get simple market schedule.

        Parameters
        ----------
        dt : datetime
            The datetime to get schedule for
        exchange : str
            Exchange identifier

        Returns
        -------
        MarketSchedule
            Simple market schedule

        """
        # All exchanges use same hours in simple implementation
        market_open = datetime.combine(dt.date(), self.default_open)
        market_close = datetime.combine(dt.date(), self.default_close)

        # Trading day if weekday
        is_trading_day = dt.weekday() < 5

        # No holidays in simple implementation
        is_holiday = False

        # Check market session
        is_pre_market = False
        is_after_hours = False
        is_market_hours = False
        minutes_to_close = 0

        if is_trading_day:
            current_time = dt.time()

            # Pre-market: 4 AM - 9:30 AM
            is_pre_market = current_time >= time(4, 0) and current_time < self.default_open

            # Market hours: 9:30 AM - 4 PM
            is_market_hours = (
                current_time >= self.default_open and current_time < self.default_close
            )

            # After hours: 4 PM - 8 PM
            is_after_hours = current_time >= self.default_close and current_time < time(20, 0)

            # Minutes to close
            if is_market_hours:
                time_to_close = market_close - dt
                minutes_to_close = int(time_to_close.total_seconds() / 60)
                minutes_to_close = max(0, minutes_to_close)

        return MarketSchedule(
            date=dt,
            exchange=exchange,
            is_trading_day=is_trading_day,
            is_holiday=is_holiday,
            market_open=market_open,
            market_close=market_close,
            is_pre_market=is_pre_market,
            is_after_hours=is_after_hours,
            is_market_hours=is_market_hours,
            minutes_to_close=minutes_to_close,
        )


class PandasCalendarSource(CalendarSource):
    """
    Real market calendar source using pandas_market_calendars.

    Provides accurate market schedules including trading hours,
    holidays, early closes, and special trading sessions for
    major exchanges worldwide.

    Attributes
    ----------
    _calendars : dict[str, mcal.MarketCalendar]
        Cache of loaded market calendars by exchange
    _schedule_cache : dict[tuple[str, datetime, datetime], pd.DataFrame]
        Cache of market schedules with (exchange, start, end) keys
    _exchange_mapping : dict[str, str]
        Mapping from exchange codes to calendar names
    _cache_ttl : timedelta
        Time-to-live for cached schedules
    _last_cache_clean : datetime
        Last time the cache was cleaned

    Examples
    --------
    >>> source = PandasCalendarSource()
    >>> schedule = source.get_schedule(datetime.now(), "NYSE")
    >>> print(f"Is trading day: {schedule.is_trading_day}")
    >>> print(f"Minutes to close: {schedule.minutes_to_close}")

    """

    def __init__(
        self,
        cache_ttl_hours: int = 24,
        fallback_source: CalendarSource | None = None,
        force_fallback: bool = False,
    ) -> None:
        """
        Initialize pandas calendar source.

        Parameters
        ----------
        cache_ttl_hours : int, default 24
            Hours to cache market schedules
        fallback_source : CalendarSource, optional
            Fallback source when pandas_market_calendars is unavailable
            or when exchange is not supported. Uses SimpleCalendarSource
            if not provided.
        force_fallback : bool, default False
            If True, force use of fallback regardless of pandas_market_calendars availability.
            Useful for testing and configuration that wants pure fallback mode.

        Raises
        ------
        ImportError
            If pandas_market_calendars is not installed and no fallback provided

        """
        # Declare the fallback attribute with type
        self._fallback: CalendarSource
        self._use_fallback: bool

        if force_fallback or not HAS_PANDAS_MARKET_CALENDARS:
            if fallback_source is None:
                msg = (
                    "pandas_market_calendars is required for PandasCalendarSource. "
                    "Install with: pip install pandas_market_calendars"
                )
                if not force_fallback:
                    logger.warning(msg)
                # Use SimpleCalendarSource as automatic fallback
                self._fallback = SimpleCalendarSource()
                self._use_fallback = True
            else:
                self._fallback = fallback_source
                self._use_fallback = True

            reason = "forced by configuration" if force_fallback else "missing pandas_market_calendars"
            logger.info(f"Using fallback calendar source due to {reason}")
        else:
            self._fallback = fallback_source or SimpleCalendarSource()
            self._use_fallback = False

        # Calendar cache
        self._calendars: dict[str, Any] = {}  # mcal.MarketCalendar instances

        # Schedule cache: (exchange, date) -> schedule DataFrame
        self._schedule_cache: dict[tuple[str, str], Any] = {}  # pd.DataFrame instances

        # Cache settings
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache_timestamps: dict[tuple[str, str], datetime] = {}

        # Exchange name mapping (common exchange codes to calendar names)
        self._exchange_mapping = {
            # US Exchanges
            "NYSE": "NYSE",
            "XNYS": "NYSE",
            "NASDAQ": "NASDAQ",
            "XNAS": "NASDAQ",
            "CME": "CME",
            "XCME": "CME",
            "CBOT": "CBOT",
            "XCBT": "CBOT",
            "ICE": "ICE",
            "XICE": "ICE",
            "CBOE": "CBOE",
            # European Exchanges
            "LSE": "LSE",
            "XLON": "LSE",
            "EUREX": "EUREX",
            "XEUR": "EUREX",
            "XETR": "XETR",  # Xetra
            # Asian Exchanges
            "JPX": "JPX",
            "XJPX": "JPX",
            "TSE": "JPX",  # Tokyo Stock Exchange (part of JPX)
            "HKEX": "HKEX",
            "XHKG": "HKEX",
            "SSE": "SSE",
            "XSHG": "SSE",
            "ASX": "ASX",
            "XASX": "ASX",
            # Crypto (24/7)
            "CRYPTO": "24/7",
            "BINANCE": "24/7",
            "COINBASE": "24/7",
        }

        # Pre-market and after-hours settings by exchange
        self._extended_hours = {
            "NYSE": {
                "pre_market_start": time(4, 0),
                "pre_market_end": time(9, 30),
                "after_hours_start": time(16, 0),
                "after_hours_end": time(20, 0),
            },
            "NASDAQ": {
                "pre_market_start": time(4, 0),
                "pre_market_end": time(9, 30),
                "after_hours_start": time(16, 0),
                "after_hours_end": time(20, 0),
            },
            "DEFAULT": {
                "pre_market_start": time(4, 0),
                "pre_market_end": time(9, 30),
                "after_hours_start": time(16, 0),
                "after_hours_end": time(20, 0),
            },
        }

        logger.info(
            f"Initialized PandasCalendarSource with cache_ttl={cache_ttl_hours}h, "
            f"fallback_enabled={self._use_fallback}",
        )

    def get_schedule(self, dt: datetime, exchange: str) -> MarketSchedule:
        """
        Get market schedule for a specific datetime and exchange.

        Parameters
        ----------
        dt : datetime
            The datetime to get schedule for
        exchange : str
            Exchange identifier (e.g., 'NYSE', 'NASDAQ', 'XNAS')

        Returns
        -------
        MarketSchedule
            Market schedule information with accurate hours and holidays

        """
        # Use fallback if necessary
        if self._use_fallback:
            return self._fallback.get_schedule(dt, exchange)

        try:
            # Get calendar name from exchange code
            calendar_name = self._exchange_mapping.get(exchange, exchange)

            # Check cache first
            cache_key = (calendar_name, dt.date().isoformat())
            if cache_key in self._schedule_cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time) < self._cache_ttl:
                    return self._build_schedule_from_cache(dt, exchange, cache_key)

            # Get or create calendar
            calendar = self._get_or_create_calendar(calendar_name)

            # For 24/7 markets (crypto)
            if calendar_name == "24/7":
                return self._get_24_7_schedule(dt, exchange)

            # Get schedule for the date (fetch a week at a time for efficiency)
            start_date = dt.date() - timedelta(days=3)
            end_date = dt.date() + timedelta(days=3)

            # Get the market schedule
            schedule = calendar.schedule(
                start_date=PD.Timestamp(start_date),
                end_date=PD.Timestamp(end_date),
            )

            # Cache the schedule
            self._schedule_cache[cache_key] = schedule
            self._cache_timestamps[cache_key] = datetime.now()

            # Build MarketSchedule from pandas schedule
            return self._build_schedule(dt, exchange, schedule)

        except Exception as e:
            logger.warning(
                f"Failed to get schedule from pandas_market_calendars for {exchange}: {e}, "
                f"using fallback",
            )
            return self._fallback.get_schedule(dt, exchange)

    def _get_or_create_calendar(self, calendar_name: str) -> Any:
        """
        Get or create a market calendar instance.

        Parameters
        ----------
        calendar_name : str
            Name of the calendar

        Returns
        -------
        mcal.MarketCalendar
            Market calendar instance

        Raises
        ------
        ValueError
            If calendar is not supported

        """
        if calendar_name not in self._calendars:
            try:
                if calendar_name == "24/7":
                    # Special handling for 24/7 markets
                    return None

                # Get calendar from pandas_market_calendars
                self._calendars[calendar_name] = MCAL.get_calendar(calendar_name)
                logger.debug(f"Loaded calendar: {calendar_name}")
            except Exception as e:
                logger.error(f"Failed to load calendar {calendar_name}: {e}")
                # Try common alternatives
                alternatives = {
                    "XNYS": "NYSE",
                    "XNAS": "NASDAQ",
                    "XLON": "LSE",
                }
                alt_name = alternatives.get(calendar_name)
                if alt_name:
                    logger.info(f"Trying alternative calendar name: {alt_name}")
                    self._calendars[calendar_name] = MCAL.get_calendar(alt_name)
                else:
                    raise ValueError(f"Unsupported calendar: {calendar_name}") from e

        return self._calendars[calendar_name]

    def _build_schedule(
        self,
        dt: datetime,
        exchange: str,
        schedule: Any,  # pd.DataFrame
    ) -> MarketSchedule:
        """
        Build MarketSchedule from pandas schedule DataFrame.

        Parameters
        ----------
        dt : datetime
            The datetime to build schedule for
        exchange : str
            Exchange identifier
        schedule : pd.DataFrame
            Pandas schedule DataFrame from market calendar

        Returns
        -------
        MarketSchedule
            Built market schedule

        """
        # Check if the date is in the schedule (trading day)
        dt_date = PD.Timestamp(dt.date())
        is_trading_day = False
        is_holiday = True
        market_open = datetime.combine(dt.date(), time(9, 30))
        market_close = datetime.combine(dt.date(), time(16, 0))

        if not schedule.empty and dt_date in schedule.index:
            is_trading_day = True
            is_holiday = False

            # Get market hours from schedule
            row = schedule.loc[dt_date]
            market_open = row["market_open"].to_pydatetime()
            market_close = row["market_close"].to_pydatetime()

        # Get extended hours settings
        extended = self._extended_hours.get(
            self._exchange_mapping.get(exchange, exchange),
            self._extended_hours["DEFAULT"],
        )

        # Check market session
        is_pre_market = False
        is_after_hours = False
        is_market_hours = False
        minutes_to_close = 0

        if is_trading_day:
            current_time = dt.time()

            # Pre-market
            is_pre_market = (
                current_time >= extended["pre_market_start"] and current_time < market_open.time()
            )

            # Regular market hours
            is_market_hours = dt >= market_open and dt < market_close

            # After-hours
            is_after_hours = (
                current_time >= market_close.time() and current_time < extended["after_hours_end"]
            )

            # Minutes to close
            if is_market_hours:
                time_to_close = market_close - dt
                minutes_to_close = int(time_to_close.total_seconds() / 60)
                minutes_to_close = max(0, minutes_to_close)

        return MarketSchedule(
            date=dt,
            exchange=exchange,
            is_trading_day=is_trading_day,
            is_holiday=is_holiday,
            market_open=market_open,
            market_close=market_close,
            is_pre_market=is_pre_market,
            is_after_hours=is_after_hours,
            is_market_hours=is_market_hours,
            minutes_to_close=minutes_to_close,
        )

    def _build_schedule_from_cache(
        self,
        dt: datetime,
        exchange: str,
        cache_key: tuple[str, str],
    ) -> MarketSchedule:
        """
        Build MarketSchedule from cached data.

        Parameters
        ----------
        dt : datetime
            The datetime to build schedule for
        exchange : str
            Exchange identifier
        cache_key : tuple[str, str]
            Cache key for the schedule

        Returns
        -------
        MarketSchedule
            Built market schedule from cache

        """
        schedule = self._schedule_cache[cache_key]
        return self._build_schedule(dt, exchange, schedule)

    def _get_24_7_schedule(self, dt: datetime, exchange: str) -> MarketSchedule:
        """
        Get schedule for 24/7 markets (crypto).

        Parameters
        ----------
        dt : datetime
            The datetime to get schedule for
        exchange : str
            Exchange identifier

        Returns
        -------
        MarketSchedule
            24/7 market schedule

        """
        # For 24/7 markets, always trading
        market_open = datetime.combine(dt.date(), time(0, 0))
        market_close = datetime.combine(dt.date(), time(23, 59, 59))

        # Always in market hours for 24/7
        minutes_to_close = int((market_close - dt).total_seconds() / 60)
        minutes_to_close = max(0, minutes_to_close)

        return MarketSchedule(
            date=dt,
            exchange=exchange,
            is_trading_day=True,
            is_holiday=False,
            market_open=market_open,
            market_close=market_close,
            is_pre_market=False,
            is_after_hours=False,
            is_market_hours=True,
            minutes_to_close=minutes_to_close,
        )

    def clear_cache(self) -> None:
        """
        Clear the schedule cache.

        Useful for long-running processes to prevent memory growth.

        """
        self._schedule_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cleared PandasCalendarSource cache")

    def get_supported_exchanges(self) -> list[str]:
        """
        Get list of supported exchange codes.

        Returns
        -------
        list[str]
            List of supported exchange codes

        """
        return list(self._exchange_mapping.keys())

    def get_holidays(
        self,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[datetime]:
        """
        Get list of holidays for an exchange in a date range.

        Parameters
        ----------
        exchange : str
            Exchange identifier
        start_date : datetime
            Start of date range
        end_date : datetime
            End of date range

        Returns
        -------
        list[datetime]
            List of holiday dates

        """
        if self._use_fallback:
            # Fallback doesn't support holiday lists
            return []

        try:
            calendar_name = self._exchange_mapping.get(exchange, exchange)
            calendar = self._get_or_create_calendar(calendar_name)

            if calendar_name == "24/7":
                return []  # No holidays for 24/7 markets

            # Get valid days (trading days)
            valid_days = calendar.valid_days(
                start_date=PD.Timestamp(start_date),
                end_date=PD.Timestamp(end_date),
            )

            # Generate all business days in range
            all_business_days = PD.bdate_range(
                start=start_date,
                end=end_date,
            )

            # Holidays are business days that are not valid trading days
            holidays = []
            for day in all_business_days:
                if day not in valid_days:
                    holidays.append(day.to_pydatetime())

            return holidays

        except Exception as e:
            logger.warning(f"Failed to get holidays: {e}")
            return []
