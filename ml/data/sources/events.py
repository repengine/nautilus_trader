"""
Event sources for economic and earnings calendars.

This module provides various sources for scheduled market events, including economic
releases, earnings announcements, and Fed meetings.

"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any

from numpy.random import default_rng


logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """
    Economic event data.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        Event release time
    name : str
        Event name (e.g., "Federal Funds Rate Decision")
    country : str
        Country code (e.g., "US", "EU")
    importance : str
        Importance level: "HIGH", "MEDIUM", "LOW"
    forecast : float, optional
        Consensus forecast value
    previous : float, optional
        Previous release value
    actual : float, optional
        Actual released value (None before release)

    """

    event_id: str
    timestamp: datetime
    name: str
    country: str
    importance: str
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None


@dataclass
class EarningsEvent:
    """
    Earnings announcement event.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        Earnings release time
    instrument_id : str
        Instrument/ticker symbol
    fiscal_quarter : str
        Fiscal quarter (e.g., "Q1", "Q2")
    fiscal_year : int
        Fiscal year
    eps_forecast : float, optional
        Consensus EPS forecast
    eps_previous : float, optional
        Previous quarter EPS
    revenue_forecast : float, optional
        Consensus revenue forecast
    revenue_previous : float, optional
        Previous quarter revenue
    eps_actual : float, optional
        Actual EPS (None before release)
    revenue_actual : float, optional
        Actual revenue (None before release)
    timing : str
        "BMO" (Before Market Open) or "AMC" (After Market Close)

    """

    event_id: str
    timestamp: datetime
    instrument_id: str
    fiscal_quarter: str
    fiscal_year: int
    eps_forecast: float | None = None
    eps_previous: float | None = None
    revenue_forecast: float | None = None
    revenue_previous: float | None = None
    eps_actual: float | None = None
    revenue_actual: float | None = None
    timing: str = "AMC"


class EventSource(ABC):
    """
    Abstract base class for event sources.
    """

    def __init__(self) -> None:
        """
        Initialize common event source state.

        This base provides a lightweight, non-blocking publish/subscribe API used by
        unit tests. It avoids any I/O and keeps latency low to satisfy hot-path
        constraints.

        """
        # Subscriber registry (id -> handler)
        self._subscribers: dict[int, Callable[[Any], None]] = {}
        self._next_subscriber_id: int = 1
        # Monotonic watermark in nanoseconds (or None if unset)
        self._watermark: int | None = None
        # Simple counter for emitted events (used in tests)
        self._event_count: int = 0

    # ---------------------------------------------------------------------
    # Lightweight streaming interface used in tests
    # ---------------------------------------------------------------------
    def subscribe(self, handler: Callable[[Any], None]) -> int:
        """
        Subscribe a synchronous handler to receive emitted events.

        Returns a subscription id which can be used to unsubscribe.

        """
        sub_id = self._next_subscriber_id
        self._next_subscriber_id += 1
        self._subscribers[sub_id] = handler
        return sub_id

    def unsubscribe(self, subscription_id: int) -> None:
        """
        Remove a previously registered subscriber.
        """
        self._subscribers.pop(subscription_id, None)

    def emit_event(self, event: Any) -> None:
        """
        Emit an event to all current subscribers.

        Exceptions thrown by individual handlers are isolated and do not affect delivery
        to other handlers.

        """
        self._event_count += 1
        # Iterate over a snapshot to allow unsubscribe during iteration.
        for handler in list(self._subscribers.values()):
            try:
                handler(event)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Subscriber handler raised; continuing", exc_info=True)

    def get_watermark(self) -> int | None:
        """
        Get the current watermark in nanoseconds, if set.
        """
        return self._watermark

    def update_watermark(self, ts: Any) -> int:
        """
        Update the watermark monotonically and return its value in nanoseconds.

        Accepts a `datetime` or an integer nanosecond timestamp.

        """
        if isinstance(ts, datetime):
            # Convert to POSIX nanoseconds from UTC-naive timestamp
            ts_ns = int(ts.timestamp() * 1e9)
        else:
            ts_ns = int(ts)

        if self._watermark is None:
            self._watermark = ts_ns
        else:
            # Ensure monotonic progression
            self._watermark = max(self._watermark, ts_ns)
        return self._watermark

    @abstractmethod
    def get_economic_events(
        self,
        start: datetime,
        end: datetime,
    ) -> list[EconomicEvent]:
        """
        Get economic events in date range.

        Parameters
        ----------
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EconomicEvent]
            List of economic events

        """
        ...

    @abstractmethod
    def get_earnings_events(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> list[EarningsEvent]:
        """
        Get earnings events for instruments in date range.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EarningsEvent]
            List of earnings events

        """
        ...


class MockEventSource(EventSource):
    """
    Mock event source for testing.

    Generates realistic but synthetic event schedules.

    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize mock event source.

        Parameters
        ----------
        seed : int, default 42
            Random seed for reproducibility

        """
        super().__init__()
        self.seed = seed
        self._rng = default_rng(seed)
        # Lightweight pub/sub state for unit tests (non-async, hot-path safe)
        self._subscribers: dict[int, Callable[[object], None]] = {}
        self._next_sub_id: int = 1
        # Watermark is stored as nanoseconds from epoch (int) or None if uninitialized
        self._watermark: int | None = None
        # Event counter for recovery tests
        self._event_count: int = 0

    # ------------------------------------------------------------------
    # Minimal pub/sub API expected by tests
    # ------------------------------------------------------------------
    def subscribe(self, handler: Callable[[object], None]) -> int:
        """
        Subscribe a synchronous event handler.

        Parameters
        ----------
        handler : Callable[[object], None]
            A function taking a single event argument.

        Returns
        -------
        int
            Subscription id which can be passed to `unsubscribe`.

        """
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self._subscribers[sub_id] = handler
        return sub_id

    def unsubscribe(self, subscription_id: int) -> None:
        """
        Unsubscribe a previously registered handler.

        Best-effort; missing ids are ignored.

        """
        try:
            self._subscribers.pop(subscription_id, None)
        except Exception:
            # Defensive: ignore unexpected errors in tests
            self._subscribers.pop(subscription_id, None)

    def emit_event(self, event: object) -> None:
        """
        Emit a single event to all subscribers.

        - Calls handlers synchronously in registration order.
        - Isolates handler exceptions so one faulty handler doesn't block others.
        - Advances the internal watermark using the event timestamp when available.

        """
        # Update watermark if event carries a datetime timestamp attribute
        try:
            ts = getattr(event, "timestamp", None)
            if isinstance(ts, datetime):
                self.update_watermark(ts)
        except Exception:
            # Ignore malformed events
            pass

        # Dispatch to subscribers, protecting isolation
        for _, handler in list(self._subscribers.items()):
            try:
                handler(event)
            except Exception:
                # Tests expect faulty handlers not to affect others
                logger.debug("Event handler raised; continuing", exc_info=True)

        # Increment simple event counter for recovery tests
        try:
            self._event_count += 1
        except Exception:
            self._event_count = getattr(self, "_event_count", 0) + 1

    # ------------------------------------------------------------------
    # Watermark helpers expected by tests
    # ------------------------------------------------------------------
    def get_watermark(self) -> int | None:
        """
        Return current watermark in nanoseconds, or None if uninitialized.
        """
        return self._watermark

    def update_watermark(self, ts: datetime) -> int:
        """
        Update watermark with a datetime, keeping it monotonic.

        Parameters
        ----------
        ts : datetime
            New candidate timestamp.

        Returns
        -------
        int
            The resulting watermark value in nanoseconds.

        """
        try:
            new_wm = int(ts.timestamp() * 1e9)
        except Exception:
            # Fallback: leave watermark unchanged on conversion failure
            return self._watermark if self._watermark is not None else 0

        if self._watermark is None or new_wm > self._watermark:
            self._watermark = new_wm
        return int(self._watermark)

    def get_economic_events(
        self,
        start: datetime,
        end: datetime,
    ) -> list[EconomicEvent]:
        """
        Generate mock economic events.

        Parameters
        ----------
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EconomicEvent]
            Mock economic events

        """
        events = []

        # Fed meetings (roughly every 6 weeks)
        current = start
        while current <= end:
            # Fed meetings typically on Wednesday
            days_to_wed = (2 - current.weekday()) % 7
            if days_to_wed == 0:
                days_to_wed = 7
            fed_date = current + timedelta(days=days_to_wed)
            fed_dt = fed_date.replace(hour=14, minute=0)

            if start <= fed_dt <= end:
                events.append(
                    EconomicEvent(
                        event_id=f"FED_{fed_date.strftime('%Y%m%d')}",
                        timestamp=fed_dt,
                        name="Federal Funds Rate Decision",
                        country="US",
                        importance="HIGH",
                        forecast=5.25 + float(self._rng.uniform(-0.5, 0.5)),
                        previous=5.25,
                        actual=None,
                    ),
                )

            # Move to next potential meeting (6 weeks later)
            current += timedelta(weeks=6)

        # CPI releases (monthly, typically around 10th-15th)
        current = start.replace(day=1)
        while current <= end:
            cpi_day = int(self._rng.integers(10, 16))
            cpi_date = current.replace(day=cpi_day, hour=8, minute=30)

            if start <= cpi_date <= end:
                events.append(
                    EconomicEvent(
                        event_id=f"CPI_{cpi_date.strftime('%Y%m')}",
                        timestamp=cpi_date,
                        name="Consumer Price Index",
                        country="US",
                        importance="HIGH",
                        forecast=3.0 + float(self._rng.uniform(-1.0, 2.0)),
                        previous=3.0,
                        actual=None,
                    ),
                )

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # NFP (Non-Farm Payrolls) - first Friday of month
        current = start.replace(day=1)
        while current <= end:
            # Find first Friday
            first_friday = current
            while first_friday.weekday() != 4:  # Friday is 4
                first_friday += timedelta(days=1)

            nfp_date = first_friday.replace(hour=8, minute=30)

            if start <= nfp_date <= end:
                events.append(
                    EconomicEvent(
                        event_id=f"NFP_{nfp_date.strftime('%Y%m')}",
                        timestamp=nfp_date,
                        name="Non-Farm Payrolls",
                        country="US",
                        importance="HIGH",
                        forecast=200000 + float(self._rng.uniform(-50000, 50000)),
                        previous=200000,
                        actual=None,
                    ),
                )

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Add some medium importance events
        for _ in range(int(self._rng.integers(5, 10))):
            days_offset = int(self._rng.integers(0, (end - start).days + 1))
            base_date = start + timedelta(days=days_offset)
            event_dt = base_date.replace(hour=10, minute=0)

            if start <= event_dt <= end:
                events.append(
                    EconomicEvent(
                        event_id=f"OTHER_{base_date.strftime('%Y%m%d')}_{int(self._rng.integers(1, 100))}",
                        timestamp=event_dt,
                        name=str(
                            self._rng.choice(
                                [
                                    "Retail Sales",
                                    "Industrial Production",
                                    "Housing Starts",
                                    "Consumer Confidence",
                                ],
                            ),
                        ),
                        country="US",
                        importance="MEDIUM",
                        forecast=float(self._rng.uniform(-2.0, 5.0)),
                        previous=float(self._rng.uniform(-2.0, 5.0)),
                        actual=None,
                    ),
                )

        return sorted(events, key=lambda e: e.timestamp)

    def get_earnings_events(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> list[EarningsEvent]:
        """
        Generate mock earnings events.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EarningsEvent]
            Mock earnings events

        """
        events = []

        for instrument in instruments:
            # Check multiple fiscal years to ensure coverage
            years_to_check = [start.year - 1, start.year, end.year, end.year + 1]

            for fiscal_year in set(years_to_check):
                # Generate quarterly earnings
                # Q4 of previous year reports in Jan
                # Q1: Jan-Mar reports in Apr
                # Q2: Apr-Jun reports in Jul
                # Q3: Jul-Sep reports in Oct

                quarters = [
                    ("Q4", 1, fiscal_year, fiscal_year - 1),  # Q4 of prev year reports in Jan
                    ("Q1", 4, fiscal_year, fiscal_year),  # April
                    ("Q2", 7, fiscal_year, fiscal_year),  # July
                    ("Q3", 10, fiscal_year, fiscal_year),  # October
                ]

                for quarter, month, report_year, actual_fiscal_year in quarters:
                    # Random day in the month (typically mid-month)
                    day = int(self._rng.integers(10, 26))

                    # Random timing (before or after market)
                    timing = str(self._rng.choice(["BMO", "AMC"]))
                    hour = 7 if timing == "BMO" else 16
                    minute = 30

                    try:
                        earnings_date = datetime(report_year, month, day, hour, minute)
                    except ValueError:
                        continue  # Skip invalid dates

                    if start <= earnings_date <= end:
                        # Generate realistic EPS values
                        eps_base = float(self._rng.uniform(0.5, 5.0))

                        events.append(
                            EarningsEvent(
                                event_id=f"{instrument}_{quarter}_{actual_fiscal_year}",
                                timestamp=earnings_date,
                                instrument_id=instrument,
                                fiscal_quarter=quarter,
                                fiscal_year=actual_fiscal_year,
                                eps_forecast=eps_base + float(self._rng.uniform(-0.2, 0.2)),
                                eps_previous=eps_base + float(self._rng.uniform(-0.3, 0.1)),
                                revenue_forecast=float(self._rng.uniform(10e9, 100e9)),
                                revenue_previous=float(self._rng.uniform(10e9, 100e9)),
                                eps_actual=None,
                                revenue_actual=None,
                                timing=timing,
                            ),
                        )

        return sorted(events, key=lambda e: e.timestamp)


class SimpleEventSource(EventSource):
    """
    Simple event source with fixed calendar.

    Provides basic Fed meetings and quarterly earnings.

    """

    def __init__(self) -> None:
        """
        Initialize simple event source.
        """
        # Fixed Fed meeting dates for 2024
        self.fed_dates_2024 = [
            datetime(2024, 1, 31, 14, 0),
            datetime(2024, 3, 20, 14, 0),
            datetime(2024, 5, 1, 14, 0),
            datetime(2024, 6, 12, 14, 0),
            datetime(2024, 7, 31, 14, 0),
            datetime(2024, 9, 18, 14, 0),
            datetime(2024, 11, 7, 14, 0),
            datetime(2024, 12, 18, 14, 0),
        ]

    def get_economic_events(
        self,
        start: datetime,
        end: datetime,
    ) -> list[EconomicEvent]:
        """
        Get simple economic events.

        Parameters
        ----------
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EconomicEvent]
            Economic events (mainly Fed meetings)

        """
        events = []

        # Add Fed meetings
        for fed_date in self.fed_dates_2024:
            if start <= fed_date <= end:
                events.append(
                    EconomicEvent(
                        event_id=f"FED_{fed_date.strftime('%Y%m%d')}",
                        timestamp=fed_date,
                        name="Federal Funds Rate Decision",
                        country="US",
                        importance="HIGH",
                        forecast=5.25,
                        previous=5.25,
                        actual=None,
                    ),
                )

        # Add monthly CPI (second Tuesday)
        current = start.replace(day=1)
        while current <= end:
            # Find second Tuesday
            tuesday_count = 0
            for day in range(1, 32):
                try:
                    date = current.replace(day=day)
                    if date.weekday() == 1:  # Tuesday
                        tuesday_count += 1
                        if tuesday_count == 2:
                            cpi_date = date.replace(hour=8, minute=30)
                            if start <= cpi_date <= end:
                                events.append(
                                    EconomicEvent(
                                        event_id=f"CPI_{cpi_date.strftime('%Y%m')}",
                                        timestamp=cpi_date,
                                        name="Consumer Price Index",
                                        country="US",
                                        importance="HIGH",
                                        forecast=3.2,
                                        previous=3.1,
                                        actual=None,
                                    ),
                                )
                            break
                except ValueError:
                    break  # Invalid day for this month

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return sorted(events, key=lambda e: e.timestamp)

    def get_earnings_events(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> list[EarningsEvent]:
        """
        Get simple earnings events (quarterly).

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers
        start : datetime
            Start of date range
        end : datetime
            End of date range

        Returns
        -------
        list[EarningsEvent]
            Quarterly earnings events

        """
        events = []

        # Simple quarterly schedule
        quarters = [
            ("Q1", 4, 20),  # April 20
            ("Q2", 7, 20),  # July 20
            ("Q3", 10, 20),  # October 20
            ("Q4", 1, 20),  # January 20 (next year)
        ]

        for instrument in instruments:
            # Need to check multiple years to catch all quarters
            # Q4 of year Y is reported in Jan of year Y+1
            years_to_check = [start.year - 1, start.year, end.year]

            for year in years_to_check:
                for quarter, month, day in quarters:
                    # Adjust year for Q4 (reported in next year)
                    if quarter == "Q4":
                        report_year = year + 1
                        fiscal_year = year
                    else:
                        report_year = year
                        fiscal_year = year

                    try:
                        earnings_date = datetime(report_year, month, day, 16, 30)  # AMC
                    except ValueError:
                        continue  # Skip invalid dates

                    if start <= earnings_date <= end:
                        events.append(
                            EarningsEvent(
                                event_id=f"{instrument}_{quarter}_{fiscal_year}",
                                timestamp=earnings_date,
                                instrument_id=instrument,
                                fiscal_quarter=quarter,
                                fiscal_year=fiscal_year,
                                eps_forecast=2.50,
                                eps_previous=2.30,
                                revenue_forecast=50e9,
                                revenue_previous=48e9,
                                eps_actual=None,
                                revenue_actual=None,
                                timing="AMC",
                            ),
                        )

        # Remove duplicates by event_id
        seen = set()
        unique_events = []
        for event in sorted(events, key=lambda e: e.timestamp):
            if event.event_id not in seen:
                seen.add(event.event_id)
                unique_events.append(event)

        return unique_events
