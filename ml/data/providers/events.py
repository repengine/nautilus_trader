"""
Event schedule provider for economic and earnings calendars.

This module provides scheduled event features including economic releases, earnings
announcements, and their temporal relationships.

"""

from __future__ import annotations

import logging
from bisect import bisect_left
from bisect import bisect_right
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from typing import Any as _Any
from typing import cast as _cast

from ml._imports import check_ml_dependencies
from ml._imports import pl as pl_runtime
from ml.data.providers.base import BaseTimeSeriesProvider


if TYPE_CHECKING:
    import polars as _pl

PL = _cast(_Any, pl_runtime)


if TYPE_CHECKING:
    from ml.data.sources.events import EarningsEvent
    from ml.data.sources.events import EconomicEvent
    from ml.data.sources.events import EventSource


logger = logging.getLogger(__name__)


class EventScheduleProvider(BaseTimeSeriesProvider):
    """
    Provider for scheduled market events.

    Provides features related to economic releases, earnings announcements,
    Fed meetings, and other scheduled events that are known in advance.
    These are ideal for TFT models' known-future inputs.

    Attributes
    ----------
    event_source : EventSource
        Source for event data
    _event_cache : dict
        Cache of loaded events by date range

    """

    _DEFAULT_EVENT_TYPES: tuple[str, ...] = (
        "fed_meeting",
        "economic_release",
        "earnings",
        "options_expiry",
    )
    _DEFAULT_HORIZON_HOURS: tuple[int, ...] = (1, 4, 24, 72)
    _EARNINGS_SEASON_MONTHS: set[int] = {1, 4, 7, 10}

    def __init__(self, event_source: EventSource) -> None:
        """
        Initialize event schedule provider.

        Parameters
        ----------
        event_source : EventSource
            Source for event data

        """
        super().__init__()
        self.event_source = event_source
        self._event_cache: dict[str, tuple[list[EconomicEvent], list[EarningsEvent]]] = {}
        logger.info(f"Initialized EventScheduleProvider with {event_source.__class__.__name__}")

    def compute_features(
        self,
        timestamps: _pl.Series,
        instruments: list[str] | None = None,
        lookback_days: int = 30,
        lookahead_days: int = 30,
    ) -> _pl.DataFrame:
        """
        Compute event-based features for timestamps.

        Parameters
        ----------
        timestamps : pl.Series
            Series of timestamps in nanoseconds since epoch
        instruments : list[str], optional
            List of instruments for earnings events
        lookback_days : int, default 30
            Days to look back for past events
        lookahead_days : int, default 30
            Days to look ahead for future events

        Returns
        -------
        pl.DataFrame
            DataFrame with event features:
            - timestamp: int
            - has_fed_event_today: bool
            - has_cpi_event_today: bool
            - has_earnings_today: bool
            - days_to_next_fed: int
            - days_to_next_cpi: int
            - days_to_next_earnings: int
            - days_since_last_fed: int
            - days_since_last_cpi: int
            - days_since_last_earnings: int
            - event_importance_score: float
            - event_clustering_score: float

        """
        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

        if instruments is None:
            instruments = []

        features = []

        # Get date range for event loading
        ts_list = timestamps.to_list()
        if ts_list:
            min_ts = min(ts_list)
            max_ts = max(ts_list)

            min_dt = datetime.fromtimestamp(min_ts / 1e9)
            max_dt = datetime.fromtimestamp(max_ts / 1e9)

            # Extend range for lookback/lookahead
            start = min_dt - timedelta(days=lookback_days)
            end = max_dt + timedelta(days=lookahead_days)

            # Load events (with caching)
            econ_events, earnings_events = self._load_events(
                start,
                end,
                instruments,
            )
        else:
            econ_events, earnings_events = [], []

        event_index, options_flags = self._prepare_event_lookup(econ_events, earnings_events)
        fomc_weeks = self._week_set(event_index.get("fed_meeting", []))
        holiday_weeks = self._week_set(event_index.get("holiday", []))
        triple_dates = {ts.date() for ts, flag in options_flags.items() if flag}

        # Process each timestamp
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts / 1e9)

            # Compute features for this timestamp
            feature_dict = self._compute_timestamp_features(
                dt,
                econ_events,
                earnings_events,
                event_index,
                options_flags,
                fomc_weeks,
                holiday_weeks,
                triple_dates,
            )
            feature_dict["timestamp"] = ts
            features.append(feature_dict)

        from typing import cast as __cast

        return __cast("_pl.DataFrame", PL.DataFrame(features))

    def _load_events(
        self,
        start: datetime,
        end: datetime,
        instruments: list[str],
    ) -> tuple[list[EconomicEvent], list[EarningsEvent]]:
        """
        Load events with caching.

        Parameters
        ----------
        start : datetime
            Start of date range
        end : datetime
            End of date range
        instruments : list[str]
            Instruments for earnings

        Returns
        -------
        tuple[list[EconomicEvent], list[EarningsEvent]]
            Economic and earnings events

        """
        # Generate cache key
        cache_key = f"{start.date()}_{end.date()}_{'_'.join(sorted(instruments))}"

        # Check cache
        if cache_key in self._event_cache:
            logger.debug(f"Event cache hit for {cache_key}")
            return self._event_cache[cache_key]

        try:
            # Load from source
            econ_events = self.event_source.get_economic_events(start, end)
            earnings_events = (
                self.event_source.get_earnings_events(
                    instruments,
                    start,
                    end,
                )
                if instruments
                else []
            )

            # Cache and return
            self._event_cache[cache_key] = (econ_events, earnings_events)
            logger.info(
                f"Loaded {len(econ_events)} economic and "
                f"{len(earnings_events)} earnings events",
            )

            return econ_events, earnings_events

        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            return [], []

    def _compute_timestamp_features(
        self,
        dt: datetime,
        econ_events: list[EconomicEvent],
        earnings_events: list[EarningsEvent],
        event_index: dict[str, list[datetime]],
        options_flags: dict[datetime, bool],
        fomc_weeks: set[tuple[int, int]],
        holiday_weeks: set[tuple[int, int]],
        triple_dates: set[date],
    ) -> dict[str, Any]:
        """
        Compute features for a single timestamp.

        Parameters
        ----------
        dt : datetime
            Timestamp to compute features for
        econ_events : list[EconomicEvent]
            Economic events
        earnings_events : list[EarningsEvent]
            Earnings events

        Returns
        -------
        dict
            Feature dictionary

        """
        # Initialize features
        features: dict[str, Any] = {
            "has_fed_event_today": False,
            "has_cpi_event_today": False,
            "has_earnings_today": False,
            "days_to_next_fed": -1,
            "days_to_next_cpi": -1,
            "days_to_next_earnings": -1,
            "days_since_last_fed": -1,
            "days_since_last_cpi": -1,
            "days_since_last_earnings": -1,
            "event_importance_score": 0.0,
            "event_clustering_score": 0.0,
        }

        # Check for events today
        today_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Economic events
        fed_events = [e for e in econ_events if "Fed" in e.name]
        cpi_events = [e for e in econ_events if "CPI" in e.name or "Consumer Price" in e.name]

        # Check today's events
        todays_events = [e for e in econ_events if today_start <= e.timestamp <= today_end]

        for event in todays_events:
            if "Fed" in event.name:
                features["has_fed_event_today"] = True
            if "CPI" in event.name or "Consumer Price" in event.name:
                features["has_cpi_event_today"] = True

        # Earnings today
        todays_earnings = [e for e in earnings_events if today_start <= e.timestamp <= today_end]
        if todays_earnings:
            features["has_earnings_today"] = True

        # Days to next events
        future_fed = [e for e in fed_events if e.timestamp > dt]
        if future_fed:
            next_fed = min(future_fed, key=lambda e: e.timestamp)
            features["days_to_next_fed"] = (next_fed.timestamp - dt).days

        future_cpi = [e for e in cpi_events if e.timestamp > dt]
        if future_cpi:
            next_cpi = min(future_cpi, key=lambda e: e.timestamp)
            features["days_to_next_cpi"] = (next_cpi.timestamp - dt).days

        future_earnings = [e for e in earnings_events if e.timestamp > dt]
        if future_earnings:
            next_earnings = min(future_earnings, key=lambda e: e.timestamp)
            features["days_to_next_earnings"] = (next_earnings.timestamp - dt).days

        # Days since last events
        past_fed = [e for e in fed_events if e.timestamp <= dt]
        if past_fed:
            last_fed = max(past_fed, key=lambda e: e.timestamp)
            features["days_since_last_fed"] = (dt - last_fed.timestamp).days

        past_cpi = [e for e in cpi_events if e.timestamp <= dt]
        if past_cpi:
            last_cpi = max(past_cpi, key=lambda e: e.timestamp)
            features["days_since_last_cpi"] = (dt - last_cpi.timestamp).days

        past_earnings = [e for e in earnings_events if e.timestamp <= dt]
        if past_earnings:
            last_earnings = max(past_earnings, key=lambda e: e.timestamp)
            features["days_since_last_earnings"] = (dt - last_earnings.timestamp).days

        # Event importance score (0-10 scale)
        importance_score = 0.0
        for event in todays_events:
            if event.importance == "HIGH":
                importance_score += 3.0
            elif event.importance == "MEDIUM":
                importance_score += 1.5
            else:
                importance_score += 0.5

        # Add earnings importance
        importance_score += len(todays_earnings) * 1.0

        # Cap at 10
        features["event_importance_score"] = min(importance_score, 10.0)

        # Event clustering score (events within ±3 days)
        window_start = dt - timedelta(days=3)
        window_end = dt + timedelta(days=3)

        nearby_events = [e for e in econ_events if window_start <= e.timestamp <= window_end]
        nearby_earnings = [e for e in earnings_events if window_start <= e.timestamp <= window_end]

        clustering_score = len(nearby_events) * 0.5 + len(nearby_earnings) * 0.3
        features["event_clustering_score"] = min(clustering_score, 10.0)

        all_events = event_index.get("all", [])
        total_events_24h = self._count_within_hours(all_events, dt, 24)
        total_events_week = self._count_within_hours(all_events, dt, 24 * 7)
        features["total_events_24h"] = total_events_24h
        features["total_events_week"] = total_events_week
        features["event_density_24h"] = total_events_24h / 24.0
        features["event_density_week"] = total_events_week / (24.0 * 7.0)
        features["days_to_next_holiday"] = self._days_to_next(event_index.get("holiday", []), dt)

        iso_week = dt.isocalendar()
        features["is_fomc_week"] = int((iso_week.year, iso_week.week) in fomc_weeks)
        features["is_holiday_week"] = int((iso_week.year, iso_week.week) in holiday_weeks)
        features["is_triple_witching"] = int(dt.date() in triple_dates)
        features["is_earnings_season"] = int(dt.month in self._EARNINGS_SEASON_MONTHS)

        for event_type in self._DEFAULT_EVENT_TYPES:
            hours_to = self._hours_to_next(event_index.get(event_type, []), dt)
            features[f"hours_to_{event_type}"] = hours_to
            features[f"has_{event_type}_in_24h"] = int(0 <= hours_to <= 24)
            features[f"has_{event_type}_in_week"] = int(0 <= hours_to <= 24 * 7)
            for horizon in self._DEFAULT_HORIZON_HOURS:
                features[f"{event_type}_within_{horizon}h"] = int(0 <= hours_to <= horizon)

        return features

    def _prepare_event_lookup(
        self,
        econ_events: list[EconomicEvent],
        earnings_events: list[EarningsEvent],
    ) -> tuple[dict[str, list[datetime]], dict[datetime, bool]]:
        event_lists: dict[str, list[datetime]] = {
            "fed_meeting": [],
            "economic_release": [],
            "options_expiry": [],
            "holiday": [],
            "earnings": [],
        }
        options_flags: dict[datetime, bool] = {}

        for event in econ_events:
            name_lower = event.name.lower()
            ts = event.timestamp
            if "fed" in name_lower:
                event_lists["fed_meeting"].append(ts)
            elif "option" in name_lower or "witch" in name_lower:
                event_lists["options_expiry"].append(ts)
                options_flags[ts] = "triple" in name_lower or "witch" in name_lower
            elif "holiday" in name_lower:
                event_lists["holiday"].append(ts)
            else:
                event_lists["economic_release"].append(ts)

        for earning_event in earnings_events:
            event_lists["earnings"].append(earning_event.timestamp)

        for key in event_lists:
            event_lists[key].sort()

        all_events = sorted({ts for lst in event_lists.values() for ts in lst})
        event_lists["all"] = all_events
        return event_lists, options_flags

    @staticmethod
    def _hours_to_next(events: list[datetime], dt: datetime) -> float:
        if not events:
            return -1.0
        idx = bisect_left(events, dt)
        if idx < len(events):
            delta = events[idx] - dt
            return delta.total_seconds() / 3600.0
        return -1.0

    @staticmethod
    def _count_within_hours(events: list[datetime], dt: datetime, horizon: int) -> int:
        if not events:
            return 0
        start_idx = bisect_left(events, dt)
        end_idx = bisect_right(events, dt + timedelta(hours=horizon))
        return max(end_idx - start_idx, 0)

    @staticmethod
    def _days_to_next(events: list[datetime], dt: datetime) -> int:
        if not events:
            return -1
        idx = bisect_left(events, dt)
        if idx < len(events):
            delta = events[idx] - dt
            return max(int(delta.total_seconds() // (3600 * 24)), 0)
        return -1

    @staticmethod
    def _week_set(events: list[datetime]) -> set[tuple[int, int]]:
        weeks: set[tuple[int, int]] = set()
        for ts in events:
            iso = ts.isocalendar()
            weeks.add((iso.year, iso.week))
        return weeks

    def load_timeseries(
        self,
        instruments: list[str],
        timestamps: _pl.Series,
    ) -> _pl.DataFrame:
        """
        Load time series event features.

        This method provides compatibility with the TimeSeriesProvider protocol.

        Parameters
        ----------
        instruments : list[str]
            List of instruments for earnings tracking
        timestamps : pl.Series
            Timestamps to compute features for

        Returns
        -------
        pl.DataFrame
            Event features

        """
        # Compute features with instruments
        df = self.compute_features(
            timestamps,
            instruments=instruments,
        )

        # Add instrument column if needed
        if instruments and len(instruments) == 1:
            df = df.with_columns(
                PL.lit(instruments[0]).alias("instrument_id"),
            )

        return df

    def _load_timeseries_impl(
        self,
        instruments: list[str],
        timestamps: _pl.Series,
    ) -> _pl.DataFrame:
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
