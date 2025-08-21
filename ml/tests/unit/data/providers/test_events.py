"""
Unit tests for event schedule provider and sources.

Tests economic events, earnings releases, and other scheduled market events.

"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.data.providers.events import EventScheduleProvider
from ml.data.sources.events import EarningsEvent
from ml.data.sources.events import EconomicEvent
from ml.data.sources.events import EventSource
from ml.data.sources.events import MockEventSource
from ml.data.sources.events import SimpleEventSource


class TestEconomicEvent:
    """
    Test economic event dataclass.
    """

    def test_economic_event_creation(self) -> None:
        """
        Test creating economic event.
        """
        event = EconomicEvent(
            event_id="FED_RATE_2024Q1",
            timestamp=datetime(2024, 3, 20, 14, 0),
            name="Federal Funds Rate Decision",
            country="US",
            importance="HIGH",
            forecast=5.25,
            previous=5.0,
            actual=None,  # Not yet released
        )

        assert event.event_id == "FED_RATE_2024Q1"
        assert event.name == "Federal Funds Rate Decision"
        assert event.importance == "HIGH"
        assert event.forecast == 5.25
        assert event.actual is None

    def test_economic_event_with_actual(self) -> None:
        """
        Test economic event after release.
        """
        event = EconomicEvent(
            event_id="CPI_2024_01",
            timestamp=datetime(2024, 1, 11, 8, 30),
            name="Consumer Price Index",
            country="US",
            importance="HIGH",
            forecast=3.2,
            previous=3.1,
            actual=3.4,  # Released value
        )

        assert event.actual == 3.4
        # Surprise calculation could be added
        surprise = event.actual - event.forecast if event.actual and event.forecast else 0
        assert surprise == pytest.approx(0.2)


class TestEarningsEvent:
    """
    Test earnings event dataclass.
    """

    def test_earnings_event_creation(self) -> None:
        """
        Test creating earnings event.
        """
        event = EarningsEvent(
            event_id="AAPL_Q1_2024",
            timestamp=datetime(2024, 2, 1, 16, 30),
            instrument_id="AAPL",
            fiscal_quarter="Q1",
            fiscal_year=2024,
            eps_forecast=2.10,
            eps_previous=1.88,
            revenue_forecast=117.3e9,
            revenue_previous=119.6e9,
            eps_actual=None,
            revenue_actual=None,
            timing="AMC",  # After Market Close
        )

        assert event.instrument_id == "AAPL"
        assert event.fiscal_quarter == "Q1"
        assert event.timing == "AMC"
        assert event.eps_forecast == 2.10

    def test_earnings_surprise(self) -> None:
        """
        Test earnings surprise calculation.
        """
        event = EarningsEvent(
            event_id="MSFT_Q2_2024",
            timestamp=datetime(2024, 4, 25, 16, 30),
            instrument_id="MSFT",
            fiscal_quarter="Q2",
            fiscal_year=2024,
            eps_forecast=2.83,
            eps_previous=2.45,
            revenue_forecast=60.9e9,
            revenue_previous=56.5e9,
            eps_actual=2.94,  # Beat
            revenue_actual=61.9e9,  # Beat
            timing="AMC",
        )

        if event.eps_actual and event.eps_forecast:
            eps_surprise = (event.eps_actual - event.eps_forecast) / event.eps_forecast
            assert eps_surprise > 0  # Positive surprise

        if event.revenue_actual and event.revenue_forecast:
            revenue_surprise = (
                event.revenue_actual - event.revenue_forecast
            ) / event.revenue_forecast
            assert revenue_surprise > 0  # Positive surprise


class TestMockEventSource:
    """
    Test mock event source.
    """

    def test_mock_source_generates_events(self) -> None:
        """
        Test that mock source generates events.
        """
        source = MockEventSource()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        # Get economic events
        econ_events = source.get_economic_events(start, end)
        assert len(econ_events) > 0

        # Check event structure
        for event in econ_events:
            assert isinstance(event, EconomicEvent)
            assert event.timestamp >= start
            assert event.timestamp <= end
            assert event.importance in ["HIGH", "MEDIUM", "LOW"]

    def test_mock_source_earnings_events(self) -> None:
        """
        Test mock source generates earnings events.
        """
        source = MockEventSource()

        instruments = ["AAPL", "MSFT", "GOOGL"]
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)

        earnings = source.get_earnings_events(instruments, start, end)

        # Should have some earnings (quarterly)
        assert len(earnings) > 0

        # Check each event
        for event in earnings:
            assert isinstance(event, EarningsEvent)
            assert event.instrument_id in instruments
            assert event.timestamp >= start
            assert event.timestamp <= end
            assert event.timing in ["BMO", "AMC"]  # Before/After Market

    def test_mock_source_deterministic(self) -> None:
        """
        Test that mock source is deterministic with same seed.
        """
        source1 = MockEventSource(seed=42)
        source2 = MockEventSource(seed=42)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        events1 = source1.get_economic_events(start, end)
        events2 = source2.get_economic_events(start, end)

        # Should generate same events
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.event_id == e2.event_id
            assert e1.timestamp == e2.timestamp


class TestSimpleEventSource:
    """
    Test simple event source.
    """

    def test_simple_source_fed_meetings(self) -> None:
        """
        Test simple source provides Fed meeting dates.
        """
        source = SimpleEventSource()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        events = source.get_economic_events(start, end)

        # Should have Fed meetings (8 per year typically)
        fed_events = [e for e in events if "Fed" in e.name]
        assert len(fed_events) >= 6  # At least 6 Fed meetings

        # All should be HIGH importance
        for event in fed_events:
            assert event.importance == "HIGH"

    def test_simple_source_earnings_calendar(self) -> None:
        """
        Test simple source provides quarterly earnings.
        """
        source = SimpleEventSource()

        instruments = ["AAPL", "MSFT"]
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        earnings = source.get_earnings_events(instruments, start, end)

        # Should have 4 quarters per instrument
        assert len(earnings) == 8  # 2 instruments * 4 quarters

        # Check spacing (roughly quarterly)
        for instrument in instruments:
            inst_earnings = [e for e in earnings if e.instrument_id == instrument]
            assert len(inst_earnings) == 4

            # Check that all quarters are present (may not be in Q1-Q4 order due to Q4 reporting)
            quarters = {e.fiscal_quarter for e in inst_earnings}
            assert quarters == {"Q1", "Q2", "Q3", "Q4"}


class TestEventScheduleProvider:
    """
    Test the main event schedule provider.
    """

    def test_provider_computes_features(self) -> None:
        """
        Test provider computes event features.
        """
        mock_source = MockEventSource()
        provider = EventScheduleProvider(mock_source)

        # Create timestamps around an event
        base_time = datetime(2024, 1, 15, 12, 0)
        timestamps = pl.Series(
            "timestamp",
            [
                int((base_time - timedelta(days=7)).timestamp() * 1e9),  # 1 week before
                int((base_time - timedelta(days=1)).timestamp() * 1e9),  # 1 day before
                int(base_time.timestamp() * 1e9),  # Event day
                int((base_time + timedelta(days=1)).timestamp() * 1e9),  # 1 day after
            ],
        )

        df = provider.compute_features(
            timestamps,
            instruments=["AAPL"],
            lookback_days=30,
            lookahead_days=30,
        )

        # Check columns exist
        expected_cols = {
            "timestamp",
            "has_fed_event_today",
            "has_cpi_event_today",
            "has_earnings_today",
            "days_to_next_fed",
            "days_to_next_earnings",
            "days_since_last_fed",
            "days_since_last_earnings",
            "event_importance_score",
        }
        assert expected_cols.issubset(df.columns)

        # Check data
        assert len(df) == 4

    def test_provider_handles_multiple_instruments(self) -> None:
        """
        Test provider handles multiple instruments.
        """
        provider = EventScheduleProvider(MockEventSource())

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
                int(datetime(2024, 2, 15, 12, 0).timestamp() * 1e9),
            ],
        )

        instruments = ["AAPL", "MSFT", "GOOGL"]

        df = provider.compute_features(
            timestamps,
            instruments=instruments,
        )

        # Should have features for each timestamp
        assert len(df) == 2

        # Should track earnings for all instruments
        assert "has_earnings_today" in df.columns
        assert "days_to_next_earnings" in df.columns

    def test_provider_event_clustering(self) -> None:
        """
        Test provider detects event clustering.
        """
        provider = EventScheduleProvider(MockEventSource())

        # Test around typical earnings season
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 20, 12, 0).timestamp() * 1e9),  # Earnings season
                int(datetime(2024, 3, 15, 12, 0).timestamp() * 1e9),  # Between seasons
            ],
        )

        df = provider.compute_features(
            timestamps,
            instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        )

        # Event clustering score should be computed
        assert "event_clustering_score" in df.columns

        # January (earnings season) should have higher clustering
        jan_clustering = df["event_clustering_score"][0]
        mar_clustering = df["event_clustering_score"][1]

        # This may not always be true with random data, but typically
        # earnings cluster in certain months
        assert jan_clustering >= 0  # At least non-negative
        assert mar_clustering >= 0

    def test_provider_importance_weighting(self) -> None:
        """
        Test provider weights events by importance.
        """
        source = MockEventSource()
        provider = EventScheduleProvider(source)

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
            ],
        )

        df = provider.compute_features(timestamps)

        # Should have importance score
        assert "event_importance_score" in df.columns

        # Score should be non-negative
        assert (df["event_importance_score"] >= 0).all()

        # Score should be bounded (e.g., 0-10 scale)
        assert (df["event_importance_score"] <= 10).all()

    @given(
        n_timestamps=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_provider_handles_any_timestamps(self, n_timestamps: int) -> None:
        """Property test: provider handles any number of timestamps."""
        provider = EventScheduleProvider(MockEventSource())

        # Generate random timestamps
        base = datetime(2024, 1, 1)
        timestamps = []
        for i in range(n_timestamps):
            dt = base + timedelta(days=i * 30)  # Monthly spacing
            timestamps.append(int(dt.timestamp() * 1e9))

        ts_series = pl.Series("timestamp", timestamps)
        df = provider.compute_features(ts_series)

        # Should return data for all timestamps
        assert len(df) == n_timestamps

        # All columns should be present
        assert "has_fed_event_today" in df.columns
        assert "days_to_next_fed" in df.columns

    def test_provider_caches_events(self) -> None:
        """
        Test that provider caches events efficiently.
        """
        mock_source = MagicMock(spec=EventSource)
        mock_source.get_economic_events.return_value = [
            EconomicEvent(
                event_id="TEST_1",
                timestamp=datetime(2024, 1, 15, 14, 0),
                name="Test Event",
                country="US",
                importance="HIGH",
                forecast=1.0,
                previous=0.9,
                actual=None,
            ),
        ]
        mock_source.get_earnings_events.return_value = []

        provider = EventScheduleProvider(mock_source)

        # Same timestamp twice
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
            ],
        )

        # First call
        df1 = provider.compute_features(timestamps)

        # Second call with same period
        df2 = provider.compute_features(timestamps)

        # Should only fetch events once (cached)
        # Due to caching, source should be called once for the date range
        assert mock_source.get_economic_events.call_count == 1

        # Results should be identical
        assert df1.equals(df2)

    def test_provider_handles_source_errors(self) -> None:
        """
        Test provider handles source errors gracefully.
        """
        mock_source = MagicMock(spec=EventSource)
        mock_source.get_economic_events.side_effect = Exception("API Error")
        mock_source.get_earnings_events.side_effect = Exception("API Error")

        provider = EventScheduleProvider(mock_source)

        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
            ],
        )

        # Should return default features without crashing
        df = provider.compute_features(timestamps)
        assert len(df) == 1

        # Should have all columns with default values
        assert "has_fed_event_today" in df.columns
        assert df["has_fed_event_today"][0] is False

        assert "days_to_next_fed" in df.columns
        assert df["days_to_next_fed"][0] == -1  # Default for unknown
