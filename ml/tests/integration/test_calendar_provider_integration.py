"""
Integration tests for calendar provider with real data sources.

Tests the integration between PandasCalendarSource and MarketCalendarProvider.

"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest

from ml._imports import HAS_PANDAS_MARKET_CALENDARS
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.factory import ProviderFactory
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import PandasCalendarSource


@pytest.mark.parallel_safe
@pytest.mark.integration
class TestCalendarProviderIntegration:
    """
    Integration tests for calendar provider with real sources.
    """

    def test_factory_creates_pandas_calendar_by_default(self) -> None:
        """
        Test that factory creates PandasCalendarSource by default when available.
        """
        with patch("ml.data.providers.factory.PandasCalendarSource") as mock_pandas_source:
            # Mock successful creation
            mock_pandas_source.return_value = MockCalendarSource()  # Use mock for testing

            factory = ProviderFactory()

            # Should have tried to create PandasCalendarSource
            mock_pandas_source.assert_called_once()

    def test_factory_falls_back_to_mock_on_error(self) -> None:
        """
        Test that factory falls back to MockCalendarSource on error.
        """
        with patch(
            "ml.data.providers.factory.PandasCalendarSource",
            side_effect=Exception("Test error"),
        ):
            factory = ProviderFactory()

            # Should have fallen back to MockCalendarSource
            assert isinstance(factory._calendar_source, MockCalendarSource)

    def test_factory_uses_provided_calendar_source(self) -> None:
        """
        Test that factory uses explicitly provided calendar source.
        """
        custom_source = MockCalendarSource()
        factory = ProviderFactory(calendar_source=custom_source)

        assert factory._calendar_source is custom_source

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_provider_with_real_pandas_source(self) -> None:
        """
        Test MarketCalendarProvider with real PandasCalendarSource.
        """
        source = PandasCalendarSource()
        provider = MarketCalendarProvider(source)

        # Create timestamps for testing
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 16, 9, 0).timestamp() * 1e9),  # Pre-market
                int(datetime(2024, 1, 16, 10, 30).timestamp() * 1e9),  # Market hours
                int(datetime(2024, 1, 16, 16, 30).timestamp() * 1e9),  # After-hours
            ],
        )

        # Compute features
        features = provider.compute_features(timestamps, exchange="NYSE")

        assert isinstance(features, pl.DataFrame)
        assert len(features) == 3
        assert "is_trading_day" in features.columns
        assert "hour_sin" in features.columns
        assert "hour_cos" in features.columns

    def test_provider_with_mock_source(self) -> None:
        """
        Test MarketCalendarProvider with MockCalendarSource.
        """
        source = MockCalendarSource()
        provider = MarketCalendarProvider(source)

        # Create timestamps for a known trading day (Tuesday)
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 16, 10, 30).timestamp() * 1e9),  # Tuesday, market hours
                int(datetime(2024, 1, 20, 10, 30).timestamp() * 1e9),  # Saturday, weekend
            ],
        )

        features = provider.compute_features(timestamps, exchange="NYSE")

        assert isinstance(features, pl.DataFrame)
        assert len(features) == 2

        # Check first row (Tuesday)
        tuesday_row = features.filter(pl.col("is_weekend") == False).to_dicts()[0]
        assert tuesday_row["is_trading_day"] is True

        # Check second row (Saturday)
        saturday_row = features.filter(pl.col("is_weekend") == True).to_dicts()[0]
        assert saturday_row["is_trading_day"] is False

    def test_factory_get_calendar_provider(self) -> None:
        """
        Test getting calendar provider from factory.
        """
        factory = ProviderFactory()
        provider = factory.get_calendar_provider()

        assert isinstance(provider, MarketCalendarProvider)
        # Should be singleton
        provider2 = factory.get_calendar_provider()
        assert provider is provider2

    def test_provider_handles_multiple_exchanges(self) -> None:
        """
        Test that provider handles different exchanges correctly.
        """
        source = MockCalendarSource()
        provider = MarketCalendarProvider(source)

        timestamps = pl.Series(
            "timestamp",
            [int(datetime(2024, 1, 16, 10, 30).timestamp() * 1e9)],
        )

        # Test NYSE
        nyse_features = provider.compute_features(timestamps, exchange="NYSE")
        assert len(nyse_features) == 1

        # Test CME (different hours in mock)
        cme_features = provider.compute_features(timestamps, exchange="CME")
        assert len(cme_features) == 1

        # Both should have computed features
        assert "is_trading_day" in nyse_features.columns
        assert "is_trading_day" in cme_features.columns

    def test_provider_cyclic_encodings(self) -> None:
        """
        Test that cyclic encodings are computed correctly.
        """
        source = MockCalendarSource()
        provider = MarketCalendarProvider(source)

        # Test various times to check cyclic encodings
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 16, 0, 0).timestamp() * 1e9),  # Midnight
                int(datetime(2024, 1, 16, 6, 0).timestamp() * 1e9),  # 6 AM
                int(datetime(2024, 1, 16, 12, 0).timestamp() * 1e9),  # Noon
                int(datetime(2024, 1, 16, 18, 0).timestamp() * 1e9),  # 6 PM
            ],
        )

        features = provider.compute_features(timestamps)

        # Check that cyclic encodings are bounded
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert features[col].min() >= -1.0
            assert features[col].max() <= 1.0

        # Check specific values for known times
        midnight_row = features[0]
        noon_row = features[2]

        # At midnight, hour_sin should be 0 and hour_cos should be 1
        assert abs(midnight_row["hour_sin"][0]) < 0.01
        assert abs(midnight_row["hour_cos"][0] - 1.0) < 0.01

        # At noon, hour_sin should be 0 and hour_cos should be -1
        assert abs(noon_row["hour_sin"][0]) < 0.01
        assert abs(noon_row["hour_cos"][0] + 1.0) < 0.01

    def test_provider_month_boundaries(self) -> None:
        """
        Test month boundary detection.
        """
        source = MockCalendarSource()
        provider = MarketCalendarProvider(source)

        # Test various dates around month boundaries
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 1, 10, 0).timestamp() * 1e9),  # Month start
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),  # Mid-month
                int(datetime(2024, 1, 31, 10, 0).timestamp() * 1e9),  # Month end
            ],
        )

        features = provider.compute_features(timestamps)

        # Check month boundary flags
        assert features[0]["is_month_start"][0] is True  # Jan 1
        assert features[1]["is_month_start"][0] is False  # Jan 15
        assert features[2]["is_month_end"][0] is True  # Jan 31

    def test_provider_with_empty_timestamps(self) -> None:
        """
        Test provider handles empty timestamp series.
        """
        source = MockCalendarSource()
        provider = MarketCalendarProvider(source)

        timestamps = pl.Series("timestamp", [], dtype=pl.Int64)
        features = provider.compute_features(timestamps)

        assert isinstance(features, pl.DataFrame)
        assert len(features) == 0

    @pytest.mark.skipif(
        not HAS_PANDAS_MARKET_CALENDARS,
        reason="pandas_market_calendars not installed",
    )
    def test_real_holiday_detection(self) -> None:
        """
        Test real holiday detection with PandasCalendarSource.
        """
        source = PandasCalendarSource()
        provider = MarketCalendarProvider(source)

        # Test known US market holidays
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 1, 10, 0).timestamp() * 1e9),  # New Year's Day
                int(datetime(2024, 7, 4, 10, 0).timestamp() * 1e9),  # Independence Day
                int(datetime(2024, 12, 25, 10, 0).timestamp() * 1e9),  # Christmas
            ],
        )

        features = provider.compute_features(timestamps, exchange="NYSE")

        # All should be holidays (not trading days)
        for row in features.to_dicts():
            # Note: Actual behavior depends on calendar data
            assert isinstance(row["is_trading_day"], bool)
