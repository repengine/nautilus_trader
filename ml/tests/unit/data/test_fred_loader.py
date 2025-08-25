#!/usr/bin/env python3
"""
Unit tests for FRED data loader.

Tests the FREDDataLoader class with mocked API responses.

"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml._imports import pl
from ml.data.loaders import FREDConfig
from ml.data.loaders import FREDDataLoader
from ml.data.loaders import FREDIndicator


class TestFREDConfig:
    """Test FRED configuration."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = FREDConfig(api_key="test_key")

        assert config.api_key == "test_key"
        assert config.cache_ttl_hours == 24
        assert config.rate_limit_calls == 120
        assert config.backfill_years == 10
        assert config.max_retries == 3

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading API key from environment."""
        monkeypatch.setenv("FRED_API_KEY", "env_test_key")

        config = FREDConfig()
        assert config.api_key == "env_test_key"

    def test_config_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when API key is missing."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        with pytest.raises(ValueError, match="FRED API key not provided"):
            FREDConfig()

    def test_cache_dir_creation(self, tmp_path: Path) -> None:
        """Test cache directory is created."""
        cache_dir = tmp_path / "fred_cache"
        config = FREDConfig(api_key="test", cache_dir=cache_dir)

        assert cache_dir.exists()
        assert config.cache_dir == cache_dir


class TestFREDIndicator:
    """Test FRED indicator dataclass."""

    def test_indicator_creation(self) -> None:
        """Test creating an indicator."""
        indicator = FREDIndicator(
            series_id="DGS10",
            name="10-Year Treasury",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="10-year constant maturity rate",
        )

        assert indicator.series_id == "DGS10"
        assert indicator.name == "10-Year Treasury"
        assert indicator.category == "interest_rates"
        assert indicator.frequency == "daily"

    def test_indicator_defaults(self) -> None:
        """Test indicator default values."""
        indicator = FREDIndicator(
            series_id="TEST",
            name="Test Indicator",
            category="test",
        )

        assert indicator.frequency == "daily"
        assert indicator.units == "percent"
        assert indicator.seasonal_adjustment == "NSA"


@pytest.fixture
def mock_fred_api() -> MagicMock:
    """Create mock FRED API client."""
    mock = MagicMock()

    # Create sample data
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    values = np.random.randn(len(dates)) * 2 + 3.5

    series = pd.Series(values, index=dates, name="DGS10")
    mock.get_series.return_value = series

    return mock


@pytest.fixture
def fred_loader(tmp_path: Path, mock_fred_api: MagicMock) -> FREDDataLoader:
    """Create FRED loader with mocked API."""
    config = FREDConfig(
        api_key="test_key",
        cache_dir=tmp_path / "cache",
        cache_ttl_hours=1,
        rate_limit_calls=10,
    )

    # Mock both the import check and the Fred class
    with patch("ml.data.loaders.fred_loader.HAS_FREDAPI", True):
        with patch("ml.data.loaders.fred_loader.Fred", return_value=mock_fred_api):
            loader = FREDDataLoader(config)

    # Replace the fred client with our mock
    loader.fred = mock_fred_api

    return loader


class TestFREDDataLoader:
    """Test FRED data loader."""

    def test_loader_initialization(self, tmp_path: Path) -> None:
        """Test loader initialization."""
        config = FREDConfig(api_key="test", cache_dir=tmp_path)

        with patch("ml.data.loaders.fred_loader.HAS_FREDAPI", True):
            with patch("ml.data.loaders.fred_loader.Fred") as mock_fred:
                loader = FREDDataLoader(config)

                assert loader.config == config
                assert len(loader.indicators) > 0
                mock_fred.assert_called_once_with(api_key="test")

    def test_custom_indicators(self, tmp_path: Path) -> None:
        """Test loader with custom indicators."""
        config = FREDConfig(api_key="test", cache_dir=tmp_path)

        indicators = [
            FREDIndicator(
                series_id="CUSTOM1",
                name="Custom Indicator 1",
                category="test",
            ),
            FREDIndicator(
                series_id="CUSTOM2",
                name="Custom Indicator 2",
                category="test",
            ),
        ]

        with patch("ml.data.loaders.fred_loader.HAS_FREDAPI", True):
            with patch("ml.data.loaders.fred_loader.Fred"):
                loader = FREDDataLoader(config, indicators=indicators)

                assert len(loader.indicators) == 2
                assert loader.indicators[0].series_id == "CUSTOM1"

    def test_fetch_indicator(self, fred_loader: FREDDataLoader) -> None:
        """Test fetching a single indicator."""
        df = fred_loader.fetch_indicator("DGS10", use_cache=False)

        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "series_id" in df.columns
        assert "value" in df.columns
        assert "timestamp_ns" in df.columns

        assert len(df) == 10
        assert df["series_id"][0] == "DGS10"

        # Check API was called
        fred_loader.fred.get_series.assert_called_once()

    def test_fetch_indicator_with_dates(self, fred_loader: FREDDataLoader) -> None:
        """Test fetching with specific date range."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)

        df = fred_loader.fetch_indicator(
            "DGS10",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        assert isinstance(df, pl.DataFrame)

        # Check API was called with correct dates
        call_args = fred_loader.fred.get_series.call_args
        assert call_args[1]["observation_start"] == start_date
        assert call_args[1]["observation_end"] == end_date

    def test_cache_functionality(self, fred_loader: FREDDataLoader) -> None:
        """Test caching of fetched data."""
        # First fetch - should call API
        df1 = fred_loader.fetch_indicator("DGS10", use_cache=True)
        assert fred_loader.fred.get_series.call_count == 1

        # Second fetch - should use cache
        df2 = fred_loader.fetch_indicator("DGS10", use_cache=True)
        assert fred_loader.fred.get_series.call_count == 1  # No additional call

        # Data should be identical
        assert df1.equals(df2)

    def test_cache_expiry(self, fred_loader: FREDDataLoader) -> None:
        """Test cache expiry."""
        # Fetch and cache data
        df1 = fred_loader.fetch_indicator("DGS10", use_cache=True)
        assert fred_loader.fred.get_series.call_count == 1

        # Modify cache metadata to simulate expiry
        metadata_path = fred_loader._get_cache_metadata_path("DGS10")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Set timestamp to 2 hours ago (cache TTL is 1 hour)
        metadata["timestamp"] = time.time() - 7200

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Fetch again - should call API due to expired cache
        df2 = fred_loader.fetch_indicator("DGS10", use_cache=True)
        assert fred_loader.fred.get_series.call_count == 2

    def test_rate_limiting(self, fred_loader: FREDDataLoader) -> None:
        """Test rate limiting functionality."""
        # Set very low rate limit for testing
        fred_loader.config.rate_limit_calls = 2
        fred_loader._rate_limit_window = 0.1  # 100ms window

        start_time = time.time()

        # Make multiple calls
        for i in range(3):
            fred_loader._rate_limit()

        elapsed = time.time() - start_time

        # Should have waited due to rate limit
        assert elapsed >= 0.1

    def test_fetch_with_retry(self, fred_loader: FREDDataLoader) -> None:
        """Test retry logic on API failure."""
        # Make API fail twice, then succeed
        fred_loader.fred.get_series.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3)),
        ]

        df = fred_loader.fetch_indicator("DGS10", use_cache=False)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert fred_loader.fred.get_series.call_count == 3

    def test_fetch_all_indicators(self, fred_loader: FREDDataLoader) -> None:
        """Test fetching all indicators."""
        # Set limited indicators for testing
        fred_loader.indicators = [
            FREDIndicator("DGS10", "10-Year", "rates"),
            FREDIndicator("DGS2", "2-Year", "rates"),
        ]

        # Mock different data for each series
        def get_series_side_effect(series_id: str, **kwargs: Any) -> pd.Series:
            dates = pd.date_range("2020-01-01", periods=5)
            if series_id == "DGS10":
                values = [3.5, 3.6, 3.7, 3.8, 3.9]
            else:
                values = [1.5, 1.6, 1.7, 1.8, 1.9]
            return pd.Series(values, index=dates, name=series_id)

        fred_loader.fred.get_series.side_effect = get_series_side_effect

        data = fred_loader.fetch_all_indicators(use_cache=False)

        assert len(data) == 2
        assert "DGS10" in data
        assert "DGS2" in data
        assert len(data["DGS10"]) == 5
        assert len(data["DGS2"]) == 5

    def test_combine_indicators(self, fred_loader: FREDDataLoader) -> None:
        """Test combining multiple indicators."""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=5)

        data = {
            "DGS10": pl.DataFrame({
                "timestamp": dates,
                "value": [3.5, 3.6, 3.7, 3.8, 3.9],
            }),
            "DGS2": pl.DataFrame({
                "timestamp": dates,
                "value": [1.5, 1.6, 1.7, 1.8, 1.9],
            }),
        }

        combined = fred_loader.combine_indicators(data)

        assert isinstance(combined, pl.DataFrame)
        assert "timestamp" in combined.columns
        assert "timestamp_ns" in combined.columns
        assert "DGS10" in combined.columns
        assert "DGS2" in combined.columns
        assert len(combined) == 5

    def test_combine_indicators_with_gaps(self, fred_loader: FREDDataLoader) -> None:
        """Test combining indicators with different date ranges."""
        dates1 = pd.date_range("2020-01-01", periods=5)
        dates2 = pd.date_range("2020-01-03", periods=5)  # Starts 2 days later

        data = {
            "DGS10": pl.DataFrame({
                "timestamp": dates1,
                "value": [3.5, 3.6, 3.7, 3.8, 3.9],
            }),
            "DGS2": pl.DataFrame({
                "timestamp": dates2,
                "value": [1.7, 1.8, 1.9, 2.0, 2.1],
            }),
        }

        combined = fred_loader.combine_indicators(data)

        assert isinstance(combined, pl.DataFrame)
        assert len(combined) == 7  # Union of dates

        # Check for nulls where data is missing
        # Find the rows for the first two dates (which should have no DGS2 data)
        combined_sorted = combined.sort("timestamp")
        dgs10_values = combined_sorted["DGS10"].to_list()
        dgs2_values = combined_sorted["DGS2"].to_list()

        # First two dates should have DGS10 data but no DGS2 data
        assert dgs10_values[0] is not None  # Has DGS10 data
        assert dgs2_values[0] is None  # No DGS2 data for first date
        assert dgs10_values[1] is not None  # Has DGS10 data
        assert dgs2_values[1] is None  # No DGS2 data for second date

    def test_store_indicators(
        self,
        fred_loader: FREDDataLoader,
        tmp_path: Path,
    ) -> None:
        """Test storing indicators in DataStore."""
        from ml.registry.data_registry import DataRegistry
        from ml.stores.data_store import DataStore

        # Create mock stores and registry
        mock_data_store = MagicMock(spec=DataStore)
        mock_registry = MagicMock(spec=DataRegistry)

        # Create sample data
        dates = pd.date_range("2020-01-01", periods=5)
        data = {
            "DGS10": pl.DataFrame({
                "timestamp": dates,
                "value": [3.5, 3.6, 3.7, 3.8, 3.9],
            }),
        }

        # Store indicators
        fred_loader.store_indicators(mock_data_store, mock_registry, data)

        # Check registry was called
        mock_registry.register_dataset.assert_called_once()
        manifest = mock_registry.register_dataset.call_args[0][0]
        assert manifest.dataset_id == "fred_economic_indicators"
        assert manifest.dataset_type.value == "features"  # Economic indicators are stored as features

        # Check data store was called
        mock_data_store.write_ingestion.assert_called()

    def test_update_realtime(
        self,
        fred_loader: FREDDataLoader,
    ) -> None:
        """Test real-time update functionality."""
        from ml.registry.data_registry import DataRegistry
        from ml.stores.data_store import DataStore

        # Create mock stores and registry
        mock_data_store = MagicMock(spec=DataStore)
        mock_registry = MagicMock(spec=DataRegistry)

        # Mock fetch to return data
        dates = pd.date_range("2020-01-01", periods=5)
        fred_loader.fred.get_series.return_value = pd.Series(
            [3.5, 3.6, 3.7, 3.8, 3.9],
            index=dates,
            name="DGS10",
        )

        # Set limited indicators for testing
        fred_loader.indicators = [
            FREDIndicator("DGS10", "10-Year", "rates"),
        ]

        # Run update
        fred_loader.update_realtime(mock_data_store, mock_registry)

        # Check that fetch was called with recent dates
        call_args = fred_loader.fred.get_series.call_args
        start_date = call_args[1]["observation_start"]
        end_date = call_args[1]["observation_end"]

        # Should fetch last 30 days
        date_diff = (end_date - start_date).days
        assert 29 <= date_diff <= 31

        # Check store was called
        mock_data_store.write_ingestion.assert_called()


class TestFREDDataLoaderIntegration:
    """Integration tests for FRED loader (requires API key)."""

    @pytest.mark.skipif(
        not os.getenv("FRED_API_KEY"),
        reason="FRED_API_KEY not set",
    )
    def test_real_api_fetch(self, tmp_path: Path) -> None:
        """Test with real FRED API (requires API key)."""
        config = FREDConfig(cache_dir=tmp_path)
        loader = FREDDataLoader(config)

        # Fetch a single day of DGS10 data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        df = loader.fetch_indicator(
            "DGS10",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "value" in df.columns

        # Values should be reasonable for treasury yields
        values = df["value"].to_list()
        assert all(0 < v < 10 for v in values if v is not None)
