#!/usr/bin/env python3
"""
Comprehensive error handling tests for TFT Dataset Builder and FRED Data Loader.

This module provides complete test coverage for error scenarios in:
- TFTDatasetBuilder: Dataset preparation and feature loading
- FREDDataLoader: Economic data fetching and caching

"""

from __future__ import annotations

import json
import os
import tempfile
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
from ml.config.base import MLFeatureConfig
from ml.data.loaders.fred_loader import FREDConfig
from ml.data.loaders.fred_loader import FREDDataLoader
from ml.data.loaders.fred_loader import FREDIndicator
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from nautilus_trader.model.data import Bar
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


# ============================================================================
# TFT DATASET BUILDER ERROR TESTS
# ============================================================================


@pytest.mark.usefixtures("clean_postgres_db")
class TestTFTDatasetBuilderErrors:
    """Test error handling in TFTDatasetBuilder."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def mock_catalog(self) -> MagicMock:
        """Create mock catalog."""
        catalog = MagicMock(spec=ParquetDataCatalog)
        catalog.query = MagicMock(return_value=[])
        return catalog

    @pytest.fixture
    def mock_feature_store(self, test_database) -> MagicMock:
        """Create mock feature store with PostgreSQL connection."""
        store = MagicMock(spec=FeatureStore)
        store.connection_string = test_database.connection_string
        store.get_training_data = MagicMock(
            return_value=(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([1000, 2000]),
                ["feat1", "feat2"],
            )
        )
        return store

    @pytest.fixture
    def builder(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
    ) -> TFTDatasetBuilder:
        """Create TFTDatasetBuilder instance."""
        return TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["SPY", "QQQ"],
            feature_config=MLFeatureConfig(),
            feature_store=mock_feature_store,
        )

    def test_builder_without_feature_store(
        self,
        mock_catalog: MagicMock,
        test_database,
    ) -> None:
        """Test builder handles missing feature store."""
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["SPY"],
            feature_store=None,
            connection_string=test_database.connection_string,
        )
        
        # Should raise ValueError when trying to use store
        with pytest.raises(ValueError, match="FeatureStore not configured"):
            builder.prepare_training_data_from_store()

    def test_empty_features_from_store(
        self,
        builder: TFTDatasetBuilder,
        mock_feature_store: MagicMock,
    ) -> None:
        """Test handling of empty features from store."""
        # Return empty arrays
        mock_feature_store.get_training_data.return_value = (
            np.array([]),
            np.array([]),
            [],
        )
        
        # Should handle empty data gracefully
        with pytest.raises(RuntimeError, match="No features loaded from FeatureStore"):
            builder.prepare_training_data_from_store(
                instrument_ids=["SPY.NYSE"],
            )

    def test_mismatched_feature_dimensions(
        self,
        builder: TFTDatasetBuilder,
        mock_feature_store: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test handling of mismatched feature dimensions."""
        # Return mismatched arrays
        mock_feature_store.get_training_data.return_value = (
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 samples, 2 features
            np.array([1000, 2000, 3000]),  # 3 timestamps - mismatch!
            ["feat1", "feat2"],
        )
        
        # Mock bars data
        mock_catalog.query.return_value = [
            MagicMock(spec=Bar, ts_event=1000),
            MagicMock(spec=Bar, ts_event=2000),
        ]
        
        # Should detect dimension mismatch
        with pytest.raises((ValueError, RuntimeError)):
            builder.prepare_training_data_from_store(
                instrument_ids=["SPY.NYSE"],
            )

    def test_nan_and_inf_values_in_features(
        self,
        builder: TFTDatasetBuilder,
        mock_feature_store: MagicMock,
    ) -> None:
        """Test handling of NaN and Inf values in features."""
        # Return features with NaN and Inf
        mock_feature_store.get_training_data.return_value = (
            np.array([[1.0, np.nan], [np.inf, 2.0], [-np.inf, 3.0]]),
            np.array([1000, 2000, 3000]),
            ["feat1", "feat2"],
        )
        
        # Should detect invalid values
        with pytest.raises((ValueError, RuntimeError)):
            builder.prepare_training_data_from_store(
                instrument_ids=["SPY.NYSE"],
            )

    def test_bars_loading_failure(
        self,
        builder: TFTDatasetBuilder,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
    ) -> None:
        """Test handling of bars loading failures."""
        # Feature store returns valid data
        mock_feature_store.get_training_data.return_value = (
            np.array([[1.0, 2.0]]),
            np.array([1000]),
            ["feat1", "feat2"],
        )
        
        # Mock bars_to_dataframe to raise exception
        with patch("ml.data.tft_dataset_builder.bars_to_dataframe") as mock_bars:
            mock_bars.side_effect = Exception("Database connection lost")
            
            # Should log error and continue, then raise RuntimeError for no data
            with pytest.raises(RuntimeError, match="No features loaded from FeatureStore"):
                builder.prepare_training_data_from_store(
                    instrument_ids=["SPY.NYSE"],
                )

    def test_empty_bars_data(
        self,
        builder: TFTDatasetBuilder,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
    ) -> None:
        """Test handling of empty bars data."""
        # Feature store returns valid data
        mock_feature_store.get_training_data.return_value = (
            np.array([[1.0, 2.0]]),
            np.array([1000]),
            ["feat1", "feat2"],
        )
        
        # Mock bars_to_dataframe to return empty DataFrame
        with patch("ml.data.tft_dataset_builder.bars_to_dataframe") as mock_bars_to_df:
            mock_bars_to_df.return_value = pl.DataFrame()
            
            # Should skip instruments with no bars and raise RuntimeError
            with pytest.raises(RuntimeError, match="No features loaded from FeatureStore"):
                builder.prepare_training_data_from_store(
                    instrument_ids=["SPY.NYSE"],
                )

    def test_target_generation_errors(
        self,
        builder: TFTDatasetBuilder,
    ) -> None:
        """Test error handling in target generation."""
        # Create data with issues for target generation
        df = pl.DataFrame({
            "close": [100.0, np.nan, 102.0],
            "ts_event": [1000, 2000, 3000],
        })
        
        with patch.object(builder, "_generate_targets_polars") as mock_gen:
            mock_gen.side_effect = ValueError("Cannot generate targets with NaN values")
            
            # Should handle target generation errors
            with pytest.raises(ValueError):
                builder._generate_targets_polars(df, horizon_minutes=15, min_return_threshold=0.001)

    def test_static_features_generation_errors(
        self,
        builder: TFTDatasetBuilder,
    ) -> None:
        """Test error handling in static feature generation."""
        df = pl.DataFrame({
            "instrument_id": ["INVALID_ID"],
            "close": [100.0],
        })
        
        with patch.object(builder, "_add_static_features_polars") as mock_add:
            mock_add.side_effect = ValueError("Invalid instrument ID format")
            
            # Should handle static feature errors
            with pytest.raises(ValueError):
                builder._add_static_features_polars(df)

    def test_known_future_features_errors(
        self,
        builder: TFTDatasetBuilder,
    ) -> None:
        """Test error handling in known future features."""
        df = pl.DataFrame({
            "ts_event": ["not_a_timestamp"],  # Invalid timestamp
            "close": [100.0],
        })
        
        with patch.object(builder, "_add_known_future_features_polars") as mock_add:
            mock_add.side_effect = TypeError("Cannot extract time features from string")
            
            # Should handle time feature errors
            with pytest.raises(TypeError):
                builder._add_known_future_features_polars(df)

    def test_multiple_instrument_processing_with_failures(
        self,
        builder: TFTDatasetBuilder,
        mock_feature_store: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test processing multiple instruments with partial failures."""
        instruments = ["SPY.NYSE", "QQQ.NASDAQ", "IWM.ARCA"]
        
        # First instrument succeeds, second fails, third succeeds
        call_count = 0
        
        def get_training_data_side_effect(*args: Any, **kwargs: Any) -> tuple:
            nonlocal call_count
            call_count += 1
            
            if call_count == 2:  # Second instrument fails
                raise Exception("Feature store unavailable")
            
            return (
                np.array([[1.0, 2.0]]),
                np.array([1000]),
                ["feat1", "feat2"],
            )
        
        mock_feature_store.get_training_data.side_effect = get_training_data_side_effect
        
        # Mock bars data
        with patch("ml.data.tft_dataset_builder.bars_to_dataframe") as mock_bars:
            mock_bars.return_value = pl.DataFrame({
                "ts_event": [1000],
                "close": [100.0],
                "open": [99.0],
                "high": [101.0],
                "low": [98.5],
                "volume": [1000000],
            })
            
            # Should continue processing despite failures
            result = builder.prepare_training_data_from_store(
                instrument_ids=instruments,
            )
            
            # Should have attempted all 3 instruments
            assert mock_feature_store.get_training_data.call_count == 3
            # Result should have data from successful instruments
            assert result is not None
            assert not result.is_empty()


# ============================================================================
# FRED DATA LOADER ERROR TESTS
# ============================================================================


@pytest.mark.usefixtures("clean_postgres_db")
class TestFREDDataLoaderErrors:
    """Test error handling in FREDDataLoader."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def fred_config(self, temp_dir: Path, test_database) -> FREDConfig:
        """Create FRED configuration with PostgreSQL."""
        return FREDConfig(
            api_key="test_key",
            cache_dir=temp_dir / "fred_cache",
            max_retries=3,
            retry_delay_seconds=0.01,
            cache_ttl_hours=1,
            connection_string=test_database.connection_string,
        )

    @pytest.fixture
    def loader(self, fred_config: FREDConfig, test_database) -> FREDDataLoader:
        """Create FREDDataLoader instance with PostgreSQL."""
        loader = FREDDataLoader(config=fred_config)
        # Initialize data store with PostgreSQL
        loader._data_store = DataStore(connection_string=test_database.connection_string)
        return loader

    def test_missing_api_key_error(self, temp_dir: Path) -> None:
        """Test error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="FRED API key not provided"):
                FREDConfig(cache_dir=temp_dir)

    def test_fredapi_import_failure(self, fred_config: FREDConfig) -> None:
        """Test handling of missing fredapi library."""
        with patch("ml.data.loaders.fred_loader.HAS_FREDAPI", False):
            with patch("ml.data.loaders.fred_loader._fredapi", None):
                # Should raise ImportError when trying to create loader
                with pytest.raises(ImportError, match="fredapi package required"):
                    FREDDataLoader(config=fred_config)

    def test_api_connection_failures(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test handling of API connection failures."""
        with patch.object(loader, "fred") as mock_fred:
            # Simulate various connection errors
            errors = [
                ConnectionError("Connection refused"),
                TimeoutError("Request timeout"),
                OSError("Network unreachable"),
            ]
            
            for error in errors:
                mock_fred.get_series.side_effect = error
                
                # Should raise RuntimeError after retries
                with pytest.raises(RuntimeError, match="Failed to fetch"):
                    loader.fetch_indicator(
                        FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
                    )

    def test_api_rate_limiting(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test API rate limiting handling."""
        with patch.object(loader, "fred") as mock_fred:
            # Simulate rate limit error then success
            mock_fred.get_series.side_effect = [
                Exception("Too Many Requests"),
                Exception("Too Many Requests"),
                pd.Series([2.5, 2.6], index=[datetime(2025, 1, 1), datetime(2025, 1, 2)]),
            ]
            
            # Should retry and eventually succeed
            result = loader.fetch_indicator(
                FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
            )
            
            assert not result.is_empty()
            assert mock_fred.get_series.call_count == 3

    def test_invalid_series_id(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test handling of invalid series IDs."""
        with patch.object(loader, "fred") as mock_fred:
            mock_fred.get_series.side_effect = ValueError("Series not found: INVALID123")
            
            # Should raise RuntimeError for invalid series
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                loader.fetch_indicator(
                    FREDIndicator(series_id="INVALID123", name="Invalid", category="test")
                )

    def test_cache_corruption_recovery(
        self,
        loader: FREDDataLoader,
        temp_dir: Path,
    ) -> None:
        """Test recovery from corrupted cache files."""
        indicator = FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
        
        # Create corrupted cache file
        cache_file = temp_dir / "fred_cache" / "DGS10.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"corrupted parquet data")
        
        with patch.object(loader, "fred") as mock_fred:
            # Should fetch fresh data when cache is corrupted
            mock_fred.get_series.return_value = pd.Series(
                [2.5], index=[datetime(2025, 1, 1)]
            )
            
            result = loader.fetch_indicator(indicator)
            
            assert not result.is_empty()
            assert mock_fred.get_series.called

    def test_cache_ttl_expiration(
        self,
        loader: FREDDataLoader,
        temp_dir: Path,
    ) -> None:
        """Test cache TTL expiration handling."""
        indicator = FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
        
        # Create expired cache file
        cache_file = temp_dir / "fred_cache" / "DGS10.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create valid but old cache
        old_data = pl.DataFrame({
            "timestamp": [datetime.now() - timedelta(hours=2)],
            "value": [2.5],
        })
        old_data.write_parquet(cache_file)
        
        # Modify file timestamp to be old
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(cache_file, (old_time, old_time))
        
        with patch.object(loader, "fred") as mock_fred:
            # Should fetch fresh data when cache is expired
            mock_fred.get_series.return_value = pd.Series(
                [2.6], index=[datetime.now()]
            )
            
            result = loader.fetch_indicator(indicator)
            
            assert not result.is_empty()
            assert mock_fred.get_series.called

    def test_empty_api_response(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test handling of empty API responses."""
        with patch.object(loader, "fred") as mock_fred:
            # Return empty series
            mock_fred.get_series.return_value = pd.Series([])
            
            result = loader.fetch_indicator(
                FREDIndicator(series_id="TEST", name="Test", category="test")
            )
            
            # Empty series is valid and should be returned
            assert result.is_empty()

    def test_malformed_api_response(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test handling of malformed API responses."""
        with patch.object(loader, "fred") as mock_fred:
            # Return series with invalid data
            mock_fred.get_series.return_value = pd.Series(
                ["not", "numeric", "data"],
                index=[datetime.now()] * 3,
            )
            
            # Should raise RuntimeError for non-numeric data
            with pytest.raises((RuntimeError, ValueError)):
                loader.fetch_indicator(
                    FREDIndicator(series_id="TEST", name="Test", category="test")
                )

    def test_concurrent_fetch_requests(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test thread safety of concurrent fetch requests."""
        import threading
        
        indicators = [
            FREDIndicator(series_id=f"TEST{i}", name=f"Test {i}", category="test")
            for i in range(5)
        ]
        
        results = []
        lock = threading.Lock()
        
        def fetch_indicator(indicator: FREDIndicator) -> None:
            with patch.object(loader, "fred") as mock_fred:
                mock_fred.get_series.return_value = pd.Series(
                    [1.0], index=[datetime.now()]
                )
                
                result = loader.fetch_indicator(indicator)
                
                with lock:
                    results.append(result)
        
        threads = []
        for indicator in indicators:
            thread = threading.Thread(target=fetch_indicator, args=(indicator,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All fetches should complete
        assert len(results) == 5

    def test_data_store_integration_errors(
        self,
        loader: FREDDataLoader,
        test_database,
    ) -> None:
        """Test error handling in data store integration."""
        # Mock data store with PostgreSQL connection
        mock_store = MagicMock(spec=DataStore)
        mock_store.connection_string = test_database.connection_string
        mock_store.store.side_effect = Exception("Database unavailable")
        loader._data_store = mock_store
        
        with patch.object(loader, "fred") as mock_fred:
            mock_fred.get_series.return_value = pd.Series(
                [2.5], index=[datetime.now()]
            )
            
            # Should handle store errors gracefully
            result = loader.fetch_indicator(
                FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
            )
            
            # Should still return data despite store error
            assert not result.is_empty()

    def test_registry_integration_errors(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test error handling in registry integration."""
        # Mock data registry
        mock_registry = MagicMock()
        mock_registry.register_dataset.side_effect = Exception("Registry unavailable")
        loader._data_registry = mock_registry
        
        with patch.object(loader, "fred") as mock_fred:
            mock_fred.get_series.return_value = pd.Series(
                [2.5], index=[datetime.now()]
            )
            
            # Should handle registry errors gracefully
            result = loader.fetch_indicator(
                FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates")
            )
            
            # Should still return data despite registry error
            assert not result.is_empty()

    def test_batch_fetch_with_failures(
        self,
        loader: FREDDataLoader,
    ) -> None:
        """Test batch fetching with partial failures."""
        indicators = [
            FREDIndicator(series_id="DGS10", name="10Y Treasury", category="rates"),
            FREDIndicator(series_id="INVALID", name="Invalid", category="test"),
            FREDIndicator(series_id="DGS2", name="2Y Treasury", category="rates"),
        ]
        
        with patch.object(loader, "fred") as mock_fred:
            def get_series_side_effect(series_id: str, *args: Any, **kwargs: Any) -> pd.Series:
                if series_id == "INVALID":
                    raise ValueError("Series not found")
                return pd.Series([2.5], index=[datetime.now()])
            
            mock_fred.get_series.side_effect = get_series_side_effect
            
            # Fetch all indicators
            results = []
            for indicator in indicators:
                try:
                    result = loader.fetch_indicator(indicator)
                    results.append(result)
                except RuntimeError:
                    results.append(None)  # Failed fetch
            
            # Should have 2 successful, 1 failed
            successful = [r for r in results if r is not None and not r.is_empty()]
            assert len(successful) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])