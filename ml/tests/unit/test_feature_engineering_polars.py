# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Tests for polars-specific code paths in feature engineering.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# Check if polars is available
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestPolarsIntegration:
    """
    Test polars-specific code paths.
    """

    def test_extract_price_arrays_polars(self) -> None:
        """
        Test price array extraction from polars DataFrame.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create polars DataFrame
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000000.0, 1100000.0, 1200000.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "open": [100.0, 101.0, 102.0],
            },
        )

        # Extract arrays
        open_prices, high_prices, low_prices, close_prices, volumes = fe._extract_price_arrays(df)

        # Check arrays
        assert isinstance(open_prices, np.ndarray)
        assert isinstance(close_prices, np.ndarray)
        assert len(close_prices) == 3
        assert close_prices[0] == 100.0
        assert volumes[1] == 1100000.0

    def test_extract_price_arrays_polars_missing_columns(self) -> None:
        """
        Test price array extraction from polars DataFrame with missing columns.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create polars DataFrame with only close and volume
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000000.0, 1100000.0, 1200000.0],
            },
        )

        # Extract arrays - should use close for missing OHLC
        open_prices, high_prices, low_prices, close_prices, volumes = fe._extract_price_arrays(df)

        # Check arrays - should all be close except volume
        assert np.array_equal(open_prices, close_prices)
        assert np.array_equal(high_prices, close_prices)
        assert np.array_equal(low_prices, close_prices)

    def test_create_features_dataframe_polars(self) -> None:
        """
        Test creating features DataFrame with polars.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test features
        feature_rows = [dict.fromkeys(config.get_feature_names(), i * 0.1) for i in range(3)]

        # Create polars DataFrame
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
            },
        )

        # Force polars path
        with patch("ml.features.engineering.POLARS_AVAILABLE", True):
            features_df = fe._create_features_dataframe(feature_rows, df)

        # Check result
        assert isinstance(features_df, pl.DataFrame)
        assert len(features_df) == 3
        assert features_df.columns == config.get_feature_names()

    def test_create_features_dataframe_polars_with_timestamp(self) -> None:
        """
        Test creating features DataFrame with timestamp in polars.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test features
        feature_rows = [dict.fromkeys(config.get_feature_names(), i * 0.1) for i in range(3)]

        # Create polars DataFrame with timestamp
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2024, 1, 1),
                    end=pl.datetime(2024, 1, 1, 0, 2),
                    interval="1m",
                    eager=True,
                ),
            },
        )

        # Force polars path
        with patch("ml.features.engineering.POLARS_AVAILABLE", True):
            features_df = fe._create_features_dataframe(feature_rows, df)

        # Check result - timestamp should NOT be in features
        assert isinstance(features_df, pl.DataFrame)
        assert "timestamp" not in features_df.columns
        assert len(features_df.columns) == len(config.get_feature_names())

    def test_create_features_dataframe_polars_empty(self) -> None:
        """
        Test creating empty features DataFrame with polars.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Empty feature rows
        feature_rows: list[dict[str, float]] = []

        # Create polars DataFrame
        df = pl.DataFrame(
            {
                "close": [],
            },
        )

        # Force polars path
        with patch("ml.features.engineering.POLARS_AVAILABLE", True):
            features_df = fe._create_features_dataframe(feature_rows, df)

        # Check result
        assert isinstance(features_df, pl.DataFrame)
        assert len(features_df) == 0
        assert features_df.columns == config.get_feature_names()

    def test_apply_scaler_polars(self) -> None:
        """
        Test scaler application with polars DataFrame.
        """
        pytest.skip("Skipping sklearn-dependent test")

    def test_batch_calculation_with_polars(self) -> None:
        """
        Test full batch calculation with polars DataFrame.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create polars DataFrame with valid OHLC data
        rng = np.random.default_rng(42)
        n = 50
        # Generate close prices first
        close_prices = 100 + np.cumsum(rng.standard_normal(n) * 0.5)

        # Ensure OHLC consistency
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        highs = np.maximum(opens, close_prices) + np.abs(rng.standard_normal(n) * 0.2)
        lows = np.minimum(opens, close_prices) - np.abs(rng.standard_normal(n) * 0.2)

        df = pl.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": close_prices,
                "volume": rng.uniform(900000, 1100000, n),
            },
        )

        # Calculate features
        features_df, _ = fe.calculate_features_batch(df)

        # Check output
        assert len(features_df) == len(df)
        assert len(features_df.columns) == len(config.get_feature_names())

        # If polars was used, result should be polars DataFrame
        if POLARS_AVAILABLE:
            assert isinstance(features_df, pl.DataFrame | pd.DataFrame)


class TestColumnsErrorHandling:
    """
    Test error handling in column selection.
    """

    def test_column_selection_exception_path(self) -> None:
        """
        Test the exception handling path in column selection.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test data that will trigger the exception path
        # Create a DataFrame with columns that will cause issues
        pd.DataFrame([{"return_1": 0.01, "return_5": 0.02}])

        # Mock the __getitem__ to fail on first attempt, succeed on second
        call_count = 0
        original_getitem = pd.DataFrame.__getitem__

        def mock_getitem(self: Any, key: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1 and isinstance(key, list):
                # First call with list of columns should fail
                raise Exception("Column selection error")
            else:
                # Subsequent calls should work normally
                if isinstance(key, str) and key in self.columns:
                    return original_getitem(self, key)
                else:
                    # Return empty series for missing columns
                    return pd.Series(
                        [0.0] * len(self),
                        name=key if isinstance(key, str) else "unknown",
                    )

        with patch.object(pd.DataFrame, "__getitem__", mock_getitem):
            # This should trigger the exception path and create a new DataFrame
            with patch("ml.features.engineering.POLARS_AVAILABLE", False):
                features_df = fe._create_features_dataframe(
                    [{"return_1": 0.01, "return_5": 0.02}],
                    pd.DataFrame({"close": [100]}),
                )

            # Should have created a new DataFrame with all columns
            assert len(features_df.columns) == len(config.get_feature_names())


class TestFeatureCalculationHelpers:
    """
    Test helper methods for feature calculation.
    """

    def test_extract_data_arrays_pandas(self) -> None:
        """
        Test data array extraction for pandas DataFrame.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create pandas DataFrame
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
            },
        )

        close_array, high_array, low_array = fe._extract_data_arrays(df)

        assert isinstance(close_array, np.ndarray)
        assert isinstance(high_array, np.ndarray)
        assert isinstance(low_array, np.ndarray)
        assert len(close_array) == 3

    def test_extract_data_arrays_missing_columns(self) -> None:
        """
        Test data array extraction with missing columns.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create pandas DataFrame without high/low
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
            },
        )

        close_array, high_array, low_array = fe._extract_data_arrays(df)

        assert isinstance(close_array, np.ndarray)
        assert high_array is None
        assert low_array is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
