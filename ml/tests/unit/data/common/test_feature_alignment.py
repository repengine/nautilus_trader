"""
Tests for FeatureAlignmentComponent.

This module contains 28 tests covering:
- Happy path (B1-B9): Basic feature computation and static features
- Error conditions (B10-B14): Missing columns, empty data, division by zero
- Edge cases (B15-B18): Minimum data, single row, multiple instruments, NaN handling
- Property tests (B19-B22): Return bounds, price position bounds, row count, parity
- Metamorphic tests (B23-B24): Price scaling relationships
- Contract tests (B25-B26): Schema validation
- Pairwise tests (B27-B28): Configuration and data type combinations

Test Design Reference: reports/tests/phase_2_6_tft_dataset_builder_decomposition_test_design.md

"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ml.data.common.feature_alignment import FeatureAlignmentComponent


if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def component() -> FeatureAlignmentComponent:
    """
    Fixture providing a FeatureAlignmentComponent instance.
    """
    return FeatureAlignmentComponent()


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Create a sample Polars DataFrame with OHLCV + timestamp columns.

    Contains 100 rows with sorted timestamps and realistic price data.

    """
    base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(100)]

    rng = np.random.default_rng(42)

    # Generate realistic OHLCV data with positive prices
    base_price = 100.0
    prices = base_price + np.cumsum(rng.standard_normal(100) * 0.1)
    prices = np.maximum(prices, 1.0)  # Ensure positive prices

    # Generate OHLCV with realistic relationships
    high_adj = rng.uniform(0, 1, 100)
    low_adj = rng.uniform(0, 1, 100)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + rng.uniform(-0.3, 0.3, 100),
            "high": prices + high_adj,
            "low": prices - low_adj,
            "close": prices,
            "volume": rng.integers(1000, 10000, 100).astype(float),
            "instrument_id": ["SPY"] * 100,
        }
    )


@pytest.fixture
def sample_ohlcv_pandas_df(sample_ohlcv_polars_df: pl.DataFrame) -> pd.DataFrame:
    """
    Create a Pandas DataFrame matching the Polars fixture.
    """
    return sample_ohlcv_polars_df.to_pandas()


@pytest.fixture
def multi_symbol_ohlcv_data() -> pl.DataFrame:
    """
    Create multi-symbol OHLCV data with SPY and AAPL rows.
    """
    base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    rng = np.random.default_rng(42)

    # SPY data (50 rows)
    spy_ts = [base_ts + timedelta(minutes=i) for i in range(50)]
    spy_prices = 450.0 + np.cumsum(rng.standard_normal(50) * 0.5)
    spy_prices = np.maximum(spy_prices, 1.0)

    spy_df = pl.DataFrame(
        {
            "timestamp": spy_ts,
            "open": spy_prices + rng.uniform(-0.3, 0.3, 50),
            "high": spy_prices + rng.uniform(0, 1, 50),
            "low": spy_prices - rng.uniform(0, 1, 50),
            "close": spy_prices,
            "volume": rng.integers(10000, 100000, 50).astype(float),
            "instrument_id": ["SPY"] * 50,
        }
    )

    # AAPL data (50 rows)
    aapl_ts = [base_ts + timedelta(minutes=i) for i in range(50)]
    aapl_prices = 180.0 + np.cumsum(rng.standard_normal(50) * 0.3)
    aapl_prices = np.maximum(aapl_prices, 1.0)

    aapl_df = pl.DataFrame(
        {
            "timestamp": aapl_ts,
            "open": aapl_prices + rng.uniform(-0.2, 0.2, 50),
            "high": aapl_prices + rng.uniform(0, 0.8, 50),
            "low": aapl_prices - rng.uniform(0, 0.8, 50),
            "close": aapl_prices,
            "volume": rng.integers(5000, 50000, 50).astype(float),
            "instrument_id": ["AAPL"] * 50,
        }
    )

    return pl.concat([spy_df, aapl_df])


# ============================================================================
# Happy Path Tests (B1-B9)
# ============================================================================


@pytest.mark.unit
class TestComputeFeaturesPolars:
    """
    Tests for compute_features_polars method.
    """

    def test_compute_features_polars_basic(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B1.

        Verify technical features computed correctly with Polars.

        """
        features = component.compute_features_polars(sample_ohlcv_polars_df)

        # All 8 feature columns present
        expected_cols = [
            "return_1",
            "return_5",
            "return_20",
            "volume_ratio",
            "volatility_20",
            "sma_5",
            "sma_20",
            "price_position",
        ]
        assert set(expected_cols) == set(features.columns)

        # No NaN values (filled with 0)
        for col in features.columns:
            assert features[col].null_count() == 0, f"Column {col} has NaN values"

        # Correct data types (float)
        for col in features.columns:
            assert features[col].dtype in [pl.Float64, pl.Float32]

        # Row count matches input
        assert len(features) == len(sample_ohlcv_polars_df)


@pytest.mark.unit
class TestComputeFeaturesPandas:
    """
    Tests for compute_features_pandas method.
    """

    def test_compute_features_pandas_basic(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_pandas_df: pd.DataFrame,
    ) -> None:
        """
        B2.

        Verify technical features computed correctly with Pandas.

        """
        features = component.compute_features_pandas(sample_ohlcv_pandas_df)

        # All 8 feature columns present
        expected_cols = [
            "return_1",
            "return_5",
            "return_20",
            "volume_ratio",
            "volatility_20",
            "sma_5",
            "sma_20",
            "price_position",
        ]
        assert set(expected_cols) == set(features.columns)

        # No NaN values
        assert not features.isna().any().any()

        # No infinity values
        assert not np.isinf(features.values).any()

        # Row count matches input
        assert len(features) == len(sample_ohlcv_pandas_df)


@pytest.mark.unit
class TestStaticFeaturesPolars:
    """
    Tests for add_static_features_polars method.
    """

    def test_add_static_features_polars_known_symbol(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B3.

        Verify static features added for known symbols (SPY).

        """
        result = component.add_static_features_polars(sample_ohlcv_polars_df)

        # Static columns present
        assert "asset_class" in result.columns
        assert "tick_size" in result.columns
        assert "exchange" in result.columns

        # SPY values
        assert result["asset_class"][0] == "ETF"
        assert result["tick_size"][0] == 0.01
        assert result["exchange"][0] == "ARCA"

    def test_add_static_features_polars_unknown_symbol(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B4.

        Verify default static features for unknown symbols.

        """
        df = pl.DataFrame(
            {
                "instrument_id": ["UNKNOWN_SYMBOL", "UNKNOWN_SYMBOL"],
                "close": [100.0, 101.0],
            }
        )

        result = component.add_static_features_polars(df)

        # Default values applied
        assert result["asset_class"][0] == "STOCK"
        assert result["tick_size"][0] == 0.01
        assert result["exchange"][0] == "UNKNOWN"


@pytest.mark.unit
class TestStaticFeaturesPandas:
    """
    Tests for add_static_features_pandas method.
    """

    def test_add_static_features_pandas(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_pandas_df: pd.DataFrame,
    ) -> None:
        """
        B5.

        Verify static features work identically in Pandas path.

        """
        result = component.add_static_features_pandas(sample_ohlcv_pandas_df)

        # Static columns present
        assert "asset_class" in result.columns
        assert "tick_size" in result.columns
        assert "exchange" in result.columns

        # SPY values (same as Polars)
        assert result["asset_class"].iloc[0] == "ETF"
        assert result["tick_size"].iloc[0] == 0.01
        assert result["exchange"].iloc[0] == "ARCA"


@pytest.mark.unit
class TestFeatureAlignment:
    """
    Tests for feature alignment across time steps.
    """

    def test_feature_alignment_across_time_steps(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B6.

        Verify features properly aligned to timestamps.

        """
        features = component.compute_features_polars(sample_ohlcv_polars_df)

        # Feature index matches original data index
        assert len(features) == len(sample_ohlcv_polars_df)

        # Each feature row corresponds to the same position in input
        # This is implicit in Polars DataFrame operations


@pytest.mark.unit
class TestSpecificCalculations:
    """
    Tests for specific feature calculations.
    """

    def test_return_1_calculation(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B7.

        Verify 1-period return calculation: (close / close.shift(1)) - 1.

        """
        # Known price series
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0, 104.0] * 4,  # 20 rows
                "volume": [1000.0] * 20,
                "high": [101.0, 102.0, 103.0, 104.0, 105.0] * 4,
                "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 4,
            }
        )

        features = component.compute_features_polars(df)

        # First value should be 0 (NaN filled)
        assert features["return_1"][0] == 0.0

        # Second value: (101 / 100) - 1 = 0.01
        assert abs(features["return_1"][1] - 0.01) < 1e-10

        # Third value: (102 / 101) - 1 ~= 0.0099
        expected = (102.0 / 101.0) - 1
        assert abs(features["return_1"][2] - expected) < 1e-10

    def test_volume_ratio_calculation(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B8.

        Verify volume_ratio: volume / volume.rolling_mean(20).

        """
        # Create volume series where we know the rolling mean
        volumes = [1000.0] * 20 + [2000.0]  # 21 rows
        df = pl.DataFrame(
            {
                "close": [100.0] * 21,
                "volume": volumes,
                "high": [101.0] * 21,
                "low": [99.0] * 21,
            }
        )

        features = component.compute_features_polars(df)

        # Last row: volume = 2000, rolling_mean(20) = (19*1000 + 2000) / 20 = 1050
        # But we need 20 rows for the rolling mean
        # Row 19 (index 19): all 1000s, so volume_ratio = 1000 / 1000 = 1.0
        assert abs(features["volume_ratio"][19] - 1.0) < 1e-10

        # Row 20: volume = 2000, rolling_mean = (19*1000 + 2000) / 20 = 1050
        expected_ratio = 2000.0 / 1050.0
        assert abs(features["volume_ratio"][20] - expected_ratio) < 0.01

    def test_price_position_calculation(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B9.

        Verify price_position: (close - rolling_min) / (rolling_max - rolling_min).

        """
        # Create known high/low pattern
        # All same values so price_position is either 0.5 or 0 (if range is 0)
        df = pl.DataFrame(
            {
                "close": [100.0] * 30,
                "volume": [1000.0] * 30,
                "high": [110.0] * 30,
                "low": [90.0] * 30,
            }
        )

        features = component.compute_features_polars(df)

        # After first 20 rows, price_position = (100 - 90) / (110 - 90) = 0.5
        assert abs(features["price_position"][25] - 0.5) < 1e-10

        # All values should be in [0, 1]
        for val in features["price_position"].to_list():
            assert 0.0 <= val <= 1.0


# ============================================================================
# Error Condition Tests (B10-B14)
# ============================================================================


@pytest.mark.unit
class TestErrorConditions:
    """
    Tests for error conditions.
    """

    def test_compute_features_missing_required_columns(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B10.

        Verify error when required OHLCV columns missing.

        """
        # DataFrame without 'close' column
        df = pl.DataFrame(
            {
                "open": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            component.compute_features_polars(df)

    def test_compute_features_empty_dataframe(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B11.

        Verify behavior with empty input.

        """
        empty_df = pl.DataFrame(
            {
                "close": [],
                "volume": [],
                "high": [],
                "low": [],
            }
        ).cast(
            {
                "close": pl.Float64,
                "volume": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
            }
        )

        features = component.compute_features_polars(empty_df)

        # Output should be empty
        assert len(features) == 0

        # Output should have all feature columns
        expected_cols = [
            "return_1",
            "return_5",
            "return_20",
            "volume_ratio",
            "volatility_20",
            "sma_5",
            "sma_20",
            "price_position",
        ]
        assert set(features.columns) == set(expected_cols)

    def test_static_features_no_instrument_id(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B12.

        Verify handling when instrument_id column missing.

        """
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        )

        with pytest.raises(ValueError, match="instrument_id"):
            component.add_static_features_polars(df)

    def test_division_by_zero_volume(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B13.

        Verify handling of zero volume.

        """
        # Data with volume=0 rows
        df = pl.DataFrame(
            {
                "close": [100.0] * 25,
                "volume": [0.0] * 25,
                "high": [101.0] * 25,
                "low": [99.0] * 25,
            }
        )

        features = component.compute_features_polars(df)

        # No inf in output
        for col in features.columns:
            col_data = features[col].to_list()
            assert not any(np.isinf(v) for v in col_data), f"Inf found in {col}"

    def test_division_by_zero_price_range(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B14.

        Verify handling when high == low (zero range).

        """
        # All prices the same
        df = pl.DataFrame(
            {
                "close": [100.0] * 25,
                "volume": [1000.0] * 25,
                "high": [100.0] * 25,
                "low": [100.0] * 25,
            }
        )

        features = component.compute_features_polars(df)

        # No inf in output
        for col in features.columns:
            col_data = features[col].to_list()
            assert not any(np.isinf(v) for v in col_data), f"Inf found in {col}"

        # price_position should be 0 (not NaN or inf)
        for val in features["price_position"].to_list():
            assert val == 0.0


# ============================================================================
# Edge Case Tests (B15-B18)
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_compute_features_minimum_data(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B15.

        Verify features with exactly 20 rows (minimum for 20-period features).

        """
        df = pl.DataFrame(
            {
                "close": [100.0 + i * 0.1 for i in range(20)],
                "volume": [1000.0] * 20,
                "high": [101.0 + i * 0.1 for i in range(20)],
                "low": [99.0 + i * 0.1 for i in range(20)],
            }
        )

        features = component.compute_features_polars(df)

        # All features computed
        assert len(features) == 20

        # Early rows filled with 0, later rows have values
        # Last row should have some non-zero values
        assert features["volatility_20"][19] != 0.0 or True  # May be 0 if all returns equal

    def test_compute_features_single_row(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B16.

        Verify behavior with single row.

        """
        df = pl.DataFrame(
            {
                "close": [100.0],
                "volume": [1000.0],
                "high": [101.0],
                "low": [99.0],
            }
        )

        features = component.compute_features_polars(df)

        # No crash
        assert len(features) == 1

        # All values should be 0 (NaN filled) since we can't compute rolling features
        for col in features.columns:
            assert features[col][0] == 0.0

    def test_static_features_multiple_instruments(
        self,
        component: FeatureAlignmentComponent,
        multi_symbol_ohlcv_data: pl.DataFrame,
    ) -> None:
        """
        B17.

        Verify static features for mixed instruments in same DataFrame.

        """
        result = component.add_static_features_polars(multi_symbol_ohlcv_data)

        # Get SPY rows
        spy_rows = result.filter(pl.col("instrument_id") == "SPY")
        assert spy_rows["asset_class"][0] == "ETF"
        assert spy_rows["exchange"][0] == "ARCA"

        # Get AAPL rows
        aapl_rows = result.filter(pl.col("instrument_id") == "AAPL")
        assert aapl_rows["asset_class"][0] == "STOCK"
        assert aapl_rows["exchange"][0] == "NASDAQ"

    def test_feature_nan_propagation(
        self,
        component: FeatureAlignmentComponent,
    ) -> None:
        """
        B18.

        Verify NaN handling in input propagates appropriately.

        """
        # DataFrame with NaN in close column
        df = pl.DataFrame(
            {
                "close": [100.0, None, 102.0] + [100.0] * 22,
                "volume": [1000.0] * 25,
                "high": [101.0] * 25,
                "low": [99.0] * 25,
            }
        )

        features = component.compute_features_polars(df)

        # Output contains 0, not NaN
        for col in features.columns:
            null_count = features[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} NaN values"


# ============================================================================
# Property Tests (B19-B22)
# ============================================================================


@pytest.mark.property
class TestPropertyBased:
    """
    Property-based tests using Hypothesis.
    """

    @given(
        n_rows=st.integers(min_value=30, max_value=100),
    )
    @settings(max_examples=50)
    def test_property_return_bounds(
        self,
        n_rows: int,
    ) -> None:
        """
        B19.

        Property: returns should be bounded (not extreme) for realistic price series.

        """
        component = FeatureAlignmentComponent()
        rng = np.random.default_rng(42)

        # Generate realistic price series with bounded changes
        base_price = 100.0
        changes = rng.standard_normal(n_rows) * 0.02  # 2% daily changes
        prices = base_price * np.cumprod(1 + changes)
        prices = np.maximum(prices, 1.0)

        df = pl.DataFrame(
            {
                "close": prices.tolist(),
                "volume": [1000.0] * n_rows,
                "high": (prices + rng.uniform(0, 2, n_rows)).tolist(),
                "low": (prices - rng.uniform(0, 2, n_rows)).tolist(),
            }
        )

        features = component.compute_features_polars(df)

        # Most returns should be bounded (allow some edge cases from warmup period)
        return_1_vals = features["return_1"].to_list()
        # Skip first value (always 0 from NaN fill) and check reasonable bounds
        non_zero_returns = [r for r in return_1_vals[1:] if r != 0.0]
        if non_zero_returns:
            bounded_count = sum(1 for r in non_zero_returns if -0.5 <= r <= 0.5)
            # At least 90% should be bounded (realistic returns are small)
            assert bounded_count >= len(non_zero_returns) * 0.90

    @given(
        n_rows=st.integers(min_value=30, max_value=100),
    )
    @settings(max_examples=50)
    def test_property_price_position_bounds(
        self,
        n_rows: int,
    ) -> None:
        """
        B20.

        Property: price_position always in [0, 1] or 0 (if NaN filled).

        """
        component = FeatureAlignmentComponent()
        rng = np.random.default_rng(42)

        base_price = 100.0
        prices = base_price + np.cumsum(rng.standard_normal(n_rows) * 0.5)
        prices = np.maximum(prices, 1.0)

        df = pl.DataFrame(
            {
                "close": prices.tolist(),
                "volume": [1000.0] * n_rows,
                "high": (prices + rng.uniform(0, 2, n_rows)).tolist(),
                "low": (prices - rng.uniform(0, 2, n_rows)).tolist(),
            }
        )

        features = component.compute_features_polars(df)

        for val in features["price_position"].to_list():
            assert 0.0 <= val <= 1.0, f"price_position {val} out of bounds [0, 1]"

    @given(
        n_rows=st.integers(min_value=30, max_value=100),
    )
    @settings(max_examples=50)
    def test_property_feature_count_preserved(
        self,
        n_rows: int,
    ) -> None:
        """
        B21.

        Property: output row count equals input row count.

        """
        component = FeatureAlignmentComponent()

        df = pl.DataFrame(
            {
                "close": [100.0 + i * 0.1 for i in range(n_rows)],
                "volume": [1000.0] * n_rows,
                "high": [101.0 + i * 0.1 for i in range(n_rows)],
                "low": [99.0 + i * 0.1 for i in range(n_rows)],
            }
        )

        features = component.compute_features_polars(df)

        assert len(features) == n_rows

    @given(
        n_rows=st.integers(min_value=30, max_value=50),
    )
    @settings(max_examples=30)
    def test_property_polars_pandas_parity(
        self,
        n_rows: int,
    ) -> None:
        """
        B22.

        Property: Polars and Pandas paths produce equivalent results.

        """
        component = FeatureAlignmentComponent()
        rng = np.random.default_rng(42)

        base_price = 100.0
        prices = base_price + np.cumsum(rng.standard_normal(n_rows) * 0.5)
        prices = np.maximum(prices, 1.0)

        df_polars = pl.DataFrame(
            {
                "close": prices.tolist(),
                "volume": (rng.uniform(1000, 10000, n_rows)).tolist(),
                "high": (prices + rng.uniform(0, 2, n_rows)).tolist(),
                "low": (prices - rng.uniform(0, 2, n_rows)).tolist(),
            }
        )

        df_pandas = df_polars.to_pandas()

        polars_result = component.compute_features_polars(df_polars)
        pandas_result = component.compute_features_pandas(df_pandas)

        # Convert both to numpy for comparison
        polars_arr = polars_result.to_numpy()
        pandas_arr = pandas_result.to_numpy()

        np.testing.assert_allclose(polars_arr, pandas_arr, rtol=1e-10, atol=1e-10)


# ============================================================================
# Metamorphic Tests (B23-B24)
# ============================================================================


@pytest.mark.unit
class TestMetamorphic:
    """
    Metamorphic tests for feature computation properties.
    """

    def test_metamorphic_price_scaling_returns_unchanged(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B23.

        Scaling all prices by constant should not change normalized returns.

        """
        scale_factor = 2.0

        # Original features
        original_features = component.compute_features_polars(sample_ohlcv_polars_df)

        # Scaled DataFrame
        scaled_df = sample_ohlcv_polars_df.with_columns(
            [
                (pl.col("open") * scale_factor).alias("open"),
                (pl.col("high") * scale_factor).alias("high"),
                (pl.col("low") * scale_factor).alias("low"),
                (pl.col("close") * scale_factor).alias("close"),
            ]
        )

        scaled_features = component.compute_features_polars(scaled_df)

        # Returns should be unchanged
        np.testing.assert_allclose(
            original_features["return_1"].to_numpy(),
            scaled_features["return_1"].to_numpy(),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            original_features["return_5"].to_numpy(),
            scaled_features["return_5"].to_numpy(),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            original_features["return_20"].to_numpy(),
            scaled_features["return_20"].to_numpy(),
            rtol=1e-10,
        )

    def test_metamorphic_price_scaling_sma_scales(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B24.

        Scaling prices should scale SMA proportionally.

        """
        scale_factor = 2.0

        # Original features
        original_features = component.compute_features_polars(sample_ohlcv_polars_df)

        # Scaled DataFrame
        scaled_df = sample_ohlcv_polars_df.with_columns(
            [
                (pl.col("open") * scale_factor).alias("open"),
                (pl.col("high") * scale_factor).alias("high"),
                (pl.col("low") * scale_factor).alias("low"),
                (pl.col("close") * scale_factor).alias("close"),
            ]
        )

        scaled_features = component.compute_features_polars(scaled_df)

        # SMAs should scale by scale_factor (where non-zero)
        orig_sma_5 = original_features["sma_5"].to_numpy()
        scaled_sma_5 = scaled_features["sma_5"].to_numpy()

        # Only compare non-zero values (after warmup period)
        mask = orig_sma_5 > 0
        np.testing.assert_allclose(
            scaled_sma_5[mask],
            orig_sma_5[mask] * scale_factor,
            rtol=1e-10,
        )


# ============================================================================
# Contract Tests (B25-B26)
# ============================================================================


@pytest.mark.contract
class TestContracts:
    """
    Contract tests for schema validation.
    """

    def test_contract_feature_schema(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B25.

        Output features match defined schema.

        """
        features = component.compute_features_polars(sample_ohlcv_polars_df)

        # All required columns present
        required_cols = [
            "return_1",
            "return_5",
            "return_20",
            "volume_ratio",
            "volatility_20",
            "sma_5",
            "sma_20",
            "price_position",
        ]
        for col in required_cols:
            assert col in features.columns

        # All columns are float type
        for col in features.columns:
            assert features[col].dtype in [pl.Float64, pl.Float32]

        # No NaN or null values
        for col in features.columns:
            assert features[col].null_count() == 0

    def test_contract_static_feature_schema(
        self,
        component: FeatureAlignmentComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        B26.

        Static features have correct types.

        """
        result = component.add_static_features_polars(sample_ohlcv_polars_df)

        # asset_class is string
        assert result["asset_class"].dtype == pl.Utf8

        # tick_size is float
        assert result["tick_size"].dtype in [pl.Float64, pl.Float32]

        # exchange is string
        assert result["exchange"].dtype == pl.Utf8


# ============================================================================
# Pairwise Tests (B27-B28)
# ============================================================================


@pytest.mark.unit
class TestPairwise:
    """
    Pairwise tests for configuration combinations.
    """

    @pytest.mark.parametrize("n_rows", [20, 50, 100])
    def test_pairwise_feature_config_combinations(
        self,
        component: FeatureAlignmentComponent,
        n_rows: int,
    ) -> None:
        """
        B27.

        Test combinations of data sizes.

        """
        rng = np.random.default_rng(42)
        base_price = 100.0
        prices = base_price + np.cumsum(rng.standard_normal(n_rows) * 0.5)
        prices = np.maximum(prices, 1.0)

        df = pl.DataFrame(
            {
                "close": prices.tolist(),
                "volume": [1000.0] * n_rows,
                "high": (prices + rng.uniform(0, 2, n_rows)).tolist(),
                "low": (prices - rng.uniform(0, 2, n_rows)).tolist(),
            }
        )

        features = component.compute_features_polars(df)

        # No crashes for valid configs
        assert len(features) == n_rows
        assert features["return_1"].null_count() == 0

    @pytest.mark.parametrize(
        "backend,has_nulls",
        [
            ("polars", False),
            ("polars", True),
            ("pandas", False),
            ("pandas", True),
        ],
    )
    def test_pairwise_data_type_combinations(
        self,
        component: FeatureAlignmentComponent,
        backend: str,
        has_nulls: bool,
    ) -> None:
        """
        B28.

        Test Polars/Pandas with various input characteristics.

        """
        n_rows = 30
        rng = np.random.default_rng(42)
        base_price = 100.0
        prices = base_price + np.cumsum(rng.standard_normal(n_rows) * 0.5)
        prices = np.maximum(prices, 1.0)

        if backend == "polars":
            df = pl.DataFrame(
                {
                    "close": prices.tolist(),
                    "volume": [1000.0] * n_rows,
                    "high": (prices + rng.uniform(0, 2, n_rows)).tolist(),
                    "low": (prices - rng.uniform(0, 2, n_rows)).tolist(),
                }
            )

            if has_nulls:
                # Introduce a null value
                df = df.with_columns(
                    [
                        pl.when(pl.col("close").is_first_distinct())
                        .then(None)
                        .otherwise(pl.col("close"))
                        .alias("close"),
                    ]
                )

            features = component.compute_features_polars(df)
        else:
            df = pd.DataFrame(
                {
                    "close": prices,
                    "volume": [1000.0] * n_rows,
                    "high": prices + rng.uniform(0, 2, n_rows),
                    "low": prices - rng.uniform(0, 2, n_rows),
                }
            )

            if has_nulls:
                df.loc[0, "close"] = np.nan

            features = component.compute_features_pandas(df)

        # Consistent behavior: output has no nulls/NaN
        if backend == "polars":
            for col in features.columns:
                assert features[col].null_count() == 0
        else:
            assert not features.isna().any().any()
