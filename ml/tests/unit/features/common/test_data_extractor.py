"""
Unit tests for DataExtractor component.

Tests the 6 methods extracted from FeatureEngineer:
1. extract_price_arrays - OHLCV extraction with fallback
2. extract_data_arrays - Close/High/Low extraction
3. extract_bid_ask_data - L2 quote data extraction
4. extract_trade_data - Trade tick data extraction
5. _ensure_float_array - Type conversion helper
6. _detect_dataframe_type - DataFrame type detection helper

Total: 22 unit tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


# ==================== 1. extract_price_arrays Tests (6 tests) ====================


def test_extract_price_arrays_from_pandas_full_ohlcv(
    data_extractor,
    pandas_ohlcv_dataframe,
):
    """
    Test extraction of complete OHLCV data from Pandas DataFrame.

    Verifies that all 5 arrays (open, high, low, close, volume) are extracted
    correctly with proper shape and dtype.
    """
    result = data_extractor.extract_price_arrays(pandas_ohlcv_dataframe)

    # Check we got 5 arrays
    assert len(result) == 5, "Should return 5 arrays (open, high, low, close, volume)"

    # Check all are numpy arrays
    assert all(isinstance(arr, np.ndarray) for arr in result), "All should be numpy arrays"

    # Check all are float64
    assert all(
        arr.dtype == np.float64 for arr in result
    ), "All arrays should have dtype np.float64"

    # Check all have same shape (100 rows)
    assert all(arr.shape == (100,) for arr in result), "All arrays should have shape (100,)"

    # Check close prices match input
    close_arr = result[3]
    np.testing.assert_allclose(
        close_arr,
        pandas_ohlcv_dataframe["close"].values,
        rtol=1e-10,
    )


def test_extract_price_arrays_from_polars_full_ohlcv(
    data_extractor,
    pandas_ohlcv_dataframe,
    polars_ohlcv_dataframe,
):
    """
    Test extraction from Polars DataFrame matches Pandas extraction.

    Verifies that Polars and Pandas DataFrames with identical data produce
    numerically identical results.
    """
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")

    # Extract from both formats
    pandas_result = data_extractor.extract_price_arrays(pandas_ohlcv_dataframe)
    polars_result = data_extractor.extract_price_arrays(polars_ohlcv_dataframe)

    # Results must be numerically identical
    for pd_arr, pl_arr in zip(pandas_result, polars_result):
        np.testing.assert_allclose(pd_arr, pl_arr, rtol=1e-10)


def test_extract_price_arrays_missing_high_low_uses_close(
    data_extractor,
    minimal_close_volume_dataframe,
):
    """
    Test fallback when high/low columns are missing.

    When high/low are absent, they should fallback to close prices.
    """
    result = data_extractor.extract_price_arrays(minimal_close_volume_dataframe)
    open_arr, high_arr, low_arr, close_arr, volume_arr = result

    # Fallback: open, high, low all equal to close
    np.testing.assert_allclose(open_arr, close_arr, rtol=1e-10)
    np.testing.assert_allclose(high_arr, close_arr, rtol=1e-10)
    np.testing.assert_allclose(low_arr, close_arr, rtol=1e-10)

    # Volume should be extracted correctly
    assert volume_arr.shape == (50,)
    assert np.allclose(volume_arr, 5000.0)


def test_extract_price_arrays_missing_volume_uses_zeros(
    data_extractor,
    ohlc_no_volume_dataframe,
):
    """
    Test fallback when volume column is missing.

    When volume is absent, it should fallback to zeros.
    """
    result = data_extractor.extract_price_arrays(ohlc_no_volume_dataframe)
    open_arr, high_arr, low_arr, close_arr, volume_arr = result

    # OHLC should be extracted correctly
    assert open_arr.shape == (75,)
    assert high_arr.shape == (75,)
    assert low_arr.shape == (75,)
    assert close_arr.shape == (75,)

    # Volume should be all zeros
    np.testing.assert_allclose(volume_arr, np.zeros(75), rtol=1e-10)


def test_extract_price_arrays_single_row(
    data_extractor,
    single_row_ohlcv_dataframe,
):
    """
    Test extraction works for single-row DataFrame.

    Edge case: DataFrame with only 1 row should produce arrays of shape (1,).
    """
    result = data_extractor.extract_price_arrays(single_row_ohlcv_dataframe)

    # All arrays should have shape (1,)
    assert all(arr.shape == (1,) for arr in result), "All arrays should have shape (1,)"

    # Check values are correct
    open_arr, high_arr, low_arr, close_arr, volume_arr = result
    assert open_arr[0] == 100.0
    assert high_arr[0] == 101.0
    assert low_arr[0] == 99.0
    assert close_arr[0] == 100.5
    assert volume_arr[0] == 10000.0


def test_extract_price_arrays_empty_dataframe_raises(
    data_extractor,
    empty_dataframe,
):
    """
    Test that empty DataFrame raises ValueError.

    Cannot extract arrays from empty DataFrame (0 rows).
    """
    with pytest.raises(ValueError, match="Cannot extract arrays from empty DataFrame"):
        data_extractor.extract_price_arrays(empty_dataframe)


# ==================== 2. extract_data_arrays Tests (4 tests) ====================


def test_extract_data_arrays_full_data(
    data_extractor,
    pandas_ohlcv_dataframe,
):
    """
    Test extraction of close, high, low arrays when all present.

    Verifies that all three arrays are extracted with proper dtype and shape.
    """
    result = data_extractor.extract_data_arrays(pandas_ohlcv_dataframe)
    close, high, low = result

    # Check close is always present
    assert isinstance(close, np.ndarray), "close should be numpy array"
    assert close.dtype == np.float64, "close should be float64"
    assert close.shape == (100,), "close should have shape (100,)"

    # Check high and low are present (not None)
    assert high is not None, "high should not be None when column present"
    assert isinstance(high, np.ndarray), "high should be numpy array"
    assert high.dtype == np.float64, "high should be float64"
    assert high.shape == (100,), "high should have shape (100,)"

    assert low is not None, "low should not be None when column present"
    assert isinstance(low, np.ndarray), "low should be numpy array"
    assert low.dtype == np.float64, "low should be float64"
    assert low.shape == (100,), "low should have shape (100,)"


def test_extract_data_arrays_missing_high_low_returns_none(
    data_extractor,
    close_only_dataframe,
):
    """
    Test that missing high/low columns return None.

    When high/low are absent, returns (close, None, None).
    """
    result = data_extractor.extract_data_arrays(close_only_dataframe)
    close, high, low = result

    # Close should be present
    assert isinstance(close, np.ndarray), "close should be numpy array"
    assert close.shape == (50,), "close should have shape (50,)"

    # High and low should be None
    assert high is None, "high should be None when column missing"
    assert low is None, "low should be None when column missing"


def test_extract_data_arrays_polars_compatibility(
    data_extractor,
    pandas_ohlcv_dataframe,
    polars_ohlcv_dataframe,
):
    """
    Test that Polars extraction matches Pandas extraction.

    Both formats should produce identical numerical results.
    """
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")

    pandas_result = data_extractor.extract_data_arrays(pandas_ohlcv_dataframe)
    polars_result = data_extractor.extract_data_arrays(polars_ohlcv_dataframe)

    # Compare close arrays
    np.testing.assert_allclose(pandas_result[0], polars_result[0], rtol=1e-10)

    # Compare high arrays (if both present)
    if pandas_result[1] is not None and polars_result[1] is not None:
        np.testing.assert_allclose(pandas_result[1], polars_result[1], rtol=1e-10)

    # Compare low arrays (if both present)
    if pandas_result[2] is not None and polars_result[2] is not None:
        np.testing.assert_allclose(pandas_result[2], polars_result[2], rtol=1e-10)


def test_extract_data_arrays_missing_close_raises(
    data_extractor,
    dataframe_no_close,
):
    """
    Test that missing close column raises KeyError.

    close is a required column - missing it should raise error.
    """
    with pytest.raises(KeyError, match="close"):
        data_extractor.extract_data_arrays(dataframe_no_close)


# ==================== 3. extract_bid_ask_data Tests (4 tests) ====================


def test_extract_bid_ask_data_full_quotes(
    data_extractor,
    l2_quote_dataframe,
):
    """
    Test extraction of L2 quote data.

    Verifies all 4 arrays (bid_price, ask_price, bid_size, ask_size) are extracted
    and bid <= ask invariant holds.
    """
    result = data_extractor.extract_bid_ask_data(l2_quote_dataframe)
    bid_prices, ask_prices, bid_sizes, ask_sizes = result

    # Check shapes
    assert bid_prices.shape == (100,), "bid_prices should have shape (100,)"
    assert ask_prices.shape == (100,), "ask_prices should have shape (100,)"
    assert bid_sizes.shape == (100,), "bid_sizes should have shape (100,)"
    assert ask_sizes.shape == (100,), "ask_sizes should have shape (100,)"

    # Check dtypes
    assert bid_prices.dtype == np.float64, "bid_prices should be float64"
    assert ask_prices.dtype == np.float64, "ask_prices should be float64"
    assert bid_sizes.dtype == np.float64, "bid_sizes should be float64"
    assert ask_sizes.dtype == np.float64, "ask_sizes should be float64"

    # Check bid <= ask invariant
    assert np.all(bid_prices <= ask_prices), "bid_price should be <= ask_price"

    # Check sizes are positive
    assert np.all(bid_sizes >= 0), "bid_sizes should be >= 0"
    assert np.all(ask_sizes >= 0), "ask_sizes should be >= 0"


def test_extract_bid_ask_data_polars_pandas_parity(
    data_extractor,
    l2_quote_dataframe,
    l2_quote_dataframe_polars,
):
    """
    Test that Polars and Pandas extraction are identical.

    Both formats should produce numerically identical results.
    """
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")

    pandas_result = data_extractor.extract_bid_ask_data(l2_quote_dataframe)
    polars_result = data_extractor.extract_bid_ask_data(l2_quote_dataframe_polars)

    # Compare all 4 arrays
    for pd_arr, pl_arr in zip(pandas_result, polars_result):
        np.testing.assert_allclose(pd_arr, pl_arr, rtol=1e-10)


def test_extract_bid_ask_data_missing_columns_raises(
    data_extractor,
    incomplete_l2_dataframe,
):
    """
    Test that missing required column raises KeyError.

    L2 quote data requires all 4 columns - missing any should raise error.
    """
    with pytest.raises(KeyError, match="ask_price"):
        data_extractor.extract_bid_ask_data(incomplete_l2_dataframe)


def test_extract_bid_ask_data_zero_sizes_allowed(
    data_extractor,
    l2_quote_with_zero_sizes,
):
    """
    Test that zero bid/ask sizes are allowed.

    Zero sizes represent no liquidity at that level - valid market condition.
    """
    result = data_extractor.extract_bid_ask_data(l2_quote_with_zero_sizes)
    bid_prices, _ask_prices, bid_sizes, ask_sizes = result

    # Extraction should succeed
    assert bid_prices.shape == (100,)

    # Some sizes should be zero
    assert np.any(bid_sizes == 0.0), "Some bid_sizes should be zero"
    assert np.any(ask_sizes == 0.0), "Some ask_sizes should be zero"


# ==================== 4. extract_trade_data Tests (4 tests) ====================


def test_extract_trade_data_full_trades(
    data_extractor,
    trade_tick_dataframe,
):
    """
    Test extraction of trade tick data.

    Verifies all 3 arrays (price, volume, side) are extracted with correct
    shape and dtype. trade_side should be -1.0 (sell) or 1.0 (buy).
    """
    result = data_extractor.extract_trade_data(trade_tick_dataframe)
    prices, volumes, sides = result

    # Check shapes
    assert prices.shape == (100,), "prices should have shape (100,)"
    assert volumes.shape == (100,), "volumes should have shape (100,)"
    assert sides.shape == (100,), "sides should have shape (100,)"

    # Check dtypes
    assert prices.dtype == np.float64, "prices should be float64"
    assert volumes.dtype == np.float64, "volumes should be float64"
    assert sides.dtype == np.float64, "sides should be float64"

    # Check trade_side values are -1.0 or 1.0
    assert np.all(np.isin(sides, [-1.0, 1.0])), "sides should be -1.0 or 1.0"

    # Check volumes are positive
    assert np.all(volumes > 0), "volumes should be positive"


def test_extract_trade_data_polars_pandas_parity(
    data_extractor,
    trade_tick_dataframe,
    trade_tick_dataframe_polars,
):
    """
    Test that Polars and Pandas extraction are identical.

    Both formats should produce numerically identical results.
    """
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")

    pandas_result = data_extractor.extract_trade_data(trade_tick_dataframe)
    polars_result = data_extractor.extract_trade_data(trade_tick_dataframe_polars)

    # Compare all 3 arrays
    for pd_arr, pl_arr in zip(pandas_result, polars_result):
        np.testing.assert_allclose(pd_arr, pl_arr, rtol=1e-10)


def test_extract_trade_data_missing_columns_raises(
    data_extractor,
    incomplete_trade_dataframe,
):
    """
    Test that missing required column raises KeyError.

    Trade data requires all 3 columns - missing any should raise error.
    """
    with pytest.raises(KeyError, match="trade_side"):
        data_extractor.extract_trade_data(incomplete_trade_dataframe)


def test_extract_trade_data_neutral_side_allowed(
    data_extractor,
    trade_tick_with_neutral_side,
):
    """
    Test that neutral trade side (0.0) is allowed.

    Neutral trades (market prints, auctions) have side=0.0 - should be allowed.
    """
    result = data_extractor.extract_trade_data(trade_tick_with_neutral_side)
    prices, _volumes, sides = result

    # Extraction should succeed
    assert prices.shape == (100,)

    # Some trades should be neutral (side=0.0)
    assert np.any(sides == 0.0), "Some trades should have neutral side (0.0)"

    # Check valid values
    assert np.all(np.isin(sides, [-1.0, 0.0, 1.0])), "sides should be -1.0, 0.0, or 1.0"


# ==================== 5. _ensure_float_array Tests (4 tests) ====================


def test_ensure_float_array_from_int_array(data_extractor):
    """
    Test integer arrays are converted to float64.

    int32 -> float64 conversion without data loss.
    """
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = data_extractor._ensure_float_array(arr)

    assert result.dtype == np.float64, "Result should be float64"
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-10)


def test_ensure_float_array_from_float32(data_extractor):
    """
    Test float32 arrays are upcast to float64.

    Ensures precision is upgraded to float64.
    """
    arr = np.array([1.5, 2.5], dtype=np.float32)
    result = data_extractor._ensure_float_array(arr)

    assert result.dtype == np.float64, "Result should be float64"
    np.testing.assert_allclose(result, [1.5, 2.5], rtol=1e-6)  # float32 precision


def test_ensure_float_array_handles_nans(data_extractor):
    """
    Test that NaN values are preserved during conversion.

    NaN should remain NaN after dtype conversion.
    """
    arr = np.array([1.0, np.nan, 3.0])
    result = data_extractor._ensure_float_array(arr)

    assert result.dtype == np.float64, "Result should be float64"
    assert np.isnan(result[1]), "NaN should be preserved"
    assert result[0] == 1.0, "First value should be 1.0"
    assert result[2] == 3.0, "Third value should be 3.0"


def test_ensure_float_array_from_list(data_extractor):
    """
    Test Python lists are converted to float64 arrays.

    Handles array-like objects (lists, tuples).
    """
    result = data_extractor._ensure_float_array([1.0, 2.0, 3.0])

    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.dtype == np.float64, "Result should be float64"
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-10)


# ==================== 6. _detect_dataframe_type Tests (implied in other tests) ====================

# Note: _detect_dataframe_type is tested implicitly through the polars_pandas_parity tests
# No separate tests needed as it's covered by all the parity tests above.
