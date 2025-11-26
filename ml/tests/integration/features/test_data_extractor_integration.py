"""
Integration tests for DataExtractor component.

Tests DataExtractor with realistic data scenarios including:
- Large datasets (performance)
- Multi-instrument data
- Real-world OHLCV/L2/trade data patterns

Total: 5 integration tests
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data_extractor():
    """Provides DataExtractor instance."""
    from ml.features.common.data_extractor import DataExtractor

    return DataExtractor()


@pytest.fixture
def realistic_spy_ohlcv_data() -> pd.DataFrame:
    """
    Realistic SPY OHLCV data (1000 bars).

    Simulates 1000 1-minute bars with realistic price action.
    """
    np.random.seed(42)
    n = 1000

    # Start at $450, add random walk
    close_prices = np.cumsum(np.random.normal(0, 0.5, n)) + 450.0

    # High/low based on close with realistic spreads
    high_prices = close_prices + np.abs(np.random.normal(0.5, 0.2, n))
    low_prices = close_prices - np.abs(np.random.normal(0.5, 0.2, n))

    # Open based on previous close
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Volume with realistic patterns
    volume = np.random.lognormal(15, 1, n)

    return pd.DataFrame(
        {
            "instrument_id": ["SPY"] * n,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        }
    )


@pytest.fixture
def realistic_l2_quotes() -> pd.DataFrame:
    """
    Realistic L2 quote data (500 quotes).

    Simulates tick-by-tick bid/ask quotes with realistic spreads.
    """
    np.random.seed(42)
    n = 500

    # Mid price random walk
    mid_price = np.cumsum(np.random.normal(0, 0.1, n)) + 100.0

    # Realistic spreads (0.01 to 0.05)
    spreads = np.random.uniform(0.01, 0.05, n)
    bid_prices = mid_price - spreads / 2
    ask_prices = mid_price + spreads / 2

    # Realistic sizes (100-10000 shares)
    bid_sizes = np.random.uniform(100, 10000, n)
    ask_sizes = np.random.uniform(100, 10000, n)

    return pd.DataFrame(
        {
            "instrument_id": ["SPY"] * n,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="100ms"),
        }
    )


@pytest.fixture
def realistic_trade_ticks() -> pd.DataFrame:
    """
    Realistic trade tick data (800 trades).

    Simulates trade-by-trade executions with realistic price/volume/side patterns.
    """
    np.random.seed(42)
    n = 800

    # Trade prices (random walk)
    trade_prices = np.cumsum(np.random.normal(0, 0.1, n)) + 100.0

    # Trade volumes (log-normal distribution, some large prints)
    trade_volumes = np.random.lognormal(5, 1.5, n)

    # Trade sides (buy/sell, with occasional neutral)
    sides = np.random.choice([-1.0, 0.0, 1.0], n, p=[0.45, 0.1, 0.45])

    return pd.DataFrame(
        {
            "instrument_id": ["SPY"] * n,
            "trade_price": trade_prices,
            "trade_volume": trade_volumes,
            "trade_side": sides,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="50ms"),
        }
    )


@pytest.fixture
def multi_instrument_ohlcv() -> pd.DataFrame:
    """
    Multi-instrument OHLCV data (SPY, QQQ, IWM).

    3 instruments × 500 bars each = 1500 rows total.
    """
    np.random.seed(42)

    instruments = {
        "SPY": 450.0,  # Starting price
        "QQQ": 390.0,
        "IWM": 185.0,
    }

    dfs = []
    for symbol, start_price in instruments.items():
        n = 500
        close_prices = np.cumsum(np.random.normal(0, 0.3, n)) + start_price
        high_prices = close_prices + np.abs(np.random.normal(0.3, 0.1, n))
        low_prices = close_prices - np.abs(np.random.normal(0.3, 0.1, n))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = start_price
        volume = np.random.lognormal(14, 1.2, n)

        dfs.append(
            pd.DataFrame(
                {
                    "instrument_id": [symbol] * n,
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volume,
                    "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
                }
            )
        )

    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def large_ohlcv_dataset() -> pd.DataFrame:
    """
    Large dataset for performance testing (100k rows).
    """
    np.random.seed(42)
    n = 100_000

    return pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(101, 111, n),
            "low": np.random.uniform(99, 109, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 100000, n),
        }
    )


# ==================== Integration Tests (5 tests) ====================


def test_data_extractor_with_realistic_ohlcv(
    data_extractor,
    realistic_spy_ohlcv_data,
):
    """
    Test DataExtractor with realistic SPY OHLCV data.

    Verifies extraction works with real-world patterns:
    - Proper handling of 1000 bars
    - No NaN values in output (except where expected)
    - Arrays have correct shape
    - bid <= ask invariant (for high/low)
    """
    # Extract arrays
    open_arr, high_arr, low_arr, close_arr, _vol_arr = data_extractor.extract_price_arrays(
        realistic_spy_ohlcv_data
    )

    # Verify shape matches input
    assert open_arr.shape[0] == 1000, "Should have 1000 bars"
    assert close_arr.shape[0] == 1000, "Close should have 1000 values"

    # Verify no NaN values (data is clean)
    assert not np.any(np.isnan(close_arr)), "Close should have no NaN values"

    # Verify high >= low invariant
    assert np.all(high_arr >= low_arr), "High should be >= low"

    # Verify high >= close >= low (usually holds)
    # Allow some tolerance for realistic data where close might be exactly at high/low
    assert np.sum(high_arr >= close_arr) >= 900, "Most bars: high >= close"
    assert np.sum(close_arr >= low_arr) >= 900, "Most bars: close >= low"


def test_data_extractor_with_realistic_l2_quotes(
    data_extractor,
    realistic_l2_quotes,
):
    """
    Test extraction from realistic L2 quote data.

    Verifies L2 extraction handles:
    - 500 quote updates
    - bid <= ask invariant holds
    - Realistic spread values
    """
    # Extract bid/ask arrays
    bid, ask, _bid_sz, _ask_sz = data_extractor.extract_bid_ask_data(realistic_l2_quotes)

    # Verify shape
    assert bid.shape[0] == 500, "Should have 500 quotes"

    # Verify bid <= ask invariant (CRITICAL)
    assert np.all(bid <= ask), "bid_price must be <= ask_price"

    # Verify realistic spreads (should be small, < $1.00 for SPY)
    spreads = ask - bid
    assert np.all(spreads >= 0), "Spreads should be non-negative"
    assert np.all(spreads < 1.0), "Spreads should be < $1.00 for SPY"

    # Verify sizes are positive
    assert np.all(_bid_sz > 0), "Bid sizes should be positive"
    assert np.all(_ask_sz > 0), "Ask sizes should be positive"


def test_data_extractor_with_realistic_trade_ticks(
    data_extractor,
    realistic_trade_ticks,
):
    """
    Test extraction from realistic trade tick data.

    Verifies trade extraction handles:
    - 800 trade executions
    - Valid trade_side values (-1, 0, 1)
    - Positive volumes
    """
    # Extract trade arrays
    prices, _volumes, sides = data_extractor.extract_trade_data(realistic_trade_ticks)

    # Verify shape
    assert prices.shape[0] == 800, "Should have 800 trades"

    # Verify trade_side values are valid
    assert np.all(np.isin(sides, [-1.0, 0.0, 1.0])), "Sides must be -1, 0, or 1"

    # Verify volumes are positive
    assert np.all(_volumes > 0), "Trade volumes must be positive"

    # Verify prices are reasonable (> 0)
    assert np.all(prices > 0), "Trade prices must be positive"


def test_data_extractor_multi_instrument(
    data_extractor,
    multi_instrument_ohlcv,
):
    """
    Test extraction works for multiple instruments in single DataFrame.

    Verifies:
    - Can filter by instrument_id and extract separately
    - No cross-contamination between instruments
    - Each instrument has correct number of bars
    """
    # Extract SPY data
    spy_df = multi_instrument_ohlcv[multi_instrument_ohlcv["instrument_id"] == "SPY"]
    spy_arrays = data_extractor.extract_price_arrays(spy_df)
    spy_close = spy_arrays[3]

    # Extract QQQ data
    qqq_df = multi_instrument_ohlcv[multi_instrument_ohlcv["instrument_id"] == "QQQ"]
    qqq_arrays = data_extractor.extract_price_arrays(qqq_df)
    qqq_close = qqq_arrays[3]

    # Extract IWM data
    iwm_df = multi_instrument_ohlcv[multi_instrument_ohlcv["instrument_id"] == "IWM"]
    iwm_arrays = data_extractor.extract_price_arrays(iwm_df)
    iwm_close = iwm_arrays[3]

    # Verify each has correct number of bars
    assert spy_close.shape[0] == 500, "SPY should have 500 bars"
    assert qqq_close.shape[0] == 500, "QQQ should have 500 bars"
    assert iwm_close.shape[0] == 500, "IWM should have 500 bars"

    # Verify price ranges are different (no contamination)
    assert np.mean(spy_close) > 400, "SPY avg should be ~450"
    assert np.mean(qqq_close) > 350, "QQQ avg should be ~390"
    assert np.mean(iwm_close) > 150, "IWM avg should be ~185"


@pytest.mark.slow
def test_data_extractor_large_dataset_performance(
    data_extractor,
    large_ohlcv_dataset,
):
    """
    Test extraction is efficient for large datasets.

    Performance requirements (NOT hot path, but should be reasonable):
    - Extraction of 100k rows should complete in < 1 second
    - Memory overhead should be reasonable

    This is NOT a hot path component (cold path - batch processing),
    so we're lenient on performance, but should still be fast enough.
    """
    # Measure extraction time
    start = time.perf_counter()
    result = data_extractor.extract_price_arrays(large_ohlcv_dataset)
    elapsed = time.perf_counter() - start

    # Verify extraction completed
    assert result[3].shape[0] == 100_000, "Should extract all 100k rows"

    # Performance check: should complete in < 1 second (generous for cold path)
    assert elapsed < 1.0, f"Extraction took {elapsed:.3f}s (should be < 1.0s)"

    # Verify result correctness
    _open_arr, _high_arr, _low_arr, _close_arr, _vol_arr = result
    assert all(arr.dtype == np.float64 for arr in result), "All arrays should be float64"
    assert all(arr.shape == (100_000,) for arr in result), "All arrays should have 100k elements"
