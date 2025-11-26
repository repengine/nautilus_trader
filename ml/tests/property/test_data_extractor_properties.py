"""
Property-based tests for DataExtractor component.

Uses Hypothesis to test invariants that should hold for all valid inputs:
1. Shape preservation (output rows == input rows)
2. Dtype consistency (all outputs are float64)
3. bid <= ask invariant (L2 quotes)
4. No data loss in roundtrip (DataFrame -> arrays -> values match)

Total: 4 property tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture
def data_extractor():
    """Provides DataExtractor instance."""
    from ml.features.common.data_extractor import DataExtractor

    return DataExtractor()


# ==================== Property Tests (4 tests) ====================


@given(
    n_rows=st.integers(min_value=1, max_value=1000),
    close_base=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=1000)
def test_property_extract_preserves_shape(n_rows, close_base):
    """
    Property: Extraction preserves DataFrame row count.

    Invariant: Output array row count == input DataFrame row count

    For ANY valid DataFrame with n_rows, the extracted arrays should
    have exactly n_rows elements.
    """
    from ml.features.common.data_extractor import DataExtractor

    extractor = DataExtractor()

    # Generate DataFrame with n_rows
    close_prices = np.linspace(close_base, close_base + 10.0, n_rows)
    df = pd.DataFrame({"close": close_prices})

    # Extract arrays
    result = extractor.extract_data_arrays(df)
    close_arr, _high_arr, _low_arr = result

    # INVARIANT: output row count == input row count
    assert close_arr.shape[0] == n_rows, f"close array should have {n_rows} elements"

    # If extracting full OHLCV with defaults
    df_full = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices * 1.01,
            "low": close_prices * 0.99,
            "close": close_prices,
            "volume": np.full(n_rows, 10000.0),
        }
    )

    result_full = extractor.extract_price_arrays(df_full)
    for arr in result_full:
        assert arr.shape[0] == n_rows, f"All OHLCV arrays should have {n_rows} elements"


@given(
    n_rows=st.integers(min_value=1, max_value=500),
    close_base=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=1000)
def test_property_extract_preserves_dtype(n_rows, close_base):
    """
    Property: All output arrays are np.float64.

    Invariant: dtype == np.float64 for ALL output arrays

    Regardless of input dtype (int32, float32, etc.), output should
    always be np.float64.
    """
    from ml.features.common.data_extractor import DataExtractor

    extractor = DataExtractor()

    # Generate DataFrame with various input dtypes
    close_prices = np.linspace(close_base, close_base + 10.0, n_rows)

    # Test with int32 input
    df_int = pd.DataFrame({"close": close_prices.astype(np.int32)})
    result_int = extractor.extract_data_arrays(df_int)
    assert result_int[0].dtype == np.float64, "Should convert int32 to float64"

    # Test with float32 input
    df_float32 = pd.DataFrame({"close": close_prices.astype(np.float32)})
    result_float32 = extractor.extract_data_arrays(df_float32)
    assert result_float32[0].dtype == np.float64, "Should convert float32 to float64"

    # Test with float64 input (no conversion needed)
    df_float64 = pd.DataFrame({"close": close_prices.astype(np.float64)})
    result_float64 = extractor.extract_data_arrays(df_float64)
    assert result_float64[0].dtype == np.float64, "Should preserve float64"


@given(
    n_rows=st.integers(min_value=1, max_value=500),
    mid_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    spread_pct=st.floats(min_value=0.0001, max_value=0.1),  # 0.01% to 10%
)
@settings(max_examples=50, deadline=1000)
def test_property_bid_ask_spread_non_negative(n_rows, mid_price, spread_pct):
    """
    Property: bid_price <= ask_price invariant always holds.

    Invariant: bid_price <= ask_price for ALL rows

    For ANY valid L2 quote data, the bid should never exceed the ask.
    This is a fundamental market microstructure invariant.
    """
    from ml.features.common.data_extractor import DataExtractor

    extractor = DataExtractor()

    # Generate L2 quotes with realistic spreads
    half_spread = mid_price * spread_pct / 2
    bid_prices = np.full(n_rows, mid_price - half_spread)
    ask_prices = np.full(n_rows, mid_price + half_spread)

    df = pd.DataFrame(
        {
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": np.full(n_rows, 100.0),
            "ask_size": np.full(n_rows, 100.0),
        }
    )

    bid, ask, _bid_sz, _ask_sz = extractor.extract_bid_ask_data(df)

    # INVARIANT: bid <= ask (always)
    assert np.all(bid <= ask), "bid_price must be <= ask_price for all rows"

    # Additional invariant: spread is non-negative
    spread = ask - bid
    assert np.all(spread >= 0), "Spread (ask - bid) must be non-negative"


@given(
    close_prices=st.lists(
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    ),
)
@settings(max_examples=50, deadline=1000)
def test_property_no_data_loss_roundtrip(close_prices):
    """
    Property: Extraction doesn't lose data (roundtrip equality).

    Invariant: DataFrame values == extracted array values (within float precision)

    When extracting arrays from a DataFrame, the values should match
    exactly (or within floating-point precision). No data should be
    lost or corrupted during extraction.
    """
    from ml.features.common.data_extractor import DataExtractor

    extractor = DataExtractor()

    # Create DataFrame
    df = pd.DataFrame({"close": close_prices})

    # Extract arrays
    result = extractor.extract_data_arrays(df)
    close_arr = result[0]

    # INVARIANT: extracted values match DataFrame values
    # Use allclose to account for float precision
    np.testing.assert_allclose(
        close_arr,
        close_prices,
        rtol=1e-10,
        err_msg="Extracted array should match DataFrame values",
    )

    # Additional check: no NaN introduced where there was none
    if not any(np.isnan(close_prices)):
        assert not np.any(np.isnan(close_arr)), "Should not introduce NaN values"
