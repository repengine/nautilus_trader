"""
Property-based tests for FeatureMetricsCollector component.

Uses Hypothesis to verify mathematical invariants:
- All rate metrics are in [0, 1] range
- Spreads are always non-negative
- Trade flow imbalance is always in [-1, 1] range
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from ml.features.common.feature_metrics_collector import FeatureMetricsCollector


@pytest.mark.property
@given(
    values=st.lists(
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.none(),
        ),
        min_size=1,
        max_size=1000,
    )
)
def test_column_metrics_rate_bounds(values: list[float | None]) -> None:
    """
    Verify all rate metrics are in [0, 1] range for any valid input.

    This property must hold for all possible inputs, including edge cases like
    empty data, all NaNs, all infinities, etc.
    """
    collector = FeatureMetricsCollector()
    col_data = pl.Series("col", values, dtype=pl.Float64)
    total_rows = len(values)

    metrics = collector._calculate_column_metrics(col_data, total_rows)

    # All rates must be in [0, 1] range
    assert 0.0 <= metrics["null_rate"] <= 1.0
    assert 0.0 <= metrics["zero_rate"] <= 1.0
    assert 0.0 <= metrics["unique_ratio"] <= 1.0
    assert 0.0 <= metrics["inf_rate"] <= 1.0
    assert 0.0 <= metrics["outlier_rate"] <= 1.0


@pytest.mark.property
@given(
    n=st.integers(min_value=10, max_value=100),
    bid_base=st.floats(min_value=0.01, max_value=1e6),
    ask_offset=st.floats(min_value=0.01, max_value=100.0),
)
def test_spread_metrics_non_negative(
    n: int, bid_base: float, ask_offset: float
) -> None:
    """
    Verify spreads are always non-negative for valid bid/ask data.

    This property ensures that when bid < ask (valid market data), all computed
    spreads and relative spreads are non-negative.
    """
    collector = FeatureMetricsCollector()

    # Generate valid bid/ask data (bid < ask)
    bid_prices = np.full(n, bid_base, dtype=np.float64)
    ask_prices = np.full(n, bid_base + ask_offset, dtype=np.float64)
    bid_sizes = np.ones(n, dtype=np.float64) * 100.0
    ask_sizes = np.ones(n, dtype=np.float64) * 100.0

    spreads, rel_spreads, imbalances, mid_prices = collector._calculate_spread_metrics(
        bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx=0, end_idx=n - 1
    )

    # All spreads must be non-negative
    assert all(s >= 0 for s in spreads)
    assert all(rs >= 0 for rs in rel_spreads)

    # All mid prices must be positive
    assert all(m > 0 for m in mid_prices)

    # All size imbalances must be in [-1, 1]
    assert all(-1.0 <= imb <= 1.0 for imb in imbalances)


@pytest.mark.property
@given(
    n=st.integers(min_value=5, max_value=100),
    price_base=st.floats(min_value=0.01, max_value=1e6),
    volume_base=st.floats(min_value=0.01, max_value=1e6),
)
def test_trade_metrics_imbalance_bounds(
    n: int, price_base: float, volume_base: float
) -> None:
    """
    Verify trade_flow_imbalance is always in [-1, 1] range.

    This property ensures that regardless of the mix of buy/sell trades,
    the computed flow imbalance is always within valid bounds.
    """
    collector = FeatureMetricsCollector()

    # Generate trade data with random buy/sell mix
    np.random.seed(42)
    trade_prices = np.full(n, price_base, dtype=np.float64)
    trade_volumes = np.full(n, volume_base, dtype=np.float64)
    trade_sides = np.random.choice([1.0, -1.0], size=n).astype(np.float64)

    flow_imb, vwap, intensity, impact, had_trades = collector._calculate_trade_metrics(
        trade_prices, trade_volumes, trade_sides, start_idx=0, end_idx=n - 1
    )

    # Flow imbalance must be in [-1, 1]
    assert -1.0 <= flow_imb <= 1.0

    # VWAP must be non-negative (since prices are positive)
    assert vwap >= 0.0

    # Trade intensity must be in [0, 5.0] (capped)
    assert 0.0 <= intensity <= 5.0

    # Price impact must be non-negative
    assert impact >= 0.0

    # had_trades should be True for valid data
    assert had_trades is True
