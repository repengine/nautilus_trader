"""
Test fixtures for FeatureStoreAccessor component tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pandas as pd
import pytest


if TYPE_CHECKING:
    from ml.stores.protocols import FeatureStoreStrictProtocol


@pytest.fixture
def mock_feature_store() -> Mock:
    """
    Mock FeatureStoreStrictProtocol for unit tests.

    Returns a mock with successful default behavior:
    - read_range() returns a sample DataFrame
    - write_features() succeeds (no exception)

    """
    mock = Mock(spec=["write_features", "flush", "read_range"])

    # Default behavior: successful read operations
    mock.read_range.return_value = pd.DataFrame(
        {
            "instrument_id": ["SPY", "SPY"],
            "ts_event": [1609459200000000000, 1609459300000000000],
            "ts_init": [1609459200000000100, 1609459300000000100],
            "price_sma_20": [100.5, 101.2],
            "rsi_14": [55.3, 58.7],
            "volume_ratio_20": [1.2, 0.9],
        }
    )

    # Default behavior: successful write operations (no exception)
    mock.write_features.return_value = None

    # Default behavior: flush succeeds
    mock.flush.return_value = None

    return mock


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """
    Sample feature DataFrame for tests.
    """
    return pd.DataFrame(
        {
            "price_sma_20": [100.5, 101.2, 102.3],
            "rsi_14": [55.3, 58.7, 60.1],
            "volume_ratio_20": [1.2, 0.9, 1.5],
            "bb_upper": [105.0, 106.0, 107.0],
            "bb_lower": [95.0, 96.0, 97.0],
        }
    )


@pytest.fixture
def valid_timestamps() -> dict[str, int]:
    """
    Valid nanosecond timestamps for tests.
    """
    return {
        "ts_event": 1609459200000000000,  # 2021-01-01 00:00:00 UTC
        "ts_init": 1609459200000000100,  # 100 ns later
        "ts_start": 1609459200000000000,
        "ts_end": 1609545600000000000,  # 2021-01-02 00:00:00 UTC
    }


# ==================== Registry Accessor Fixtures (Phase 2.1.2) ====================


@pytest.fixture
def mock_stores_with_registries() -> Mock:
    """
    Mock stores container with all 4 registries.

    Simulates the ActorStoresRegistries container returned by
    init_ml_stores_and_registries() with all registries present.

    Returns
    -------
    Mock
        Mock object with feature_registry, model_registry,
        strategy_registry, and data_registry attributes.

    """
    mock_stores = Mock()

    # Create mock registries
    mock_stores.feature_registry = Mock(spec=["register", "get", "list"])
    mock_stores.model_registry = Mock(spec=["register", "get", "list"])
    mock_stores.strategy_registry = Mock(spec=["register", "get", "list"])
    mock_stores.data_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


@pytest.fixture
def mock_stores_missing_feature_registry() -> Mock:
    """
    Mock stores container WITHOUT feature_registry attribute.

    Used to test defensive programming when stores lacks feature_registry.

    """
    mock_stores = Mock(spec=["model_registry", "strategy_registry", "data_registry"])

    # Only include 3 of 4 registries (missing feature_registry)
    mock_stores.model_registry = Mock(spec=["register", "get", "list"])
    mock_stores.strategy_registry = Mock(spec=["register", "get", "list"])
    mock_stores.data_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


@pytest.fixture
def mock_stores_missing_model_registry() -> Mock:
    """
    Mock stores container WITHOUT model_registry attribute.
    """
    mock_stores = Mock(spec=["feature_registry", "strategy_registry", "data_registry"])

    mock_stores.feature_registry = Mock(spec=["register", "get", "list"])
    mock_stores.strategy_registry = Mock(spec=["register", "get", "list"])
    mock_stores.data_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


@pytest.fixture
def mock_stores_missing_strategy_registry() -> Mock:
    """
    Mock stores container WITHOUT strategy_registry attribute.
    """
    mock_stores = Mock(spec=["feature_registry", "model_registry", "data_registry"])

    mock_stores.feature_registry = Mock(spec=["register", "get", "list"])
    mock_stores.model_registry = Mock(spec=["register", "get", "list"])
    mock_stores.data_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


@pytest.fixture
def mock_stores_missing_data_registry() -> Mock:
    """
    Mock stores container WITHOUT data_registry attribute.
    """
    mock_stores = Mock(spec=["feature_registry", "model_registry", "strategy_registry"])

    mock_stores.feature_registry = Mock(spec=["register", "get", "list"])
    mock_stores.model_registry = Mock(spec=["register", "get", "list"])
    mock_stores.strategy_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


@pytest.fixture
def mock_stores_partial() -> Mock:
    """
    Mock stores container with only 2 registries (feature + model).

    Used to test partial dependency injection scenarios.

    """
    mock_stores = Mock(spec=["feature_registry", "model_registry"])

    # Only include 2 registries
    mock_stores.feature_registry = Mock(spec=["register", "get", "list"])
    mock_stores.model_registry = Mock(spec=["register", "get", "list"])

    return mock_stores


# ==================== DataExtractor Fixtures (Phase 2.1.5) ====================

import numpy as np

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


@pytest.fixture
def data_extractor():
    """Provides DataExtractor instance."""
    from ml.features.common.data_extractor import DataExtractor

    return DataExtractor()


@pytest.fixture
def pandas_ohlcv_dataframe() -> pd.DataFrame:
    """Pandas DataFrame with 100 rows of OHLCV data."""
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 105.0, 100),
            "high": np.linspace(101.0, 106.0, 100),
            "low": np.linspace(99.0, 104.0, 100),
            "close": np.linspace(100.5, 105.5, 100),
            "volume": np.full(100, 10000.0),
        }
    )


@pytest.fixture
def polars_ohlcv_dataframe(pandas_ohlcv_dataframe) -> pd.DataFrame | None:
    """Polars DataFrame with same data as pandas_ohlcv_dataframe."""
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")
    return pl.from_pandas(pandas_ohlcv_dataframe)


@pytest.fixture
def minimal_close_volume_dataframe() -> pd.DataFrame:
    """DataFrame with only close and volume (no OHLC)."""
    return pd.DataFrame(
        {
            "close": np.linspace(100.0, 110.0, 50),
            "volume": np.full(50, 5000.0),
        }
    )


@pytest.fixture
def ohlc_no_volume_dataframe() -> pd.DataFrame:
    """DataFrame with OHLC but no volume."""
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 105.0, 75),
            "high": np.linspace(101.0, 106.0, 75),
            "low": np.linspace(99.0, 104.0, 75),
            "close": np.linspace(100.5, 105.5, 75),
        }
    )


@pytest.fixture
def single_row_ohlcv_dataframe() -> pd.DataFrame:
    """Single-row OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10000.0],
        }
    )


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame with OHLCV columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def close_only_dataframe() -> pd.DataFrame:
    """DataFrame with only close column."""
    return pd.DataFrame({"close": np.linspace(100.0, 110.0, 50)})


@pytest.fixture
def dataframe_no_close() -> pd.DataFrame:
    """DataFrame missing close column (invalid)."""
    return pd.DataFrame(
        {
            "high": np.linspace(101.0, 106.0, 50),
            "low": np.linspace(99.0, 104.0, 50),
        }
    )


@pytest.fixture
def l2_quote_dataframe() -> pd.DataFrame:
    """L2 quote data with bid/ask prices and sizes."""
    return pd.DataFrame(
        {
            "bid_price": np.linspace(99.9, 104.9, 100),
            "ask_price": np.linspace(100.1, 105.1, 100),
            "bid_size": np.full(100, 1000.0),
            "ask_size": np.full(100, 1000.0),
        }
    )


@pytest.fixture
def l2_quote_dataframe_polars(l2_quote_dataframe) -> pd.DataFrame | None:
    """Polars version of L2 quote data."""
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")
    return pl.from_pandas(l2_quote_dataframe)


@pytest.fixture
def incomplete_l2_dataframe() -> pd.DataFrame:
    """L2 quote data missing ask_price column."""
    return pd.DataFrame(
        {
            "bid_price": np.linspace(99.9, 104.9, 100),
            "bid_size": np.full(100, 1000.0),
            "ask_size": np.full(100, 1000.0),
        }
    )


@pytest.fixture
def l2_quote_with_zero_sizes() -> pd.DataFrame:
    """L2 quotes with some zero bid/ask sizes (no liquidity)."""
    bid_sizes = [1000.0] * 50 + [0.0] * 50
    ask_sizes = [1000.0] * 40 + [0.0] * 60
    return pd.DataFrame(
        {
            "bid_price": np.linspace(99.9, 104.9, 100),
            "ask_price": np.linspace(100.1, 105.1, 100),
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        }
    )


@pytest.fixture
def trade_tick_dataframe() -> pd.DataFrame:
    """Trade tick data with price, volume, side."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame(
        {
            "trade_price": np.linspace(100.0, 105.0, 100),
            "trade_volume": np.random.uniform(10, 1000, 100),
            "trade_side": np.random.choice([-1.0, 1.0], 100),
        }
    )


@pytest.fixture
def trade_tick_dataframe_polars(trade_tick_dataframe) -> pd.DataFrame | None:
    """Polars version of trade tick data."""
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")
    return pl.from_pandas(trade_tick_dataframe)


@pytest.fixture
def incomplete_trade_dataframe() -> pd.DataFrame:
    """Trade data missing trade_side column."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "trade_price": np.linspace(100.0, 105.0, 100),
            "trade_volume": np.random.uniform(10, 1000, 100),
        }
    )


@pytest.fixture
def trade_tick_with_neutral_side() -> pd.DataFrame:
    """Trade data with some neutral sides (side=0)."""
    np.random.seed(42)
    sides = [-1.0] * 40 + [0.0] * 20 + [1.0] * 40
    return pd.DataFrame(
        {
            "trade_price": np.linspace(100.0, 105.0, 100),
            "trade_volume": np.random.uniform(10, 1000, 100),
            "trade_side": sides,
        }
    )


@pytest.fixture
def large_ohlcv_dataframe() -> pd.DataFrame:
    """Large DataFrame with 100,000 rows for performance tests."""
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
