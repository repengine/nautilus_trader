"""
Test fixtures for feature component integration tests.

Fixtures are registered via the shared pytest plug-in defined in ``ml/tests/__init__.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml.features.engineering import FeatureConfig, IndicatorManager


if TYPE_CHECKING:
    pass


# ==================== Registry Mock Fixtures ====================


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


# ==================== FeatureConfig Fixtures ====================


@pytest.fixture
def feature_config() -> FeatureConfig:
    """
    Standard FeatureConfig for integration tests.

    Provides a comprehensive configuration with extended return/momentum
    periods suitable for integration test workflows.
    """
    return FeatureConfig(
        return_periods=[1, 2, 5, 10],
        momentum_periods=[1, 3, 5],
        volume_ma_periods=[10, 20],
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        enable_returns=True,
        enable_momentum=True,
        enable_volatility=True,
        enable_technical=True,
    )


# ==================== DataFrame Fixtures ====================


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    """
    DataFrame simulating training data (1000 bars).

    Uses seed 42 for reproducibility. Suitable for integration tests
    that require a larger dataset for scaler fitting and training workflows.
    """
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1min")
    close_prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.randn(n_bars) * 0.2,
            "high": close_prices + np.abs(np.random.randn(n_bars) * 0.3),
            "low": close_prices - np.abs(np.random.randn(n_bars) * 0.3),
            "close": close_prices,
            "volume": np.random.uniform(900000, 1100000, n_bars),
        }
    )


@pytest.fixture
def inference_bars() -> list[dict[str, float]]:
    """
    List of bars simulating real-time inference data.

    Returns 100 bars with seed 123 for reproducibility. Suitable for
    testing online workflow with warmup and inference phases.
    """
    np.random.seed(123)
    bars = []
    price = 100.0
    for i in range(100):
        price += np.random.randn() * 0.1
        bars.append(
            {
                "open": price - 0.05,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price,
                "volume": 1000000.0 + np.random.uniform(-100000, 100000),
            }
        )
    return bars


# ==================== FeatureMetricsCollector Fixtures (Phase 2.1.3) ====================


@pytest.fixture
def realistic_column_data() -> pl.Series:
    """
    Realistic column data for integration testing.

    Generates ~1000 data points with:
    - ~95% normal distribution values
    - ~3% nulls
    - ~0.5% outliers
    - ~1% zeros
    - ~0.5% duplicates
    """
    np.random.seed(42)
    values = np.random.normal(100, 10, 950).tolist()
    values.extend([None] * 30)  # 3% nulls
    values.extend([0.0, 200.0, -50.0, 250.0, 300.0])  # outliers
    values.extend([0.0] * 10)  # zeros
    values.extend([100.0] * 5)  # duplicates
    return pl.Series("feature", values, dtype=pl.Float64)


@pytest.fixture
def realistic_spread_data() -> dict[str, np.ndarray]:
    """
    Realistic L2 spread data for integration testing.

    Generates 50 ticks of bid/ask data with:
    - Base price around 100.0
    - Spreads between 0.01 and 0.1
    - Sizes between 50 and 500
    """
    np.random.seed(42)
    n = 50
    base_price = 100.0

    bid_prices = base_price - np.random.uniform(0.01, 0.1, n)
    ask_prices = base_price + np.random.uniform(0.01, 0.1, n)
    bid_sizes = np.random.uniform(50, 500, n)
    ask_sizes = np.random.uniform(50, 500, n)

    return {
        "bid_prices": bid_prices,
        "ask_prices": ask_prices,
        "bid_sizes": bid_sizes,
        "ask_sizes": ask_sizes,
    }


@pytest.fixture
def realistic_trade_data() -> dict[str, np.ndarray]:
    """
    Realistic trade data for integration testing.

    Generates 50 trades with:
    - Prices around 100.0 (+/- 2.0)
    - Volumes between 1.0 and 100.0
    - Mix of buy and sell trades
    """
    np.random.seed(42)
    n = 50

    trade_prices = 100.0 + np.random.uniform(-2.0, 2.0, n)
    trade_volumes = np.random.uniform(1.0, 100.0, n)
    trade_sides = np.random.choice([1.0, -1.0], size=n)

    return {
        "trade_prices": trade_prices,
        "trade_volumes": trade_volumes,
        "trade_sides": trade_sides,
    }
