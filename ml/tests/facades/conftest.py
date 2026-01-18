"""
Pytest configuration for facade parity tests.

Fixtures are registered via the shared pytest plug-in defined in ``ml/tests/__init__.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from ml.config.base import MLFeatureConfig
from ml.features.config import FeatureConfig
from ml.features.indicators import IndicatorManager


if TYPE_CHECKING:
    pass


# ==================== FeatureConfig Fixtures ====================


@pytest.fixture
def feature_config() -> FeatureConfig:
    """
    Standard FeatureConfig for facade testing.

    Provides a comprehensive configuration with all feature types enabled
    for testing facade behavior across batch and online modes.
    """
    return FeatureConfig(
        return_periods=[1, 2, 5],
        momentum_periods=[1, 3],
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
        include_microstructure=False,
        include_trade_flow=False,
    )


@pytest.fixture
def benchmark_config() -> dict:
    """
    Standard benchmark configuration for performance tests.

    Returns:
        dict with n_warmup, n_iterations, p99_threshold_ms, overhead_threshold_pct
    """
    return {
        "n_warmup": 100,
        "n_iterations": 1000,
        "p99_threshold_ms": 5.0,  # HOT PATH requirement
        "overhead_threshold_pct": 10.0,  # Max 10% overhead vs calculator
    }


# ==================== DataFrame Fixtures ====================


@pytest.fixture
def sample_ohlcv_dataframe() -> pd.DataFrame:
    """
    DataFrame with 100 bars of synthetic OHLCV data.

    Uses seed 42 for reproducibility. Contains timestamp, open, high,
    low, close, and volume columns suitable for feature computation.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="1min")
    close_prices = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(100) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(100) * 0.3)
    open_prices = close_prices + np.random.randn(100) * 0.2
    volumes = np.random.uniform(900000, 1100000, 100)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


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


# ==================== Bar Dict Fixtures ====================


@pytest.fixture
def current_bar_dict() -> dict[str, float]:
    """
    Single bar as dict for online mode testing.

    Represents a single OHLCV bar in dictionary format as expected by
    calculate_features_online.
    """
    return {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000000.0,
    }


@pytest.fixture
def sample_bar_dict() -> dict[str, float]:
    """
    Single bar dict for online mode benchmarks.

    Alias for current_bar_dict for performance test clarity.
    """
    return {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000000.0,
    }


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


# ==================== IndicatorManager Fixtures ====================


@pytest.fixture
def indicator_manager_with_history(feature_config: FeatureConfig) -> IndicatorManager:
    """
    IndicatorManager pre-warmed with 50 bars of history.

    Ready for inference with indicators properly warmed up.
    Uses deterministic synthetic data for reproducibility.
    """
    manager = IndicatorManager(feature_config)
    for i in range(50):
        manager.update_from_values(
            close=100.0 + i * 0.1,
            high=101.0 + i * 0.1,
            low=99.0 + i * 0.1,
            volume=1000000.0 + i * 1000,
        )
    return manager


@pytest.fixture
def prepared_indicator_manager(feature_config: FeatureConfig) -> IndicatorManager:
    """
    IndicatorManager with 50 bars of history (ready for inference).

    Alias for indicator_manager_with_history for performance test clarity.
    """
    manager = IndicatorManager(feature_config)
    for i in range(50):
        manager.update_from_values(
            close=100.0 + i * 0.1,
            high=101.0 + i * 0.1,
            low=99.0 + i * 0.1,
            volume=1000000.0 + i * 1000,
        )
    return manager


# ==================== Mock Result Fixtures ====================


@pytest.fixture
def mock_calculator_result_batch() -> tuple[pd.DataFrame, None]:
    """Mock return value for batch mode delegation tests."""
    return (pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]}), None)


@pytest.fixture
def mock_calculator_result_online() -> np.ndarray:
    """Mock return value for online mode delegation tests."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


# ==================== MLFeatureConfig Fixtures (for parity tests) ====================


@pytest.fixture
def ml_feature_config() -> MLFeatureConfig:
    """
    MLFeatureConfig for parity tests (msgspec-based config).

    Note: This is distinct from `feature_config` which returns FeatureConfig
    from ml.features.engineering for facade tests.
    """
    return MLFeatureConfig(
        lookback_window=120,
        normalize_features=True,
        fill_missing_with=0.0,
        average_volume=1000000.0,
    )


# ==================== Nautilus Bar Fixtures ====================


@pytest.fixture
def test_bar() -> Bar:
    """
    Single Nautilus Bar object for parity testing.

    Creates a properly constructed Bar with realistic OHLCV data.
    """
    instrument_id = InstrumentId.from_str("SPY.NYSE")
    bar_type = BarType.from_str("SPY.NYSE-1-MINUTE-LAST-EXTERNAL")

    return Bar(
        bar_type=bar_type,
        open=Price.from_str("100.00"),
        high=Price.from_str("101.00"),
        low=Price.from_str("99.00"),
        close=Price.from_str("100.50"),
        volume=Quantity.from_str("1000000"),
        ts_event=1609459200000000000,  # 2021-01-01 00:00:00 UTC
        ts_init=1609459200000000001,
    )


@pytest.fixture
def test_bars(test_bar: Bar) -> list[Bar]:
    """
    List of Nautilus Bar objects (100 bars).

    Creates a sequence of bars for parity testing with compute_features.
    """
    np.random.seed(42)
    bar_type = BarType.from_str("SPY.NYSE-1-MINUTE-LAST-EXTERNAL")

    bars = []
    price = 100.0
    base_ts = 1609459200000000000  # 2021-01-01 00:00:00 UTC

    for i in range(100):
        price_change = np.random.randn() * 0.5
        close = price + price_change
        open_price = close + np.random.randn() * 0.2

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_price, close) + abs(np.random.randn() * 0.3)
        low = min(open_price, close) - abs(np.random.randn() * 0.3)

        price = close  # Update for next bar

        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.2f}"),
                high=Price.from_str(f"{high:.2f}"),
                low=Price.from_str(f"{low:.2f}"),
                close=Price.from_str(f"{close:.2f}"),
                volume=Quantity.from_str("1000000"),
                ts_event=base_ts + i * 60_000_000_000,  # 1-minute intervals
                ts_init=base_ts + i * 60_000_000_000 + 1,
            )
        )

    return bars
