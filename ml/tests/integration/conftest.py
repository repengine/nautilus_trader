
"""
Shared pytest fixtures for ML integration tests.

This module provides fixtures for:
- Mock ParquetDataCatalog with test data
- Generating test Bar objects with realistic data
- Generating test ML signals
- Test configuration using Nautilus patterns

"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_XGBOOST
from ml._imports import lgb
from ml._imports import xgb
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider


# Constants for testing
TEST_VENUE = Venue("SIM")
TEST_SYMBOL = Symbol("EURUSD")
TEST_INSTRUMENT_ID = InstrumentId(TEST_SYMBOL, TEST_VENUE)


@pytest.fixture
def test_instrument() -> CurrencyPair:
    """
    Provide a test CurrencyPair instrument for EURUSD.

    Returns
    -------
    CurrencyPair
        Test EURUSD instrument

    """
    return TestInstrumentProvider.default_fx_ccy("EURUSD", venue=TEST_VENUE)


@pytest.fixture
def test_bar_type() -> BarType:
    """
    Provide a test BarType for 1-minute bars.

    Returns
    -------
    BarType
        Test bar type for 1-minute EURUSD bars

    """
    return BarType.from_str("EURUSD.SIM-1-MINUTE-LAST-EXTERNAL")


@pytest.fixture
def generate_test_bars(test_bar_type: BarType, test_instrument: CurrencyPair) -> list[Bar]:
    """
    Generate realistic test Bar objects with correlated OHLCV data.

    Parameters
    ----------
    test_bar_type : BarType
        The bar type for generated bars
    test_instrument : CurrencyPair
        The instrument for generated bars

    Returns
    -------
    list[Bar]
        List of test bars with realistic price movement

    """
    bars = []
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))
    interval_ns = 60_000_000_000  # 1 minute in nanoseconds

    # Start price around 1.0900 for EURUSD
    current_price = 1.0900

    for i in range(100):  # Generate 100 bars
        # Add realistic random walk with mean reversion
        drift = 0.00001  # Small upward drift
        volatility = 0.0001  # Realistic FX volatility

        # Generate price movement
        returns = np.random.normal(drift, volatility, 4)

        # Calculate OHLC with realistic constraints
        open_price = current_price
        high_price = open_price + abs(returns[0]) * 2
        low_price = open_price - abs(returns[1]) * 2
        close_price = open_price + returns[2]

        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume with some correlation to price movement
        volume = random.uniform(1000, 5000) * (1 + abs(returns[3]) * 10)

        bar = Bar(
            bar_type=test_bar_type,
            open=Price(open_price, precision=5),
            high=Price(high_price, precision=5),
            low=Price(low_price, precision=5),
            close=Price(close_price, precision=5),
            volume=Quantity(volume, precision=0),
            ts_event=base_timestamp + i * interval_ns,
            ts_init=base_timestamp + i * interval_ns + 1000,  # 1 microsecond later
        )

        bars.append(bar)
        current_price = close_price  # Next bar opens at previous close

    return bars


@pytest.fixture
def mock_parquet_catalog(tmp_path: Path, generate_test_bars: list[Bar]) -> ParquetDataCatalog:
    """
    Create a mock ParquetDataCatalog with test data.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path from pytest
    generate_test_bars : list[Bar]
        Test bars to populate the catalog

    Returns
    -------
    ParquetDataCatalog
        Mock catalog with test data

    """
    # Create catalog in temporary directory
    catalog = ParquetDataCatalog(str(tmp_path))

    # Write test bars to catalog (expects flat list, not list of lists)
    catalog.write_data(generate_test_bars)

    return catalog


@pytest.fixture
def test_ml_signals(generate_test_bars: list[Bar]) -> list[dict[str, Any]]:
    """
    Generate test ML signals correlated with bar data.

    Parameters
    ----------
    generate_test_bars : list[Bar]
        Test bars to generate signals from

    Returns
    -------
    list[dict[str, Any]]
        List of ML signal dictionaries

    """
    signals = []

    for i, bar in enumerate(generate_test_bars[1:], 1):  # Skip first bar
        prev_bar = generate_test_bars[i - 1]

        # Generate signal based on simple momentum
        price_change = float(bar.close) - float(prev_bar.close)

        # Threshold for signal generation
        threshold = 0.00005  # 0.5 pips

        if abs(price_change) > threshold:
            prediction = 1 if price_change > 0 else -1
            confidence = min(abs(price_change) / threshold, 1.0)
        else:
            prediction = 0
            confidence = 0.0

        signal = {
            "instrument_id": TEST_INSTRUMENT_ID,
            "timestamp": bar.ts_event,
            "prediction": prediction,  # -1, 0, 1
            "confidence": confidence,
            "features": {
                "price_change": price_change,
                "volume": float(bar.volume),
                "high_low_spread": float(bar.high) - float(bar.low),
            },
        }

        signals.append(signal)

    return signals


@pytest.fixture
def test_ml_config() -> dict[str, Any]:
    """
    Provide test configuration for ML components.

    Returns
    -------
    dict[str, Any]
        Test configuration dictionary

    """
    return {
        "feature_config": {
            "indicators": {
                "sma": {"periods": [10, 20, 50]},
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
            },
            "lookback_period": 20,
            "normalize": True,
        },
        "model_config": {
            "model_type": "xgboost",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "objective": "multi:softprob",
                "num_class": 3,
            },
        },
        "signal_config": {
            "confidence_threshold": 0.7,
            "max_positions": 3,
            "signal_validity_seconds": 60,
        },
        "risk_config": {
            "max_position_size": 0.1,  # 10% of capital
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
            "max_daily_loss": 0.02,  # 2% daily loss limit
        },
    }


@pytest.fixture
def xgboost_test_model(test_ml_config: dict[str, Any]) -> Any:
    """
    Create a simple XGBoost model for testing.

    Parameters
    ----------
    test_ml_config : dict[str, Any]
        Test configuration

    Returns
    -------
    Any
        XGBoost model or None if not available

    """
    if not HAS_XGBOOST:
        pytest.skip("XGBoost not installed")

    # Create dummy training data
    n_samples = 100
    n_features = 10

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 3, n_samples)  # 3 classes: sell, hold, buy

    # Train simple model
    model = xgb.XGBClassifier(**test_ml_config["model_config"]["hyperparameters"])
    model.fit(X, y)

    return model


@pytest.fixture
def lightgbm_test_model(test_ml_config: dict[str, Any]) -> Any:
    """
    Create a simple LightGBM model for testing.

    Parameters
    ----------
    test_ml_config : dict[str, Any]
        Test configuration

    Returns
    -------
    Any
        LightGBM model or None if not available

    """
    if not HAS_LIGHTGBM:
        pytest.skip("LightGBM not installed")

    # Create dummy training data
    n_samples = 100
    n_features = 10

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 3, n_samples)  # 3 classes

    # Train simple model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multiclass",
        num_class=3,
    )
    model.fit(X, y)

    return model


def create_onnx_model_for_features(
    n_features: int,
    tmp_path: Path,
    model_name: str = "test_model.onnx",
) -> Path:
    """
    Create an ONNX model that matches the given feature count.
    
    This creates a simple classifier that accepts n_features as input
    and outputs a probability.
    
    Parameters
    ----------
    n_features : int
        Number of input features the model should accept
    tmp_path : Path
        Temporary directory to save the model
    model_name : str
        Name for the ONNX file
        
    Returns
    -------
    Path
        Path to the created ONNX model
        
    """
    if not HAS_XGBOOST:
        pytest.skip("XGBoost not installed")
    if not HAS_ONNX:
        pytest.skip("ONNX Runtime not installed")
        
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        pytest.skip("onnxmltools not installed (required for XGBoost to ONNX conversion)")
    
    # Create a simple XGBoost model with the right dimensions
    import numpy as np
    import xgboost as xgb
    
    # Generate dummy training data with correct feature count
    X_train = np.random.randn(100, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    
    # Train a simple model
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Convert to ONNX with correct input shape
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    
    # Save to file
    model_path = tmp_path / model_name
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    return model_path


@pytest.fixture
def onnx_test_model_path(xgboost_test_model: Any, tmp_path: Path) -> Path:
    """
    Legacy fixture - creates ONNX model with 10 features for backward compatibility.
    
    For new tests, use create_onnx_model_for_features() directly.
    """
    return create_onnx_model_for_features(10, tmp_path)


@pytest.fixture
def multi_instrument_bars() -> dict[InstrumentId, list[Bar]]:
    """
    Generate correlated bars for multiple instruments.

    Returns
    -------
    dict[InstrumentId, list[Bar]]
        Dictionary mapping instrument IDs to bar lists

    """
    instruments = {
        "EURUSD": 1.0900,  # Base prices
        "GBPUSD": 1.2700,
        "USDJPY": 148.50,
    }

    bars_dict = {}
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))
    interval_ns = 60_000_000_000  # 1 minute

    # Generate correlated returns
    n_bars = 100
    common_factor = np.random.normal(0, 0.0001, n_bars)  # Market factor

    for symbol_str, base_price in instruments.items():
        instrument_id = InstrumentId(Symbol(symbol_str), TEST_VENUE)

        bar_type = BarType.from_str(f"{symbol_str}.{TEST_VENUE}-1-MINUTE-LAST-EXTERNAL")

        bars = []
        current_price = base_price

        for i in range(n_bars):
            # Combine common market factor with idiosyncratic noise
            idio_return = np.random.normal(0, 0.00005)
            total_return = common_factor[i] + idio_return

            open_price = current_price
            close_price = open_price * (1 + total_return)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.00002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.00002)))

            # Adjust precision based on instrument
            precision = 5 if symbol_str != "USDJPY" else 3

            bar = Bar(
                bar_type=bar_type,
                open=Price(open_price, precision=precision),
                high=Price(high_price, precision=precision),
                low=Price(low_price, precision=precision),
                close=Price(close_price, precision=precision),
                volume=Quantity(random.uniform(1000, 5000), precision=0),
                ts_event=base_timestamp + i * interval_ns,
                ts_init=base_timestamp + i * interval_ns + 1000,
            )

            bars.append(bar)
            current_price = close_price

        bars_dict[instrument_id] = bars

    return bars_dict


@pytest.fixture
def test_feature_data() -> pd.DataFrame:
    """
    Generate test feature data for ML training.

    Returns
    -------
    pd.DataFrame
        DataFrame with test features and labels

    """
    n_samples = 1000

    # Generate correlated features
    rng = np.random.default_rng(44)
    base_feature = rng.standard_normal(n_samples)

    features = pd.DataFrame(
        {
            "sma_ratio": base_feature + rng.standard_normal(n_samples) * 0.1,
            "rsi": np.clip(50 + base_feature * 20 + rng.standard_normal(n_samples) * 10, 0, 100),
            "volume_ratio": np.exp(rng.standard_normal(n_samples)),
            "high_low_spread": np.abs(rng.standard_normal(n_samples)) * 0.001,
            "momentum": base_feature * 0.5 + rng.standard_normal(n_samples) * 0.2,
        },
    )

    # Generate labels based on features (with some noise)
    signal = features["sma_ratio"] * 0.3 + features["momentum"] * 0.7
    labels = pd.cut(signal, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)

    features["label"] = labels

    return features
