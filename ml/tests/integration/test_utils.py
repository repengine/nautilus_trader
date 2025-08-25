"""
Helper functions for ML integration testing.

This module provides utilities for:
- Generating realistic OHLCV data
- Creating correlated multi-instrument data
- Validating feature parity with extreme precision
- Creating test models (XGBoost/LightGBM stubs)

"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_XGBOOST
from ml._imports import lgb
from ml._imports import xgb
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


def generate_realistic_ohlcv(
    instrument_id: InstrumentId,
    start_time: datetime,
    n_bars: int = 1000,
    bar_interval_minutes: int = 1,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    price_precision: int = 2,
    volume_mean: float = 10000.0,
    volume_std: float = 2000.0,
) -> list[Bar]:
    """
    Generate realistic OHLCV data with specified characteristics.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument to generate bars for
    start_time : datetime
        Starting timestamp for the bars
    n_bars : int, default 1000
        Number of bars to generate
    bar_interval_minutes : int, default 1
        Interval between bars in minutes
    base_price : float, default 100.0
        Starting price level
    volatility : float, default 0.02
        Price volatility (standard deviation of returns)
    trend : float, default 0.0001
        Drift/trend in price (mean return)
    price_precision : int, default 2
        Decimal precision for prices
    volume_mean : float, default 10000.0
        Mean volume per bar
    volume_std : float, default 2000.0
        Standard deviation of volume

    Returns
    -------
    list[Bar]
        List of generated bars with realistic OHLCV data

    """
    # Initialize random generator for reproducibility
    rng = np.random.default_rng(42)

    bar_type = BarType.from_str(
        f"{instrument_id}-{bar_interval_minutes}-MINUTE-LAST-EXTERNAL",
    )

    bars = []
    current_price = base_price
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(start_time))
    interval_ns = bar_interval_minutes * 60_000_000_000  # Convert to nanoseconds

    for i in range(n_bars):
        # Generate log returns for more realistic price movement
        log_return = rng.normal(trend, volatility)

        # Calculate OHLC with realistic intrabar movement
        open_price = current_price

        # Simulate intrabar price path with multiple ticks
        n_ticks = 20
        tick_returns = rng.normal(0, volatility / np.sqrt(n_ticks), n_ticks)
        tick_prices = open_price * np.exp(np.cumsum(tick_returns))

        high_price = np.max(tick_prices)
        low_price = np.min(tick_prices)
        close_price = open_price * np.exp(log_return)

        # Ensure OHLC constraints
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume with correlation to price movement
        volume_multiplier = 1 + abs(log_return) * 10  # Higher volume on larger moves
        volume = max(100, rng.normal(volume_mean * volume_multiplier, volume_std))

        bar = Bar(
            bar_type=bar_type,
            open=Price(open_price, precision=price_precision),
            high=Price(high_price, precision=price_precision),
            low=Price(low_price, precision=price_precision),
            close=Price(close_price, precision=price_precision),
            volume=Quantity(int(volume), precision=0),
            ts_event=base_timestamp + i * interval_ns,
            ts_init=base_timestamp + i * interval_ns + 1000,  # 1 microsecond later
        )

        bars.append(bar)
        current_price = close_price

    return bars


def create_correlated_multi_instrument_data(
    instruments: dict[str, float],
    correlation_matrix: npt.NDArray[np.float64] | None = None,
    start_time: datetime = datetime(2024, 1, 1),
    n_bars: int = 1000,
    common_volatility: float = 0.01,
    idiosyncratic_volatility: float = 0.005,
) -> dict[InstrumentId, list[Bar]]:
    """
    Create correlated bar data for multiple instruments.

    Parameters
    ----------
    instruments : dict[str, float]
        Mapping of instrument symbols to base prices
    correlation_matrix : npt.NDArray[np.float64], optional
        Correlation matrix for returns. If None, uses default correlations
    start_time : datetime
        Starting timestamp for the data
    n_bars : int, default 1000
        Number of bars to generate per instrument
    common_volatility : float, default 0.01
        Volatility of common market factor
    idiosyncratic_volatility : float, default 0.005
        Volatility of instrument-specific returns

    Returns
    -------
    dict[InstrumentId, list[Bar]]
        Dictionary mapping instrument IDs to bar lists

    """
    n_instruments = len(instruments)

    # Initialize random generator for reproducibility
    rng = np.random.default_rng(42)

    # Default correlation matrix if not provided
    if correlation_matrix is None:
        # Create a realistic correlation matrix
        correlation_matrix = np.eye(n_instruments)
    from numpy.random import default_rng

    _rng = default_rng(0)
    for i in range(n_instruments):
        for j in range(i + 1, n_instruments):
            corr = 0.3 + float(_rng.random()) * 0.4  # Correlations between 0.3 and 0.7
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    # Generate correlated returns using Cholesky decomposition
    L = np.linalg.cholesky(correlation_matrix)

    # Generate independent random returns
    independent_returns = rng.normal(0, 1, (n_bars, n_instruments))

    # Create correlated returns
    correlated_returns = independent_returns @ L.T * common_volatility

    # Add idiosyncratic component
    idiosyncratic_returns = rng.normal(0, idiosyncratic_volatility, (n_bars, n_instruments))
    total_returns = correlated_returns + idiosyncratic_returns

    # Generate bars for each instrument
    result = {}
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(start_time))
    interval_ns = 60_000_000_000  # 1 minute

    for idx, (symbol, base_price) in enumerate(instruments.items()):
        instrument_id = InstrumentId(Symbol(symbol), Venue("SIM"))

        bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")

        bars = []
        current_price = base_price

        for i in range(n_bars):
            ret = total_returns[i, idx]

            open_price = current_price
            close_price = open_price * (1 + ret)

            # Add realistic high/low
            high_offset = abs(rng.normal(0, idiosyncratic_volatility * 0.5))
            low_offset = abs(rng.normal(0, idiosyncratic_volatility * 0.5))

            high_price = max(open_price, close_price) * (1 + high_offset)
            low_price = min(open_price, close_price) * (1 - low_offset)

            # Determine precision based on typical FX conventions
            if "JPY" in symbol:
                precision = 3
            else:
                precision = 5

            volume = rng.lognormal(np.log(10000), 0.5)

            bar = Bar(
                bar_type=bar_type,
                open=Price(open_price, precision=precision),
                high=Price(high_price, precision=precision),
                low=Price(low_price, precision=precision),
                close=Price(close_price, precision=precision),
                volume=Quantity(int(volume), precision=0),
                ts_event=base_timestamp + i * interval_ns,
                ts_init=base_timestamp + i * interval_ns + 1000,
            )

            bars.append(bar)
            current_price = close_price

        result[instrument_id] = bars

    return result


def validate_feature_parity(
    batch_features: npt.NDArray[np.float64],
    online_features: npt.NDArray[np.float64],
    tolerance: float = 1e-10,
    feature_names: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate that batch and online feature calculations match with extreme precision.

    Parameters
    ----------
    batch_features : npt.NDArray[np.float64]
        Features calculated in batch mode (training)
    online_features : npt.NDArray[np.float64]
        Features calculated online (inference)
    tolerance : float, default 1e-10
        Maximum allowed relative tolerance
    feature_names : list[str], optional
        Names of features for detailed reporting

    Returns
    -------
    tuple[bool, dict[str, Any]]
        (is_valid, detailed_report) where is_valid indicates if features match
        and detailed_report contains statistics about differences

    """
    if batch_features.shape != online_features.shape:
        return False, {
            "error": "Shape mismatch",
            "batch_shape": batch_features.shape,
            "online_shape": online_features.shape,
        }

    # Calculate various difference metrics
    abs_diff = np.abs(batch_features - online_features)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(
            batch_features != 0,
            abs_diff / np.abs(batch_features),
            abs_diff,  # Use absolute diff where batch is zero
        )

    # Detailed statistics
    report = {
        "shape": batch_features.shape,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "n_exact_matches": int(np.sum(batch_features == online_features)),
        "n_within_tolerance": int(np.sum(rel_diff <= tolerance)),
        "total_features": int(batch_features.size),
    }

    # Per-feature analysis if names provided
    if feature_names is not None and len(feature_names) == batch_features.shape[1]:
        feature_report = {}
        for i, name in enumerate(feature_names):
            feature_abs_diff = abs_diff[:, i]
            feature_rel_diff = rel_diff[:, i]

            feature_report[name] = {
                "max_abs_diff": float(np.max(feature_abs_diff)),
                "mean_abs_diff": float(np.mean(feature_abs_diff)),
                "max_rel_diff": float(np.max(feature_rel_diff)),
                "mean_rel_diff": float(np.mean(feature_rel_diff)),
                "within_tolerance": bool(np.all(feature_rel_diff <= tolerance)),
            }

        report["per_feature"] = feature_report

    # Check if all differences are within tolerance
    is_valid = bool(np.all(rel_diff <= tolerance))
    report["is_valid"] = is_valid
    report["tolerance"] = tolerance

    return is_valid, report


def create_test_xgboost_model(
    n_features: int = 10,
    n_classes: int = 3,
    n_samples: int = 1000,
    **kwargs: Any,
) -> Any:
    """
    Create a simple XGBoost model for testing purposes.

    Parameters
    ----------
    n_features : int, default 10
        Number of input features
    n_classes : int, default 3
        Number of output classes
    n_samples : int, default 1000
        Number of training samples
    **kwargs
        Additional parameters for XGBClassifier

    Returns
    -------
    Any
        Trained XGBoost model or None if XGBoost not available

    """
    if not HAS_XGBOOST:
        return None

    # Generate synthetic training data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Create labels with some structure (not completely random)
    # This makes the model learn something meaningful
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n_samples) * 0.2
    y = pd.cut(signal, bins=n_classes, labels=False).astype(int)

    # Default parameters for testing
    default_params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        "random_state": 42,
    }

    if n_classes > 2:
        default_params["num_class"] = n_classes

    # Override with provided parameters
    default_params.update(kwargs)

    # Train model
    model = xgb.XGBClassifier(**default_params)
    model.fit(X, y)

    return model


def create_test_lightgbm_model(
    n_features: int = 10,
    n_classes: int = 3,
    n_samples: int = 1000,
    **kwargs: Any,
) -> Any:
    """
    Create a simple LightGBM model for testing purposes.

    Parameters
    ----------
    n_features : int, default 10
        Number of input features
    n_classes : int, default 3
        Number of output classes
    n_samples : int, default 1000
        Number of training samples
    **kwargs
        Additional parameters for LGBMClassifier

    Returns
    -------
    Any
        Trained LightGBM model or None if LightGBM not available

    """
    if not HAS_LIGHTGBM:
        return None

    # Generate synthetic training data
    rng = np.random.default_rng(43)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Create structured labels
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n_samples) * 0.2
    y = pd.cut(signal, bins=n_classes, labels=False).astype(int)

    # Default parameters for testing
    default_params: dict[str, Any] = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "multiclass" if n_classes > 2 else "binary",
        "random_state": 42,
        "verbose": -1,  # Suppress output during testing
    }

    if n_classes > 2:
        default_params["num_class"] = n_classes

    # Override with provided parameters
    default_params.update(kwargs)

    # Train model
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y)

    return model


def compute_nautilus_indicators(bars: list[Bar]) -> pd.DataFrame:
    """
    Compute standard Nautilus indicators for testing feature parity.

    Parameters
    ----------
    bars : list[Bar]
        List of bars to compute indicators from

    Returns
    -------
    pd.DataFrame
        DataFrame with computed indicator values

    """
    # Initialize indicators
    sma_10 = SimpleMovingAverage(10)
    sma_20 = SimpleMovingAverage(20)
    rsi = RelativeStrengthIndex(14)
    macd = MovingAverageConvergenceDivergence(
        fast_period=12,
        slow_period=26,
    )

    # Compute indicators
    results = []

    for bar in bars:
        # Update indicators
        sma_10.update_raw(float(bar.close))
        sma_20.update_raw(float(bar.close))
        rsi.update_raw(float(bar.close))
        macd.update_raw(float(bar.close))

        # Store results
        result = {
            "timestamp": bar.ts_event,
            "close": float(bar.close),
            "volume": float(bar.volume),
            "sma_10": float(sma_10.value) if sma_10.initialized else np.nan,
            "sma_20": float(sma_20.value) if sma_20.initialized else np.nan,
            "rsi": float(rsi.value) if rsi.initialized else np.nan,
            "macd": float(macd.value) if macd.initialized else np.nan,
        }

        results.append(result)

    return pd.DataFrame(results)


def generate_mock_ml_signals(
    bars: list[Bar],
    signal_frequency: float = 0.1,
    confidence_mean: float = 0.75,
    confidence_std: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Generate mock ML signals based on bar data.

    Parameters
    ----------
    bars : list[Bar]
        Bars to generate signals from
    signal_frequency : float, default 0.1
        Probability of generating a signal for each bar
    confidence_mean : float, default 0.75
        Mean confidence level for signals
    confidence_std : float, default 0.15
        Standard deviation of confidence

    Returns
    -------
    list[dict[str, Any]]
        List of ML signal dictionaries

    """
    signals = []

    # Initialize random generator for reproducibility
    rng = np.random.default_rng(42)

    for i, bar in enumerate(bars[1:], 1):
        # Randomly decide whether to generate a signal
        if float(rng.random()) > signal_frequency:
            continue

        prev_bar = bars[i - 1]

        # Simple momentum-based signal
        price_change = (float(bar.close) - float(prev_bar.close)) / float(prev_bar.close)
        volume_ratio = (
            float(bar.volume) / float(prev_bar.volume) if float(prev_bar.volume) > 0 else 1.0
        )

        # Determine signal direction
        if abs(price_change) < 0.0001:  # No significant move
            continue

        prediction = 1 if price_change > 0 else -1

        # Generate confidence with some correlation to price change magnitude
        base_confidence = min(abs(price_change) * 100, 1.0)
        confidence = np.clip(
            rng.normal(
                base_confidence * confidence_mean,
                confidence_std,
            ),
            0.5,
            1.0,
        )

        signal = {
            "instrument_id": bar.bar_type.instrument_id,
            "timestamp": bar.ts_event,
            "prediction": prediction,
            "confidence": float(confidence),
            "price_change": price_change,
            "volume_ratio": volume_ratio,
            "bar_index": i,
        }

        signals.append(signal)

    return signals
