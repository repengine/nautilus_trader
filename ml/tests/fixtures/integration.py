"""
Integration test fixtures consolidated for reuse.

These fixtures were moved from ml/tests/integration/conftest.py to centralize fixture
discovery. Tests can continue to rely on pytest fixture resolution; no direct imports
are required.

"""

from __future__ import annotations

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

    bars: list[Bar] = []
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))
    interval_ns = 60_000_000_000  # 1 minute in nanoseconds

    # Start price around 1.0900 for EURUSD
    current_price = 1.0900

    for i in range(100):  # Generate 100 bars
        drift = 0.00001
        volatility = 0.0001
        from numpy.random import default_rng

        _rng = default_rng(0)
        returns = _rng.normal(drift, volatility, 4)

        open_price = current_price
        high_price = open_price + abs(returns[0]) * 2
        low_price = open_price - abs(returns[1]) * 2
        close_price = open_price + returns[2]

        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        from numpy.random import default_rng as _dr

        _rng_vol = _dr(7)
        volume = float(_rng_vol.uniform(1000, 5000)) * (1 + abs(returns[3]) * 10)

        bar = Bar(
            bar_type=test_bar_type,
            open=Price(open_price, precision=5),
            high=Price(high_price, precision=5),
            low=Price(low_price, precision=5),
            close=Price(close_price, precision=5),
            volume=Quantity(volume, precision=0),
            ts_event=base_timestamp + i * interval_ns,
            ts_init=base_timestamp + i * interval_ns + 1000,
        )

        bars.append(bar)
        current_price = close_price

    return bars


@pytest.fixture
def mock_parquet_catalog(tmp_path: Path, generate_test_bars: list[Bar]) -> ParquetDataCatalog:
    """
    Create a mock ParquetDataCatalog with test data.
    """

    catalog = ParquetDataCatalog(str(tmp_path))
    catalog.write_data(generate_test_bars)
    return catalog


@pytest.fixture
def test_ml_signals(generate_test_bars: list[Bar]) -> list[dict[str, Any]]:
    """
    Generate test ML signals correlated with bar data.
    """

    signals: list[dict[str, Any]] = []
    for i, bar in enumerate(generate_test_bars[1:], 1):
        prev_bar = generate_test_bars[i - 1]
        price_change = float(bar.close) - float(prev_bar.close)
        threshold = 0.00005
        if abs(price_change) > threshold:
            prediction = 1 if price_change > 0 else -1
            confidence = min(abs(price_change) / threshold, 1.0)
        else:
            prediction = 0
            confidence = 0.0
        signal: dict[str, Any] = {
            "instrument_id": TEST_INSTRUMENT_ID,
            "timestamp": bar.ts_event,
            "prediction": prediction,
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
            "max_position_size": 0.1,
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
            "max_daily_loss": 0.02,
        },
    }


@pytest.fixture
def xgboost_test_model(test_ml_config: dict[str, Any]) -> Any:
    """
    Create a simple XGBoost model for testing.
    """

    if not HAS_XGBOOST:
        pytest.skip("XGBoost not installed")

    n_samples = 100
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 3, n_samples)

    model = xgb.XGBClassifier(**test_ml_config["model_config"]["hyperparameters"])
    model.fit(X, y)
    return model


@pytest.fixture
def lightgbm_test_model(test_ml_config: dict[str, Any]) -> Any:
    """
    Create a simple LightGBM model for testing.
    """

    if not HAS_LIGHTGBM:
        pytest.skip("LightGBM not installed")

    n_samples = 100
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 3, n_samples)

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
    """

    if not HAS_ONNX:
        pytest.skip("ONNX not installed")

    try:
        import numpy as _np
        import xgboost as _xgb
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.helpers.onnx_helper import save_onnx_model
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
        from skl2onnx.helpers.onnx_helper import set_model_input_types

        rng = _np.random.default_rng(42)
        X_train = rng.standard_normal((200, n_features)).astype(_np.float32)
        y_train = rng.integers(0, 2, 200)

        model = _xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            random_state=42,
        )
        model.fit(X_train, y_train)

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = to_onnx(model, X_train[:1], target_opset=12, initial_types=initial_type)
        onnx_model = select_model_inputs_outputs(onnx_model, ["float_input", "probabilities"])
        set_model_input_types(onnx_model, initial_type)

        model_path = tmp_path / model_name
        save_onnx_model(onnx_model, str(model_path))
        return model_path
    except Exception:
        # Fallback older converter path (best-effort)
        try:
            from skl2onnx.common.data_types import FloatTensorType
            from skl2onnx import convert_sklearn as convert_xgboost

            rng = np.random.default_rng(42)
            X = rng.standard_normal((200, n_features)).astype(np.float32)
            y = rng.integers(0, 2, 200)
            import xgboost as _xgb2

            model = _xgb2.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1)
            model.fit(X, y)
            initial_type2 = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model2 = convert_xgboost(model, initial_types=initial_type2)
            model_path2 = tmp_path / model_name
            with open(model_path2, "wb") as f:
                f.write(onnx_model2.SerializeToString())
            return model_path2
        except Exception:
            # As a last resort, skip integration tests that require ONNX export
            pytest.skip("ONNX export for XGBoost unavailable in this environment")


@pytest.fixture
def onnx_test_model_path(xgboost_test_model: Any, tmp_path: Path) -> Path:
    """
    Legacy fixture — creates ONNX model with 10 features.
    """

    return create_onnx_model_for_features(10, tmp_path)


@pytest.fixture
def multi_instrument_bars() -> dict[InstrumentId, list[Bar]]:
    """
    Generate correlated bars for multiple instruments.
    """

    instruments: dict[str, float] = {
        "EURUSD": 1.0900,
        "GBPUSD": 1.2700,
        "USDJPY": 148.50,
    }

    bars_dict: dict[InstrumentId, list[Bar]] = {}
    base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))
    interval_ns = 60_000_000_000

    from numpy.random import default_rng

    _rng2 = default_rng(42)
    common_factor = _rng2.normal(0, 0.0001, 100)

    for symbol_str, base_price in instruments.items():
        instrument_id = InstrumentId(Symbol(symbol_str), TEST_VENUE)
        bar_type = BarType.from_str(f"{symbol_str}.{TEST_VENUE}-1-MINUTE-LAST-EXTERNAL")
        bars: list[Bar] = []
        current_price = base_price
        for i in range(100):
            idio_return = _rng2.normal(0, 0.00005)
            total_return = common_factor[i] + idio_return
            open_price = current_price
            close_price = open_price * (1 + total_return)
            high_price = max(open_price, close_price) * (1 + abs(_rng2.normal(0, 0.00002)))
            low_price = min(open_price, close_price) * (1 - abs(_rng2.normal(0, 0.00002)))
            precision = 5 if symbol_str != "USDJPY" else 3
            bar = Bar(
                bar_type=bar_type,
                open=Price(open_price, precision=precision),
                high=Price(high_price, precision=precision),
                low=Price(low_price, precision=precision),
                close=Price(close_price, precision=precision),
                volume=Quantity(float(_rng2.uniform(1000, 5000)), precision=0),
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
    """

    n_samples = 1000
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
    signal = features["sma_ratio"] * 0.3 + features["momentum"] * 0.7
    labels = pd.cut(signal, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)
    features["label"] = labels
    return features


__all__ = [
    "TEST_INSTRUMENT_ID",
    "TEST_SYMBOL",
    "TEST_VENUE",
    "create_onnx_model_for_features",
    "generate_test_bars",
    "lightgbm_test_model",
    "mock_parquet_catalog",
    "multi_instrument_bars",
    "onnx_test_model_path",
    "test_bar_type",
    "test_feature_data",
    "test_instrument",
    "test_ml_config",
    "test_ml_signals",
    "xgboost_test_model",
]
