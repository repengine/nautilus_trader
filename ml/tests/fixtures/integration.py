"""
Integration test fixtures consolidated for reuse.

These fixtures were moved from ml/tests/integration/conftest.py to centralize fixture
discovery. Tests can continue to rely on pytest fixture resolution; no direct imports
are required.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

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

_REAL_CATALOG_PREFERRED_SYMBOLS = ("AAPL", "MSFT", "SPY")
_REAL_CATALOG_START_OFFSET_MINUTES = 60
_REAL_CATALOG_WINDOW_MINUTES = 240


@dataclass(frozen=True, slots=True)
class RealCatalogSlice:
    """Slice definition for running tests against a real Parquet catalog."""

    catalog_path: Path
    instrument_id: str
    symbol: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            msg = "real catalog window start must be < end"
            raise ValueError(msg)


def _parse_catalog_timestamp(value: str) -> datetime:
    token = value.rstrip("Z")
    date_raw, time_raw = token.split("T", maxsplit=1)
    year_str, month_str, day_str = date_raw.split("-")
    time_parts = time_raw.split("-")
    if len(time_parts) < 3:
        msg = f"Invalid catalog timestamp: {value}"
        raise ValueError(msg)
    hour_str, minute_str, second_str = time_parts[:3]
    nanos_raw = time_parts[3] if len(time_parts) > 3 else "0"
    nanos_padded = nanos_raw.ljust(9, "0")
    microsecond = int(nanos_padded[:6])
    return datetime(
        int(year_str),
        int(month_str),
        int(day_str),
        int(hour_str),
        int(minute_str),
        int(second_str),
        microsecond,
        tzinfo=UTC,
    )


def _parse_catalog_range(path: Path) -> tuple[datetime, datetime]:
    stem = path.stem
    try:
        start_raw, end_raw = stem.split("_", maxsplit=1)
    except ValueError as exc:
        msg = f"Unexpected catalog filename: {path.name}"
        raise ValueError(msg) from exc
    return (_parse_catalog_timestamp(start_raw), _parse_catalog_timestamp(end_raw))


def _find_real_catalog_bar_dataset(catalog_root: Path) -> tuple[Path, str, str]:
    bar_root = catalog_root / "data" / "bar"
    if not bar_root.exists():
        raise FileNotFoundError("Catalog bar root missing")

    for symbol in _REAL_CATALOG_PREFERRED_SYMBOLS:
        for path in sorted(bar_root.glob(f"{symbol}.*-1-MINUTE-LAST-EXTERNAL")):
            if path.is_dir():
                instrument_id = path.name.split("-1-MINUTE", maxsplit=1)[0]
                return path, instrument_id, symbol

    for path in sorted(bar_root.glob("*-1-MINUTE-LAST-EXTERNAL")):
        if path.is_dir():
            instrument_id = path.name.split("-1-MINUTE", maxsplit=1)[0]
            symbol = instrument_id.split(".", maxsplit=1)[0]
            return path, instrument_id, symbol

    raise FileNotFoundError("No minute bar datasets found")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.data import BuildResult


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

    .. deprecated:: 2025-11-03
        Use test_data_factory.bars() instead for better performance.
        This fixture now delegates to the test_data_factory internally.

        Old pattern:
            def test_bars(generate_test_bars):
                bars = generate_test_bars

        New pattern:
            def test_bars(test_data_factory):
                bars = test_data_factory.bars(n=100)

    This fixture is kept for backward compatibility and will be maintained
    but new tests should use test_data_factory directly.

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
    from ml.tests.fixtures.model_factory import TestDataFactory

    # Delegate to factory (transparent migration)
    factory = TestDataFactory()
    return factory.bars(
        n=100,
        instrument_id=test_instrument.id,
        bar_type=test_bar_type,
    )


@pytest.fixture
def mock_parquet_catalog(tmp_path: Path, generate_test_bars: list[Bar]) -> ParquetDataCatalog:
    """
    Create a mock ParquetDataCatalog with test data.
    """

    catalog = ParquetDataCatalog(str(tmp_path))
    catalog.write_data(generate_test_bars)
    return catalog


@pytest.fixture(scope="session")
def real_catalog_slice() -> RealCatalogSlice:
    """
    Provide a deterministic time slice from the real Parquet catalog.
    """
    catalog_root = Path("data/catalog")
    if not catalog_root.exists():
        pytest.skip("Real catalog not available")

    try:
        dataset_dir, instrument_id, symbol = _find_real_catalog_bar_dataset(catalog_root)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        pytest.skip("Real catalog has no parquet files for selected instrument")

    start_dt, end_dt = _parse_catalog_range(parquet_files[0])
    offset = timedelta(minutes=_REAL_CATALOG_START_OFFSET_MINUTES)
    window = timedelta(minutes=_REAL_CATALOG_WINDOW_MINUTES)
    start = start_dt + offset
    end = start + window

    if end > end_dt:
        end = end_dt - timedelta(minutes=1)
        start = end - window
    if start < start_dt:
        start = start_dt
    if start >= end:
        pytest.skip("Real catalog window too small for deterministic slice")

    return RealCatalogSlice(
        catalog_path=catalog_root,
        instrument_id=instrument_id,
        symbol=symbol,
        start=start,
        end=end,
    )


@pytest.fixture(scope="session")
def real_catalog_dataset(
    real_catalog_slice: RealCatalogSlice,
    tmp_path_factory: pytest.TempPathFactory,
) -> BuildResult:
    """
    Build a bounded TFT dataset from the real catalog for integration tests.
    """
    from ml._imports import HAS_POLARS
    from ml._imports import pl

    if not HAS_POLARS:
        pytest.skip("Polars not installed")

    from ml.data import DatasetBuildConfig
    from ml.data import build_tft_dataset
    from ml.tests.utils.targets import build_default_target_semantics

    out_dir = tmp_path_factory.mktemp("real_catalog_dataset")
    cfg = DatasetBuildConfig(
        data_dir=real_catalog_slice.catalog_path,
        out_dir=out_dir,
        symbols=[real_catalog_slice.symbol],
        instrument_ids=[real_catalog_slice.instrument_id],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        auto_refresh_macro=False,
        target_semantics=build_default_target_semantics(
            horizon_minutes=5,
            threshold=0.0005,
        ),
        lookback_periods=30,
        start=real_catalog_slice.start,
        end=real_catalog_slice.end,
    )

    result = build_tft_dataset(cfg)
    if pl is not None:
        dataset_df = pl.read_parquet(str(result.dataset_parquet))
        if dataset_df.is_empty():
            pytest.skip("Real catalog slice produced empty dataset")
        if not result.feature_names:
            pytest.skip("Real catalog dataset has no feature columns")
    return result


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
                "device": "cpu",
                "tree_method": "hist",
                "predictor": "cpu_predictor",
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

    Requires XGBoost to be installed.
    """
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

    Requires LightGBM to be installed.
    """
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

    Requires ONNX Runtime, XGBoost, and onnxmltools to be installed.
    """
    import numpy as _np
    import xgboost as _xgb
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    rng = _np.random.default_rng(42)
    X_train = rng.standard_normal((200, n_features)).astype(_np.float32)
    y_train = rng.integers(0, 2, 200)

    model = _xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        device="cpu",
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

    model_path = tmp_path / model_name
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return model_path


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
            "volume_ratio_20": np.exp(rng.standard_normal(n_samples)),
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
    "RealCatalogSlice",
    "create_onnx_model_for_features",
    "generate_test_bars",
    "lightgbm_test_model",
    "mock_parquet_catalog",
    "multi_instrument_bars",
    "onnx_test_model_path",
    "real_catalog_dataset",
    "real_catalog_slice",
    "test_bar_type",
    "test_feature_data",
    "test_instrument",
    "test_ml_config",
    "test_ml_signals",
    "xgboost_test_model",
]
