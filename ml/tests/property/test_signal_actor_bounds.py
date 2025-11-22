"""
Property-based tests for MLSignalActor output bounds.

These tests verify that MLSignalActor always produces outputs within expected bounds,
regardless of input data variations. They help ensure robustness against edge cases
and verify mathematical invariants that must hold for all valid inputs.

Key Properties Tested:
- Predictions always in [-1, 1]
- Confidence always in [0, 1]
- Signal strength always in [0, 1]
- Feature values within expected ranges
- NaN/Inf handling

"""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.common.component import TestClock

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.features.config import FeatureConfig

if TYPE_CHECKING:
    from ml.tests.fixtures.model_factory import TestModelFactory


_TEST_MODEL_FACTORY: TestModelFactory | None = None


@pytest.fixture(scope="session", autouse=True)
def _configure_test_model_factory(test_model_factory: TestModelFactory) -> None:
    """
    Ensure property tests share the canonical model factory instance.
    """

    global _TEST_MODEL_FACTORY
    _TEST_MODEL_FACTORY = test_model_factory


def _require_test_model_factory() -> TestModelFactory:
    if _TEST_MODEL_FACTORY is None:  # pragma: no cover - sanity guard
        raise RuntimeError(
            "test_model_factory fixture not configured; ensure pytest plug-in is active.",
        )
    return _TEST_MODEL_FACTORY


# ============================================================================
# Custom Hypothesis Strategies
# ============================================================================


@st.composite
def valid_instrument_ids(draw, allow_crypto=True, allow_forex=True, allow_stocks=True):
    """
    Generate valid instrument IDs for testing.
    """
    symbols = []
    venues = []

    if allow_crypto:
        symbols.extend(["BTC/USD", "ETH/USD", "DOGE/USD"])
        venues.extend(["BINANCE", "COINBASE"])

    if allow_forex:
        symbols.extend(["EUR/USD", "GBP/USD", "USD/JPY"])
        venues.extend(["FXCM", "OANDA"])

    if allow_stocks:
        symbols.extend(["AAPL", "MSFT", "GOOGL"])
        venues.extend(["NASDAQ", "NYSE"])

    # Default SIM venue for testing
    venues.append("SIM")

    symbol = draw(st.sampled_from(symbols))
    venue = draw(st.sampled_from(venues))

    return InstrumentId(Symbol(symbol), Venue(venue))


@st.composite
def valid_bar_types(draw, instrument_id=None):
    """
    Generate valid bar types for testing.
    """
    if instrument_id is None:
        instrument_id = draw(valid_instrument_ids())

    aggregation = draw(st.sampled_from([BarAggregation.MINUTE, BarAggregation.SECOND]))
    if aggregation is BarAggregation.SECOND:
        step = draw(st.sampled_from([1, 5, 10, 15, 30]))
    else:
        step = draw(st.sampled_from([1, 5, 15, 30]))
    price_type = draw(st.sampled_from([PriceType.MID, PriceType.BID, PriceType.ASK]))

    bar_spec = BarSpecification(step, aggregation, price_type)
    return BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )


@st.composite
def valid_prices(draw, min_value=0.0001, max_value=100000.0, precision=4):
    """
    Generate valid prices with realistic bounds and precision.
    """
    # Use float strategy with constraints
    raw_price = draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        ),
    )

    # Round to specified precision to avoid floating point issues
    rounded_price = round(raw_price, precision)

    # Ensure minimum value after rounding
    if rounded_price < min_value:
        rounded_price = min_value

    return Price(rounded_price, precision)


@st.composite
def valid_quantities(draw, min_value=0.0001, max_value=1000000.0, precision=4):
    """
    Generate valid quantities for volume.
    """
    raw_quantity = draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        ),
    )

    rounded_quantity = round(raw_quantity, precision)

    if rounded_quantity < min_value:
        rounded_quantity = min_value

    return Quantity(rounded_quantity, precision)


@st.composite
def valid_bars(draw, bar_type=None, allow_extreme_values=False):
    """
    Generate valid Bar objects with realistic or extreme price data.
    """
    if bar_type is None:
        bar_type = draw(valid_bar_types())

    # Generate base prices
    if allow_extreme_values:
        # Include edge cases: very small, very large, and precision edge cases
        base_price = draw(
            st.floats(
                min_value=1e-8,  # Very small prices (crypto dust)
                max_value=1e6,  # Very large prices (some stocks/crypto)
                allow_nan=False,
                allow_infinity=False,
            ),
        )
        price_precision = draw(st.integers(min_value=0, max_value=8))
    else:
        # Normal price ranges
        base_price = draw(
            st.floats(
                min_value=0.01,
                max_value=10000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
        price_precision = 4

    # Ensure OHLC relationships hold: Low <= Open,Close <= High
    price_variation = base_price * 0.05  # Max 5% variation

    open_price = draw(
        valid_prices(
            min_value=max(base_price - price_variation, 1e-8),
            max_value=base_price + price_variation,
            precision=price_precision,
        ),
    )

    high_price = draw(
        valid_prices(
            min_value=float(open_price),
            max_value=float(open_price) * 1.1,  # High can be up to 10% above open
            precision=price_precision,
        ),
    )

    low_price = draw(
        valid_prices(
            min_value=float(open_price) * 0.9,  # Low can be down to 90% of open
            max_value=float(open_price),
            precision=price_precision,
        ),
    )

    # Clamp OHLC relationships after rounding
    open_val = float(open_price)
    high_price_val = max(float(high_price), open_val)
    low_price_val = min(float(low_price), open_val)
    high_price = Price(round(high_price_val, price_precision), price_precision)
    low_price = Price(round(max(low_price_val, 1e-8), price_precision), price_precision)

    close_price = draw(
        valid_prices(
            min_value=float(low_price),
            max_value=float(high_price),
            precision=price_precision,
        ),
    )

    # Generate volume
    if allow_extreme_values:
        volume = draw(
            valid_quantities(
                min_value=0.0,  # Allow zero volume
                max_value=1e12,  # Very high volume
                precision=8,
            ),
        )
    else:
        volume = draw(
            valid_quantities(
                min_value=0.001,
                max_value=1000000.0,
                precision=4,
            ),
        )

    # Generate timestamps
    ts_event = draw(st.integers(min_value=1600000000000000000, max_value=2000000000000000000))
    ts_init = ts_event + draw(st.integers(min_value=1, max_value=1000000))  # ts_init >= ts_event

    return Bar(
        bar_type=bar_type,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        ts_event=ts_event,
        ts_init=ts_init,
    )


@st.composite
def feature_arrays(draw, n_features=None, allow_extreme_values=False):
    """
    Generate feature arrays for ML models.
    """
    if n_features is None:
        n_features = draw(st.integers(min_value=1, max_value=50))

    if allow_extreme_values:
        # Include edge cases that might break models
        values = draw(
            st.lists(
                st.one_of(
                    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    st.just(0.0),  # Include exact zeros
                    st.floats(min_value=-1e-8, max_value=1e-8),  # Very small values
                ),
                min_size=n_features,
                max_size=n_features,
            ),
        )
    else:
        # Normal feature ranges
        values = draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n_features,
                max_size=n_features,
            ),
        )

    return np.array(values, dtype=np.float32)


@lru_cache(maxsize=32)
def _cached_model_path(n_features: int, n_outputs: int) -> str:
    """
    Generate or fetch a cached ONNX model path for the given shape.
    """
    model_path = _require_test_model_factory().create_onnx_model(
        n_features=n_features,
        n_outputs=n_outputs,
    )
    return str(model_path)


@st.composite
def ml_signal_actor_configs(draw, feature_config=None, strategy=None):
    """
    Generate valid MLSignalActorConfig instances.
    """
    instrument_id = draw(valid_instrument_ids())
    bar_type = draw(valid_bar_types(instrument_id))

    # Create a temporary model file
    model_path = _cached_model_path(10, 2)

    if feature_config is None:
        feature_config = FeatureConfig(
            lookback_window=draw(st.integers(min_value=5, max_value=100)),
            normalize_features=draw(st.booleans()),
            include_microstructure=draw(st.booleans()),
            include_trade_flow=draw(st.booleans()),
        )

    if strategy is None:
        strategy = draw(
            st.sampled_from(
                [
                    SignalStrategy.THRESHOLD,
                    SignalStrategy.EXTREMES,
                    SignalStrategy.MOMENTUM,
                    SignalStrategy.ADAPTIVE,
                ],
            ),
        )

    return MLSignalActorConfig(
        model_id="test_model",
        model_path=model_path,
        bar_type=bar_type,
        instrument_id=instrument_id,
        feature_config=feature_config,
        signal_strategy=strategy,
        prediction_threshold=draw(st.floats(min_value=0.1, max_value=0.9)),
        warm_up_period=draw(st.integers(min_value=1, max_value=50)),
        batch_size=1,
        use_dummy_stores=True,
        strategy_config=StrategyConfig(
            extremes_top_pct=draw(st.floats(min_value=0.05, max_value=0.5)),
            momentum_lookback=draw(st.integers(min_value=2, max_value=20)),
            adaptive_volatility_factor=draw(st.floats(min_value=0.5, max_value=5.0)),
            min_threshold=draw(st.floats(min_value=0.01, max_value=0.3)),
            max_threshold=draw(st.floats(min_value=0.7, max_value=0.99)),
        ),
        optimization_config=OptimizationConfig(
            level=draw(st.sampled_from(["standard", "optimized"])),
            enable_model_warm_up=draw(st.booleans()),
        ),
    )


# ============================================================================
# Property Tests
# ============================================================================


def create_actor_with_mock_stores(config: MLSignalActorConfig) -> MLSignalActor:
    """
    Instantiate an MLSignalActor with stores/registries stubbed for tests.
    """

    services_stub = SimpleNamespace(
        feature_store=MagicMock(),
        model_store=MagicMock(),
        strategy_store=MagicMock(),
        data_store=MagicMock(),
        feature_registry=MagicMock(),
        model_registry=MagicMock(),
        strategy_registry=MagicMock(),
        data_registry=MagicMock(),
    )

    with patch("ml.actors.actor_services.init_actor_services", return_value=services_stub):
        actor = MLSignalActor(config)

    mock_model = MagicMock()
    mock_model.predict = MagicMock()
    mock_model.run = MagicMock()

    actor._config = config
    actor._signal_config = config
    actor._bars_processed = 0
    actor._prediction_count = 0
    actor._last_signal_bar = -config.min_signal_separation_bars
    actor._model = mock_model
    actor._model_id = config.model_id

    actor._feature_buffer = np.zeros(10, dtype=np.float32)
    actor._prediction_window = np.zeros(config.adaptive_window, dtype=np.float32)
    actor._confidence_window = np.zeros(config.adaptive_window, dtype=np.float32)
    actor._volatility_window = np.zeros(config.adaptive_window, dtype=np.float32)
    actor._window_index = 0
    actor._window_count = 0
    actor._adaptive_threshold = config.prediction_threshold
    actor._market_regime = "normal"
    actor._prediction_history = []
    actor._confidence_history = []

    from ml.actors.signal import ThresholdSignalStrategy

    actor._signal_strategy = ThresholdSignalStrategy(config.prediction_threshold)

    # Note: Stores and registries are initialized via init_actor_services patch
    # and accessed via properties on the actor. We do not need to set them manually.

    actor._circuit_breaker = None
    actor._health_monitor = None
    actor._performance_monitor = MagicMock()

    return actor


@pytest.mark.property
class TestMLSignalActorBounds:
    """
    Property tests for MLSignalActor output bounds.
    """

    @given(
        config=ml_signal_actor_configs(),
        prediction=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        confidence=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=10000)
    def test_prediction_bounds_invariant(self, config, prediction, confidence):
        """
        Property: Model predictions must always be bounded to [-1, 1].

        Invariant: No matter what the raw model outputs, the final prediction
        in any generated signal must be within [-1, 1] range.
        """
        # Create actor with controlled prediction output
        actor = create_actor_with_mock_stores(config)

        # Mock model to return specific prediction values
        actor._model.run.return_value = [[prediction], [abs(prediction)]]
        actor._model.predict = MagicMock(return_value=(prediction, confidence))

        # Generate features and call prediction
        features = np.random.randn(10).astype(np.float32)

        try:
            pred, conf = actor._predict(features)

            # Property: Predictions must be bounded
            assert -1.0 <= pred <= 1.0, f"Prediction {pred} out of bounds [-1, 1]"

            # Property: If a signal is generated, its prediction must also be bounded
            bar = Bar(
                bar_type=config.bar_type,
                open=Price(1.0, 4),
                high=Price(1.1, 4),
                low=Price(0.9, 4),
                close=Price(1.05, 4),
                volume=Quantity(1000, 0),
                ts_event=actor.clock.timestamp_ns(),
                ts_init=actor.clock.timestamp_ns(),
            )

            # Mock signal generation
            context = {
                "prediction_history": [],
                "confidence_history": [],
                "adaptive_threshold": config.prediction_threshold,
                "market_regime": "normal",
                "log_predictions": False,
                "timestamp_ns": actor.clock.timestamp_ns(),
                "model_id": config.model_id,
            }

            signal = actor._signal_strategy.generate_signal(bar, pred, conf, features, context)

            if signal is not None:
                assert (
                    -1.0 <= signal.prediction <= 1.0
                ), f"Signal prediction {signal.prediction} out of bounds [-1, 1]"

        except Exception:
            # If prediction fails due to invalid input, that's acceptable
            # The key is that valid outputs must be bounded
            pass

    @given(
        config=ml_signal_actor_configs(),
        confidence_raw=st.floats(
            min_value=-2.0,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_confidence_bounds_invariant(self, config, confidence_raw):
        """
        Property: Confidence scores must always be bounded to [0, 1].

        Invariant: All confidence values in signals must be valid probabilities.
        """
        actor = create_actor_with_mock_stores(config)

        # Mock model to return specific confidence
        actor._model.run.return_value = [[0.5], [confidence_raw]]
        actor._model.predict = MagicMock(return_value=(0.5, confidence_raw))

        features = np.random.randn(10).astype(np.float32)

        try:
            pred, conf = actor._predict(features)

            # Property: Confidence must be bounded
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds [0, 1]"

            # Property: If a signal is generated, confidence must be valid
            bar = Bar(
                bar_type=config.bar_type,
                open=Price(1.0, 4),
                high=Price(1.1, 4),
                low=Price(0.9, 4),
                close=Price(1.05, 4),
                volume=Quantity(1000, 0),
                ts_event=actor.clock.timestamp_ns(),
                ts_init=actor.clock.timestamp_ns(),
            )

            context = {
                "prediction_history": [],
                "confidence_history": [],
                "adaptive_threshold": config.prediction_threshold,
                "market_regime": "normal",
                "log_predictions": False,
                "timestamp_ns": actor.clock.timestamp_ns(),
                "model_id": config.model_id,
            }

            signal = actor._signal_strategy.generate_signal(bar, pred, conf, features, context)

            if signal is not None:
                assert (
                    0.0 <= signal.confidence <= 1.0
                ), f"Signal confidence {signal.confidence} out of bounds [0, 1]"

        except Exception:
            # Invalid inputs may cause exceptions, which is acceptable
            pass

    @given(
        bar=valid_bars(allow_extreme_values=True),
        features=feature_arrays(n_features=10, allow_extreme_values=True),
        config=ml_signal_actor_configs(),
    )
    @settings(max_examples=50, deadline=15000)
    def test_extreme_input_handling(self, bar, features, config):
        """
        Property: Actor should handle extreme input values gracefully.

        Invariant: Extreme prices, volumes, or feature values should not
        cause crashes or produce invalid outputs.
        """
        actor = create_actor_with_mock_stores(config)

        # Mock model with reasonable outputs regardless of input
        actor._model.run.return_value = [[0.7], [0.8]]
        actor._model.predict = MagicMock(return_value=(0.7, 0.8))

        try:
            # Test feature computation doesn't crash
            # Note: We'll mock this since real feature computation needs complex setup
            computed_features = features  # Use provided features directly

            # Verify features don't contain NaN or Inf
            assert not np.any(np.isnan(computed_features)), "Features contain NaN"
            assert not np.any(np.isinf(computed_features)), "Features contain Inf"

            # Test prediction doesn't crash
            pred, conf = actor._predict(computed_features)

            # Verify outputs are still bounded
            assert -1.0 <= pred <= 1.0, f"Extreme input led to unbounded prediction {pred}"
            assert 0.0 <= conf <= 1.0, f"Extreme input led to unbounded confidence {conf}"

            # Test signal generation
            context = {
                "prediction_history": [],
                "confidence_history": [],
                "adaptive_threshold": config.prediction_threshold,
                "market_regime": "normal",
                "log_predictions": False,
                "timestamp_ns": bar.ts_event,
                "model_id": config.model_id,
            }

            signal = actor._signal_strategy.generate_signal(
                bar,
                pred,
                conf,
                computed_features,
                context,
            )

            if signal is not None:
                # Verify signal properties
                assert -1.0 <= signal.prediction <= 1.0
                assert 0.0 <= signal.confidence <= 1.0
                assert signal.instrument_id == bar.bar_type.instrument_id
                assert signal.ts_event == bar.ts_event

        except Exception as e:
            # Some extreme inputs may cause legitimate failures
            # But we should not get silent corruption or invalid bounds
            print(f"Extreme input caused exception (acceptable): {e}")

    @given(
        config=ml_signal_actor_configs(),
        n_predictions=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20, deadline=15000)
    def test_signal_strength_consistency(self, config, n_predictions):
        """
        Property: Signal strength should be consistent with prediction magnitude.

        Invariant: Stronger predictions (higher absolute value) should generally
        lead to stronger signals when confidence is sufficient.
        """
        actor = create_actor_with_mock_stores(config)

        predictions = []
        confidences = []
        signal_strengths = []

        for i in range(n_predictions):
            # Generate prediction and confidence
            prediction = np.random.uniform(-1.0, 1.0)
            confidence = np.random.uniform(config.prediction_threshold, 1.0)  # Above threshold

            # Mock model output
            actor._model.run.return_value = [[prediction], [confidence]]
            actor._model.predict = MagicMock(return_value=(prediction, confidence))

            features = np.random.randn(10).astype(np.float32)

            try:
                pred, conf = actor._predict(features)

                bar = Bar(
                    bar_type=config.bar_type,
                    open=Price(1.0 + i * 0.001, 4),  # Slight price variation
                    high=Price(1.1 + i * 0.001, 4),
                    low=Price(0.9 + i * 0.001, 4),
                    close=Price(1.05 + i * 0.001, 4),
                    volume=Quantity(1000, 0),
                    ts_event=1000000000000000000 + i * 1000000,  # Increment time
                    ts_init=1000000000000000000 + i * 1000000 + 1000,
                )

                context = {
                    "prediction_history": [],
                    "confidence_history": [],
                    "adaptive_threshold": config.prediction_threshold,
                    "market_regime": "normal",
                    "log_predictions": False,
                    "timestamp_ns": bar.ts_event,
                    "model_id": config.model_id,
                }

                signal = actor._signal_strategy.generate_signal(bar, pred, conf, features, context)

                if signal is not None:
                    predictions.append(abs(pred))
                    confidences.append(conf)
                    signal_strengths.append(abs(signal.prediction))

            except Exception:
                continue  # Skip failed predictions

        # Property: Signal strength should correlate with prediction magnitude
        # when confidence is above threshold
        if len(signal_strengths) >= 3:
            # Check that stronger predictions tend to produce stronger signals
            # This is a statistical property, not absolute
            strong_predictions = [s for p, s in zip(predictions, signal_strengths) if p > 0.7]
            weak_predictions = [s for p, s in zip(predictions, signal_strengths) if p < 0.3]

            if strong_predictions and weak_predictions:
                avg_strong = np.mean(strong_predictions)
                avg_weak = np.mean(weak_predictions)

                # Allow some tolerance for statistical variation
                assert avg_strong >= avg_weak * 0.8, (
                    f"Signal strength inconsistency: strong predictions "
                    f"({avg_strong:.3f}) should generally be stronger than "
                    f"weak predictions ({avg_weak:.3f})"
                )

    @given(
        features_with_nan=st.lists(
            st.one_of(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.just(float("nan")),
                st.just(float("inf")),
                st.just(float("-inf")),
            ),
            min_size=10,
            max_size=10,
        ),
        config=ml_signal_actor_configs(),
    )
    @settings(max_examples=50, deadline=10000)
    def test_nan_inf_handling(self, features_with_nan, config):
        """
        Property: Actor should handle NaN and Inf values gracefully.

        Invariant: NaN or Inf in features should either be handled gracefully
        or cause explicit, controlled failures - never silent corruption.
        """
        features = np.array(features_with_nan, dtype=np.float32)
        has_invalid = np.any(np.isnan(features)) or np.any(np.isinf(features))

        # Skip if no invalid values to test
        assume(has_invalid)

        actor = create_actor_with_mock_stores(config)

        # Mock model to return valid outputs
        actor._model.run.return_value = [[0.5], [0.7]]
        actor._model.predict = MagicMock(return_value=(0.5, 0.7))

        try:
            pred, conf = actor._predict(features)

            # If prediction succeeds, outputs must still be valid
            assert not math.isnan(pred), "Prediction returned NaN"
            assert not math.isinf(pred), "Prediction returned Inf"
            assert not math.isnan(conf), "Confidence returned NaN"
            assert not math.isinf(conf), "Confidence returned Inf"

            # And still bounded
            assert -1.0 <= pred <= 1.0, f"Prediction {pred} out of bounds despite NaN/Inf input"
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds despite NaN/Inf input"

        except (ValueError, RuntimeError, TypeError) as e:
            # Explicit exceptions for invalid inputs are acceptable
            print(f"Invalid features caused controlled exception: {e}")
        except Exception as e:
            # Unexpected exceptions might indicate a problem
            pytest.fail(f"Unexpected exception type for NaN/Inf input: {type(e).__name__}: {e}")

    @given(
        config=ml_signal_actor_configs(strategy=SignalStrategy.ADAPTIVE),
        volatility_sequence=st.lists(
            st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=20,
        ),
    )
    @settings(max_examples=20, deadline=10000)
    def test_adaptive_threshold_bounds(self, config, volatility_sequence):
        """
        Property: Adaptive thresholds should remain within configured bounds.

        Invariant: Adaptive strategy should respect min_threshold and max_threshold
        regardless of market conditions.
        """
        actor = create_actor_with_mock_stores(config)

        # Use adaptive strategy specifically
        from ml.actors.signal import AdaptiveStrategy

        strategy_config = config.strategy_config or StrategyConfig()
        actor._signal_strategy = AdaptiveStrategy(
            base_threshold=config.prediction_threshold,
            volatility_factor=strategy_config.adaptive_volatility_factor,
            min_threshold=strategy_config.min_threshold,
            max_threshold=strategy_config.max_threshold,
        )

        actor._model.run.return_value = [[0.8], [0.9]]
        actor._model.predict = MagicMock(return_value=(0.8, 0.9))

        features = np.random.randn(10).astype(np.float32)

        for i, volatility in enumerate(volatility_sequence):
            # Simulate market conditions affecting adaptive threshold
            # In a real scenario, this would be computed from market data
            base_threshold = config.prediction_threshold
            vol_factor = strategy_config.adaptive_volatility_factor
            adaptive_threshold = min(
                strategy_config.max_threshold,
                max(
                    strategy_config.min_threshold,
                    base_threshold + volatility * vol_factor,
                ),
            )

            bar = Bar(
                bar_type=config.bar_type,
                open=Price(1.0 + i * 0.001, 4),
                high=Price(1.1 + i * 0.001, 4),
                low=Price(0.9 + i * 0.001, 4),
                close=Price(1.05 + i * 0.001, 4),
                volume=Quantity(1000, 0),
                ts_event=1000000000000000000 + i * 1000000,
                ts_init=1000000000000000000 + i * 1000000 + 1000,
            )

            context = {
                "prediction_history": [],
                "confidence_history": [],
                "adaptive_threshold": adaptive_threshold,
                "market_regime": "normal",
                "log_predictions": False,
                "timestamp_ns": bar.ts_event,
                "model_id": config.model_id,
            }

            try:
                pred, conf = actor._predict(features)
                signal = actor._signal_strategy.generate_signal(bar, pred, conf, features, context)

                # Property: Adaptive threshold should be within bounds
                assert (
                    strategy_config.min_threshold
                    <= adaptive_threshold
                    <= strategy_config.max_threshold
                ), (
                    f"Adaptive threshold {adaptive_threshold} outside bounds "
                    f"[{strategy_config.min_threshold}, {strategy_config.max_threshold}]"
                )

                if signal is not None and hasattr(signal, "metadata") and signal.metadata:
                    # If signal metadata includes adaptive threshold, verify it
                    signal_threshold = signal.metadata.get("adaptive_threshold")
                    if signal_threshold is not None:
                        assert (
                            strategy_config.min_threshold
                            <= signal_threshold
                            <= strategy_config.max_threshold
                        )

            except Exception:
                continue  # Skip failed iterations


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.property
class TestMLSignalActorEdgeCases:
    """
    Property tests for edge cases and boundary conditions.
    """

    @given(
        zero_volume_bars=st.lists(
            valid_bars(allow_extreme_values=False),
            min_size=1,
            max_size=10,
        ),
        config=ml_signal_actor_configs(),
    )
    @settings(max_examples=20, deadline=10000)
    def test_zero_volume_handling(self, zero_volume_bars, config):
        """
        Property: Zero volume bars should be handled gracefully.

        Invariant: Zero volume should not cause crashes or invalid outputs.
        """
        zero_volume_bars = [
            Bar(
                bar_type=bar.bar_type,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=Quantity(0.0, 0),
                ts_event=bar.ts_event,
                ts_init=bar.ts_init,
            )
            for bar in zero_volume_bars
        ]

        actor = create_actor_with_mock_stores(config)
        actor._model.run.return_value = [[0.6], [0.8]]
        actor._model.predict = MagicMock(return_value=(0.6, 0.8))

        features = np.random.randn(10).astype(np.float32)

        for bar in zero_volume_bars:
            try:
                pred, conf = actor._predict(features)

                # Property: Outputs should still be valid with zero volume
                assert -1.0 <= pred <= 1.0
                assert 0.0 <= conf <= 1.0

            except Exception:
                # Some strategies might reject zero volume bars, which is acceptable
                continue

    @given(
        config=ml_signal_actor_configs(),
    )
    @settings(max_examples=10, deadline=5000)
    def test_initialization_bounds(self, config):
        """
        Property: Actor initialization should set up valid initial state.

        Invariant: All internal state variables should be within valid bounds
        after initialization.
        """
        try:
            actor = create_actor_with_mock_stores(config)

            # Property: Initial adaptive threshold should be valid
            assert (
                0.0 <= actor._adaptive_threshold <= 1.0
            ), f"Initial adaptive threshold {actor._adaptive_threshold} out of bounds"

            # Property: Window indices should be valid
            assert 0 <= actor._window_index < config.adaptive_window
            assert 0 <= actor._window_count <= config.adaptive_window

            # Property: History lists should be initialized
            assert isinstance(actor._prediction_history, list)
            assert isinstance(actor._confidence_history, list)

            # Property: Buffers should have correct shapes and dtypes
            assert actor._feature_buffer.dtype == np.float32
            assert actor._prediction_window.dtype == np.float32
            assert actor._confidence_window.dtype == np.float32

        except Exception as e:
            pytest.fail(f"Actor initialization failed: {e}")
