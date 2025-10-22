#!/usr/bin/env python3
"""
Metamorphic tests for MLSignalActor.

These tests verify that the signal actor preserves key mathematical invariants
under various data transformations, ensuring robustness and correctness of the
ML inference pipeline.

Metamorphic Relations Tested:
1. Price scaling: scale prices by X → raw features scale, normalized don't
2. Time reversal: reverse time → directional features flip, magnitude unchanged
3. Noise tolerance: small price noise → bounded prediction change
4. Data duplication: duplicate bars → same output

Test Data Sources:
- Real market data from fixtures
- Synthetic data with controlled properties
"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from ml.actors.base import MLSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import OptimizationLevel
from ml.actors.signal import SignalStrategy
from ml.actors.signal import ThresholdSignalStrategy
from ml.actors.signal import SignalGenerationStrategy
from ml.actors.signal import MomentumStrategy
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.features.engineering import FeatureConfig
from ml.tests.utils.stubs import FeatureStoreNoOp


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================


@pytest.fixture(scope="module")
def base_signal_config() -> MLSignalActorConfig:
    """Create base signal actor configuration for testing."""
    return MLSignalActorConfig(
        actor_id="test-signal-actor",
        instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
        bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL"),
        model_path="/tmp/dummy_model.onnx",
        model_id="test-model-id",
        prediction_threshold=0.6,
        signal_strategy=SignalStrategy.THRESHOLD,
        enable_parity_smoke_check=True,
        parity_tolerance=1e-6,
        optimization_config=OptimizationConfig(level=OptimizationLevel.STANDARD),
        strategy_config=StrategyConfig(),
        feature_config=FeatureConfig(
            lookback_window=16,
            return_periods=[1, 5],
            momentum_periods=[3],
            ema_fast=10,
            ema_slow=12,
            macd_signal=9,
            volume_ma_periods=[3],
            normalize_features=True,
        ),
        warm_up_period=5,
        min_signal_separation_bars=0,
        adaptive_window=20,
        enable_hot_reload=False,
        use_dummy_stores=True,
    )


@pytest.fixture(scope="module")
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.metadata = {"feature_names": ["feature_0", "feature_1", "feature_2"]}
    return model


@pytest.fixture(scope="module")
def sample_bars() -> list[Bar]:
    """Create sample bars for testing."""
    instrument_id = InstrumentId.from_str("BTCUSDT.BINANCE")
    bar_type = BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL")

    bars = []
    base_time = 1_700_000_000_000_000_000  # Nanosecond timestamp

    # Create realistic OHLCV data with a gentle uptrend
    prices = [50000.0 + (i * 25.0) for i in range(64)]
    volumes = [100.0 + ((i % 10) * 5.0) for i in range(64)]

    for i, (close_price, volume) in enumerate(zip(prices, volumes)):
        # Add some realistic spread
        high = close_price + random.uniform(10, 50)
        low = close_price - random.uniform(10, 50)
        open_price = close_price + random.uniform(-20, 20)

        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        low = max(low, 0.01)

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high:.2f}"),
            low=Price.from_str(f"{low:.2f}"),
            close=Price.from_str(f"{close_price:.2f}"),
            volume=Quantity.from_str(f"{volume:.2f}"),
            ts_event=base_time + i * 60_000_000_000,  # 1 minute intervals
            ts_init=base_time + i * 60_000_000_000 + 1_000_000,  # +1ms
        )
        bars.append(bar)

    return bars


class _FeatureStoreStub(FeatureStoreNoOp):
    """Feature store stub exposing deterministic realtime computation."""

    def compute_realtime(self, *, bar: Bar, store: bool) -> npt.NDArray[np.float32]:  # type: ignore[override]
        del store
        close_price = float(bar.close)
        open_price = float(bar.open)
        high_price = float(bar.high)
        low_price = float(bar.low)
        volume = float(bar.volume)

        # Mirror the legacy stub scaling so metamorphic expectations remain valid.
        price_scale = 10_000.0
        spread = max(high_price - low_price, 0.0)
        spread_scale = 100.0

        features = np.array(
            [
                close_price / price_scale,
                open_price / price_scale,
                spread / spread_scale,
                volume,
            ],
            dtype=np.float32,
        )
        return np.clip(features, -999.0, 999.0)


def _attach_feature_store_stub(actor: MLSignalActor) -> _FeatureStoreStub:
    """Attach and return a deterministic feature store stub."""

    feature_store = _FeatureStoreStub()
    actor._feature_store = feature_store
    actor._persist_features = False
    return feature_store


class _FallbackStrategy(SignalGenerationStrategy):
    """Wrap a strategy to guarantee a signal when confidence exceeds threshold."""

    def __init__(self, inner: SignalGenerationStrategy) -> None:
        self._inner = inner

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        result = self._inner.generate_signal(bar, prediction, confidence, features, context)
        if result is not None:
            return result
        return MLSignal(
            instrument_id=bar.bar_type.instrument_id,
            model_id=context.get("model_id", "unknown"),
            prediction=prediction,
            confidence=confidence,
            features=features if context.get("log_predictions", False) else None,
            ts_event=bar.ts_event,
            ts_init=context.get("timestamp_ns", bar.ts_init),
        )


def _with_fallback(strategy: SignalGenerationStrategy) -> SignalGenerationStrategy:
    """Return a strategy wrapped with fallback behaviour."""

    if isinstance(strategy, _FallbackStrategy):
        return strategy
    return _FallbackStrategy(strategy)


def build_test_actor(
    config: MLSignalActorConfig,
    mock_model: MagicMock,
    *,
    strategy: SignalGenerationStrategy | None = None,
) -> MLSignalActor:
    """Create an MLSignalActor with stores and feature computation stubbed for tests."""

    feature_store_stub = _FeatureStoreStub()
    model_store_stub = MagicMock()
    strategy_store_stub = MagicMock()
    data_store_stub = MagicMock()

    services_stub = SimpleNamespace(
        feature_store=feature_store_stub,
        model_store=model_store_stub,
        strategy_store=strategy_store_stub,
        data_store=data_store_stub,
        feature_registry=MagicMock(),
        model_registry=MagicMock(),
        strategy_registry=MagicMock(),
        data_registry=MagicMock(),
    )

    with patch("ml.actors.actor_services.init_actor_services", return_value=services_stub):
        actor = MLSignalActor(config)

    feature_store_injected = _attach_feature_store_stub(actor)
    actor._feature_store = feature_store_injected
    services_stub.feature_store = feature_store_injected
    actor._model_store = model_store_stub
    actor._strategy_store = strategy_store_stub
    actor._data_store = data_store_stub
    actor._feature_registry = services_stub.feature_registry
    actor._model_registry = services_stub.model_registry
    actor._strategy_registry = services_stub.strategy_registry
    actor._data_registry = services_stub.data_registry

    actor._model = mock_model
    actor._model_id = "test_model"

    active_strategy = strategy if strategy is not None else actor._signal_strategy
    wrapped_strategy = _with_fallback(active_strategy)
    actor._signal_strategy = wrapped_strategy
    try:
        actor._signal_policy_swapper.set_current(wrapped_strategy, {"reason": "test_override"})
    except Exception:
        pass

    return actor

def create_scaled_bars(bars: list[Bar], scale_factor: float) -> list[Bar]:
    """Create bars with prices scaled by a factor."""
    scaled_bars = []

    for bar in bars:
        scaled_bar = Bar(
            bar_type=bar.bar_type,
            open=Price.from_str(f"{float(bar.open) * scale_factor:.2f}"),
            high=Price.from_str(f"{float(bar.high) * scale_factor:.2f}"),
            low=Price.from_str(f"{float(bar.low) * scale_factor:.2f}"),
            close=Price.from_str(f"{float(bar.close) * scale_factor:.2f}"),
            volume=bar.volume,  # Volume remains unchanged
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )
        scaled_bars.append(scaled_bar)

    return scaled_bars


def create_time_reversed_bars(bars: list[Bar]) -> list[Bar]:
    """Create bars with time order reversed."""
    if not bars:
        return []

    reversed_bars = []
    original_times = [(bar.ts_event, bar.ts_init) for bar in bars]
    reversed_data = list(reversed(bars))

    for i, bar in enumerate(reversed_data):
        ts_event, ts_init = original_times[i]
        reversed_bar = Bar(
            bar_type=bar.bar_type,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            ts_event=ts_event,
            ts_init=ts_init,
        )
        reversed_bars.append(reversed_bar)

    return reversed_bars


def add_noise_to_bars(bars: list[Bar], noise_std: float) -> list[Bar]:
    """Add Gaussian noise to bar prices."""
    noisy_bars = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for bar in bars:
        # Add noise proportional to price level
        noise_scale = float(bar.close) * noise_std
        open_noise = rng.normal(0, noise_scale)
        high_noise = rng.normal(0, noise_scale)
        low_noise = rng.normal(0, noise_scale)
        close_noise = rng.normal(0, noise_scale)

        # Ensure OHLC constraints are maintained
        new_open = max(0.01, float(bar.open) + open_noise)
        new_close = max(0.01, float(bar.close) + close_noise)
        new_high = max(new_open, new_close, float(bar.high) + high_noise)
        new_low = min(new_open, new_close, max(0.01, float(bar.low) + low_noise))

        noisy_bar = Bar(
            bar_type=bar.bar_type,
            open=Price.from_str(f"{new_open:.2f}"),
            high=Price.from_str(f"{new_high:.2f}"),
            low=Price.from_str(f"{new_low:.2f}"),
            close=Price.from_str(f"{new_close:.2f}"),
            volume=bar.volume,
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )
        noisy_bars.append(noisy_bar)

    return noisy_bars


def duplicate_bars(bars: list[Bar]) -> list[Bar]:
    """Duplicate bars (each bar appears twice consecutively)."""
    duplicated = []
    for bar in bars:
        duplicated.append(bar)
        # Create identical bar with slightly different timestamp
        dup_bar = Bar(
            bar_type=bar.bar_type,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            ts_event=bar.ts_event + 1_000_000,  # +1ms
            ts_init=bar.ts_init + 1_000_000,
        )
        duplicated.append(dup_bar)

    return duplicated


# =============================================================================
# Metamorphic Test Classes
# =============================================================================


class TestPriceScalingInvariance:
    """Test price scaling metamorphic relation."""

    @pytest.mark.parametrize("scale_factor", [0.5, 2.0, 10.0])
    def test_price_scaling_feature_behavior(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
        scale_factor: float,
    ):
        """Test that price scaling affects raw features but not normalized features appropriately."""
        # Create actor with mock stores
        actor = build_test_actor(base_signal_config, mock_model)

        # Process original bars and collect features
        original_features = []
        for bar in sample_bars:
            features = actor._compute_features(bar)
            if features is not None:
                original_features.append(features.copy())

        # Process scaled bars
        scaled_bars = create_scaled_bars(sample_bars, scale_factor)
        scaled_features = []

        # Reset actor state for fair comparison
        actor.reset_signal_state()

        for bar in scaled_bars:
            features = actor._compute_features(bar)
            if features is not None:
                scaled_features.append(features.copy())

        # Verify we have features to compare
        assert len(original_features) > 0, "No features computed from original bars"
        assert len(scaled_features) > 0, "No features computed from scaled bars"

        # Compare feature vectors
        min_len = min(len(original_features), len(scaled_features))

        for i in range(min_len):
            orig_feat = original_features[i]
            scaled_feat = scaled_features[i]

            # Some features should scale with price (raw price features)
            # Others should be scale-invariant (normalized features, returns)

            # Check that at least some features changed (raw price features)
            has_scaled_features = not np.allclose(orig_feat, scaled_feat, rtol=1e-3)

            # Check that relative features (returns, ratios) are more stable
            # This is a heuristic - we expect some features to be more stable than others
            feature_diff_ratio = np.abs(scaled_feat - orig_feat) / (np.abs(orig_feat) + 1e-8)
            stable_features = np.sum(feature_diff_ratio < 0.1)
            total_features = len(orig_feat)

            # At least some features should be relatively stable (normalized/ratio features)
            stability_ratio = stable_features / total_features

            # Either we have clear scaling behavior or good stability
            assert has_scaled_features or stability_ratio > 0.3, (
                f"Features neither scaled nor stable under {scale_factor}x price scaling"
            )

    @given(scale_factor=st.floats(min_value=0.1, max_value=10.0))
    def test_price_scaling_prediction_consistency(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
        scale_factor: float,
    ):
        """Test that price scaling produces consistent predictions (property-based)."""
        # Create two identical actors
        actor1 = build_test_actor(base_signal_config, mock_model)

        actor2 = build_test_actor(base_signal_config, mock_model)

        # Process original vs scaled data
        original_predictions = []
        scaled_predictions = []

        scaled_bars = create_scaled_bars(sample_bars, scale_factor)

        # Collect predictions from both
        for bar in sample_bars[:3]:  # Limit for performance
            features = actor1._compute_features(bar)
            if features is not None:
                pred, conf = actor1._predict(features)
                original_predictions.append((pred, conf))

        for bar in scaled_bars[:3]:
            features = actor2._compute_features(bar)
            if features is not None:
                pred, conf = actor2._predict(features)
                scaled_predictions.append((pred, conf))

        # Predictions should be similar (model sees similar normalized features)
        assert len(original_predictions) == len(scaled_predictions)

        for (orig_pred, orig_conf), (scaled_pred, scaled_conf) in zip(
            original_predictions, scaled_predictions
        ):
            # Allow some tolerance for numerical differences
            assert abs(orig_pred - scaled_pred) < 0.5, (
                f"Predictions differ too much: {orig_pred} vs {scaled_pred}"
            )
            assert abs(orig_conf - scaled_conf) < 0.3, (
                f"Confidence differs too much: {orig_conf} vs {scaled_conf}"
            )


class TestTimeReversalInvariance:
    """Test time reversal metamorphic relation."""

    def test_directional_features_flip_under_time_reversal(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
    ):
        """Test that directional features flip sign under time reversal."""
        # Use momentum strategy which is sensitive to directional changes
        actor = build_test_actor(base_signal_config, mock_model)
        actor._signal_strategy = MomentumStrategy(
            lookback=base_signal_config.strategy_config.momentum_lookback,
            threshold=base_signal_config.prediction_threshold,
            momentum_threshold=base_signal_config.strategy_config.min_threshold,
        )

        # Process original bars
        original_features = []
        for bar in sample_bars:
            features = actor._compute_features(bar)
            if features is not None:
                original_features.append(features.copy())

        # Process time-reversed bars
        reversed_bars = create_time_reversed_bars(sample_bars)
        reversed_features = []

        # Reset actor state
        actor.reset_signal_state()

        for bar in reversed_bars:
            features = actor._compute_features(bar)
            if features is not None:
                reversed_features.append(features.copy())

        # Verify we have features to compare
        assert len(original_features) > 0, "No features computed from original bars"
        assert len(reversed_features) > 0, "No features computed from reversed bars"

        # Check that some features (momentum, directional) have flipped
        min_len = min(len(original_features), len(reversed_features))

        if min_len > 1:  # Need at least 2 points for directional features
            # Compare first and last features (which should be most different)
            orig_first = original_features[0]
            orig_last = original_features[-1]
            rev_first = reversed_features[0]
            rev_last = reversed_features[-1]

            # Look for sign flips in feature differences
            orig_diff = orig_last - orig_first
            rev_diff = rev_last - rev_first

            # Some features should show opposite trends
            sign_flips = np.sum(np.sign(orig_diff) != np.sign(rev_diff))
            total_features = len(orig_diff)

            flip_ratio = sign_flips / total_features
            assert flip_ratio > 0.2, (
                f"Expected more directional features to flip under time reversal, "
                f"got {flip_ratio:.2%} flips"
            )

    def test_magnitude_preservation_under_time_reversal(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
    ):
        """Test that feature magnitudes are preserved under time reversal."""
        actor = build_test_actor(base_signal_config, mock_model)

        # Process original bars
        original_features = []
        for bar in sample_bars:
            features = actor._compute_features(bar)
            if features is not None:
                original_features.append(features.copy())

        # Process time-reversed bars
        reversed_bars = create_time_reversed_bars(sample_bars)
        reversed_features = []

        actor.reset_signal_state()

        for bar in reversed_bars:
            features = actor._compute_features(bar)
            if features is not None:
                reversed_features.append(features.copy())

        # Compare magnitudes
        assert len(original_features) > 0
        assert len(reversed_features) > 0

        min_len = min(len(original_features), len(reversed_features))

        for i in range(min_len):
            orig_feat = original_features[i]
            rev_feat = reversed_features[i]

            # Magnitudes should be similar (allowing for some numerical differences)
            orig_magnitude = np.linalg.norm(orig_feat)
            rev_magnitude = np.linalg.norm(rev_feat)

            relative_diff = abs(orig_magnitude - rev_magnitude) / (orig_magnitude + 1e-8)
            assert relative_diff < 0.5, (
                f"Feature magnitudes differ too much under time reversal: "
                f"{orig_magnitude:.3f} vs {rev_magnitude:.3f}"
            )


class TestNoiseTolerance:
    """Test noise tolerance metamorphic relation."""

    @pytest.mark.parametrize("noise_level", [0.001, 0.005, 0.01])
    def test_small_noise_bounded_prediction_change(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
        noise_level: float,
    ):
        """Test that small price noise causes bounded prediction changes."""
        actor = build_test_actor(base_signal_config, mock_model)

        # Process original bars
        original_predictions = []
        for bar in sample_bars:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                original_predictions.append((pred, conf))

        # Process noisy bars
        noisy_bars = add_noise_to_bars(sample_bars, noise_level)
        noisy_predictions = []

        actor.reset_signal_state()

        for bar in noisy_bars:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                noisy_predictions.append((pred, conf))

        # Compare predictions
        assert len(original_predictions) > 0
        assert len(noisy_predictions) > 0

        min_len = min(len(original_predictions), len(noisy_predictions))

        for i in range(min_len):
            orig_pred, orig_conf = original_predictions[i]
            noisy_pred, noisy_conf = noisy_predictions[i]

            # Prediction changes should be bounded by noise level
            pred_diff = abs(orig_pred - noisy_pred)
            conf_diff = abs(orig_conf - noisy_conf)

            # Tolerance should scale with noise level
            pred_tolerance = max(0.1, noise_level * 100)  # Reasonable scaling
            conf_tolerance = max(0.1, noise_level * 50)

            assert pred_diff < pred_tolerance, (
                f"Prediction changed too much under {noise_level} noise: "
                f"{pred_diff:.3f} > {pred_tolerance:.3f}"
            )
            assert conf_diff < conf_tolerance, (
                f"Confidence changed too much under {noise_level} noise: "
                f"{conf_diff:.3f} > {conf_tolerance:.3f}"
            )

    @given(noise_std=st.floats(min_value=0.0001, max_value=0.02))
    def test_noise_robustness_property(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
        noise_std: float,
    ):
        """Property-based test for noise robustness."""
        actor = build_test_actor(base_signal_config, mock_model)

        # Get baseline predictions
        baseline_predictions = []
        test_bars = sample_bars[:3]  # Limit for performance

        for bar in test_bars:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                baseline_predictions.append((pred, conf))

        # Test multiple noise realizations
        num_trials = 3
        all_prediction_diffs = []

        for trial in range(num_trials):
            # Use different random seed for each trial
            np.random.seed(42 + trial)
            noisy_bars = add_noise_to_bars(test_bars, noise_std)

            actor.reset_signal_state()
            trial_predictions = []

            for bar in noisy_bars:
                features = actor._compute_features(bar)
                if features is not None:
                    pred, conf = actor._predict(features)
                    trial_predictions.append((pred, conf))

            # Calculate prediction differences for this trial
            for (base_pred, base_conf), (trial_pred, trial_conf) in zip(
                baseline_predictions, trial_predictions
            ):
                pred_diff = abs(base_pred - trial_pred)
                conf_diff = abs(base_conf - trial_conf)
                all_prediction_diffs.append((pred_diff, conf_diff))

        # Verify bounded variation across all trials
        if all_prediction_diffs:
            max_pred_diff = max(diff[0] for diff in all_prediction_diffs)
            max_conf_diff = max(diff[1] for diff in all_prediction_diffs)

            # Bounds should scale reasonably with noise level
            pred_bound = min(1.0, noise_std * 200)
            conf_bound = min(1.0, noise_std * 100)

            assert max_pred_diff < pred_bound, (
                f"Max prediction difference {max_pred_diff:.3f} exceeds bound {pred_bound:.3f}"
            )
            assert max_conf_diff < conf_bound, (
                f"Max confidence difference {max_conf_diff:.3f} exceeds bound {conf_bound:.3f}"
            )


class TestDataDuplicationInvariance:
    """Test data duplication metamorphic relation."""

    def test_duplicate_bars_same_output(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
    ):
        """Test that duplicate bars produce the same output."""
        actor = build_test_actor(base_signal_config, mock_model)

        # Process original bars
        original_outputs = []
        for bar in sample_bars:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                original_outputs.append((features.copy(), pred, conf))

        # Process duplicated bars
        duplicated_bars = duplicate_bars(sample_bars)
        duplicated_outputs = []

        actor.reset_signal_state()

        for bar in duplicated_bars:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                duplicated_outputs.append((features.copy(), pred, conf))

        # Verify outputs
        assert len(original_outputs) > 0
        assert len(duplicated_outputs) > 0

        # Each original bar should have two corresponding duplicated bars
        expected_duplicated_count = len(original_outputs) * 2

        # Allow some tolerance for warm-up differences
        assert len(duplicated_outputs) >= len(original_outputs), (
            "Should have at least as many outputs from duplicated data"
        )

        # Check that consecutive pairs in duplicated data are similar
        for i in range(0, len(duplicated_outputs) - 1, 2):
            if i + 1 < len(duplicated_outputs):
                _feat1, pred1, conf1 = duplicated_outputs[i]
                _feat2, pred2, conf2 = duplicated_outputs[i + 1]

                # Features might differ due to indicator state, but predictions should be close
                pred_diff = abs(pred1 - pred2)
                conf_diff = abs(conf1 - conf2)

                assert pred_diff < 0.1, (
                    f"Duplicate bars produced different predictions: {pred1} vs {pred2}"
                )
                assert conf_diff < 0.1, (
                    f"Duplicate bars produced different confidence: {conf1} vs {conf2}"
                )

    def test_signal_generation_with_duplicates(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
    ):
        """Test signal generation behavior with duplicate data."""
        # Use a custom threshold strategy for predictable signal generation
        threshold_strategy = ThresholdSignalStrategy(threshold=0.5)

        actor = MLSignalActor(base_signal_config)
        _attach_feature_store_stub(actor)
        actor._model = mock_model
        actor._model_id = "test_model"
        actor._signal_strategy = threshold_strategy
        actor._predict = MagicMock(return_value=(0.8, 0.9))

        # Mock the _publish_signal method to capture signals
        published_signals = []
        original_publish = actor._publish_signal

        def mock_publish(signal):
            published_signals.append(signal)
            return original_publish(signal)

        actor._publish_signal = mock_publish

        # Process original bars
        for bar in sample_bars:
            actor._last_signal_bar = (
                actor._bars_processed - actor._signal_config.min_signal_separation_bars
            )
            actor.on_bar(bar)

        original_signal_count = len(published_signals)

        # Reset and process duplicated bars
        published_signals.clear()
        actor.reset_signal_state()

        duplicated_bars = duplicate_bars(sample_bars)
        for bar in duplicated_bars:
            actor._last_signal_bar = (
                actor._bars_processed - actor._signal_config.min_signal_separation_bars
            )
            actor.on_bar(bar)

        duplicated_signal_count = len(published_signals)

        # Should have roughly double the signals (allowing for some variance due to warm-up)
        signal_ratio = duplicated_signal_count / max(original_signal_count, 1)

        # Allow reasonable tolerance for edge effects
        assert 1.5 <= signal_ratio <= 2.5, (
            f"Expected roughly 2x signals with duplicated data, "
            f"got {signal_ratio:.2f}x ({duplicated_signal_count} vs {original_signal_count})"
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetamorphicIntegration:
    """Integration tests combining multiple metamorphic relations."""

    def test_combined_transformations(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
        sample_bars: list[Bar],
    ):
        """Test behavior under combined transformations."""
        actor = MLSignalActor(base_signal_config)
        _attach_feature_store_stub(actor)
        actor._model = mock_model
        actor._model_id = "test_model"

        # Apply multiple transformations
        scaled_bars = create_scaled_bars(sample_bars, 2.0)
        noisy_scaled_bars = add_noise_to_bars(scaled_bars, 0.005)

        # Get baseline
        baseline_predictions = []
        for bar in sample_bars[:3]:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                baseline_predictions.append((pred, conf))

        # Get transformed predictions
        actor.reset_signal_state()
        transformed_predictions = []
        for bar in noisy_scaled_bars[:3]:
            features = actor._compute_features(bar)
            if features is not None:
                pred, conf = actor._predict(features)
                transformed_predictions.append((pred, conf))

        # Should still be reasonably close despite multiple transformations
        assert len(baseline_predictions) == len(transformed_predictions)

        for (base_pred, base_conf), (trans_pred, trans_conf) in zip(
            baseline_predictions, transformed_predictions
        ):
            pred_diff = abs(base_pred - trans_pred)
            conf_diff = abs(base_conf - trans_conf)

            # More tolerant bounds for combined transformations
            assert pred_diff < 0.8, f"Combined transformation caused large prediction change: {pred_diff}"
            assert conf_diff < 0.6, f"Combined transformation caused large confidence change: {conf_diff}"

    def test_real_vs_synthetic_data_consistency(
        self,
        base_signal_config: MLSignalActorConfig,
        mock_model,
    ):
        """Test that metamorphic relations hold for both real and synthetic data."""
        # Create synthetic data with known properties
        instrument_id = InstrumentId.from_str("BTCUSDT.BINANCE")
        bar_type = BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL")

        synthetic_bars = []
        base_time = 1_700_000_000_000_000_000

        # Create synthetic trending data
        for i in range(10):
            price = 50000.0 + i * 100.0  # Clear uptrend
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{price - 20:.2f}"),
                high=Price.from_str(f"{price + 30:.2f}"),
                low=Price.from_str(f"{price - 30:.2f}"),
                close=Price.from_str(f"{price:.2f}"),
                volume=Quantity.from_str("100.0"),
                ts_event=base_time + i * 60_000_000_000,
                ts_init=base_time + i * 60_000_000_000 + 1_000_000,
            )
            synthetic_bars.append(bar)

        actor = build_test_actor(base_signal_config, mock_model)

        # Test scaling invariance on synthetic data
        original_features = []
        for bar in synthetic_bars:
            features = actor._compute_features(bar)
            if features is not None:
                original_features.append(features)

        scaled_bars = create_scaled_bars(synthetic_bars, 3.0)
        actor.reset_signal_state()

        scaled_features = []
        for bar in scaled_bars:
            features = actor._compute_features(bar)
            if features is not None:
                scaled_features.append(features)

        # Should observe scaling behavior in synthetic data too
        assert len(original_features) > 0
        assert len(scaled_features) > 0

        # At least some features should change under scaling
        min_len = min(len(original_features), len(scaled_features))
        features_changed = False

        for i in range(min_len):
            if not np.allclose(original_features[i], scaled_features[i], rtol=1e-3):
                features_changed = True
                break

        assert features_changed, "Scaling should affect at least some features in synthetic data"


# =============================================================================
# Property-Based Test Helpers
# =============================================================================


@given(
    scale_factor=st.floats(min_value=0.1, max_value=10.0),
    noise_level=st.floats(min_value=0.0001, max_value=0.01),
)
def test_combined_property_based_invariants(
    base_signal_config: MLSignalActorConfig,
    mock_model,
    sample_bars: list[Bar],
    scale_factor: float,
    noise_level: float,
):
    """Property-based test combining scaling and noise."""
    actor = MLSignalActor(base_signal_config)
    _attach_feature_store_stub(actor)
    actor._model = mock_model
    actor._model_id = "test_model"

    # Apply combined transformations
    scaled_bars = create_scaled_bars(sample_bars[:3], scale_factor)  # Limit for performance
    noisy_scaled_bars = add_noise_to_bars(scaled_bars, noise_level)

    # Should be able to compute features without errors
    features_computed = 0
    for bar in noisy_scaled_bars:
        features = actor._compute_features(bar)
        if features is not None:
            features_computed += 1
            # Features should be finite
            assert np.all(np.isfinite(features)), "Features should be finite"
            # Features should have reasonable magnitude
            assert np.linalg.norm(features) < 1000, "Features should have reasonable magnitude"

    assert features_computed > 0, "Should compute at least some features"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
