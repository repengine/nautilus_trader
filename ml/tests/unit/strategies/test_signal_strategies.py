"""
Unit tests for Signal Generation Strategies and Plugin Architecture.

Tests the new strategy pattern implementation including:
- All built-in signal generation strategies
- Plugin architecture for custom strategies
- Optimization levels (STANDARD vs OPTIMIZED)
- Configuration system

"""

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import AdaptiveStrategy
from ml.actors.signal import EnsembleStrategy
from ml.actors.signal import ExtremesStrategy
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import MomentumStrategy
from ml.actors.signal import OptimizationConfig
from ml.actors.signal import OptimizationLevel
from ml.actors.signal import SignalGenerationStrategy
from ml.actors.signal import SignalStrategy
from ml.actors.signal import StrategyConfig
from ml.actors.signal import ThresholdSignalStrategy
from ml.actors.signal import ThresholdStrategy
from nautilus_trader.common.component import TestClock
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class CustomSignalStrategy(SignalGenerationStrategy):
    """
    Custom signal generation strategy for testing plugin architecture.
    """

    def __init__(self, multiplier: float = 2.0) -> None:
        self.multiplier = multiplier
        self.signals_generated = 0

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        # Custom logic: multiply confidence by multiplier
        adjusted_confidence = confidence * self.multiplier
        if adjusted_confidence > 1.0:
            self.signals_generated += 1
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id="test_model",
                prediction=prediction,
                confidence=min(adjusted_confidence, 1.0),
                features=(
                    features.astype(np.float32)
                    if context.get("log_predictions", False) and features is not None
                    else None
                ),
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class TestSignalStrategies:
    """
    Test signal generation strategies.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.clock = TestClock()
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

    def create_test_bar(self, close_price: float = 1.1000) -> Bar:
        """
        Create a test bar.
        """
        return Bar(
            bar_type=self.bar_type,
            open=Price.from_str(str(close_price - 0.0002)),
            high=Price.from_str(str(close_price + 0.0003)),
            low=Price.from_str(str(close_price - 0.0004)),
            close=Price.from_str(str(close_price)),
            volume=Quantity.from_str("1000"),
            ts_event=self.clock.timestamp_ns(),
            ts_init=self.clock.timestamp_ns(),
        )

    def test_threshold_strategy(self) -> None:
        """
        Test threshold signal strategy.
        """
        strategy = ThresholdSignalStrategy(threshold=0.7)
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])
        context = {
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        # Test below threshold
        signal = strategy.generate_signal(bar, 0.8, 0.6, features, context)
        assert signal is None

        # Test above threshold
        signal = strategy.generate_signal(bar, 0.8, 0.8, features, context)
        assert signal is not None
        assert signal.prediction == 0.8
        assert signal.confidence == 0.8

    def test_extremes_strategy(self) -> None:
        """
        Test extremes signal strategy.
        """
        strategy = ExtremesStrategy(top_pct=0.1, threshold=0.5, window_size=10)
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test with insufficient history
        context = {
            "prediction_history": [0.5, 0.6],
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }
        signal = strategy.generate_signal(bar, 0.9, 0.8, features, context)
        assert signal is None

        # Test with sufficient history - extreme value
        context["prediction_history"] = list(range(50, 60))  # [50, 51, ..., 59]
        signal = strategy.generate_signal(bar, 59.5, 0.8, features, context)
        assert signal is not None

        # Test with middle value
        signal = strategy.generate_signal(bar, 54.5, 0.8, features, context)
        assert signal is None

    def test_momentum_strategy(self) -> None:
        """
        Test momentum signal strategy.
        """
        strategy = MomentumStrategy(lookback=3, threshold=0.5, momentum_threshold=0.05)
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test with insufficient history
        context = {
            "prediction_history": [0.5],
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }
        signal = strategy.generate_signal(bar, 0.8, 0.8, features, context)
        assert signal is None

        # Test with positive momentum
        context["prediction_history"] = [0.4, 0.5, 0.6, 0.7]  # Increasing
        signal = strategy.generate_signal(bar, 0.8, 0.8, features, context)
        assert signal is not None
        assert signal.prediction != 0.8  # Adjusted by momentum

        # Test with flat momentum
        context["prediction_history"] = [0.5, 0.5, 0.5, 0.5]  # Flat
        signal = strategy.generate_signal(bar, 0.8, 0.8, features, context)
        assert signal is None  # No momentum

    def test_ensemble_strategy(self) -> None:
        """
        Test ensemble signal strategy.
        """
        strategies = {
            "threshold": ThresholdSignalStrategy(0.5),
            "extremes": ExtremesStrategy(0.2, 0.5, 5),
            "momentum": MomentumStrategy(3, 0.5, 0.01),
        }
        weights = {"threshold": 0.5, "extremes": 0.3, "momentum": 0.2}
        ensemble = EnsembleStrategy(strategies, weights, threshold=0.5)

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])
        context = {
            "prediction_history": [0.3, 0.4, 0.5, 0.6, 0.7],
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        # Test ensemble generation
        signal = ensemble.generate_signal(bar, 0.8, 0.7, features, context)
        # May or may not generate signal depending on weighted voting
        if signal is not None:
            assert isinstance(signal, MLSignal)

    def test_adaptive_strategy(self) -> None:
        """
        Test adaptive signal strategy.
        """
        strategy = AdaptiveStrategy(
            base_threshold=0.5,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test with high adaptive threshold
        context = {
            "model_id": "test_model",
            "adaptive_threshold": 0.9,
            "market_regime": "volatile",
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        # Low confidence relative to threshold
        signal = strategy.generate_signal(bar, 0.7, 0.5, features, context)
        assert signal is None  # Signal strength < 1.0

        # High confidence relative to threshold
        signal = strategy.generate_signal(bar, 0.7, 0.95, features, context)
        assert signal is not None
        assert isinstance(signal, AdaptiveSignal)
        assert signal.metadata["adaptive_threshold"] == 0.9
        assert signal.metadata["market_regime"] == "volatile"

    def test_custom_strategy_plugin(self) -> None:
        """
        Test custom strategy plugin architecture.
        """
        custom_strategy = CustomSignalStrategy(multiplier=1.5)

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])
        context = {
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        # Test custom logic
        signal = custom_strategy.generate_signal(bar, 0.8, 0.5, features, context)
        assert signal is None  # 0.5 * 1.5 = 0.75 < 1.0

        signal = custom_strategy.generate_signal(bar, 0.8, 0.7, features, context)
        assert signal is not None  # 0.7 * 1.5 = 1.05 > 1.0
        assert signal.confidence == 1.0  # Capped at 1.0
        assert custom_strategy.signals_generated == 1


# Define model class at module level for pickling
class SimpleTestModel:
    """
    Simple model for testing that can be pickled.
    """

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([0.8])


class TestOptimizationLevels:
    """
    Test optimization level configurations.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        # Create temporary model file with a simple picklable model
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)

        model = SimpleTestModel()
        with open(self.temp_model_file.name, "wb") as f:
            pickle.dump(model, f)
        self.temp_model_file.close()

        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

    def teardown_method(self) -> None:
        """
        Clean up test fixtures.
        """
        Path(self.temp_model_file.name).unlink(missing_ok=True)

    def test_standard_optimization_level(self) -> None:
        """
        Test standard optimization level.
        """
        opt_config = OptimizationConfig(level=OptimizationLevel.STANDARD)

        assert opt_config.level == OptimizationLevel.STANDARD
        assert not opt_config.enable_zero_copy
        assert not opt_config.enable_model_warm_up
        assert opt_config.pre_allocate_buffers
        assert not opt_config.use_lock_free_buffers

    def test_optimized_level_configuration(self) -> None:
        """
        Test optimized level configuration.
        """
        opt_config = OptimizationConfig(
            level=OptimizationLevel.OPTIMIZED,
            enable_zero_copy=True,
            enable_model_warm_up=True,
            warm_up_iterations=200,
            use_lock_free_buffers=True,
            reservoir_sample_size=2000,
        )

        assert opt_config.level == OptimizationLevel.OPTIMIZED
        assert opt_config.enable_zero_copy
        assert opt_config.enable_model_warm_up
        assert opt_config.warm_up_iterations == 200
        assert opt_config.use_lock_free_buffers
        assert opt_config.reservoir_sample_size == 2000

    def test_actor_with_standard_optimization(self) -> None:
        """
        Test actor with standard optimization.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-STD",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            optimization_config=OptimizationConfig(level=OptimizationLevel.STANDARD),
            use_dummy_stores=True,  # Use dummy stores for testing
        )

        actor = MLSignalActor(config)
        assert actor._opt_config.level == OptimizationLevel.STANDARD
        assert actor._performance_monitor is not None
        # Standard level uses default reservoir size
        assert actor._performance_monitor.reservoir_size == 100

    def test_actor_with_optimized_level(self) -> None:
        """
        Test actor with optimized level.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-OPT",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                reservoir_sample_size=1500,
            ),
            use_dummy_stores=True,  # Use dummy stores for testing
        )

        actor = MLSignalActor(config)
        assert actor._opt_config.level == OptimizationLevel.OPTIMIZED
        assert actor._performance_monitor is not None
        # Optimized level uses configured reservoir size
        assert actor._performance_monitor.reservoir_size == 1500


class TestConfigurationSystem:
    """
    Test configuration system for ML Signal Actor.
    """

    def test_strategy_config_defaults(self) -> None:
        """
        Test StrategyConfig default values.
        """
        config = StrategyConfig()

        assert config.extremes_top_pct == 0.1
        assert config.momentum_lookback == 5
        assert config.ensemble_weights is None
        assert config.adaptive_volatility_factor == 2.0
        assert config.min_threshold == 0.1
        assert config.max_threshold == 0.95
        assert config.update_frequency == 10

    def test_strategy_config_custom_values(self) -> None:
        """
        Test StrategyConfig with custom values.
        """
        custom_weights = {"threshold": 0.6, "extremes": 0.2, "momentum": 0.2}
        config = StrategyConfig(
            extremes_top_pct=0.05,
            momentum_lookback=10,
            ensemble_weights=custom_weights,
            adaptive_volatility_factor=1.5,
            min_threshold=0.2,
            max_threshold=0.9,
            update_frequency=20,
        )

        assert config.extremes_top_pct == 0.05
        assert config.momentum_lookback == 10
        assert config.ensemble_weights == custom_weights
        assert config.adaptive_volatility_factor == 1.5
        assert config.min_threshold == 0.2
        assert config.max_threshold == 0.9
        assert config.update_frequency == 20

    def test_threshold_strategy_enum(self) -> None:
        """
        Test ThresholdStrategy enum values.
        """
        assert ThresholdStrategy.STATIC.value == "static"
        assert ThresholdStrategy.REGIME_AWARE.value == "regime_aware"
        assert ThresholdStrategy.DYNAMIC.value == "dynamic"

    def test_signal_strategy_enum(self) -> None:
        """
        Test SignalStrategy enum values.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"

    def test_optimization_level_enum(self) -> None:
        """
        Test OptimizationLevel enum values.
        """
        assert OptimizationLevel.STANDARD.value == "standard"
        assert OptimizationLevel.OPTIMIZED.value == "optimized"


class TestPerformanceMonitoring:
    """
    Test performance monitoring functionality.
    """

    def test_performance_monitor_initialization(self) -> None:
        """
        Test PerformanceMonitor initialization.
        """
        from ml.actors.signal import PerformanceMonitor

        monitor = PerformanceMonitor(reservoir_size=500)
        assert monitor.reservoir_size == 500
        assert monitor.prediction_count == 0
        assert monitor.signal_count == 0
        assert monitor.error_count == 0

    def test_performance_monitor_timing_recording(self) -> None:
        """
        Test recording timing measurements.
        """
        from ml.actors.signal import PerformanceMonitor

        monitor = PerformanceMonitor(reservoir_size=100)

        # Record some timings (in nanoseconds)
        monitor.record_timing(500_000, 2_000_000, 2_500_000)
        monitor.record_timing(600_000, 2_100_000, 2_700_000)
        monitor.record_timing(550_000, 1_900_000, 2_450_000)

        stats = monitor.get_current_stats()
        assert stats["prediction_count"] == 3
        assert stats["avg_feature_time_ms"] > 0
        assert stats["avg_inference_time_ms"] > 0
        assert stats["avg_total_time_ms"] > 0
        assert stats["p99_total_time_ms"] > 0

    def test_performance_monitor_latency_percentiles(self) -> None:
        """
        Test latency percentile calculations.
        """
        from ml.actors.signal import PerformanceMonitor

        monitor = PerformanceMonitor(reservoir_size=100)

        # Record timings with varying latencies
        for i in range(10):
            feature_time = 400_000 + i * 50_000  # 0.4ms to 0.85ms
            inference_time = 1_800_000 + i * 100_000  # 1.8ms to 2.7ms
            total_time = feature_time + inference_time
            monitor.record_timing(feature_time, inference_time, total_time)

        percentiles = monitor.get_latency_percentiles()

        assert "feature_computation" in percentiles
        assert "inference" in percentiles
        assert "total" in percentiles

        # Check percentile keys
        for category in percentiles.values():
            assert 50.0 in category
            assert 90.0 in category
            assert 95.0 in category
            assert 99.0 in category

    def test_performance_monitor_reservoir_sampling(self) -> None:
        """
        Test reservoir sampling bounds.
        """
        from ml.actors.signal import PerformanceMonitor

        monitor = PerformanceMonitor(reservoir_size=10)

        # Record more than reservoir size
        for i in range(20):
            monitor.record_timing(500_000, 2_000_000, 2_500_000)

        # Should be bounded to reservoir size
        assert len(monitor.feature_times) == 10
        assert len(monitor.inference_times) == 10
        assert len(monitor.total_times) == 10
        assert monitor.prediction_count == 20  # Count continues
