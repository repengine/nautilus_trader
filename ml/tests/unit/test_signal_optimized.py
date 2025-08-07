# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Unit tests for optimized ML Signal Actor.

Tests the hot path optimization features including zero-copy operations, pre-allocated
buffers, and performance characteristics.

"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

# Skip tests if ONNX is not available - import check first
from ml._imports import HAS_ONNX


pytestmark = pytest.mark.skipif(not HAS_ONNX, reason="ONNX not available")

from ml.actors.feature_cache import LockFreeRingBuffer
from ml.actors.feature_cache import PreAllocatedFeatureCache
from ml.actors.feature_cache import ReservoirSampler
from ml.actors.signal_config import ONNXOptimizationConfig
from ml.actors.signal_config import OptimizedMLSignalActorConfig
from ml.actors.signal_config import SignalStrategy
from ml.actors.signal_config import ThresholdStrategy
from ml.actors.signal_optimized import ModelSwapper
from ml.actors.signal_optimized import OptimizedMLSignal
from ml.actors.signal_optimized import OptimizedMLSignalActor
from ml.actors.signal_optimized import PerformanceMonitor
from ml.config.base import MLFeatureConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


class TestLockFreeRingBuffer:
    """
    Test lock-free ring buffer implementation.
    """

    def test_ring_buffer_initialization(self):
        """
        Test ring buffer initializes correctly.
        """
        buffer = LockFreeRingBuffer(10)

        assert buffer.size == 10
        assert buffer.count == 0
        assert not buffer.is_full
        assert len(buffer.get_all()) == 0

    def test_ring_buffer_append_single_values(self):
        """
        Test appending single values to ring buffer.
        """
        buffer = LockFreeRingBuffer(3)

        # Add first value
        buffer.append(1.0)
        assert buffer.count == 1
        np.testing.assert_array_equal(buffer.get_all(), [1.0])

        # Add second value
        buffer.append(2.0)
        assert buffer.count == 2
        np.testing.assert_array_equal(buffer.get_all(), [1.0, 2.0])

        # Fill buffer
        buffer.append(3.0)
        assert buffer.count == 3
        assert buffer.is_full
        np.testing.assert_array_equal(buffer.get_all(), [1.0, 2.0, 3.0])

    def test_ring_buffer_wrap_around(self):
        """
        Test ring buffer wrap-around behavior.
        """
        buffer = LockFreeRingBuffer(3)

        # Fill buffer
        buffer.append(1.0)
        buffer.append(2.0)
        buffer.append(3.0)

        # Add one more to trigger wrap-around
        buffer.append(4.0)
        assert buffer.count == 3  # Size should stay at capacity
        np.testing.assert_array_equal(buffer.get_all(), [2.0, 3.0, 4.0])

        # Add another
        buffer.append(5.0)
        np.testing.assert_array_equal(buffer.get_all(), [3.0, 4.0, 5.0])

    def test_ring_buffer_get_last(self):
        """
        Test getting last n values.
        """
        buffer = LockFreeRingBuffer(5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

        for value in values:
            buffer.append(value)

        # Buffer should contain [3.0, 4.0, 5.0, 6.0, 7.0]
        np.testing.assert_array_equal(buffer.get_last(2), [6.0, 7.0])
        np.testing.assert_array_equal(buffer.get_last(3), [5.0, 6.0, 7.0])
        np.testing.assert_array_equal(buffer.get_all(), [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_ring_buffer_statistics(self):
        """
        Test ring buffer statistical methods.
        """
        buffer = LockFreeRingBuffer(5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for value in values:
            buffer.append(value)

        assert buffer.mean() == 3.0
        assert abs(buffer.std() - np.std(values)) < 1e-6  # Relaxed precision for floating point
        assert buffer.percentile(50) == 3.0

    def test_ring_buffer_reset(self):
        """
        Test ring buffer reset functionality.
        """
        buffer = LockFreeRingBuffer(3)
        buffer.append(1.0)
        buffer.append(2.0)

        buffer.reset()
        assert buffer.count == 0
        assert not buffer.is_full
        assert len(buffer.get_all()) == 0


class TestReservoirSampler:
    """
    Test reservoir sampling implementation.
    """

    def test_reservoir_sampler_initialization(self):
        """
        Test reservoir sampler initializes correctly.
        """
        sampler = ReservoirSampler(100)

        assert sampler.reservoir_size == 100
        assert sampler.count == 0
        assert sampler.total_seen == 0

    def test_reservoir_sampler_fill_reservoir(self):
        """
        Test filling reservoir up to capacity.
        """
        sampler = ReservoirSampler(5)

        for i in range(5):
            sampler.add_sample(float(i))

        assert sampler.count == 5
        assert sampler.total_seen == 5

        sample = sampler.get_sample()
        assert len(sample) == 5
        np.testing.assert_array_equal(sample, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_reservoir_sampler_overflow(self):
        """
        Test reservoir sampling with more values than capacity.
        """
        sampler = ReservoirSampler(3)

        # Add more values than capacity
        for i in range(10):
            sampler.add_sample(float(i))

        assert sampler.count == 3  # Reservoir size
        assert sampler.total_seen == 10  # Total values seen

        sample = sampler.get_sample()
        assert len(sample) == 3
        # Sample should contain 3 values, but we can't predict which ones due to randomness

    def test_reservoir_sampler_percentiles(self):
        """
        Test percentile calculation from reservoir.
        """
        sampler = ReservoirSampler(100)

        # Add known distribution
        values = list(range(100))
        for value in values:
            sampler.add_sample(float(value))

        # Test percentiles
        p50 = sampler.get_percentile(50.0)
        p90 = sampler.get_percentile(90.0)

        # Should be close to expected values
        assert 45 <= p50 <= 55  # 50th percentile around 49.5
        assert 85 <= p90 <= 95  # 90th percentile around 89.1

    def test_reservoir_sampler_multiple_percentiles(self):
        """
        Test calculating multiple percentiles efficiently.
        """
        sampler = ReservoirSampler(50)

        for i in range(100):
            sampler.add_sample(float(i))

        percentiles = sampler.get_percentiles([25.0, 50.0, 75.0, 90.0])

        assert len(percentiles) == 4
        assert all(isinstance(v, float) for v in percentiles.values())
        # Values should be in ascending order
        values_list = list(percentiles.values())
        assert values_list == sorted(values_list)


class TestPreAllocatedFeatureCache:
    """
    Test pre-allocated feature cache.
    """

    def test_feature_cache_initialization(self):
        """
        Test feature cache initializes correctly.
        """
        cache = PreAllocatedFeatureCache(n_features=10, history_size=5)

        assert cache.n_features == 10
        assert cache.history_size == 5
        assert cache.history_count == 0

        # Test buffer shapes
        assert cache.get_current_buffer().shape == (10,)
        assert cache.get_normalized_buffer().shape == (10,)
        assert cache.get_onnx_input_buffer().shape == (1, 10)

    def test_feature_cache_buffer_operations(self):
        """
        Test buffer operations and memory views.
        """
        cache = PreAllocatedFeatureCache(n_features=5)

        # Test current buffer modification
        current_buffer = cache.get_current_buffer()
        current_buffer[0] = 1.0
        current_buffer[1] = 2.0

        # Should be reflected in memory view
        view = cache.get_current_view()
        assert view[0] == 1.0
        assert view[1] == 2.0

    def test_feature_cache_onnx_preparation(self):
        """
        Test ONNX input preparation.
        """
        cache = PreAllocatedFeatureCache(n_features=3)

        # Set current features
        current_buffer = cache.get_current_buffer()
        current_buffer[:] = [1.0, 2.0, 3.0]

        # Prepare ONNX input
        onnx_input = cache.prepare_onnx_input(use_normalized=False)

        assert onnx_input.shape == (1, 3)
        np.testing.assert_array_equal(onnx_input[0], [1.0, 2.0, 3.0])

    def test_feature_cache_history_storage(self):
        """
        Test feature history storage and retrieval.
        """
        cache = PreAllocatedFeatureCache(n_features=2, history_size=3)

        # Store multiple feature vectors
        current_buffer = cache.get_current_buffer()

        # First vector
        current_buffer[:] = [1.0, 2.0]
        cache.store_current_features()

        # Second vector
        current_buffer[:] = [3.0, 4.0]
        cache.store_current_features()

        # Third vector
        current_buffer[:] = [5.0, 6.0]
        cache.store_current_features()

        assert cache.history_count == 3

        # Get history
        history = cache.get_feature_history()
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(history, expected)

    def test_feature_cache_history_wraparound(self):
        """
        Test history storage with wrap-around.
        """
        cache = PreAllocatedFeatureCache(n_features=2, history_size=2)
        current_buffer = cache.get_current_buffer()

        # Store 3 vectors (more than capacity)
        current_buffer[:] = [1.0, 2.0]
        cache.store_current_features()

        current_buffer[:] = [3.0, 4.0]
        cache.store_current_features()

        current_buffer[:] = [5.0, 6.0]
        cache.store_current_features()

        # Should only have last 2 vectors
        assert cache.history_count == 2
        history = cache.get_feature_history()
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(history, expected)


class TestPerformanceMonitor:
    """
    Test performance monitoring implementation.
    """

    def test_performance_monitor_initialization(self):
        """
        Test performance monitor initializes correctly.
        """
        monitor = PerformanceMonitor(reservoir_size=100)

        stats = monitor.get_current_stats()
        assert stats["prediction_count"] == 0
        assert stats["signal_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0

    def test_performance_monitor_timing_recording(self):
        """
        Test timing measurement recording.
        """
        monitor = PerformanceMonitor(reservoir_size=10)

        # Record some timings
        monitor.record_timing(
            feature_time_ns=500_000,  # 0.5ms
            inference_time_ns=1_500_000,  # 1.5ms
            total_time_ns=2_000_000,  # 2ms
        )

        stats = monitor.get_current_stats()
        assert stats["prediction_count"] == 1
        assert stats["last_feature_time_ms"] == 0.5
        assert stats["last_inference_time_ms"] == 1.5
        assert stats["last_total_time_ms"] == 2.0

    def test_performance_monitor_percentiles(self):
        """
        Test latency percentile calculation.
        """
        monitor = PerformanceMonitor(reservoir_size=100)

        # Record multiple measurements
        for i in range(50):
            monitor.record_timing(
                feature_time_ns=int(i * 10_000),  # Increasing times
                inference_time_ns=int(i * 20_000),
                total_time_ns=int(i * 30_000),
            )

        percentiles = monitor.get_latency_percentiles()

        assert "feature_computation" in percentiles
        assert "inference" in percentiles
        assert "total" in percentiles

        # Check that percentiles are in ascending order
        feature_percentiles = percentiles["feature_computation"]
        values = [feature_percentiles[p] for p in [50.0, 90.0, 95.0, 99.0]]
        assert values == sorted(values)

    def test_performance_monitor_error_tracking(self):
        """
        Test error and signal tracking.
        """
        monitor = PerformanceMonitor()

        # Record some events
        monitor.record_timing(100_000, 200_000, 300_000)
        monitor.record_signal()
        monitor.record_timing(150_000, 250_000, 400_000)
        monitor.record_error()

        stats = monitor.get_current_stats()
        assert stats["prediction_count"] == 2
        assert stats["signal_count"] == 1
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.5
        assert stats["signal_rate"] == 0.5


class TestModelSwapper:
    """
    Test atomic model swapping implementation.
    """

    def test_model_swapper_initialization(self):
        """
        Test model swapper initializes correctly.
        """
        swapper = ModelSwapper()

        assert swapper.current_model is None
        assert swapper.current_metadata is None
        assert not swapper.swap_pending
        assert swapper.load_error is None

    def test_model_swapper_set_initial_model(self):
        """
        Test setting initial model.
        """
        swapper = ModelSwapper()
        mock_model = MagicMock()
        metadata = {"input_names": ["features"], "output_names": ["prediction"]}

        swapper.set_current_model(mock_model, metadata)

        assert swapper.current_model is mock_model
        assert swapper.current_metadata == metadata
        assert not swapper.swap_pending

    def test_model_swapper_prepare_swap(self):
        """
        Test preparing model swap.
        """
        swapper = ModelSwapper()
        old_model = MagicMock()
        new_model = MagicMock()

        swapper.set_current_model(old_model)
        swapper.prepare_swap(new_model, {"version": "2.0"})

        assert swapper.current_model is old_model  # Still old model
        assert swapper.swap_pending
        assert swapper.load_error is None

    def test_model_swapper_execute_swap(self):
        """
        Test executing atomic model swap.
        """
        swapper = ModelSwapper()
        old_model = MagicMock()
        new_model = MagicMock()

        swapper.set_current_model(old_model, {"version": "1.0"})
        swapper.prepare_swap(new_model, {"version": "2.0"})

        # Execute swap
        result = swapper.execute_swap()

        assert result is True
        assert swapper.current_model is new_model
        assert swapper.current_metadata == {"version": "2.0"}
        assert not swapper.swap_pending

    def test_model_swapper_no_pending_swap(self):
        """
        Test executing swap when no swap is pending.
        """
        swapper = ModelSwapper()
        model = MagicMock()

        swapper.set_current_model(model)

        # Try to execute swap without preparing
        result = swapper.execute_swap()

        assert result is False
        assert swapper.current_model is model  # Unchanged

    def test_model_swapper_error_handling(self):
        """
        Test error handling during model loading.
        """
        swapper = ModelSwapper()
        error = RuntimeError("Model loading failed")

        swapper.prepare_swap_with_error(error)

        assert swapper.load_error is error
        assert not swapper.swap_pending


class TestOptimizedMLSignal:
    """
    Test optimized ML signal data structure.
    """

    def test_optimized_signal_initialization(self):
        """
        Test optimized signal initializes correctly.
        """
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))

        signal = OptimizedMLSignal(
            instrument_id=instrument_id,
            prediction=0.8,
            confidence=0.9,
            signal_strength=1.2,
            market_regime="trending",
            adaptive_threshold=0.75,
            feature_computation_time_ns=400_000,
            inference_time_ns=1_800_000,
            total_latency_ns=2_500_000,
            ts_event=1640995200000000000,
            ts_init=1640995200000000001,
        )

        assert signal.instrument_id == instrument_id
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.signal_strength == 1.2
        assert signal.market_regime == "trending"
        assert signal.adaptive_threshold == 0.75

        # Test time properties
        assert signal.feature_computation_time_ms == 0.4
        assert signal.inference_time_ms == 1.8
        assert signal.total_latency_ms == 2.5

        assert signal.ts_event == 1640995200000000000
        assert signal.ts_init == 1640995200000000001


class TestOptimizedMLSignalActorConfig:
    """
    Test optimized ML signal actor configuration.
    """

    def test_optimized_config_defaults(self):
        """
        Test optimized configuration with default values.
        """
        config = OptimizedMLSignalActorConfig(
            actor_id="test_actor",
            bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE"),
            model_path="test_model.onnx",
        )

        assert config.signal_strategy == SignalStrategy.ADAPTIVE
        assert config.threshold_strategy == ThresholdStrategy.REGIME_AWARE
        assert config.enable_hot_reload is True
        assert config.enable_model_warm_up is True
        assert config.warm_up_iterations == 100

    def test_optimized_config_helper_methods(self):
        """
        Test configuration helper methods.
        """
        config = OptimizedMLSignalActorConfig(
            actor_id="test_actor",
            bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE"),
            model_path="test_model.onnx",
        )

        # Test helper methods return default configs
        onnx_config = config.get_onnx_config()
        assert isinstance(onnx_config, ONNXOptimizationConfig)

        adaptive_config = config.get_adaptive_config()
        assert adaptive_config.base_threshold == 0.7

        hotpath_config = config.get_hotpath_config()
        assert hotpath_config.enable_zero_copy is True

        ensemble_weights = config.get_ensemble_weights()
        assert "threshold" in ensemble_weights
        assert "extremes" in ensemble_weights
        assert "momentum" in ensemble_weights


@pytest.mark.slow
class TestOptimizedMLSignalActorIntegration:
    """
    Integration tests for optimized ML signal actor.
    """

    @pytest.fixture
    def mock_model(self):
        """
        Create a mock ONNX model.
        """
        mock_model = MagicMock()
        mock_model.run.return_value = [np.array([[0.8]]), np.array([[0.9]])]

        # Mock model inputs/outputs
        mock_input = MagicMock()
        mock_input.name = "features"
        mock_input.shape = [1, 10]

        mock_output = MagicMock()
        mock_output.name = "prediction"
        mock_output.shape = [1, 1]

        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]

        return mock_model

    @pytest.fixture
    def optimized_config(self):
        """
        Create optimized signal actor configuration.
        """
        return OptimizedMLSignalActorConfig(
            actor_id="test_optimized_actor",
            bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE"),
            model_path="test_model.onnx",
            signal_strategy=SignalStrategy.ADAPTIVE,
            threshold_strategy=ThresholdStrategy.REGIME_AWARE,
            feature_config=MLFeatureConfig(
                lookback_window=50,
                normalize_features=False,  # Disable for performance
            ),
            warm_up_iterations=10,  # Reduced for tests
        )

    @pytest.fixture
    def test_bar(self):
        """
        Create a test bar.
        """
        instrument = TestInstrumentProvider.default_fx_ccy("EURUSD")
        bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE")

        return Bar(
            bar_type=bar_type,
            open=Price.from_str("1.1000"),
            high=Price.from_str("1.1010"),
            low=Price.from_str("1.0990"),
            close=Price.from_str("1.1005"),
            volume=Quantity.from_int(1000),
            ts_event=1640995200000000000,
            ts_init=1640995200000000000,
        )

    def test_optimized_actor_initialization(self, optimized_config, mock_model):
        """
        Test optimized actor initializes correctly.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_model

            # Mock base class initialization
            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)

                assert actor._optimized_config == optimized_config
                assert isinstance(actor._performance_monitor, PerformanceMonitor)
                assert isinstance(actor._feature_cache, PreAllocatedFeatureCache)
                assert isinstance(actor._prediction_buffer, LockFreeRingBuffer)
                assert isinstance(actor._model_swapper, ModelSwapper)

    def test_performance_monitoring_integration(self, optimized_config):
        """
        Test performance monitoring during signal generation.
        """
        with patch.object(OptimizedMLSignalActor, "_load_model"):
            actor = OptimizedMLSignalActor(optimized_config)

            # Mock required components
            actor._model = MagicMock()
            actor._model_swapper.set_current_model(actor._model)
            actor._indicator_manager = MagicMock()
            actor._indicator_manager.all_initialized.return_value = True
            actor._feature_engineer = MagicMock()
            actor._feature_engineer.calculate_features_online.return_value = np.array([1.0] * 10)

            # Process some predictions
            for _ in range(5):
                features = np.random.randn(10).astype(np.float32)
                with patch.object(actor, "_predict_optimized", return_value=(0.8, 0.9)):
                    actor._performance_monitor.record_timing(
                        feature_time_ns=400_000,
                        inference_time_ns=1_500_000,
                        total_time_ns=2_000_000,
                    )

            # Check performance stats
            stats = actor.get_performance_stats()
            assert stats["prediction_count"] == 5
            assert "latency_percentiles" in stats
            assert "feature_cache_history_count" in stats

    @pytest.mark.parametrize(
        "strategy",
        [
            SignalStrategy.ADAPTIVE,
            SignalStrategy.THRESHOLD,
            SignalStrategy.EXTREMES,
            SignalStrategy.MOMENTUM,
        ],
    )
    def test_signal_strategies_optimization(self, optimized_config, strategy):
        """
        Test different signal strategies with optimizations.
        """
        optimized_config = OptimizedMLSignalActorConfig(
            actor_id="test_actor",
            bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE"),
            model_path="test_model.onnx",
            signal_strategy=strategy,
        )

        with patch.object(OptimizedMLSignalActor, "_load_model"):
            actor = OptimizedMLSignalActor(optimized_config)

            # Each strategy should initialize specific components
            assert actor._optimized_config.signal_strategy == strategy

            # Check that buffers are initialized for all strategies
            assert isinstance(actor._prediction_buffer, LockFreeRingBuffer)
            assert isinstance(actor._confidence_buffer, LockFreeRingBuffer)

    def test_memory_stability_simulation(self, optimized_config):
        """
        Test memory stability over many operations.
        """
        with patch.object(OptimizedMLSignalActor, "_load_model"):
            actor = OptimizedMLSignalActor(optimized_config)

            # Mock components
            actor._model = MagicMock()
            actor._model.run.return_value = [np.array([[0.8]])]
            actor._model_swapper.set_current_model(actor._model)

            # Simulate many operations
            initial_buffer_count = actor._prediction_buffer.count

            for i in range(1000):
                # Simulate feature computation and prediction
                features = np.random.randn(10).astype(np.float32)
                actor._prediction_buffer.append(float(i % 100))
                actor._confidence_buffer.append(0.8)

                # Periodically check buffer bounds
                if i % 100 == 0:
                    assert actor._prediction_buffer.count <= actor._prediction_buffer.size
                    assert actor._confidence_buffer.count <= actor._confidence_buffer.size

            # Verify no memory growth (buffers should be bounded)
            assert actor._prediction_buffer.count <= actor._prediction_buffer.size
            assert actor._confidence_buffer.count <= actor._confidence_buffer.size

            # Performance monitor should handle many samples
            final_stats = actor.get_performance_stats()
            assert isinstance(final_stats, dict)
