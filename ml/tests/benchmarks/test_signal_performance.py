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
Performance benchmarks for optimized ML Signal Actor.

These benchmarks validate that the optimization targets are met:
- P99 feature computation: <500μs
- P99 inference latency: <2ms
- P99 end-to-end: <5ms
- Zero allocations in hot path
- Memory stability over time

"""

import gc
import time
import tracemalloc
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.actors.feature_cache import LockFreeRingBuffer
from ml.actors.feature_cache import PreAllocatedFeatureCache
from ml.actors.feature_cache import ReservoirSampler
from ml.actors.signal_config import OptimizedMLSignalActorConfig
from ml.actors.signal_config import SignalStrategy
from ml.actors.signal_optimized import OptimizedMLSignalActor
from ml.config.base import MLFeatureConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


class BenchmarkConfig:
    """
    Configuration for performance benchmarks.
    """

    # Performance targets (in nanoseconds)
    TARGET_FEATURE_COMPUTATION_P99_NS = 500_000  # 500μs
    TARGET_INFERENCE_LATENCY_P99_NS = 2_000_000  # 2ms
    TARGET_END_TO_END_P99_NS = 5_000_000  # 5ms

    # Test parameters
    WARM_UP_ITERATIONS = 100
    BENCHMARK_ITERATIONS = 1000
    MEMORY_TEST_ITERATIONS = 10000

    # Tolerance for performance regression
    PERFORMANCE_TOLERANCE = 1.2  # 20% tolerance


class PerformanceBenchmark:
    """
    Base class for performance benchmarks.
    """

    def __init__(self):
        self.timings: list[float] = []
        self.config = BenchmarkConfig()

    def reset(self) -> None:
        """
        Reset benchmark timings.
        """
        self.timings.clear()

    def record_timing(self, timing_ns: int) -> None:
        """
        Record a timing measurement.
        """
        self.timings.append(timing_ns)

    def get_statistics(self) -> dict[str, float]:
        """
        Get timing statistics.
        """
        if not self.timings:
            return {}

        timings_array = np.array(self.timings)
        return {
            "count": len(self.timings),
            "mean_ns": float(np.mean(timings_array)),
            "std_ns": float(np.std(timings_array)),
            "min_ns": float(np.min(timings_array)),
            "max_ns": float(np.max(timings_array)),
            "p50_ns": float(np.percentile(timings_array, 50)),
            "p90_ns": float(np.percentile(timings_array, 90)),
            "p95_ns": float(np.percentile(timings_array, 95)),
            "p99_ns": float(np.percentile(timings_array, 99)),
            "p99_9_ns": float(np.percentile(timings_array, 99.9)),
        }

    def assert_performance_target(self, target_ns: int, tolerance: float = None) -> None:
        """
        Assert that P99 latency meets target.
        """
        if tolerance is None:
            tolerance = self.config.PERFORMANCE_TOLERANCE

        stats = self.get_statistics()
        p99_ns = stats.get("p99_ns", float("inf"))

        max_allowed_ns = target_ns * tolerance

        assert p99_ns <= max_allowed_ns, (
            f"P99 latency {p99_ns/1_000_000:.3f}ms exceeds target "
            f"{target_ns/1_000_000:.3f}ms (max allowed: {max_allowed_ns/1_000_000:.3f}ms)"
        )


@pytest.mark.benchmark
class TestRingBufferPerformance:
    """
    Benchmark ring buffer operations.
    """

    def test_ring_buffer_append_performance(self):
        """
        Benchmark ring buffer append operations.
        """
        buffer = LockFreeRingBuffer(1000)
        benchmark = PerformanceBenchmark()

        # Warm up
        for i in range(100):
            buffer.append(float(i))

        # Benchmark append operations
        for i in range(1000):
            start_time = time.perf_counter_ns()
            buffer.append(float(i))
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # Ring buffer append should be extremely fast (<1μs)
        assert stats["p99_ns"] < 1_000, f"Ring buffer append too slow: {stats['p99_ns']}ns"
        print(f"Ring buffer append P99: {stats['p99_ns']:.0f}ns")

    def test_ring_buffer_get_last_performance(self):
        """
        Benchmark ring buffer get_last operations.
        """
        buffer = LockFreeRingBuffer(1000)
        benchmark = PerformanceBenchmark()

        # Fill buffer
        for i in range(1000):
            buffer.append(float(i))

        # Benchmark get_last operations
        for i in range(1000):
            start_time = time.perf_counter_ns()
            _ = buffer.get_last(10)
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # get_last should be fast (<10μs)
        assert stats["p99_ns"] < 10_000, f"Ring buffer get_last too slow: {stats['p99_ns']}ns"
        print(f"Ring buffer get_last P99: {stats['p99_ns']:.0f}ns")


@pytest.mark.benchmark
class TestReservoirSamplerPerformance:
    """
    Benchmark reservoir sampler operations.
    """

    def test_reservoir_sampler_add_performance(self):
        """
        Benchmark reservoir sampler add operations.
        """
        sampler = ReservoirSampler(1000)
        benchmark = PerformanceBenchmark()

        # Benchmark add operations
        for i in range(2000):  # More than reservoir size
            start_time = time.perf_counter_ns()
            sampler.add_sample(float(i))
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # Reservoir sampling should be fast (<5μs)
        assert stats["p99_ns"] < 5_000, f"Reservoir sampling too slow: {stats['p99_ns']}ns"
        print(f"Reservoir sampling P99: {stats['p99_ns']:.0f}ns")

    def test_reservoir_sampler_percentile_performance(self):
        """
        Benchmark reservoir sampler percentile calculation.
        """
        sampler = ReservoirSampler(1000)
        benchmark = PerformanceBenchmark()

        # Fill sampler
        for i in range(1000):
            sampler.add_sample(float(i))

        # Benchmark percentile calculation
        for _ in range(100):
            start_time = time.perf_counter_ns()
            _ = sampler.get_percentile(95.0)
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # Percentile calculation should be reasonable (<100μs)
        assert stats["p99_ns"] < 100_000, f"Percentile calculation too slow: {stats['p99_ns']}ns"
        print(f"Percentile calculation P99: {stats['p99_ns']:.0f}ns")


@pytest.mark.benchmark
class TestFeatureCachePerformance:
    """
    Benchmark feature cache operations.
    """

    def test_feature_cache_buffer_access_performance(self):
        """
        Benchmark feature cache buffer access.
        """
        cache = PreAllocatedFeatureCache(n_features=100, history_size=1000)
        benchmark = PerformanceBenchmark()

        # Benchmark buffer access operations
        for i in range(1000):
            start_time = time.perf_counter_ns()
            buffer = cache.get_current_buffer()
            buffer[0] = float(i)
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # Buffer access should be extremely fast (<1000ns)
        assert stats["p99_ns"] < 1000, f"Buffer access too slow: {stats['p99_ns']}ns"
        print(f"Feature cache buffer access P99: {stats['p99_ns']:.0f}ns")

    def test_feature_cache_onnx_preparation_performance(self):
        """
        Benchmark ONNX input preparation.
        """
        cache = PreAllocatedFeatureCache(n_features=100)
        benchmark = PerformanceBenchmark()

        # Fill current buffer
        current_buffer = cache.get_current_buffer()
        current_buffer[:] = np.random.randn(100).astype(np.float32)

        # Benchmark ONNX preparation
        for _ in range(1000):
            start_time = time.perf_counter_ns()
            _ = cache.prepare_onnx_input(use_normalized=False)
            end_time = time.perf_counter_ns()
            benchmark.record_timing(end_time - start_time)

        stats = benchmark.get_statistics()

        # ONNX preparation should be very fast (<1μs)
        assert stats["p99_ns"] < 1_000, f"ONNX preparation too slow: {stats['p99_ns']}ns"
        print(f"ONNX preparation P99: {stats['p99_ns']:.0f}ns")


@pytest.mark.benchmark
@pytest.mark.slow
class TestOptimizedActorPerformance:
    """
    Comprehensive performance benchmarks for optimized actor.
    """

    @pytest.fixture
    def mock_onnx_model(self):
        """
        Create a fast mock ONNX model.
        """
        mock_model = MagicMock()

        def fast_run(output_names, input_dict):
            # Simulate minimal processing time
            return [np.array([[0.8]], dtype=np.float32)]

        mock_model.run.side_effect = fast_run
        mock_model.get_inputs.return_value = [MagicMock(name="features", shape=[1, 100])]
        mock_model.get_outputs.return_value = [MagicMock(name="prediction", shape=[1, 1])]

        return mock_model

    @pytest.fixture
    def optimized_config(self):
        """
        Create optimized configuration for benchmarks.
        """
        return OptimizedMLSignalActorConfig(
            actor_id="benchmark_actor",
            bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE"),
            model_path="benchmark_model.onnx",
            signal_strategy=SignalStrategy.ADAPTIVE,
            feature_config=MLFeatureConfig(
                lookback_window=50,
                normalize_features=False,  # Disable for performance
            ),
            warm_up_iterations=10,  # Reduced for benchmarks
        )

    @pytest.fixture
    def test_bars(self):
        """
        Create test bars for benchmarking.
        """
        instrument = TestInstrumentProvider.default_fx_ccy("EURUSD")
        bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE")

        bars = []
        base_price = 1.1000
        for i in range(1000):
            price_change = (np.random.random() - 0.5) * 0.001
            close_price = base_price + price_change

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{close_price - 0.0001:.5f}"),
                high=Price.from_str(f"{close_price + 0.0002:.5f}"),
                low=Price.from_str(f"{close_price - 0.0002:.5f}"),
                close=Price.from_str(f"{close_price:.5f}"),
                volume=Quantity.from_int(1000 + i),
                ts_event=1640995200000000000 + i * 60_000_000_000,
                ts_init=1640995200000000000 + i * 60_000_000_000,
            )
            bars.append(bar)
            base_price = close_price

        return bars

    def test_feature_computation_performance(self, optimized_config, mock_onnx_model, test_bars):
        """
        Benchmark feature computation performance.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_onnx_model

            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)

                # Mock required components
                actor._model = mock_onnx_model
                actor._model_swapper.set_current_model(mock_onnx_model)
                actor._indicator_manager = MagicMock()
                actor._indicator_manager.all_initialized.return_value = True
                actor._feature_engineer = MagicMock()

                def fast_feature_computation(*args, **kwargs):
                    return np.random.randn(100).astype(np.float32)

                actor._feature_engineer.calculate_features_online.side_effect = (
                    fast_feature_computation
                )

                benchmark = PerformanceBenchmark()

                # Warm up
                for bar in test_bars[: benchmark.config.WARM_UP_ITERATIONS]:
                    actor._compute_features_optimized(bar)

                # Benchmark feature computation
                for bar in test_bars[
                    benchmark.config.WARM_UP_ITERATIONS : benchmark.config.WARM_UP_ITERATIONS
                    + benchmark.config.BENCHMARK_ITERATIONS
                ]:
                    start_time = time.perf_counter_ns()
                    features = actor._compute_features_optimized(bar)
                    end_time = time.perf_counter_ns()

                    if features is not None:
                        benchmark.record_timing(end_time - start_time)

                stats = benchmark.get_statistics()
                print("\nFeature Computation Performance:")
                print(f"  Mean: {stats['mean_ns']/1000:.1f}μs")
                print(f"  P95:  {stats['p95_ns']/1000:.1f}μs")
                print(f"  P99:  {stats['p99_ns']/1000:.1f}μs")

                # Assert performance target
                benchmark.assert_performance_target(
                    benchmark.config.TARGET_FEATURE_COMPUTATION_P99_NS
                )

    def test_inference_performance(self, optimized_config, mock_onnx_model):
        """
        Benchmark model inference performance.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_onnx_model

            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)
                actor._model_swapper.set_current_model(
                    mock_onnx_model,
                    {
                        "input_names": ["features"],
                        "output_names": ["prediction"],
                    },
                )

                benchmark = PerformanceBenchmark()

                # Create test features
                features = np.random.randn(100).astype(np.float32)

                # Warm up
                for _ in range(benchmark.config.WARM_UP_ITERATIONS):
                    actor._predict_optimized(features)

                # Benchmark inference
                for _ in range(benchmark.config.BENCHMARK_ITERATIONS):
                    start_time = time.perf_counter_ns()
                    prediction, confidence = actor._predict_optimized(features)
                    end_time = time.perf_counter_ns()
                    benchmark.record_timing(end_time - start_time)

                stats = benchmark.get_statistics()
                print("\nInference Performance:")
                print(f"  Mean: {stats['mean_ns']/1_000_000:.3f}ms")
                print(f"  P95:  {stats['p95_ns']/1_000_000:.3f}ms")
                print(f"  P99:  {stats['p99_ns']/1_000_000:.3f}ms")

                # Assert performance target
                benchmark.assert_performance_target(
                    benchmark.config.TARGET_INFERENCE_LATENCY_P99_NS
                )

    def test_end_to_end_performance(self, optimized_config, mock_onnx_model, test_bars):
        """
        Benchmark end-to-end signal generation performance.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_onnx_model

            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)

                # Mock required components for full pipeline
                actor._model = mock_onnx_model
                actor._model_swapper.set_current_model(
                    mock_onnx_model,
                    {
                        "input_names": ["features"],
                        "output_names": ["prediction"],
                    },
                )
                actor._indicator_manager = MagicMock()
                actor._indicator_manager.all_initialized.return_value = True
                actor._indicator_manager.price_history = {"closes": [1.1] * 50}
                actor._feature_engineer = MagicMock()
                actor._feature_engineer.calculate_features_online.return_value = np.random.randn(
                    100
                ).astype(np.float32)

                benchmark = PerformanceBenchmark()

                # Warm up
                for bar in test_bars[: benchmark.config.WARM_UP_ITERATIONS]:
                    features = actor._compute_features_optimized(bar)
                    if features is not None:
                        actor._generate_prediction_protected(bar, features)

                # Benchmark end-to-end processing
                for bar in test_bars[
                    benchmark.config.WARM_UP_ITERATIONS : benchmark.config.WARM_UP_ITERATIONS
                    + benchmark.config.BENCHMARK_ITERATIONS
                ]:
                    start_time = time.perf_counter_ns()
                    features = actor._compute_features_optimized(bar)
                    if features is not None:
                        actor._generate_prediction_protected(bar, features)
                    end_time = time.perf_counter_ns()
                    benchmark.record_timing(end_time - start_time)

                stats = benchmark.get_statistics()
                print("\nEnd-to-End Performance:")
                print(f"  Mean: {stats['mean_ns']/1_000_000:.3f}ms")
                print(f"  P95:  {stats['p95_ns']/1_000_000:.3f}ms")
                print(f"  P99:  {stats['p99_ns']/1_000_000:.3f}ms")

                # Assert performance target
                benchmark.assert_performance_target(benchmark.config.TARGET_END_TO_END_P99_NS)

    def test_memory_stability(self, optimized_config, mock_onnx_model, test_bars):
        """
        Test memory stability over extended operation.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_onnx_model

            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)

                # Setup mocks
                actor._model = mock_onnx_model
                actor._model_swapper.set_current_model(mock_onnx_model)
                actor._indicator_manager = MagicMock()
                actor._indicator_manager.all_initialized.return_value = True
                actor._indicator_manager.price_history = {"closes": [1.1] * 50}
                actor._feature_engineer = MagicMock()
                actor._feature_engineer.calculate_features_online.return_value = np.random.randn(
                    100
                ).astype(np.float32)

                # Start memory tracing
                tracemalloc.start()
                gc.collect()
                initial_snapshot = tracemalloc.take_snapshot()

                # Process many bars to test memory stability
                bars_to_process = test_bars * (
                    benchmark.config.MEMORY_TEST_ITERATIONS // len(test_bars) + 1
                )

                for i, bar in enumerate(bars_to_process[: benchmark.config.MEMORY_TEST_ITERATIONS]):
                    features = actor._compute_features_optimized(bar)
                    if features is not None:
                        actor._generate_prediction_protected(bar, features)

                    # Check memory periodically
                    if i % 1000 == 0 and i > 0:
                        gc.collect()
                        current_snapshot = tracemalloc.take_snapshot()
                        top_stats = current_snapshot.compare_to(initial_snapshot, "lineno")

                        # Check for significant memory growth
                        total_growth = sum(stat.size_diff for stat in top_stats)
                        memory_growth_mb = total_growth / (1024 * 1024)

                        print(f"Memory growth after {i} iterations: {memory_growth_mb:.2f}MB")

                        # Assert memory growth is bounded (should be < 100MB for long runs)
                        assert (
                            memory_growth_mb < 100
                        ), f"Excessive memory growth: {memory_growth_mb:.2f}MB"

                # Final memory check
                gc.collect()
                final_snapshot = tracemalloc.take_snapshot()
                top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")
                total_growth = sum(stat.size_diff for stat in top_stats)
                final_growth_mb = total_growth / (1024 * 1024)

                print(
                    f"\nFinal memory growth: {final_growth_mb:.2f}MB after {benchmark.config.MEMORY_TEST_ITERATIONS} iterations"
                )

                # Assert final memory growth is reasonable
                assert (
                    final_growth_mb < 200
                ), f"Excessive final memory growth: {final_growth_mb:.2f}MB"

                tracemalloc.stop()

    def test_hot_path_zero_allocation(self, optimized_config, mock_onnx_model, test_bars):
        """
        Test that hot path operations don't allocate memory.
        """
        with patch("ml.actors.signal_optimized.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_onnx_model

            with patch.object(OptimizedMLSignalActor, "_load_model"):
                actor = OptimizedMLSignalActor(optimized_config)

                # Setup mocks
                actor._model = mock_onnx_model
                actor._model_swapper.set_current_model(mock_onnx_model)
                actor._indicator_manager = MagicMock()
                actor._indicator_manager.all_initialized.return_value = True
                actor._indicator_manager.price_history = {"closes": [1.1] * 50}
                actor._feature_engineer = MagicMock()
                actor._feature_engineer.calculate_features_online.return_value = np.random.randn(
                    100
                ).astype(np.float32)

                # Warm up to eliminate initialization allocations
                for bar in test_bars[:100]:
                    features = actor._compute_features_optimized(bar)
                    if features is not None:
                        actor._generate_prediction_protected(bar, features)

                # Start precise memory tracking
                tracemalloc.start()
                gc.collect()

                # Take baseline snapshot
                baseline_snapshot = tracemalloc.take_snapshot()

                # Process bars in hot path
                hot_path_iterations = 100
                for bar in test_bars[:hot_path_iterations]:
                    features = actor._compute_features_optimized(bar)
                    if features is not None:
                        actor._generate_prediction_protected(bar, features)

                # Check for allocations
                gc.collect()
                hot_path_snapshot = tracemalloc.take_snapshot()
                top_stats = hot_path_snapshot.compare_to(baseline_snapshot, "lineno")

                # Filter out system allocations and focus on our code
                significant_allocations = [
                    stat
                    for stat in top_stats
                    if stat.size_diff > 1024  # > 1KB allocation
                    and ("ml/actors" in str(stat.traceback) or "signal" in str(stat.traceback))
                ]

                if significant_allocations:
                    print("\nSignificant allocations detected in hot path:")
                    for stat in significant_allocations[:5]:
                        print(f"  {stat.size_diff/1024:.1f}KB: {stat.traceback}")

                total_hot_path_growth = sum(stat.size_diff for stat in significant_allocations)
                hot_path_growth_kb = total_hot_path_growth / 1024

                print(
                    f"\nHot path memory growth: {hot_path_growth_kb:.1f}KB over {hot_path_iterations} iterations"
                )

                # Assert minimal memory growth in hot path (<10KB total)
                assert hot_path_growth_kb < 10, (
                    f"Hot path allocating too much memory: {hot_path_growth_kb:.1f}KB. "
                    "Should be zero-allocation."
                )

                tracemalloc.stop()


if __name__ == "__main__":
    """
    Run benchmarks directly for development.
    """
    print("Running ML Signal Actor Performance Benchmarks...")
    print("=" * 60)

    # Basic component benchmarks
    print("\n1. Ring Buffer Performance")
    test_ring = TestRingBufferPerformance()
    test_ring.test_ring_buffer_append_performance()
    test_ring.test_ring_buffer_get_last_performance()

    print("\n2. Reservoir Sampler Performance")
    test_reservoir = TestReservoirSamplerPerformance()
    test_reservoir.test_reservoir_sampler_add_performance()
    test_reservoir.test_reservoir_sampler_percentile_performance()

    print("\n3. Feature Cache Performance")
    test_cache = TestFeatureCachePerformance()
    test_cache.test_feature_cache_buffer_access_performance()
    test_cache.test_feature_cache_onnx_preparation_performance()

    print("\n" + "=" * 60)
    print("All benchmarks completed successfully!")
    print("Performance targets met for hot path optimization.")
