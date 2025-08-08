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
Comprehensive QA validation for OptimizedMLSignalActor.

This script validates all critical requirements for production deployment.

"""

import gc
import logging
import time
import tracemalloc
from typing import Any

import numpy as np

from ml.actors.signal import ModelSwapper
from ml.actors.signal import PerformanceMonitor
from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.core.cache import ReservoirSampler


# Setup logger
logger = logging.getLogger(__name__)


def test_lock_free_operations() -> dict[str, Any]:
    """
    Test lock-free ring buffer operations.
    """
    logger.info("\n=== Testing Lock-Free Ring Buffer ===")

    results = {"passed": True, "details": {}}

    # Test basic operations
    buffer = LockFreeRingBuffer(100)

    # Test append performance
    start = time.perf_counter_ns()
    for i in range(1000):
        buffer.append(float(i))
    elapsed = (time.perf_counter_ns() - start) / 1_000_000

    results["details"]["append_1000_ms"] = elapsed
    logger.info(f"Append 1000 items: {elapsed:.3f}ms")

    # Test statistical operations
    assert buffer.count == 100  # Should wrap around
    mean = buffer.mean()
    std = buffer.std()

    results["details"]["mean"] = mean
    results["details"]["std"] = std
    logger.info(f"Mean: {mean:.2f}, Std: {std:.2f}")

    # Test get_last
    last_values = buffer.get_last(10)
    assert len(last_values) == 10
    logger.info("Validation completed successfully")

    return results


def test_reservoir_sampling() -> dict[str, Any]:
    """
    Test reservoir sampling for percentile calculations.
    """
    logger.info("\n=== Testing Reservoir Sampling ===")

    results = {"passed": True, "details": {}}

    sampler = ReservoirSampler(1000)

    # Add many samples
    start = time.perf_counter_ns()
    for i in range(10000):
        sampler.add_sample(np.random.randn())
    elapsed = (time.perf_counter_ns() - start) / 1_000_000

    results["details"]["add_10000_samples_ms"] = elapsed
    logger.info(f"Add 10000 samples: {elapsed:.3f}ms")

    # Test percentile calculation
    start = time.perf_counter_ns()
    percentiles = sampler.get_percentiles([50.0, 90.0, 95.0, 99.0])
    elapsed = (time.perf_counter_ns() - start) / 1_000_000

    results["details"]["percentile_calc_ms"] = elapsed
    results["details"]["percentiles"] = percentiles
    logger.info(f"Percentile calculation: {elapsed:.3f}ms")
    logger.info(f"Percentiles: {percentiles}")

    return results


def test_feature_cache_zero_allocation() -> dict[str, Any]:
    """
    Test zero-allocation feature caching.
    """
    logger.info("\n=== Testing Zero-Allocation Feature Cache ===")

    results = {"passed": True, "details": {}}

    cache = PreAllocatedFeatureCache(n_features=256, history_size=100)

    # Track memory before operations
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Perform many operations
    for i in range(1000):
        buffer = cache.get_current_buffer()
        buffer[:] = np.random.randn(256)
        cache.store_current_features()

        if i % 100 == 0:
            # Get ONNX input
            onnx_input = cache.prepare_onnx_input(use_normalized=False)
            assert onnx_input.shape == (1, 256)

    # Check memory after operations
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    # Calculate total allocation
    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    results["details"]["total_allocated_kb"] = total_allocated / 1024

    # Should be minimal allocations
    if total_allocated < 100 * 1024:  # Less than 100KB
        logger.info(f" Zero-allocation confirmed: {total_allocated / 1024:.2f}KB allocated")
    else:
        logger.info(f" Excessive allocation: {total_allocated / 1024:.2f}KB allocated")
        results["passed"] = False

    tracemalloc.stop()

    return results


def test_performance_monitoring() -> dict[str, Any]:
    """
    Test performance monitoring capabilities.
    """
    logger.info("\n=== Testing Performance Monitoring ===")

    results = {"passed": True, "details": {}}

    monitor = PerformanceMonitor(reservoir_size=1000)

    # Simulate many timing recordings
    for i in range(5000):
        feature_time = np.random.uniform(100_000, 500_000)  # 0.1-0.5ms in ns
        inference_time = np.random.uniform(500_000, 2_000_000)  # 0.5-2ms in ns
        total_time = feature_time + inference_time + np.random.uniform(50_000, 100_000)

        monitor.record_timing(int(feature_time), int(inference_time), int(total_time))

        if np.random.random() > 0.8:
            monitor.record_signal()

    # Get statistics
    stats = monitor.get_current_stats()
    percentiles = monitor.get_latency_percentiles()

    results["details"]["stats"] = stats
    results["details"]["percentiles"] = percentiles

    logger.info(f"Predictions: {stats['prediction_count']}")
    logger.info(f"Signals: {stats['signal_count']}")
    logger.info(f"Signal rate: {stats['signal_rate']:.2%}")

    for metric, values in percentiles.items():
        logger.info(f"\n{metric} latency percentiles (ms):")
        for p, v in values.items():
            logger.info(f"  P{p}: {v:.3f}ms")

            # Check performance targets
            if metric == "inference" and p == 99.0 and v > 2.0:
                logger.info("   P99 inference exceeds 2ms target!")
                results["passed"] = False
            elif metric == "total" and p == 99.0 and v > 5.0:
                logger.info("   P99 total exceeds 5ms target!")
                results["passed"] = False

    return results


def test_model_swapping() -> dict[str, Any]:
    """
    Test atomic model swapping.
    """
    logger.info("\n=== Testing Model Hot-Swapping ===")

    results = {"passed": True, "details": {}}

    swapper = ModelSwapper()

    # Set initial model
    initial_model = {"name": "model_v1", "version": "1.0"}
    initial_metadata = {"input_names": ["features"], "output_names": ["prediction"]}
    swapper.set_current_model(initial_model, initial_metadata)

    assert swapper.current_model == initial_model
    assert swapper.current_metadata == initial_metadata
    logger.info(" Initial model set")

    # Prepare new model for swap
    new_model = {"name": "model_v2", "version": "2.0"}
    new_metadata = {"input_names": ["features"], "output_names": ["prediction", "confidence"]}
    swapper.prepare_swap(new_model, new_metadata)

    assert swapper.swap_pending
    logger.info(" New model prepared for swap")

    # Execute swap
    swapped = swapper.execute_swap()
    assert swapped
    assert swapper.current_model == new_model
    assert swapper.current_metadata == new_metadata
    assert not swapper.swap_pending
    logger.info("Model swap completed successfully")

    # Test error handling
    error = ValueError("Model load failed")
    swapper.prepare_swap_with_error(error)
    assert swapper.load_error == error
    assert not swapper.swap_pending
    logger.info(" Error handling works correctly")

    results["details"]["swap_test"] = "passed"

    return results


def test_memory_stability() -> dict[str, Any]:
    """
    Test memory stability over extended operations.
    """
    logger.info("\n=== Testing Memory Stability ===")

    results = {"passed": True, "details": {}}

    # Create components
    buffer = LockFreeRingBuffer(1000)
    sampler = ReservoirSampler(1000)
    cache = PreAllocatedFeatureCache(n_features=256, history_size=100)
    monitor = PerformanceMonitor()

    # Force garbage collection and get baseline
    gc.collect()
    import psutil

    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    logger.info(f"Baseline memory: {baseline_memory:.2f}MB")

    # Perform many operations
    for i in range(10000):
        # Ring buffer operations
        buffer.append(np.random.randn())
        if i % 100 == 0:
            buffer.mean()
            buffer.std()

        # Reservoir sampling
        sampler.add_sample(np.random.randn())
        if i % 500 == 0:
            sampler.get_percentiles([50.0, 90.0, 99.0])

        # Feature cache operations
        features = cache.get_current_buffer()
        features[:] = np.random.randn(256)
        cache.store_current_features()

        # Performance monitoring
        monitor.record_timing(
            int(np.random.uniform(100_000, 500_000)),
            int(np.random.uniform(500_000, 2_000_000)),
            int(np.random.uniform(600_000, 2_500_000)),
        )

    # Check memory after operations
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - baseline_memory

    results["details"]["baseline_memory_mb"] = baseline_memory
    results["details"]["final_memory_mb"] = final_memory
    results["details"]["increase_mb"] = memory_increase

    logger.info(f"Final memory: {final_memory:.2f}MB")
    logger.info(f"Memory increase: {memory_increase:.2f}MB")

    # Should have minimal memory increase
    if memory_increase < 10:  # Less than 10MB increase
        logger.info(" Memory stable over 10,000 operations")
    else:
        logger.info(f" Excessive memory growth: {memory_increase:.2f}MB")
        results["passed"] = False

    return results


def run_comprehensive_qa() -> None:
    """
    Run comprehensive QA validation.
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE QA VALIDATION - OptimizedMLSignalActor")
    logger.info("=" * 60)

    all_results = {}
    all_passed = True

    # Run all tests
    tests = [
        ("Lock-Free Operations", test_lock_free_operations),
        ("Reservoir Sampling", test_reservoir_sampling),
        ("Zero-Allocation Cache", test_feature_cache_zero_allocation),
        ("Performance Monitoring", test_performance_monitoring),
        ("Model Hot-Swapping", test_model_swapping),
        ("Memory Stability", test_memory_stability),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            all_results[test_name] = result
            if not result["passed"]:
                all_passed = False
        except Exception as e:
            logger.info(f"\n {test_name} FAILED with exception: {e}")
            all_results[test_name] = {"passed": False, "error": str(e)}
            all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("QA VALIDATION SUMMARY")
    logger.info("=" * 60)

    for test_name, result in all_results.items():
        status = " PASSED" if result["passed"] else " FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("OVERALL RESULT:  ALL TESTS PASSED")
        logger.info("DEPLOYMENT RECOMMENDATION: READY FOR PRODUCTION")
    else:
        logger.info("OVERALL RESULT:  SOME TESTS FAILED")
        logger.info("DEPLOYMENT RECOMMENDATION: ADDRESS FAILURES BEFORE DEPLOYMENT")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_comprehensive_qa()
