#!/usr/bin/env python3
"""
Core ML Performance Test - Focused on ML Components Only

This test focuses purely on the ML components without full Nautilus integration
to isolate and measure the actual ML performance claims.
"""

import gc
import json
import logging
import statistics
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil

from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache

# Core ML imports - avoid full Nautilus imports that cause FFI issues
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import OptimizationLevel
from ml.config.actors import OptimizationConfig
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer
from ml.features.indicators import IndicatorManager
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """
    Performance measurement result.
    """

    name: str
    min_latency_us: float
    mean_latency_us: float
    p99_latency_us: float
    throughput_ops_per_sec: float
    memory_allocated_bytes: int
    passes_500us_claim: bool
    passes_5ms_claim: bool
    passes_zero_alloc_claim: bool


@contextmanager
def memory_tracker():
    """
    Track memory allocations.
    """
    tracemalloc.start()
    gc.collect()
    snapshot_before = tracemalloc.take_snapshot()

    try:
        yield
    finally:
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        tracemalloc.stop()
        memory_tracker.allocated_bytes = total_allocated


class CoreMLPerformanceTester:
    """
    Test core ML components performance.
    """

    def __init__(self):
        self.results = {}

    def test_feature_computation_only(self, n_iterations: int = 10000) -> PerformanceResult:
        """
        Test pure feature computation without Nautilus Bar objects.
        """
        logger.info(f"Testing core feature computation ({n_iterations} iterations)...")

        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize with simple price data (avoid complex Nautilus objects)
        for i in range(50):
            price = 1.1000 + i * 0.0001
            volume = 1000000 + i * 1000
            indicator_mgr.price_history["closes"].append(price)
            indicator_mgr.price_history["volumes"].append(volume)
            indicator_mgr.price_history["highs"].append(price + 0.0001)
            indicator_mgr.price_history["lows"].append(price - 0.0001)

        # Warm up
        for _ in range(100):
            current_bar = {
                "open": 1.1000,
                "high": 1.1001,
                "low": 1.0999,
                "close": 1.1000,
                "volume": 1000000,
            }
            engineer.calculate_features_online(current_bar, indicator_mgr)

        # Performance measurement
        latencies = []

        with memory_tracker():
            for i in range(n_iterations):
                # Simulate realistic price movement
                base_price = 1.1000 + (i * 0.00001)
                current_bar = {
                    "open": base_price + np.random.normal(0, 0.00001),
                    "high": base_price + abs(np.random.normal(0, 0.00002)),
                    "low": base_price - abs(np.random.normal(0, 0.00002)),
                    "close": base_price + np.random.normal(0, 0.00001),
                    "volume": 1000000 + np.random.exponential(100000),
                }

                start = time.perf_counter()
                features = engineer.calculate_features_online(current_bar, indicator_mgr)
                end = time.perf_counter()

                latency_us = (end - start) * 1_000_000
                latencies.append(latency_us)

                # Update history (realistic usage pattern)
                indicator_mgr.price_history["closes"].append(current_bar["close"])
                indicator_mgr.price_history["volumes"].append(current_bar["volume"])
                indicator_mgr.price_history["highs"].append(current_bar["high"])
                indicator_mgr.price_history["lows"].append(current_bar["low"])

                # Keep history bounded
                for key in indicator_mgr.price_history:
                    if len(indicator_mgr.price_history[key]) > 100:
                        indicator_mgr.price_history[key] = indicator_mgr.price_history[key][-100:]

        allocated_bytes = getattr(memory_tracker, "allocated_bytes", 0)
        total_time = sum(latencies) / 1_000_000

        return PerformanceResult(
            name="Feature Computation Core",
            min_latency_us=min(latencies),
            mean_latency_us=statistics.mean(latencies),
            p99_latency_us=np.percentile(latencies, 99),
            throughput_ops_per_sec=n_iterations / total_time,
            memory_allocated_bytes=allocated_bytes,
            passes_500us_claim=np.percentile(latencies, 99) < 500,
            passes_5ms_claim=np.percentile(latencies, 99) < 5000,
            passes_zero_alloc_claim=allocated_bytes < 1000,  # Allow 1KB overhead
        )

    def test_cache_performance(self, n_operations: int = 100000) -> PerformanceResult:
        """Test cache performance - key to zero allocation claims."""
        logger.info(f"Testing cache performance ({n_operations} operations)...")

        # Test ring buffer
        buffer = LockFreeRingBuffer(1000)

        # Fill buffer
        for i in range(500):
            buffer.append(float(i))

        latencies = []

        with memory_tracker():
            for i in range(n_operations):
                start = time.perf_counter()

                # Typical cache operations
                buffer.append(float(i + 500))
                view = buffer.get_last(10)
                _ = view[0]  # Access data

                end = time.perf_counter()

                latency_us = (end - start) * 1_000_000
                latencies.append(latency_us)

        allocated_bytes = getattr(memory_tracker, "allocated_bytes", 0)
        total_time = sum(latencies) / 1_000_000

        return PerformanceResult(
            name="Cache Operations",
            min_latency_us=min(latencies),
            mean_latency_us=statistics.mean(latencies),
            p99_latency_us=np.percentile(latencies, 99),
            throughput_ops_per_sec=n_operations / total_time,
            memory_allocated_bytes=allocated_bytes,
            passes_500us_claim=np.percentile(latencies, 99) < 500,
            passes_5ms_claim=np.percentile(latencies, 99) < 5000,
            passes_zero_alloc_claim=allocated_bytes < 1000,
        )

    def test_feature_cache_performance(self, n_operations: int = 50000) -> PerformanceResult:
        """
        Test pre-allocated feature cache performance.
        """
        logger.info(f"Testing feature cache performance ({n_operations} operations)...")

        cache = PreAllocatedFeatureCache(n_features=20, history_size=1000)

        latencies = []

        with memory_tracker():
            for i in range(n_operations):
                start = time.perf_counter()

                # Typical feature cache operations
                current_buffer = cache.get_current_buffer()
                current_buffer[:] = np.random.random(20).astype(np.float32)
                cache.store_current_features()
                history = cache.get_feature_history(10)
                _ = history[0, 0]  # Access data

                end = time.perf_counter()

                latency_us = (end - start) * 1_000_000
                latencies.append(latency_us)

        allocated_bytes = getattr(memory_tracker, "allocated_bytes", 0)
        total_time = sum(latencies) / 1_000_000

        return PerformanceResult(
            name="Feature Cache",
            min_latency_us=min(latencies),
            mean_latency_us=statistics.mean(latencies),
            p99_latency_us=np.percentile(latencies, 99),
            throughput_ops_per_sec=n_operations / total_time,
            memory_allocated_bytes=allocated_bytes,
            passes_500us_claim=np.percentile(latencies, 99) < 500,
            passes_5ms_claim=np.percentile(latencies, 99) < 5000,
            passes_zero_alloc_claim=allocated_bytes < 1000,
        )

    def test_sustained_performance(self, duration_minutes: int = 5) -> PerformanceResult:
        """
        Test sustained performance over time.
        """
        logger.info(f"Testing sustained performance ({duration_minutes} minutes)...")

        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize
        for i in range(50):
            price = 1.1000 + i * 0.0001
            volume = 1000000 + i * 1000
            indicator_mgr.price_history["closes"].append(price)
            indicator_mgr.price_history["volumes"].append(volume)
            indicator_mgr.price_history["highs"].append(price + 0.0001)
            indicator_mgr.price_history["lows"].append(price - 0.0001)

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        latencies = []
        iterations = 0
        memory_samples = []

        while time.time() < end_time:
            current_bar = {
                "open": 1.1000 + np.random.normal(0, 0.0001),
                "high": 1.1001 + np.random.normal(0, 0.0001),
                "low": 1.0999 + np.random.normal(0, 0.0001),
                "close": 1.1000 + np.random.normal(0, 0.0001),
                "volume": 1000000 + np.random.exponential(100000),
            }

            iter_start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr)
            iter_end = time.perf_counter()

            latency_us = (iter_end - iter_start) * 1_000_000
            latencies.append(latency_us)

            # Sample memory usage
            if iterations % 1000 == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

                if iterations % 10000 == 0:
                    gc.collect()

            iterations += 1

        total_time = time.time() - start_time
        memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0

        return PerformanceResult(
            name=f"Sustained ({duration_minutes}min)",
            min_latency_us=min(latencies),
            mean_latency_us=statistics.mean(latencies),
            p99_latency_us=np.percentile(latencies, 99),
            throughput_ops_per_sec=iterations / total_time,
            memory_allocated_bytes=int(memory_growth * 1024 * 1024),
            passes_500us_claim=np.percentile(latencies, 99) < 500,
            passes_5ms_claim=np.percentile(latencies, 99) < 5000,
            passes_zero_alloc_claim=memory_growth < 10,  # Less than 10MB growth
        )

    def run_all_tests(self) -> dict[str, PerformanceResult]:
        """
        Run all core ML performance tests.
        """
        logger.info("Starting core ML performance tests...")

        # Test 1: Core feature computation
        self.results["feature_computation"] = self.test_feature_computation_only(10000)

        # Test 2: Cache performance
        self.results["cache_performance"] = self.test_cache_performance(100000)

        # Test 3: Feature cache performance
        self.results["feature_cache"] = self.test_feature_cache_performance(50000)

        # Test 4: Sustained performance
        self.results["sustained"] = self.test_sustained_performance(2)

        return self.results

    def generate_report(self) -> str:
        """
        Generate performance report.
        """
        if not self.results:
            self.run_all_tests()

        lines = [
            "=" * 80,
            "CORE ML PERFORMANCE ASSESSMENT",
            "=" * 80,
            "",
            "Testing ML components in isolation to measure actual performance",
            "against documented claims:",
            "",
            "CLAIMS:",
            "• Hot path <5ms P99 latency",
            "• Feature computation <500μs",
            "• Zero allocations in hot path",
            "",
        ]

        # Summary
        feature_test = self.results.get("feature_computation")
        if feature_test:
            lines.extend(
                [
                    "FEATURE COMPUTATION VERDICT:",
                    f"• P99 Latency: {feature_test.p99_latency_us:.1f}μs",
                    f"• <500μs claim: {'✓ PASS' if feature_test.passes_500us_claim else '✗ FAIL'}",
                    f"• Zero alloc: {'✓ PASS' if feature_test.passes_zero_alloc_claim else '✗ FAIL'}",
                    f"• Throughput: {feature_test.throughput_ops_per_sec:,.0f} ops/sec",
                    "",
                ],
            )

        # Detailed results
        for name, result in self.results.items():
            lines.extend(
                [
                    f"TEST: {result.name}",
                    "-" * (len(result.name) + 6),
                    "Latency (μs):",
                    f"  Min:     {result.min_latency_us:8.1f}",
                    f"  Mean:    {result.mean_latency_us:8.1f}",
                    f"  P99:     {result.p99_latency_us:8.1f} ⭐",
                    f"Throughput:  {result.throughput_ops_per_sec:10,.0f} ops/sec",
                    f"Memory:      {result.memory_allocated_bytes:10,} bytes allocated",
                    "Claims:",
                    f"  <500μs:    {'✓ PASS' if result.passes_500us_claim else '✗ FAIL'}",
                    f"  <5ms:      {'✓ PASS' if result.passes_5ms_claim else '✗ FAIL'}",
                    f"  Zero alloc: {'✓ PASS' if result.passes_zero_alloc_claim else '✗ FAIL'}",
                    "",
                ],
            )

        # Final verdict
        all_feature_pass = all(r.passes_500us_claim for r in self.results.values())
        all_zero_alloc = all(r.passes_zero_alloc_claim for r in self.results.values())

        lines.extend(
            [
                "FINAL ASSESSMENT:",
                "=" * 18,
            ],
        )

        if all_feature_pass and all_zero_alloc:
            lines.extend(
                [
                    "🎉 ML SYSTEM MEETS PERFORMANCE CLAIMS",
                    "The core ML components successfully meet their requirements.",
                ],
            )
        else:
            lines.extend(
                [
                    "⚠️  ML SYSTEM FAILS TO MEET SOME CLAIMS",
                    "Issues found:",
                ],
            )
            if not all_feature_pass:
                lines.append("• Feature computation exceeds 500μs in some tests")
            if not all_zero_alloc:
                lines.append("• Memory allocations detected in hot path")

        lines.extend(
            [
                "",
                "Note: This test isolates ML components from full Nautilus integration.",
                "Real-world performance may vary with complete system integration.",
                "=" * 80,
            ],
        )

        return "\n".join(lines)

    def save_results(self, output_dir: str = "performance_results"):
        """
        Save results to files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save JSON results
        json_data = {}
        for name, result in self.results.items():
            json_data[name] = {
                "name": result.name,
                "min_latency_us": result.min_latency_us,
                "mean_latency_us": result.mean_latency_us,
                "p99_latency_us": result.p99_latency_us,
                "throughput_ops_per_sec": result.throughput_ops_per_sec,
                "memory_allocated_bytes": result.memory_allocated_bytes,
                "passes_500us_claim": result.passes_500us_claim,
                "passes_5ms_claim": result.passes_5ms_claim,
                "passes_zero_alloc_claim": result.passes_zero_alloc_claim,
            }

        with open(output_path / "core_ml_results.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Save text report
        report = self.generate_report()
        with open(output_path / "core_ml_assessment.txt", "w") as f:
            f.write(report)

        logger.info(f"Results saved to {output_path}/")


def main():
    """
    Run core ML performance assessment.
    """
    tester = CoreMLPerformanceTester()

    try:
        results = tester.run_all_tests()
        report = tester.generate_report()
        tester.save_results()

        print(report)

        # Check if all tests pass
        all_pass = all(r.passes_500us_claim and r.passes_zero_alloc_claim for r in results.values())

        return 0 if all_pass else 1

    except Exception as e:
        logger.error(f"Core ML performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
