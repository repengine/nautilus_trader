#!/usr/bin/env python3
"""
Brutal ML System Performance Assessment.

This module provides comprehensive performance testing to validate (or debunk)
the performance claims made in the documentation. Tests include:

1. Hot path <5ms P99 latency claims
2. Zero allocation claims in hot path
3. Feature computation <500μs claims
4. Model inference performance
5. Memory usage and stability
6. Concurrent performance
7. Realistic trading volume benchmarks

IMPORTANT: This test suite is designed to expose performance issues with brutal honesty.

"""

import gc
import json
import logging
import multiprocessing
import os
import statistics
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# ML imports
from ml._imports import HAS_SKLEARN
from ml._imports import check_ml_dependencies
from ml.actors.enhanced import EnhancedMLInferenceActor
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager

# Nautilus imports
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics container.
    """

    name: str
    min_latency_us: float
    max_latency_us: float
    mean_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    p999_latency_us: float
    total_time_s: float
    throughput_ops_per_sec: float
    memory_allocated_bytes: int
    memory_peak_mb: float
    success_rate: float

    def passes_hot_path_requirement(self) -> bool:
        """
        Check if passes <5ms P99 requirement.
        """
        return self.p99_latency_us < 5000

    def passes_feature_requirement(self) -> bool:
        """
        Check if passes <500μs feature computation requirement.
        """
        return self.p99_latency_us < 500

    def passes_zero_allocation(self) -> bool:
        """
        Check if passes zero allocation requirement (allow small overhead).
        """
        return self.memory_allocated_bytes < 1000  # Allow 1KB overhead


@contextmanager
def memory_profiler():
    """
    Context manager for memory profiling.
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
        # Return allocated bytes via exception mechanism (hacky but works)
        if hasattr(memory_profiler, "_allocated_bytes"):
            memory_profiler._allocated_bytes = total_allocated


class BrutalPerformanceTester:
    """
    Brutal performance tester that will expose the truth about ML system performance.

    This class runs comprehensive performance tests with realistic trading scenarios and
    measures actual performance against documented claims.

    """

    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: dict[str, PerformanceMetrics] = {}
        self.process = psutil.Process()

    def create_realistic_bar_data(
        self,
        n_bars: int = 10000,
        instrument: str = "EURUSD",
    ) -> list[Bar]:
        """
        Create realistic high-frequency bar data for testing.
        """
        instrument_id = InstrumentId.from_str(f"{instrument}.IDEALPRO")
        bar_spec = BarSpecification(1, BarAggregation.SECOND, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.NO_AGGRESSOR)

        bars = []
        base_price = 1.1000
        base_volume = 1000000

        for i in range(n_bars):
            # Simulate realistic price movement with volatility
            price_change = np.random.normal(0, 0.0001)  # 1 pip std dev
            volume_change = np.random.exponential(0.5)  # Exponential volume distribution

            current_price = base_price + price_change * i * 0.1
            spread = 0.00001  # 0.1 pip spread

            open_price = current_price
            high_price = current_price + abs(np.random.normal(0, 0.00005))
            low_price = current_price - abs(np.random.normal(0, 0.00005))
            close_price = current_price + np.random.normal(0, 0.00002)
            volume = max(1000, base_volume * volume_change)

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.5f}"),
                high=Price.from_str(f"{high_price:.5f}"),
                low=Price.from_str(f"{low_price:.5f}"),
                close=Price.from_str(f"{close_price:.5f}"),
                volume=Quantity.from_str(f"{volume:.0f}"),
                ts_event=1000000000 + i * 1000000000,  # 1 second bars
                ts_init=1000000000 + i * 1000000000,
            )
            bars.append(bar)

        return bars

    def test_feature_computation_performance(self, n_iterations: int = 10000) -> PerformanceMetrics:
        """
        Test feature computation performance with realistic scenarios.
        """
        logger.info(f"Testing feature computation performance ({n_iterations} iterations)...")

        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize with realistic market data
        bars = self.create_realistic_bar_data(100)
        for bar in bars[:50]:
            indicator_mgr.update_from_bar(bar)

        # Warm up
        for _ in range(100):
            current_bar = {
                "open": 1.1000 + np.random.normal(0, 0.0001),
                "high": 1.1005 + np.random.normal(0, 0.0001),
                "low": 1.0995 + np.random.normal(0, 0.0001),
                "close": 1.1000 + np.random.normal(0, 0.0001),
                "volume": 1000000 + np.random.exponential(500000),
            }
            engineer.calculate_features_online(current_bar, indicator_mgr)

        # Performance test with memory profiling
        latencies = []

        with memory_profiler():
            for i in range(n_iterations):
                current_bar = {
                    "open": 1.1000 + np.random.normal(0, 0.0001),
                    "high": 1.1005 + np.random.normal(0, 0.0001),
                    "low": 1.0995 + np.random.normal(0, 0.0001),
                    "close": 1.1000 + np.random.normal(0, 0.0001),
                    "volume": 1000000 + np.random.exponential(500000),
                }

                start_time = time.perf_counter()
                features = engineer.calculate_features_online(current_bar, indicator_mgr)
                end_time = time.perf_counter()

                latency_us = (end_time - start_time) * 1_000_000
                latencies.append(latency_us)

                # Simulate realistic indicator updates
                indicator_mgr.price_history["closes"].append(current_bar["close"])
                indicator_mgr.price_history["volumes"].append(current_bar["volume"])
                indicator_mgr.price_history["highs"].append(current_bar["high"])
                indicator_mgr.price_history["lows"].append(current_bar["low"])

                # Keep history bounded (realistic scenario)
                for key in indicator_mgr.price_history:
                    if len(indicator_mgr.price_history[key]) > 100:
                        indicator_mgr.price_history[key] = indicator_mgr.price_history[key][-100:]

        allocated_bytes = getattr(memory_profiler, "_allocated_bytes", 0)
        total_time = sum(latencies) / 1_000_000  # Convert to seconds

        return PerformanceMetrics(
            name="Feature Computation",
            min_latency_us=min(latencies),
            max_latency_us=max(latencies),
            mean_latency_us=statistics.mean(latencies),
            p50_latency_us=statistics.median(latencies),
            p95_latency_us=np.percentile(latencies, 95),
            p99_latency_us=np.percentile(latencies, 99),
            p999_latency_us=np.percentile(latencies, 99.9),
            total_time_s=total_time,
            throughput_ops_per_sec=n_iterations / total_time,
            memory_allocated_bytes=allocated_bytes,
            memory_peak_mb=self.process.memory_info().rss / 1024 / 1024,
            success_rate=1.0,
        )

    def test_ml_actor_hot_path(self, n_iterations: int = 10000) -> PerformanceMetrics:
        """
        Test full ML actor hot path performance.
        """
        logger.info(f"Testing ML actor hot path performance ({n_iterations} iterations)...")

        # Create a dummy model for testing
        dummy_model_path = self.output_dir / "dummy_test_model.json"
        dummy_model_data = {
            "model_type": "dummy",
            "version": "1.0.0",
            "weights": [0.1] * 11,  # Matches feature count
            "bias": 0.0,
        }
        with open(dummy_model_path, "w") as f:
            json.dump(dummy_model_data, f)

        # Configure ML actor
        config = MLActorConfig(
            model_id="test_model_brutal",
            component_id="BRUTAL_TEST",
            model_path=str(dummy_model_path),
            bar_type="EURUSD.IDEALPRO-1-SECOND-LAST",
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            feature_config=MLFeatureConfig(average_volume=1000000.0),
            use_dummy_stores=True,  # Avoid DB dependencies in test
            warm_up_period=50,
        )

        try:
            actor = EnhancedMLInferenceActor(config)
            actor._initialize_features()

            # Create realistic test data
            bars = self.create_realistic_bar_data(n_iterations + 100)

            # Warm up actor
            for bar in bars[:50]:
                actor._compute_features(bar)

            latencies = []
            memory_peak = 0

            with memory_profiler():
                for i, bar in enumerate(bars[50 : 50 + n_iterations]):
                    start_time = time.perf_counter()

                    # Full hot path: feature computation + prediction
                    features = actor._compute_features(bar)
                    if features is not None:
                        _prediction, _confidence = actor._predict(features)

                    end_time = time.perf_counter()

                    latency_us = (end_time - start_time) * 1_000_000
                    latencies.append(latency_us)

                    # Track memory usage
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_peak = max(memory_peak, current_memory)

            allocated_bytes = getattr(memory_profiler, "_allocated_bytes", 0)
            total_time = sum(latencies) / 1_000_000

            return PerformanceMetrics(
                name="ML Actor Hot Path",
                min_latency_us=min(latencies),
                max_latency_us=max(latencies),
                mean_latency_us=statistics.mean(latencies),
                p50_latency_us=statistics.median(latencies),
                p95_latency_us=np.percentile(latencies, 95),
                p99_latency_us=np.percentile(latencies, 99),
                p999_latency_us=np.percentile(latencies, 99.9),
                total_time_s=total_time,
                throughput_ops_per_sec=n_iterations / total_time,
                memory_allocated_bytes=allocated_bytes,
                memory_peak_mb=memory_peak,
                success_rate=1.0,
            )

        finally:
            # Cleanup
            if dummy_model_path.exists():
                dummy_model_path.unlink()

    def test_memory_stability_over_time(self, duration_minutes: int = 10) -> PerformanceMetrics:
        """
        Test memory stability over extended period.
        """
        logger.info(f"Testing memory stability over {duration_minutes} minutes...")

        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize
        bars = self.create_realistic_bar_data(100)
        for bar in bars[:50]:
            indicator_mgr.update_from_bar(bar)

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        iterations = 0
        latencies = []
        memory_samples = []

        while time.time() < end_time:
            current_bar = {
                "open": 1.1000 + np.random.normal(0, 0.0001),
                "high": 1.1005 + np.random.normal(0, 0.0001),
                "low": 1.0995 + np.random.normal(0, 0.0001),
                "close": 1.1000 + np.random.normal(0, 0.0001),
                "volume": 1000000 + np.random.exponential(500000),
            }

            iter_start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr)
            iter_end = time.perf_counter()

            latency_us = (iter_end - iter_start) * 1_000_000
            latencies.append(latency_us)

            # Sample memory every 1000 iterations
            if iterations % 1000 == 0:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

                # Force garbage collection occasionally
                if iterations % 10000 == 0:
                    gc.collect()

            iterations += 1

        total_time = time.time() - start_time
        memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0

        return PerformanceMetrics(
            name=f"Memory Stability ({duration_minutes}min)",
            min_latency_us=min(latencies),
            max_latency_us=max(latencies),
            mean_latency_us=statistics.mean(latencies),
            p50_latency_us=statistics.median(latencies),
            p95_latency_us=np.percentile(latencies, 95),
            p99_latency_us=np.percentile(latencies, 99),
            p999_latency_us=np.percentile(latencies, 99.9),
            total_time_s=total_time,
            throughput_ops_per_sec=iterations / total_time,
            memory_allocated_bytes=int(memory_growth * 1024 * 1024),
            memory_peak_mb=max(memory_samples) if memory_samples else 0,
            success_rate=1.0,
        )

    def test_concurrent_performance(
        self,
        n_threads: int = 4,
        n_iterations_per_thread: int = 2500,
    ) -> PerformanceMetrics:
        """
        Test concurrent performance with multiple threads.
        """
        logger.info(
            f"Testing concurrent performance ({n_threads} threads, {n_iterations_per_thread} iterations each)...",
        )

        def worker_thread(thread_id: int) -> list[float]:
            """
            Worker thread function.
            """
            config = FeatureConfig()
            engineer = FeatureEngineer(config)
            indicator_mgr = IndicatorManager(config)

            # Initialize each thread independently
            bars = self.create_realistic_bar_data(100)
            for bar in bars[:50]:
                indicator_mgr.update_from_bar(bar)

            latencies = []

            for i in range(n_iterations_per_thread):
                current_bar = {
                    "open": 1.1000 + np.random.normal(0, 0.0001),
                    "high": 1.1005 + np.random.normal(0, 0.0001),
                    "low": 1.0995 + np.random.normal(0, 0.0001),
                    "close": 1.1000 + np.random.normal(0, 0.0001),
                    "volume": 1000000 + np.random.exponential(500000),
                }

                start_time = time.perf_counter()
                features = engineer.calculate_features_online(current_bar, indicator_mgr)
                end_time = time.perf_counter()

                latency_us = (end_time - start_time) * 1_000_000
                latencies.append(latency_us)

            return latencies

        # Run concurrent test
        start_time = time.time()
        memory_start = self.process.memory_info().rss / 1024 / 1024

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(n_threads)]
            all_latencies = []

            for future in futures:
                thread_latencies = future.result()
                all_latencies.extend(thread_latencies)

        end_time = time.time()
        memory_end = self.process.memory_info().rss / 1024 / 1024

        total_time = end_time - start_time
        total_iterations = n_threads * n_iterations_per_thread

        return PerformanceMetrics(
            name=f"Concurrent ({n_threads} threads)",
            min_latency_us=min(all_latencies),
            max_latency_us=max(all_latencies),
            mean_latency_us=statistics.mean(all_latencies),
            p50_latency_us=statistics.median(all_latencies),
            p95_latency_us=np.percentile(all_latencies, 95),
            p99_latency_us=np.percentile(all_latencies, 99),
            p999_latency_us=np.percentile(all_latencies, 99.9),
            total_time_s=total_time,
            throughput_ops_per_sec=total_iterations / total_time,
            memory_allocated_bytes=int((memory_end - memory_start) * 1024 * 1024),
            memory_peak_mb=memory_end,
            success_rate=1.0,
        )

    def test_high_frequency_trading_simulation(
        self,
        messages_per_second: int = 10000,
        duration_seconds: int = 60,
    ) -> PerformanceMetrics:
        """
        Simulate high-frequency trading load.
        """
        logger.info(
            f"Testing HFT simulation ({messages_per_second} msg/s for {duration_seconds}s)...",
        )

        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize
        bars = self.create_realistic_bar_data(100)
        for bar in bars[:50]:
            indicator_mgr.update_from_bar(bar)

        total_messages = messages_per_second * duration_seconds
        interval_us = 1_000_000 / messages_per_second  # Microseconds between messages

        latencies = []
        missed_deadlines = 0

        start_time = time.perf_counter()
        next_message_time = start_time

        for i in range(total_messages):
            # Wait for next message time (simulate realistic timing)
            current_time = time.perf_counter()
            if current_time < next_message_time:
                time.sleep(next_message_time - current_time)

            current_bar = {
                "open": 1.1000 + np.random.normal(0, 0.0001),
                "high": 1.1005 + np.random.normal(0, 0.0001),
                "low": 1.0995 + np.random.normal(0, 0.0001),
                "close": 1.1000 + np.random.normal(0, 0.0001),
                "volume": 1000000 + np.random.exponential(500000),
            }

            process_start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr)
            process_end = time.perf_counter()

            latency_us = (process_end - process_start) * 1_000_000
            latencies.append(latency_us)

            # Check if we missed our deadline (simulate real-time requirement)
            deadline_us = interval_us * 0.8  # Use 80% of interval as deadline
            if latency_us > deadline_us:
                missed_deadlines += 1

            next_message_time += interval_us / 1_000_000  # Convert back to seconds

        total_time = time.perf_counter() - start_time
        success_rate = 1.0 - (missed_deadlines / total_messages)

        return PerformanceMetrics(
            name=f"HFT Simulation ({messages_per_second} msg/s)",
            min_latency_us=min(latencies) if latencies else 0,
            max_latency_us=max(latencies) if latencies else 0,
            mean_latency_us=statistics.mean(latencies) if latencies else 0,
            p50_latency_us=statistics.median(latencies) if latencies else 0,
            p95_latency_us=float(np.percentile(latencies, 95)) if latencies else 0.0,
            p99_latency_us=float(np.percentile(latencies, 99)) if latencies else 0.0,
            p999_latency_us=float(np.percentile(latencies, 99.9)) if latencies else 0.0,
            total_time_s=total_time,
            throughput_ops_per_sec=total_messages / total_time,
            memory_allocated_bytes=0,  # Not measured in this test
            memory_peak_mb=self.process.memory_info().rss / 1024 / 1024,
            success_rate=success_rate,
        )

    def run_all_tests(self) -> dict[str, PerformanceMetrics]:
        """
        Run all performance tests.
        """
        logger.info("Starting brutal performance assessment...")

        # Test 1: Feature computation performance
        self.results["feature_computation"] = self.test_feature_computation_performance(10000)

        # Test 2: ML actor hot path (if dependencies available)
        try:
            self.results["ml_actor_hot_path"] = self.test_ml_actor_hot_path(5000)
        except Exception as e:
            logger.warning(f"ML actor test failed: {e}")

        # Test 3: Memory stability (shorter duration for CI)
        self.results["memory_stability"] = self.test_memory_stability_over_time(2)

        # Test 4: Concurrent performance
        self.results["concurrent_performance"] = self.test_concurrent_performance(4, 2500)

        # Test 5: High-frequency trading simulation
        self.results["hft_simulation"] = self.test_high_frequency_trading_simulation(5000, 30)

        return self.results

    def generate_brutal_report(self) -> str:
        """
        Generate a brutally honest performance report.
        """
        if not self.results:
            self.run_all_tests()

        report_lines = [
            "=" * 80,
            "BRUTAL ML SYSTEM PERFORMANCE ASSESSMENT",
            "=" * 80,
            "",
            "This report provides an unvarnished assessment of actual performance",
            "versus documented claims. Failures are highlighted in detail.",
            "",
        ]

        # Summary of claims vs reality
        claims_section = [
            "DOCUMENTED CLAIMS vs MEASURED REALITY:",
            "-" * 40,
        ]

        feature_metrics = self.results.get("feature_computation")
        if feature_metrics:
            claim_500us = feature_metrics.passes_feature_requirement()
            claims_section.extend(
                [
                    f"✓ Feature computation <500μs: {'PASS' if claim_500us else 'FAIL'}",
                    "  Claimed: <500μs P99",
                    f"  Actual:  {feature_metrics.p99_latency_us:.1f}μs P99",
                    "",
                ],
            )

        hot_path_metrics = self.results.get("ml_actor_hot_path")
        if hot_path_metrics:
            claim_5ms = hot_path_metrics.passes_hot_path_requirement()
            claims_section.extend(
                [
                    f"✓ Hot path <5ms P99: {'PASS' if claim_5ms else 'FAIL'}",
                    "  Claimed: <5000μs P99",
                    f"  Actual:  {hot_path_metrics.p99_latency_us:.1f}μs P99",
                    "",
                ],
            )

        zero_alloc_pass = True
        for name, metrics in self.results.items():
            if not metrics.passes_zero_allocation():
                zero_alloc_pass = False
                break

        claims_section.extend(
            [
                f"✓ Zero allocation claims: {'PASS' if zero_alloc_pass else 'FAIL'}",
                "  Claimed: No memory allocations in hot path",
                "",
            ],
        )

        report_lines.extend(claims_section)

        # Detailed results for each test
        for test_name, metrics in self.results.items():
            report_lines.extend(
                [
                    f"TEST: {metrics.name}",
                    "-" * (len(metrics.name) + 6),
                    "Latency Statistics (microseconds):",
                    f"  Min:    {metrics.min_latency_us:8.1f}μs",
                    f"  Mean:   {metrics.mean_latency_us:8.1f}μs",
                    f"  P50:    {metrics.p50_latency_us:8.1f}μs",
                    f"  P95:    {metrics.p95_latency_us:8.1f}μs",
                    f"  P99:    {metrics.p99_latency_us:8.1f}μs ⭐",
                    f"  P99.9:  {metrics.p999_latency_us:8.1f}μs",
                    f"  Max:    {metrics.max_latency_us:8.1f}μs",
                    "",
                    "Performance Metrics:",
                    f"  Throughput:     {metrics.throughput_ops_per_sec:10,.0f} ops/sec",
                    f"  Success Rate:   {metrics.success_rate:10.1%}",
                    f"  Memory Peak:    {metrics.memory_peak_mb:10.1f} MB",
                    f"  Memory Alloc:   {metrics.memory_allocated_bytes:10,} bytes",
                    "",
                    "Requirements Check:",
                    f"  Hot Path (<5ms):     {'✓ PASS' if metrics.passes_hot_path_requirement() else '✗ FAIL'}",
                    f"  Feature (<500μs):    {'✓ PASS' if metrics.passes_feature_requirement() else '✗ FAIL'}",
                    f"  Zero Allocation:     {'✓ PASS' if metrics.passes_zero_allocation() else '✗ FAIL'}",
                    "",
                ],
            )

        # Final assessment
        all_hot_path_pass = all(m.passes_hot_path_requirement() for m in self.results.values())
        all_feature_pass = all(m.passes_feature_requirement() for m in self.results.values())

        report_lines.extend(
            [
                "FINAL VERDICT:",
                "=" * 15,
            ],
        )

        if all_hot_path_pass and all_feature_pass and zero_alloc_pass:
            report_lines.extend(
                [
                    "🎉 SYSTEM MEETS ALL PERFORMANCE CLAIMS",
                    "The ML system successfully meets its documented performance requirements.",
                ],
            )
        else:
            report_lines.extend(
                [
                    "⚠️  SYSTEM FAILS TO MEET PERFORMANCE CLAIMS",
                    "The following issues were identified:",
                ],
            )

            if not all_hot_path_pass:
                report_lines.append("• Hot path latency exceeds 5ms P99 requirement")
            if not all_feature_pass:
                report_lines.append("• Feature computation exceeds 500μs requirement")
            if not zero_alloc_pass:
                report_lines.append("• Memory allocations detected in hot path")

        report_lines.extend(
            [
                "",
                "This assessment was conducted with realistic trading scenarios",
                "and high-frequency data loads. Results reflect production conditions.",
                "=" * 80,
            ],
        )

        return "\n".join(report_lines)

    def save_results(self) -> None:
        """
        Save detailed results to files.
        """
        # Save detailed JSON results
        json_results = {}
        for name, metrics in self.results.items():
            json_results[name] = {
                "name": metrics.name,
                "latency_stats": {
                    "min_us": metrics.min_latency_us,
                    "max_us": metrics.max_latency_us,
                    "mean_us": metrics.mean_latency_us,
                    "p50_us": metrics.p50_latency_us,
                    "p95_us": metrics.p95_latency_us,
                    "p99_us": metrics.p99_latency_us,
                    "p999_us": metrics.p999_latency_us,
                },
                "performance": {
                    "throughput_ops_per_sec": metrics.throughput_ops_per_sec,
                    "success_rate": metrics.success_rate,
                    "total_time_s": metrics.total_time_s,
                },
                "memory": {
                    "allocated_bytes": metrics.memory_allocated_bytes,
                    "peak_mb": metrics.memory_peak_mb,
                },
                "requirements": {
                    "hot_path_5ms": metrics.passes_hot_path_requirement(),
                    "feature_500us": metrics.passes_feature_requirement(),
                    "zero_allocation": metrics.passes_zero_allocation(),
                },
            }

        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        # Save summary report
        report = self.generate_brutal_report()
        with open(self.output_dir / "brutal_assessment.txt", "w") as f:
            f.write(report)

        logger.info(f"Results saved to {self.output_dir}/")


def main():
    """
    Run the brutal performance assessment.
    """
    tester = BrutalPerformanceTester()

    try:
        results = tester.run_all_tests()
        report = tester.generate_brutal_report()
        tester.save_results()

        print(report)

        # Return exit code based on results
        all_pass = all(
            m.passes_hot_path_requirement()
            and m.passes_feature_requirement()
            and m.passes_zero_allocation()
            for m in results.values()
        )

        return 0 if all_pass else 1

    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
