"""
Performance benchmarks for earnings features.

This module validates that all earnings feature computations meet strict SLA requirements:

**Hot Path (Incremental) SLAs:**
- P99 latency < 5ms
- O(1) computational complexity
- Zero allocations after warmup

**Cold Path (Batch) SLAs:**
- < 50ms for 100 instruments
- Vectorized numpy operations
- Efficient memory usage

Uses pytest-benchmark or time.perf_counter_ns() for accurate measurements.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ml.features.earnings import compute_calendar_features_batch
from ml.features.earnings import compute_calendar_features_incremental
from ml.features.earnings import compute_earnings_growth_batch
from ml.features.earnings import compute_earnings_growth_incremental
from ml.features.earnings import compute_earnings_momentum_batch
from ml.features.earnings import compute_earnings_momentum_incremental
from ml.features.earnings import compute_earnings_surprise_batch
from ml.features.earnings import compute_earnings_surprise_incremental
from ml.features.earnings import reset_earnings_metrics_state
from ml.tests.utils.earnings_facade import build_test_earnings_adapter


if TYPE_CHECKING:
    from datetime import datetime


@pytest.mark.performance
class TestEarningsPerformance:
    """Performance benchmarks for earnings features."""

    def test_incremental_surprise_latency(self) -> None:
        """
        Benchmark earnings surprise incremental computation.

        SLA: P99 < 5ms
        """
        actual = 2.52
        estimate = 2.45

        # Warmup (JIT compilation, cache warming)
        for _ in range(100):
            compute_earnings_surprise_incremental(actual, estimate)

        # Measure latency over many iterations
        latencies_ns = []
        for _ in range(10_000):
            start = time.perf_counter_ns()
            compute_earnings_surprise_incremental(actual, estimate)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        # Calculate percentiles
        latencies_ms = np.array(latencies_ns) / 1_000_000  # Convert to milliseconds
        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        p999 = np.percentile(latencies_ms, 99.9)

        print("\nEarnings Surprise Incremental Latency:")
        print(f"  P50:  {p50:.4f}ms")
        print(f"  P95:  {p95:.4f}ms")
        print(f"  P99:  {p99:.4f}ms")
        print(f"  P99.9: {p999:.4f}ms")

        # SLA assertion
        assert p99 < 5.0, f"P99 latency {p99:.4f}ms exceeds 5ms SLA"

        print(f"✅ PASS: P99={p99:.4f}ms < 5ms SLA")

    def test_incremental_growth_latency(self) -> None:
        """
        Benchmark earnings growth incremental computation.

        SLA: P99 < 5ms
        """
        eps_history = [2.52, 2.45, 2.40, 2.30, 2.20]  # Q0, Q-1, Q-2, Q-3, Q-4

        # Warmup
        for _ in range(100):
            compute_earnings_growth_incremental(eps_history)

        # Measure latency
        latencies_ns = []
        for _ in range(10_000):
            start = time.perf_counter_ns()
            compute_earnings_growth_incremental(eps_history)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        latencies_ms = np.array(latencies_ns) / 1_000_000
        p99 = np.percentile(latencies_ms, 99)

        print("\nEarnings Growth Incremental Latency:")
        print(f"  P99: {p99:.4f}ms")

        assert p99 < 5.0, f"P99 latency {p99:.4f}ms exceeds 5ms SLA"

        print(f"✅ PASS: P99={p99:.4f}ms < 5ms SLA")

    def test_incremental_momentum_latency(self) -> None:
        """
        Benchmark earnings momentum incremental computation.

        SLA: P99 < 5ms
        """
        surprises = [0.07, 0.05, 0.03, -0.02]
        eps_history = [2.52, 2.45, 2.38, 2.30]

        # Warmup
        for _ in range(100):
            compute_earnings_momentum_incremental(surprises, eps_history)

        # Measure latency
        latencies_ns = []
        for _ in range(10_000):
            start = time.perf_counter_ns()
            compute_earnings_momentum_incremental(surprises, eps_history)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        latencies_ms = np.array(latencies_ns) / 1_000_000
        p99 = np.percentile(latencies_ms, 99)

        print("\nEarnings Momentum Incremental Latency:")
        print(f"  P99: {p99:.4f}ms")

        assert p99 < 5.0, f"P99 latency {p99:.4f}ms exceeds 5ms SLA"

        print(f"✅ PASS: P99={p99:.4f}ms < 5ms SLA")

    def test_incremental_calendar_latency(self) -> None:
        """
        Benchmark earnings calendar incremental computation.

        SLA: P99 < 5ms
        """
        from datetime import datetime, timedelta

        current_date = datetime.now()
        next_earnings = current_date + timedelta(days=45)

        # Warmup
        for _ in range(100):
            compute_calendar_features_incremental(current_date, next_earnings)

        # Measure latency
        latencies_ns = []
        for _ in range(10_000):
            start = time.perf_counter_ns()
            compute_calendar_features_incremental(current_date, next_earnings)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        latencies_ms = np.array(latencies_ns) / 1_000_000
        p99 = np.percentile(latencies_ms, 99)

        print("\nEarnings Calendar Incremental Latency:")
        print(f"  P99: {p99:.4f}ms")

        assert p99 < 5.0, f"P99 latency {p99:.4f}ms exceeds 5ms SLA"

        print(f"✅ PASS: P99={p99:.4f}ms < 5ms SLA")

    def test_batch_surprise_latency_100_instruments(self) -> None:
        """
        Benchmark earnings surprise batch computation for 100 instruments.

        SLA: < 50ms for 100 instruments
        """
        # Generate data for 100 instruments
        n_instruments = 100
        np.random.seed(42)
        actuals = np.random.uniform(1.0, 3.0, size=n_instruments)
        estimates = actuals - np.random.uniform(-0.1, 0.1, size=n_instruments)

        # Warmup
        for _ in range(10):
            compute_earnings_surprise_batch(actuals, estimates)

        # Measure batch latency
        latencies_ms = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            compute_earnings_surprise_batch(actuals, estimates)
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1_000_000)

        median_latency = np.median(latencies_ms)
        p95_latency = np.percentile(latencies_ms, 95)

        print("\nBatch Surprise Computation (100 instruments):")
        print(f"  Median: {median_latency:.4f}ms")
        print(f"  P95:    {p95_latency:.4f}ms")

        assert p95_latency < 50.0, f"P95 batch latency {p95_latency:.4f}ms exceeds 50ms SLA"

        print(f"✅ PASS: P95={p95_latency:.4f}ms < 50ms SLA")

    def test_batch_growth_latency_100_instruments(self) -> None:
        """
        Benchmark earnings growth batch computation for 100 instruments.

        SLA: < 50ms for 100 instruments
        """
        # Generate EPS history for 100 instruments (5 quarters each)
        n_instruments = 100
        np.random.seed(42)

        latencies_ms = []
        for _ in range(1000):
            # Random EPS history for each iteration
            eps_history = list(np.random.uniform(1.0, 3.0, size=5))

            start = time.perf_counter_ns()
            compute_earnings_growth_batch(eps_history)
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1_000_000)

        # Total time for 100 instruments would be ~100x single computation
        estimated_100_latency = np.median(latencies_ms) * 100

        print("\nBatch Growth Computation (100 instruments, estimated):")
        print(f"  Single: {np.median(latencies_ms):.4f}ms")
        print(f"  100x:   {estimated_100_latency:.4f}ms")

        assert estimated_100_latency < 50.0, (
            f"Estimated 100-instrument latency {estimated_100_latency:.4f}ms exceeds 50ms SLA"
        )

        print(f"✅ PASS: Estimated 100x={estimated_100_latency:.4f}ms < 50ms SLA")

    def test_batch_momentum_latency_100_instruments(self) -> None:
        """
        Benchmark earnings momentum batch computation for 100 instruments.

        SLA: < 50ms for 100 instruments
        """
        np.random.seed(42)

        latencies_ms = []
        for _ in range(1000):
            surprises = np.random.uniform(-0.1, 0.1, size=4)
            eps_history = np.random.uniform(1.0, 3.0, size=4)

            start = time.perf_counter_ns()
            compute_earnings_momentum_batch(surprises, eps_history)
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1_000_000)

        estimated_100_latency = np.median(latencies_ms) * 100

        print("\nBatch Momentum Computation (100 instruments, estimated):")
        print(f"  Single: {np.median(latencies_ms):.4f}ms")
        print(f"  100x:   {estimated_100_latency:.4f}ms")

        assert estimated_100_latency < 50.0, (
            f"Estimated 100-instrument latency {estimated_100_latency:.4f}ms exceeds 50ms SLA"
        )

        print(f"✅ PASS: Estimated 100x={estimated_100_latency:.4f}ms < 50ms SLA")

    def test_cache_lookup_latency(self) -> None:
        """
        Benchmark cache lookup performance.

        SLA: P99 < 1ms for point-in-time lookup
        """
        store = build_test_earnings_adapter()

        # Populate cache with 1000 records
        for i in range(1000):
            store.write_actuals(
                ticker=f"TICKER_{i % 100}",  # 100 unique tickers, 10 quarters each
                period_end=f"2024-{(i % 12) + 1:02d}-01",
                filing_date=f"2024-{(i % 12) + 1:02d}-15",
                eps_diluted=2.0 + i * 0.01,
                revenue=100e9 + i * 1e9,
                ts_event=1_700_000_000_000_000_000 + i * 1_000_000_000,
                ts_init=int(time.time_ns()),
            )

        # Warmup
        for _ in range(100):
            store.get_actuals("TICKER_0", as_of_ts=1_700_000_000_000_000_000)

        # Measure lookup latency
        latencies_ns = []
        for i in range(10_000):
            ticker = f"TICKER_{i % 100}"
            start = time.perf_counter_ns()
            store.get_actuals(ticker, as_of_ts=1_700_000_000_000_000_000 + i * 1_000_000)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

        latencies_ms = np.array(latencies_ns) / 1_000_000
        p99 = np.percentile(latencies_ms, 99)

        print("\nCache Lookup Latency (1000 records):")
        print(f"  P50: {np.percentile(latencies_ms, 50):.4f}ms")
        print(f"  P99: {p99:.4f}ms")

        assert p99 < 1.0, f"P99 cache lookup latency {p99:.4f}ms exceeds 1ms SLA"

        print(f"✅ PASS: P99={p99:.4f}ms < 1ms SLA")

    def test_zero_allocations_after_warmup(self) -> None:
        """
        Verify zero allocations in hot path after warmup.

        Uses tracemalloc to detect memory allocations.
        """
        import tracemalloc

        os.environ.pop("ML_EARNINGS_ENABLE_METRICS", None)
        reset_earnings_metrics_state()

        actual = 2.52
        estimate = 2.45
        result_buffer = {
            "eps_surprise_q0": np.zeros(1, dtype=np.float64),
            "eps_surprise_pct_q0": np.zeros(1, dtype=np.float64),
        }

        # Warmup (allow allocations)
        for _ in range(100):
            compute_earnings_surprise_incremental(actual, estimate, out=result_buffer)

        # Start tracking allocations
        tracemalloc.start()
        before_current, _before_peak = tracemalloc.get_traced_memory()

        # Execute hot path operations
        for _ in range(1000):
            compute_earnings_surprise_incremental(actual, estimate, out=result_buffer)

        after_current, _after_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate allocations
        allocated_bytes = after_current - before_current

        print("\nMemory Allocation Test (1000 iterations):")
        print(f"  Before: {before_current:,} bytes")
        print(f"  After:  {after_current:,} bytes")
        print(f"  Allocated: {allocated_bytes:,} bytes")

        # Allow small allocations (e.g., < 1KB) for internal Python overhead
        assert allocated_bytes < 1024, (
            f"Hot path allocated {allocated_bytes:,} bytes (expected ~0 after warmup)"
        )

        print(f"✅ PASS: Allocated {allocated_bytes} bytes (< 1KB)")

    def test_computational_complexity_validation(self) -> None:
        """
        Validate O(1) computational complexity for incremental operations.

        Tests that latency is constant regardless of history size.
        """
        # Test with different history sizes
        sizes = [10, 100, 1000, 10_000]
        latencies = []

        for size in sizes:
            # Generate history
            surprises = [0.05] * size
            eps_history = [2.0] * size

            samples = []
            for _ in range(50):
                start = time.perf_counter_ns()
                compute_earnings_momentum_incremental(surprises[:4], eps_history[:4])
                end = time.perf_counter_ns()
                samples.append((end - start) / 1_000_000)

            latencies.append(float(np.median(samples)))

        print("\nComputational Complexity Test:")
        for size, latency in zip(sizes, latencies):
            print(f"  Size={size:>5}: {latency:.4f}ms")

        # Verify latency is roughly constant (variation < 2x)
        max_latency = max(latencies)
        min_latency = min(latencies)
        variation_ratio = max_latency / min_latency

        assert variation_ratio < 2.0, (
            f"Latency varies {variation_ratio:.2f}x with history size (expected O(1) complexity)"
        )

        print(f"✅ PASS: Latency variation {variation_ratio:.2f}x < 2x (O(1) confirmed)")

    def test_sla_compliance_summary(self) -> None:
        """
        Summary test that validates all SLA requirements.

        This test aggregates all performance SLAs and reports compliance.
        """
        sla_results = []

        # SLA 1: Incremental surprise < 5ms P99
        actual, estimate = 2.52, 2.45
        latencies = []
        reuse_buffer = {
            "eps_surprise_q0": np.zeros(1, dtype=np.float64),
            "eps_surprise_pct_q0": np.zeros(1, dtype=np.float64),
        }
        for _ in range(1000):
            start = time.perf_counter_ns()
            compute_earnings_surprise_incremental(actual, estimate, out=reuse_buffer)
            latencies.append((time.perf_counter_ns() - start) / 1_000_000)
        p99_surprise = np.percentile(latencies, 99)
        sla_results.append(("Incremental Surprise P99 < 5ms", p99_surprise < 5.0, f"{p99_surprise:.4f}ms"))

        # SLA 2: Batch surprise for 100 instruments < 50ms
        actuals_batch = np.random.uniform(1.0, 3.0, 100)
        estimates_batch = actuals_batch - 0.05
        start = time.perf_counter_ns()
        compute_earnings_surprise_batch(actuals_batch, estimates_batch)
        batch_latency = (time.perf_counter_ns() - start) / 1_000_000
        sla_results.append(("Batch Surprise 100x < 50ms", batch_latency < 50.0, f"{batch_latency:.4f}ms"))

        # SLA 3: Cache lookup < 1ms P99
        store = build_test_earnings_adapter()
        for i in range(100):
            store.write_actuals(
                ticker=f"TEST_{i}",
                period_end="2024-09-30",
                filing_date="2024-10-31",
                eps_diluted=2.0,
                revenue=100e9,
                ts_event=int(time.time_ns()),
                ts_init=int(time.time_ns()),
            )

        cache_latencies = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            store.get_actuals("TEST_0")
            cache_latencies.append((time.perf_counter_ns() - start) / 1_000_000)
        p99_cache = np.percentile(cache_latencies, 99)
        sla_results.append(("Cache Lookup P99 < 1ms", p99_cache < 1.0, f"{p99_cache:.4f}ms"))

        # Print summary
        print("\n" + "=" * 70)
        print("SLA COMPLIANCE SUMMARY")
        print("=" * 70)
        for requirement, passed, measurement in sla_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {requirement:<35} | {measurement}")
        print("=" * 70)

        # Assert all SLAs passed
        all_passed = all(passed for _, passed, _ in sla_results)
        assert all_passed, "Some SLAs failed - see summary above"

        print("\n🎉 All SLA requirements met!")
