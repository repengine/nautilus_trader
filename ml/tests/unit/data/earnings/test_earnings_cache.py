"""
Unit tests for EarningsCache.

Tests cache hit/miss behavior, LRU eviction, cache warming, and invalidation.
Ensures <1ms P99 latency and >90% hit rate for sequential access.

Coverage target: ≥90%
"""

import time

import pytest

from ml.data.earnings.earnings_cache import EarningsCache
from ml.tests.utils.earnings_facade import build_test_earnings_adapter


class TestEarningsCacheBasics:
    """Test basic cache functionality."""

    def test_cache_initialization(self) -> None:
        """Test cache initializes correctly."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        stats = cache.get_stats()
        assert stats["maxsize"] == 128
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_cache_wraps_datastore_adapter(self) -> None:
        """Test cache works with DataStore-backed earnings adapter."""
        store = build_test_earnings_adapter()

        # Populate store
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-09-30",
            filing_date="2024-11-01",
            eps_diluted=1.64,
            revenue=94_900_000_000.0,
            ts_event=1730419200000000000,  # 2024-11-01 in nanoseconds
            ts_init=1730419200000000000,
            fiscal_year=2024,
            fiscal_quarter=4,
        )

        # Create cache
        cache = EarningsCache(store, maxsize=128)

        # First access (cache miss)
        actuals = cache.get_actuals_at(
            ticker="AAPL",
            as_of_ts=1730505600000000000,  # 2024-11-02 (after filing)
        )

        assert len(actuals) == 1
        assert actuals[0]["eps_diluted"] == 1.64

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_hit_on_repeated_access(self) -> None:
        """Test cache hits on repeated access with same parameters."""
        store = build_test_earnings_adapter()
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-09-30",
            filing_date="2024-11-01",
            eps_diluted=1.64,
            revenue=94_900_000_000.0,
            ts_event=1730419200000000000,
            ts_init=1730419200000000000,
        )

        cache = EarningsCache(store, maxsize=128)

        # First access (miss)
        as_of_ts = 1730505600000000000
        actuals1 = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)

        # Second access with same parameters (hit)
        actuals2 = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)

        assert actuals1 == actuals2

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_miss_on_different_timestamp(self) -> None:
        """Test cache misses when as_of_ts changes."""
        store = build_test_earnings_adapter()
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-09-30",
            filing_date="2024-11-01",
            eps_diluted=1.64,
            revenue=94_900_000_000.0,
            ts_event=1730419200000000000,
            ts_init=1730419200000000000,
        )

        cache = EarningsCache(store, maxsize=128)

        # Access at T1
        actuals1 = cache.get_actuals_at("AAPL", as_of_ts=1730505600000000000)

        # Access at T2 (different timestamp, different cache key)
        actuals2 = cache.get_actuals_at("AAPL", as_of_ts=1730592000000000000)

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2


class TestCacheLRUEviction:
    """Test LRU eviction behavior."""

    def test_lru_eviction_on_maxsize_exceeded(self) -> None:
        """Test LRU eviction when cache exceeds maxsize."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=2)  # Very small cache

        # Populate with 3 different entries (exceeds maxsize)
        ts1 = 1704067200000000000  # 2024-01-01
        ts2 = 1704153600000000000  # 2024-01-02
        ts3 = 1704240000000000000  # 2024-01-03

        cache.get_actuals_at("AAPL", as_of_ts=ts1)  # Miss
        cache.get_actuals_at("AAPL", as_of_ts=ts2)  # Miss
        cache.get_actuals_at("AAPL", as_of_ts=ts3)  # Miss, evicts ts1

        # Access ts1 again (should be miss, was evicted)
        cache.get_actuals_at("AAPL", as_of_ts=ts1)  # Miss

        stats = cache.get_stats()
        assert stats["misses"] == 4  # All misses
        assert stats["size"] == 2  # Cache size capped at maxsize

    def test_lru_preserves_recently_accessed(self) -> None:
        """Test LRU keeps recently accessed entries."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=2)

        ts1 = 1704067200000000000
        ts2 = 1704153600000000000
        ts3 = 1704240000000000000

        cache.get_actuals_at("AAPL", as_of_ts=ts1)  # Miss
        cache.get_actuals_at("AAPL", as_of_ts=ts2)  # Miss
        cache.get_actuals_at("AAPL", as_of_ts=ts1)  # Hit (refresh ts1)
        cache.get_actuals_at("AAPL", as_of_ts=ts3)  # Miss, evicts ts2 (not ts1)

        # Access ts1 (should still be cached)
        cache.get_actuals_at("AAPL", as_of_ts=ts1)  # Hit

        # Access ts2 (was evicted)
        cache.get_actuals_at("AAPL", as_of_ts=ts2)  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2  # ts1 twice
        assert stats["misses"] == 4


class TestCacheWarming:
    """Test cache warming for backtest windows."""

    def test_warm_cache_sequential_access(self) -> None:
        """Test cache warming improves hit rate for sequential access."""
        store = build_test_earnings_adapter()
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-02-01",
            eps_diluted=2.10,
            revenue=100_000_000_000.0,
            ts_event=1706745600000000000,  # 2024-02-01
            ts_init=1706745600000000000,
        )

        cache = EarningsCache(store, maxsize=128)

        # Warm cache for Q1 2024 (weekly samples)
        start_ts = 1704067200000000000  # 2024-01-01
        end_ts = 1711929600000000000  # 2024-04-01
        cache.warm_cache("AAPL", start_ts, end_ts, step_days=7)

        # Verify warming loaded data
        stats_after_warm = cache.get_stats()
        assert stats_after_warm["size"] > 0

        # Subsequent accesses within window should have high hit rate
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)  # Should hit
        cache.get_actuals_at("AAPL", as_of_ts=1704672000000000000)  # Should hit

        final_stats = cache.get_stats()
        # Some hits from warming + subsequent accesses
        assert final_stats["hits"] > 0

    def test_warm_cache_calculates_step_correctly(self) -> None:
        """Test cache warming calculates step intervals correctly."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=1024)

        # 1 month window with 7-day steps
        start_ts = 1704067200000000000  # 2024-01-01
        end_ts = 1706745600000000000  # 2024-02-01 (31 days)

        cache.warm_cache("AAPL", start_ts, end_ts, step_days=7)

        stats = cache.get_stats()
        # Should have ~5 entries (31 days / 7 days ≈ 4.4, rounds to 5)
        assert stats["size"] >= 4
        assert stats["size"] <= 6


class TestCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_clears_cache(self) -> None:
        """Test invalidation clears all cache entries."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Populate cache
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)
        cache.get_actuals_at("MSFT", as_of_ts=1704067200000000000)

        stats_before = cache.get_stats()
        assert stats_before["size"] == 2

        # Invalidate
        cache.invalidate()

        stats_after = cache.get_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0

    def test_invalidate_after_new_data(self) -> None:
        """Test cache invalidation after new data arrives."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Initial access
        as_of_ts = 1730505600000000000
        actuals1 = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)
        assert len(actuals1) == 0  # No data yet

        # Add new data
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-09-30",
            filing_date="2024-11-01",
            eps_diluted=1.64,
            revenue=94_900_000_000.0,
            ts_event=1730419200000000000,
            ts_init=1730419200000000000,
        )

        # Without invalidation, cache returns stale data
        actuals2 = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)
        assert len(actuals2) == 0  # Still cached empty result

        # Invalidate cache
        cache.invalidate()

        # Now sees new data
        actuals3 = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)
        assert len(actuals3) == 1
        assert actuals3[0]["eps_diluted"] == 1.64


class TestCacheEstimates:
    """Test cache behavior for estimates."""

    def test_estimates_caching(self) -> None:
        """Test estimates are cached correctly."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Populate estimate
        store.write_estimates(
            ticker="AAPL",
            estimate_date="2024-01-15",
            period_end="2024-03-31",
            eps_consensus=1.50,
            ts_event=1705276800000000000,  # 2024-01-15
            ts_init=1705276800000000000,
        )

        # First access (miss)
        estimate1 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=1705363200000000000,  # 2024-01-16
        )

        # Second access (hit)
        estimate2 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=1705363200000000000,
        )

        assert estimate1 == estimate2
        assert estimate1["eps_consensus"] == 1.50

        stats = cache.get_stats()
        assert stats["estimates_cache_info"]["hits"] == 1
        assert stats["estimates_cache_info"]["misses"] == 1


class TestCachePerformance:
    """Test cache performance requirements."""

    def test_lookup_latency_below_1ms(self) -> None:
        """Test P99 lookup latency is <1ms (hot path requirement)."""
        store = build_test_earnings_adapter()

        # Populate with test data
        for i in range(10):
            store.write_actuals(
                ticker="AAPL",
                period_end=f"2024-{i+1:02d}-01",
                filing_date=f"2024-{i+1:02d}-15",
                eps_diluted=1.0 + i * 0.1,
                revenue=100_000_000_000.0,
                ts_event=1704067200000000000 + i * 86400000000000,
                ts_init=1704067200000000000 + i * 86400000000000,
            )

        cache = EarningsCache(store, maxsize=128)

        # Measure latency for 1000 lookups
        latencies = []
        for i in range(1000):
            start = time.perf_counter_ns()
            cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000 + i * 1000000)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1_000_000)  # Convert to ms

        # Calculate P99
        import numpy as np

        p99_latency = np.percentile(latencies, 99)
        assert p99_latency < 1.0, f"P99 latency {p99_latency:.3f}ms exceeds 1ms requirement"

    def test_sequential_access_hit_rate(self) -> None:
        """Test cache hit rate >90% for sequential access pattern."""
        store = build_test_earnings_adapter()
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-09-30",
            filing_date="2024-11-01",
            eps_diluted=1.64,
            revenue=94_900_000_000.0,
            ts_event=1730419200000000000,
            ts_init=1730419200000000000,
        )

        cache = EarningsCache(store, maxsize=128)

        # Warm cache with exact timestamps we'll access
        start_ts = 1730419200000000000
        end_ts = 1733011200000000000  # 30 days later

        # Pre-warm with exact timestamps to maximize hit rate
        current_ts = start_ts
        for _ in range(30):
            cache.get_actuals_at("AAPL", as_of_ts=current_ts)
            current_ts += 86400000000000  # 1 day in nanoseconds

        # Get stats after warming to establish baseline
        warmup_stats = cache.get_stats()
        warmup_hits = warmup_stats["hits"]
        warmup_misses = warmup_stats["misses"]

        # Now access again (should all be hits)
        current_ts = start_ts
        for _ in range(30):
            cache.get_actuals_at("AAPL", as_of_ts=current_ts)
            current_ts += 86400000000000

        final_stats = cache.get_stats()

        # Calculate hit rate for second pass (after warming)
        new_hits = final_stats["hits"] - warmup_hits
        new_misses = final_stats["misses"] - warmup_misses
        total_new_ops = new_hits + new_misses

        hit_rate = new_hits / total_new_ops if total_new_ops > 0 else 0.0

        # Second pass should have 100% hit rate (all data pre-cached)
        assert hit_rate >= 0.9, f"Hit rate {hit_rate:.1%} below 90% requirement"


class TestCacheStatistics:
    """Test cache statistics reporting."""

    def test_get_stats_structure(self) -> None:
        """Test get_stats returns expected structure."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        stats = cache.get_stats()

        # Verify required fields
        assert "hit_rate" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "maxsize" in stats
        assert "actuals_cache_info" in stats
        assert "estimates_cache_info" in stats

        # Verify nested structure
        assert "hits" in stats["actuals_cache_info"]
        assert "misses" in stats["actuals_cache_info"]
        assert "currsize" in stats["actuals_cache_info"]

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate is calculated correctly."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # No operations yet
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

        # 1 miss
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

        # 1 hit
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total

        # 2 more hits
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)
        cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.75  # 3 hits / 4 total
