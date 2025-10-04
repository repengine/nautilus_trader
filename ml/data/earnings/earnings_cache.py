"""
Earnings cache for point-in-time backtesting correctness.

This module provides an LRU-cached wrapper around EarningsStore to ensure:
1. No look-ahead bias: At time T, only data with ts_event < T is visible
2. High cache hit rate (>90%) for sequential access patterns
3. <1ms P99 latency per lookup

Key principles:
- Immutable cache keys (ticker, as_of_ts) for LRU compatibility
- All timestamps in nanoseconds (ts_event for point-in-time correctness)
- Cache warming for backtest windows
- Invalidation when new data arrives

Performance targets: <1ms per lookup (P99), >90% cache hit rate for sequential access
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# ===== Metrics =====
_cache_hits = get_counter("ml_earnings_cache_hits_total", "Total earnings cache hits")
_cache_misses = get_counter("ml_earnings_cache_misses_total", "Total earnings cache misses")
_cache_latency = get_histogram(
    "ml_earnings_cache_latency_seconds",
    "Earnings cache lookup latency",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
)


# ===== Store Protocol =====


class EarningsStoreProtocol(Protocol):
    """
    Protocol for earnings store implementations.

    Defines minimal interface for EarningsCache to work with any conforming store.
    Both EarningsStore and DummyEarningsStore satisfy this protocol.
    """

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get actual earnings with point-in-time filtering."""
        ...

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        """Get consensus estimate for a specific period."""
        ...

    def flush(self) -> None:
        """Flush any pending writes."""
        ...


# ===== Cache Implementation =====


class EarningsCache:
    """
    Point-in-time earnings cache with LRU eviction.

    Wraps an EarningsStore to provide high-performance cached access with temporal
    correctness guarantees. Ensures backtests see only data available at time T.

    Parameters
    ----------
    earnings_store : EarningsStoreProtocol
        Store implementation (EarningsStore or DummyEarningsStore)
    maxsize : int
        Maximum number of entries in LRU cache (default: 1024)

    Notes
    -----
    - Cache keys are immutable tuples (ticker, as_of_ts) for LRU compatibility
    - Returns tuples instead of lists for hashability
    - Cache invalidation via clear() when new data arrives
    - Metrics track hit/miss rates for monitoring

    Examples
    --------
    >>> from ml.stores import EarningsStore
    >>> store = EarningsStore("postgresql://...")
    >>> cache = EarningsCache(store, maxsize=1024)
    >>> actuals = cache.get_actuals_at("AAPL", as_of_ts=1704067200000000000)  # 2024-01-01
    >>> stats = cache.get_stats()
    >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    Hit rate: 92.3%
    """

    def __init__(
        self,
        earnings_store: EarningsStoreProtocol,
        maxsize: int = 1024,
    ) -> None:
        self._store = earnings_store
        self._maxsize = maxsize

        # Create LRU-cached versions of store methods
        # Cache returns immutable tuples for hashability
        self._cached_get_actuals = lru_cache(maxsize=maxsize)(self._get_actuals_impl)
        self._cached_get_estimates = lru_cache(maxsize=maxsize)(self._get_estimates_impl)

        # Track statistics
        self._total_requests = 0
        self._cache_hits = 0

        logger.info(
            "Initialized EarningsCache",
            extra={"maxsize": maxsize, "store_type": type(earnings_store).__name__},
        )

    def _get_actuals_impl(
        self,
        ticker: str,
        as_of_ts: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """
        Internal implementation that returns immutable tuple for LRU cache.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        as_of_ts : int
            Point-in-time timestamp in nanoseconds (only data with ts_event < as_of_ts)
        start_date : str | None
            Start date filter (ISO format)
        end_date : str | None
            End date filter (ISO format)

        Returns
        -------
        tuple[dict[str, Any], ...]
            Immutable tuple of actual earnings records
        """
        actuals = self._store.get_actuals(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )
        # Convert list to tuple for immutability (required for LRU cache)
        return tuple(actuals)

    def _get_estimates_impl(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int,
    ) -> dict[str, Any] | None:
        """
        Internal implementation for estimates lookup.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format)
        as_of_ts : int
            Point-in-time timestamp in nanoseconds

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found
        """
        return self._store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        )

    def get_actuals_at(
        self,
        ticker: str,
        as_of_ts: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get actual earnings at a specific point in time (cached).

        This is the primary API for temporal-correct lookups. At time T, only
        returns data where ts_event < T (prevents look-ahead bias).

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        as_of_ts : int
            Point-in-time timestamp in nanoseconds
        start_date : str | None
            Start date filter (ISO format, optional)
        end_date : str | None
            End date filter (ISO format, optional)

        Returns
        -------
        list[dict[str, Any]]
            List of actual earnings records visible at as_of_ts

        Notes
        -----
        - Cache key is (ticker, as_of_ts, start_date, end_date)
        - Returns mutable list for API compatibility (converted from cached tuple)
        - Updates metrics for hit/miss tracking

        Examples
        --------
        >>> # Backtest at 2024-01-01
        >>> as_of_ts = 1704067200000000000  # 2024-01-01 00:00:00 UTC in nanoseconds
        >>> actuals = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts)
        >>> # Only shows Q4 2023 and earlier (Q1 2024 filed after 2024-01-01)
        """
        start = time.perf_counter()
        self._total_requests += 1

        # Check cache info before call
        cache_info_before = self._cached_get_actuals.cache_info()

        # Call cached implementation
        result_tuple = self._cached_get_actuals(
            ticker=ticker,
            as_of_ts=as_of_ts,
            start_date=start_date,
            end_date=end_date,
        )

        # Check cache info after call
        cache_info_after = self._cached_get_actuals.cache_info()

        # Detect cache hit/miss
        if cache_info_after.hits > cache_info_before.hits:
            self._cache_hits += 1
            _cache_hits.inc()
        else:
            _cache_misses.inc()

        # Record latency
        elapsed = time.perf_counter() - start
        _cache_latency.observe(elapsed)

        # Convert tuple back to list for API compatibility
        return list(result_tuple)

    def get_estimates_at(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int,
    ) -> dict[str, Any] | None:
        """
        Get consensus estimate at a specific point in time (cached).

        Returns the most recent estimate available before as_of_ts.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format: 'YYYY-MM-DD')
        as_of_ts : int
            Point-in-time timestamp in nanoseconds

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found

        Notes
        -----
        - Cache key is (ticker, period_end, as_of_ts)
        - Returns most recent estimate with ts_event < as_of_ts

        Examples
        --------
        >>> # Get estimate for Q1 2024 as of 2024-01-01
        >>> estimate = cache.get_estimates_at(
        ...     "AAPL",
        ...     period_end="2024-03-31",
        ...     as_of_ts=1704067200000000000
        ... )
        """
        start = time.perf_counter()
        self._total_requests += 1

        # Check cache info before call
        cache_info_before = self._cached_get_estimates.cache_info()

        # Call cached implementation
        result = self._cached_get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        )

        # Check cache info after call
        cache_info_after = self._cached_get_estimates.cache_info()

        # Detect cache hit/miss
        if cache_info_after.hits > cache_info_before.hits:
            self._cache_hits += 1
            _cache_hits.inc()
        else:
            _cache_misses.inc()

        # Record latency
        elapsed = time.perf_counter() - start
        _cache_latency.observe(elapsed)

        return result

    def warm_cache(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        step_days: int = 7,
    ) -> None:
        """
        Warm cache for a backtest window.

        Pre-loads data for multiple timestamps to improve cache hit rate during
        sequential backtesting.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_ts : int
            Start timestamp in nanoseconds
        end_ts : int
            End timestamp in nanoseconds
        step_days : int
            Days between cache warming points (default: 7 for weekly)

        Notes
        -----
        - Samples timestamps at regular intervals across the window
        - Improves cache hit rate for sequential access
        - Use before running backtests

        Examples
        --------
        >>> # Warm cache for 2023 backtest
        >>> start_ts = 1672531200000000000  # 2023-01-01 00:00:00 UTC
        >>> end_ts = 1704067200000000000    # 2024-01-01 00:00:00 UTC
        >>> cache.warm_cache("AAPL", start_ts, end_ts, step_days=7)
        """
        logger.info(
            "Warming cache",
            extra={
                "ticker": ticker,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "step_days": step_days,
            },
        )

        # Calculate step in nanoseconds
        step_ns = step_days * 24 * 60 * 60 * 1_000_000_000

        # Sample timestamps
        current_ts = start_ts
        warm_count = 0

        while current_ts <= end_ts:
            # Warm actuals cache
            self.get_actuals_at(ticker=ticker, as_of_ts=current_ts)
            warm_count += 1
            current_ts += step_ns

        logger.info(
            "Cache warming complete",
            extra={"ticker": ticker, "timestamps_warmed": warm_count},
        )

    def invalidate(self, ticker: str | None = None) -> None:
        """
        Invalidate cache entries.

        Call when new data arrives to ensure cache consistency.

        Parameters
        ----------
        ticker : str | None
            Ticker to invalidate (if None, clears entire cache)

        Notes
        -----
        - Partial invalidation by ticker not supported by functools.lru_cache
        - Currently clears entire cache
        - Consider using cachetools for finer-grained invalidation if needed

        Examples
        --------
        >>> # Invalidate after loading new AAPL earnings data
        >>> cache.invalidate("AAPL")
        """
        # functools.lru_cache doesn't support partial invalidation
        # Clear entire cache (consider cachetools for selective invalidation)
        self._cached_get_actuals.cache_clear()
        self._cached_get_estimates.cache_clear()

        # Reset statistics
        self._total_requests = 0
        self._cache_hits = 0

        logger.info("Cache invalidated", extra={"ticker": ticker})

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict[str, Any]
            Statistics including:
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - hits: Total cache hits
            - misses: Total cache misses
            - size: Current cache size
            - maxsize: Maximum cache size
            - actuals_cache_info: LRU cache info for actuals
            - estimates_cache_info: LRU cache info for estimates

        Examples
        --------
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Cache size: {stats['size']}/{stats['maxsize']}")
        """
        actuals_info = self._cached_get_actuals.cache_info()
        estimates_info = self._cached_get_estimates.cache_info()

        total_hits = actuals_info.hits + estimates_info.hits
        total_misses = actuals_info.misses + estimates_info.misses
        total_requests = total_hits + total_misses

        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        return {
            "hit_rate": hit_rate,
            "hits": total_hits,
            "misses": total_misses,
            "size": actuals_info.currsize + estimates_info.currsize,
            "maxsize": self._maxsize,
            "actuals_cache_info": {
                "hits": actuals_info.hits,
                "misses": actuals_info.misses,
                "currsize": actuals_info.currsize,
            },
            "estimates_cache_info": {
                "hits": estimates_info.hits,
                "misses": estimates_info.misses,
                "currsize": estimates_info.currsize,
            },
        }


__all__ = [
    "EarningsCache",
    "EarningsStoreProtocol",
]
