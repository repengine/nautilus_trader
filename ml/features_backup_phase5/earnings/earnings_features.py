"""
Earnings Feature Computation - Corporate Fundamentals Integration.

Provides earnings surprise, growth, momentum, and calendar features with guaranteed
parity between incremental (hot path) and batch (cold path) implementations.

Performance Targets:
- Hot path: P99 < 5ms, O(1) complexity, zero allocations
- Cold path: Vectorized batch computation
- Parity: Validated to rtol=1e-10

Data Sources:
- Actuals: SEC EDGAR 10-Q/10-K filings (via edgartools)
- Estimates: Yahoo Finance consensus (via yfinance)
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np


if TYPE_CHECKING:
    from datetime import datetime

# ===== Module metrics (idempotent) =====
_metrics_init = False
_surprise_updates_total = None
_surprise_latency_seconds = None
_growth_updates_total = None
_growth_latency_seconds = None
_momentum_updates_total = None
_momentum_latency_seconds = None
_env_flag_raw: str | None = None
_env_flag_enabled = False


def _init_module_metrics() -> None:
    """Initialize module-level metrics once (idempotent)."""
    global _metrics_init
    global _surprise_updates_total, _surprise_latency_seconds
    global _growth_updates_total, _growth_latency_seconds
    global _momentum_updates_total, _momentum_latency_seconds

    if _metrics_init:
        return

    from ml.common.metrics_manager import MetricsManager

    mm = MetricsManager.default()

    _surprise_updates_total = mm.counter(
        "ml_earnings_surprise_updates_total",
        "Total earnings surprise updates performed",
        ["path_type"],
    )
    _surprise_latency_seconds = mm.histogram(
        "ml_earnings_surprise_latency_seconds",
        "Earnings surprise computation latency",
        ["path_type"],
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    )

    _growth_updates_total = mm.counter(
        "ml_earnings_growth_updates_total",
        "Total earnings growth updates performed",
        ["path_type"],
    )
    _growth_latency_seconds = mm.histogram(
        "ml_earnings_growth_latency_seconds",
        "Earnings growth computation latency",
        ["path_type"],
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    )

    _momentum_updates_total = mm.counter(
        "ml_earnings_momentum_updates_total",
        "Total earnings momentum updates performed",
        ["path_type"],
    )
    _momentum_latency_seconds = mm.histogram(
        "ml_earnings_momentum_latency_seconds",
        "Earnings momentum computation latency",
        ["path_type"],
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    )

    _metrics_init = True


def _default_earnings_metrics_enabled() -> bool:
    global _env_flag_raw, _env_flag_enabled
    raw = os.getenv("ML_EARNINGS_ENABLE_METRICS", "0")
    if raw != _env_flag_raw:
        _env_flag_raw = raw
        _env_flag_enabled = raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return _env_flag_enabled


def earnings_metrics_enabled(*, env: Mapping[str, str] | None = None) -> bool:
    """
    Determine whether earnings metrics are enabled.

    Uses a cached lookup for process-wide defaults while still permitting explicit
    mappings during tests.
    """
    if env is None:
        return _default_earnings_metrics_enabled()
    source = env
    raw = source.get("ML_EARNINGS_ENABLE_METRICS", "0")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _metrics_enabled() -> bool:
    """
    Return True when metrics instrumentation is enabled and ensure initialization.
    """
    if earnings_metrics_enabled():
        _init_module_metrics()
        return True
    return False


EPS_EPSILON = 1e-12


def reset_earnings_metrics_state() -> None:
    """Reset cached metrics state to keep tests deterministic."""
    global _env_flag_raw, _env_flag_enabled
    global _metrics_init
    global _surprise_updates_total, _surprise_latency_seconds
    global _growth_updates_total, _growth_latency_seconds
    global _momentum_updates_total, _momentum_latency_seconds

    _env_flag_raw = None
    _env_flag_enabled = False

    _metrics_init = False
    _surprise_updates_total = None
    _surprise_latency_seconds = None
    _growth_updates_total = None
    _growth_latency_seconds = None
    _momentum_updates_total = None
    _momentum_latency_seconds = None


def _assign_hot_value(
    container: MutableMapping[str, Any],
    key: str,
    value: float,
) -> None:
    """Assign to mapping key, updating numpy buffers in-place when provided."""
    target = container.get(key)
    if isinstance(target, np.ndarray):
        target[...] = value
        return

    container[key] = value

# ===== Public API =====
__all__ = [
    "compute_calendar_features_batch",
    "compute_calendar_features_incremental",
    "compute_earnings_growth_batch",
    "compute_earnings_growth_incremental",
    "compute_earnings_momentum_batch",
    "compute_earnings_momentum_incremental",
    "compute_earnings_surprise_batch",
    "compute_earnings_surprise_incremental",
    "earnings_metrics_enabled",
    "reset_earnings_metrics_state",
]


# ===== Earnings Surprise Features =====


def compute_earnings_surprise_incremental(
    actual: float,
    estimate: float,
    *,
    out: MutableMapping[str, float] | None = None,
) -> MutableMapping[str, float]:
    """
    Compute earnings surprise incrementally (hot path - O(1)).

    Calculates dollar surprise and percentage surprise from actual vs consensus.
    Designed for real-time trading with zero allocations.

    Parameters
    ----------
    actual : float
        Actual reported EPS from SEC EDGAR 10-Q filing
    estimate : float
        Consensus EPS estimate from Yahoo Finance
    out : MutableMapping[str, float] | None, optional
        Optional pre-allocated mapping to populate with results for zero-alloc hot paths.
        Provide 1-length ``numpy`` arrays (e.g., ``np.zeros(1, dtype=np.float64)``)
        to update buffers in-place without new allocations.

    Returns
    -------
    MutableMapping[str, float]
        Mapping with keys:
        - 'eps_surprise_q0': Dollar surprise (actual - estimate)
        - 'eps_surprise_pct_q0': Percentage surprise ((actual - estimate) / estimate * 100)

    Notes
    -----
    - Returns 0.0 for percentage surprise if estimate is zero
    - Hot path: <1ms P99, zero allocations
    - Division by zero protection with 1e-12 threshold
    - Provide ``out`` to reuse a caller-managed buffer and avoid allocations

    Examples
    --------
    >>> surprise = compute_earnings_surprise_incremental(2.52, 2.45)
    >>> surprise['eps_surprise_q0']
    0.07
    >>> surprise['eps_surprise_pct_q0']
    2.857142857142857
    """
    metrics_enabled = (
        _metrics_enabled()
        and _surprise_latency_seconds is not None
        and _surprise_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    result_any: MutableMapping[str, Any]
    if out is not None:
        result_any = cast(MutableMapping[str, Any], out)
    else:
        result_any = {
            "eps_surprise_q0": 0.0,
            "eps_surprise_pct_q0": 0.0,
        }

    # Handle None values
    if actual is None or estimate is None:
        _assign_hot_value(result_any, "eps_surprise_q0", 0.0)
        _assign_hot_value(result_any, "eps_surprise_pct_q0", 0.0)
    else:
        # Calculate dollar surprise
        eps_surprise = actual - estimate
        _assign_hot_value(result_any, "eps_surprise_q0", eps_surprise)

        # Calculate percentage surprise (handle division by zero)
        if abs(estimate) > EPS_EPSILON:
            _assign_hot_value(result_any, "eps_surprise_pct_q0", (eps_surprise / estimate) * 100.0)
        else:
            _assign_hot_value(result_any, "eps_surprise_pct_q0", 0.0)

    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _surprise_latency_seconds)
        updates_counter = cast(Any, _surprise_updates_total)
        latency_hist.labels(path_type="incremental").observe(elapsed)
        updates_counter.labels(path_type="incremental").inc()

    return cast(MutableMapping[str, float], result_any)


def compute_earnings_surprise_batch(
    actuals: np.ndarray,
    estimates: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute earnings surprise in batch (cold path - vectorized).

    Computes dollar and percentage surprises for arrays of actuals and estimates.
    Suitable for backtesting, model training, and offline analysis.

    Parameters
    ----------
    actuals : np.ndarray
        Array of actual reported EPS values, shape (n,)
    estimates : np.ndarray
        Array of consensus EPS estimates, shape (n,)

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'eps_surprise_q0': Dollar surprises, shape (n,)
        - 'eps_surprise_pct_q0': Percentage surprises, shape (n,)

    Notes
    -----
    - Percentage surprise is 0.0 where estimate is near zero (< 1e-12)
    - Uses same formula as incremental version for parity
    - Cold path: No performance constraints, vectorized for speed
    - Validates parity with incremental computation to rtol=1e-10

    Raises
    ------
    ValueError
        If arrays have different lengths

    Examples
    --------
    >>> import numpy as np
    >>> actuals = np.array([2.52, 2.45, 2.38, 2.30])
    >>> estimates = np.array([2.45, 2.40, 2.35, 2.28])
    >>> surprises = compute_earnings_surprise_batch(actuals, estimates)
    >>> surprises['eps_surprise_q0']
    array([0.07, 0.05, 0.03, 0.02])
    """
    metrics_enabled = (
        _metrics_enabled()
        and _surprise_latency_seconds is not None
        and _surprise_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    # Validation
    if actuals.shape != estimates.shape:
        msg = (
            f"Shape mismatch: actuals {actuals.shape} != estimates {estimates.shape}"
        )
        raise ValueError(msg)

    n = len(actuals)
    if n == 0:
        return {
            "eps_surprise_q0": np.array([]),
            "eps_surprise_pct_q0": np.array([]),
        }

    # Calculate dollar surprise (vectorized)
    eps_surprise = actuals - estimates

    # Calculate percentage surprise (vectorized with division by zero protection)
    eps_surprise_pct = np.zeros(n, dtype=np.float64)
    valid_estimates = np.abs(estimates) > EPS_EPSILON
    eps_surprise_pct[valid_estimates] = (
        eps_surprise[valid_estimates] / estimates[valid_estimates]
    ) * 100.0

    # Record metrics
    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _surprise_latency_seconds)
        updates_counter = cast(Any, _surprise_updates_total)
        latency_hist.labels(path_type="batch").observe(elapsed)
        updates_counter.labels(path_type="batch").inc()

    return {
        "eps_surprise_q0": eps_surprise,
        "eps_surprise_pct_q0": eps_surprise_pct,
    }


# ===== Earnings Growth Features =====


def compute_earnings_growth_incremental(
    eps_history: list[float],
    *,
    out: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute earnings growth incrementally (hot path - O(1)).

    Calculates year-over-year (YoY) and quarter-over-quarter (QoQ) EPS growth.
    Designed for real-time trading with zero allocations.

    Parameters
    ----------
    eps_history : list[float]
        EPS for last 5 quarters in order [Q0, Q-1, Q-2, Q-3, Q-4]
        where Q0 is most recent quarter
    out : dict[str, float] | None, optional
        Optional pre-allocated mapping to populate for zero-allocation hot paths.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - 'eps_growth_yoy': Year-over-year growth percentage
        - 'eps_growth_qoq': Quarter-over-quarter growth percentage

    Notes
    -----
    - Returns 0.0 if denominator is zero
    - Hot path: <1ms P99, zero allocations
    - Requires at least 5 quarters of history
    - Division by zero protection with 1e-12 threshold
    - Provide ``out`` to reuse caller buffers without allocating

    Examples
    --------
    >>> eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]
    >>> growth = compute_earnings_growth_incremental(eps_history)
    >>> growth['eps_growth_yoy']
    14.545454545454545
    """
    metrics_enabled = (
        _metrics_enabled()
        and _growth_latency_seconds is not None
        and _growth_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    result = out if out is not None else {
        "eps_growth_yoy": 0.0,
        "eps_growth_qoq": 0.0,
    }

    if len(eps_history) >= 2:
        eps_q0 = eps_history[0]
        eps_q1 = eps_history[1]

        if abs(eps_q1) > EPS_EPSILON:
            result["eps_growth_qoq"] = ((eps_q0 - eps_q1) / eps_q1) * 100.0
        else:
            result["eps_growth_qoq"] = 0.0

        if len(eps_history) >= 5:
            eps_q4 = eps_history[4]
            if abs(eps_q4) > EPS_EPSILON:
                result["eps_growth_yoy"] = ((eps_q0 - eps_q4) / eps_q4) * 100.0
            else:
                result["eps_growth_yoy"] = 0.0
        else:
            result["eps_growth_yoy"] = 0.0
    else:
        result["eps_growth_yoy"] = 0.0
        result["eps_growth_qoq"] = 0.0

    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _growth_latency_seconds)
        updates_counter = cast(Any, _growth_updates_total)
        latency_hist.labels(path_type="incremental").observe(elapsed)
        updates_counter.labels(path_type="incremental").inc()

    return result


def compute_earnings_growth_batch(
    eps_series: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute earnings growth in batch (cold path - vectorized).

    Computes YoY and QoQ growth rates for a time series of EPS values.
    Suitable for backtesting, model training, and offline analysis.

    Parameters
    ----------
    eps_series : np.ndarray
        Time series of EPS values, shape (n,), ordered chronologically

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'eps_growth_yoy': Year-over-year growth percentages, shape (n,)
        - 'eps_growth_qoq': Quarter-over-quarter growth percentages, shape (n,)

    Notes
    -----
    - First 4 values for YoY will be 0.0 (insufficient history)
    - First value for QoQ will be 0.0 (no prior quarter)
    - Uses same formula as incremental version for parity
    - Cold path: No performance constraints, vectorized for speed
    - Validates parity with incremental computation to rtol=1e-10

    Examples
    --------
    >>> import numpy as np
    >>> eps_series = np.array([2.20, 2.30, 2.38, 2.45, 2.52])
    >>> growth = compute_earnings_growth_batch(eps_series)
    >>> growth['eps_growth_yoy'][-1]
    14.545454545454545
    """
    metrics_enabled = (
        _metrics_enabled()
        and _growth_latency_seconds is not None
        and _growth_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    n = len(eps_series)
    if n == 0:
        return {
            "eps_growth_yoy": np.array([]),
            "eps_growth_qoq": np.array([]),
        }

    # Pre-allocate output arrays
    eps_growth_yoy = np.zeros(n, dtype=np.float64)
    eps_growth_qoq = np.zeros(n, dtype=np.float64)

    # Calculate YoY growth (need 4 quarters of history)
    for i in range(4, n):
        eps_q0 = eps_series[i]
        eps_q4 = eps_series[i - 4]

        if abs(eps_q4) > EPS_EPSILON:
            eps_growth_yoy[i] = ((eps_q0 - eps_q4) / eps_q4) * 100.0

    # Calculate QoQ growth (need 1 quarter of history)
    for i in range(1, n):
        eps_q0 = eps_series[i]
        eps_q1 = eps_series[i - 1]

        if abs(eps_q1) > EPS_EPSILON:
            eps_growth_qoq[i] = ((eps_q0 - eps_q1) / eps_q1) * 100.0

    # Record metrics
    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _growth_latency_seconds)
        updates_counter = cast(Any, _growth_updates_total)
        latency_hist.labels(path_type="batch").observe(elapsed)
        updates_counter.labels(path_type="batch").inc()

    return {
        "eps_growth_yoy": eps_growth_yoy,
        "eps_growth_qoq": eps_growth_qoq,
    }


# ===== Earnings Momentum Features =====


def compute_earnings_momentum_incremental(
    surprises: list[float],
    eps_history: list[float],
    *,
    out: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute earnings momentum incrementally (hot path - O(1)).

    Calculates consecutive beat streak and EPS volatility.
    Designed for real-time trading with zero allocations.

    Parameters
    ----------
    surprises : list[float]
        Earnings surprises (dollar amount) for recent quarters
        Ordered most recent first [Q0, Q-1, Q-2, ...]
    eps_history : list[float]
        EPS values for last 4 quarters [Q0, Q-1, Q-2, Q-3]
    out : dict[str, float] | None, optional
        Optional pre-allocated mapping to populate for zero-allocation hot paths.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - 'earnings_beat_streak': Number of consecutive positive surprises
        - 'eps_volatility_4q': Coefficient of variation (std/mean) for last 4 quarters

    Notes
    -----
    - Beat streak counts from most recent quarter backwards
    - Volatility is coefficient of variation (handles scale differences)
    - Returns 0.0 for volatility if mean EPS is near zero
    - Hot path: <1ms P99, zero allocations when ``out`` provided

    Examples
    --------
    >>> surprises = [0.07, 0.05, 0.03, -0.02]
    >>> eps_history = [2.52, 2.45, 2.38, 2.30]
    >>> momentum = compute_earnings_momentum_incremental(surprises, eps_history)
    >>> momentum['earnings_beat_streak']
    3
    """
    metrics_enabled = (
        _metrics_enabled()
        and _momentum_latency_seconds is not None
        and _momentum_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    result = out if out is not None else {
        "earnings_beat_streak": 0.0,
        "eps_volatility_4q": 0.0,
    }

    # Calculate beat streak (consecutive positive surprises)
    beat_streak = 0
    for surprise in surprises:
        if surprise is None or surprise <= 0:
            break
        beat_streak += 1

    result["earnings_beat_streak"] = float(beat_streak)

    # Calculate EPS volatility (coefficient of variation)
    eps_volatility = 0.0
    if len(eps_history) >= 4:
        first_four = eps_history[:4]
        try:
            eps0 = float(first_four[0])
            eps1 = float(first_four[1])
            eps2 = float(first_four[2])
            eps3 = float(first_four[3])
        except (TypeError, ValueError, IndexError):
            eps_mean = 0.0
        else:
            eps_mean = (eps0 + eps1 + eps2 + eps3) / 4.0
            if abs(eps_mean) > EPS_EPSILON:
                diff0 = eps0 - eps_mean
                diff1 = eps1 - eps_mean
                diff2 = eps2 - eps_mean
                diff3 = eps3 - eps_mean
                variance = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3) / 3.0
                eps_std = math.sqrt(variance)
                eps_volatility = eps_std / eps_mean

    result["eps_volatility_4q"] = eps_volatility

    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _momentum_latency_seconds)
        updates_counter = cast(Any, _momentum_updates_total)
        latency_hist.labels(path_type="incremental").observe(elapsed)
        updates_counter.labels(path_type="incremental").inc()

    return result


def compute_earnings_momentum_batch(
    surprises_series: np.ndarray,
    eps_series: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute earnings momentum in batch (cold path - vectorized).

    Computes beat streaks and EPS volatility for time series data.
    Suitable for backtesting, model training, and offline analysis.

    Parameters
    ----------
    surprises_series : np.ndarray
        Time series of earnings surprises (dollar), shape (n,), chronological order
    eps_series : np.ndarray
        Time series of EPS values, shape (n,), chronological order

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'earnings_beat_streak': Beat streak counts, shape (n,)
        - 'eps_volatility_4q': 4-quarter volatility, shape (n,)

    Notes
    -----
    - Beat streak is calculated as of each point in time
    - Volatility requires 4 quarters of history (first 3 values are 0.0)
    - Uses same formula as incremental version for parity
    - Cold path: No performance constraints
    - Validates parity with incremental computation to rtol=1e-10

    Examples
    --------
    >>> import numpy as np
    >>> surprises = np.array([0.07, 0.05, 0.03, -0.02, 0.01])
    >>> eps_series = np.array([2.52, 2.45, 2.38, 2.30, 2.20])
    >>> momentum = compute_earnings_momentum_batch(surprises, eps_series)
    """
    metrics_enabled = (
        _metrics_enabled()
        and _momentum_latency_seconds is not None
        and _momentum_updates_total is not None
    )
    start: float | None = time.perf_counter() if metrics_enabled else None

    # Validation
    if surprises_series.shape != eps_series.shape:
        msg = f"Shape mismatch: surprises {surprises_series.shape} != eps {eps_series.shape}"
        raise ValueError(msg)

    n = len(surprises_series)
    if n == 0:
        return {
            "earnings_beat_streak": np.array([]),
            "eps_volatility_4q": np.array([]),
        }

    # Pre-allocate output arrays
    earnings_beat_streak = np.zeros(n, dtype=np.float64)
    eps_volatility_4q = np.zeros(n, dtype=np.float64)

    # Calculate beat streak for each point in time
    for i in range(n):
        streak = 0
        # Count backwards from current position
        for j in range(i, -1, -1):
            if surprises_series[j] > 0:
                streak += 1
            else:
                break
        earnings_beat_streak[i] = float(streak)

    # Calculate rolling 4-quarter volatility
    for i in range(3, n):
        window = eps_series[i - 3 : i + 1]  # Last 4 quarters
        eps_mean = np.mean(window)
        eps_std = np.std(window, ddof=1)

        if abs(eps_mean) > 1e-12:
            eps_volatility_4q[i] = eps_std / eps_mean

    # Record metrics
    if metrics_enabled and start is not None:
        elapsed = time.perf_counter() - start
        latency_hist = cast(Any, _momentum_latency_seconds)
        updates_counter = cast(Any, _momentum_updates_total)
        latency_hist.labels(path_type="batch").observe(elapsed)
        updates_counter.labels(path_type="batch").inc()

    return {
        "earnings_beat_streak": earnings_beat_streak,
        "eps_volatility_4q": eps_volatility_4q,
    }


# ===== Earnings Calendar Features =====


def compute_calendar_features_incremental(
    next_earnings_date: datetime,
    current_date: datetime,
    *,
    out: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Compute earnings calendar features incrementally (hot path - O(1)).

    Calculates days until next earnings announcement.
    Designed for real-time trading with zero allocations.

    Parameters
    ----------
    next_earnings_date : datetime
        Scheduled next earnings announcement date
    current_date : datetime
        Current date for calculation
    out : dict[str, int] | None, optional
        Optional pre-allocated mapping to populate for zero-allocation hot paths.

    Returns
    -------
    dict[str, int]
        Dictionary with keys:
        - 'days_to_next_earnings': Calendar days until next earnings (can be negative if past due)

    Notes
    -----
    - Returns negative value if earnings date has passed
    - Hot path: <1ms P99, zero allocations when ``out`` provided
    - Simple date arithmetic (no timezone conversions)

    Examples
    --------
    >>> from datetime import datetime
    >>> next_earnings = datetime(2024, 1, 30)
    >>> current = datetime(2024, 1, 1)
    >>> calendar = compute_calendar_features_incremental(next_earnings, current)
    >>> calendar['days_to_next_earnings']
    29
    """
    # Simple date difference
    delta = next_earnings_date - current_date
    days_to_earnings = delta.days

    result = out if out is not None else {"days_to_next_earnings": 0}
    result["days_to_next_earnings"] = days_to_earnings
    return result


def compute_calendar_features_batch(
    next_earnings_dates: np.ndarray,
    current_dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute earnings calendar features in batch (cold path - vectorized).

    Computes days until next earnings for arrays of dates.
    Suitable for backtesting, model training, and offline analysis.

    Parameters
    ----------
    next_earnings_dates : np.ndarray
        Array of next earnings dates, shape (n,), dtype datetime64
    current_dates : np.ndarray
        Array of current dates, shape (n,), dtype datetime64

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'days_to_next_earnings': Days to next earnings, shape (n,), dtype int64

    Notes
    -----
    - Uses vectorized date arithmetic
    - Same result as incremental version (exact parity)
    - Cold path: No performance constraints

    Raises
    ------
    ValueError
        If arrays have different lengths

    Examples
    --------
    >>> import numpy as np
    >>> next_earnings = np.array(['2024-01-30', '2024-02-15'], dtype='datetime64[D]')
    >>> current = np.array(['2024-01-01', '2024-01-20'], dtype='datetime64[D]')
    >>> calendar = compute_calendar_features_batch(next_earnings, current)
    >>> calendar['days_to_next_earnings']
    array([29, 26])
    """
    # Validation
    if next_earnings_dates.shape != current_dates.shape:
        msg = (
            f"Shape mismatch: next_earnings_dates {next_earnings_dates.shape} != "
            f"current_dates {current_dates.shape}"
        )
        raise ValueError(msg)

    # Vectorized date arithmetic
    # Convert to timedelta64 then extract days
    delta = next_earnings_dates - current_dates

    # Convert timedelta64 to days (integer)
    days_to_earnings = delta.astype("timedelta64[D]").astype(np.int64)

    return {
        "days_to_next_earnings": days_to_earnings,
    }
