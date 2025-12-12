"""
Post-processing utilities for feature parity.

This module provides shared post-processing functions used by both the
legacy FeatureEngineer and the new FeatureCalculator to ensure identical
feature values (training/serving parity).

Task 1.1c: Extracted from ml/features/engineering.py to avoid duplication.
"""

from __future__ import annotations

import math


def stable_rsi(prices: list[float], period: int) -> float:
    """
    Compute Wilder's RSI in double precision for the last value.

    Uses fractional returns to improve scale invariance under rounded inputs.
    This is the canonical RSI calculation for parity between legacy and facade.

    Parameters
    ----------
    prices : list[float]
        Closing prices in chronological order.
    period : int
        RSI period (e.g., 14).

    Returns
    -------
    float
        RSI in [0, 100].

    Examples
    --------
    >>> prices = [100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 105.0,
    ...           104.5, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0]
    >>> rsi = stable_rsi(prices, period=14)
    >>> 0.0 <= rsi <= 100.0
    True

    """
    n = len(prices)
    if n <= period:
        return 50.0

    # Use fractional returns to improve scale invariance under rounded inputs
    gains: list[float] = []
    losses: list[float] = []

    for i in range(1, n):
        prev = float(prices[i - 1])
        curr = float(prices[i])

        if math.isclose(prev, 0.0, abs_tol=1e-20):
            ret = 0.0
        else:
            ret = (curr - prev) / prev

        gains.append(max(ret, 0.0))
        losses.append(max(-ret, 0.0))

    # Initial averages
    avg_gain = sum(gains[:period]) / float(period)
    avg_loss = sum(losses[:period]) / float(period)

    # Wilder smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / float(period)
        avg_loss = (avg_loss * (period - 1) + losses[i]) / float(period)

    if math.isclose(avg_loss, 0.0, abs_tol=1e-20):
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def normalize_rsi(rsi_value: float, precision: int = 8) -> float:
    """
    Normalize RSI from [0, 100] to [-1, 1] range with rounding.

    Parameters
    ----------
    rsi_value : float
        RSI value in [0, 100] range.
    precision : int
        Number of decimal places to round to (default 8).

    Returns
    -------
    float
        Normalized RSI in [-1, 1] range, rounded to specified precision.

    Examples
    --------
    >>> normalize_rsi(50.0)
    0.0
    >>> normalize_rsi(100.0)
    1.0
    >>> normalize_rsi(0.0)
    -1.0

    """
    # Map [0, 100] -> [-1, 1]: (rsi / 100 - 0.5) * 2
    normalized = (rsi_value / 100.0 - 0.5) * 2.0
    return round(normalized, precision)


def round_hl_spread(hl_spread: float, precision: int = 6) -> float:
    """
    Round high-low spread for numerical stability.

    Parameters
    ----------
    hl_spread : float
        High-low spread value.
    precision : int
        Number of decimal places (default 6).

    Returns
    -------
    float
        Rounded hl_spread value.

    """
    return round(float(hl_spread), precision)


def apply_compute_features_post_processing(
    features: dict[str, float],
    closes: list[float] | None = None,
    rsi_period: int = 14,
) -> dict[str, float]:
    """
    Apply post-processing to computed features for parity with legacy.

    This function applies:
    1. RSI recalculation using stable_rsi() and normalization to [-1, 1]
    2. hl_spread rounding for numerical stability

    Parameters
    ----------
    features : dict[str, float]
        Raw computed features dictionary.
    closes : list[float] | None
        Closing prices for RSI calculation. If None, RSI is not recalculated.
    rsi_period : int
        RSI period (default 14).

    Returns
    -------
    dict[str, float]
        Features with post-processing applied.

    Notes
    -----
    This function modifies the input dictionary in-place and returns it.
    The RSI recalculation is only applied if closes is provided and has
    enough data points (>= rsi_period + 1).

    """
    # Apply RSI recalculation if closes provided
    if closes is not None and len(closes) >= rsi_period + 1:
        try:
            rsi_val = stable_rsi(closes, rsi_period)
            features["rsi"] = normalize_rsi(rsi_val)
        except Exception:
            # Fall back silently; this is a compatibility helper
            pass

    # Apply hl_spread rounding for numerical stability
    if "hl_spread" in features:
        features["hl_spread"] = round_hl_spread(features["hl_spread"])

    return features
