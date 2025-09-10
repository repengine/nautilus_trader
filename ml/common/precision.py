"""
Precision helpers for constructing nautilus Price/Quantity safely.

Nautilus enforces a maximum precision of 16 decimal places for Prices. This module
provides small helpers to clamp floating values to a safe number of decimals before
constructing string representations.

"""

from __future__ import annotations

from typing import Final


MAX_PRICE_DECIMALS: Final[int] = 16


def clamp_price_str(value: float, decimals: int = 9) -> str:
    """
    Clamp a float to a safe decimal precision for Price.from_str.

    Parameters
    ----------
    value : float
        The price value.
    decimals : int, default 9
        Number of decimal places to include (must be <= MAX_PRICE_DECIMALS).

    Returns
    -------
    str
        A string representation with at most `decimals` decimals.

    """
    if decimals > MAX_PRICE_DECIMALS:
        decimals = MAX_PRICE_DECIMALS
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)
