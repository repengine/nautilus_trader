"""
Shared validation strategy helpers for training and evaluation splits.
"""

from __future__ import annotations

from typing import Final, Literal


CVStrategy = Literal["time_series", "purged"]
HoldoutStrategy = Literal["time_window", "purged"]

CV_STRATEGIES: Final[tuple[str, ...]] = ("time_series", "purged")
HOLDOUT_STRATEGIES: Final[tuple[str, ...]] = ("time_window", "purged")

DEFAULT_CV_STRATEGY: Final[str] = "time_series"
DEFAULT_HOLDOUT_STRATEGY: Final[str] = "purged"


def normalize_strategy(value: str) -> str:
    """
    Normalize a strategy string for comparison.

    Parameters
    ----------
    value : str
        Strategy string to normalize.

    Returns
    -------
    str
        Normalized, lowercase strategy.
    """
    return value.strip().lower()


def require_cv_strategy(value: str) -> str:
    """
    Validate cross-validation strategy values.

    Parameters
    ----------
    value : str
        Strategy string to validate.

    Returns
    -------
    str
        Normalized strategy value.

    Raises
    ------
    ValueError
        If the strategy is not supported.
    """
    normalized = normalize_strategy(value)
    if normalized not in CV_STRATEGIES:
        raise ValueError(
            f"cv_strategy must be one of {CV_STRATEGIES}, got {value!r}",
        )
    return normalized


def require_holdout_strategy(value: str) -> str:
    """
    Validate holdout validation strategy values.

    Parameters
    ----------
    value : str
        Strategy string to validate.

    Returns
    -------
    str
        Normalized strategy value.

    Raises
    ------
    ValueError
        If the strategy is not supported.
    """
    normalized = normalize_strategy(value)
    if normalized not in HOLDOUT_STRATEGIES:
        raise ValueError(
            f"validation_strategy must be one of {HOLDOUT_STRATEGIES}, got {value!r}",
        )
    return normalized


__all__ = [
    "CV_STRATEGIES",
    "DEFAULT_CV_STRATEGY",
    "DEFAULT_HOLDOUT_STRATEGY",
    "HOLDOUT_STRATEGIES",
    "CVStrategy",
    "HoldoutStrategy",
    "normalize_strategy",
    "require_cv_strategy",
    "require_holdout_strategy",
]
