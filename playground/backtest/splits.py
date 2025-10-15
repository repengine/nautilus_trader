"""
Train/test split utilities for out-of-sample backtesting validation.

This module provides utilities for creating temporal train/test splits that
prevent look-ahead bias in factor model backtesting. It supports both simple
static splits and walk-forward analysis for robust validation.

Key Features:
- Temporal train/test split with no look-ahead bias
- Walk-forward analysis with rolling windows
- Validation functions to detect data leakage
- Type-safe configuration dataclasses

Performance Targets (Cold Path):
- Split generation: < 100ms for 15 years of data
- Validation checks: < 50ms per split
- No performance-critical operations (offline analysis only)

Hot/Cold Path Separation:
- This is a cold-path module (backtesting is offline analysis)
- No real-time constraints, optimized for correctness over speed

Integration Notes:
- Compatible with FactorBacktester from engine.py
- All dates are timezone-aware (UTC)
- Splits are guaranteed to have no temporal overlap
- Follows Phase 3.2.1 requirements from 3D_Risk_Model_Roadmap.md

Examples
--------
Basic train/test split:

>>> split = define_train_test_split()
>>> print(f"Training: {split.train_start} to {split.train_end}")
Training: 2010-01-01 to 2018-12-31
>>> print(f"Testing: {split.test_start} to {split.test_end}")
Testing: 2019-01-01 to 2024-12-31

Walk-forward analysis:

>>> splits = walk_forward_splits(
...     start_date=datetime(2010, 1, 1),
...     end_date=datetime(2024, 12, 31),
...     train_years=5,
...     test_years=1,
... )
>>> for i, split in enumerate(splits, 1):
...     print(f"Fold {i}: Train {split.train_start.year}-{split.train_end.year}, Test {split.test_start.year}")
Fold 1: Train 2010-2014, Test 2015
Fold 2: Train 2011-2015, Test 2016
...

Validation:

>>> split = define_train_test_split()
>>> split.validate_no_overlap()  # Raises if test starts before training ends
>>> assert validate_no_lookahead(split)  # Check no future data leakage
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

import structlog


if TYPE_CHECKING:
    from collections.abc import Sequence
    from playground.backtest.regime_analysis import MarketRegime


LOGGER = structlog.get_logger(__name__)


# ===== Configuration Classes =====


@dataclass(slots=True, frozen=True)
class TrainTestSplit:
    """
    Configuration for a single train/test split with temporal separation.

    This class represents a single train/test split with explicit start and end
    dates for both periods. All dates are timezone-aware (UTC).

    The split is designed to prevent look-ahead bias:
    - Training period ends BEFORE test period begins
    - No temporal overlap between periods
    - Test period follows chronologically after training

    Attributes
    ----------
    train_start : datetime
        Start date of training period (inclusive), timezone-aware UTC
    train_end : datetime
        End date of training period (inclusive), timezone-aware UTC
    test_start : datetime
        Start date of testing period (inclusive), timezone-aware UTC
    test_end : datetime
        End date of testing period (inclusive), timezone-aware UTC

    Properties
    ----------
    train_years : float
        Number of years in training period
    test_years : float
        Number of years in testing period
    train_days : int
        Number of calendar days in training period
    test_days : int
        Number of calendar days in testing period
    gap_days : int
        Number of calendar days between training end and test start

    Methods
    -------
    validate_no_overlap()
        Validate that test period starts after training ends
    to_dict()
        Convert split configuration to dictionary

    Examples
    --------
    >>> split = TrainTestSplit(
    ...     train_start=datetime(2010, 1, 1, tzinfo=UTC),
    ...     train_end=datetime(2018, 12, 31, tzinfo=UTC),
    ...     test_start=datetime(2019, 1, 1, tzinfo=UTC),
    ...     test_end=datetime(2024, 12, 31, tzinfo=UTC),
    ... )
    >>> split.validate_no_overlap()
    >>> print(f"Training: {split.train_years:.1f} years")
    Training: 9.0 years
    >>> print(f"Testing: {split.test_years:.1f} years")
    Testing: 6.0 years

    Raises
    ------
    ValueError
        If dates are invalid (e.g., end before start, overlap, missing timezone)

    Notes
    -----
    - All datetime objects must be timezone-aware (preferably UTC)
    - Training period must end before test period begins
    - Both periods must have positive duration
    - Frozen dataclass ensures immutability (prevents accidental modification)
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        """Validate split configuration on initialization."""
        # Check timezone awareness
        if self.train_start.tzinfo is None:
            msg = "train_start must be timezone-aware"
            raise ValueError(msg)
        if self.train_end.tzinfo is None:
            msg = "train_end must be timezone-aware"
            raise ValueError(msg)
        if self.test_start.tzinfo is None:
            msg = "test_start must be timezone-aware"
            raise ValueError(msg)
        if self.test_end.tzinfo is None:
            msg = "test_end must be timezone-aware"
            raise ValueError(msg)

        # Check chronological order within each period
        if self.train_end <= self.train_start:
            msg = (
                f"Training end ({self.train_end}) must be after training start "
                f"({self.train_start})"
            )
            raise ValueError(msg)

        if self.test_end <= self.test_start:
            msg = (
                f"Test end ({self.test_end}) must be after test start "
                f"({self.test_start})"
            )
            raise ValueError(msg)

        # Check no overlap (test must start after training ends)
        if self.test_start <= self.train_end:
            msg = (
                f"Test start ({self.test_start}) must be after training end "
                f"({self.train_end}) to prevent look-ahead bias"
            )
            raise ValueError(msg)

    def validate_no_overlap(self) -> None:
        """
        Validate that test period starts strictly after training ends.

        This method provides an explicit validation that can be called
        programmatically to ensure no temporal overlap exists between
        training and testing periods.

        Raises
        ------
        ValueError
            If test period starts before or on training end date

        Examples
        --------
        >>> split = TrainTestSplit(...)
        >>> split.validate_no_overlap()  # Passes if valid
        """
        if self.test_start <= self.train_end:
            msg = (
                f"Look-ahead bias detected: test starts ({self.test_start}) "
                f"before training ends ({self.train_end})"
            )
            raise ValueError(msg)

    @property
    def train_years(self) -> float:
        """Calculate number of years in training period."""
        delta = self.train_end - self.train_start
        return delta.total_seconds() / (365.25 * 86400)

    @property
    def test_years(self) -> float:
        """Calculate number of years in testing period."""
        delta = self.test_end - self.test_start
        return delta.total_seconds() / (365.25 * 86400)

    @property
    def train_days(self) -> int:
        """Calculate number of calendar days in training period."""
        delta = self.train_end - self.train_start
        return delta.days + 1  # Inclusive of both start and end

    @property
    def test_days(self) -> int:
        """Calculate number of calendar days in testing period."""
        delta = self.test_end - self.test_start
        return delta.days + 1  # Inclusive of both start and end

    @property
    def gap_days(self) -> int:
        """Calculate number of calendar days between training and testing."""
        delta = self.test_start - self.train_end
        return delta.days

    def to_dict(self) -> dict[str, str]:
        """
        Convert split configuration to dictionary representation.

        Returns
        -------
        dict[str, str]
            Dictionary with ISO format date strings

        Examples
        --------
        >>> split = define_train_test_split()
        >>> config = split.to_dict()
        >>> print(config["train_start"])
        2010-01-01T00:00:00+00:00
        """
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


# ===== Primary Split Definition =====


def define_train_test_split(
    train_start: datetime | None = None,
    train_end: datetime | None = None,
    test_start: datetime | None = None,
    test_end: datetime | None = None,
) -> TrainTestSplit:
    """
    Define the primary train/test split for the 3D Factor Risk Model.

    This function creates the standard 8-year training / 6-year testing split
    as specified in Phase 3.2.1 of the 3D Risk Model Roadmap:
    - Training: 2010-01-01 to 2018-12-31 (8 years)
    - Testing: 2019-01-01 to 2024-12-31 (6 years)

    Parameters
    ----------
    train_start : datetime, optional
        Override default training start (2010-01-01)
    train_end : datetime, optional
        Override default training end (2018-12-31)
    test_start : datetime, optional
        Override default test start (2019-01-01)
    test_end : datetime, optional
        Override default test end (2024-12-31)

    Returns
    -------
    TrainTestSplit
        Validated train/test split configuration

    Raises
    ------
    ValueError
        If custom dates create invalid split (overlap, wrong order, etc.)

    Examples
    --------
    Default split:

    >>> split = define_train_test_split()
    >>> print(f"Train: {split.train_start.year}-{split.train_end.year}")
    Train: 2010-2018
    >>> print(f"Test: {split.test_start.year}-{split.test_end.year}")
    Test: 2019-2024

    Custom split:

    >>> split = define_train_test_split(
    ...     train_start=datetime(2012, 1, 1, tzinfo=UTC),
    ...     test_end=datetime(2023, 12, 31, tzinfo=UTC),
    ... )

    Notes
    -----
    Rationale for 8-year training period:
    - Captures multiple market regimes (2010-2018 includes recovery, growth, volatility)
    - Sufficient data for stable factor estimation (~2000 trading days)
    - Allows for rolling window beta estimation (252-day windows)

    Rationale for 6-year testing period:
    - Tests model through COVID-19 pandemic (2020)
    - Includes rate hike cycle (2022-2023)
    - Validates model in unseen regimes
    - Sufficient length for statistical significance
    """
    # Default dates from Phase 3.2.1 specification
    default_train_start = datetime(2010, 1, 1, tzinfo=UTC)
    default_train_end = datetime(2018, 12, 31, tzinfo=UTC)
    default_test_start = datetime(2019, 1, 1, tzinfo=UTC)
    default_test_end = datetime(2024, 12, 31, tzinfo=UTC)

    # Apply overrides with timezone handling
    final_train_start = train_start or default_train_start
    final_train_end = train_end or default_train_end
    final_test_start = test_start or default_test_start
    final_test_end = test_end or default_test_end

    # Ensure timezone awareness
    if final_train_start.tzinfo is None:
        final_train_start = final_train_start.replace(tzinfo=UTC)
    if final_train_end.tzinfo is None:
        final_train_end = final_train_end.replace(tzinfo=UTC)
    if final_test_start.tzinfo is None:
        final_test_start = final_test_start.replace(tzinfo=UTC)
    if final_test_end.tzinfo is None:
        final_test_end = final_test_end.replace(tzinfo=UTC)

    split = TrainTestSplit(
        train_start=final_train_start,
        train_end=final_train_end,
        test_start=final_test_start,
        test_end=final_test_end,
    )

    LOGGER.info(
        "Defined train/test split",
        train_period=f"{split.train_start.date()} to {split.train_end.date()}",
        test_period=f"{split.test_start.date()} to {split.test_end.date()}",
        train_years=f"{split.train_years:.1f}",
        test_years=f"{split.test_years:.1f}",
    )

    return split


# ===== Walk-Forward Analysis =====


def walk_forward_splits(
    start_date: datetime,
    end_date: datetime,
    train_years: int = 5,
    test_years: int = 1,
    *,
    step_years: int = 1,
) -> list[TrainTestSplit]:
    """
    Generate walk-forward analysis splits with rolling windows.

    Walk-forward analysis is a robust validation technique that simulates
    real-world model deployment by progressively rolling the training and
    testing windows forward through time.

    Algorithm:
    1. Start with initial training window (train_years)
    2. Test on next test_years period
    3. Roll forward by step_years
    4. Repeat until end_date is reached

    Parameters
    ----------
    start_date : datetime
        Earliest possible training start date
    end_date : datetime
        Latest possible testing end date
    train_years : int, default 5
        Number of years in each training window
    test_years : int, default 1
        Number of years in each testing window
    step_years : int, default 1
        Number of years to roll forward between folds

    Returns
    -------
    list[TrainTestSplit]
        List of non-overlapping train/test splits, ordered chronologically

    Raises
    ------
    ValueError
        If parameters are invalid (negative years, insufficient data range, etc.)

    Examples
    --------
    Standard walk-forward (5-year train, 1-year test):

    >>> splits = walk_forward_splits(
    ...     start_date=datetime(2010, 1, 1, tzinfo=UTC),
    ...     end_date=datetime(2024, 12, 31, tzinfo=UTC),
    ...     train_years=5,
    ...     test_years=1,
    ... )
    >>> len(splits)
    10
    >>> print(f"First fold: {splits[0].train_start.year}-{splits[0].train_end.year}")
    First fold: 2010-2014
    >>> print(f"First test: {splits[0].test_start.year}")
    First test: 2015

    Anchored walk-forward (expanding window):

    >>> splits = walk_forward_splits(
    ...     start_date=datetime(2010, 1, 1, tzinfo=UTC),
    ...     end_date=datetime(2024, 12, 31, tzinfo=UTC),
    ...     train_years=5,
    ...     test_years=1,
    ...     step_years=1,
    ... )

    Notes
    -----
    Walk-forward validation advantages:
    - More realistic than single train/test split
    - Tests model across different market regimes
    - Reduces selection bias from single split choice
    - Provides distribution of performance metrics

    Limitations:
    - More computationally expensive (multiple model fits)
    - Overlapping training data between folds
    - Results may be serially correlated

    Common configurations:
    - Conservative: train_years=5, test_years=1 (standard)
    - Aggressive: train_years=3, test_years=1 (faster adaptation)
    - Long-term: train_years=7, test_years=2 (stable estimation)
    """
    # Validate inputs
    if train_years <= 0:
        msg = f"train_years must be positive, got {train_years}"
        raise ValueError(msg)

    if test_years <= 0:
        msg = f"test_years must be positive, got {test_years}"
        raise ValueError(msg)

    if step_years <= 0:
        msg = f"step_years must be positive, got {step_years}"
        raise ValueError(msg)

    if end_date <= start_date:
        msg = "end_date must be after start_date"
        raise ValueError(msg)

    # Ensure timezone awareness
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    # Calculate minimum required period
    min_required_years = train_years + test_years
    total_years = (end_date - start_date).days / 365.25

    if total_years < min_required_years:
        msg = (
            f"Insufficient data range: need {min_required_years} years "
            f"(train={train_years} + test={test_years}), "
            f"but only have {total_years:.1f} years"
        )
        raise ValueError(msg)

    splits: list[TrainTestSplit] = []
    current_train_start = start_date

    while True:
        # Calculate training period
        train_end = current_train_start + timedelta(days=int(train_years * 365.25)) - timedelta(days=1)

        # Calculate testing period
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=int(test_years * 365.25)) - timedelta(days=1)

        # Check if we've exceeded the available data range
        if test_end > end_date:
            break

        # Create split
        split = TrainTestSplit(
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        splits.append(split)

        # Roll forward by step_years
        current_train_start = current_train_start + timedelta(days=int(step_years * 365.25))

    if not splits:
        msg = (
            f"Could not create any splits with train_years={train_years}, "
            f"test_years={test_years} in range {start_date} to {end_date}"
        )
        raise ValueError(msg)

    LOGGER.info(
        "Generated walk-forward splits",
        num_splits=len(splits),
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
        first_split=f"{splits[0].train_start.date()} to {splits[0].test_end.date()}",
        last_split=f"{splits[-1].train_start.date()} to {splits[-1].test_end.date()}",
    )

    return splits


# ===== Validation Functions =====


def validate_no_lookahead(split: TrainTestSplit) -> bool:
    """
    Validate that a split has no look-ahead bias.

    Check that test period starts strictly after training period ends,
    ensuring no future data leaks into the training set.

    Parameters
    ----------
    split : TrainTestSplit
        Split configuration to validate

    Returns
    -------
    bool
        True if valid (no look-ahead bias), False otherwise

    Examples
    --------
    >>> split = define_train_test_split()
    >>> assert validate_no_lookahead(split)

    >>> # Invalid split
    >>> bad_split = TrainTestSplit(
    ...     train_start=datetime(2010, 1, 1, tzinfo=UTC),
    ...     train_end=datetime(2020, 1, 1, tzinfo=UTC),
    ...     test_start=datetime(2019, 1, 1, tzinfo=UTC),  # Overlaps!
    ...     test_end=datetime(2024, 1, 1, tzinfo=UTC),
    ... )  # Raises ValueError in __post_init__

    Notes
    -----
    This function is primarily used for explicit validation in testing
    and debugging. The TrainTestSplit dataclass already enforces this
    constraint in its __post_init__ method.
    """
    return split.test_start > split.train_end


def validate_splits_disjoint(splits: Sequence[TrainTestSplit]) -> bool:
    """
    Validate that test periods don't overlap across multiple splits.

    This is important for walk-forward analysis to ensure that test sets
    are truly independent (though training sets may overlap).

    Parameters
    ----------
    splits : Sequence[TrainTestSplit]
        List of splits to validate

    Returns
    -------
    bool
        True if all test periods are disjoint, False otherwise

    Examples
    --------
    >>> splits = walk_forward_splits(
    ...     start_date=datetime(2010, 1, 1, tzinfo=UTC),
    ...     end_date=datetime(2020, 12, 31, tzinfo=UTC),
    ...     train_years=5,
    ...     test_years=1,
    ... )
    >>> assert validate_splits_disjoint(splits)

    Notes
    -----
    Training periods may overlap in walk-forward analysis (by design).
    This function only checks test period disjointness.

    Overlapping test periods would lead to:
    - Double-counting performance metrics
    - Biased estimates of out-of-sample performance
    - Invalid statistical inference
    """
    if len(splits) < 2:
        return True

    # Check each pair of splits
    for i, split_i in enumerate(splits):
        for split_j in splits[i + 1:]:
            # Check if test periods overlap
            # Two periods overlap if: start1 <= end2 AND start2 <= end1
            if (
                split_i.test_start <= split_j.test_end
                and split_j.test_start <= split_i.test_end
            ):
                LOGGER.warning(
                    "Test period overlap detected",
                    split_1=f"{split_i.test_start} to {split_i.test_end}",
                    split_2=f"{split_j.test_start} to {split_j.test_end}",
                )
                return False

    return True


def validate_sufficient_training_data(
    split: TrainTestSplit,
    min_trading_days: int = 252,
) -> bool:
    """
    Validate that training period has sufficient data for factor estimation.

    Factor models require adequate historical data to estimate betas and
    factor parameters reliably. This function checks minimum duration.

    Parameters
    ----------
    split : TrainTestSplit
        Split configuration to validate
    min_trading_days : int, default 252
        Minimum required trading days (default 252 = 1 year)

    Returns
    -------
    bool
        True if training period is sufficient, False otherwise

    Examples
    --------
    >>> split = define_train_test_split()
    >>> assert validate_sufficient_training_data(split, min_trading_days=252)
    >>> assert validate_sufficient_training_data(split, min_trading_days=1260)  # 5 years

    Notes
    -----
    Recommended minimum training periods:
    - Static factor models: 252 days (1 year)
    - Rolling beta estimation: 504 days (2 years)
    - Stable factor parameter estimation: 1260 days (5 years)
    - Regime-robust models: 2520 days (10 years)

    This function uses calendar days as a proxy. Actual trading days
    will be ~70% of calendar days (accounting for weekends and holidays).
    """
    # Approximate trading days (252 per year, ~69% of calendar days)
    trading_day_ratio = 252 / 365.25
    estimated_trading_days = int(split.train_days * trading_day_ratio)

    if estimated_trading_days < min_trading_days:
        LOGGER.warning(
            "Insufficient training data",
            estimated_trading_days=estimated_trading_days,
            min_required=min_trading_days,
            train_period=f"{split.train_start} to {split.train_end}",
        )
        return False

    return True


def build_regime_aligned_split(
    regime: "MarketRegime",
    *,
    train_years: int = 5,
    buffer_days: int = 1,
) -> TrainTestSplit:
    """
    Build a train/test split aligned to a specific market regime.

    The training window ends ``buffer_days`` before the regime starts (defaults to
    a 1-day gap to prevent look-ahead), while the testing window spans the entire
    regime. Training length is specified in years (calendar approximation).

    Parameters
    ----------
    regime : MarketRegime
        Regime definition from ``regime_analysis``.
    train_years : int, default 5
        Number of calendar years to include in the training window.
    buffer_days : int, default 1
        Gap between training end and regime start to prevent look-ahead.

    Returns
    -------
    TrainTestSplit
        Split aligned to the supplied regime.

    Raises
    ------
    ValueError
        If parameters are invalid or the regime starts too early for the requested
        training horizon.
    """
    if train_years <= 0:
        msg = f"train_years must be positive, got {train_years}"
        raise ValueError(msg)

    if buffer_days < 0:
        msg = f"buffer_days must be non-negative, got {buffer_days}"
        raise ValueError(msg)

    train_end = regime.start - timedelta(days=buffer_days)
    train_duration_days = max(1, int(train_years * 365.25))
    train_start = train_end - timedelta(days=train_duration_days - 1)

    if train_start >= train_end:
        msg = "Training period must have positive length"
        raise ValueError(msg)

    if train_end >= regime.start:
        msg = "Training period must end before regime begins"
        raise ValueError(msg)

    split = TrainTestSplit(
        train_start=train_start,
        train_end=train_end,
        test_start=regime.start,
        test_end=regime.end,
    )
    return split


def build_phase3_regime_splits(
    *,
    train_years: int = 5,
    buffer_days: int = 1,
) -> dict[str, TrainTestSplit]:
    """
    Construct regime-aligned splits for all Phase 3 regimes.

    Parameters
    ----------
    train_years : int, default 5
        Number of years to use for the training window preceding each regime.
    buffer_days : int, default 1
        Gap between the training end and regime start.

    Returns
    -------
    dict[str, TrainTestSplit]
        Mapping of regime name -> split configuration.
    """
    from playground.backtest.regime_analysis import define_market_regimes

    splits: dict[str, TrainTestSplit] = {}
    for regime in define_market_regimes():
        try:
            splits[regime.name] = build_regime_aligned_split(
                regime,
                train_years=train_years,
                buffer_days=buffer_days,
            )
        except ValueError:
            LOGGER.exception("Unable to build split for regime", regime=regime.name)
    return splits


# ===== Public API =====

__all__ = [
    "TrainTestSplit",
    "define_train_test_split",
    "build_phase3_regime_splits",
    "build_regime_aligned_split",
    "validate_no_lookahead",
    "validate_splits_disjoint",
    "validate_sufficient_training_data",
    "walk_forward_splits",
]
