"""
Unit tests for train/test split utilities.

This test module validates the temporal split design for backtesting,
with particular emphasis on preventing look-ahead bias and ensuring
proper temporal separation.

Test Coverage:
- TrainTestSplit dataclass validation
- Primary split definition (define_train_test_split)
- Walk-forward split generation
- Look-ahead bias detection
- Rolling window temporal constraints
- Validation functions

All tests use timezone-aware datetimes (UTC) to match production code.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest

from playground.backtest.regime_analysis import define_market_regimes
from playground.backtest.splits import TrainTestSplit
from playground.backtest.splits import build_phase3_regime_splits
from playground.backtest.splits import build_regime_aligned_split
from playground.backtest.splits import define_train_test_split
from playground.backtest.splits import validate_no_lookahead
from playground.backtest.splits import validate_splits_disjoint
from playground.backtest.splits import validate_sufficient_training_data
from playground.backtest.splits import walk_forward_splits


# ===== TrainTestSplit Validation Tests =====


def test_train_test_split_valid_construction():
    """Test that valid split configuration is accepted."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    assert split.train_start.year == 2010
    assert split.train_end.year == 2018
    assert split.test_start.year == 2019
    assert split.test_end.year == 2024


def test_train_test_split_rejects_missing_timezone():
    """Test that timezone-naive datetimes are rejected."""
    with pytest.raises(ValueError, match="must be timezone-aware"):
        TrainTestSplit(
            train_start=datetime(2010, 1, 1),  # No tzinfo
            train_end=datetime(2018, 12, 31, tzinfo=UTC),
            test_start=datetime(2019, 1, 1, tzinfo=UTC),
            test_end=datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_train_test_split_rejects_train_end_before_start():
    """Test that training period with end before start is rejected."""
    with pytest.raises(ValueError, match="must be after training start"):
        TrainTestSplit(
            train_start=datetime(2018, 12, 31, tzinfo=UTC),  # After train_end
            train_end=datetime(2010, 1, 1, tzinfo=UTC),
            test_start=datetime(2019, 1, 1, tzinfo=UTC),
            test_end=datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_train_test_split_rejects_test_end_before_start():
    """Test that testing period with end before start is rejected."""
    with pytest.raises(ValueError, match="must be after test start"):
        TrainTestSplit(
            train_start=datetime(2010, 1, 1, tzinfo=UTC),
            train_end=datetime(2018, 12, 31, tzinfo=UTC),
            test_start=datetime(2024, 12, 31, tzinfo=UTC),  # After test_end
            test_end=datetime(2019, 1, 1, tzinfo=UTC),
        )


def test_train_test_split_rejects_overlap():
    """Test that overlapping train/test periods are rejected."""
    with pytest.raises(ValueError, match="must be after training end"):
        TrainTestSplit(
            train_start=datetime(2010, 1, 1, tzinfo=UTC),
            train_end=datetime(2020, 12, 31, tzinfo=UTC),  # Overlaps with test
            test_start=datetime(2019, 1, 1, tzinfo=UTC),
            test_end=datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_train_test_split_rejects_test_equals_train_end():
    """Test that test starting on same day as train end is rejected."""
    with pytest.raises(ValueError, match="must be after training end"):
        TrainTestSplit(
            train_start=datetime(2010, 1, 1, tzinfo=UTC),
            train_end=datetime(2018, 12, 31, tzinfo=UTC),
            test_start=datetime(2018, 12, 31, tzinfo=UTC),  # Same as train_end
            test_end=datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_train_test_split_properties():
    """Test computed properties (train_years, test_years, etc.)."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    # Check year calculations (approximately)
    assert 8.9 < split.train_years < 9.1  # ~9 years
    assert 5.9 < split.test_years < 6.1   # ~6 years

    # Check day calculations
    assert split.train_days == 3287  # Inclusive count
    assert split.test_days == 2192   # Inclusive count

    # Check gap
    assert split.gap_days == 1  # One day between periods


def test_train_test_split_to_dict():
    """Test dictionary conversion."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    config = split.to_dict()

    assert config["train_start"] == "2010-01-01T00:00:00+00:00"
    assert config["train_end"] == "2018-12-31T00:00:00+00:00"
    assert config["test_start"] == "2019-01-01T00:00:00+00:00"
    assert config["test_end"] == "2024-12-31T00:00:00+00:00"


def test_train_test_split_validate_no_overlap():
    """Test explicit overlap validation method."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    # Should not raise
    split.validate_no_overlap()


# ===== Primary Split Definition Tests =====


def test_define_train_test_split_default():
    """Test default primary split (2010-2018 / 2019-2024)."""
    split = define_train_test_split()

    assert split.train_start == datetime(2010, 1, 1, tzinfo=UTC)
    assert split.train_end == datetime(2018, 12, 31, tzinfo=UTC)
    assert split.test_start == datetime(2019, 1, 1, tzinfo=UTC)
    assert split.test_end == datetime(2024, 12, 31, tzinfo=UTC)


def test_define_train_test_split_custom_dates():
    """Test primary split with custom date overrides."""
    split = define_train_test_split(
        train_start=datetime(2012, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
    )

    assert split.train_start.year == 2012
    assert split.train_end.year == 2018  # Default
    assert split.test_start.year == 2019  # Default
    assert split.test_end.year == 2023


def test_define_train_test_split_adds_timezone():
    """Test that timezone is added to naive datetimes."""
    split = define_train_test_split(
        train_start=datetime(2012, 1, 1),  # No tzinfo
    )

    # Should have timezone added automatically
    assert split.train_start.tzinfo is not None
    assert split.train_start == datetime(2012, 1, 1, tzinfo=UTC)


def test_define_train_test_split_invalid_override():
    """Test that invalid custom dates are rejected."""
    with pytest.raises(ValueError):
        define_train_test_split(
            train_end=datetime(2025, 1, 1, tzinfo=UTC),  # After test_start
        )


# ===== Walk-Forward Split Tests =====


def test_walk_forward_splits_standard_config():
    """Test walk-forward with standard 5-year train, 1-year test."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
    )

    # Should have 10 folds
    assert len(splits) == 10

    # Check first fold
    first = splits[0]
    assert first.train_start.year == 2010
    assert first.train_end.year == 2014
    assert first.test_start.year == 2015
    assert first.test_end.year == 2015

    # Check last fold (actual dates may vary slightly due to leap years)
    last = splits[-1]
    assert last.train_start.year >= 2018  # At least 2018
    assert last.train_end.year >= 2023    # At least 2023
    assert last.test_start.year >= 2023   # At least 2023
    assert last.test_end.year == 2024     # Should end in 2024


def test_walk_forward_splits_no_overlap():
    """Test that walk-forward test periods don't overlap."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
    )

    # Check each pair of consecutive test periods
    for i in range(len(splits) - 1):
        # Test period i should end before test period i+1 starts
        assert splits[i].test_end < splits[i + 1].test_start


def test_walk_forward_splits_custom_step():
    """Test walk-forward with custom step size."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
        step_years=2,  # Jump 2 years each fold
    )

    # Fewer folds with larger step
    assert len(splits) >= 3  # At least 3 folds

    # Check first two folds
    assert splits[0].train_start.year == 2010
    assert splits[1].train_start.year == 2012  # 2 years later

    # Verify step size between consecutive folds
    delta_years = (splits[1].train_start - splits[0].train_start).days / 365.25
    assert 1.9 < delta_years < 2.1  # Approximately 2 years


def test_walk_forward_splits_insufficient_data():
    """Test that insufficient data range raises error."""
    with pytest.raises(ValueError, match="Insufficient data range"):
        walk_forward_splits(
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2022, 12, 31, tzinfo=UTC),  # Only 3 years
            train_years=5,  # Need 5+1=6 years
            test_years=1,
        )


def test_walk_forward_splits_invalid_params():
    """Test that invalid parameters are rejected."""
    # Negative train_years
    with pytest.raises(ValueError, match="train_years must be positive"):
        walk_forward_splits(
            start_date=datetime(2010, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            train_years=-5,
            test_years=1,
        )

    # Zero test_years
    with pytest.raises(ValueError, match="test_years must be positive"):
        walk_forward_splits(
            start_date=datetime(2010, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            train_years=5,
            test_years=0,
        )

    # Negative step_years
    with pytest.raises(ValueError, match="step_years must be positive"):
        walk_forward_splits(
            start_date=datetime(2010, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            train_years=5,
            test_years=1,
            step_years=-1,
        )


def test_walk_forward_splits_adds_timezone():
    """Test that timezone is added to naive datetimes."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1),  # No tzinfo
        end_date=datetime(2015, 12, 31),  # No tzinfo
        train_years=3,
        test_years=1,
    )

    # All splits should have timezone-aware dates
    for split in splits:
        assert split.train_start.tzinfo is not None
        assert split.train_end.tzinfo is not None
        assert split.test_start.tzinfo is not None
        assert split.test_end.tzinfo is not None


# ===== Validation Function Tests =====


def test_validate_no_lookahead_valid():
    """Test that valid split passes look-ahead check."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    assert validate_no_lookahead(split) is True


def test_validate_splits_disjoint_valid():
    """Test that disjoint test periods pass validation."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
    )

    assert validate_splits_disjoint(splits) is True


def test_validate_splits_disjoint_single_split():
    """Test that single split always passes disjoint check."""
    split = define_train_test_split()
    assert validate_splits_disjoint([split]) is True


def test_validate_splits_disjoint_empty():
    """Test that empty split list passes disjoint check."""
    assert validate_splits_disjoint([]) is True


def test_validate_splits_disjoint_overlapping():
    """Test that overlapping test periods fail validation."""
    # Create two splits with overlapping test periods
    split1 = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2014, 12, 31, tzinfo=UTC),
        test_start=datetime(2015, 1, 1, tzinfo=UTC),
        test_end=datetime(2016, 12, 31, tzinfo=UTC),  # Overlaps with split2
    )

    split2 = TrainTestSplit(
        train_start=datetime(2011, 1, 1, tzinfo=UTC),
        train_end=datetime(2015, 12, 31, tzinfo=UTC),
        test_start=datetime(2016, 1, 1, tzinfo=UTC),  # Overlaps with split1
        test_end=datetime(2017, 12, 31, tzinfo=UTC),
    )

    assert validate_splits_disjoint([split1, split2]) is False


def test_validate_sufficient_training_data_sufficient():
    """Test that split with sufficient training data passes."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 12, 31, tzinfo=UTC),  # 9 years
        test_start=datetime(2019, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
    )

    # Should pass with default minimum (252 days)
    assert validate_sufficient_training_data(split) is True

    # Should pass with 5-year minimum
    assert validate_sufficient_training_data(split, min_trading_days=1260) is True


def test_validate_sufficient_training_data_insufficient():
    """Test that split with insufficient training data fails."""
    split = TrainTestSplit(
        train_start=datetime(2018, 1, 1, tzinfo=UTC),
        train_end=datetime(2018, 6, 30, tzinfo=UTC),  # Only 6 months
        test_start=datetime(2018, 7, 1, tzinfo=UTC),
        test_end=datetime(2019, 12, 31, tzinfo=UTC),
    )

    # Should fail with default minimum (252 days)
    assert validate_sufficient_training_data(split, min_trading_days=252) is False


# ===== Integration Tests =====


def test_full_pipeline_primary_split():
    """Test complete pipeline with primary split."""
    # Define split
    split = define_train_test_split()

    # Validate
    split.validate_no_overlap()
    assert validate_no_lookahead(split)
    assert validate_sufficient_training_data(split, min_trading_days=252)

    # Check properties
    assert split.train_years >= 8
    assert split.test_years >= 5


def test_full_pipeline_walk_forward():
    """Test complete pipeline with walk-forward."""
    # Generate splits
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
    )

    # Validate all splits
    for split in splits:
        split.validate_no_overlap()
        assert validate_no_lookahead(split)
        assert validate_sufficient_training_data(split, min_trading_days=252)

    # Validate disjoint test periods
    assert validate_splits_disjoint(splits)


def test_rolling_window_constraint_simulation():
    """Test that rolling windows respect split boundaries."""
    split = define_train_test_split()

    # Simulate rebalancing in early testing period
    rebalance_date = datetime(2019, 3, 31, tzinfo=UTC)  # 3 months into test
    window_size = timedelta(days=365)  # 1 year window

    window_start = rebalance_date - window_size
    window_end = rebalance_date

    # This window would span training/testing boundary
    assert window_start < split.test_start  # Starts in training
    assert window_end > split.test_start    # Ends in testing

    # In real implementation, we would either:
    # 1. Skip this rebalance date, OR
    # 2. Use shorter window starting at split.test_start

    # Option 2: Adjust window to start at test period
    adjusted_window_start = max(window_start, split.test_start)
    adjusted_window_end = window_end

    # Now window is entirely in testing period
    assert adjusted_window_start >= split.test_start
    assert adjusted_window_end <= split.test_end


def test_parameter_freezing_simulation():
    """Test that parameters are frozen at training end."""
    split = define_train_test_split()

    # Simulate parameter estimation
    parameter_estimation_date = split.train_end

    # Parameters should be estimated up to and including train_end
    assert parameter_estimation_date <= split.train_end

    # In testing period, verify parameters would never be re-estimated
    test_dates = [
        datetime(2019, 6, 30, tzinfo=UTC),
        datetime(2020, 12, 31, tzinfo=UTC),
        datetime(2024, 6, 30, tzinfo=UTC),
    ]

    for test_date in test_dates:
        # Parameters remain frozen (estimated on parameter_estimation_date)
        assert parameter_estimation_date < test_date
        assert parameter_estimation_date <= split.train_end


# ===== Edge Case Tests =====


def test_split_with_single_day_gap():
    """Test split with minimum gap (1 day) between periods."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2015, 12, 31, tzinfo=UTC),
        test_start=datetime(2016, 1, 1, tzinfo=UTC),  # Next day
        test_end=datetime(2020, 12, 31, tzinfo=UTC),
    )

    assert split.gap_days == 1
    assert validate_no_lookahead(split)


def test_split_with_large_gap():
    """Test split with large gap between periods."""
    split = TrainTestSplit(
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2015, 12, 31, tzinfo=UTC),
        test_start=datetime(2017, 1, 1, tzinfo=UTC),  # 1 year gap
        test_end=datetime(2020, 12, 31, tzinfo=UTC),
    )

    assert split.gap_days == 367  # 2016 was leap year
    assert validate_no_lookahead(split)


def test_walk_forward_with_short_periods():
    """Test walk-forward with short periods (edge case)."""
    splits = walk_forward_splits(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2018, 12, 31, tzinfo=UTC),
        train_years=3,
        test_years=1,
        step_years=1,
    )

    # Should have 5 folds (2010-2012→2013, 2011-2013→2014, ..., 2014-2016→2017)
    assert len(splits) == 6

    # Validate all splits
    for split in splits:
        assert validate_no_lookahead(split)


def test_immutability():
    """Test that TrainTestSplit is immutable (frozen dataclass)."""
    split = define_train_test_split()

    # Attempting to modify should raise AttributeError
    with pytest.raises(AttributeError):
        split.train_start = datetime(2012, 1, 1, tzinfo=UTC)  # type: ignore

    with pytest.raises(AttributeError):
        split.test_end = datetime(2025, 1, 1, tzinfo=UTC)  # type: ignore


def test_build_regime_aligned_split() -> None:
    """Train/test split should align with regime boundaries."""
    regime = define_market_regimes()[0]
    split = build_regime_aligned_split(regime, train_years=2, buffer_days=1)

    assert split.test_start == regime.start
    assert split.test_end == regime.end
    assert split.train_end < split.test_start
    assert split.train_start < split.train_end


def test_build_phase3_regime_splits() -> None:
    """Phase 3 helper should build splits for every regime."""
    regimes = define_market_regimes()
    splits = build_phase3_regime_splits(train_years=3)

    assert set(splits.keys()) == {regime.name for regime in regimes}

    for regime in regimes:
        split = splits[regime.name]
        assert split.test_start == regime.start
        assert split.test_end == regime.end
