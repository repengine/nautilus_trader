"""
Unit tests for PurgedCrossValidator.

Tests purged walk-forward cross-validation with embargo to prevent information leakage.

"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.preprocessing.stationarity import PurgedCrossValidator


class TestPurgedCrossValidator:
    """Test purged cross-validation functionality."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        cv = PurgedCrossValidator(n_splits=5, purge_gap=10, embargo_pct=0.01)
        assert cv.n_splits == 5
        assert cv.purge_gap == 10
        assert cv.embargo_pct == 0.01

    def test_init_invalid_n_splits(self) -> None:
        """Test initialization with invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            PurgedCrossValidator(n_splits=1)

    def test_init_invalid_purge_gap(self) -> None:
        """Test initialization with invalid purge_gap."""
        with pytest.raises(ValueError, match="purge_gap must be non-negative"):
            PurgedCrossValidator(purge_gap=-1)

    def test_init_invalid_embargo_pct(self) -> None:
        """Test initialization with invalid embargo_pct."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            PurgedCrossValidator(embargo_pct=1.5)

    def test_basic_split_no_purge_no_embargo(self) -> None:
        """Test basic splitting without purge or embargo."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedCrossValidator(n_splits=5, purge_gap=0, embargo_pct=0)

        splits = cv.split(X)

        # Should have 5 splits
        assert len(splits) == 5

        # Check each split
        for i, (train_idx, test_idx) in enumerate(splits):
            # Test set should be contiguous
            assert np.all(np.diff(test_idx) == 1)

            # Test sets should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0

            # Each test set should have ~20 samples (100/5)
            assert 19 <= len(test_idx) <= 21

    def test_split_with_purge_gap(self) -> None:
        """Test splitting with purge gap."""
        X = np.arange(100).reshape(-1, 1)
        purge_gap = 5
        cv = PurgedCrossValidator(n_splits=5, purge_gap=purge_gap, embargo_pct=0)

        splits = cv.split(X)

        for train_idx, test_idx in splits:
            # Check purge gap: no training sample should be within purge_gap of test
            test_min, test_max = test_idx.min(), test_idx.max()

            # Training samples before test should respect purge gap
            train_before = train_idx[train_idx < test_min]
            if len(train_before) > 0:
                assert train_before.max() < test_min - purge_gap

            # Training samples after test should respect purge gap
            train_after = train_idx[train_idx > test_max]
            if len(train_after) > 0:
                assert train_after.min() > test_max + purge_gap

    def test_split_with_embargo(self) -> None:
        """Test splitting with embargo."""
        X = np.arange(100).reshape(-1, 1)
        embargo_pct = 0.1  # 10% embargo
        cv = PurgedCrossValidator(n_splits=5, purge_gap=0, embargo_pct=embargo_pct)

        splits = cv.split(X)
        embargo_size = int(100 * embargo_pct)  # 10 samples

        # Check embargo in all but last split
        for i, (train_idx, test_idx) in enumerate(splits[:-1]):
            test_end = test_idx.max()
            embargo_end = test_end + embargo_size

            # No training samples should be in embargo zone
            embargo_zone = set(range(test_end + 1, min(embargo_end + 1, 100)))
            train_set = set(train_idx)
            assert len(embargo_zone & train_set) == 0

    def test_get_n_splits(self) -> None:
        """Test get_n_splits method."""
        cv = PurgedCrossValidator(n_splits=7)
        assert cv.get_n_splits() == 7
        assert cv.get_n_splits(X=np.arange(100)) == 7

    @given(
        n_samples=st.integers(min_value=20, max_value=200),
        n_splits=st.integers(min_value=2, max_value=10),
        purge_gap=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_split_property_no_overlap(
        self,
        n_samples: int,
        n_splits: int,
        purge_gap: int,
    ) -> None:
        """Property: train and test sets should never overlap."""
        X = np.arange(n_samples).reshape(-1, 1)
        cv = PurgedCrossValidator(n_splits=n_splits, purge_gap=purge_gap, embargo_pct=0)

        splits = cv.split(X)

        for train_idx, test_idx in splits:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

            # All indices should be valid
            assert np.all(train_idx >= 0)
            assert np.all(train_idx < n_samples)
            assert np.all(test_idx >= 0)
            assert np.all(test_idx < n_samples)

    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_splits=st.integers(min_value=3, max_value=8),
        purge_gap=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_split_property_purge_gap_respected(
        self,
        n_samples: int,
        n_splits: int,
        purge_gap: int,
    ) -> None:
        """Property: purge gap should always be respected."""
        X = np.arange(n_samples).reshape(-1, 1)
        cv = PurgedCrossValidator(n_splits=n_splits, purge_gap=purge_gap, embargo_pct=0)

        splits = cv.split(X)

        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            # Calculate minimum distance between train and test
            for test_i in test_idx:
                for train_i in train_idx:
                    distance = abs(test_i - train_i)
                    # Distance should be > purge_gap (not >= because indices are discrete)
                    assert distance > purge_gap, (
                        f"Purge gap violated: distance {distance} <= {purge_gap}"
                    )

    def test_split_coverage(self) -> None:
        """Test that all samples appear in at least one test set."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedCrossValidator(n_splits=5, purge_gap=0, embargo_pct=0)

        splits = cv.split(X)

        # Collect all test indices
        all_test_indices: set[int] = set()
        for _, test_idx in splits:
            all_test_indices.update(test_idx)

        # Should cover all samples
        assert all_test_indices == set(range(100))

    def test_split_temporal_order(self) -> None:
        """Test that splits maintain temporal order."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedCrossValidator(n_splits=5, purge_gap=2, embargo_pct=0.05)

        splits = cv.split(X)

        # Test sets should be in temporal order
        test_starts = []
        for _, test_idx in splits:
            test_starts.append(test_idx.min())

        # Test sets should progress forward in time
        assert test_starts == sorted(test_starts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
