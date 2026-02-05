#!/usr/bin/env python3

"""End-to-end tests for vintage policy enforcement in orchestration."""

from datetime import datetime

import pandas as pd
import pytest

from ml.orchestration.vintage import VintageWindowPolicy


@pytest.mark.e2e
class TestOrchestratorVintage:
    """E2E tests for vintage policy enforcement."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample dataset spanning full year 2024.

        Returns
        -------
        pd.DataFrame
            DataFrame with daily timestamps from 2024-01-01 to 2024-12-31

        """
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "close": range(len(dates)),
                "volume": [1000 + i * 10 for i in range(len(dates))],
            }
        )

    def test_e2e_vintage_policy_enforcement(self, sample_data: pd.DataFrame) -> None:
        """Test vintage policy enforcement with temporal boundaries.

        Property Under Test: Vintage policies enforce temporal boundaries

        Given:
        - Training data with timestamps ranging from 2024-01-01 to 2024-12-31
        - Vintage policy: max_age_days=30 (only use data <30 days old)
        - Current time set as 2024-12-15

        When:
        - Building dataset with vintage policy enabled
        - Only data from 2024-11-15 to 2024-12-15 should be included
        - Data before 2024-11-15 should be filtered out

        Then:
        - Dataset contains only data within vintage window
        - Row count matches expected filtered count
        - Oldest timestamp in dataset >= 2024-11-15
        - Vintage metadata saved (policy applied, filter date)

        """
        # Given: Vintage policy with 30-day window
        policy = VintageWindowPolicy(max_age_days=30)

        # When: Apply vintage filtering with current date as 2024-12-15
        current_date = datetime(2024, 12, 15)
        filtered_df = policy.filter_by_vintage(
            sample_data,
            current_date,
            timestamp_column="timestamp",
        )

        # Then: Verify vintage boundary enforcement
        oldest_ts = filtered_df["timestamp"].min()
        newest_ts = filtered_df["timestamp"].max()

        # Assert oldest timestamp >= cutoff (2024-11-15)
        assert oldest_ts >= pd.Timestamp("2024-11-15"), (
            f"Oldest timestamp {oldest_ts} is before cutoff 2024-11-15"
        )

        # Assert newest timestamp <= current date (2024-12-15)
        assert newest_ts <= pd.Timestamp("2024-12-15"), (
            f"Newest timestamp {newest_ts} is after current date 2024-12-15"
        )

        # Assert correct row count (31 days: Nov 15-Dec 15 inclusive)
        expected_rows = 31  # 30 days + current day
        assert len(filtered_df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(filtered_df)}"
        )

        # Verify metadata computation
        metadata = policy.compute_vintage_metadata(
            current_date=current_date,
            original_count=len(sample_data),
            filtered_count=len(filtered_df),
        )

        # Assert metadata contains expected fields
        assert "vintage_policy" in metadata
        vintage_meta = metadata["vintage_policy"]

        assert vintage_meta["max_age_days"] == 30
        assert vintage_meta["cutoff_date"] == "2024-11-15"
        assert vintage_meta["current_date"] == "2024-12-15"
        assert vintage_meta["original_count"] == len(sample_data)
        assert vintage_meta["filtered_count"] == expected_rows
        assert vintage_meta["rows_removed"] == len(sample_data) - expected_rows

    def test_vintage_policy_with_different_windows(
        self,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test vintage policy with different time windows.

        Verifies that different max_age_days settings produce correct filtering.

        """
        # Test 7-day window
        policy_7d = VintageWindowPolicy(max_age_days=7)
        current_date = datetime(2024, 12, 15)

        filtered_7d = policy_7d.filter_by_vintage(
            sample_data,
            current_date,
            timestamp_column="timestamp",
        )

        # Should have 8 days (Dec 8-15 inclusive)
        assert len(filtered_7d) == 8
        assert filtered_7d["timestamp"].min() >= pd.Timestamp("2024-12-08")

        # Test 90-day window
        policy_90d = VintageWindowPolicy(max_age_days=90)
        filtered_90d = policy_90d.filter_by_vintage(
            sample_data,
            current_date,
            timestamp_column="timestamp",
        )

        # Should have 91 days (Sep 16-Dec 15 inclusive)
        assert len(filtered_90d) == 91
        assert filtered_90d["timestamp"].min() >= pd.Timestamp("2024-09-16")

    def test_vintage_policy_validation(self) -> None:
        """Test vintage policy parameter validation.

        Verifies that invalid parameters raise appropriate errors.

        """
        # Test invalid max_age_days
        with pytest.raises(ValueError, match="max_age_days must be > 0"):
            VintageWindowPolicy(max_age_days=0)

        with pytest.raises(ValueError, match="max_age_days must be > 0"):
            VintageWindowPolicy(max_age_days=-10)

    def test_vintage_policy_missing_timestamp_column(
        self,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test error handling when timestamp column is missing.

        Verifies that missing timestamp columns raise descriptive errors.

        """
        policy = VintageWindowPolicy(max_age_days=30)
        current_date = datetime(2024, 12, 15)

        # Test with non-existent column
        with pytest.raises(ValueError, match="Timestamp column 'nonexistent' not found"):
            policy.filter_by_vintage(
                sample_data,
                current_date,
                timestamp_column="nonexistent",
            )

    def test_vintage_policy_with_custom_timestamp_column(self) -> None:
        """Test vintage policy with custom timestamp column name.

        Verifies that vintage filtering works with different column names.

        """
        # Create data with custom timestamp column
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        df = pd.DataFrame(
            {
                "trade_time": dates,
                "price": range(len(dates)),
            }
        )

        policy = VintageWindowPolicy(max_age_days=30)
        current_date = datetime(2024, 12, 15)

        filtered = policy.filter_by_vintage(
            df,
            current_date,
            timestamp_column="trade_time",
        )

        # Verify filtering worked with custom column
        assert len(filtered) == 31
        assert filtered["trade_time"].min() >= pd.Timestamp("2024-11-15")

    def test_vintage_policy_edge_case_empty_dataframe(self) -> None:
        """Test vintage policy with empty dataframe.

        Verifies graceful handling of edge cases.

        """
        policy = VintageWindowPolicy(max_age_days=30)
        current_date = datetime(2024, 12, 15)

        empty_df = pd.DataFrame({"timestamp": [], "value": []})
        empty_df["timestamp"] = pd.to_datetime(empty_df["timestamp"])

        filtered = policy.filter_by_vintage(
            empty_df,
            current_date,
            timestamp_column="timestamp",
        )

        # Should return empty dataframe without error
        assert len(filtered) == 0
        assert list(filtered.columns) == ["timestamp", "value"]

    def test_vintage_policy_all_data_filtered(self, sample_data: pd.DataFrame) -> None:
        """Test vintage policy when all data is outside window.

        Verifies handling when all data is too old.

        """
        policy = VintageWindowPolicy(max_age_days=1)

        # Use a current date far in the future
        current_date = datetime(2025, 12, 31)

        filtered = policy.filter_by_vintage(
            sample_data,
            current_date,
            timestamp_column="timestamp",
        )

        # All 2024 data should be filtered out (more than 1 day old)
        assert len(filtered) == 0

    def test_vintage_policy_no_data_filtered(self, sample_data: pd.DataFrame) -> None:
        """Test vintage policy when no data is filtered.

        Verifies handling when all data is within window.

        """
        policy = VintageWindowPolicy(max_age_days=365)

        # Use end of year as current date
        current_date = datetime(2024, 12, 31)

        filtered = policy.filter_by_vintage(
            sample_data,
            current_date,
            timestamp_column="timestamp",
        )

        # All data should pass (entire year is within 365-day window)
        assert len(filtered) == len(sample_data)
        assert len(filtered) == 366  # 2024 is a leap year

    def test_vintage_metadata_calculation(self) -> None:
        """Test vintage metadata calculation.

        Verifies that metadata accurately reflects filtering results.

        """
        policy = VintageWindowPolicy(max_age_days=30)
        current_date = datetime(2024, 12, 15)

        metadata = policy.compute_vintage_metadata(
            current_date=current_date,
            original_count=365,
            filtered_count=31,
        )

        assert metadata["vintage_policy"]["max_age_days"] == 30
        assert metadata["vintage_policy"]["cutoff_date"] == "2024-11-15"
        assert metadata["vintage_policy"]["current_date"] == "2024-12-15"
        assert metadata["vintage_policy"]["original_count"] == 365
        assert metadata["vintage_policy"]["filtered_count"] == 31
        assert metadata["vintage_policy"]["rows_removed"] == 334

        # Check removal percentage calculation
        expected_pct = round(100.0 * 334 / 365, 2)
        assert metadata["vintage_policy"]["removal_pct"] == expected_pct
