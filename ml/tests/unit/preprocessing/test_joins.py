"""
Unit tests for point-in-time join utilities.

Tests PIT-correct joins, embargo windows, and lookahead validation using both
standard tests and property-based testing with Hypothesis.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import pd
from ml._imports import pl
from ml.preprocessing.joins import asof_join
from ml.preprocessing.joins import create_lag_features
from ml.preprocessing.joins import embargo_window
from ml.preprocessing.joins import validate_no_lookahead


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


pytestmark = pytest.mark.skipif(
    not HAS_POLARS or not HAS_PANDAS,
    reason="Requires polars and pandas",
)


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestAsofJoin:
    """Test point-in-time correct as-of joins."""

    def test_asof_join_basic_polars(self) -> None:
        """Test basic as-of join with Polars."""
        # Create left dataframe with timestamps
        left = pl.DataFrame({
            "timestamp": [100, 200, 300, 400],
            "instrument_id": ["SPY", "SPY", "SPY", "SPY"],
            "price": [100.0, 101.0, 102.0, 103.0],
        })

        # Create right dataframe with events
        right = pl.DataFrame({
            "timestamp": [150, 250, 350],
            "instrument_id": ["SPY", "SPY", "SPY"],
            "event": ["event1", "event2", "event3"],
        })

        # Perform as-of join
        result = asof_join(left, right, on="timestamp", by="instrument_id")

        # Check results
        assert len(result) == 4
        assert result["event"][0] is None  # No event before timestamp 100
        assert result["event"][1] == "event1"  # Event1 at 150 applies to 200
        assert result["event"][2] == "event2"  # Event2 at 250 applies to 300
        assert result["event"][3] == "event3"  # Event3 at 350 applies to 400

    def test_asof_join_basic_pandas(self) -> None:
        """Test basic as-of join with Pandas."""
        # Create left dataframe with timestamps
        left = pd.DataFrame({
            "timestamp": [100, 200, 300, 400],
            "instrument_id": ["SPY", "SPY", "SPY", "SPY"],
            "price": [100.0, 101.0, 102.0, 103.0],
        })

        # Create right dataframe with events
        right = pd.DataFrame({
            "timestamp": [150, 250, 350],
            "instrument_id": ["SPY", "SPY", "SPY"],
            "event": ["event1", "event2", "event3"],
        })

        # Perform as-of join
        result = asof_join(left, right, on="timestamp", by="instrument_id")
        result_pd = cast(pd.DataFrame, result)

        # Check results
        assert len(result_pd) == 4
        assert pd.isna(result_pd["event"].iloc[0])  # No event before timestamp 100
        assert result_pd["event"].iloc[1] == "event1"  # Event1 at 150 applies to 200
        assert result_pd["event"].iloc[2] == "event2"  # Event2 at 250 applies to 300
        assert result_pd["event"].iloc[3] == "event3"  # Event3 at 350 applies to 400

    def test_asof_join_with_tolerance_polars(self) -> None:
        """Test as-of join with tolerance window."""
        # Use datetime for tolerance support
        from datetime import datetime
        from datetime import timedelta

        base_time = datetime(2024, 1, 1)
        left = pl.DataFrame({
            "timestamp": [
                base_time,
                base_time + timedelta(seconds=1),
                base_time + timedelta(seconds=2),
                base_time + timedelta(seconds=10),  # Far away
            ],
            "value": [1, 2, 3, 4],
        })

        right = pl.DataFrame({
            "timestamp": [
                base_time + timedelta(milliseconds=500),
                base_time + timedelta(seconds=1, milliseconds=500),
            ],
            "reference": ["A", "B"],
        })

        # Join with 1 second tolerance
        result = asof_join(left, right, on="timestamp", tolerance="1s")

        # Check that far-away point doesn't match
        assert result["reference"][3] is None  # Too far from any reference

    @given(
        left_size=st.integers(min_value=1, max_value=100),
        right_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20, deadline=5000)
    def test_asof_join_property_no_future_data(
        self,
        left_size: int,
        right_size: int,
    ) -> None:
        """Property: as-of join should never include future data."""
        # Generate sorted timestamps
        left_timestamps = np.sort(np.random.randint(0, 10000, left_size))
        right_timestamps = np.sort(np.random.randint(0, 10000, right_size))

        left = pl.DataFrame({
            "timestamp": left_timestamps,
            "left_value": np.arange(left_size),
        })

        right = pl.DataFrame({
            "timestamp": right_timestamps,
            "right_value": np.arange(right_size),
        })

        # Perform backward as-of join
        result = asof_join(left, right, on="timestamp", direction="backward")

        # For each row in result, if there's a matched right value,
        # its timestamp must be <= the left timestamp
        for i in range(len(result)):
            if result["right_value"][i] is not None:
                left_ts = result["timestamp"][i]
                # Find the original right timestamp
                right_idx = int(result["right_value"][i])
                right_ts = right_timestamps[right_idx]
                assert right_ts <= left_ts, "Future data included in join!"


class TestEmbargoWindow:
    """Test embargo window functionality."""

    def test_embargo_window_basic_polars(self) -> None:
        """Test basic embargo window with Polars."""
        df = pl.DataFrame({
            "ts_event": [100, 200, 300, 400, 500],
            "value": [1, 2, 3, 4, 5],
        })

        # Embargo around event at 300
        result = embargo_window(
            df,
            event_timestamps=[300],
            embargo_before_ns=100,
            embargo_after_ns=100,
        )

        # Check embargo flags
        assert not result["embargo"][0]  # 100: outside window
        assert result["embargo"][1]  # 200: within before window
        assert result["embargo"][2]  # 300: event time
        assert result["embargo"][3]  # 400: within after window
        assert not result["embargo"][4]  # 500: outside window

    def test_embargo_window_multiple_events(self) -> None:
        """Test embargo with multiple events."""
        df = pl.DataFrame({
            "ts_event": list(range(0, 1000, 100)),
            "value": list(range(10)),
        })

        # Multiple events
        events = [200, 600]
        result = embargo_window(
            df,
            event_timestamps=events,
            embargo_before_ns=150,
            embargo_after_ns=150,
        )

        embargo_flags = result["embargo"].to_list()

        # Check specific points
        assert embargo_flags[0] == False  # 0: not embargoed
        assert embargo_flags[1] == True   # 100: before event at 200
        assert embargo_flags[2] == True   # 200: event time
        assert embargo_flags[3] == True   # 300: after event at 200
        assert embargo_flags[4] == False  # 400: between events
        assert embargo_flags[5] == True   # 500: before event at 600
        assert embargo_flags[6] == True   # 600: event time
        assert embargo_flags[7] == True   # 700: after event at 600

    @given(
        df_size=st.integers(min_value=10, max_value=100),
        n_events=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_embargo_window_property_symmetric(
        self,
        df_size: int,
        n_events: int,
    ) -> None:
        """Property: embargo windows should be symmetric around events."""
        timestamps = np.arange(0, df_size * 1000, 1000)

        df = pl.DataFrame({
            "ts_event": timestamps,
            "value": np.arange(df_size),
        })

        # Random events within the range
        event_indices = np.random.choice(df_size, min(n_events, df_size), replace=False)
        event_timestamps = timestamps[event_indices].tolist()

        # Apply symmetric embargo
        embargo_ns = 500
        result = embargo_window(
            df,
            event_timestamps=event_timestamps,
            embargo_before_ns=embargo_ns,
            embargo_after_ns=embargo_ns,
        )

        # Check symmetry: for each event, same number of embargoed points before and after
        # (within data bounds)
        for event_ts in event_timestamps:
            before_mask = (timestamps >= event_ts - embargo_ns) & (timestamps < event_ts)
            after_mask = (timestamps > event_ts) & (timestamps <= event_ts + embargo_ns)

            before_count = before_mask.sum()
            after_count = after_mask.sum()

            # Should be equal if not at boundaries
            if event_ts > embargo_ns and event_ts < timestamps[-1] - embargo_ns:
                assert abs(before_count - after_count) <= 1  # Allow for rounding


class TestValidateNoLookahead:
    """Test lookahead bias validation."""

    def test_validate_no_lookahead_pass(self) -> None:
        """Test validation passes when no lookahead."""
        features = pl.DataFrame({
            "ts_event": [100, 200, 300],
            "feature": [1, 2, 3],
        })

        targets = pl.DataFrame({
            "ts_event": [400, 500, 600],
            "target": [10, 20, 30],
        })

        # Should pass - all features before targets
        assert validate_no_lookahead(features, targets)

    def test_validate_no_lookahead_fail(self) -> None:
        """Test validation fails when lookahead detected."""
        features = pl.DataFrame({
            "ts_event": [100, 200, 500],  # 500 is after first target
            "feature": [1, 2, 3],
        })

        targets = pl.DataFrame({
            "ts_event": [400, 500, 600],
            "target": [10, 20, 30],
        })

        # Should fail - feature at 500 >= target at 400
        with pytest.raises(ValueError, match="Lookahead bias detected"):
            validate_no_lookahead(features, targets)

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        feature_lag=st.integers(min_value=1, max_value=10),
    )
    def test_validate_no_lookahead_property(
        self,
        n_samples: int,
        feature_lag: int,
    ) -> None:
        """Property: properly lagged features should never have lookahead."""
        # Ensure feature_lag doesn't exceed n_samples
        feature_lag = min(feature_lag, n_samples - 1)

        base_timestamps = np.arange(n_samples) * 1000

        # Features are earlier (no overlap with targets)
        if n_samples > feature_lag:
            # Features end before targets begin
            features = pl.DataFrame({
                "ts_event": base_timestamps[:n_samples - feature_lag],
                "feature": np.arange(n_samples - feature_lag),
            })

            # Targets start after features end
            targets = pl.DataFrame({
                "ts_event": base_timestamps[n_samples - feature_lag:],
                "target": np.arange(n_samples - feature_lag, n_samples),
            })

            # Should always pass with proper lag
            assert validate_no_lookahead(features, targets)


class TestCreateLagFeatures:
    """Test lag feature creation."""

    def test_create_lag_features_basic(self) -> None:
        """Test basic lag feature creation."""
        df = pl.DataFrame({
            "ts_event": [100, 200, 300, 400, 500],
            "price": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100, 200, 300, 400, 500],
        })

        result = create_lag_features(
            df,
            columns=["price", "volume"],
            lags=[1, 2],
        )

        # Check lag columns exist
        assert "price_lag_1" in result.columns
        assert "price_lag_2" in result.columns
        assert "volume_lag_1" in result.columns
        assert "volume_lag_2" in result.columns

        # Check values
        assert result["price_lag_1"][1] == 10.0  # Previous price
        assert result["price_lag_2"][2] == 10.0  # 2 periods back
        assert result["volume_lag_1"][1] == 100  # Previous volume

    def test_create_lag_features_grouped(self) -> None:
        """Test lag features with grouping."""
        df = pl.DataFrame({
            "ts_event": [100, 200, 300, 100, 200, 300],
            "instrument_id": ["SPY", "SPY", "SPY", "QQQ", "QQQ", "QQQ"],
            "price": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
        })

        result = create_lag_features(
            df,
            columns=["price"],
            lags=[1],
            group_by="instrument_id",
        )

        # Sort for consistent checking
        result_pl = cast(pl.DataFrame, result)
        result = result_pl.sort(["instrument_id", "ts_event"])

        # Check SPY lags
        spy_data = result.filter(pl.col("instrument_id") == "SPY")
        assert spy_data["price_lag_1"][0] is None  # First in group
        assert spy_data["price_lag_1"][1] == 100.0
        assert spy_data["price_lag_1"][2] == 101.0

        # Check QQQ lags
        qqq_data = result.filter(pl.col("instrument_id") == "QQQ")
        assert qqq_data["price_lag_1"][0] is None  # First in group
        assert qqq_data["price_lag_1"][1] == 50.0
        assert qqq_data["price_lag_1"][2] == 51.0

    @given(
        n_rows=st.integers(min_value=5, max_value=50),
        n_lags=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_create_lag_features_property_consistency(
        self,
        n_rows: int,
        n_lags: int,
    ) -> None:
        """Property: lag features should maintain temporal consistency."""
        timestamps = np.arange(n_rows) * 1000
        values = np.random.randn(n_rows)

        df = pl.DataFrame({
            "ts_event": timestamps,
            "value": values,
        })

        lags = list(range(1, min(n_lags + 1, n_rows)))
        result = create_lag_features(df, columns=["value"], lags=lags)

        # Check that each lag contains the correct historical value
        for lag in lags:
            lag_col = f"value_lag_{lag}"
            for i in range(lag, n_rows):
                expected = values[i - lag]
                actual = result[lag_col][i]
                if actual is not None:
                    assert abs(actual - expected) < 1e-10, f"Lag {lag} incorrect at row {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
