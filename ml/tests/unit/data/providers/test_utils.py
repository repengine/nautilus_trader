"""
Unit tests for data provider utility functions.

Using hypothesis for property-based testing of mathematical functions.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.data.providers.utils import align_timeseries
from ml.data.providers.utils import cyclic_encode
from ml.data.providers.utils import time_to_event
from ml.data.providers.utils import validate_timestamps


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestCyclicEncode:
    """Test cyclic encoding function."""

    def test_cyclic_encode_basic(self) -> None:
        """Test basic cyclic encoding values."""
        # Test known values
        sin_val, cos_val = cyclic_encode(0, 24)  # Midnight
        assert np.abs(sin_val - 0.0) < 1e-10
        assert np.abs(cos_val - 1.0) < 1e-10

        sin_val, cos_val = cyclic_encode(6, 24)  # 6 AM
        assert np.abs(sin_val - 1.0) < 1e-10
        assert np.abs(cos_val - 0.0) < 1e-10

        sin_val, cos_val = cyclic_encode(12, 24)  # Noon
        assert np.abs(sin_val - 0.0) < 1e-10
        assert np.abs(cos_val - (-1.0)) < 1e-10

        sin_val, cos_val = cyclic_encode(18, 24)  # 6 PM
        assert np.abs(sin_val - (-1.0)) < 1e-10
        assert np.abs(cos_val - 0.0) < 1e-10

    @given(
        value=st.floats(min_value=0, max_value=1000, allow_nan=False),
        period=st.floats(min_value=0.1, max_value=1000, allow_nan=False),
    )
    def test_cyclic_encode_unit_circle_property(self, value: float, period: float) -> None:
        """Property: sin^2 + cos^2 = 1 (unit circle)."""
        sin_val, cos_val = cyclic_encode(value, period)

        # Should be on unit circle
        radius_squared = sin_val**2 + cos_val**2
        assert np.abs(radius_squared - 1.0) < 1e-10

    @given(
        value=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        period=st.floats(min_value=0.1, max_value=1000, allow_nan=False),
    )
    def test_cyclic_encode_range_property(self, value: float, period: float) -> None:
        """Property: values always in [-1, 1]."""
        sin_val, cos_val = cyclic_encode(value, period)

        assert -1 <= sin_val <= 1
        assert -1 <= cos_val <= 1

    @given(
        value=st.floats(min_value=0, max_value=100, allow_nan=False),
        period=st.floats(min_value=1, max_value=100, allow_nan=False),
        n_periods=st.integers(min_value=1, max_value=10),
    )
    def test_cyclic_encode_periodicity(
        self,
        value: float,
        period: float,
        n_periods: int,
    ) -> None:
        """Property: encoding repeats after period."""
        sin1, cos1 = cyclic_encode(value, period)
        sin2, cos2 = cyclic_encode(value + n_periods * period, period)

        # Should be identical after full periods
        assert np.abs(sin1 - sin2) < 1e-10
        assert np.abs(cos1 - cos2) < 1e-10

    @given(
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
    )
    def test_time_encoding_continuity(self, hour: int, minute: int) -> None:
        """Property: small time changes result in small encoding changes."""
        time1 = hour + minute / 60
        time2 = (hour + (minute + 1) / 60) % 24  # One minute later

        sin1, cos1 = cyclic_encode(time1, 24)
        sin2, cos2 = cyclic_encode(time2, 24)

        # One minute change should result in small encoding change
        # 1 minute = 1/1440 of a day = angle change of 2π/1440 ≈ 0.00436
        # sin/cos change should be approximately this small
        assert np.abs(sin2 - sin1) < 0.1  # Conservative bound
        assert np.abs(cos2 - cos1) < 0.1


class TestTimeToEvent:
    """Test time to event calculation."""

    def test_time_to_event_hours(self) -> None:
        """Test time to event in hours."""
        current = datetime(2024, 1, 1, 12, 0)
        event = datetime(2024, 1, 1, 15, 0)

        result = time_to_event(current, event, "hours")
        assert result == 3.0

    def test_time_to_event_days(self) -> None:
        """Test time to event in days."""
        current = datetime(2024, 1, 1)
        event = datetime(2024, 1, 8)

        result = time_to_event(current, event, "days")
        assert result == 7.0

    def test_time_to_event_minutes(self) -> None:
        """Test time to event in minutes."""
        current = datetime(2024, 1, 1, 12, 0)
        event = datetime(2024, 1, 1, 12, 30)

        result = time_to_event(current, event, "minutes")
        assert result == 30.0

    def test_time_to_event_negative(self) -> None:
        """Test time to past event (negative result)."""
        current = datetime(2024, 1, 1, 15, 0)
        event = datetime(2024, 1, 1, 12, 0)

        result = time_to_event(current, event, "hours")
        assert result == -3.0

    def test_time_to_event_invalid_unit(self) -> None:
        """Test invalid time unit raises error."""
        current = datetime(2024, 1, 1)
        event = datetime(2024, 1, 2)

        with pytest.raises(ValueError, match="Unknown unit"):
            time_to_event(current, event, "weeks")

    @given(
        hours_delta=st.integers(min_value=-168, max_value=168),
        unit=st.sampled_from(["hours", "days", "minutes"]),
    )
    def test_time_to_event_consistency(self, hours_delta: int, unit: str) -> None:
        """Property: time calculations are internally consistent."""
        current = datetime(2024, 1, 1, 12, 0)
        event = current + timedelta(hours=hours_delta)

        result = time_to_event(current, event, unit)

        if unit == "hours":
            assert np.abs(result - hours_delta) < 1e-10
        elif unit == "days":
            assert np.abs(result - hours_delta / 24) < 1e-10
        elif unit == "minutes":
            assert np.abs(result - hours_delta * 60) < 1e-10


class TestValidateTimestamps:
    """Test timestamp validation."""

    def test_validate_timestamps_valid(self) -> None:
        """Test validation of valid timestamps."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Valid sorted timestamps
        ts = pl.Series([100, 200, 300, 400])
        assert validate_timestamps(ts)

    def test_validate_timestamps_with_nulls(self) -> None:
        """Test validation fails with nulls."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        ts = pl.Series([100, None, 300])
        assert not validate_timestamps(ts)

    def test_validate_timestamps_unsorted(self) -> None:
        """Test validation fails with unsorted timestamps."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        ts = pl.Series([300, 100, 200])
        assert not validate_timestamps(ts)

    def test_validate_timestamps_negative(self) -> None:
        """Test validation fails with negative timestamps."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        ts = pl.Series([-100, 0, 100])
        assert not validate_timestamps(ts)

    def test_validate_timestamps_future(self) -> None:
        """Test validation fails with far future timestamps."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Year 2200 in nanoseconds
        future_ts = 7258118400000000000
        ts = pl.Series([100, 200, future_ts])
        assert not validate_timestamps(ts)

    @given(
        timestamps=st.lists(
            st.integers(min_value=0, max_value=4102444800000000000),
            min_size=1,
            max_size=100,
            unique=True,
        ),
    )
    def test_validate_timestamps_property(self, timestamps: list[int]) -> None:
        """Property: sorted unique timestamps should be valid."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        sorted_ts = sorted(timestamps)
        ts = pl.Series(sorted_ts)
        assert validate_timestamps(ts)


class TestAlignTimeseries:
    """Test timeseries alignment."""

    def test_align_timeseries_inner(self) -> None:
        """Test inner join alignment."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        df1 = pl.DataFrame({
            "timestamp": [100, 200, 300],
            "value1": [1, 2, 3],
        })

        df2 = pl.DataFrame({
            "timestamp": [200, 300, 400],
            "value2": [4, 5, 6],
        })

        aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "inner")

        # Should only have common timestamps (200, 300)
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        assert aligned1["timestamp"].to_list() == [200, 300]
        assert aligned2["timestamp"].to_list() == [200, 300]

    def test_align_timeseries_left(self) -> None:
        """Test left join alignment."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        df1 = pl.DataFrame({
            "timestamp": [100, 200, 300],
            "value1": [1, 2, 3],
        })

        df2 = pl.DataFrame({
            "timestamp": [200, 300, 400],
            "value2": [4, 5, 6],
        })

        aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "left")

        # df1 should keep all its timestamps
        assert len(aligned1) == 3
        assert aligned1["timestamp"].to_list() == [100, 200, 300]

        # df2 should only have matching timestamps
        assert len(aligned2) == 2
        assert aligned2["timestamp"].to_list() == [200, 300]

    def test_align_timeseries_outer(self) -> None:
        """Test outer join alignment."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        df1 = pl.DataFrame({
            "timestamp": [100, 200],
            "value1": [1, 2],
        })

        df2 = pl.DataFrame({
            "timestamp": [200, 300],
            "value2": [3, 4],
        })

        aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "outer")

        # Both should have union of timestamps

        # df1 should have its original data for 100, 200
        assert len(aligned1) == 2
        assert set(aligned1["timestamp"].to_list()) == {100, 200}

        # df2 should have its original data for 200, 300
        assert len(aligned2) == 2
        assert set(aligned2["timestamp"].to_list()) == {200, 300}

    @given(
        n_common=st.integers(min_value=1, max_value=20),
        n_unique1=st.integers(min_value=0, max_value=10),
        n_unique2=st.integers(min_value=0, max_value=10),
    )
    def test_align_timeseries_property(
        self,
        n_common: int,
        n_unique1: int,
        n_unique2: int,
    ) -> None:
        """Property: alignment preserves data integrity."""
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Create timestamps
        common_ts = list(range(0, n_common * 100, 100))
        unique1_ts = list(range(10000, 10000 + n_unique1 * 100, 100))
        unique2_ts = list(range(20000, 20000 + n_unique2 * 100, 100))

        df1 = pl.DataFrame({
            "timestamp": common_ts + unique1_ts,
            "value1": list(range(n_common + n_unique1)),
        })

        df2 = pl.DataFrame({
            "timestamp": common_ts + unique2_ts,
            "value2": list(range(n_common + n_unique2)),
        })

        # Inner join should only have common timestamps
        aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "inner")
        assert len(aligned1) == n_common
        assert len(aligned2) == n_common

        # Left join should preserve all df1 timestamps
        aligned1, aligned2 = align_timeseries(df1, df2, "timestamp", "left")
        assert len(aligned1) == n_common + n_unique1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
