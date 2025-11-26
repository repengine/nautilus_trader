"""
Tests for TimeSeriesWindowingComponent.

This module contains 24 tests covering:
- Happy path (A1-A8): Basic functionality
- Error conditions (A9-A13): Edge cases and error handling
- Edge cases (A14-A18): Boundary conditions
- Property tests (A19-A22): Invariant testing with Hypothesis
- Contract tests (A23-A24): Schema validation

Test Design Reference: reports/tests/phase_2_6_tft_dataset_builder_decomposition_test_design.md

"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ml.data.common.time_series_windowing import TimeSeriesWindowingComponent


if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def component() -> TimeSeriesWindowingComponent:
    """
    Fixture providing a TimeSeriesWindowingComponent instance.
    """
    return TimeSeriesWindowingComponent()


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Create a sample Polars DataFrame with OHLCV + timestamp columns.

    Contains 100 rows with sorted timestamps spanning one month.

    """
    base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(100)]

    rng = np.random.default_rng(42)

    # Generate realistic OHLCV data
    base_price = 100.0
    prices = base_price + np.cumsum(rng.standard_normal(100) * 0.1)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + rng.uniform(-0.5, 0.5, 100),
            "high": prices + rng.uniform(0, 1, 100),
            "low": prices - rng.uniform(0, 1, 100),
            "close": prices,
            "volume": rng.integers(1000, 10000, 100).astype(float),
        }
    )


@pytest.fixture
def multi_symbol_ohlcv_data() -> dict[str, pl.DataFrame]:
    """
    Create multi-symbol OHLCV data with partially overlapping timestamps.

    SPY: timestamps 0-79
    QQQ: timestamps 20-99
    Overlap: timestamps 20-79

    """
    base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

    rng = np.random.default_rng(42)

    # SPY: first 80 minutes
    spy_timestamps = [base_ts + timedelta(minutes=i) for i in range(80)]
    spy_prices = 100.0 + np.cumsum(rng.standard_normal(80) * 0.1)
    spy_df = pl.DataFrame(
        {
            "timestamp": spy_timestamps,
            "close": spy_prices,
            "volume": rng.integers(1000, 10000, 80).astype(float),
        }
    )

    # QQQ: last 80 minutes (minutes 20-99)
    qqq_timestamps = [base_ts + timedelta(minutes=i) for i in range(20, 100)]
    qqq_prices = 400.0 + np.cumsum(rng.standard_normal(80) * 0.5)
    qqq_df = pl.DataFrame(
        {
            "timestamp": qqq_timestamps,
            "close": qqq_prices,
            "volume": rng.integers(1000, 10000, 80).astype(float),
        }
    )

    return {"SPY": spy_df, "QQQ": qqq_df}


# ============================================================================
# Happy Path Tests (A1-A8)
# ============================================================================


@pytest.mark.unit
class TestFrameTimeBounds:
    """
    Tests for frame_time_bounds method.
    """

    def test_frame_time_bounds_returns_min_max_timestamps(
        self,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A1.

        Verify _frame_time_bounds correctly returns (min, max) timestamp tuple.

        """
        bounds = TimeSeriesWindowingComponent.frame_time_bounds(sample_ohlcv_polars_df)

        # Verify return type
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2

        min_ts, max_ts = bounds

        # Both should be integers (nanoseconds)
        assert isinstance(min_ts, int)
        assert isinstance(max_ts, int)

        # Min should be less than max
        assert min_ts < max_ts

        # Verify against actual DataFrame values
        actual_min = sample_ohlcv_polars_df["timestamp"].min()
        actual_max = sample_ohlcv_polars_df["timestamp"].max()

        expected_min = TimeSeriesWindowingComponent.coerce_to_ns(actual_min)
        expected_max = TimeSeriesWindowingComponent.coerce_to_ns(actual_max)

        assert min_ts == expected_min
        assert max_ts == expected_max

    def test_frame_time_bounds_with_ts_event_column(self) -> None:
        """
        A2.

        Verify fallback to ts_event when timestamp not present.

        """
        # Create DataFrame with ts_event instead of timestamp
        base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        timestamps = [base_ts + timedelta(minutes=i) for i in range(10)]

        df = pl.DataFrame(
            {
                "ts_event": timestamps,
                "close": [100.0 + i for i in range(10)],
            }
        )

        bounds = TimeSeriesWindowingComponent.frame_time_bounds(df)

        assert bounds[0] is not None
        assert bounds[1] is not None
        assert bounds[0] < bounds[1]


@pytest.mark.unit
class TestWindowByTimeRange:
    """
    Tests for window_by_time_range method.
    """

    def test_window_by_time_range(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A3.

        Verify slicing data by start/end datetime.

        """
        base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        # Window for minutes 20-40 (20 minutes)
        start = base_ts + timedelta(minutes=20)
        end = base_ts + timedelta(minutes=40)

        result = component.window_by_time_range(sample_ohlcv_polars_df, start, end)

        # Verify row count
        assert len(result) == 20  # 20 minutes of data

        # Verify all timestamps are within range
        for ts in result["timestamp"].to_list():
            assert ts >= start
            assert ts < end


@pytest.mark.unit
class TestSlidingWindows:
    """
    Tests for create_sliding_windows method.
    """

    def test_sliding_window_creation(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A4.

        Verify creation of overlapping sliding windows for sequence models.

        """
        window_size = 20
        stride = 5

        windows = component.create_sliding_windows(
            sample_ohlcv_polars_df,
            window_size=window_size,
            stride=stride,
        )

        # Expected windows: (100 - 20) / 5 + 1 = 17
        expected_count = (len(sample_ohlcv_polars_df) - window_size) // stride + 1
        assert len(windows) == expected_count

        # Each window should have exactly window_size rows
        for window in windows:
            assert len(window) == window_size

        # Verify overlap: consecutive windows should share (window_size - stride) rows
        if len(windows) >= 2:
            w1_last = windows[0]["timestamp"][-stride:].to_list()
            w2_first = windows[1]["timestamp"][:stride].to_list()
            # The last 'stride' rows of window 1 should NOT be the first 'stride' of window 2
            # Actually, the first 'window_size - stride' of window 2 should overlap with last 'window_size - stride' of window 1
            overlap_size = window_size - stride
            w1_overlap = windows[0]["timestamp"][-overlap_size:].to_list()
            w2_overlap = windows[1]["timestamp"][:overlap_size].to_list()
            assert w1_overlap == w2_overlap


@pytest.mark.unit
class TestAlignMultiSymbol:
    """
    Tests for align_multi_symbol_timestamps method.
    """

    def test_align_multi_symbol_timestamps(
        self,
        component: TimeSeriesWindowingComponent,
        multi_symbol_ohlcv_data: dict[str, pl.DataFrame],
    ) -> None:
        """
        A5.

        Verify alignment of multiple symbols to common timestamp grid.

        """
        aligned = component.align_multi_symbol_timestamps(multi_symbol_ohlcv_data)

        # Both outputs should have identical timestamp sets
        spy_ts = set(aligned["SPY"]["timestamp"].to_list())
        qqq_ts = set(aligned["QQQ"]["timestamp"].to_list())

        assert spy_ts == qqq_ts

        # Should have 60 overlapping timestamps (minutes 20-79)
        assert len(spy_ts) == 60

        # No NaN in timestamp column
        assert aligned["SPY"]["timestamp"].null_count() == 0
        assert aligned["QQQ"]["timestamp"].null_count() == 0


@pytest.mark.unit
class TestCoerceToNs:
    """
    Tests for coerce_to_ns method.
    """

    def test_coerce_to_ns_datetime_input(self) -> None:
        """
        A6.

        Verify coerce_to_ns handles datetime objects.

        """
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = TimeSeriesWindowingComponent.coerce_to_ns(dt)

        assert isinstance(result, int)
        # 2024-01-01 00:00:00 UTC in nanoseconds
        expected = 1704067200_000_000_000
        assert result == expected

    def test_coerce_to_ns_numpy_generic(self) -> None:
        """
        A7.

        Verify coerce_to_ns handles numpy datetime types.

        """
        # Test with numpy datetime64
        np_dt = np.datetime64("2024-01-01T00:00:00", "ns")
        result = TimeSeriesWindowingComponent.coerce_to_ns(np_dt)

        assert isinstance(result, int)
        # numpy datetime64 in nanoseconds
        expected = 1704067200_000_000_000
        assert result == expected

    def test_datetime_to_ns_with_fallback(self) -> None:
        """
        A8.

        Verify datetime_to_ns uses fallback for None input.

        """
        fallback = 1000000000
        result = TimeSeriesWindowingComponent.datetime_to_ns(None, fallback=fallback)

        assert result == fallback


# ============================================================================
# Error Condition Tests (A9-A13)
# ============================================================================


@pytest.mark.unit
class TestErrorConditions:
    """
    Tests for error conditions and edge cases.
    """

    def test_frame_time_bounds_empty_dataframe(self) -> None:
        """
        A9.

        Verify behavior with empty DataFrame.

        """
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "close": [],
            }
        ).cast({"timestamp": pl.Datetime("ns", "UTC")})

        bounds = TimeSeriesWindowingComponent.frame_time_bounds(empty_df)

        assert bounds == (None, None)

    def test_frame_time_bounds_no_timestamp_column(self) -> None:
        """
        A10.

        Verify behavior when neither timestamp nor ts_event exists.

        """
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

        bounds = TimeSeriesWindowingComponent.frame_time_bounds(df)

        assert bounds == (None, None)

    def test_window_by_time_invalid_range(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A11.

        Verify error when start > end.

        """
        start = datetime(2024, 1, 20, tzinfo=UTC)
        end = datetime(2024, 1, 15, tzinfo=UTC)

        with pytest.raises(ValueError, match=r"start.*end"):
            component.window_by_time_range(sample_ohlcv_polars_df, start, end)

    def test_sliding_window_insufficient_data(
        self,
        component: TimeSeriesWindowingComponent,
    ) -> None:
        """
        A12.

        Verify behavior when data smaller than window_size.

        """
        small_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(10)
                ],
                "close": [100.0 + i for i in range(10)],
            }
        )

        windows = component.create_sliding_windows(small_df, window_size=20)

        # Should return empty list when data is smaller than window
        assert windows == []

    def test_coerce_to_ns_invalid_type(self) -> None:
        """
        A13.

        Verify handling of non-convertible types.

        """
        result = TimeSeriesWindowingComponent.coerce_to_ns("not a timestamp")

        assert result is None


# ============================================================================
# Edge Case Tests (A14-A18)
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_frame_time_bounds_single_row(self) -> None:
        """
        A14.

        Verify bounds with single-row DataFrame.

        """
        ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts],
                "close": [100.0],
            }
        )

        min_ts, max_ts = TimeSeriesWindowingComponent.frame_time_bounds(df)

        # Both should be the same for single row
        assert min_ts == max_ts
        assert min_ts is not None
        assert isinstance(min_ts, int)

    def test_window_covering_all_data(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A15.

        Verify window that encompasses entire dataset.

        """
        # Window far before and after data
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 1, tzinfo=UTC)

        result = component.window_by_time_range(sample_ohlcv_polars_df, start, end)

        # All data should be preserved
        assert len(result) == len(sample_ohlcv_polars_df)

    def test_window_outside_data_range(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A16.

        Verify window completely outside data range.

        """
        # Data is from 2024-01-01, window for 2025
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 2, 1, tzinfo=UTC)

        result = component.window_by_time_range(sample_ohlcv_polars_df, start, end)

        # Output should be empty
        assert result.is_empty()
        # But schema should be preserved
        assert result.columns == sample_ohlcv_polars_df.columns

    def test_align_disjoint_timestamps(
        self,
        component: TimeSeriesWindowingComponent,
    ) -> None:
        """
        A17.

        Verify alignment when symbols have no overlapping timestamps.

        """
        base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

        # SPY: first 10 minutes
        spy_df = pl.DataFrame(
            {
                "timestamp": [base_ts + timedelta(minutes=i) for i in range(10)],
                "close": [100.0 + i for i in range(10)],
            }
        )

        # QQQ: minutes 20-30 (no overlap)
        qqq_df = pl.DataFrame(
            {
                "timestamp": [base_ts + timedelta(minutes=i) for i in range(20, 30)],
                "close": [400.0 + i for i in range(10)],
            }
        )

        frames = {"SPY": spy_df, "QQQ": qqq_df}
        aligned = component.align_multi_symbol_timestamps(frames)

        # Both outputs should be empty
        assert aligned["SPY"].is_empty()
        assert aligned["QQQ"].is_empty()

        # Schema should be preserved
        assert aligned["SPY"].columns == spy_df.columns
        assert aligned["QQQ"].columns == qqq_df.columns

    def test_datetime_to_ns_timezone_naive(self) -> None:
        """
        A18.

        Verify handling of timezone-naive datetime.

        """
        # Naive datetime (no tzinfo)
        naive_dt = datetime(2024, 1, 1, 0, 0, 0)

        result = TimeSeriesWindowingComponent.datetime_to_ns(naive_dt, fallback=0)

        # Should assume UTC
        expected = 1704067200_000_000_000
        assert result == expected


# ============================================================================
# Property Tests (A19-A22)
# ============================================================================


@pytest.mark.property
class TestPropertyBased:
    """
    Property-based tests using Hypothesis.
    """

    @given(
        timestamps=st.lists(
            st.integers(min_value=0, max_value=10**18),
            min_size=1,
            max_size=100,
        ),
    )
    @settings(max_examples=50)
    def test_property_timestamp_bounds_within_data(
        self,
        timestamps: list[int],
    ) -> None:
        """
        A19.

        Property: bounds always within or equal to actual data bounds.

        """
        df = pl.DataFrame({"timestamp": timestamps})

        min_bound, max_bound = TimeSeriesWindowingComponent.frame_time_bounds(df)

        actual_min = min(timestamps)
        actual_max = max(timestamps)

        if min_bound is not None:
            assert min_bound >= actual_min
        if max_bound is not None:
            assert max_bound <= actual_max

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        start_offset=st.integers(min_value=0, max_value=50),
        window_len=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_property_window_preserves_ordering(
        self,
        n_rows: int,
        start_offset: int,
        window_len: int,
    ) -> None:
        """
        A20.

        Property: windowed data maintains timestamp ordering.

        """
        assume(start_offset + window_len <= n_rows)

        component = TimeSeriesWindowingComponent()
        base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        timestamps = [base_ts + timedelta(minutes=i) for i in range(n_rows)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0 + i for i in range(n_rows)],
            }
        )

        start = base_ts + timedelta(minutes=start_offset)
        end = base_ts + timedelta(minutes=start_offset + window_len)

        result = component.window_by_time_range(df, start, end)

        if not result.is_empty():
            ts_list = result["timestamp"].to_list()
            # Verify monotonically increasing
            for i in range(len(ts_list) - 1):
                assert ts_list[i] <= ts_list[i + 1]

    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        window_size=st.integers(min_value=2, max_value=20),
        stride=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_property_sliding_windows_cover_data(
        self,
        n_rows: int,
        window_size: int,
        stride: int,
    ) -> None:
        """
        A21.

        Property: windows have correct structure and overlap when stride <= window_size.

        """
        assume(window_size <= n_rows)
        # Only test coverage property when stride ensures overlap
        assume(stride <= window_size)

        component = TimeSeriesWindowingComponent()
        base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        timestamps = [base_ts + timedelta(minutes=i) for i in range(n_rows)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0 + i for i in range(n_rows)],
            }
        )

        windows = component.create_sliding_windows(df, window_size, stride)

        if windows:
            # Each window should have exactly window_size rows
            for window in windows:
                assert len(window) == window_size

            # Window count should match expected formula
            expected_count = (n_rows - window_size) // stride + 1
            assert len(windows) == expected_count

            # Consecutive windows should overlap by (window_size - stride) rows
            if len(windows) >= 2 and stride < window_size:
                overlap_size = window_size - stride
                for i in range(len(windows) - 1):
                    w1_tail = windows[i]["close"][-overlap_size:].to_list()
                    w2_head = windows[i + 1]["close"][:overlap_size].to_list()
                    assert w1_tail == w2_head

    @given(
        dt=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2100, 1, 1),
        ),
    )
    @settings(max_examples=50)
    def test_property_coerce_roundtrip(self, dt: datetime) -> None:
        """
        A22.

        Property: datetime -> ns -> datetime preserves value.

        """
        # Add UTC timezone
        dt_utc = dt.replace(tzinfo=UTC)

        ns = TimeSeriesWindowingComponent.coerce_to_ns(dt_utc)
        assert ns is not None

        # Convert back to datetime
        recovered = datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)

        # Should preserve to microsecond precision
        assert abs((recovered - dt_utc).total_seconds()) < 0.000001


# ============================================================================
# Contract Tests (A23-A24)
# ============================================================================


@pytest.mark.contract
class TestContracts:
    """
    Contract tests for schema validation.
    """

    def test_contract_output_schema_bounds(
        self,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A23.

        Output schema for time bounds must match specification.

        """
        bounds = TimeSeriesWindowingComponent.frame_time_bounds(sample_ohlcv_polars_df)

        # Must be a tuple of exactly 2 elements
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2

        min_ts, max_ts = bounds

        # Each element must be int or None
        assert min_ts is None or isinstance(min_ts, int)
        assert max_ts is None or isinstance(max_ts, int)

        # If both present, min <= max
        if min_ts is not None and max_ts is not None:
            assert min_ts <= max_ts

    def test_contract_output_schema_windowed_data(
        self,
        component: TimeSeriesWindowingComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        A24.

        Windowed DataFrame preserves input schema.

        """
        start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        end = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)

        result = component.window_by_time_range(sample_ohlcv_polars_df, start, end)

        # Column set should be identical
        assert set(result.columns) == set(sample_ohlcv_polars_df.columns)

        # Column types should be preserved
        for col in result.columns:
            assert result[col].dtype == sample_ohlcv_polars_df[col].dtype
