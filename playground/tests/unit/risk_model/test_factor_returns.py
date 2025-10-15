"""Unit tests for factor return calculation."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from playground.exposure.factor_exposure import prepare_factor_returns


class TestPrepareFactorReturns:
    """Test suite for prepare_factor_returns function."""

    @pytest.fixture
    def sample_factor_data(self) -> pl.DataFrame:
        """Create sample factor level data."""
        dates = [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "factor_duration": [2.0, 2.1, 2.05, 2.15, 2.2, 2.3, 2.25, 2.4, 2.35, 2.5],
                "factor_credit": [3.0, 3.2, 3.1, 3.3, 3.5, 3.4, 3.6, 3.7, 3.8, 3.9],
                "factor_liquidity": [1.0, 1.1, 1.05, 1.15, 1.2, 1.3, 1.25, 1.4, 1.35, 1.5],
            },
        )

    def test_difference_method(self, sample_factor_data: pl.DataFrame) -> None:
        """Test additive return calculation."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration", "factor_credit", "factor_liquidity"],
            method="difference",
        )

        assert "timestamp" in result.columns
        assert "factor_duration" in result.columns
        assert result.height == 9  # One less due to diff()

        # Check first return is difference
        first_return = result["factor_duration"][0]
        expected = 2.1 - 2.0
        assert abs(first_return - expected) < 1e-10

    def test_difference_method_single_column(self, sample_factor_data: pl.DataFrame) -> None:
        """Test additive returns for single column."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration"],
            method="difference",
        )

        assert result.height == 9
        returns_list = result["factor_duration"].to_list()

        # Verify all differences manually
        expected_returns = [
            2.1 - 2.0,
            2.05 - 2.1,
            2.15 - 2.05,
            2.2 - 2.15,
            2.3 - 2.2,
            2.25 - 2.3,
            2.4 - 2.25,
            2.35 - 2.4,
            2.5 - 2.35,
        ]
        for actual, expected in zip(returns_list, expected_returns):
            assert abs(actual - expected) < 1e-10

    def test_pct_change_method(self, sample_factor_data: pl.DataFrame) -> None:
        """Test multiplicative return calculation."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration"],
            method="pct_change",
        )

        first_return = result["factor_duration"][0]
        expected = (2.1 - 2.0) / 2.0
        assert abs(first_return - expected) < 1e-10

    def test_pct_change_method_multiple_columns(self, sample_factor_data: pl.DataFrame) -> None:
        """Test pct_change for multiple columns."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration", "factor_credit"],
            method="pct_change",
        )

        # Check factor_credit first return
        credit_return = result["factor_credit"][0]
        expected_credit = (3.2 - 3.0) / 3.0
        assert abs(credit_return - expected_credit) < 1e-10

    def test_winsorization(self) -> None:
        """Test winsorization of extreme outliers."""
        # Create dataset with 100 points to ensure winsorization works
        # Pattern: mostly 0.0 changes with one extreme outlier
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(101)],
                "factor_duration": [1.0] * 50 + [100.0] + [1.0] * 50,  # One extreme outlier
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
            winsorize_percentile=0.99,
        )

        # The extreme jump of 99.0 should be winsorized
        # With 100 observations after diff(), 99th percentile should clip the outlier
        max_return = float(result["factor_duration"].max())  # type: ignore[arg-type]
        # 99th percentile of mostly 0.0 values with one 99.0 should be close to 0
        assert max_return < 5.0  # Should be clipped to near-zero

    def test_winsorization_small_dataset(self) -> None:
        """Test that winsorization is skipped for small datasets."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 6)],
                "factor_duration": [1.0, 1.0, 100.0, 1.0, 1.0],  # Outlier in small dataset
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
            winsorize_percentile=0.99,
        )

        # Outlier should NOT be winsorized (dataset too small: n=4 after diff)
        # The extreme value of 99.0 should remain
        max_return = float(result["factor_duration"].max())  # type: ignore[arg-type]
        assert max_return > 90.0  # Should NOT be clipped

    def test_winsorization_disabled(self) -> None:
        """Test that winsorization can be disabled."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 21)],
                "factor_duration": [1.0] * 10 + [100.0] + [1.0] * 9,
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
            winsorize_percentile=None,  # Disable winsorization
        )

        # Extreme value should NOT be capped
        max_return = float(result["factor_duration"].max())  # type: ignore[arg-type]
        assert max_return > 90.0  # Should NOT be clipped

    def test_inf_handling_pct_change(self) -> None:
        """Test handling of infinite values from pct_change."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 4)],
                "factor_duration": [0.0, 1.0, 0.0],  # Will create inf with pct_change
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="pct_change",
        )

        # Should not contain inf
        assert result["factor_duration"].is_infinite().sum() == 0

        # Verify replacement values
        returns_list = result["factor_duration"].to_list()
        # First return: (1.0 - 0.0) / 0.0 = inf -> replaced with 10.0
        # Second return: (0.0 - 1.0) / 1.0 = -1.0 (valid)
        assert returns_list[0] == 10.0  # Positive infinity replaced
        assert abs(returns_list[1] - (-1.0)) < 1e-10

    def test_negative_inf_handling(self) -> None:
        """Test handling of negative infinite values."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 4)],
                "factor_duration": [0.0, -1.0, 0.0],  # Will create -inf with pct_change
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="pct_change",
        )

        # First return: (-1.0 - 0.0) / 0.0 = -inf -> replaced with -10.0
        assert result["factor_duration"][0] == -10.0

    def test_empty_dataframe(self) -> None:
        """Test handling of empty input."""
        data = pl.DataFrame(
            {
                "timestamp": [],
                "factor_duration": [],
            },
            schema={"timestamp": pl.Datetime, "factor_duration": pl.Float64},
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        assert result.is_empty()

    def test_single_row_dataframe(self) -> None:
        """Test handling of single row (no returns possible)."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "factor_duration": [1.0],
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        # After diff(), first row is null and gets dropped
        assert result.is_empty()

    def test_missing_columns(self) -> None:
        """Test error handling for missing columns."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "factor_duration": [1.0],
            },
        )

        with pytest.raises(Exception):  # Should raise column not found
            prepare_factor_returns(
                data,
                columns=["factor_nonexistent"],
                method="difference",
            )

    def test_invalid_method(self, sample_factor_data: pl.DataFrame) -> None:
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            prepare_factor_returns(
                sample_factor_data,
                columns=["factor_duration"],
                method="invalid",
            )

    def test_null_values_in_input(self) -> None:
        """Test handling of null values in input data."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 6)],
                "factor_duration": [1.0, 1.1, None, 1.3, 1.4],
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        # Nulls should be dropped
        assert result["factor_duration"].null_count() == 0
        # After diff() and dropping nulls, we should have 2 valid returns
        # (1.1-1.0, 1.4-1.3)
        assert result.height == 2

    def test_timestamp_sorting(self) -> None:
        """Test that timestamps are sorted before differencing."""
        # Create unsorted data
        data = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 3, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2020, 1, 2, tzinfo=UTC),
                ],
                "factor_duration": [3.0, 1.0, 2.0],
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        # After sorting by timestamp, differences should be: 2.0-1.0=1.0, 3.0-2.0=1.0
        returns_list = result["factor_duration"].to_list()
        assert abs(returns_list[0] - 1.0) < 1e-10
        assert abs(returns_list[1] - 1.0) < 1e-10

    def test_preserves_timestamp_column(self, sample_factor_data: pl.DataFrame) -> None:
        """Test that timestamp column is preserved in output."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration"],
            method="difference",
        )

        assert "timestamp" in result.columns
        assert result["timestamp"].dtype == pl.Datetime

    def test_multiple_factors_simultaneously(self, sample_factor_data: pl.DataFrame) -> None:
        """Test computing returns for multiple factors at once."""
        result = prepare_factor_returns(
            sample_factor_data,
            columns=["factor_duration", "factor_credit", "factor_liquidity"],
            method="difference",
        )

        # All factor columns should be present
        assert "factor_duration" in result.columns
        assert "factor_credit" in result.columns
        assert "factor_liquidity" in result.columns

        # All should have same number of rows
        assert result.height == 9

        # Check one value from each
        assert abs(result["factor_duration"][0] - 0.1) < 1e-10
        assert abs(result["factor_credit"][0] - 0.2) < 1e-10
        assert abs(result["factor_liquidity"][0] - 0.1) < 1e-10

    @given(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100,
        ),
    )
    def test_returns_always_finite(self, values: list[float]) -> None:
        """Property test: returns should always be finite after processing."""
        dates = [
            datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(values))
        ]
        data = pl.DataFrame(
            {
                "timestamp": dates,
                "factor_duration": values,
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        # All returns should be finite
        if not result.is_empty():
            assert result["factor_duration"].is_finite().all()

    @given(
        st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100,
        ),
    )
    def test_returns_shape_invariant(self, values: list[float]) -> None:
        """Property test: output has one less row than input (due to diff)."""
        dates = [
            datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(values))
        ]
        data = pl.DataFrame(
            {
                "timestamp": dates,
                "factor_duration": values,
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
        )

        # After diff and dropping nulls, should have n-1 rows (or 0 if n=1)
        if len(values) > 1:
            assert result.height == len(values) - 1
        else:
            assert result.is_empty()

    @given(
        st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50,
        ),
    )
    def test_pct_change_finite_positive(self, values: list[float]) -> None:
        """Property test: pct_change is finite for positive values."""
        dates = [
            datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(values))
        ]
        data = pl.DataFrame(
            {
                "timestamp": dates,
                "factor_duration": values,
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="pct_change",
        )

        # All returns should be finite when base values are positive
        if not result.is_empty():
            assert result["factor_duration"].is_finite().all()

    def test_winsorization_symmetric(self) -> None:
        """Test that winsorization is symmetric for positive and negative outliers."""
        # Create larger dataset (201 points) with symmetric outliers
        # Pattern: mostly small changes with two extreme outliers
        baseline_values = [1.0 + 0.01 * i for i in range(98)]  # 98 values: small increments
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(201)],
                "factor_duration": (
                    baseline_values  # 98 values
                    + [100.0]  # 1 value: positive outlier
                    + baseline_values[::-1]  # 98 values: reverse
                    + [-100.0]  # 1 value: negative outlier
                    + [1.0] * 3  # 3 values to reach 201 total
                ),
            },
        )

        result = prepare_factor_returns(
            data,
            columns=["factor_duration"],
            method="difference",
            winsorize_percentile=0.99,
        )

        max_return = float(result["factor_duration"].max())  # type: ignore[arg-type]
        min_return = float(result["factor_duration"].min())  # type: ignore[arg-type]

        # With 200 returns after diff(), 99th percentile should clip extreme outliers
        # Most returns are ~0.01, so outliers of 98+ and -98+ should be clipped
        assert max_return < 10.0  # Should be clipped
        assert min_return > -10.0  # Should be clipped

    def test_iterable_columns_parameter(self, sample_factor_data: pl.DataFrame) -> None:
        """Test that columns parameter accepts any iterable."""
        # Test with tuple
        result_tuple = prepare_factor_returns(
            sample_factor_data,
            columns=("factor_duration", "factor_credit"),
            method="difference",
        )
        assert result_tuple.height == 9

        # Test with set (order may vary)
        result_set = prepare_factor_returns(
            sample_factor_data,
            columns={"factor_duration", "factor_credit"},
            method="difference",
        )
        assert result_set.height == 9

        # Test with generator
        result_gen = prepare_factor_returns(
            sample_factor_data,
            columns=(col for col in ["factor_duration", "factor_credit"]),
            method="difference",
        )
        assert result_gen.height == 9
