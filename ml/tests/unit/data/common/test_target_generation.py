"""
Unit tests for TargetGenerationComponent.

This module contains 22 tests as specified in the Phase 2.6.3 test design:
- Happy path tests (C1-C7)
- Error condition tests (C8-C12)
- Edge case tests (C13-C15)
- Property tests (C16-C19)
- Metamorphic tests (C20-C21)
- Contract tests (C22)

Test Design Reference: reports/tests/phase_2_6_tft_dataset_builder_decomposition_test_design.md

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


from ml.data.common.target_generation import TargetGenerationComponent


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def component() -> TargetGenerationComponent:
    """
    Fixture providing a fresh TargetGenerationComponent instance.
    """
    return TargetGenerationComponent()


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Sample Polars DataFrame with OHLCV data for target generation tests.
    """
    np.random.seed(42)
    n_rows = 100
    base_price = 100.0
    # Generate realistic price series with slight trend
    returns = np.random.normal(0.0001, 0.01, n_rows)
    prices = base_price * np.cumprod(1 + returns)

    return pl.DataFrame(
        {
            "timestamp": list(range(n_rows)),
            "open": prices * (1 - np.random.uniform(0, 0.001, n_rows)),
            "high": prices * (1 + np.random.uniform(0, 0.005, n_rows)),
            "low": prices * (1 - np.random.uniform(0, 0.005, n_rows)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_rows),
        }
    )


@pytest.fixture
def sample_ohlcv_pandas_df(sample_ohlcv_polars_df: pl.DataFrame) -> pd.DataFrame:
    """
    Sample Pandas DataFrame matching the Polars fixture.
    """
    return sample_ohlcv_polars_df.to_pandas()


# ============================================================================
# Happy Path Tests (C1-C7)
# ============================================================================


class TestHappyPath:
    """
    Happy path tests for TargetGenerationComponent.
    """

    def test_generate_targets_polars_basic(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C1. Verify binary target generation with Polars.

        Tests that generate_targets_polars returns DataFrame with 'y' (binary) and
        'forward_return' columns with correct types and values.

        """
        targets = component.generate_targets_polars(
            sample_ohlcv_polars_df,
            horizon_minutes=15,
            threshold=0.001,
        )

        # Check columns exist
        assert "y" in targets.columns, "Missing 'y' column"
        assert "forward_return" in targets.columns, "Missing 'forward_return' column"

        # Check y only contains 0 and 1
        unique_y = set(targets["y"].unique().to_list())
        assert unique_y <= {0, 1}, f"y should only contain 0 or 1, got {unique_y}"

        # Check forward_return is float
        assert targets["forward_return"].dtype == pl.Float32

        # Check y is Int32
        assert targets["y"].dtype == pl.Int32

    def test_generate_targets_pandas_basic(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_pandas_df: pd.DataFrame,
    ) -> None:
        """
        C2. Verify binary target generation with Pandas.

        Same test as C1 but using Pandas implementation.

        """
        targets = component.generate_targets_pandas(
            sample_ohlcv_pandas_df,
            horizon_minutes=15,
            threshold=0.001,
        )

        # Check columns exist
        assert "y" in targets.columns, "Missing 'y' column"
        assert "forward_return" in targets.columns, "Missing 'forward_return' column"

        # Check y only contains 0 and 1
        unique_y = set(targets["y"].unique())
        assert unique_y <= {0, 1}, f"y should only contain 0 or 1, got {unique_y}"

        # Check forward_return is float
        assert targets["forward_return"].dtype == float

    def test_target_calculation_formula(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C3. Verify forward_return = (future_price - current_price) / current_price.

        Uses known prices to verify the exact calculation.
        """
        # Create known price series: price jumps from 100 to 110 at horizon
        prices = [100.0] * 5 + [110.0]  # 6 rows, horizon=5
        df = pl.DataFrame({"close": prices})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=5,
            threshold=0.05,  # 5% threshold
        )

        # forward_return at t=0: (110-100)/100 = 0.10 (10%)
        expected_return = 0.10
        actual_return = targets["forward_return"][0]

        assert (
            abs(actual_return - expected_return) < 1e-6
        ), f"Expected forward_return {expected_return}, got {actual_return}"

    def test_binary_classification_threshold(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C4. Verify y=1 when forward_return > threshold.

        Tests the binary classification logic at the threshold boundary.

        """
        # Prices that produce exactly 0.2% return
        prices = [100.0, 100.2]  # 0.2% return over horizon=1
        df = pl.DataFrame({"close": prices})

        # With threshold=0.001 (0.1%), return of 0.2% should give y=1
        targets_above = component.generate_targets_polars(
            df,
            horizon_minutes=1,
            threshold=0.001,  # 0.1%
        )
        assert targets_above["y"][0] == 1, "y should be 1 when return > threshold"

        # With threshold=0.003 (0.3%), return of 0.2% should give y=0
        targets_below = component.generate_targets_polars(
            df,
            horizon_minutes=1,
            threshold=0.003,  # 0.3%
        )
        assert targets_below["y"][0] == 0, "y should be 0 when return <= threshold"

    def test_target_nan_filling(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C5. Verify NaN in last `horizon` rows filled with 0.

        The last horizon rows cannot have valid forward-looking targets and should be
        filled with 0.

        """
        horizon = 15
        targets = component.generate_targets_polars(
            sample_ohlcv_polars_df,
            horizon_minutes=horizon,
            threshold=0.001,
        )

        # No NaN in output
        assert targets["y"].is_null().sum() == 0, "y should have no NaN"
        assert targets["forward_return"].is_null().sum() == 0, "forward_return should have no NaN"

        # Last horizon rows should be 0 (filled NaN)
        last_y = targets["y"].tail(horizon).to_list()
        last_return = targets["forward_return"].tail(horizon).to_list()

        assert all(y == 0 for y in last_y), f"Last {horizon} y values should be 0"
        assert all(
            abs(r) < 1e-10 for r in last_return
        ), f"Last {horizon} forward_return values should be 0.0"

    def test_different_horizons(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C6. Verify targets work with various horizons (5, 15, 30, 60 minutes).

        Different horizons should produce different targets.

        """
        horizons = [5, 15, 30]
        results: list[pl.DataFrame] = []

        for horizon in horizons:
            targets = component.generate_targets_polars(
                sample_ohlcv_polars_df,
                horizon_minutes=horizon,
                threshold=0.001,
            )
            results.append(targets)

        # Check that different horizons produce different results
        # (at least some values should differ)
        for i in range(len(results) - 1):
            # Compare non-trailing values (since trailing values are all 0)
            safe_idx = min(5, len(results[i]) - 1)  # First 5 values should differ
            vals1 = results[i]["forward_return"].head(safe_idx).to_list()
            vals2 = results[i + 1]["forward_return"].head(safe_idx).to_list()

            assert (
                vals1 != vals2
            ), f"Horizons {horizons[i]} and {horizons[i+1]} should produce different targets"

    def test_different_thresholds(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C7. Verify thresholds affect binary classification.

        Lower threshold should result in more y=1 labels.

        """
        thresholds = [0.0001, 0.001, 0.01]
        y_counts: list[int] = []

        for threshold in thresholds:
            targets = component.generate_targets_polars(
                sample_ohlcv_polars_df,
                horizon_minutes=15,
                threshold=threshold,
            )
            y_count = int(targets["y"].sum())
            y_counts.append(y_count)

        # Lower threshold should have >= positive labels than higher threshold
        assert (
            y_counts[0] >= y_counts[1] >= y_counts[2]
        ), f"y=1 count should decrease with higher threshold: {y_counts}"


# ============================================================================
# Error Condition Tests (C8-C12)
# ============================================================================


class TestErrorConditions:
    """
    Error condition tests for TargetGenerationComponent.
    """

    def test_generate_targets_missing_close(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C8. Verify error when 'close' column missing.

        Should raise KeyError when required column is absent.

        """
        df_no_close = pl.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
            }
        )

        with pytest.raises(KeyError, match="close"):
            component.generate_targets_polars(df_no_close, horizon_minutes=1, threshold=0.001)

    def test_generate_targets_empty_dataframe(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C9. Verify behavior with empty DataFrame.

        Should return empty DataFrame with correct columns.

        """
        empty_df = pl.DataFrame({"close": []}).cast({"close": pl.Float64})

        targets = component.generate_targets_polars(
            empty_df,
            horizon_minutes=15,
            threshold=0.001,
        )

        assert len(targets) == 0, "Output should be empty"
        assert "y" in targets.columns, "Should have 'y' column"
        assert "forward_return" in targets.columns, "Should have 'forward_return' column"

    def test_generate_targets_horizon_exceeds_data(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C10. Verify behavior when horizon > data length.

        All targets should be 0 (filled NaN).

        """
        df = pl.DataFrame({"close": [100.0] * 10})  # 10 rows

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=20,  # horizon > data length
            threshold=0.001,
        )

        # All y should be 0
        assert targets["y"].sum() == 0, "All y should be 0 when horizon > data length"

        # All forward_return should be 0.0
        assert all(
            abs(r) < 1e-10 for r in targets["forward_return"].to_list()
        ), "All forward_return should be 0.0"

    def test_generate_targets_negative_horizon(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C11. Verify handling of negative/zero horizon (invalid).

        Should raise ValueError for invalid horizon.

        """
        df = pl.DataFrame({"close": [100.0, 101.0]})

        with pytest.raises(ValueError, match="horizon minutes"):
            component.generate_targets_polars(df, horizon_minutes=0, threshold=0.001)

        with pytest.raises(ValueError, match="horizon minutes"):
            component.generate_targets_polars(df, horizon_minutes=-5, threshold=0.001)

    def test_generate_targets_zero_prices(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C12. Verify handling when close=0 (division by zero).

        Inf should be replaced with 0.

        """
        df = pl.DataFrame({"close": [0.0, 100.0, 0.0, 100.0]})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=1,
            threshold=0.001,
        )

        # No inf in output
        assert not targets["forward_return"].is_infinite().any(), "Should have no inf values"

        # No NaN in output
        assert not targets["forward_return"].is_nan().any(), "Should have no NaN values"


# ============================================================================
# Edge Case Tests (C13-C15)
# ============================================================================


class TestEdgeCases:
    """
    Edge case tests for TargetGenerationComponent.
    """

    def test_generate_targets_minimum_data(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C13. Verify targets with exactly horizon+1 rows.

        Only first row should have a valid forward-looking target.

        """
        horizon = 15
        n_rows = horizon + 1  # 16 rows
        prices = [100.0 + i * 0.1 for i in range(n_rows)]
        df = pl.DataFrame({"close": prices})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=horizon,
            threshold=0.001,
        )

        # First row should have calculated target
        assert (
            targets["forward_return"][0] != 0.0 or abs(prices[-1] - prices[0]) < 1e-10
        ), "First row should have calculated target"

        # Remaining horizon rows should have filled values
        assert len(targets) == n_rows

    def test_generate_targets_constant_prices(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C14. Verify targets when all prices equal.

        All forward_return = 0.0 and all y = 0.

        """
        df = pl.DataFrame({"close": [100.0] * 20})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=5,
            threshold=0.001,
        )

        # All returns should be 0
        all_returns = targets["forward_return"].to_list()
        assert all(abs(r) < 1e-10 for r in all_returns), "All returns should be ~0"

        # All classifications should be 0
        assert targets["y"].sum() == 0, "All y should be 0"

    def test_generate_targets_extreme_prices(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C15. Verify handling of extreme price changes.

        Large returns should be calculated correctly without overflow.

        """
        # Extreme jump: 1 to 1,000,000
        df = pl.DataFrame({"close": [1.0, 1000000.0]})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=1,
            threshold=0.001,
        )

        # Return should be very large but finite
        forward_return = targets["forward_return"][0]
        assert np.isfinite(forward_return), "Return should be finite"

        # Expected return: (1000000 - 1) / 1 = 999999
        expected = 999999.0
        assert abs(forward_return - expected) < 1.0, f"Expected ~{expected}, got {forward_return}"

        # Should definitely be classified as positive
        assert targets["y"][0] == 1, "Extreme positive return should have y=1"


# ============================================================================
# Property Tests (C16-C19)
# ============================================================================


class TestPropertyBased:
    """
    Property-based tests for TargetGenerationComponent.
    """

    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
        horizon=st.integers(min_value=1, max_value=20),
        threshold=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_binary_target_values(
        self,
        prices: list[float],
        horizon: int,
        threshold: float,
    ) -> None:
        """
        C16. Property: y column only contains 0 or 1.

        For any valid input, the output y column should only have binary values.

        """
        component = TargetGenerationComponent()
        df = pl.DataFrame({"close": prices})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=horizon,
            threshold=threshold,
        )

        unique_y = set(targets["y"].unique().to_list())
        assert unique_y <= {0, 1}, f"y should only contain 0 or 1, got {unique_y}"

    @given(
        prices=st.lists(
            st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_forward_return_bounded(
        self,
        prices: list[float],
        horizon: int,
    ) -> None:
        """
        C17. Property: forward_return bounded by [-1, max_reasonable].

        Cannot lose more than 100% of investment.

        """
        component = TargetGenerationComponent()
        df = pl.DataFrame({"close": prices})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=horizon,
            threshold=0.001,
        )

        # All forward_return >= -1.0 (can't lose more than 100%)
        min_return = targets["forward_return"].min()
        assert (
            min_return >= -1.0 or abs(min_return + 1.0) < 1e-6
        ), f"forward_return should be >= -1.0, got {min_return}"

    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50,
        ),
        horizon=st.integers(min_value=1, max_value=5),
        threshold=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_polars_pandas_target_parity(
        self,
        prices: list[float],
        horizon: int,
        threshold: float,
    ) -> None:
        """
        C18. Property: Polars and Pandas produce identical targets.

        Both implementations should produce the same output for identical inputs.

        """
        component = TargetGenerationComponent()

        df_polars = pl.DataFrame({"close": prices})
        df_pandas = pd.DataFrame({"close": prices})

        targets_polars = component.generate_targets_polars(
            df_polars,
            horizon_minutes=horizon,
            threshold=threshold,
        )
        targets_pandas = component.generate_targets_pandas(
            df_pandas,
            horizon_minutes=horizon,
            threshold=threshold,
        )

        # Compare y values
        np.testing.assert_array_equal(
            targets_polars["y"].to_numpy(),
            targets_pandas["y"].to_numpy(),
            err_msg="y values should match between Polars and Pandas",
        )

        # Compare forward_return values (with tolerance)
        np.testing.assert_allclose(
            targets_polars["forward_return"].to_numpy(),
            targets_pandas["forward_return"].to_numpy(),
            rtol=1e-6,
            err_msg="forward_return values should match between Polars and Pandas",
        )

    def test_property_no_lookahead_bias(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C19. Property: target at time t only uses data at t+horizon or later.

        Verify that shift direction is negative (forward-looking).

        """
        # Create a price series where we know exact behavior
        prices = list(range(1, 101))  # 1, 2, 3, ..., 100
        df = pl.DataFrame({"close": [float(p) for p in prices]})

        targets = component.generate_targets_polars(
            df,
            horizon_minutes=10,
            threshold=0.0,
        )

        # forward_return at t=0 should be (price[10] - price[0]) / price[0]
        # = (11 - 1) / 1 = 10.0
        expected_return_0 = (prices[10] - prices[0]) / prices[0]
        actual_return_0 = targets["forward_return"][0]

        assert (
            abs(actual_return_0 - expected_return_0) < 1e-6
        ), f"Target at t=0 should use price at t=10. Expected {expected_return_0}, got {actual_return_0}"


# ============================================================================
# Metamorphic Tests (C20-C21)
# ============================================================================


class TestMetamorphic:
    """
    Metamorphic tests for TargetGenerationComponent.
    """

    def test_metamorphic_threshold_monotonicity(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C20. Increasing threshold should not increase count of y=1.

        Higher threshold means stricter criteria, so fewer positive labels.

        """
        thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        y_counts: list[int] = []

        for threshold in thresholds:
            targets = component.generate_targets_polars(
                sample_ohlcv_polars_df,
                horizon_minutes=15,
                threshold=threshold,
            )
            y_counts.append(int(targets["y"].sum()))

        # y_counts should be non-increasing
        for i in range(len(y_counts) - 1):
            assert y_counts[i] >= y_counts[i + 1], (
                f"count(y=1) should be non-increasing with threshold: "
                f"threshold={thresholds[i]} had {y_counts[i]}, "
                f"threshold={thresholds[i+1]} had {y_counts[i+1]}"
            )

    def test_metamorphic_horizon_independence_for_direction(
        self,
        component: TargetGenerationComponent,
    ) -> None:
        """
        C21. Direction of return shouldn't depend on horizon (sign preserved).

        If price goes up, forward_return > 0 regardless of horizon length.

        """
        # Create steadily increasing prices
        prices = [100.0 + i * 1.0 for i in range(50)]  # 100, 101, 102, ...
        df = pl.DataFrame({"close": prices})

        horizons = [5, 10, 15, 20]

        for horizon in horizons:
            targets = component.generate_targets_polars(
                df,
                horizon_minutes=horizon,
                threshold=0.0,
            )

            # All non-trailing forward_returns should be positive
            valid_idx = len(targets) - horizon
            valid_returns = targets["forward_return"].head(valid_idx).to_list()

            for i, ret in enumerate(valid_returns):
                assert ret > 0, (
                    f"With increasing prices, forward_return should be > 0. "
                    f"horizon={horizon}, idx={i}, return={ret}"
                )


# ============================================================================
# Contract Tests (C22)
# ============================================================================


class TestContracts:
    """
    Contract tests for TargetGenerationComponent.
    """

    def test_contract_target_schema(
        self,
        component: TargetGenerationComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        C22. Target output matches defined schema.

        Verifies:
        - y is Int32
        - forward_return is Float32
        - No null values
        - y only contains 0 or 1

        """
        targets = component.generate_targets_polars(
            sample_ohlcv_polars_df,
            horizon_minutes=15,
            threshold=0.001,
        )

        # Type validation
        assert targets["y"].dtype == pl.Int32, f"y should be Int32, got {targets['y'].dtype}"
        assert (
            targets["forward_return"].dtype == pl.Float32
        ), f"forward_return should be Float32, got {targets['forward_return'].dtype}"

        # No nulls
        assert targets["y"].is_null().sum() == 0, "y should have no nulls"
        assert targets["forward_return"].is_null().sum() == 0, "forward_return should have no nulls"

        # Binary constraint
        unique_y = set(targets["y"].unique().to_list())
        assert unique_y <= {0, 1}, f"y should only contain 0 or 1, got {unique_y}"

        # No infinities
        assert (
            not targets["forward_return"].is_infinite().any()
        ), "forward_return should have no inf"
