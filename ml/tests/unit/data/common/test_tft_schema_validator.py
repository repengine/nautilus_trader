"""
Unit tests for TFTSchemaValidatorComponent.

This module contains 18 tests as specified in the Phase 2.6.4 test design:
- Happy path tests (D1-D5)
- Error condition tests (D6-D11)
- Edge case tests (D12-D14)
- Property tests (D15-D16)
- Contract tests (D17-D18)

Test Design Reference: reports/tests/phase_2_6_tft_dataset_builder_decomposition_test_design.md

"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


from ml.data.common.tft_schema_validator import (
    SchemaValidationError,
    TFTSchemaValidatorComponent,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def validator() -> TFTSchemaValidatorComponent:
    """
    Fixture providing a fresh TFTSchemaValidatorComponent instance.
    """
    return TFTSchemaValidatorComponent()


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """
    Sample Polars DataFrame with OHLCV data for schema validation tests.
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
    ).cast(
        {
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


@pytest.fixture
def sample_ohlcv_pandas_df(sample_ohlcv_polars_df: pl.DataFrame) -> pd.DataFrame:
    """
    Sample Pandas DataFrame matching the Polars fixture.
    """
    return sample_ohlcv_polars_df.to_pandas()


# ============================================================================
# Happy Path Tests (D1-D5)
# ============================================================================


class TestHappyPath:
    """
    Happy path tests for TFTSchemaValidatorComponent.
    """

    def test_validate_required_columns_present(
        self,
        validator: TFTSchemaValidatorComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        D1. Verify validation passes when all required columns present.

        Tests that validate() does not raise when DataFrame contains timestamp, close,
        open, high, low, volume columns.

        """
        # Should not raise
        validator.validate(sample_ohlcv_polars_df)

        # Also test with Pandas
        pandas_df = sample_ohlcv_polars_df.to_pandas()
        validator.validate(pandas_df)

    def test_validate_timestamp_column_types(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D2. Verify timestamp column has correct type.

        Validation passes for Datetime and Int64 (nanoseconds) timestamp types.

        """
        # Test with Int64 timestamp
        df_int64 = pl.DataFrame(
            {
                "timestamp": [1000000000, 2000000000, 3000000000],
                "close": [100.0, 101.0, 102.0],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})

        validator.validate_column_types(df_int64)

        # Test with Datetime timestamp
        df_datetime = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                ],
                "close": [100.0, 101.0, 102.0],
            }
        )

        validator.validate_column_types(df_datetime)

    def test_validate_numeric_columns(
        self,
        validator: TFTSchemaValidatorComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        D3. Verify OHLCV columns are numeric.

        Validation passes for Float32, Float64, Int64 OHLCV columns.

        """
        # Test with Float64 (default)
        validator.validate_column_types(sample_ohlcv_polars_df)

        # Test with Float32
        df_float32 = sample_ohlcv_polars_df.cast(
            {
                "open": pl.Float32,
                "high": pl.Float32,
                "low": pl.Float32,
                "close": pl.Float32,
                "volume": pl.Float32,
            }
        )
        validator.validate_column_types(df_float32)

        # Test with Int64
        df_int64 = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": [100, 101, 102],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Int64})
        validator.validate_column_types(df_int64)

    def test_validate_data_shape_minimum_rows(
        self,
        validator: TFTSchemaValidatorComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        D4. Verify minimum row count check.

        Validation passes when rows >= minimum.

        """
        # Sample has 100 rows, minimum=50 should pass
        validator.validate_row_count(sample_ohlcv_polars_df, minimum=50)

        # Minimum=100 should also pass (exactly at boundary)
        validator.validate_row_count(sample_ohlcv_polars_df, minimum=100)

        # Also test with Pandas
        pandas_df = sample_ohlcv_polars_df.to_pandas()
        validator.validate_row_count(pandas_df, minimum=50)

    def test_validate_instrument_id_column(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D5. Verify instrument_id column is string/categorical.

        Validation passes when instrument_id is Utf8/Categorical type.
        Note: This is validated via OHLCV numeric check - instrument_id
        should NOT trigger numeric type validation since it's not in
        NUMERIC_COLUMNS list.

        """
        df = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
                "instrument_id": ["SPY", "SPY", "SPY"],
            }
        ).cast(
            {
                "timestamp": pl.Int64,
                "close": pl.Float64,
                "instrument_id": pl.Utf8,
            }
        )

        # instrument_id should not cause validation errors
        # (it's not in NUMERIC_COLUMNS)
        validator.validate(df)

        # Test with Categorical type
        df_categorical = df.with_columns(
            pl.col("instrument_id").cast(pl.Categorical),
        )
        validator.validate(df_categorical)


# ============================================================================
# Error Condition Tests (D6-D11)
# ============================================================================


class TestErrorConditions:
    """
    Error condition tests for TFTSchemaValidatorComponent.
    """

    def test_validate_missing_timestamp_column(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D6. Verify error when timestamp column missing.

        Should raise SchemaValidationError with 'timestamp' in message.

        """
        df_no_timestamp = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

        with pytest.raises(SchemaValidationError, match="timestamp"):
            validator.validate(df_no_timestamp)

    def test_validate_missing_close_column(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D7. Verify error when close column missing.

        Should raise SchemaValidationError with 'close' in message.

        """
        df_no_close = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
            }
        )

        with pytest.raises(SchemaValidationError, match="close"):
            validator.validate(df_no_close)

    def test_validate_wrong_column_type(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D8. Verify error when column has wrong type.

        Should raise SchemaValidationError with 'type' in message.

        """
        # close column as string type
        df_wrong_type = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": ["100", "101", "102"],  # String instead of numeric
            }
        ).cast(
            {
                "timestamp": pl.Int64,
                "close": pl.Utf8,
            }
        )

        with pytest.raises(SchemaValidationError, match="type"):
            validator.validate_column_types(df_wrong_type)

    def test_validate_insufficient_rows(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D9. Verify error when row count below minimum.

        Should raise SchemaValidationError with 'rows' in message.

        """
        df_small = pl.DataFrame(
            {
                "timestamp": list(range(10)),
                "close": [100.0 + i for i in range(10)],
            }
        )

        with pytest.raises(SchemaValidationError, match="row"):
            validator.validate_row_count(df_small, minimum=50)

    def test_validate_nan_in_required_column(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D10. Verify error when required column contains NaN/null.

        Should raise SchemaValidationError with column name in message.

        """
        df_with_null = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": [100.0, None, 102.0],  # Null in required column
            }
        )

        with pytest.raises(SchemaValidationError, match="close"):
            validator.validate_no_nulls(df_with_null, ["close"])

        # Also test validate() catches nulls
        with pytest.raises(SchemaValidationError, match="null"):
            validator.validate(df_with_null)

    def test_validate_negative_prices(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D11. Verify error when OHLCV values are negative.

        Should raise SchemaValidationError with 'negative' in message.

        """
        df_negative = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": [100.0, -50.0, 102.0],  # Negative price
            }
        )

        with pytest.raises(SchemaValidationError, match="negative"):
            validator.validate(df_negative)


# ============================================================================
# Edge Case Tests (D12-D14)
# ============================================================================


class TestEdgeCases:
    """
    Edge case tests for TFTSchemaValidatorComponent.
    """

    def test_validate_exactly_minimum_rows(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D12. Verify validation passes at exact minimum.

        No exception should be raised at the boundary.

        """
        df = pl.DataFrame(
            {
                "timestamp": list(range(50)),
                "close": [100.0 + i * 0.1 for i in range(50)],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})

        # Exactly at minimum should pass
        validator.validate_row_count(df, minimum=50)

        # One less should fail
        df_small = df.head(49)
        with pytest.raises(SchemaValidationError):
            validator.validate_row_count(df_small, minimum=50)

    def test_validate_optional_columns_absent(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D13. Verify validation passes when optional columns absent.

        Only timestamp and close are required; open, high, low, volume are optional.

        """
        # DataFrame with only required columns
        df_minimal = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})

        # Should pass validation - optional columns are not required
        validator.validate(df_minimal)

    def test_validate_extra_columns_present(
        self,
        validator: TFTSchemaValidatorComponent,
        sample_ohlcv_polars_df: pl.DataFrame,
    ) -> None:
        """
        D14. Verify validation passes with extra columns.

        Extra columns beyond the schema should be ignored.

        """
        # Add extra columns
        df_extra = sample_ohlcv_polars_df.with_columns(
            [
                pl.lit("extra_value").alias("extra_col1"),
                pl.lit(999).alias("extra_col2"),
                pl.lit(True).alias("extra_col3"),
            ]
        )

        # Should pass - extra columns are ignored
        validator.validate(df_extra)


# ============================================================================
# Property Tests (D15-D16)
# ============================================================================


class TestPropertyBased:
    """
    Property-based tests for TFTSchemaValidatorComponent.
    """

    @given(
        n_rows=st.integers(min_value=1, max_value=100),
        base_price=st.floats(
            min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_validation_deterministic(
        self,
        n_rows: int,
        base_price: float,
    ) -> None:
        """
        D15. Property: validation result is deterministic.

        validate(df) should produce the same result when called multiple times.

        """
        validator = TFTSchemaValidatorComponent()

        # Create a valid DataFrame
        df = pl.DataFrame(
            {
                "timestamp": list(range(n_rows)),
                "close": [base_price + i * 0.1 for i in range(n_rows)],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})

        # Multiple calls should have the same result (no exception)
        validator.validate(df)
        validator.validate(df)
        validator.validate(df)

        # Also test with invalid DataFrame - should consistently raise
        df_invalid = pl.DataFrame(
            {
                "close": [base_price],  # Missing timestamp
            }
        )

        raised_count = 0
        for _ in range(3):
            try:
                validator.validate(df_invalid)
            except SchemaValidationError:
                raised_count += 1

        assert raised_count == 3, "Validation should consistently raise for invalid input"

    @given(
        n_rows=st.integers(min_value=10, max_value=50),
        prices=st.lists(
            st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_valid_data_always_passes(
        self,
        n_rows: int,
        prices: list[float],
    ) -> None:
        """
        D16. Property: data meeting all criteria always passes.

        DataFrames with all required columns, correct types, and no NaN should always
        pass validation.

        """
        # Use the smaller of n_rows and len(prices)
        actual_size = min(n_rows, len(prices))
        assume(actual_size >= 1)

        validator = TFTSchemaValidatorComponent()

        df = pl.DataFrame(
            {
                "timestamp": list(range(actual_size)),
                "close": prices[:actual_size],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})

        # Should always pass
        validator.validate(df)


# ============================================================================
# Contract Tests (D17-D18)
# ============================================================================


class TestContracts:
    """
    Contract tests for TFTSchemaValidatorComponent.
    """

    def test_contract_required_columns_list(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D17. Document required columns via test.

        Verifies that the REQUIRED_COLUMNS class attribute matches the documented
        requirements.

        """
        # Required columns should be exactly ['timestamp', 'close']
        expected_required = ["timestamp", "close"]
        assert validator.REQUIRED_COLUMNS == expected_required, (
            f"REQUIRED_COLUMNS should be {expected_required}, " f"got {validator.REQUIRED_COLUMNS}"
        )

        # Optional columns should be exactly ['open', 'high', 'low', 'volume']
        expected_optional = ["open", "high", "low", "volume"]
        assert validator.OPTIONAL_COLUMNS == expected_optional, (
            f"OPTIONAL_COLUMNS should be {expected_optional}, " f"got {validator.OPTIONAL_COLUMNS}"
        )

        # Timestamp columns should include 'timestamp' and 'ts_event'
        expected_ts = ["timestamp", "ts_event"]
        assert validator.TIMESTAMP_COLUMNS == expected_ts, (
            f"TIMESTAMP_COLUMNS should be {expected_ts}, " f"got {validator.TIMESTAMP_COLUMNS}"
        )

    def test_contract_error_message_format(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        D18. Error messages are user-friendly.

        Error messages should include the column name and a description of the specific
        issue.

        """
        # Test missing timestamp
        df_no_ts = pl.DataFrame({"close": [100.0]})
        try:
            validator.validate(df_no_ts)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            error_msg = str(e)
            assert (
                "timestamp" in error_msg.lower()
            ), f"Error message should mention 'timestamp': {error_msg}"

        # Test missing close
        df_no_close = pl.DataFrame({"timestamp": [1]})
        try:
            validator.validate(df_no_close)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            error_msg = str(e)
            assert (
                "close" in error_msg.lower()
            ), f"Error message should mention 'close': {error_msg}"

        # Test wrong type
        df_wrong_type = pl.DataFrame(
            {
                "timestamp": [1],
                "close": ["not_a_number"],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Utf8})
        try:
            validator.validate(df_wrong_type)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            error_msg = str(e)
            assert (
                "close" in error_msg.lower()
            ), f"Error message should mention column name: {error_msg}"
            assert "type" in error_msg.lower(), f"Error message should mention 'type': {error_msg}"

        # Test negative values
        df_negative = pl.DataFrame(
            {
                "timestamp": [1],
                "close": [-100.0],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})
        try:
            validator.validate(df_negative)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            error_msg = str(e)
            assert (
                "close" in error_msg.lower()
            ), f"Error message should mention column name: {error_msg}"
            assert (
                "negative" in error_msg.lower()
            ), f"Error message should mention 'negative': {error_msg}"

        # Test insufficient rows
        df_small = pl.DataFrame(
            {
                "timestamp": [1],
                "close": [100.0],
            }
        ).cast({"timestamp": pl.Int64, "close": pl.Float64})
        try:
            validator.validate_row_count(df_small, minimum=100)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            error_msg = str(e)
            assert "1" in error_msg, f"Error message should mention actual row count: {error_msg}"
            assert "100" in error_msg, f"Error message should mention minimum required: {error_msg}"


# ============================================================================
# Additional Tests - ts_event fallback
# ============================================================================


class TestTimestampFallback:
    """
    Additional tests for ts_event timestamp fallback.
    """

    def test_validate_with_ts_event_column(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        Verify validation passes with ts_event instead of timestamp.
        """
        df = pl.DataFrame(
            {
                "ts_event": [1000000000, 2000000000, 3000000000],
                "close": [100.0, 101.0, 102.0],
            }
        ).cast({"ts_event": pl.Int64, "close": pl.Float64})

        # Should pass - ts_event is a valid timestamp column
        validator.validate(df)

    def test_validate_prefers_timestamp_over_ts_event(
        self,
        validator: TFTSchemaValidatorComponent,
    ) -> None:
        """
        Verify that 'timestamp' is preferred over 'ts_event' when both exist.
        """
        df = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "ts_event": [4, 5, 6],
                "close": [100.0, 101.0, 102.0],
            }
        ).cast({"timestamp": pl.Int64, "ts_event": pl.Int64, "close": pl.Float64})

        # Should pass with both columns
        validator.validate(df)

        # The internal method should find 'timestamp' first
        ts_col = validator._find_timestamp_column(df)
        assert ts_col == "timestamp", "Should prefer 'timestamp' over 'ts_event'"
