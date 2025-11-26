"""
TFT Schema validation component for TFTDatasetBuilder.

This component provides schema validation specific to TFT (Temporal Fusion
Transformer) datasets, ensuring DataFrames meet the required structure
before processing.

Validation Rules:
1. Required columns: timestamp (or ts_event), close must exist
2. Numeric types: OHLCV columns must be numeric (Float32, Float64, Int64)
3. Timestamp types: Datetime or Int64 (nanoseconds)
4. Minimum rows: Configurable minimum row count
5. No nulls: Required columns should not contain nulls
6. No negative prices: OHLCV values should be non-negative

All methods are COLD path (not performance-critical).

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception
# ============================================================================


class SchemaValidationError(Exception):
    """
    Exception raised when TFT schema validation fails.

    This exception provides detailed information about the validation failure,
    including the column name and specific issue encountered.

    Attributes:
        message: Detailed error message describing the validation failure.
        column: Optional column name that caused the failure.
        issue: Optional description of the specific issue.

    Example:
        >>> raise SchemaValidationError(
        ...     "Missing required column",
        ...     column="timestamp",
        ...     issue="Column not found in DataFrame"
        ... )

    """

    def __init__(
        self,
        message: str,
        *,
        column: str | None = None,
        issue: str | None = None,
    ) -> None:
        """
        Initialize SchemaValidationError.

        Args:
            message: The main error message.
            column: Optional column name related to the error.
            issue: Optional specific issue description.

        """
        self.column = column
        self.issue = issue

        # Build detailed message
        parts = [message]
        if column:
            parts.append(f"column='{column}'")
        if issue:
            parts.append(f"issue='{issue}'")

        full_message = " - ".join(parts)
        super().__init__(full_message)


# ============================================================================
# TFTSchemaValidatorComponent
# ============================================================================


class TFTSchemaValidatorComponent:
    """
    Schema validation component for TFT (Temporal Fusion Transformer) datasets.

    This component validates that DataFrames conform to the expected structure
    for TFT model training. It supports both Polars and Pandas DataFrames.

    Class Attributes:
        REQUIRED_COLUMNS: Columns that must be present in the DataFrame.
        OPTIONAL_COLUMNS: Columns that are optional but validated if present.
        TIMESTAMP_COLUMNS: Valid names for timestamp columns.
        NUMERIC_COLUMNS: OHLCV columns that must be numeric.
        VALID_NUMERIC_TYPES_POLARS: Valid Polars numeric dtypes.
        VALID_NUMERIC_TYPES_PANDAS: Valid Pandas numeric dtype strings.
        VALID_TIMESTAMP_TYPES_POLARS: Valid Polars timestamp dtypes.
        VALID_TIMESTAMP_TYPES_PANDAS: Valid Pandas timestamp dtype strings.

    Example:
        >>> validator = TFTSchemaValidatorComponent()
        >>> validator.validate(df)  # Raises SchemaValidationError if invalid
        >>> validator.validate_row_count(df, minimum=50)

    """

    # Required columns - timestamp can be 'timestamp' or 'ts_event'
    REQUIRED_COLUMNS: ClassVar[list[str]] = ["timestamp", "close"]

    # Optional columns that are validated if present
    OPTIONAL_COLUMNS: ClassVar[list[str]] = ["open", "high", "low", "volume"]

    # Valid timestamp column names
    TIMESTAMP_COLUMNS: ClassVar[list[str]] = ["timestamp", "ts_event"]

    # OHLCV columns that must be numeric
    NUMERIC_COLUMNS: ClassVar[list[str]] = ["open", "high", "low", "close", "volume"]

    # Valid numeric types for Polars
    VALID_NUMERIC_TYPES_POLARS: ClassVar[list[str]] = [
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ]

    # Valid numeric types for Pandas (lowercase substrings)
    VALID_NUMERIC_TYPES_PANDAS: ClassVar[list[str]] = [
        "float",
        "int",
        "uint",
    ]

    # Valid timestamp types for Polars
    VALID_TIMESTAMP_TYPES_POLARS: ClassVar[list[str]] = [
        "Datetime",
        "Int64",
        "UInt64",
    ]

    # Valid timestamp types for Pandas (lowercase substrings)
    VALID_TIMESTAMP_TYPES_PANDAS: ClassVar[list[str]] = [
        "datetime",
        "int",
        "uint",
    ]

    def __init__(self) -> None:
        """
        Initialize TFTSchemaValidatorComponent.

        No configuration required - all validation rules are defined as
        class attributes.

        """

    # =========================================================================
    # Public API
    # =========================================================================

    def validate(self, df: _pl.DataFrame | _pd.DataFrame) -> None:
        """
        Perform full TFT schema validation on a DataFrame.

        Validates all schema requirements including:
        - Required columns present
        - Column types correct
        - No nulls in required columns
        - No negative prices

        Args:
            df: Polars or Pandas DataFrame to validate.

        Raises:
            SchemaValidationError: If any validation check fails.

        Example:
            >>> validator = TFTSchemaValidatorComponent()
            >>> validator.validate(df)  # No exception means valid

        """
        # Validate required columns
        self._validate_required_columns(df)

        # Validate column types
        self.validate_column_types(df)

        # Validate no nulls in required columns
        required_cols = self._get_required_columns(df)
        self.validate_no_nulls(df, required_cols)

        # Validate no negative prices
        self._validate_no_negative_prices(df)

        logger.debug("TFT schema validation passed")

    def validate_column_types(self, df: _pl.DataFrame | _pd.DataFrame) -> None:
        """
        Validate that columns have expected types.

        Checks that:
        - Timestamp columns are Datetime or Int64
        - OHLCV columns are numeric (Float32, Float64, Int64)

        Args:
            df: Polars or Pandas DataFrame to validate.

        Raises:
            SchemaValidationError: If column type is incorrect.

        Example:
            >>> validator = TFTSchemaValidatorComponent()
            >>> validator.validate_column_types(df)

        """
        is_polars = self._is_polars(df)
        columns = self._get_columns(df)

        # Validate timestamp column type
        ts_col = self._find_timestamp_column(df)
        if ts_col is not None:
            ts_dtype = str(self._get_dtype(df, ts_col))

            if is_polars:
                valid = any(t in ts_dtype for t in self.VALID_TIMESTAMP_TYPES_POLARS)
            else:
                valid = any(t in ts_dtype.lower() for t in self.VALID_TIMESTAMP_TYPES_PANDAS)

            if not valid:
                raise SchemaValidationError(
                    f"Invalid type for timestamp column: expected Datetime or Int64, got {ts_dtype}",
                    column=ts_col,
                    issue=f"type mismatch - got {ts_dtype}",
                )

        # Validate numeric columns
        for col in self.NUMERIC_COLUMNS:
            if col in columns:
                dtype = str(self._get_dtype(df, col))

                if is_polars:
                    valid = any(t in dtype for t in self.VALID_NUMERIC_TYPES_POLARS)
                else:
                    valid = any(t in dtype.lower() for t in self.VALID_NUMERIC_TYPES_PANDAS)

                if not valid:
                    raise SchemaValidationError(
                        f"Invalid type for column '{col}': expected numeric, got {dtype}",
                        column=col,
                        issue=f"type mismatch - expected numeric, got {dtype}",
                    )

    def validate_row_count(
        self,
        df: _pl.DataFrame | _pd.DataFrame,
        minimum: int,
    ) -> None:
        """
        Validate that DataFrame has at least minimum rows.

        Args:
            df: Polars or Pandas DataFrame to validate.
            minimum: Minimum required number of rows.

        Raises:
            SchemaValidationError: If row count is below minimum.

        Example:
            >>> validator = TFTSchemaValidatorComponent()
            >>> validator.validate_row_count(df, minimum=50)

        """
        row_count = self._get_row_count(df)

        if row_count < minimum:
            raise SchemaValidationError(
                f"Insufficient rows: got {row_count}, minimum required is {minimum}",
                issue=f"row count {row_count} < minimum {minimum}",
            )

    def validate_no_nulls(
        self,
        df: _pl.DataFrame | _pd.DataFrame,
        columns: list[str],
    ) -> None:
        """
        Validate that specified columns contain no null values.

        Args:
            df: Polars or Pandas DataFrame to validate.
            columns: List of column names to check for nulls.

        Raises:
            SchemaValidationError: If any specified column contains nulls.

        Example:
            >>> validator = TFTSchemaValidatorComponent()
            >>> validator.validate_no_nulls(df, ["timestamp", "close"])

        """
        df_columns = self._get_columns(df)

        for col in columns:
            if col not in df_columns:
                continue

            null_count = self._count_nulls(df, col)

            if null_count > 0:
                raise SchemaValidationError(
                    f"Column '{col}' contains {null_count} null values",
                    column=col,
                    issue=f"contains {null_count} nulls",
                )

    # =========================================================================
    # Internal Validation Methods
    # =========================================================================

    def _validate_required_columns(self, df: _pl.DataFrame | _pd.DataFrame) -> None:
        """
        Validate that required columns are present.

        Checks for 'close' and either 'timestamp' or 'ts_event'.

        Args:
            df: DataFrame to validate.

        Raises:
            SchemaValidationError: If required column is missing.

        """
        columns = self._get_columns(df)

        # Check for timestamp column (timestamp or ts_event)
        ts_col = self._find_timestamp_column(df)
        if ts_col is None:
            raise SchemaValidationError(
                "Missing required timestamp column: neither 'timestamp' nor 'ts_event' found",
                column="timestamp",
                issue="column not found",
            )

        # Check for close column
        if "close" not in columns:
            raise SchemaValidationError(
                "Missing required column 'close'",
                column="close",
                issue="column not found",
            )

    def _validate_no_negative_prices(self, df: _pl.DataFrame | _pd.DataFrame) -> None:
        """
        Validate that OHLCV columns contain no negative values.

        Args:
            df: DataFrame to validate.

        Raises:
            SchemaValidationError: If negative values are found.

        """
        columns = self._get_columns(df)

        for col in self.NUMERIC_COLUMNS:
            if col not in columns:
                continue

            negative_count = self._count_negative(df, col)

            if negative_count > 0:
                raise SchemaValidationError(
                    f"Column '{col}' contains {negative_count} negative values",
                    column=col,
                    issue=f"contains {negative_count} negative values",
                )

    # =========================================================================
    # Helper Methods - DataFrame Type Detection and Operations
    # =========================================================================

    def _is_polars(self, df: _pl.DataFrame | _pd.DataFrame) -> bool:
        """
        Check if DataFrame is a Polars DataFrame.

        Args:
            df: DataFrame to check.

        Returns:
            True if Polars DataFrame, False otherwise.

        """
        # Check module name to determine type
        if hasattr(df, "__module__"):
            return "polars" in str(df.__module__)
        return False

    def _get_columns(self, df: _pl.DataFrame | _pd.DataFrame) -> list[str]:
        """
        Get list of column names from DataFrame.

        Args:
            df: DataFrame.

        Returns:
            List of column names.

        """
        df_any = cast(Any, df)
        if hasattr(df_any, "columns"):
            return list(df_any.columns)
        return []

    def _get_row_count(self, df: _pl.DataFrame | _pd.DataFrame) -> int:
        """
        Get the number of rows in a DataFrame.

        Args:
            df: DataFrame.

        Returns:
            Number of rows.

        """
        df_any = cast(Any, df)
        if hasattr(df_any, "__len__"):
            return len(df_any)
        return 0

    def _get_dtype(self, df: _pl.DataFrame | _pd.DataFrame, column: str) -> str:
        """
        Get the dtype of a column.

        Args:
            df: DataFrame.
            column: Column name.

        Returns:
            String representation of the dtype.

        """
        df_any = cast(Any, df)
        return str(df_any[column].dtype)

    def _count_nulls(self, df: _pl.DataFrame | _pd.DataFrame, column: str) -> int:
        """
        Count null values in a column.

        Args:
            df: DataFrame.
            column: Column name.

        Returns:
            Number of null values.

        """
        df_any = cast(Any, df)

        if self._is_polars(df):
            # Polars
            return int(df_any[column].is_null().sum())
        else:
            # Pandas
            return int(df_any[column].isna().sum())

    def _count_negative(self, df: _pl.DataFrame | _pd.DataFrame, column: str) -> int:
        """
        Count negative values in a column.

        Args:
            df: DataFrame.
            column: Column name.

        Returns:
            Number of negative values.

        """
        df_any = cast(Any, df)

        try:
            if self._is_polars(df):
                # Polars
                return int((df_any[column] < 0).sum())
            else:
                # Pandas
                return int((df_any[column] < 0).sum())
        except (TypeError, ValueError):
            # Column might not support comparison (e.g., strings)
            return 0

    def _find_timestamp_column(self, df: _pl.DataFrame | _pd.DataFrame) -> str | None:
        """
        Find the timestamp column in the DataFrame.

        Checks for 'timestamp' first, then falls back to 'ts_event'.

        Args:
            df: DataFrame.

        Returns:
            Name of timestamp column, or None if not found.

        """
        columns = self._get_columns(df)

        for ts_col in self.TIMESTAMP_COLUMNS:
            if ts_col in columns:
                return ts_col

        return None

    def _get_required_columns(self, df: _pl.DataFrame | _pd.DataFrame) -> list[str]:
        """
        Get the list of required columns that exist in the DataFrame.

        Args:
            df: DataFrame.

        Returns:
            List of required column names that exist.

        """
        columns = self._get_columns(df)
        required: list[str] = []

        # Add timestamp column
        ts_col = self._find_timestamp_column(df)
        if ts_col is not None:
            required.append(ts_col)

        # Add close column
        if "close" in columns:
            required.append("close")

        return required
