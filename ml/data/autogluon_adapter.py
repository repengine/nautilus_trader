"""
AutoGluon TimeSeriesDataFrame adapter for Nautilus Trader datasets.

This module provides conversion utilities to transform Nautilus datasets
(Polars DataFrames with ns-precision timestamps) to AutoGluon's
TimeSeriesDataFrame format for training with Chronos foundation models.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import TimeSeriesDataFrame
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl

    from ml.config.autogluon import AutoGluonDataConfig
    from ml.config.autogluon import ChronosTrainingConfig
    PandasDataFrame: TypeAlias = _pd.DataFrame
    PolarsDataFrame: TypeAlias = _pl.DataFrame
else:
    PandasDataFrame: TypeAlias = Any
    PolarsDataFrame: TypeAlias = Any


__all__ = [
    "canonicalize_timestamp_column",
    "compute_forward_return",
    "convert_to_timeseries_dataframe",
    "convert_to_timeseries_pandas",
    "extract_covariates",
    "validate_nautilus_dataset",
]


logger = logging.getLogger(__name__)


@overload
def canonicalize_timestamp_column(
    df: PandasDataFrame,
    *,
    timestamp_column: str,
) -> PandasDataFrame: ...


@overload
def canonicalize_timestamp_column(
    df: PolarsDataFrame,
    *,
    timestamp_column: str,
) -> PolarsDataFrame: ...


def canonicalize_timestamp_column(
    df: PandasDataFrame | PolarsDataFrame,
    *,
    timestamp_column: str,
) -> PandasDataFrame | PolarsDataFrame:
    """
    Ensure ts_event is the canonical timestamp column when required.

    This helper accepts legacy "timestamp" columns but normalizes them to
    "ts_event" when the configured timestamp column requires it.

    Args:
        df: Polars or Pandas DataFrame to normalize.
        timestamp_column: Configured timestamp column name.

    Returns:
        DataFrame with timestamp column normalized when needed.

    """
    if timestamp_column != "ts_event":
        return df

    has_ts_event = "ts_event" in df.columns
    has_timestamp = "timestamp" in df.columns

    if has_ts_event and has_timestamp:
        if HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
            logger.info("Dropping redundant 'timestamp' column (ts_event is canonical)")
            return cast(PolarsDataFrame, df).drop("timestamp")
        if HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):
            logger.info("Dropping redundant 'timestamp' column (ts_event is canonical)")
            return cast(PandasDataFrame, df).drop(columns=["timestamp"])
        return df

    if not has_ts_event and has_timestamp:
        if HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
            logger.info("Renaming 'timestamp' column -> 'ts_event'")
            return cast(PolarsDataFrame, df).rename({"timestamp": "ts_event"})
        if HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):
            logger.info("Renaming 'timestamp' column -> 'ts_event'")
            return cast(PandasDataFrame, df).rename(columns={"timestamp": "ts_event"})
        return df

    return df


def validate_nautilus_dataset(
    df: _pl.DataFrame | _pd.DataFrame,
    config: AutoGluonDataConfig,
) -> list[str]:
    """
    Validate that a Nautilus dataset has required columns for conversion.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input Nautilus dataset (Polars or Pandas).
    config : AutoGluonDataConfig
        Data configuration specifying column names.

    Returns
    -------
    list[str]
        List of validation errors. Empty if valid.

    """
    errors: list[str] = []
    df = canonicalize_timestamp_column(df, timestamp_column=config.timestamp_column)

    # Check required columns
    required = [config.item_id_column, config.timestamp_column]
    if config.target_column:
        required.append(config.target_column)

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Validate timestamp column is numeric (nanoseconds) or datetime - only for Polars DataFrames
    # For Pandas DataFrames, we skip this check as the type system is different
    if config.timestamp_column in df.columns and HAS_POLARS and pl is not None:
        # Check if this is a Polars DataFrame by checking for Polars-specific method
        if hasattr(df, "select"):  # Polars DataFrame has select method
            ts_col = df[config.timestamp_column]
            ts_dtype = ts_col.dtype
            # Use hasattr to check for is_integer method (Polars-specific)
            is_int = hasattr(ts_dtype, "is_integer") and ts_dtype.is_integer()
            is_int64 = ts_dtype == pl.Int64 or ts_dtype == pl.UInt64
            # Also accept Datetime types (already converted)
            is_datetime = hasattr(ts_dtype, "is_temporal") and ts_dtype.is_temporal()
            if not (is_int or is_int64 or is_datetime):
                errors.append(
                    f"Timestamp column '{config.timestamp_column}' must be integer (nanoseconds) or datetime, "
                    f"got {ts_dtype}"
                )

    # Check covariates exist
    for cov in config.known_covariates:
        if cov not in df.columns:
            errors.append(f"Missing known covariate column: {cov}")

    for cov in config.past_covariates:
        if cov not in df.columns:
            errors.append(f"Missing past covariate column: {cov}")

    for feat in config.static_features:
        if feat not in df.columns:
            errors.append(f"Missing static feature column: {feat}")

    return errors


def compute_forward_return(
    df: _pl.DataFrame,
    *,
    horizon: int = 15,
    price_col: str = "close",
    output_col: str = "forward_return",
    item_id_column: str = "instrument_id",
    timestamp_column: str = "ts_event",
) -> _pl.DataFrame:
    """
    Compute forward returns as regression target.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with price data.
    horizon : int, default 15
        Number of periods ahead for return calculation.
    price_col : str, default "close"
        Column name for price data.
    output_col : str, default "forward_return"
        Output column name for forward returns.
    item_id_column : str, default "instrument_id"
        Column name for series identifier (used to prevent cross-series leakage).
    timestamp_column : str, default "ts_event"
        Column name for sorting within each series.

    Returns
    -------
    pl.DataFrame
        DataFrame with forward return column added.

    """
    if not HAS_POLARS or pl is None:
        check_ml_dependencies(["polars"])
        raise ImportError("Polars not available")

    df = canonicalize_timestamp_column(df, timestamp_column=timestamp_column)

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    if item_id_column not in df.columns:
        raise ValueError(f"Item id column '{item_id_column}' not found in DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    # Sort to ensure shifts happen within each series in time order.
    df_sorted = df.sort([item_id_column, timestamp_column])

    # Compute forward return per series: (price_{t+h} - price_t) / price_t
    return df_sorted.with_columns(
        (
            (
                pl.col(price_col).shift(-horizon).over(item_id_column)
                - pl.col(price_col)
            )
            / pl.col(price_col)
        ).alias(output_col)
    )


def extract_covariates(
    df: _pl.DataFrame | _pd.DataFrame,
    config: AutoGluonDataConfig,
) -> dict[str, list[str]]:
    """
    Extract and categorize covariates from the dataset.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input DataFrame (Polars or Pandas).
    config : AutoGluonDataConfig
        Data configuration specifying covariate columns.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys 'known', 'past', 'static' containing
        lists of column names that exist in the DataFrame.

    """
    result: dict[str, list[str]] = {
        "known": [],
        "past": [],
        "static": [],
    }

    # Get columns as a set for efficient lookup
    columns = set(df.columns)

    for cov in config.known_covariates:
        if cov in columns:
            result["known"].append(cov)

    for cov in config.past_covariates:
        if cov in columns:
            result["past"].append(cov)

    for feat in config.static_features:
        if feat in columns:
            result["static"].append(feat)

    return result


def convert_to_timeseries_dataframe(
    df: _pl.DataFrame | _pd.DataFrame,
    config: ChronosTrainingConfig | AutoGluonDataConfig,
) -> Any:  # TimeSeriesDataFrame
    """
    Convert a Nautilus dataset to AutoGluon TimeSeriesDataFrame.

    This function transforms a Polars or Pandas DataFrame with nanosecond
    timestamps into AutoGluon's TimeSeriesDataFrame format suitable for
    training Chronos models.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataset with columns:
        - instrument_id (or item_id_column): Time series identifier
        - ts_event (or timestamp_column): Nanosecond timestamp
        - target column: Prediction target
        - Optional covariates
    config : ChronosTrainingConfig | AutoGluonDataConfig
        Configuration specifying column mappings and covariate assignments.

    Returns
    -------
    TimeSeriesDataFrame
        AutoGluon TimeSeriesDataFrame ready for training.

    Raises
    ------
    ImportError
        If AutoGluon or required dependencies are not available.
    ValueError
        If required columns are missing or data validation fails.

    Examples
    --------
    >>> from ml.config.autogluon import ChronosTrainingConfig
    >>> config = ChronosTrainingConfig(target_column="forward_return")
    >>> tsdf = convert_to_timeseries_dataframe(df, config)

    """
    if not HAS_AUTOGLUON:
        check_ml_dependencies(["autogluon"])

    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    df_pandas = convert_to_timeseries_pandas(df, config)

    # Build TimeSeriesDataFrame
    # Note: AutoGluon expects specific column structure
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df_pandas,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    return tsdf


def convert_to_timeseries_pandas(
    df: _pl.DataFrame | _pd.DataFrame,
    config: ChronosTrainingConfig | AutoGluonDataConfig,
) -> _pd.DataFrame:
    """
    Convert a Nautilus dataset to a pandas DataFrame in AutoGluon format.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataset with columns:
        - instrument_id (or item_id_column): Time series identifier
        - ts_event (or timestamp_column): Nanosecond timestamp
        - target column: Prediction target
        - Optional covariates
    config : ChronosTrainingConfig | AutoGluonDataConfig
        Configuration specifying column mappings and covariate assignments.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with standardized columns: item_id, timestamp, target.

    """
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    # Get data config
    if hasattr(config, "get_data_config"):
        data_config = config.get_data_config()
    else:
        # ChronosTrainingConfig passed directly - cast to AutoGluonDataConfig
        from ml.config.autogluon import AutoGluonDataConfig

        data_config = AutoGluonDataConfig() if not isinstance(config, AutoGluonDataConfig) else config

    df = canonicalize_timestamp_column(df, timestamp_column=data_config.timestamp_column)

    # Convert Polars to Pandas if needed
    if HAS_POLARS and hasattr(df, "to_pandas"):
        # Validate before conversion (df is Polars DataFrame)
        polars_df = df  # Already a Polars DataFrame
        validation_errors = validate_nautilus_dataset(polars_df, data_config)
        if validation_errors:
            raise ValueError(f"Dataset validation failed: {validation_errors}")

        # Use getattr to avoid mypy issues with duck typing
        to_pandas_method = getattr(df, "to_pandas")
        df_pandas = to_pandas_method()
    else:
        df_pandas = df

    # Convert nanosecond timestamps to datetime
    ts_col = data_config.timestamp_column
    if ts_col in df_pandas.columns:
        if df_pandas[ts_col].dtype in [np.int64, np.uint64, "int64", "Int64"]:
            # Convert nanoseconds to datetime
            df_pandas[ts_col] = pd.to_datetime(df_pandas[ts_col], unit="ns")
        elif not pd.api.types.is_datetime64_any_dtype(df_pandas[ts_col]):
            # Try to parse as datetime
            df_pandas[ts_col] = pd.to_datetime(df_pandas[ts_col])

        # AutoGluon requires timezone-naive datetime64, so strip timezone if present
        if hasattr(df_pandas[ts_col], "dt") and df_pandas[ts_col].dt.tz is not None:
            df_pandas[ts_col] = df_pandas[ts_col].dt.tz_localize(None)

    # Rename columns to AutoGluon expected format
    rename_map = {
        data_config.item_id_column: "item_id",
        data_config.timestamp_column: "timestamp",
    }

    # Only rename target if it's different from expected
    target_col = data_config.target_column
    if target_col and target_col != "target":
        rename_map[target_col] = "target"

    df_pandas = df_pandas.rename(columns=rename_map)

    # Ensure item_id is string type
    if "item_id" in df_pandas.columns:
        df_pandas["item_id"] = df_pandas["item_id"].astype(str)

    # Sort by item_id and timestamp for proper time series structure
    df_pandas = df_pandas.sort_values(["item_id", "timestamp"])

    # Drop rows with NaN target (e.g., from forward return computation)
    if "target" in df_pandas.columns:
        initial_rows = len(df_pandas)
        df_pandas = df_pandas.dropna(subset=["target"])
        dropped = initial_rows - len(df_pandas)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN target values")

    # Get covariates that exist in the DataFrame
    covariates = extract_covariates(df_pandas, data_config)
    known_covariates = covariates["known"]
    static_features = covariates["static"]

    logger.info(
        "Prepared pandas time series frame: %d rows, %d time series, %d known covariates, %d static features",
        len(df_pandas),
        df_pandas["item_id"].nunique(),
        len(known_covariates),
        len(static_features),
    )

    return cast(PandasDataFrame, df_pandas)
