"""
Target generation component for TFT dataset building.

This component extracts and handles binary target generation from the legacy
TFTDatasetBuilder, supporting both Polars and Pandas DataFrames.

Extracted methods:
- _generate_targets_polars() (lines 1929-1960)
- _generate_targets_pandas() (lines 1961-1989)

Target Generation Logic:
1. Forward return calculation: forward_return = (close.shift(-horizon) - close) / close
2. Binary target: y = 1 if forward_return > threshold else 0
3. NaN filling: Last `horizon` rows have NaN targets which are filled with 0
4. No lookahead bias: Targets use future prices (negative shift)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

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


class TargetGenerationComponent:
    """
    Component for generating binary classification targets for TFT models.

    This component computes forward-looking returns and binary classification
    targets from price data. The targets indicate whether the price will
    exceed a threshold return within a specified horizon.

    Target Columns Generated:
        - y: Binary classification target (1 if forward_return > threshold, else 0)
        - forward_return: Continuous forward return for Sharpe ratio computation

    Key Invariants:
        - y values are always in {0, 1}
        - forward_return >= -1.0 (cannot lose more than 100%)
        - No NaN values in output (filled with 0)
        - No lookahead bias: targets use future prices only

    Example:
        >>> component = TargetGenerationComponent()
        >>> targets_df = component.generate_targets_polars(df, horizon_minutes=15, threshold=0.001)
        >>> assert set(targets_df["y"].unique().to_list()) <= {0, 1}
        >>> assert targets_df["forward_return"].is_null().sum() == 0

    """

    def generate_targets_polars(
        self,
        df: _pl.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pl.DataFrame:
        """
        Generate binary targets using Polars.

        Computes forward-looking returns by shifting close prices by the
        negative horizon, then creates binary classification targets based
        on whether the forward return exceeds the threshold.

        Args:
            df: Polars DataFrame with 'close' column containing price data.
            horizon_minutes: Number of periods to look ahead for target calculation.
                Must be positive. Represents the prediction horizon.
            threshold: Minimum return threshold for positive classification (y=1).
                Should be non-negative. Values like 0.001 (0.1%) are typical.

        Returns:
            Polars DataFrame with two columns:
            - y (Int32): Binary target (1 if forward_return > threshold, else 0)
            - forward_return (Float32): Continuous forward return

        Raises:
            KeyError: If 'close' column is missing from input DataFrame.
            ValueError: If horizon_minutes is not positive.

        Example:
            >>> df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
            >>> targets = component.generate_targets_polars(df, horizon_minutes=2, threshold=0.01)
            >>> # forward_return[0] = (102-100)/100 = 0.02 > 0.01, so y[0] = 1
            >>> assert targets["y"][0] == 1

        Notes:
            - The last `horizon_minutes` rows will have NaN in forward_return due
              to the forward shift, which are filled with 0.
            - Division by zero (when close=0) produces inf, which is filled with 0.

        """
        # Validate horizon
        if horizon_minutes <= 0:
            raise ValueError(
                f"horizon_minutes must be positive, got {horizon_minutes}",
            )

        # Validate close column exists
        if "close" not in df.columns:
            raise KeyError("Missing required 'close' column for target generation")

        # Handle empty DataFrame
        if df.is_empty():
            empty_df: _pl.DataFrame = pl.DataFrame(
                {
                    "y": pl.Series([], dtype=pl.Int32),
                    "forward_return": pl.Series([], dtype=pl.Float32),
                }
            )
            return empty_df

        # Calculate forward returns
        future_prices = pl.col("close").shift(-horizon_minutes)
        current_prices = pl.col("close")
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        targets = df.select(
            [
                (forward_returns > threshold).cast(pl.Int32).alias("y"),
                forward_returns.cast(pl.Float32).alias("forward_return"),
            ],
        )

        # Fill trailing NaNs introduced by the horizon shift
        targets = targets.with_columns(
            [
                pl.col("y").fill_null(0),
                pl.col("forward_return").fill_null(0.0),
            ],
        )

        # Handle infinities from division by zero
        targets = targets.with_columns(
            [
                pl.when(pl.col("y").is_infinite() | pl.col("y").is_nan())
                .then(0)
                .otherwise(pl.col("y"))
                .alias("y"),
                pl.when(pl.col("forward_return").is_infinite() | pl.col("forward_return").is_nan())
                .then(0.0)
                .otherwise(pl.col("forward_return"))
                .cast(pl.Float32)
                .alias("forward_return"),
            ],
        )

        return targets

    def generate_targets_pandas(
        self,
        df: _pd.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pd.DataFrame:
        """
        Generate binary targets using Pandas.

        Identical logic to generate_targets_polars but using Pandas operations.
        Both implementations produce identical outputs for the same inputs.

        Args:
            df: Pandas DataFrame with 'close' column containing price data.
            horizon_minutes: Number of periods to look ahead for target calculation.
                Must be positive. Represents the prediction horizon.
            threshold: Minimum return threshold for positive classification (y=1).
                Should be non-negative. Values like 0.001 (0.1%) are typical.

        Returns:
            Pandas DataFrame with two columns:
            - y (int): Binary target (1 if forward_return > threshold, else 0)
            - forward_return (float): Continuous forward return

        Raises:
            KeyError: If 'close' column is missing from input DataFrame.
            ValueError: If horizon_minutes is not positive.

        Example:
            >>> df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
            >>> targets = component.generate_targets_pandas(df, horizon_minutes=2, threshold=0.01)
            >>> # forward_return[0] = (102-100)/100 = 0.02 > 0.01, so y[0] = 1
            >>> assert targets["y"].iloc[0] == 1

        Notes:
            - The last `horizon_minutes` rows will have NaN in forward_return due
              to the forward shift, which are filled with 0.
            - Division by zero (when close=0) produces inf, which is filled with 0.

        """
        # Validate horizon
        if horizon_minutes <= 0:
            raise ValueError(
                f"horizon_minutes must be positive, got {horizon_minutes}",
            )

        # Validate close column exists
        if "close" not in df.columns:
            raise KeyError("Missing required 'close' column for target generation")

        # Handle empty DataFrame
        if len(df) == 0:
            empty_df: _pd.DataFrame = pd.DataFrame(
                {
                    "y": pd.Series([], dtype=int),
                    "forward_return": pd.Series([], dtype=float),
                }
            )
            return empty_df

        # Calculate forward returns
        future_prices = df["close"].shift(-horizon_minutes)
        current_prices = df["close"]
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        targets = pd.DataFrame(
            {
                "y": (forward_returns > threshold).astype(int),
                "forward_return": forward_returns.astype(float),
            },
        )

        # Fill trailing NaNs introduced by the horizon shift
        targets = targets.fillna({"y": 0, "forward_return": 0.0})

        # Handle infinities from division by zero
        targets = targets.replace([np.inf, -np.inf], 0.0)

        # Ensure correct types after fillna/replace
        targets["y"] = targets["y"].astype(int)
        targets["forward_return"] = targets["forward_return"].astype(float)

        return cast("_pd.DataFrame", targets)
