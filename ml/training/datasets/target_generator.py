"""
Target generation for TFT training datasets.

This module provides dual Polars/Pandas implementations for generating prediction
targets from price data, supporting forward returns and binary classification.

Extracted from TFTDatasetBuilder as part of Phase 3.1 decomposition.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
else:
    pd = Any
    pl = Any


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class TargetGeneratorProtocol(Protocol):
    """Protocol for target generation operations."""

    def generate_targets_polars(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate targets using Polars.

        Parameters
        ----------
        df : polars.DataFrame
            Input dataframe
        horizon_minutes : int
            Horizon in minutes
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        polars.DataFrame
            DataFrame with target columns

        """
        ...

    def generate_targets_pandas(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate targets using Pandas.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe
        horizon_minutes : int
            Horizon in minutes
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        pandas.DataFrame
            DataFrame with target columns

        """
        ...

    def generate_targets(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
        use_polars: bool = True,
    ) -> Any:
        """
        Generate targets using specified implementation.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        horizon_minutes : int
            Forward-looking horizon in minutes
        threshold : float
            Binary classification threshold (e.g., 0.001 = 10bps)
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation

        Returns
        -------
        polars.DataFrame | pandas.DataFrame
            DataFrame with target columns added

        """
        ...


# ========================================================================
# TargetGenerator Implementation
# ========================================================================


class TargetGenerator:
    """
    Generates prediction targets for TFT models.

    Computes forward-looking returns and binary classification labels
    with configurable horizons and thresholds.

    Targets generated:
    - forward_return: (future_close - current_close) / current_close
    - y: Binary label (1 if forward_return > threshold, else 0)

    """

    def __init__(self) -> None:
        """Initialize target generator."""
        logger.debug("TargetGenerator initialized")

    def generate_targets(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
        use_polars: bool = True,
    ) -> Any:
        """
        Generate targets using specified implementation.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        horizon_minutes : int
            Forward-looking horizon in minutes
        threshold : float
            Binary classification threshold (e.g., 0.001 = 10bps)
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation

        Returns
        -------
        polars.DataFrame | pandas.DataFrame
            DataFrame with target columns added

        """
        if use_polars:
            return self.generate_targets_polars(df, horizon_minutes, threshold)
        else:
            return self.generate_targets_pandas(df, horizon_minutes, threshold)

    def generate_targets_polars(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate targets using Polars.

        Computes:
        - forward_return: (future_close - current_close) / current_close
        - y: Binary label (1 if forward_return > threshold, else 0)

        Parameters
        ----------
        df : polars.DataFrame
            Input dataframe with 'close' column
        horizon_minutes : int
            Horizon in minutes (how many periods to look ahead)
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        polars.DataFrame
            Input dataframe with added 'y' and 'forward_return' columns

        """
        # Import polars at runtime
        try:
            import polars as pl
        except ImportError as e:
            msg = "Polars is required for TargetGenerator but not installed"
            raise ImportError(msg) from e

        # Calculate forward returns
        future_prices = pl.col("close").shift(-horizon_minutes)
        current_prices = pl.col("close")
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        # Use with_columns to preserve all input columns
        return df.with_columns(
            [
                (forward_returns > threshold).cast(pl.Int32).alias("y"),
                forward_returns.cast(pl.Float32).alias("forward_return"),
            ],
        ).with_columns(
            [
                pl.col("y").fill_null(0),
                pl.col("forward_return").fill_null(0.0),
            ],
        )

    def generate_targets_pandas(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate targets using Pandas.

        Identical logic to generate_targets_polars but using Pandas API.
        Ensures target parity between implementations.

        Computes:
        - forward_return: (future_close - current_close) / current_close
        - y: Binary label (1 if forward_return > threshold, else 0)

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with 'close' column
        horizon_minutes : int
            Horizon in minutes (how many periods to look ahead)
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        pandas.DataFrame
            Input dataframe with added 'y' and 'forward_return' columns

        """
        # Import pandas at runtime
        try:
            import pandas as pd
        except ImportError as e:
            msg = "Pandas is required for TargetGenerator but not installed"
            raise ImportError(msg) from e

        # Calculate forward returns
        future_prices = df["close"].shift(-horizon_minutes)
        current_prices = df["close"]
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        # Add targets as new columns to existing dataframe
        df = df.copy()
        df["y"] = (forward_returns > threshold).astype(int)
        df["forward_return"] = forward_returns.astype(float)

        # Fill trailing NaNs introduced by the horizon shift
        df[["y", "forward_return"]] = df[["y", "forward_return"]].fillna(
            {"y": 0, "forward_return": 0.0},
        )

        return df
