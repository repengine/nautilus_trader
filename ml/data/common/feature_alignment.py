"""
Feature alignment component for TFT dataset building.

This component extracts and handles feature computation, alignment,
and static feature addition from the legacy TFTDatasetBuilder.

Extracted methods:
- _compute_features_polars() (lines 1864-1895)
- _compute_features_pandas() (lines 1896-1928)
- _add_static_features_polars() (lines 1990-2043)
- _add_static_features_pandas() (lines 2044-2075)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

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


class FeatureAlignmentComponent:
    """
    Component for feature computation and alignment for TFT datasets.

    This component provides methods for:
    - Computing technical features (returns, volume ratio, volatility, SMAs, price position)
    - Adding static instrument features (asset class, tick size, exchange)
    - Supporting both Polars and Pandas DataFrames with identical outputs

    All computed features are filled with 0 to handle NaN values, which is
    critical for TFT model training.

    Feature List:
        - return_1: 1-period return
        - return_5: 5-period return
        - return_20: 20-period return
        - volume_ratio: volume / 20-period rolling mean volume
        - volatility_20: 20-period rolling standard deviation of return_1
        - sma_5: 5-period simple moving average of close
        - sma_20: 20-period simple moving average of close
        - price_position: (close - rolling_min) / (rolling_max - rolling_min) in [0, 1]

    Static Features:
        - asset_class: ETF, STOCK, etc.
        - tick_size: Minimum price increment
        - exchange: Trading venue (ARCA, NASDAQ, etc.)

    Example:
        >>> component = FeatureAlignmentComponent()
        >>> features = component.compute_features_polars(df)
        >>> df_with_static = component.add_static_features_polars(df)

    """

    # Static feature mapping for known symbols
    STATIC_FEATURE_MAP: ClassVar[dict[str, dict[str, Any]]] = {
        "SPY": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "ARCA"},
        "QQQ": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "NASDAQ"},
        "AAPL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "MSFT": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "NVDA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "AMZN": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "META": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "GOOGL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        "TSLA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    }

    DEFAULT_STATIC_FEATURES: ClassVar[dict[str, Any]] = {
        "asset_class": "STOCK",
        "tick_size": 0.01,
        "exchange": "UNKNOWN",
    }

    # Required columns for feature computation
    REQUIRED_COLUMNS: ClassVar[list[str]] = ["close", "volume", "high", "low"]

    def compute_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Compute technical features using Polars.

        Computes 8 technical features from OHLCV data. All NaN values are
        filled with 0 to ensure TFT model compatibility.

        Args:
            df: Polars DataFrame with OHLCV columns (close, volume, high, low)

        Returns:
            Polars DataFrame with 8 feature columns:
            - return_1, return_5, return_20
            - volume_ratio
            - volatility_20
            - sma_5, sma_20
            - price_position

        Raises:
            ValueError: If required columns (close, volume, high, low) are missing

        Example:
            >>> df = pl.DataFrame({
            ...     "close": [100.0, 101.0, 102.0],
            ...     "volume": [1000.0, 1100.0, 1200.0],
            ...     "high": [101.0, 102.0, 103.0],
            ...     "low": [99.0, 100.0, 101.0],
            ... })
            >>> features = component.compute_features_polars(df)
            >>> assert "return_1" in features.columns

        """
        # Handle empty DataFrame
        if df.is_empty():
            feature_cols = [
                "return_1",
                "return_5",
                "return_20",
                "volume_ratio",
                "volatility_20",
                "sma_5",
                "sma_20",
                "price_position",
            ]
            empty_df: _pl.DataFrame = pl.DataFrame(
                {col: [] for col in feature_cols},
            ).cast(dict.fromkeys(feature_cols, pl.Float64))
            return empty_df

        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for feature computation: {missing_cols}",
            )

        # Compute rolling min/max first to handle division by zero
        rolling_min_20 = pl.col("low").rolling_min(20)
        rolling_max_20 = pl.col("high").rolling_max(20)
        price_range = rolling_max_20 - rolling_min_20

        # Safe division for price_position - use when/then/otherwise
        price_position_expr = (
            pl.when(price_range > 0)
            .then((pl.col("close") - rolling_min_20) / price_range)
            .otherwise(0.0)
            .alias("price_position")
        )

        # Safe division for volume_ratio
        vol_mean_20 = pl.col("volume").rolling_mean(20)
        volume_ratio_expr = (
            pl.when(vol_mean_20 > 0)
            .then(pl.col("volume") / vol_mean_20)
            .otherwise(0.0)
            .alias("volume_ratio")
        )

        base = df.with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1"),
                (pl.col("close") / pl.col("close").shift(5) - 1).alias("return_5"),
                (pl.col("close") / pl.col("close").shift(20) - 1).alias("return_20"),
                volume_ratio_expr,
                pl.col("close").rolling_mean(5).alias("sma_5"),
                pl.col("close").rolling_mean(20).alias("sma_20"),
                price_position_expr,
            ]
        )

        features = base.select(
            [
                "return_1",
                "return_5",
                "return_20",
                "volume_ratio",
                pl.col("return_1").rolling_std(20).alias("volatility_20"),
                "sma_5",
                "sma_20",
                "price_position",
            ]
        ).fill_null(0)

        # Replace any infinities with 0
        features = features.with_columns(
            [
                pl.when(pl.col(col).is_infinite()).then(0.0).otherwise(pl.col(col)).alias(col)
                for col in features.columns
            ]
        )

        return features

    def compute_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Compute technical features using Pandas.

        Computes the same 8 technical features as the Polars implementation.
        All NaN and infinity values are filled with 0.

        Args:
            df: Pandas DataFrame with OHLCV columns (close, volume, high, low)

        Returns:
            Pandas DataFrame with 8 feature columns matching Polars output.

        Raises:
            ValueError: If required columns (close, volume, high, low) are missing

        Example:
            >>> df = pd.DataFrame({
            ...     "close": [100.0, 101.0, 102.0],
            ...     "volume": [1000.0, 1100.0, 1200.0],
            ...     "high": [101.0, 102.0, 103.0],
            ...     "low": [99.0, 100.0, 101.0],
            ... })
            >>> features = component.compute_features_pandas(df)
            >>> assert "return_1" in features.columns

        """
        # Handle empty DataFrame
        if len(df) == 0:
            empty_df: _pd.DataFrame = pd.DataFrame(
                {
                    "return_1": pd.Series([], dtype=float),
                    "return_5": pd.Series([], dtype=float),
                    "return_20": pd.Series([], dtype=float),
                    "volume_ratio": pd.Series([], dtype=float),
                    "volatility_20": pd.Series([], dtype=float),
                    "sma_5": pd.Series([], dtype=float),
                    "sma_20": pd.Series([], dtype=float),
                    "price_position": pd.Series([], dtype=float),
                }
            )
            return empty_df

        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for feature computation: {missing_cols}",
            )

        features = pd.DataFrame()

        # Price-based features
        features["return_1"] = df["close"].pct_change(1)
        features["return_5"] = df["close"].pct_change(5)
        features["return_20"] = df["close"].pct_change(20)

        # Volume features - handle division by zero
        vol_mean_20 = df["volume"].rolling(20).mean()
        features["volume_ratio"] = np.where(
            vol_mean_20 > 0,
            df["volume"] / vol_mean_20,
            0.0,
        )

        # Volatility
        features["volatility_20"] = features["return_1"].rolling(20).std()

        # Simple moving averages
        features["sma_5"] = df["close"].rolling(5).mean()
        features["sma_20"] = df["close"].rolling(20).mean()

        # Price position - handle division by zero
        rolling_min = df["low"].rolling(20).min()
        rolling_max = df["high"].rolling(20).max()
        price_range = rolling_max - rolling_min
        features["price_position"] = np.where(
            price_range > 0,
            (df["close"] - rolling_min) / price_range,
            0.0,
        )

        # Fill NaN and infinity values with 0
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)

        return cast("_pd.DataFrame", features)

    def add_static_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Add static instrument features using Polars.

        Adds asset_class, tick_size, and exchange columns based on the
        instrument_id column. Uses default values for unknown symbols.

        Args:
            df: Polars DataFrame with instrument_id column

        Returns:
            DataFrame with added static feature columns.

        Raises:
            ValueError: If instrument_id column is missing

        Example:
            >>> df = pl.DataFrame({
            ...     "instrument_id": ["SPY", "SPY"],
            ...     "close": [450.0, 451.0],
            ... })
            >>> result = component.add_static_features_polars(df)
            >>> assert result["asset_class"][0] == "ETF"
            >>> assert result["exchange"][0] == "ARCA"

        """
        if "instrument_id" not in df.columns:
            raise ValueError("Missing required 'instrument_id' column for static features")

        if df.is_empty():
            # Add empty columns with correct types
            return df.with_columns(
                [
                    pl.lit(None).cast(pl.Utf8).alias("asset_class"),
                    pl.lit(None).cast(pl.Float64).alias("tick_size"),
                    pl.lit(None).cast(pl.Utf8).alias("exchange"),
                ]
            )

        # Get unique instruments
        instruments = df["instrument_id"].unique().to_list()

        result = df

        # Add static features for each instrument
        for instrument in instruments:
            static = self.STATIC_FEATURE_MAP.get(
                instrument,
                self.DEFAULT_STATIC_FEATURES,
            )

            result = result.with_columns(
                [
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["asset_class"]))
                    .otherwise(
                        (
                            pl.col("asset_class")
                            if "asset_class" in result.columns
                            else pl.lit("UNKNOWN")
                        ),
                    )
                    .alias("asset_class"),
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["tick_size"]))
                    .otherwise(
                        pl.col("tick_size") if "tick_size" in result.columns else pl.lit(0.01),
                    )
                    .alias("tick_size"),
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["exchange"]))
                    .otherwise(
                        pl.col("exchange") if "exchange" in result.columns else pl.lit("UNKNOWN"),
                    )
                    .alias("exchange"),
                ]
            )

        return result

    def add_static_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Add static instrument features using Pandas.

        Adds the same static features as the Polars implementation.

        Args:
            df: Pandas DataFrame with instrument_id column

        Returns:
            DataFrame with added static feature columns.

        Raises:
            ValueError: If instrument_id column is missing

        Example:
            >>> df = pd.DataFrame({
            ...     "instrument_id": ["SPY", "SPY"],
            ...     "close": [450.0, 451.0],
            ... })
            >>> result = component.add_static_features_pandas(df)
            >>> assert result["asset_class"].iloc[0] == "ETF"

        """
        if "instrument_id" not in df.columns:
            raise ValueError("Missing required 'instrument_id' column for static features")

        if len(df) == 0:
            # Add empty columns with correct types
            result = df.copy()
            result["asset_class"] = pd.Series([], dtype=str)
            result["tick_size"] = pd.Series([], dtype=float)
            result["exchange"] = pd.Series([], dtype=str)
            return result

        result = df.copy()

        # Add static features using map with explicit function
        def get_asset_class(x: str) -> str:
            return str(
                self.STATIC_FEATURE_MAP.get(
                    x,
                    self.DEFAULT_STATIC_FEATURES,
                ).get("asset_class", "STOCK"),
            )

        def get_tick_size(x: str) -> float:
            return float(
                self.STATIC_FEATURE_MAP.get(
                    x,
                    self.DEFAULT_STATIC_FEATURES,
                ).get("tick_size", 0.01),
            )

        def get_exchange(x: str) -> str:
            return str(
                self.STATIC_FEATURE_MAP.get(
                    x,
                    self.DEFAULT_STATIC_FEATURES,
                ).get("exchange", "UNKNOWN"),
            )

        result["asset_class"] = result["instrument_id"].map(get_asset_class)
        result["tick_size"] = result["instrument_id"].map(get_tick_size)
        result["exchange"] = result["instrument_id"].map(get_exchange)

        return result
