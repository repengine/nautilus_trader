"""
TFT Dataset Builder for quick training data preparation.

This module provides a fast path to create TFT-compatible training datasets
from existing collected market data.

"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml.config.base import MLFeatureConfig
from ml.data.catalog_utils import bars_to_dataframe
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


class TFTDatasetBuilder:
    """Fast TFT dataset builder using existing collected data."""
    
    def __init__(
        self,
        catalog: ParquetDataCatalog,
        symbols: list[str],
        feature_config: MLFeatureConfig | None = None,
    ) -> None:
        """
        Initialize TFT dataset builder.
        
        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for accessing market data
        symbols : list[str]
            List of symbols to include in dataset
        feature_config : MLFeatureConfig, optional
            Feature engineering configuration
        
        """
        self.catalog = catalog
        self.symbols = symbols
        self.feature_config = feature_config or MLFeatureConfig()
        # Skip feature engineer for now - will use simplified features
        
        logger.info(f"Initialized TFTDatasetBuilder with {len(symbols)} symbols")
    
    def build_training_dataset(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Build complete TFT training dataset.
        
        Parameters
        ----------
        horizon_minutes : int, default 15
            Prediction horizon in minutes
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%)
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation
        use_polars : bool, default True
            Whether to use Polars for faster processing
        
        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset
        
        """
        all_data = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            # Load data using catalog
            try:
                # Assuming symbol needs venue suffix (e.g., NYSE, NASDAQ)
                instrument_id = f"{symbol}.NYSE"  # Default to NYSE, could be configurable
                df = bars_to_dataframe(
                    self.catalog,
                    [instrument_id],
                    start=None,  # Load all available data
                    end=None,
                )
                
                if df.is_empty():
                    # Try with NASDAQ if NYSE doesn't work
                    instrument_id = f"{symbol}.NASDAQ"
                    df = bars_to_dataframe(
                        self.catalog,
                        [instrument_id],
                        start=None,
                        end=None,
                    )
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                df = None
            
            if df is None:
                logger.warning(f"No data found for {symbol}, skipping")
                continue
            
            # Process with Polars or Pandas
            if use_polars:
                processed = self._process_symbol_polars(
                    df, symbol, horizon_minutes, min_return_threshold, lookback_periods
                )
            else:
                processed = self._process_symbol_pandas(
                    df, symbol, horizon_minutes, min_return_threshold, lookback_periods
                )
            
            if processed is not None:
                all_data.append(processed)
        
        if not all_data:
            logger.error("No data processed for any symbol")
            return pd.DataFrame() if not use_polars else pl.DataFrame()
        
        # Combine all symbols
        if use_polars:
            final_df = pl.concat(all_data, how="vertical")
        else:
            final_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Built dataset with shape: {final_df.shape}")
        
        return final_df
    
    def _process_symbol_polars(
        self,
        df: pl.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> pl.DataFrame | None:
        """Process single symbol with Polars."""
        
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None
        
        # Sort by time and create sequential index
        df = df.sort("timestamp" if "timestamp" in df.columns else df.columns[0])
        df = df.with_columns([
            pl.arange(0, len(df)).alias("time_index"),
            pl.lit(symbol).alias("instrument_id"),
        ])
        
        # Generate features
        features = self._compute_features_polars(df)
        
        # Generate targets
        targets = self._generate_targets_polars(df, horizon_minutes, threshold)
        
        # Combine
        dataset = pl.concat([
            df.select(["time_index", "instrument_id"]),
            features,
            targets,
        ], how="horizontal")
        
        # Filter for sufficient history
        dataset = dataset.slice(lookback_periods, len(dataset))
        
        # Add static and known-future features
        dataset = self._add_static_features_polars(dataset)
        dataset = self._add_known_future_features_polars(dataset)
        
        return dataset
    
    def _process_symbol_pandas(
        self,
        df: pd.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> pd.DataFrame | None:
        """Process single symbol with Pandas."""
        
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None
        
        # Sort by index (timestamps) and create sequential index
        df = df.sort_index().reset_index(drop=False)
        # Rename the index to timestamp if it's called ts_event
        if 'ts_event' in df.columns:
            df = df.rename(columns={'ts_event': 'timestamp'})
        elif df.index.name == 'ts_event':
            df = df.reset_index().rename(columns={'ts_event': 'timestamp'})
        
        df["time_index"] = range(len(df))
        df["instrument_id"] = symbol
        
        # Generate features
        features = self._compute_features_pandas(df)
        
        # Generate targets
        targets = self._generate_targets_pandas(df, horizon_minutes, threshold)
        
        # Combine
        dataset = pd.concat([
            df[["time_index", "instrument_id"]],
            features,
            targets,
        ], axis=1)
        
        # Filter for sufficient history
        dataset = dataset.iloc[lookback_periods:].copy()
        
        # Add static and known-future features
        dataset = self._add_static_features_pandas(dataset)
        dataset = self._add_known_future_features_pandas(dataset)
        
        return dataset
    
    def _compute_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute technical features using Polars."""
        
        # Simple technical indicators (can be enhanced with full FeatureEngineer)
        features = df.select([
            # Price-based features
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("return_5"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("return_20"),
            
            # Volume features
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
            
            # Volatility
            pl.col("return_1").rolling_std(20).alias("volatility_20"),
            
            # Simple moving averages
            pl.col("close").rolling_mean(5).alias("sma_5"),
            pl.col("close").rolling_mean(20).alias("sma_20"),
            
            # Price position
            ((pl.col("close") - pl.col("low").rolling_min(20)) /
             (pl.col("high").rolling_max(20) - pl.col("low").rolling_min(20))).alias("price_position"),
        ])
        
        # Fill NaN values
        features = features.fill_null(0)
        
        return features
    
    def _compute_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features using Pandas."""
        
        features = pd.DataFrame()
        
        # Price-based features
        features["return_1"] = df["close"].pct_change(1)
        features["return_5"] = df["close"].pct_change(5)
        features["return_20"] = df["close"].pct_change(20)
        
        # Volume features
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        # Volatility
        features["volatility_20"] = features["return_1"].rolling(20).std()
        
        # Simple moving averages
        features["sma_5"] = df["close"].rolling(5).mean()
        features["sma_20"] = df["close"].rolling(20).mean()
        
        # Price position
        rolling_min = df["low"].rolling(20).min()
        rolling_max = df["high"].rolling(20).max()
        features["price_position"] = (df["close"] - rolling_min) / (rolling_max - rolling_min)
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _generate_targets_polars(
        self,
        df: pl.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> pl.DataFrame:
        """Generate binary targets using Polars."""
        
        # Calculate forward returns
        future_prices = pl.col("close").shift(-horizon_minutes)
        current_prices = pl.col("close")
        forward_returns = (future_prices - current_prices) / current_prices
        
        # Binary classification
        targets = df.select([
            (forward_returns > threshold).cast(pl.Int32).alias("y")
        ])
        
        # Fill NaN at the end
        targets = targets.fill_null(0)
        
        return targets
    
    def _generate_targets_pandas(
        self,
        df: pd.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> pd.DataFrame:
        """Generate binary targets using Pandas."""
        
        # Calculate forward returns
        future_prices = df["close"].shift(-horizon_minutes)
        current_prices = df["close"]
        forward_returns = (future_prices - current_prices) / current_prices
        
        # Binary classification
        targets = pd.DataFrame({
            "y": (forward_returns > threshold).astype(int)
        })
        
        # Fill NaN at the end
        targets = targets.fillna(0)
        
        return targets
    
    def _add_static_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add static instrument features using Polars."""
        
        # Simple static feature mapping
        static_map = {
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
        
        # Get unique instruments
        instruments = df["instrument_id"].unique().to_list()
        
        # Add static features for each instrument
        for instrument in instruments:
            static = static_map.get(instrument, {
                "asset_class": "STOCK",
                "tick_size": 0.01,
                "exchange": "UNKNOWN",
            })
            
            df = df.with_columns([
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["asset_class"]))
                .otherwise(pl.col("asset_class") if "asset_class" in df.columns else pl.lit("UNKNOWN"))
                .alias("asset_class"),
                
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["tick_size"]))
                .otherwise(pl.col("tick_size") if "tick_size" in df.columns else pl.lit(0.01))
                .alias("tick_size"),
                
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["exchange"]))
                .otherwise(pl.col("exchange") if "exchange" in df.columns else pl.lit("UNKNOWN"))
                .alias("exchange"),
            ])
        
        return df
    
    def _add_static_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add static instrument features using Pandas."""
        
        # Simple static feature mapping
        static_map = {
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
        
        # Add static features
        for col in ["asset_class", "tick_size", "exchange"]:
            df[col] = df["instrument_id"].map(
                lambda x: static_map.get(x, {
                    "asset_class": "STOCK",
                    "tick_size": 0.01,
                    "exchange": "UNKNOWN",
                }).get(col)
            )
        
        return df
    
    def _add_known_future_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add known-future time features using Polars."""
        
        # Create hour and minute from time_index (assuming minute bars)
        df = df.with_columns([
            ((pl.col("time_index") // 60) % 24).alias("hour"),
            (pl.col("time_index") % 60).alias("minute"),
        ])
        
        # Time of day features (cyclical encoding)
        df = df.with_columns([
            (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60)).sin().alias("tod_sin"),
            (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60)).cos().alias("tod_cos"),
        ])
        
        # Day of week (simplified - assuming continuous trading for now)
        df = df.with_columns([
            ((pl.col("time_index") // (24 * 60)) % 7).alias("dow"),
        ])
        
        df = df.with_columns([
            (2 * np.pi * pl.col("dow") / 7).sin().alias("dow_sin"),
            (2 * np.pi * pl.col("dow") / 7).cos().alias("dow_cos"),
        ])
        
        # Market session flags
        df = df.with_columns([
            ((pl.col("hour") >= 9) & (pl.col("hour") < 16)).cast(pl.Int32).alias("is_market_open"),
            ((pl.col("hour") >= 4) & (pl.col("hour") < 9)).cast(pl.Int32).alias("is_premarket"),
            ((pl.col("hour") >= 16) & (pl.col("hour") < 20)).cast(pl.Int32).alias("is_aftermarket"),
        ])
        
        return df
    
    def _add_known_future_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add known-future time features using Pandas."""
        
        # Create hour and minute from time_index (assuming minute bars)
        df["hour"] = (df["time_index"] // 60) % 24
        df["minute"] = df["time_index"] % 60
        
        # Time of day features (cyclical encoding)
        time_in_minutes = df["hour"] * 60 + df["minute"]
        df["tod_sin"] = np.sin(2 * np.pi * time_in_minutes / (24 * 60))
        df["tod_cos"] = np.cos(2 * np.pi * time_in_minutes / (24 * 60))
        
        # Day of week (simplified - assuming continuous trading for now)
        df["dow"] = (df["time_index"] // (24 * 60)) % 7
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        
        # Market session flags
        df["is_market_open"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
        df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
        df["is_aftermarket"] = ((df["hour"] >= 16) & (df["hour"] < 20)).astype(int)
        
        return df