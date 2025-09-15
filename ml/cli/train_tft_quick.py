#!/usr/bin/env python3
"""
Quick TFT training script for immediate model training.

This script provides the fastest path to training a TFT teacher model using existing
collected market data.

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import polars as pl

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Quick TFT training on collected data.
    """
    # Configuration
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]  # Start with priority symbols
    horizon_minutes = 15  # Predict 15-minute ahead
    min_return_threshold = 0.002  # 0.2% threshold for binary classification
    lookback_periods = 50  # Ensure enough history for features

    # Data directories to check (prefer curated tier1)
    data_dirs = [
        Path("data/tier1"),
        Path("/home/nate/projects/nautilus_trader/data/tier1"),
        Path("data"),
    ]

    # Find valid data directory
    data_dir = None
    for dir_path in data_dirs:
        if dir_path.exists():
            data_dir = dir_path
            logger.info(f"Using data directory: {data_dir}")
            break

    if data_dir is None:
        logger.error("No valid data directory found")
        return 1

    # Build dataset
    logger.info("Building TFT dataset...")
    # Initialize ParquetDataCatalog at the chosen path
    catalog = ParquetDataCatalog(path=str(data_dir))
    builder = TFTDatasetBuilder(catalog, symbols)

    try:
        # Try with Polars first (faster)
        df = builder.build_training_dataset(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=True,
        )

        # Convert to Pandas for TFT (if using Polars)
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df

    except Exception as e:
        logger.warning(f"Polars processing failed: {e}, falling back to Pandas")
        df = builder.build_training_dataset(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=False,
        )
        from typing import cast

        df_pd = cast(pd.DataFrame, df)

    if df_pd.empty:
        logger.error("No data was processed successfully")
        return 1

    logger.info(f"Dataset shape: {df_pd.shape}")
    logger.info(f"Columns: {list(df_pd.columns)}")

    # Check target distribution
    if "y" in df_pd.columns:
        target_dist = df_pd["y"].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")

        # Check for class imbalance
        if len(target_dist) > 0:
            minority_pct = target_dist.min() / target_dist.sum() * 100
            logger.info(f"Minority class percentage: {minority_pct:.2f}%")

    # Save dataset for inspection and future use
    output_dir = Path("/home/nate/projects/nautilus_trader/data/tft_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as both CSV and Parquet
    csv_path = output_dir / "tft_training_data.csv"
    parquet_path = output_dir / "tft_training_data.parquet"

    df_pd.to_csv(csv_path, index=False)
    df_pd.to_parquet(parquet_path, index=False)

    logger.info("Saved training data to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Parquet: {parquet_path}")

    # Print sample of the data
    logger.info("\nFirst 5 rows of training data:")
    print(df_pd.head())

    logger.info("\nData types:")
    print(df_pd.dtypes)

    logger.info("\nBasic statistics:")
    print(df_pd.describe())

    # Check for required TFT columns
    required_cols = ["time_index", "instrument_id", "y"]
    missing_cols = [col for col in required_cols if col not in df_pd.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return 1

    # Identify feature categories for TFT
    static_categoricals = ["asset_class", "exchange"]
    static_reals = ["tick_size"]

    time_varying_known_reals = [
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "is_market_open",
        "is_premarket",
        "is_aftermarket",
    ]

    # All other numeric columns are time_varying_unknown_reals
    time_varying_unknown_reals = [
        col
        for col in df_pd.select_dtypes(include=[np.number]).columns
        if col not in ["time_index", "y"] + static_reals + time_varying_known_reals
    ]

    logger.info("\nTFT Feature Categories:")
    logger.info(f"  Static Categoricals: {static_categoricals}")
    logger.info(f"  Static Reals: {static_reals}")
    logger.info(f"  Time-Varying Known Reals: {time_varying_known_reals}")
    logger.info(f"  Time-Varying Unknown Reals: {time_varying_unknown_reals}")

    # Now train TFT if the teacher module is available
    try:
        from ml.training.teacher.tft_teacher import TFTTeacher
        from ml.training.teacher.tft_teacher import TFTTeacherConfig

        logger.info("\n" + "=" * 50)
        logger.info("Starting TFT Teacher Training")
        logger.info("=" * 50)

        # Create configuration (keep defaults for quick run)
        config = TFTTeacherConfig()

        # Initialize teacher
        teacher = TFTTeacher(
            config=config,
            max_encoder_length=20,  # Reduced for speed
            max_prediction_length=1,  # Binary classification
            hidden_size=32,  # Small model for quick training
            lstm_layers=2,
            dropout=0.1,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
        )

        # Split data for training/validation
        train_size = int(len(df_pd) * 0.8)
        train_df = df_pd.iloc[:train_size].copy()
        val_df = df_pd.iloc[train_size:].copy()

        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")

        # Train the model
        logger.info("\nTraining TFT model...")
        teacher.fit(train_df)

        # Make predictions on validation set
        logger.info("\nMaking predictions on validation set...")
        predictions = teacher.predict_logits(val_df.head(100))

        # Convert logits to probabilities
        if predictions is not None:
            probabilities = 1 / (1 + np.exp(-predictions))
            logger.info("Sample predictions (first 10):")
            for i in range(min(10, len(probabilities))):
                actual = val_df.iloc[i]["y"]
                prob = probabilities[i]
                logger.info(f"  Sample {i}: Actual={actual}, Predicted Prob={prob:.4f}")

        # Note: model saving/export handled via registry in production flows

        logger.info("\n✅ TFT training complete!")

    except ImportError as e:
        logger.warning(f"Could not import TFT teacher module: {e}")
        logger.info("Dataset prepared successfully. You can train the model manually.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Dataset prepared successfully. Check the error and try training manually.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
