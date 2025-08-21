#!/usr/bin/env python3
"""
Example of using TFT Dataset Builder with FeatureStore integration.

This script demonstrates how to use the TFT Dataset Builder with the FeatureStore to
ensure training/inference parity.

"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from ml.config.base import MLFeatureConfig
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.features.engineering import FeatureConfig
from ml.stores.feature_store import FeatureStore
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def main() -> None:
    """
    Demonstrate TFT Dataset Builder with FeatureStore.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Create temporary directory for catalog
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog_path = Path(temp_dir) / "catalog"
        catalog_path.mkdir()

        # Initialize catalog
        catalog = ParquetDataCatalog(str(catalog_path))

        # Configure features
        feature_config = MLFeatureConfig(
            lookback_window=20,
            return_periods=[1, 5, 10],
            rsi_period=14,
            bb_period=20,
        )

        # Get PostgreSQL connection from environment or use default
        connection_string = os.getenv(
            "NAUTILUS_DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        )

        # Initialize FeatureStore (optional - will work without it)
        feature_store = None
        try:
            feature_store = FeatureStore(
                connection_string=connection_string,
                feature_config=FeatureConfig(
                    lookback_window=feature_config.lookback_window,
                    return_periods=feature_config.return_periods,
                    rsi_period=feature_config.rsi_period,
                    bb_period=feature_config.bb_period,
                ),
            )
            logger.info("FeatureStore initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize FeatureStore: {e}")
            logger.info("Will use direct feature computation instead")

        # Define symbols to train on
        symbols = ["SPY", "QQQ", "IWM"]

        # Initialize TFT Dataset Builder
        builder = TFTDatasetBuilder(
            catalog=catalog,
            symbols=symbols,
            feature_config=feature_config,
            feature_store=feature_store,  # Optional - works with or without
        )

        logger.info(f"Initialized TFT Dataset Builder with {len(symbols)} symbols")

        # Prepare training data
        # The builder will automatically use FeatureStore if available,
        # otherwise fall back to direct computation
        try:
            # Define time range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            logger.info(f"Preparing training data from {start_date} to {end_date}")

            # Build dataset - automatically selects best method
            dataset = builder.prepare_training_data(
                start=start_date,
                end=end_date,
                horizon_minutes=15,
                min_return_threshold=0.001,
                lookback_periods=30,
                use_polars=True,  # Return Polars DataFrame
            )

            # Display results
            logger.info(f"Dataset shape: {dataset.shape}")
            logger.info(f"Columns: {dataset.columns}")

            # Show first few rows
            logger.info("First 5 rows:")
            logger.info(dataset.head(5))

            # Alternative method - using build_training_dataset (backward compatible)
            logger.info("\nUsing build_training_dataset method (backward compatible):")
            dataset2 = builder.build_training_dataset(
                horizon_minutes=15,
                min_return_threshold=0.001,
                start=start_date,
                end=end_date,
            )

            logger.info(f"Dataset2 shape: {dataset2.shape}")

            # Demonstrate feature source logging
            if feature_store:
                logger.info("\n✅ Training data prepared using FeatureStore")
                logger.info("This ensures perfect training/inference parity")
            else:
                logger.info("\n⚠️ Training data prepared using direct computation")
                logger.info("Consider setting up FeatureStore for production use")

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise

        # Example: Load features from specific instruments
        if feature_store:
            try:
                logger.info("\nDemonstrating instrument-specific loading:")

                # Load for specific instruments
                specific_dataset = builder.prepare_training_data_from_store(
                    instrument_ids=["SPY.NYSE", "QQQ.NASDAQ"],
                    start=start_date,
                    end=end_date,
                )

                logger.info(f"Specific dataset shape: {specific_dataset.shape}")

            except ValueError as e:
                logger.warning(f"Could not load from FeatureStore: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TFT Dataset Builder Summary:")
        logger.info(f"  - Symbols: {symbols}")
        logger.info(f"  - FeatureStore: {'Enabled' if feature_store else 'Disabled'}")
        logger.info(
            f"  - Feature source: {'FeatureStore' if feature_store else 'Direct computation'}",
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
