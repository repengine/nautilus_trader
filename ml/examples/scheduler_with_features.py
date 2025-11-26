#!/usr/bin/env python3
"""
Example usage of DataScheduler with FeatureStore integration.

This script demonstrates how to set up and use the DataScheduler with automated feature
computation and storage.

"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.common.logging_config import configure_logging
from ml.data.scheduler import DataScheduler
from ml.features import FeatureConfig
from ml.features import FeatureEngineer
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def main() -> None:
    """
    Demonstrate DataScheduler with FeatureStore integration.
    """
    # Setup logging
    configure_logging(level="INFO")
    logger = logging.getLogger(__name__)

    # 1. Setup catalog for data storage
    catalog_path = Path("./data/catalog")
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    logger.info(f"Initialized catalog at: {catalog_path}")

    # 2. Configure feature engineering
    feature_config = FeatureConfig(
        lookback_window=50,  # Use 50 bars for feature calculation
        return_periods=[1, 5, 10, 20],  # Multiple return horizons
        momentum_periods=[5, 10, 20],  # Momentum indicators
        rsi_period=14,  # RSI configuration
        bb_period=20,  # Bollinger Bands
        bb_std=2.0,
        atr_period=20,  # Average True Range
        normalize_features=True,  # Normalize for ML models
        average_volume=1_000_000.0,  # For volume normalization
    )
    feature_engineer = FeatureEngineer(feature_config)
    logger.info("Initialized feature engineer with custom configuration")

    # 3. Configure scheduler with feature store
    scheduler_config = SchedulerConfig(
        # Symbol universe
        symbols=[
            "SPY.XNAS",  # S&P 500 ETF
            "QQQ.XNAS",  # NASDAQ ETF
            "IWM.XNAS",  # Russell 2000 ETF
            "AAPL.XNAS",  # Apple
            "MSFT.XNAS",  # Microsoft
        ],
        # Data collection settings
        collection_time="04:00",  # 4 AM UTC
        retention_days=90,  # Keep 3 months of data
        # Databento configuration
        databento=DatabentoConfig(
            dataset="GLBX.MDP3",  # CME Globex
            schema="ohlcv-1m",  # 1-minute bars
            stype_in="raw_symbol",
            use_temporary_files=True,
            temp_data_dir="./temp_databento",
        ),
        # Feature store settings
        feature_store_enabled=True,
        feature_store_connection=os.getenv(
            "NAUTILUS_DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        ),
        # Data types to collect
        enable_l2_depth=False,  # L2 order book (requires subscription)
        enable_trades=False,  # Trade ticks
        enable_quotes=False,  # Quote ticks
        # Retry logic
        max_retries=3,
        retry_delay_seconds=5.0,
    )

    # 4. Create scheduler with all components
    scheduler = DataScheduler(
        catalog=catalog,
        config=scheduler_config,
        feature_engineer=feature_engineer,
    )

    # 5. Display scheduler status
    status = scheduler.get_status()
    logger.info("Scheduler Status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

    # 6. Run a manual update (for testing)
    # Note: This requires DATABENTO_API_KEY environment variable
    if os.getenv("DATABENTO_API_KEY"):
        logger.info("Running manual data update...")
        try:
            scheduler.run_daily_update()
            logger.info("Daily update completed successfully!")
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
    else:
        logger.warning(
            "DATABENTO_API_KEY not set. Skipping data collection.\n"
            "To enable data collection, set: export DATABENTO_API_KEY=your_key",
        )

    # 7. Schedule automated updates (commented out for demo)
    # scheduler.schedule_updates("0 4 * * *")  # Daily at 4 AM
    # logger.info("Scheduled automated daily updates")

    # 8. Example of accessing computed features
    if scheduler._feature_store is not None:
        logger.info("\nFeature Store is initialized and ready for:")
        logger.info("  - Storing batch-computed features from historical data")
        logger.info("  - Providing features for ML model training")
        logger.info("  - Ensuring training/inference feature parity")
        logger.info("  - Tracking feature computation metrics")

        # Example query (would work with actual data)
        # from datetime import datetime, timedelta
        # features = scheduler._feature_store.get_training_data(
        #     instrument_id="SPY.NASDAQ",
        #     start=datetime.now() - timedelta(days=30),
        #     end=datetime.now(),
        # )
    else:
        logger.info("\nFeature Store not initialized (feature computation disabled)")


if __name__ == "__main__":
    main()
