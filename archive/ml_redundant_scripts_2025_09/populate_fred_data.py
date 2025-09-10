#!/usr/bin/env python3
"""
Script to populate initial FRED economic data.

This script fetches historical FRED data and stores it in the DataStore
with proper registration in the DataRegistry.

Usage:
    python ml/scripts/populate_fred_data.py [OPTIONS]

Options:
    --api-key TEXT         FRED API key (defaults to FRED_API_KEY env var)
    --backfill-years INT   Number of years to backfill (default: 10)
    --cache-dir PATH       Cache directory (default: /tmp/fred_cache)
    --db-path PATH         Database path (default: ml/stores/ml_data.db)
    --registry-path PATH   Registry path (default: ml/registry/data)
    --update-only          Only fetch recent data for updates
    --dry-run              Show what would be fetched without storing

Examples:
    # Full backfill with 10 years of history
    python ml/scripts/populate_fred_data.py

    # Update with last 30 days only
    python ml/scripts/populate_fred_data.py --update-only

    # Custom backfill period
    python ml/scripts/populate_fred_data.py --backfill-years 5

    # Dry run to see what would be fetched
    python ml/scripts/populate_fred_data.py --dry-run

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.data.loaders import FREDConfig
from ml.data.loaders import FREDDataLoader
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Populate FRED economic data for ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FRED API key (defaults to FRED_API_KEY env var)",
    )

    parser.add_argument(
        "--backfill-years",
        type=int,
        default=10,
        help="Number of years to backfill (default: 10)",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/fred_cache"),
        help="Cache directory (default: /tmp/fred_cache)",
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("ml/stores/ml_data.db"),
        help="Database path (default: ml/stores/ml_data.db)",
    )

    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("ml/registry/data"),
        help="Registry path (default: ml/registry/data)",
    )

    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only fetch recent data for updates",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without storing",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for FRED data population.
    """
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Configure FRED loader
    config = FREDConfig(
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        backfill_years=args.backfill_years,
    )

    try:
        loader = FREDDataLoader(config)
    except (ImportError, ValueError) as e:
        logger.error(f"Failed to initialize FRED loader: {e}")
        sys.exit(1)

    # Log configuration
    logger.info("=" * 60)
    logger.info("FRED Data Population Script")
    logger.info("=" * 60)
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Backfill years: {config.backfill_years}")
    logger.info(f"Update only: {args.update_only}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Indicators to fetch: {len(loader.indicators)}")
    logger.info("=" * 60)

    # Determine date range
    end_date = datetime.now()

    if args.update_only:
        # For updates, only fetch last 30 days
        start_date = end_date - timedelta(days=30)
        logger.info(f"Update mode: fetching from {start_date.date()} to {end_date.date()}")
    else:
        # Full backfill
        start_date = end_date - timedelta(days=365 * config.backfill_years)
        logger.info(f"Backfill mode: fetching from {start_date.date()} to {end_date.date()}")

    # List indicators by category
    categories = {}
    for indicator in loader.indicators:
        if indicator.category not in categories:
            categories[indicator.category] = []
        categories[indicator.category].append(indicator)

    logger.info("\nIndicators by category:")
    for category, indicators in sorted(categories.items()):
        logger.info(f"  {category}: {len(indicators)} indicators")
        for ind in indicators[:3]:  # Show first 3
            logger.info(f"    - {ind.series_id}: {ind.name}")
        if len(indicators) > 3:
            logger.info(f"    ... and {len(indicators) - 3} more")

    # Fetch all indicators
    logger.info("\n" + "=" * 60)
    logger.info("Fetching indicators...")
    logger.info("=" * 60)

    try:
        data = loader.fetch_all_indicators(
            start_date=start_date,
            end_date=end_date,
            use_cache=not args.update_only,  # Don't use cache for updates
        )
    except Exception as e:
        logger.error(f"Failed to fetch indicators: {e}")
        sys.exit(1)

    # Show summary
    logger.info("\n" + "=" * 60)
    logger.info("Fetch Summary")
    logger.info("=" * 60)
    logger.info(f"Successfully fetched: {len(data)} indicators")

    total_rows = 0
    for series_id, df in data.items():
        rows = len(df)
        total_rows += rows
        indicator = next((i for i in loader.indicators if i.series_id == series_id), None)
        name = indicator.name if indicator else series_id
        logger.debug(f"  {series_id} ({name}): {rows} rows")

    logger.info(f"Total data points: {total_rows:,}")

    # Combine indicators
    combined_df = loader.combine_indicators(data)
    logger.info(f"Combined DataFrame: {len(combined_df)} rows x {len(combined_df.columns)} columns")

    # Show data preview
    if not combined_df.is_empty():
        logger.info("\nData preview (last 5 rows):")
        preview_cols = [c for c in combined_df.columns if c not in ["timestamp", "timestamp_ns"]][
            :5
        ]
        if "timestamp" in combined_df.columns:
            preview_df = combined_df.select(["timestamp"] + preview_cols).tail(5)
        else:
            preview_df = combined_df.select(preview_cols).tail(5)
        logger.info(f"\n{preview_df}")

    # Store in DataStore (unless dry run)
    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("Storing in DataStore...")
        logger.info("=" * 60)

        try:
            # Initialize stores
            db_path = args.db_path.absolute()
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create connection string
            if db_path.suffix == ".db":
                # SQLite
                connection_string = f"sqlite:///{db_path}"
            else:
                # Assume PostgreSQL
                connection_string = os.getenv(
                    "ML_DB_CONNECTION",
                    "postgresql://user:password@localhost:5432/nautilus_ml",
                )

            # Initialize stores
            feature_store = FeatureStore(connection_string=connection_string)
            model_store = ModelStore(connection_string=connection_string)
            strategy_store = StrategyStore(connection_string=connection_string)

            # Initialize DataRegistry
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=args.registry_path,
            )
            data_registry = DataRegistry(
                registry_path=args.registry_path,
                persistence_config=persistence_config,
            )

            # Initialize DataStore
            data_store = DataStore(
                feature_store=feature_store,
                model_store=model_store,
                strategy_store=strategy_store,
                data_registry=data_registry,
            )

            # Store indicators
            loader.store_indicators(data_store, data_registry, data)

            logger.info("✓ Successfully stored FRED data in DataStore")

            # Show registry info
            dataset_id = "fred_economic_indicators"
            manifest = data_registry.get_manifest(dataset_id)
            if manifest:
                logger.info("\nDataset registered:")
                logger.info(f"  ID: {manifest.dataset_id}")
                logger.info(f"  Type: {manifest.dataset_type}")
                logger.info(f"  Source: {manifest.source}")
                logger.info(f"  Columns: {len(manifest.contract.columns)}")
                logger.info(f"  Location: {manifest.location}")

        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            sys.exit(1)

    else:
        logger.info("\n✓ Dry run completed - no data stored")

    logger.info("\n" + "=" * 60)
    logger.info("Script completed successfully")
    logger.info("=" * 60)

    # Print next steps
    if not args.dry_run:
        logger.info("\nNext steps:")
        logger.info("1. Verify data in database:")
        logger.info(
            f"   sqlite3 {args.db_path} 'SELECT COUNT(*) FROM ingestion_data;'",
        )
        logger.info("2. Schedule regular updates:")
        logger.info("   python ml/scripts/populate_fred_data.py --update-only")
        logger.info("3. Use data in ML pipeline:")
        logger.info("   from ml.stores.data_store import DataStore")
        logger.info("   data = data_store.read_ingestion('fred_economic_indicators')")


if __name__ == "__main__":
    main()
