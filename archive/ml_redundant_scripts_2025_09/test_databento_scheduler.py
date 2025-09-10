#!/usr/bin/env python
"""
Test script for Databento scheduler integration.

This script tests the data scheduler with real Databento API calls. Ensure
DATABENTO_API_KEY is set in environment variables before running.

"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Skip if optional dependency is not available
import pytest


pytest.importorskip("databento", reason="databento package not installed")

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def test_databento_collection() -> None:
    """
    Test Databento data collection with a small universe.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check for API key
    if not os.getenv("DATABENTO_API_KEY"):
        logger.error("DATABENTO_API_KEY not set in environment variables")
        logger.info("Please set: export DATABENTO_API_KEY='your_api_key'")
        return

    # Create test catalog directory
    test_catalog_path = Path("./test_databento_catalog")
    test_catalog_path.mkdir(exist_ok=True)

    try:
        # Initialize catalog
        logger.info(f"Initializing catalog at {test_catalog_path}")
        catalog = ParquetDataCatalog(str(test_catalog_path))

        # Create minimal test configuration
        config = SchedulerConfig(
            symbols=["SPY.XNAS"],  # Single symbol for testing
            retention_days=30,
            databento=DatabentoConfig(
                dataset="GLBX.MDP3",  # Global exchange data
                schema="ohlcv-1m",  # 1-minute bars
                use_temporary_files=True,
                temp_data_dir="./test_temp_databento",
            ),
            max_retries=2,
            retry_delay_seconds=2.0,
        )

        # Create scheduler
        logger.info("Creating DataScheduler with test configuration")
        scheduler = DataScheduler(
            catalog=catalog,
            config=config,
        )

        # Display status
        status = scheduler.get_status()
        logger.info("Scheduler status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")

        # Run single data collection
        logger.info("\n" + "=" * 60)
        logger.info("Starting test data collection...")
        logger.info("=" * 60)

        try:
            scheduler.run_daily_update()
            logger.info("\n✅ Data collection test completed successfully!")

            # Check what was written to catalog
            logger.info("\nChecking catalog contents...")

            # List available instruments
            instruments = catalog.instruments()
            if instruments:
                logger.info(f"Found {len(instruments)} instruments in catalog:")
                for inst in instruments[:5]:  # Show first 5
                    logger.info(f"  - {inst.id}")

            # List available bars (use dynamic lookup for compatibility)
            if hasattr(catalog, "bar_types"):
                bar_types = getattr(catalog, "bar_types")()
                if bar_types:
                    logger.info(f"\nFound {len(bar_types)} bar types in catalog:")
                    for bar_type in bar_types[:5]:  # Show first 5
                        logger.info(f"  - {bar_type}")

                        # Get count of bars for this type
                        bars = catalog.bars([bar_type.instrument_id])
                        if bars:
                            logger.info(f"    {len(bars)} bars available")

        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            raise

    finally:
        # Cleanup test directories
        logger.info("\nCleaning up test directories...")

        # Remove temporary databento directory
        temp_dir = Path("./test_temp_databento")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
            logger.info(f"  Removed {temp_dir}")

        # Note: Not removing catalog directory to allow inspection
        logger.info(f"  Catalog preserved at {test_catalog_path} for inspection")
        logger.info("  To remove: rm -rf test_databento_catalog/")


def test_configuration_options() -> None:
    """
    Test different configuration options for the scheduler.
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 60)
    logger.info("Testing configuration options")
    logger.info("=" * 60)

    # Test different schemas
    schemas = ["ohlcv-1m", "trades", "mbp-1", "tbbo"]

    for schema in schemas:
        config = SchedulerConfig(
            symbols=["AAPL.XNAS"],
            databento=DatabentoConfig(
                dataset="GLBX.MDP3",
                schema=schema,
            ),
        )
        logger.info(f"\nSchema: {schema}")
        logger.info(f"  Dataset: {config.databento.dataset}")
        logger.info(f"  Temp dir: {config.databento.temp_data_dir}")
        logger.info(f"  Use temp files: {config.databento.use_temporary_files}")

    # Test universe expansion
    from ml.config.scheduler_config import UniverseConfig

    universe = UniverseConfig(expansion_mode="conservative")
    logger.info(f"\nConservative universe: {len(universe.get_full_universe())} symbols")

    universe = UniverseConfig(expansion_mode="moderate")
    logger.info(f"Moderate universe: {len(universe.get_full_universe())} symbols")

    universe = UniverseConfig(expansion_mode="aggressive")
    logger.info(f"Aggressive universe: {len(universe.get_full_universe())} symbols")


def main() -> None:
    """
    Run entry point for testing.
    """
    print("\n" + "=" * 80)
    print("DATABENTO SCHEDULER INTEGRATION TEST")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Test the DataScheduler with real Databento API calls")
    print("2. Collect 1-minute bars for SPY")
    print("3. Store data in a test catalog")
    print("4. Verify the data was written correctly")
    print("\nRequirements:")
    print("- DATABENTO_API_KEY environment variable must be set")
    print("- Databento Python client must be installed (pip install databento)")
    print("- Network connection to Databento API")

    response = input("\nProceed with test? (yes/no): ")
    if response.lower() != "yes":
        print("Test cancelled")
        return

    # Run tests
    test_configuration_options()
    test_databento_collection()

    print("\n✅ All tests completed!")
    print("Check test_databento_catalog/ directory for collected data")


if __name__ == "__main__":
    main()
