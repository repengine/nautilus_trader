#!/usr/bin/env python3
"""
Example demonstrating DataScheduler with Prometheus metrics.

This script shows how to:
1. Initialize the DataScheduler with metrics enabled
2. Access metrics via the HTTP endpoint
3. Monitor data collection and feature computation

"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import requests

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def check_metrics_endpoint(port: int = 8000) -> None:
    """
    Check the metrics endpoint and display key metrics.

    Parameters
    ----------
    port : int
        Port number where metrics server is running

    """
    try:
        # Fetch metrics from endpoint
        response = requests.get(f"http://localhost:{port}/metrics")
        if response.status_code == 200:
            metrics_text = response.text

            # Parse and display key metrics
            print("\n=== Key Metrics ===")

            # Look for specific metrics
            for line in metrics_text.split("\n"):
                if line.startswith("nautilus_ml_"):
                    # Skip TYPE and HELP lines
                    if not line.startswith("# "):
                        print(line)

            print("\n=== Health Check ===")
            health_response = requests.get(f"http://localhost:{port}/health")
            if health_response.status_code == 200:
                print(f"Health: {health_response.json()}")
        else:
            print(f"Failed to fetch metrics: HTTP {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to metrics server on port {port}")
    except Exception as e:
        print(f"Error checking metrics: {e}")


def main() -> None:
    """
    Run example DataScheduler with metrics monitoring.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Create temporary directory for catalog
    temp_dir = tempfile.mkdtemp(prefix="nautilus_scheduler_")
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        # Initialize catalog
        catalog = ParquetDataCatalog(temp_dir)

        # Create scheduler configuration
        config = SchedulerConfig(
            symbols=["AAPL.XNAS", "MSFT.XNAS", "GOOGL.XNAS"],
            retention_days=30,
            databento=DatabentoConfig(
                dataset="EQUS.MINI",
                schema="ohlcv-1m",
                use_temporary_files=True,
                temp_data_dir=str(Path(temp_dir) / "databento_temp"),
            ),
            max_retries=2,
            retry_delay_seconds=1.0,
            feature_store_enabled=False,  # Disable for demo
        )

        # Initialize scheduler with metrics enabled
        logger.info("Starting DataScheduler with metrics on port 8000...")
        scheduler = DataScheduler(
            catalog=catalog,
            config=config,
            metrics_port=8000,
            start_metrics_server=True,
        )

        # Give metrics server time to start
        time.sleep(2)

        # Check initial metrics
        print("\n" + "=" * 60)
        print("INITIAL METRICS STATE")
        print("=" * 60)
        check_metrics_endpoint(8000)

        # Get scheduler status
        status = scheduler.get_status()
        print("\n=== Scheduler Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")

        # Simulate some operations to generate metrics
        print("\n" + "=" * 60)
        print("SIMULATING OPERATIONS")
        print("=" * 60)

        # Try to run a data collection (will fail without API key, but generates metrics)
        print("\nAttempting data collection (will fail without API key)...")
        try:
            scheduler.run_daily_update()
        except ValueError as e:
            logger.warning(f"Expected error: {e}")

        # Run cleanup (always succeeds)
        print("\nRunning data cleanup...")
        scheduler._clean_old_data()

        # Give metrics time to update
        time.sleep(1)

        # Check metrics after operations
        print("\n" + "=" * 60)
        print("METRICS AFTER OPERATIONS")
        print("=" * 60)
        check_metrics_endpoint(8000)

        # Example of accessing metrics programmatically
        print("\n" + "=" * 60)
        print("ACCESSING METRICS PROGRAMMATICALLY")
        print("=" * 60)

        from ml._imports import HAS_PROMETHEUS

        if HAS_PROMETHEUS:
            from ml._imports import generate_latest

            # Get raw metrics data
            metrics_bytes = generate_latest()
            metrics_lines = metrics_bytes.decode("utf-8").split("\n")

            # Count different metric types
            counters = sum(
                1 for line in metrics_lines if "_total" in line and not line.startswith("#")
            )
            histograms = sum(1 for line in metrics_lines if "_bucket" in line)
            gauges = sum(
                1
                for line in metrics_lines
                if "nautilus_ml_" in line
                and "_total" not in line
                and "_bucket" not in line
                and not line.startswith("#")
            )

            print("Metrics Summary:")
            print(f"  Counters: {counters}")
            print(f"  Histograms: {histograms} buckets")
            print(f"  Gauges: {gauges}")

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("\nMetrics server is still running on port 8000")
        print("You can view metrics at: http://localhost:8000/metrics")
        print("Press Ctrl+C to stop...")

        # Keep running to allow metric inspection
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nShutting down...")

    finally:
        # Clean up
        scheduler.stop()

        # Clean up temporary directory
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
