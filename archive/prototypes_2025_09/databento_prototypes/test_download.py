#!/usr/bin/env python3
"""
Test script to verify download works correctly.
"""
import warnings
from datetime import datetime

import databento as db


warnings.filterwarnings("ignore")

def test_single_download():
    """Test a single small download to check format."""
    try:
        client = db.Historical()
        print("✅ Connected to Databento")

        # Test small download - use a date that definitely has data
        end_date = datetime(2025, 8, 30)
        start_date = datetime(2025, 8, 29)  # Just 2 days

        print(f"Testing download: ohlcv-1d for AAPL from {start_date.date()} to {end_date.date()}")

        df = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=["AAPL"],
            schema="ohlcv-1d",
            start=start_date,
            end=end_date,
        ).to_df()

        print(f"Downloaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Index: {df.index.name}")
        print("\nFirst few rows:")
        print(df.head())

        # Test save
        df.to_parquet("test_download.parquet")
        print("✅ Saved successfully to test_download.parquet")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_single_download()
