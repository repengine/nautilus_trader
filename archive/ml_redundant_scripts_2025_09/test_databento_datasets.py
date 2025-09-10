#!/usr/bin/env python3
"""
Test different Databento datasets to find the right one for our symbols.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest


# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_datasets():
    """
    Test different datasets to find what works.
    """
    # Setup
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        pytest.skip("databento API key not set; skipping")

    pytest.importorskip("databento", reason="databento package not installed")
    import databento as db

    client = db.Historical(api_key)

    # Test parameters - use a date we know has data
    test_date = datetime(2024, 1, 15)  # Recent past date
    end_date = datetime(2024, 1, 16)

    # Symbols to test
    test_symbols = ["SPY", "AAPL", "MSFT"]

    # Datasets to try
    datasets = [
        "XNAS.ITCH",  # Nasdaq TotalView-ITCH
        "GLBX.MDP3",  # CME Globex MDP 3.0
        "OPRA.PILLAR",  # OPRA
        "DBEQ.BASIC",  # Databento Equities Basic
        "BATS.PITCH",  # Cboe Europe
    ]

    print("Testing Databento datasets...")
    print("=" * 60)

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 40)

        for symbol in test_symbols:
            try:
                # Try to get just 1 day of data
                data = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    start=test_date,
                    end=end_date,
                    schema="ohlcv-1d",
                    limit=10,
                )

                df = data.to_df()

                if not df.empty:
                    print(f"  ✓ {symbol}: {len(df)} rows")
                else:
                    print(f"  ✗ {symbol}: No data")

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"  ✗ {symbol}: Error - {error_msg}")

    # Now test what we know works from the pilot
    print("\n" + "=" * 60)
    print("Testing known working configuration (from pilot):")
    print("-" * 40)

    try:
        # This is what worked in the collector.py
        data = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            schema="ohlcv-1d",
            limit=100,
        )

        df = data.to_df()
        print(f"XNAS.ITCH with AAPL: {len(df)} rows")

        if not df.empty:
            print("\nSample data:")
            print(df.head(3))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_datasets()
