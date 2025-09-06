#!/usr/bin/env python3
"""
Test batching and combining functionality.
"""
import warnings
from datetime import datetime

import databento as db
import pandas as pd


warnings.filterwarnings("ignore")


def test_batch_combine():
    """
    Test downloading multiple batches and combining them.
    """
    try:
        client = db.Historical()
        print("✅ Connected to Databento")

        # Test multiple small batches
        batches = [
            (datetime(2025, 8, 26), datetime(2025, 8, 27)),
            (datetime(2025, 8, 28), datetime(2025, 8, 29)),
        ]

        all_dfs = []

        for start, end in batches:
            print(f"Downloading batch: {start.date()} to {end.date()}")
            df = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=["AAPL"],
                schema="ohlcv-1d",
                start=start,
                end=end,
            ).to_df()

            print(f"  Got {len(df)} records")
            if len(df) > 0:
                all_dfs.append(df)
                print(f"  Index type: {type(df.index)}")
                print(f"  Index name: {df.index.name}")

        if all_dfs:
            print("\nCombining batches...")
            combined_df = pd.concat(all_dfs)  # Keep the index (ts_event)
            print(f"Combined shape: {combined_df.shape}")
            print(f"Index after concat: {combined_df.index.name}")

            # Sort and deduplicate by ts_event index
            combined_df = combined_df.sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

            print(f"After dedup: {combined_df.shape}")
            print("\nFirst few rows:")
            print(combined_df.head())

            # Test save
            combined_df.to_parquet("test_combined.parquet")
            print("✅ Saved successfully")

            return True
        else:
            print("❌ No data to combine")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_batch_combine()
