#!/usr/bin/env python
"""
Collect missing L1 trades for 2023-2024.
The enhanced collector only got 2025 data due to the XNAS.BASIC error.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path


# Priority symbols for multi-year L1 trades
PRIORITY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "XLF", "XLK", "XLE",
    "XLV", "VXX", "UVXY", "TLT", "GLD"
]

# Extended symbols for 1 year
EXTENDED_SYMBOLS = [
    "HD", "PFE", "CVX", "MRK", "ABBV", "DIS", "PEP", "KO",
    "NKE", "MCD", "TMO", "LLY", "CAT", "BA", "HON", "UNP",
    "AMD", "INTC", "QCOM", "CRM", "ADBE", "NFLX", "AVGO",
    "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"
]

def collect_l1_trades() -> None:
    """Collect missing L1 trades for 2023-2024."""
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")

    # Import Databento lazily to avoid module import side-effects during doctest collection
    import databento as db  # local import

    client = db.Historical(api_key)
    data_dir = Path("/home/nate/projects/nautilus_trader/data/enhanced")

    # Fixed end date (last available data) kept for reference
    # end_date = datetime(2025, 8, 16)

    print("=" * 80)
    print("COLLECTING MISSING L1 TRADES (2023-2024)")
    print("=" * 80)

    total_collected = 0
    total_size_gb = 0.0

    # Collect 2 years for priority symbols
    print("\n📈 PRIORITY SYMBOLS (2023-2024)")
    print("-" * 40)

    for i, symbol in enumerate(PRIORITY_SYMBOLS, 1):
        print(f"\n[{i}/{len(PRIORITY_SYMBOLS)}] {symbol}:")
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        # Collect 2024
        year_2024_file = symbol_dir / "trades_2024.parquet"
        if not year_2024_file.exists():
            try:
                print("  Collecting trades for 2024...")
                start_2024 = datetime(2024, 8, 16)
                end_2024 = datetime(2025, 8, 15)

                data = client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_2024,
                    end=end_2024,
                    schema="trades",
                    limit=10000000
                )

                df = data.to_df()
                if not df.empty:
                    df.to_parquet(year_2024_file)
                    size_gb = year_2024_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    total_collected += 1
                    print(f"  ✓ 2024: {len(df):,} trades, {size_gb:.3f} GB")
                else:
                    print("  ✗ No trades for 2024")
            except Exception as e:
                print(f"  ✗ Error for 2024: {str(e)[:100]}")

            time.sleep(1.0)  # Rate limit
        else:
            print("  ✓ 2024 already collected")

        # Collect 2023
        year_2023_file = symbol_dir / "trades_2023.parquet"
        if not year_2023_file.exists():
            try:
                print("  Collecting trades for 2023...")
                start_2023 = datetime(2023, 8, 17)
                end_2023 = datetime(2024, 8, 15)

                data = client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_2023,
                    end=end_2023,
                    schema="trades",
                    limit=10000000
                )

                df = data.to_df()
                if not df.empty:
                    df.to_parquet(year_2023_file)
                    size_gb = year_2023_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    total_collected += 1
                    print(f"  ✓ 2023: {len(df):,} trades, {size_gb:.3f} GB")
                else:
                    print("  ✗ No trades for 2023")
            except Exception as e:
                print(f"  ✗ Error for 2023: {str(e)[:100]}")

            time.sleep(1.0)  # Rate limit
        else:
            print("  ✓ 2023 already collected")

        # Check storage every 5 symbols
        if i % 5 == 0:
            current_gb = sum(
                f.stat().st_size for f in data_dir.rglob("*.parquet")
            ) / (1024**3)
            print(f"\nProgress: {total_collected} files collected, {current_gb:.1f} GB total")

            # Stop if approaching 1TB limit
            if current_gb > 950:
                print("\n⚠️  Approaching 1TB limit, stopping collection")
                break

    # Collect 2024 for extended symbols
    print("\n\n📊 EXTENDED SYMBOLS (2024 only)")
    print("-" * 40)

    for i, symbol in enumerate(EXTENDED_SYMBOLS[:30], 1):  # Limit to 30
        print(f"\n[{i}/30] {symbol}:")
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        year_2024_file = symbol_dir / "trades_2024.parquet"
        if not year_2024_file.exists():
            try:
                print("  Collecting trades for 2024...")
                start_2024 = datetime(2024, 8, 16)
                end_2024 = datetime(2025, 8, 15)

                data = client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_2024,
                    end=end_2024,
                    schema="trades",
                    limit=10000000
                )

                df = data.to_df()
                if not df.empty:
                    df.to_parquet(year_2024_file)
                    size_gb = year_2024_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    total_collected += 1
                    print(f"  ✓ 2024: {len(df):,} trades, {size_gb:.3f} GB")
                else:
                    print("  ✗ No trades for 2024")
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")

            time.sleep(0.5)  # Rate limit
        else:
            print("  ✓ 2024 already collected")

        # Check storage
        if i % 10 == 0:
            current_gb = sum(
                f.stat().st_size for f in data_dir.rglob("*.parquet")
            ) / (1024**3)
            if current_gb > 950:
                print("\n⚠️  Approaching 1TB limit, stopping")
                break

    # Final summary
    current_gb = sum(
        f.stat().st_size for f in data_dir.rglob("*.parquet")
    ) / (1024**3)

    print("\n" + "=" * 80)
    print("L1 TRADES COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Files collected: {total_collected}")
    print(f"New data size: {total_size_gb:.2f} GB")
    print(f"Total storage: {current_gb:.2f} GB")
    print("\n✅ Multi-year L1 trades now available for TFT training!")


if __name__ == "__main__":
    print("COLLECTING MISSING L1 TRADES")
    print("This will collect 2023-2024 trade data")
    print("Estimated size: ~400-500GB")
    print("Estimated time: 2-3 hours")
    print()

    import sys
    if not sys.stdin.isatty():
        print("Auto-proceeding in background mode...")
    else:
        response = input("Proceed? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled")
            sys.exit(0)

    collect_l1_trades()
