#!/usr/bin/env python3
"""
Auto Gap Filler - Analyzes existing tier1 data and downloads ALL missing data from Databento subscription.

This script:
1. Scans your existing data/tier1 directory
2. Identifies ALL gaps between what you have and what your subscription allows
3. Downloads ONLY the missing data (no redownloading)
4. Works with your existing file structure
"""
import argparse
import logging
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import databento as db
import pandas as pd


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Setup logging - quieter output
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


def get_subscription_entitlements() -> dict[str, tuple[datetime, datetime]]:
    """
    Get what you should have based on Databento subscription.
    """
    today = datetime.now().date()
    yesterday = datetime.combine(today - timedelta(days=1), datetime.min.time())
    yesterday = yesterday.replace(tzinfo=None)

    return {
        "core": (yesterday - timedelta(days=365 * 7), yesterday),  # 7 years OHLCV
        "l1": (yesterday - timedelta(days=365), yesterday),  # 1 year L1
        "l2": (yesterday - timedelta(days=30), yesterday),  # 30 days L2
    }


def get_existing_date_range(file_path: Path) -> tuple[datetime, datetime] | None:
    """
    Get date range from existing file, handling both old and new formats.
    """
    if not file_path.exists():
        return None

    try:
        # Read only a small sample for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_parquet(file_path, nrows=1000)  # Only read first 1000 rows for speed

        if len(df) == 0:
            return None

        # Handle different timestamp formats
        if "ts_event" in df.columns:
            # New format: ts_event column (nanoseconds) - need to read full file for date range
            df_full = pd.read_parquet(file_path, columns=["ts_event"])
            min_ts = df_full["ts_event"].min()
            max_ts = df_full["ts_event"].max()
            min_date = pd.to_datetime(min_ts, unit="ns", utc=True).tz_localize(None)
            max_date = pd.to_datetime(max_ts, unit="ns", utc=True).tz_localize(None)
        elif df.index.name == "ts_event" or (
            hasattr(df.index, "name") and "ts_event" in str(df.index.name)
        ):
            # Old format: ts_event index (datetime) - need to read full file to get complete range
            df_full = pd.read_parquet(file_path)
            min_date = df_full.index.min()
            max_date = df_full.index.max()
            # Convert to timezone-naive
            if hasattr(min_date, "tz_localize") and min_date.tz is not None:
                min_date = min_date.tz_localize(None)
                max_date = max_date.tz_localize(None)
        else:
            return None

        # Convert to python datetime
        if hasattr(min_date, "to_pydatetime"):
            min_date = min_date.to_pydatetime()
            max_date = max_date.to_pydatetime()
        elif hasattr(min_date, "replace") and min_date.tzinfo is not None:
            min_date = min_date.replace(tzinfo=None)
            max_date = max_date.replace(tzinfo=None)

        return min_date, max_date

    except Exception:
        return None


def calculate_gaps(
    existing_range: tuple[datetime, datetime] | None,
    target_range: tuple[datetime, datetime],
) -> list[tuple[datetime, datetime]]:
    """
    Calculate missing date ranges.
    """
    if not existing_range:
        return [target_range]

    target_start, target_end = target_range
    existing_start, existing_end = existing_range
    gaps = []

    # Gap before existing data
    if target_start < existing_start:
        gaps.append((target_start, existing_start - timedelta(days=1)))

    # Gap after existing data
    if existing_end < target_end:
        gaps.append((existing_end + timedelta(days=1), target_end))

    return gaps


def analyze_symbol_gaps(symbol_dir: Path, entitlements: dict) -> dict:
    """
    Analyze what's missing for a specific symbol.
    """
    symbol = symbol_dir.name
    analysis = {
        "symbol": symbol,
        "existing": {},
        "gaps": {},
        "downloads_needed": [],
    }

    # File pattern mapping: (file_pattern, schema_name, data_type, output_filename)
    file_patterns = [
        ("daily_*.parquet", "ohlcv-1d", "core", "bars_ohlcv-1d.parquet"),
        ("hourly_*.parquet", "ohlcv-1m", "core", "bars_ohlcv-1m.parquet"),
        ("l1/*bbo*.parquet", "tbbo", "l1", "bbo_tbbo.parquet"),
        ("l1/*trades*.parquet", "trades", "l1", "trades.parquet"),
        ("l2/*.parquet", "mbp-10", "l2", "mbp-10.parquet"),
    ]

    for pattern, schema, data_type, output_file in file_patterns:
        files = list(symbol_dir.glob(pattern))

        if files:
            # Get existing data range
            existing_range = get_existing_date_range(files[0])
            if existing_range:
                analysis["existing"][schema] = {
                    "file": files[0],
                    "range": existing_range,
                    "days": (existing_range[1] - existing_range[0]).days,
                }

                # Calculate gaps
                target_range = entitlements[data_type]
                gaps = calculate_gaps(existing_range, target_range)

                if gaps:
                    analysis["gaps"][schema] = gaps
                    for gap_start, gap_end in gaps:
                        analysis["downloads_needed"].append(
                            {
                                "schema": schema,
                                "start": gap_start,
                                "end": gap_end,
                                "days": (gap_end - gap_start).days,
                                "output_file": symbol_dir / output_file,
                                "data_type": data_type,
                            },
                        )
        else:
            # No existing data - need full range
            if data_type in entitlements:
                target_start, target_end = entitlements[data_type]
                analysis["gaps"][schema] = [(target_start, target_end)]
                analysis["downloads_needed"].append(
                    {
                        "schema": schema,
                        "start": target_start,
                        "end": target_end,
                        "days": (target_end - target_start).days,
                        "output_file": symbol_dir / output_file,
                        "data_type": data_type,
                    },
                )

    return analysis


def download_gap(client: db.Historical, download_info: dict, dry_run: bool = False) -> bool:
    """
    Download a specific gap.
    """
    symbol = download_info["output_file"].parent.name
    schema = download_info["schema"]
    start_date = download_info["start"]
    end_date = download_info["end"]
    output_file = download_info["output_file"]

    if dry_run:
        print(
            f"  [DRY RUN] Would download {schema} for {symbol}: {start_date.date()} to {end_date.date()} ({download_info['days']} days)",
        )
        return True

    try:
        print(
            f"Downloading {schema} for {symbol}: {start_date.date()} to {end_date.date()} ({download_info['days']} days)",
        )

        # Download data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = client.timeseries.get_range(
                dataset="EQUS.MINI",
                symbols=[symbol],
                schema=schema,
                start=start_date,
                end=end_date,
            ).to_df()

        if df.empty:
            print(f"  ⚠️  No data returned for {symbol} {schema}")
            return False

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary file first
        temp_file = output_file.with_suffix(".tmp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df.to_parquet(temp_file)

        # Atomic rename
        temp_file.rename(output_file)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved {len(df):,} records ({size_mb:.1f} MB) to {output_file.name}")

        return True

    except Exception as e:
        error_msg = str(e)
        if any(code in error_msg for code in ("403", "license", "unauthorized")):
            print(f"  ⚠️  License restriction for {symbol} {schema}")
            return True  # Not an error - just not licensed

        print(f"  ❌ Failed to download {symbol} {schema}: {error_msg}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto Gap Filler - Download ALL missing data from your Databento subscription",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tier1"),
        help="Directory to scan for existing data",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to process (default: all found in directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit processing to first N symbols (for testing)",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory {args.data_dir} does not exist")
        return

    # Initialize Databento client
    client = None
    if not args.dry_run:
        try:
            client = db.Historical()
            print("✅ Connected to Databento")
        except Exception as e:
            print(f"❌ Failed to connect to Databento: {e}")
            return

    # Get subscription entitlements
    entitlements = get_subscription_entitlements()
    print("📊 Databento Subscription Entitlements:")
    print(
        f"  Core (OHLCV): {entitlements['core'][0].date()} to {entitlements['core'][1].date()} (7 years)",
    )
    print(
        f"  L1 (BBO/Trades): {entitlements['l1'][0].date()} to {entitlements['l1'][1].date()} (1 year)",
    )
    print(f"  L2 (MBP): {entitlements['l2'][0].date()} to {entitlements['l2'][1].date()} (30 days)")

    # Find symbols to process
    if args.symbols:
        symbol_dirs = [
            args.data_dir / symbol for symbol in args.symbols if (args.data_dir / symbol).is_dir()
        ]
    else:
        symbol_dirs = [d for d in args.data_dir.iterdir() if d.is_dir()]

    if not symbol_dirs:
        print("❌ No symbol directories found")
        return

    symbol_dirs.sort()
    if args.max_symbols:
        symbol_dirs = symbol_dirs[: args.max_symbols]

    print(f"🔍 Analyzing {len(symbol_dirs)} symbols for gaps...")
    print()

    # Analyze each symbol
    total_downloads = 0
    total_gaps = 0
    symbols_with_gaps = 0

    all_downloads = []

    for i, symbol_dir in enumerate(symbol_dirs, 1):
        analysis = analyze_symbol_gaps(symbol_dir, entitlements)
        symbol = analysis["symbol"]

        print(f"[{i:3d}/{len(symbol_dirs)}] {symbol}")

        # Show existing coverage
        if analysis["existing"]:
            print("  ✅ Current data:")
            for schema, info in analysis["existing"].items():
                days = info["days"]
                start = info["range"][0].strftime("%Y-%m-%d")
                end = info["range"][1].strftime("%Y-%m-%d")
                print(f"    {schema}: {days} days ({start} to {end})")

        # Show gaps
        if analysis["downloads_needed"]:
            print("  📥 Missing data (will download):")
            symbols_with_gaps += 1
            for download in analysis["downloads_needed"]:
                days = download["days"]
                start = download["start"].strftime("%Y-%m-%d")
                end = download["end"].strftime("%Y-%m-%d")
                print(f"    {download['schema']}: {days} days ({start} to {end})")
                total_downloads += 1
                total_gaps += days
                all_downloads.append(download)
        else:
            print("  🎉 Complete coverage!")

        print()

    # Summary
    print("=" * 80)
    print("📈 GAP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Symbols analyzed: {len(symbol_dirs)}")
    print(f"Symbols with gaps: {symbols_with_gaps}")
    print(f"Total downloads needed: {total_downloads}")
    print(f"Total days to download: {total_gaps:,}")
    print()

    if not all_downloads:
        print("🎉 No gaps found! You have complete data coverage.")
        return

    if args.dry_run:
        print("🔍 DRY RUN MODE - No data was downloaded")
        print("Remove --dry-run flag to execute the downloads")
        return

    # Execute downloads
    print("🚀 Starting gap-filling downloads...")
    print("=" * 80)

    successful_downloads = 0
    failed_downloads = 0

    for i, download in enumerate(all_downloads, 1):
        print(f"[{i:3d}/{len(all_downloads)}] ", end="")
        if download_gap(client, download):
            successful_downloads += 1
        else:
            failed_downloads += 1

    # Final summary
    print()
    print("=" * 80)
    print("🏁 DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"✅ Successful: {successful_downloads}")
    print(f"❌ Failed: {failed_downloads}")
    print(f"📊 Success rate: {successful_downloads/len(all_downloads)*100:.1f}%")
    print()
    print("🎯 Next steps:")
    print("1. Verify data completeness by running this script again")
    print("2. Set up daily incremental updates")
    print("3. Use your complete dataset for ML training!")


if __name__ == "__main__":
    main()
