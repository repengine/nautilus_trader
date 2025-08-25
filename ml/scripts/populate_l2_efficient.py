#!/usr/bin/env python3
"""
Efficient L2 data population script.

Handles large L2 data volumes by:
- Downloading day by day
- Processing one symbol at a time
- Showing progress and data sizes

Usage:
    python ml/scripts/populate_l2_efficient.py --symbols SPY AAPL --days 7
    python ml/scripts/populate_l2_efficient.py --tier 1 --days 7
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import databento as db
import pandas as pd
import psutil
import pyarrow.parquet as pq


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_tier1_symbols() -> list[str]:
    """Get list of Tier 1 symbols."""
    # Try to get from completed L1 data
    progress_file = Path("tier1_l1_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            symbols = sorted(set(data.get("completed_bbo", [])))
            if symbols:
                return symbols

    # Fallback to default list (minus VIX)
    return [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "XLF", "XLK", "XLE", "XLV", "XLI",
        "TLT", "GLD", "SLV", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "TSLA", "BRK.B", "AMD", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA",
        "DIS", "BAC", "ADBE", "CRM", "NFLX", "KO", "PEP", "TMO", "ABBV", "CVX",
        "WMT", "MRK", "LLY", "AVGO", "NKE", "ORCL", "ACN", "COST", "MCD", "ABT",
        "TXN", "GS", "MS", "WFC", "C", "XOM", "COP", "CAT", "BA", "GE",
        "MMM", "VZ", "T", "EFA", "EEM", "VEA", "VWO", "UUP", "FXE", "USO",
        "UNG", "PLTR", "SOFI", "RIVN", "LCID", "COIN", "MSTR", "VNQ"
    ]


def download_l2_daily(
    client: db.Historical,
    symbol: str,
    date: datetime,
    output_dir: Path
) -> int:
    """Download L2 data for a single day."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    try:
        # Download the data
        df = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=[symbol],
            schema="mbp-10",
            start=start,
            end=end
        ).to_df()

        if df.empty:
            return 0

        # Save to dated file
        date_str = date.strftime("%Y%m%d")
        output_file = output_dir / f"{symbol}_mbp10_{date_str}.parquet"
        df.to_parquet(output_file)

        # Get file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {date.date()}: {len(df):,} records, {size_mb:.1f} MB")

        return len(df)

    except Exception as e:
        if "403" in str(e) or "license" in str(e):
            logger.warning(f"  {date.date()}: Skipped (too recent for EQUS.MINI)")
        else:
            logger.error(f"  {date.date()}: Error - {e}")
        return 0


def combine_daily_files(symbol: str, output_dir: Path) -> None:
    """Combine daily L2 files into a single file using memory-efficient chunking."""
    daily_files = sorted(output_dir.glob(f"{symbol}_mbp10_*.parquet"))

    if not daily_files:
        return

    logger.info(f"Combining {len(daily_files)} daily files for {symbol}...")

    # Calculate total size to determine optimal chunk size
    total_size_mb = sum(f.stat().st_size for f in daily_files) / (1024 * 1024)

    # Adaptive chunking based on total size
    # Aim for ~300-400MB per chunk to stay well within memory limits
    if total_size_mb > 2500:  # > 2.5GB total (like NVDA)
        chunk_size = 2  # Very conservative for huge datasets
    elif total_size_mb > 2000:  # > 2GB total
        chunk_size = 3
    elif total_size_mb > 1000:  # > 1GB total
        chunk_size = 5
    elif total_size_mb > 500:  # > 500MB total
        chunk_size = 7
    else:
        chunk_size = 10  # Small files, can process more at once

    logger.info(f"  Total size: {total_size_mb:.1f} MB, using chunk size: {chunk_size}")

    temp_files = []

    try:
        # Process files in chunks
        for i in range(0, len(daily_files), chunk_size):
            chunk_files = daily_files[i:i + chunk_size]

            # Read and combine this chunk
            dfs = []
            for file in chunk_files:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"    Skipping corrupted file {file.name}: {e}")
                    continue

            if dfs:
                chunk_combined = pd.concat(dfs, ignore_index=False)
                chunk_combined = chunk_combined.sort_index()

                # Save chunk to temp file
                temp_file = output_dir / f"{symbol}_temp_chunk_{i//chunk_size}.parquet"
                chunk_combined.to_parquet(temp_file)
                temp_files.append(temp_file)

                # Clear memory aggressively
                del dfs, chunk_combined
                gc.collect()

                # Log memory usage
                mem_percent = psutil.virtual_memory().percent
                logger.info(f"  Processed chunk {i//chunk_size + 1}/{(len(daily_files) + chunk_size - 1)//chunk_size} (Memory: {mem_percent:.1f}%)")

        # Now combine temp chunks
        output_file = output_dir / f"{symbol}_mbp-10.parquet"

        if len(temp_files) == 1:
            # Only one chunk, just rename it
            temp_files[0].rename(output_file)
        else:
            # Multiple chunks - combine them more efficiently
            logger.info(f"  Merging {len(temp_files)} temp chunks...")

            # For very large datasets, use incremental merging
            if total_size_mb > 1500:  # > 1.5GB
                # Use ultra-conservative approach for huge datasets
                if total_size_mb > 2500:  # NVDA-sized datasets
                    logger.info("  Using ultra-conservative merge strategy for very large dataset")

                    # Don't merge all at once - use append strategy
                    # Start with first temp file as base
                    base_file = temp_files[0]

                    for idx, temp_file in enumerate(temp_files[1:], 1):
                        logger.info(f"    Appending file {idx}/{len(temp_files)-1}")

                        # Read both files
                        base_df = pd.read_parquet(base_file)
                        append_df = pd.read_parquet(temp_file)

                        # Merge and sort
                        merged = pd.concat([base_df, append_df], ignore_index=False)
                        merged = merged.sort_index()

                        # Write to new temp file
                        new_base = output_dir / f"{symbol}_temp_incremental_{idx}.parquet"
                        merged.to_parquet(new_base)

                        # Clean up
                        del base_df, append_df, merged
                        gc.collect()

                        # Remove old files
                        if base_file != temp_files[0]:  # Don't delete original temps yet
                            base_file.unlink()
                        temp_file.unlink()

                        # Update base for next iteration
                        base_file = new_base

                        # Check memory
                        mem_percent = psutil.virtual_memory().percent
                        if mem_percent > 80:
                            logger.warning(f"    High memory usage: {mem_percent:.1f}%")
                            gc.collect()
                            import time
                            time.sleep(2)  # Give system time to recover

                    # Final file is our output
                    base_file.rename(output_file)

                    # Clean up first temp file
                    if temp_files[0].exists():
                        temp_files[0].unlink()
                else:
                    # Standard incremental merge for 1.5-2.5GB
                    merge_round = 0
                    while len(temp_files) > 1:
                        new_temp_files = []

                        # Process pairs
                        for i in range(0, len(temp_files), 2):
                            if i + 1 < len(temp_files):
                                # Merge pair
                                df1 = pd.read_parquet(temp_files[i])
                                df2 = pd.read_parquet(temp_files[i + 1])
                                merged = pd.concat([df1, df2], ignore_index=False).sort_index()

                                # Save merged with unique name using round counter
                                merged_file = output_dir / f"{symbol}_temp_merged_r{merge_round}_p{i//2}.parquet"
                                merged.to_parquet(merged_file)
                                new_temp_files.append(merged_file)

                                # Clean up
                                del df1, df2, merged
                                gc.collect()

                                # Remove original temp files
                                temp_files[i].unlink()
                                temp_files[i + 1].unlink()
                            else:
                                # Odd file, keep for next round
                                new_temp_files.append(temp_files[i])

                        temp_files = new_temp_files
                        merge_round += 1
                        logger.info(f"    Reduced to {len(temp_files)} temp files")

                    # Final file should be the output
                    if temp_files:
                        temp_files[0].rename(output_file)
            else:
                # Standard approach for smaller datasets
                # Use first temp file as base
                combined = pd.read_parquet(temp_files[0])

                # Append others one by one
                for i, temp_file in enumerate(temp_files[1:], 1):
                    df = pd.read_parquet(temp_file)
                    combined = pd.concat([combined, df], ignore_index=False)
                    del df
                    gc.collect()

                    if i % 3 == 0:
                        logger.info(f"    Merged {i}/{len(temp_files)-1} chunks")

                combined = combined.sort_index()
                combined.to_parquet(output_file)
                del combined
                gc.collect()

                # Clean up temp files
                for temp_file in temp_files:
                    if temp_file.exists():
                        temp_file.unlink()

        # Remove daily files
        for file in daily_files:
            file.unlink()

        # Get final stats
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            # Count records without loading full file
            pf = pq.ParquetFile(output_file)
            total_records = pf.metadata.num_rows
            logger.info(f"Created {output_file.name}: {total_records:,} records, {size_mb:.1f} MB")

    except Exception as e:
        # Clean up all temp files on error
        for pattern in [f"{symbol}_temp_chunk_*.parquet",
                       f"{symbol}_temp_merged_*.parquet"]:
            for temp_file in output_dir.glob(pattern):
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file.name}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Efficient L2 data downloader")

    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--tier", type=int, choices=[1], help="Use Tier 1 symbols")
    parser.add_argument("--days", type=int, default=7, help="Number of days to download")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tier1"),
                       help="Data directory")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from existing data")

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    elif args.tier == 1:
        symbols = get_tier1_symbols()
    else:
        logger.error("Specify either --symbols or --tier")
        return 1

    # Initialize client
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set")
        return 1

    client = db.Historical(api_key)

    # Calculate date range (account for EQUS.MINI delay)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days - 1)

    logger.info(f"Downloading L2 data for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 50)

    total_records = 0
    total_size_mb = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        # Create output directory
        output_dir = args.data_dir / symbol / "l2"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists
        final_file = output_dir / f"{symbol}_mbp-10.parquet"
        if final_file.exists() and args.resume:
            size_mb = final_file.stat().st_size / (1024 * 1024)
            logger.info(f"  Already exists: {size_mb:.1f} MB - skipping")
            total_size_mb += size_mb
            continue

        # Download day by day
        symbol_records = 0
        current_date = start_date

        while current_date <= end_date:
            records = download_l2_daily(client, symbol, current_date, output_dir)
            symbol_records += records
            current_date += timedelta(days=1)

        # Combine daily files
        if symbol_records > 0:
            combine_daily_files(symbol, output_dir)
            if final_file.exists():
                size_mb = final_file.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb

        total_records += symbol_records
        logger.info(f"  Total: {symbol_records:,} records")

    logger.info("\n" + "=" * 50)
    logger.info(f"COMPLETE: {total_records:,} total records, {total_size_mb:.1f} MB")


if __name__ == "__main__":
    sys.exit(main() or 0)
