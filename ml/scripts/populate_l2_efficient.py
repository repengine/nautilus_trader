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
import json
import logging
import os
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import databento as db
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
    """Download L2 data for a single day with basic retries."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    # Skip weekends to avoid futile requests and gateway warnings
    if start.weekday() >= 5:
        return 0

    # Simple retry with backoff for transient gateway or rate limit errors
    attempts = 3
    delay_secs = 2.0
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            df = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[symbol],
                schema="mbp-10",
                start=start,
                end=end,
            ).to_df()

            if df.empty:
                return 0

            date_str = date.strftime("%Y%m%d")
            output_file = output_dir / f"{symbol}_mbp10_{date_str}.parquet"
            df.to_parquet(output_file)

            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {date.date()}: {len(df):,} records, {size_mb:.1f} MB")
            return len(df)

        except Exception as e:
            msg = str(e)
            last_err = e
            # Known benign cases
            if "403" in msg or "license" in msg:
                logger.warning(f"  {date.date()}: Skipped (too recent for EQUS.MINI)")
                return 0
            # Retry on 5xx, timeouts, and 429
            transient = any(code in msg for code in ("504", "502", "500", "timeout", "429"))
            if transient and attempt < attempts:
                logger.warning(f"  {date.date()}: Transient error (attempt {attempt}/{attempts}) - {msg}")
                import time
                time.sleep(delay_secs)
                delay_secs *= 2.0
                continue
            logger.error(f"  {date.date()}: Error - {msg}")
            return 0
    # Should not reach here, but in case loop exits without return
    if last_err is not None:
        logger.error(f"  {date.date()}: Error - {last_err}")
    return 0


def _clean_stale_temp_files(symbol: str, output_dir: Path) -> None:
    """Remove any leftover temp parquet chunks from previous runs."""
    for pattern in (f"{symbol}_temp_chunk_*.parquet", f"{symbol}_temp_merged_*.parquet"):
        for f in output_dir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def _stream_merge_daily_files(daily_files: list[Path], tmp_output: Path) -> None:
    """Stream row groups from daily files into a single Parquet file."""
    writer: pq.ParquetWriter | None = None
    try:
        for idx, file in enumerate(daily_files, 1):
            pf = pq.ParquetFile(file)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, pf.schema_arrow)
            for rg in range(pf.num_row_groups or 1):
                table = pf.read_row_group(rg) if pf.num_row_groups else pf.read()
                if writer.schema and table.schema != writer.schema:
                    table = table.select(writer.schema.names)
                writer.write_table(table)
            if idx % 3 == 0 or idx == len(daily_files):
                mem_percent = psutil.virtual_memory().percent
                logger.info(f"  Appended {idx}/{len(daily_files)} daily files (Memory: {mem_percent:.1f}%)")
    finally:
        if writer is not None:
            writer.close()


def combine_daily_files(symbol: str, output_dir: Path) -> None:
    """Combine daily L2 files into a single Parquet via streaming writes."""
    _clean_stale_temp_files(symbol, output_dir)

    daily_files = sorted(output_dir.glob(f"{symbol}_mbp10_*.parquet"))
    if not daily_files:
        return

    logger.info(f"Combining {len(daily_files)} daily files for {symbol}...")
    total_size_mb = sum(f.stat().st_size for f in daily_files) / (1024 * 1024)
    logger.info(f"  Total size: {total_size_mb:.1f} MB; streaming merge")

    output_file = output_dir / f"{symbol}_mbp-10.parquet"
    tmp_output = output_dir / f"{symbol}_mbp-10.tmp.parquet"

    try:
        _stream_merge_daily_files(daily_files, tmp_output)
        if tmp_output.exists():
            tmp_output.replace(output_file)

        for file in daily_files:
            try:
                file.unlink()
            except OSError:
                pass

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            pf_final = pq.ParquetFile(output_file)
            total_records = pf_final.metadata.num_rows
            logger.info(f"Created {output_file.name}: {total_records:,} records, {size_mb:.1f} MB")
    except Exception as e:
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except OSError:
                pass
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
