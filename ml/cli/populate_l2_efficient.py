#!/usr/bin/env python3
"""
Efficient L2 data population script with gap detection and filling.

Handles large L2 data volumes by:
- Downloading day by day
- Processing one symbol at a time
- Checking for missing data gaps
- Merging new data with existing files
- Showing progress and data sizes

Usage:
    # Download last 30 days for specific symbols (default)
    python ml/cli/populate_l2_efficient.py --symbols SPY AAPL

    # Download tier 1 symbols for date range, fill gaps automatically
    python ml/cli/populate_l2_efficient.py --tier 1 --start-date 2025-07-26 --end-date 2025-08-29

    # Download last 7 days with custom period
    python ml/cli/populate_l2_efficient.py --tier 1 --days 7

    # Force re-download all data (ignoring existing)
    python ml/cli/populate_l2_efficient.py --tier 1 --days 30 --force

    # Check and fill gaps in existing data
    python ml/cli/populate_l2_efficient.py --tier 1 --start-date 2025-07-26 --end-date 2025-08-29 --check-gaps

"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid as _uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from random import shuffle as _shuffle
from typing import Any, cast

import databento as db
import pandas as pd
import polars as pl
import psutil
import pyarrow.parquet as pq

from ml.common.logging_config import bind_log_context

# Setup logging
from ml.common.logging_config import configure_logging
from ml.config.universes import TIER1_CORE
from ml.data.ingest.common import load_progress_json
from ml.data.ingest.common import save_progress_json


configure_logging()
_run_id: str = f"cli_populate_l2_efficient_{_uuid.uuid4().hex[:8]}"
bind_log_context(run_id=_run_id, component="ml.cli.populate_l2_efficient")
logger = logging.getLogger(__name__)


def _load_progress(path: Path) -> dict[str, list[str]]:
    data_any = load_progress_json(path)
    if isinstance(data_any, dict):
        out: dict[str, list[str]] = {}
        for k, v in data_any.items():
            try:
                out[str(k)] = [str(x) for x in (v if isinstance(v, list) else [])]
            except Exception:
                out[str(k)] = []
        return out
    return {}


def _save_progress(path: Path, prog: dict[str, list[str]]) -> None:
    save_progress_json(path, prog)


def validate_data_integrity(file_path: Path, symbol: str, expected_date: datetime) -> bool:
    """
    Validate that a data file contains reasonable data for the given date.
    """
    if not file_path.exists():
        return False

    try:
        df = pl.read_parquet(file_path)

        if df.is_empty():
            logger.warning(f"  {expected_date.date()}: File exists but is empty")
            return False

        # Check if data is from the expected date
        min_ts = df["ts_event"].min()
        max_ts = df["ts_event"].max()
        if min_ts is None or max_ts is None:
            return False
        min_ts_i: int = cast(int, min_ts)
        max_ts_i: int = cast(int, max_ts)
        min_date = pd.to_datetime(min_ts_i, unit="ns").date()
        max_date = pd.to_datetime(max_ts_i, unit="ns").date()

        if min_date != expected_date.date() or max_date != expected_date.date():
            logger.warning(
                f"  {expected_date.date()}: Data spans {min_date} to {max_date} (date mismatch)",
            )
            return False

        # Check for reasonable amount of L2 data
        record_count = len(df)
        if record_count < 1000:  # Very low for a full trading day
            logger.warning(
                f"  {expected_date.date()}: Only {record_count:,} records (likely incomplete)",
            )
            return False

        # Check for data during market hours (rough validation)
        market_hours_data = df.filter(
            (pl.from_epoch("ts_event", time_unit="ns").dt.hour() >= 9)
            & (pl.from_epoch("ts_event", time_unit="ns").dt.hour() <= 16),
        )

        if len(market_hours_data) < record_count * 0.8:  # Most data should be during market hours
            logger.warning(
                f"  {expected_date.date()}: Low market hours coverage ({len(market_hours_data)}/{record_count} records)",
            )

        return True

    except Exception as e:
        logger.warning("  %s: Error validating file: %s", expected_date.date(), e, exc_info=True)
        return False


def get_business_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    Get list of business dates (Monday-Friday) in the range.
    """
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current)
        current += timedelta(days=1)
    return dates


def detect_data_gaps(
    symbol: str,
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> list[datetime]:
    """
    Detect missing dates in existing L2 data for a symbol, including integrity
    validation.
    """
    final_file = output_dir / f"{symbol}_mbp-10.parquet"

    if not final_file.exists():
        # No existing data - need all dates
        return get_business_dates(start_date, end_date)

    # Validate the existing file integrity first
    if not validate_data_integrity(final_file, symbol, start_date):
        logger.warning(
            "Existing data file failed integrity check - will re-download affected dates",
        )

    try:
        # Read existing data to find covered dates
        df = pl.read_parquet(final_file)
        if df.is_empty():
            return get_business_dates(start_date, end_date)

        # Get unique dates from existing data
        existing_dates = (
            df.select(
                pl.from_epoch("ts_event", time_unit="ns").dt.date().alias("date"),
            )
            .unique()
            .sort("date")
        )

        dates_in_data = set(existing_dates.to_series().to_list())

        # Find missing business dates in the requested range
        expected_dates = get_business_dates(start_date, end_date)
        missing_dates = []

        for date in expected_dates:
            if date.date() not in dates_in_data:
                missing_dates.append(date)
            else:
                # Date exists in data, but validate daily integrity if we have daily files
                daily_file = output_dir / f"{symbol}_mbp10_{date.strftime('%Y%m%d')}.parquet"
                if daily_file.exists() and not validate_data_integrity(daily_file, symbol, date):
                    logger.info(f"Re-downloading {date.date()} due to integrity issues")
                    missing_dates.append(date)

        return missing_dates

    except Exception as e:
        logger.warning("Error reading existing data for %s: %s", symbol, e, exc_info=True)
        # If we can't read existing data, assume we need all dates
        return get_business_dates(start_date, end_date)


def merge_new_with_existing(symbol: str, output_dir: Path) -> None:
    """
    Merge new daily files with existing L2 data using streaming writes.

    Avoids loading the entire final file into memory.

    """
    daily_files = sorted(output_dir.glob(f"{symbol}_mbp10_*.parquet"))
    if not daily_files:
        return

    final_file = output_dir / f"{symbol}_mbp-10.parquet"
    tmp_file = output_dir / f"{symbol}_mbp-10.tmp.parquet"

    def _append_files_to_writer(
        writer: pq.ParquetWriter | None,
        files: list[Path],
    ) -> pq.ParquetWriter | None:
        for file in files:
            try:
                pf = pq.ParquetFile(file)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_file, pf.schema_arrow)
                for rg in range(pf.num_row_groups or 1):
                    table = pf.read_row_group(rg) if pf.num_row_groups else pf.read()
                    # Ensure column order matches writer schema
                    if writer.schema and table.schema != writer.schema:
                        table = table.select(writer.schema.names)
                    writer.write_table(table)
            except Exception as e:
                logger.error("  Error appending %s: %s", file.name, e, exc_info=True)
                continue
        return writer

    writer: pq.ParquetWriter | None = None
    try:
        # Append existing final first (if present), then new daily files
        if final_file.exists():
            writer = _append_files_to_writer(writer, [final_file])
        writer = _append_files_to_writer(writer, daily_files)
        if writer is not None:
            writer.close()
        # Replace final atomically
        tmp_file.replace(final_file)
        # Cleanup daily files
        for file in daily_files:
            try:
                file.unlink()
            except OSError as exc:
                logger.debug("Failed to unlink daily file %s: %s", file, exc, exc_info=True)
        size_mb = final_file.stat().st_size / (1024 * 1024)
        logger.info(f"Merged data (streaming): {size_mb:.1f} MB -> {final_file.name}")
    except Exception as e:
        try:
            if writer is not None:
                writer.close()
        except Exception as exc:
            logger.debug("Closing Parquet writer failed: %s", exc, exc_info=True)
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError as exc:
                logger.debug("Failed to remove tmp file %s: %s", tmp_file, exc, exc_info=True)
        raise e


def get_tier1_symbols() -> list[str]:
    """
    Get list of Tier 1 symbols.
    """
    # Try to get from completed L1 data
    progress_file = Path("tier1_l1_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            symbols = sorted(set(data.get("completed_bbo", [])))
            if symbols:
                return symbols

    # Fallback to centralized Tier1 core universe
    return list(TIER1_CORE)


def download_l2_daily(
    client: db.Historical,
    symbol: str,
    date: datetime,
    output_dir: Path,
) -> int:
    """
    Download L2 data for a single day with basic retries.
    """
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    # Skip weekends to avoid futile requests and gateway warnings
    if start.weekday() >= 5:
        return 0

    # Typed retry with backoff for transient gateway or rate limit errors
    class _TransientError(Exception):
        pass

    class _NonTransientSkip(Exception):
        pass

    def _attempt() -> int:
        # Get the data stream first
        data_stream = client.timeseries.get_range(
            dataset="EQUS.MINI",
            symbols=[symbol],
            schema="mbp-10",
            start=start,
            end=end,
        )

        # Handle empty data cases before conversion
        try:
            df = data_stream.to_df()
        except Exception as pandas_err:
            # Handle pandas processing errors on empty/malformed data
            if "No data found" in str(pandas_err):
                logger.info(f"  {date.date()}: No data available")
                return 0
            # Re-raise other pandas errors
            raise pandas_err

        if df is None or df.empty:
            logger.info(f"  {date.date()}: No data available")
            return 0

        # Validate data integrity
        if len(df) < 100:  # Suspiciously small for L2 data
            logger.warning(f"  {date.date()}: Only {len(df)} records (may be incomplete)")

        date_str = date.strftime("%Y%m%d")
        output_file = output_dir / f"{symbol}_mbp10_{date_str}.parquet"
        df.to_parquet(output_file)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {date.date()}: {len(df):,} records, {size_mb:.1f} MB")
        return len(df)

    def _on_exc(attempt: int, exc: BaseException) -> None:
        msg = str(exc)
        # Known benign cases (fail fast without retries)
        if "403" in msg or "license" in msg:
            raise _NonTransientSkip(msg)
        # Retry on 5xx, timeouts, and 429
        transient = any(code in msg for code in ("504", "502", "500", "timeout", "429"))
        if transient:
            logger.warning(
                f"  {date.date()}: Transient error (attempt {attempt + 1}) - {msg}",
            )
        else:
            # Promote to non-retriable by raising different exception
            raise _NonTransientSkip(msg)

    try:
        from ml.common.retry_utils import retry_with_backoff as _retry

        res = _retry(
            _attempt,
            max_attempts=3,
            initial_delay=2.0,
            multiplier=2.0,
            max_delay=30.0,
            jitter=0.0,
            on_exception=_on_exc,
            retry_on=(_TransientError, Exception),
        )
        return int(res)
    except _NonTransientSkip as e:
        logger.error("  %s: Error - %s", date.date(), e, exc_info=True)
        return 0


def _clean_stale_temp_files(symbol: str, output_dir: Path) -> None:
    """
    Remove any leftover temp parquet chunks from previous runs.
    """
    for pattern in (f"{symbol}_temp_chunk_*.parquet", f"{symbol}_temp_merged_*.parquet"):
        for f in output_dir.glob(pattern):
            try:
                f.unlink()
            except OSError as exc:
                logger.debug("Failed to unlink stale temp file %s: %s", f, exc, exc_info=True)


def _validate_daily_file(file_path: Path) -> bool:
    """
    Validate that a daily parquet file is readable and not corrupted.
    """
    try:
        # Quick validation - try to read parquet metadata
        pf = pq.ParquetFile(file_path)
        if pf.metadata is None or pf.metadata.num_rows == 0:
            return False
        return True
    except Exception as e:
        logger.warning("  Corrupted file detected: %s - %s", file_path.name, e, exc_info=True)
        return False


def _stream_merge_daily_files(daily_files: list[Path], tmp_output: Path) -> None:
    """
    Stream row groups from daily files into a single Parquet file.
    """
    writer: pq.ParquetWriter | None = None
    valid_files = []

    # Pre-validate all files
    for file in daily_files:
        if _validate_daily_file(file):
            valid_files.append(file)
        else:
            logger.warning(
                f"  Skipping corrupted file: {file.name} (will be re-downloaded on next run)",
            )
            # Don't auto-delete here - let the main script handle re-downloading

    if not valid_files:
        raise ValueError("No valid daily files found to combine")

    try:
        for idx, file in enumerate(valid_files, 1):
            try:
                pf = pq.ParquetFile(file)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_output, pf.schema_arrow)
                for rg in range(pf.num_row_groups or 1):
                    table = pf.read_row_group(rg) if pf.num_row_groups else pf.read()
                    if writer.schema and table.schema != writer.schema:
                        table = table.select(writer.schema.names)
                    writer.write_table(table)
                if idx % 3 == 0 or idx == len(valid_files):
                    mem_percent = psutil.virtual_memory().percent
                    logger.info(
                        f"  Appended {idx}/{len(valid_files)} daily files (Memory: {mem_percent:.1f}%)",
                    )
            except Exception as e:
                logger.error("  Error processing %s: %s", file.name, e, exc_info=True)
                # Don't fail entire operation for one bad file
                continue
    finally:
        if writer is not None:
            writer.close()


def combine_daily_files(symbol: str, output_dir: Path) -> None:
    """
    Combine daily L2 files into a single Parquet via streaming writes.
    """
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
            except OSError as exc:
                logger.debug("Failed to unlink daily file %s: %s", file, exc, exc_info=True)

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            pf_final = pq.ParquetFile(output_file)
            total_records = pf_final.metadata.num_rows
            logger.info(f"Created {output_file.name}: {total_records:,} records, {size_mb:.1f} MB")
    except Exception as e:
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except OSError as exc:
                logger.debug("Failed to unlink tmp output %s: %s", tmp_output, exc, exc_info=True)
        raise e


def main() -> int:
    parser = argparse.ArgumentParser(description="Efficient L2 data downloader")

    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--tier", type=int, choices=[1], help="Use Tier 1 symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tier1"),
        help="Data directory",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("data/tier1/.l2_progress.json"),
        help="Path to JSON progress file for resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing data",
    )
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--check-gaps",
        action="store_true",
        default=True,
        help="Check for and fill data gaps",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all data, ignoring existing files",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols processed in this run",
    )
    parser.add_argument(
        "--symbol-offset",
        type=int,
        default=0,
        help="Start offset into the symbol list (for sharding)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle symbol order to spread progress",
    )
    parser.add_argument("--rate-limit", type=int, default=60, help="API calls per minute throttle")
    parser.add_argument(
        "--sleep-between-symbols",
        type=float,
        default=0.0,
        help="Seconds to sleep between symbols (reduce burstiness)",
    )

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

    # Calculate date range
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error("Invalid date format: %s", e, exc_info=True)
        return 1
    else:
        # Default: account for EQUS.MINI delay
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days - 1)

    # Optional sharding
    symbols_work = list(symbols)
    if args.shuffle:
        _shuffle(symbols_work)
    if args.symbol_offset:
        symbols_work = symbols_work[args.symbol_offset :]
    if args.max_symbols is not None:
        symbols_work = symbols_work[: max(0, int(args.max_symbols))]

    logger.info(f"Downloading L2 data for {len(symbols_work)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 50)

    total_records = 0
    total_size_mb = 0

    # Progress and signal handling
    progress_path: Path = args.progress_file
    progress: dict[str, list[str]] = _load_progress(progress_path)
    _terminate: dict[str, bool] = {"stop": False}

    def _handle_sig(_signum: int, _frame: Any) -> None:
        _terminate["stop"] = True
        _save_progress(progress_path, progress)
        logger.info("Received termination signal; progress saved.")

    try:
        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)
    except Exception as exc:
        logger.debug("Signal handler setup failed: %s", exc, exc_info=True)

    min_interval = 60.0 / max(1, int(args.rate_limit))
    last_call_ts = 0.0

    for i, symbol in enumerate(symbols_work, 1):
        logger.info(f"\n[{i}/{len(symbols_work)}] Processing {symbol}...")

        # Create output directory
        output_dir = args.data_dir / symbol / "l2"
        output_dir.mkdir(parents=True, exist_ok=True)

        final_file = output_dir / f"{symbol}_mbp-10.parquet"

        # Handle force mode or gap detection
        dates_to_download: list[datetime] = []

        if args.force:
            # Force mode: download all dates
            dates_to_download = get_business_dates(start_date, end_date)
            if final_file.exists():
                logger.info("  Force mode: removing existing data")
                try:
                    final_file.unlink()
                except OSError as exc:
                    logger.debug(
                        "Failed to unlink existing final file %s: %s",
                        final_file,
                        exc,
                        exc_info=True,
                    )
        elif args.check_gaps:
            # Prefer progress-based gap detection to avoid loading large files
            done_dates = set(progress.get(symbol, []))
            if done_dates:
                expected = get_business_dates(start_date, end_date)
                dates_to_download = [
                    d for d in expected if d.strftime("%Y-%m-%d") not in done_dates
                ]
            else:
                # Fallback: inspect existing files
                dates_to_download = detect_data_gaps(symbol, output_dir, start_date, end_date)
            if not dates_to_download:
                if final_file.exists():
                    size_mb = final_file.stat().st_size / (1024 * 1024)
                    logger.info(f"  No gaps found: {size_mb:.1f} MB - complete")
                    total_size_mb += size_mb
                else:
                    logger.info("  No existing data and no gaps to fill")
                continue
            else:
                logger.info(f"  Found {len(dates_to_download)} missing dates to download")
        else:
            # Resume mode: skip if exists, otherwise download all
            if final_file.exists() and args.resume:
                size_mb = final_file.stat().st_size / (1024 * 1024)
                logger.info(f"  Already exists: {size_mb:.1f} MB - skipping")
                total_size_mb += size_mb
                continue
            dates_to_download = get_business_dates(start_date, end_date)

        # Download only the dates we need
        symbol_records = 0
        done_dates = set(progress.get(symbol, []))
        for date in dates_to_download:
            if _terminate["stop"]:
                break
            date_iso = date.strftime("%Y-%m-%d")
            if date_iso in done_dates:
                logger.info(f"  {date.date()}: already completed (progress file)")
                continue
            # Throttle
            now = time.time()
            elapsed = now - last_call_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            records = download_l2_daily(client, symbol, date, output_dir)
            last_call_ts = time.time()
            symbol_records += records
            if records > 0:
                progress.setdefault(symbol, []).append(date_iso)
                _save_progress(progress_path, progress)

        # Combine/merge files
        if symbol_records > 0:
            if args.check_gaps and final_file.exists():
                # Merge new data with existing
                merge_new_with_existing(symbol, output_dir)
            else:
                # Fresh download - combine daily files normally
                combine_daily_files(symbol, output_dir)

            if final_file.exists():
                size_mb = final_file.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb

        total_records += symbol_records
        logger.info(f"  Downloaded: {symbol_records:,} new records")
        if args.sleep_between_symbols > 0:
            time.sleep(float(args.sleep_between_symbols))

    logger.info("\n" + "=" * 50)
    logger.info(f"COMPLETE: {total_records:,} total records, {total_size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
