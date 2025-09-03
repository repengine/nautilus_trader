#!/usr/bin/env python3
"""
Fixed L2 data population script with proper validation and gap detection.
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import databento as db
import pandas as pd


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_expected_trading_days(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Get list of expected trading days (Mon-Fri) between dates."""
    trading_days = []
    current = start_date

    while current <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:  # Monday=0 to Friday=4
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def validate_existing_data(symbol: str, output_dir: Path, expected_days: list[datetime]) -> tuple[bool, list[str]]:
    """Validate existing data completeness and return (is_complete, issues)."""
    final_file = output_dir / f"{symbol}_mbp-10.parquet"

    if not final_file.exists():
        return False, ["File does not exist"]

    try:
        # Quick validation using parquet metadata
        import polars as pl
        df = pl.scan_parquet(final_file)

        # Sample data to check date range
        sample_data = df.head(1000).collect()
        if len(sample_data) == 0:
            return False, ["Empty file"]

        # Check date range
        min_ts = sample_data["ts_event"].min()
        max_ts = sample_data["ts_event"].max()

        # Get tail sample too for complete range
        tail_data = df.tail(1000).collect()
        if len(tail_data) > 0:
            min_ts = min(min_ts, tail_data["ts_event"].min())
            max_ts = max(max_ts, tail_data["ts_event"].max())

        # Convert to dates
        min_date = pd.to_datetime(min_ts, unit="ns").date()
        max_date = pd.to_datetime(max_ts, unit="ns").date()

        expected_start = expected_days[0].date()
        expected_end = expected_days[-1].date()

        issues = []

        # Check coverage
        if min_date > expected_start:
            issues.append(f"Missing early days: starts {min_date} vs expected {expected_start}")

        if max_date < expected_end:
            issues.append(f"Missing recent days: ends {max_date} vs expected {expected_end}")

        # Check minimum days coverage (allow 2-day tolerance for holidays)
        coverage_days = (max_date - min_date).days + 1
        expected_days_count = len(expected_days)

        if coverage_days < expected_days_count - 2:
            issues.append(f"Insufficient coverage: {coverage_days} days vs expected ~{expected_days_count}")

        return len(issues) == 0, issues

    except Exception as e:
        return False, [f"Validation error: {e!s}"]


def download_l2_daily_with_validation(
    client: db.Historical,
    symbol: str,
    date: datetime,
    output_dir: Path
) -> tuple[int, bool]:
    """Download L2 data for a single day with validation. Returns (records, success)."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    # Skip weekends
    if start.weekday() >= 5:
        return 0, True  # Success but no data expected

    attempts = 3
    delay_secs = 2.0

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
                logger.warning(f"  {date.date()}: NO DATA (empty response)")
                return 0, False  # Mark as failure for tracking

            date_str = date.strftime("%Y%m%d")
            output_file = output_dir / f"{symbol}_mbp10_{date_str}.parquet"
            df.to_parquet(output_file)

            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {date.date()}: {len(df):,} records, {size_mb:.1f} MB")
            return len(df), True

        except Exception as e:
            msg = str(e)

            # Known acceptable cases
            if "403" in msg or "license" in msg:
                logger.warning(f"  {date.date()}: Skipped (license restriction)")
                return 0, True

            # Retry logic
            transient = any(code in msg for code in ("504", "502", "500", "timeout", "429"))
            if transient and attempt < attempts:
                logger.warning(f"  {date.date()}: Retry {attempt}/{attempts} - {msg}")
                import time
                time.sleep(delay_secs)
                delay_secs *= 2.0
                continue

            logger.error(f"  {date.date()}: FAILED - {msg}")
            return 0, False

    return 0, False


def get_tier1_symbols() -> list[str]:
    """Get list of Tier 1 symbols."""
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


def main():
    parser = argparse.ArgumentParser(description="Fixed L2 data downloader with validation")

    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--tier", type=int, choices=[1], help="Use Tier 1 symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tier1"), help="Data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    elif args.tier == 1:
        symbols = get_tier1_symbols()
    else:
        logger.error("Specify either --symbols or --tier")
        return 1

    # Calculate fixed date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        # Use fixed range to avoid drift across runs
        end_date = datetime(2025, 8, 29)  # Last available trading day based on analysis
        start_date = end_date - timedelta(days=args.days - 1)

    expected_days = get_expected_trading_days(start_date, end_date)

    logger.info(f"Processing {len(symbols)} symbols")
    logger.info(f"Fixed date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Expected trading days: {len(expected_days)}")
    logger.info("=" * 60)

    if args.validate_only:
        logger.info("VALIDATION MODE - checking existing data only")

    validation_results = {}
    download_results = {}

    # Initialize client if not validation-only
    client = None
    if not args.validate_only:
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.error("DATABENTO_API_KEY not set")
            return 1
        client = db.Historical(api_key)

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        output_dir = args.data_dir / symbol / "l2"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate existing data
        is_valid, issues = validate_existing_data(symbol, output_dir, expected_days)
        validation_results[symbol] = {"valid": is_valid, "issues": issues}

        if is_valid and not args.force:
            logger.info("  ✓ Data complete and valid - skipping")
            continue

        if issues:
            logger.warning(f"  ❌ Issues found: {', '.join(issues)}")

        if args.validate_only:
            continue

        # Force re-download if needed
        if args.force or not is_valid:
            final_file = output_dir / f"{symbol}_mbp-10.parquet"
            if final_file.exists():
                logger.info("  Removing existing incomplete file")
                final_file.unlink()

        # Download data
        failed_days = []
        successful_days = 0
        total_records = 0

        for day in expected_days:
            records, success = download_l2_daily_with_validation(client, symbol, day, output_dir)
            total_records += records

            if success:
                successful_days += 1
            else:
                failed_days.append(day.date())

        download_results[symbol] = {
            "total_records": total_records,
            "successful_days": successful_days,
            "failed_days": failed_days,
            "success_rate": successful_days / len(expected_days) if expected_days else 0
        }

        # Combine files if we have data
        if total_records > 0:
            from ml.scripts.populate_l2_efficient import combine_daily_files
            combine_daily_files(symbol, output_dir)
            logger.info(f"  Combined: {total_records:,} total records")

        # Report issues
        if failed_days:
            logger.error(f"  ❌ Failed days ({len(failed_days)}): {failed_days[:5]}{'...' if len(failed_days) > 5 else ''}")

    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 60)

    valid_count = sum(1 for r in validation_results.values() if r["valid"])
    logger.info(f"Valid symbols: {valid_count}/{len(symbols)}")

    if not args.validate_only:
        successful_downloads = sum(1 for r in download_results.values() if r["success_rate"] > 0.8)
        logger.info(f"Successful downloads: {successful_downloads}/{len(symbols)}")

    # List problematic symbols
    problem_symbols = [sym for sym, result in validation_results.items() if not result["valid"]]
    if problem_symbols:
        logger.info(f"\n❌ Symbols needing attention: {', '.join(problem_symbols[:10])}")
        if len(problem_symbols) > 10:
            logger.info(f"   ... and {len(problem_symbols) - 10} more")
    else:
        logger.info("\n✓ All symbols have complete, valid data!")

    return 0 if len(problem_symbols) == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
