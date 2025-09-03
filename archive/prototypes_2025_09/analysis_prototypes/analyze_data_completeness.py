#!/usr/bin/env python3
"""
Data completeness analysis for L2 market data across all symbols.
Identifies missing dates, inconsistent coverage, and temporal alignment issues.
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl


def get_parquet_files():
    """Get all L2 parquet files."""
    data_dir = Path("data/tier1")
    return list(data_dir.glob("**/l2/*_mbp-10.parquet"))

def analyze_file_metadata(file_path):
    """Analyze basic metadata of a parquet file without loading full data."""
    try:
        # Read just schema and basic stats
        df = pl.scan_parquet(file_path)

        # Get basic info
        symbol = file_path.stem.replace("_mbp-10", "")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Sample first and last few rows to get date range
        first_rows = df.head(100).collect()
        last_rows = df.tail(100).collect()

        if len(first_rows) == 0:
            return {
                "symbol": symbol,
                "file_size_mb": file_size_mb,
                "record_count": 0,
                "date_range": None,
                "error": "Empty file"
            }

        # Extract date range from ts_event (assuming it's in nanoseconds)
        min_ts = min(first_rows["ts_event"].min(), last_rows["ts_event"].min())
        max_ts = max(first_rows["ts_event"].max(), last_rows["ts_event"].max())

        # Convert nanoseconds to datetime
        min_date = pd.to_datetime(min_ts, unit="ns").date()
        max_date = pd.to_datetime(max_ts, unit="ns").date()

        # Get approximate record count
        record_count = df.select(pl.count()).collect().item()

        return {
            "symbol": symbol,
            "file_size_mb": round(file_size_mb, 1),
            "record_count": record_count,
            "min_date": min_date,
            "max_date": max_date,
            "date_range_days": (max_date - min_date).days + 1,
            "error": None
        }

    except Exception as e:
        symbol = file_path.stem.replace("_mbp-10", "")
        return {
            "symbol": symbol,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "record_count": None,
            "date_range": None,
            "error": str(e)
        }

def analyze_trading_days():
    """Generate expected trading days for the period Aug 3 - Sep 1, 2025."""
    from datetime import date

    start_date = date(2025, 8, 3)  # Sunday
    end_date = date(2025, 9, 1)   # Sunday

    trading_days = []
    current = start_date

    while current <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:  # Monday=0 to Friday=4
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days

def main():
    print("🔍 Analyzing L2 Data Completeness and Alignment")
    print("=" * 60)

    # Get all parquet files
    parquet_files = get_parquet_files()
    print(f"Found {len(parquet_files)} parquet files")

    # Analyze each file
    results = []
    errors = []

    print("\nAnalyzing file metadata...")
    for i, file_path in enumerate(parquet_files, 1):
        print(f"  [{i:2d}/{len(parquet_files)}] {file_path.stem}...", end="")

        result = analyze_file_metadata(file_path)
        results.append(result)

        if result["error"]:
            errors.append(result)
            print(f" ERROR: {result['error']}")
        else:
            print(f" ✓ {result['record_count']:,} records ({result['date_range_days']} days)")

    # Expected trading days
    expected_days = analyze_trading_days()
    expected_start = expected_days[0]
    expected_end = expected_days[-1]
    expected_count = len(expected_days)

    print("\n📅 Expected Trading Period:")
    print(f"  Start: {expected_start}")
    print(f"  End: {expected_end}")
    print(f"  Trading Days: {expected_count}")

    # Analyze completeness
    print("\n📊 Data Completeness Analysis:")

    complete_files = [r for r in results if not r["error"] and r["record_count"] > 0]

    if not complete_files:
        print("❌ No valid data files found!")
        return

    # Date range analysis
    min_start = min(r["min_date"] for r in complete_files)
    max_end = max(r["max_date"] for r in complete_files)

    print(f"  Actual Data Range: {min_start} to {max_end}")
    print(f"  Expected Range: {expected_start} to {expected_end}")

    # Check alignment
    start_aligned = min_start <= expected_start
    end_aligned = max_end >= expected_end

    print(f"  Start Alignment: {'✓' if start_aligned else '❌'} ({min_start} vs {expected_start})")
    print(f"  End Alignment: {'✓' if end_aligned else '❌'} ({max_end} vs {expected_end})")

    # Analyze individual symbol coverage
    print("\n📋 Symbol Coverage Analysis:")

    coverage_issues = []
    perfect_coverage = []

    for result in complete_files:
        symbol = result["symbol"]
        days_coverage = result["date_range_days"]

        if days_coverage < expected_count - 2:  # Allow 2-day tolerance for weekends/holidays
            coverage_issues.append({
                "symbol": symbol,
                "days": days_coverage,
                "missing_days": expected_count - days_coverage,
                "start": result["min_date"],
                "end": result["max_date"]
            })
        else:
            perfect_coverage.append(symbol)

    print(f"  Perfect Coverage: {len(perfect_coverage)}/{len(complete_files)} symbols")
    print(f"  Coverage Issues: {len(coverage_issues)} symbols")

    if coverage_issues:
        print("\n❌ Symbols with Incomplete Coverage:")
        coverage_issues.sort(key=lambda x: x["missing_days"], reverse=True)

        for issue in coverage_issues[:10]:  # Show top 10 worst
            print(f"    {issue['symbol']:6s}: {issue['days']:2d} days "
                  f"(missing {issue['missing_days']:2d}) "
                  f"[{issue['start']} to {issue['end']}]")

        if len(coverage_issues) > 10:
            print(f"    ... and {len(coverage_issues) - 10} more symbols")

    # Volume analysis for data quality
    print("\n📈 Volume Distribution:")

    # Sort by record count
    complete_files.sort(key=lambda x: x["record_count"], reverse=True)

    print("  Top 5 Highest Volume:")
    for result in complete_files[:5]:
        print(f"    {result['symbol']:6s}: {result['record_count']:>12,} records "
              f"({result['file_size_mb']:6.1f} MB)")

    print("  Bottom 5 Lowest Volume:")
    for result in complete_files[-5:]:
        print(f"    {result['symbol']:6s}: {result['record_count']:>12,} records "
              f"({result['file_size_mb']:6.1f} MB)")

    # Summary statistics
    total_records = sum(r["record_count"] for r in complete_files)
    total_size = sum(r["file_size_mb"] for r in complete_files)

    print("\n📋 Summary:")
    print(f"  Total Symbols: {len(complete_files)}")
    print(f"  Total Records: {total_records:,}")
    print(f"  Total Size: {total_size:,.1f} MB")
    print(f"  Average Records/Symbol: {total_records // len(complete_files):,}")

    if errors:
        print(f"\n❌ Errors Found ({len(errors)}):")
        for error in errors:
            print(f"    {error['symbol']}: {error['error']}")

    # Generate recommendations
    print("\n🔧 Recommendations:")

    if coverage_issues:
        print(f"  1. Re-download {len(coverage_issues)} symbols with incomplete coverage")
        print("  2. Verify data availability for missing date ranges")

    if not start_aligned or not end_aligned:
        print("  3. Standardize date ranges across all symbols")

    if errors:
        print(f"  4. Fix {len(errors)} files with errors")

    if len(coverage_issues) == 0 and start_aligned and end_aligned and len(errors) == 0:
        print("  ✓ Data appears complete and well-aligned!")

    return {
        "total_symbols": len(complete_files),
        "coverage_issues": len(coverage_issues),
        "errors": len(errors),
        "perfect_alignment": start_aligned and end_aligned,
        "total_records": total_records
    }

if __name__ == "__main__":
    main()
