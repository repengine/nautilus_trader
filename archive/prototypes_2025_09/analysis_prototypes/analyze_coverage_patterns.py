#!/usr/bin/env python3
"""
Analyze actual data coverage patterns to determine optimal date ranges.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars as pl


def analyze_all_coverage():
    """
    Analyze actual date coverage across all symbols.
    """
    data_dir = Path("data/tier1")
    parquet_files = list(data_dir.glob("**/l2/*_mbp-10.parquet"))

    coverage_data = []

    for file_path in parquet_files:
        symbol = file_path.stem.replace("_mbp-10", "")

        try:
            # Sample data to get date range
            df = pl.scan_parquet(file_path)

            first_rows = df.head(1000).collect()
            last_rows = df.tail(1000).collect()

            if len(first_rows) == 0:
                continue

            # Get full date range
            min_ts = min(first_rows["ts_event"].min(), last_rows["ts_event"].min())
            max_ts = max(first_rows["ts_event"].max(), last_rows["ts_event"].max())

            min_date = pd.to_datetime(min_ts, unit="ns").date()
            max_date = pd.to_datetime(max_ts, unit="ns").date()

            coverage_data.append(
                {
                    "symbol": symbol,
                    "start_date": min_date,
                    "end_date": max_date,
                    "days_covered": (max_date - min_date).days + 1,
                },
            )

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    return coverage_data


def main():
    print("📊 Analyzing Actual Data Coverage Patterns")
    print("=" * 60)

    coverage_data = analyze_all_coverage()

    if not coverage_data:
        print("No data found!")
        return

    # Sort by start date
    coverage_data.sort(key=lambda x: x["start_date"])

    # Find common patterns
    start_dates = defaultdict(int)
    end_dates = defaultdict(int)

    for item in coverage_data:
        start_dates[item["start_date"]] += 1
        end_dates[item["end_date"]] += 1

    print("🗓️ Most Common Start Dates:")
    for date, count in sorted(start_dates.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {date}: {count} symbols")

    print("\n🗓️ Most Common End Dates:")
    for date, count in sorted(end_dates.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {date}: {count} symbols")

    # Find the optimal range that maximizes symbol coverage
    all_starts = [item["start_date"] for item in coverage_data]
    all_ends = [item["end_date"] for item in coverage_data]

    optimal_start = max(all_starts)  # Latest start (most conservative)
    optimal_end = min(all_ends)  # Earliest end (most conservative)

    print("\n🎯 Optimal Conservative Range:")
    print(f"  Start: {optimal_start}")
    print(f"  End: {optimal_end}")

    if optimal_end >= optimal_start:
        optimal_days = (optimal_end - optimal_start).days + 1
        print(f"  Days: {optimal_days}")

        # Count symbols with complete coverage in this range
        complete_symbols = []
        for item in coverage_data:
            if item["start_date"] <= optimal_start and item["end_date"] >= optimal_end:
                complete_symbols.append(item["symbol"])

        print(f"  Complete symbols: {len(complete_symbols)}/{len(coverage_data)}")
        print(
            f"  Symbols: {', '.join(sorted(complete_symbols)[:10])}{'...' if len(complete_symbols) > 10 else ''}",
        )
    else:
        print("  No overlapping coverage found!")

    # Alternative: Find best balance range
    print("\n⚖️ Coverage Analysis by Potential Ranges:")

    # Test different start dates
    unique_starts = sorted(set(all_starts))
    unique_ends = sorted(set(all_ends), reverse=True)

    best_coverage = 0
    best_range = None

    for start in unique_starts[-3:]:  # Try last 3 starts
        for end in unique_ends[:3]:  # Try first 3 ends
            if end >= start:
                # Count coverage
                covered = sum(
                    1
                    for item in coverage_data
                    if item["start_date"] <= start and item["end_date"] >= end
                )
                days = (end - start).days + 1

                print(
                    f"  {start} to {end} ({days:2d} days): {covered:2d}/{len(coverage_data)} symbols ({100*covered/len(coverage_data):4.1f}%)",
                )

                if covered > best_coverage:
                    best_coverage = covered
                    best_range = (start, end, days)

    if best_range:
        start, end, days = best_range
        print("\n🏆 Recommended Range for Maximum Coverage:")
        print(f"  Start: {start}")
        print(f"  End: {end}")
        print(f"  Days: {days}")
        print(
            f"  Coverage: {best_coverage}/{len(coverage_data)} symbols ({100*best_coverage/len(coverage_data):.1f}%)",
        )

        # List symbols with issues in this range
        problem_symbols = []
        for item in coverage_data:
            if not (item["start_date"] <= start and item["end_date"] >= end):
                problem_symbols.append(f"{item['symbol']}({item['start_date']}-{item['end_date']})")

        if problem_symbols:
            print(f"\n❌ Symbols needing re-download: {len(problem_symbols)}")
            print(f"  {', '.join(problem_symbols[:8])}{'...' if len(problem_symbols) > 8 else ''}")

    # Show extreme cases
    print("\n📈 Coverage Distribution:")
    print(f"  Shortest coverage: {min(item['days_covered'] for item in coverage_data)} days")
    print(f"  Longest coverage: {max(item['days_covered'] for item in coverage_data)} days")
    print(
        f"  Average coverage: {sum(item['days_covered'] for item in coverage_data) / len(coverage_data):.1f} days",
    )


if __name__ == "__main__":
    main()
