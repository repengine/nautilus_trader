#!/usr/bin/env python3
"""
Consolidated data analysis tool for Nautilus Trader ML pipeline.

Combines functionality from:
- analyze_existing_gaps.py
- analyze_data_completeness.py
- analyze_coverage_patterns.py
- data_consolidation_analysis.py
"""

import argparse
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl


def analyze_data_gaps(data_dir="data/tier1"):
    """Analyze gaps in existing data files."""
    print("🔍 Data Gap Analysis")
    print("=" * 40)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    gaps_found = 0
    total_files = 0

    for schema_dir in data_path.iterdir():
        if not schema_dir.is_dir():
            continue

        print(f"\n📊 Schema: {schema_dir.name}")
        schema_files = list(schema_dir.glob("*.parquet"))
        total_files += len(schema_files)

        if not schema_files:
            print("   No parquet files found")
            continue

        # Analyze date coverage
        dates = []
        for file in schema_files:
            # Extract date from filename (assuming YYYY-MM-DD format)
            date_str = file.stem.split("_")[-1] if "_" in file.stem else file.stem
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.append(date)
            except ValueError:
                continue

        if dates:
            dates.sort()
            date_range = pd.date_range(dates[0], dates[-1], freq="D")
            missing_dates = [d.date() for d in date_range if d.date() not in dates]

            if missing_dates:
                gaps_found += len(missing_dates)
                print(f"   📅 Date range: {dates[0]} to {dates[-1]}")
                print(f"   ⚠️  Missing {len(missing_dates)} dates")
                if len(missing_dates) <= 5:
                    print(f"      {missing_dates}")
            else:
                print(f"   ✅ Complete coverage: {dates[0]} to {dates[-1]}")

    print("\n📈 Summary:")
    print(f"   Total files analyzed: {total_files}")
    print(f"   Total gaps found: {gaps_found}")


def analyze_data_completeness(data_dir="data/tier1"):
    """Analyze completeness of data within files."""
    print("\n🎯 Data Completeness Analysis")
    print("=" * 40)

    data_path = Path(data_dir)
    total_records = 0
    total_nulls = 0

    for schema_dir in data_path.iterdir():
        if not schema_dir.is_dir():
            continue

        print(f"\n📊 Schema: {schema_dir.name}")

        # Sample a few files for analysis
        files = list(schema_dir.glob("*.parquet"))[:3]

        for file in files:
            try:
                df = pl.read_parquet(file)
                records = len(df)
                nulls = df.null_count().sum_horizontal().sum()

                total_records += records
                total_nulls += nulls

                null_pct = (nulls / (records * len(df.columns))) * 100 if records > 0 else 0

                print(f"   📄 {file.name}")
                print(f"      Records: {records:,}, Nulls: {nulls:,} ({null_pct:.1f}%)")

            except Exception as e:
                print(f"   ❌ Error reading {file.name}: {e}")

    overall_null_pct = (total_nulls / total_records) * 100 if total_records > 0 else 0
    print("\n📈 Overall Completeness:")
    print(f"   Total records sampled: {total_records:,}")
    print(f"   Overall null rate: {overall_null_pct:.2f}%")


def analyze_coverage_patterns(data_dir="data/tier1"):
    """Analyze coverage patterns across schemas and dates."""
    print("\n📈 Coverage Pattern Analysis")
    print("=" * 40)

    data_path = Path(data_dir)
    schema_stats = {}

    for schema_dir in data_path.iterdir():
        if not schema_dir.is_dir():
            continue

        files = list(schema_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)

        # Extract dates
        dates = []
        for file in files:
            date_str = file.stem.split("_")[-1] if "_" in file.stem else file.stem
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.append(date)
            except ValueError:
                continue

        if dates:
            dates.sort()
            schema_stats[schema_dir.name] = {
                "files": len(files),
                "size_mb": total_size / (1024 * 1024),
                "date_range": (dates[0], dates[-1]),
                "days_covered": len(dates),
                "days_total": (dates[-1] - dates[0]).days + 1 if dates else 0
            }

    # Display results
    for schema, stats in sorted(schema_stats.items()):
        coverage_pct = (stats["days_covered"] / stats["days_total"]) * 100 if stats["days_total"] > 0 else 0

        print(f"\n📊 {schema}:")
        print(f"   Files: {stats['files']}, Size: {stats['size_mb']:.1f} MB")
        print(f"   Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"   Coverage: {stats['days_covered']}/{stats['days_total']} days ({coverage_pct:.1f}%)")


def consolidation_recommendations(data_dir="data/tier1"):
    """Provide data consolidation recommendations."""
    print("\n💡 Consolidation Recommendations")
    print("=" * 40)

    data_path = Path(data_dir)
    recommendations = []

    # Check for small files that could be consolidated
    for schema_dir in data_path.iterdir():
        if not schema_dir.is_dir():
            continue

        files = list(schema_dir.glob("*.parquet"))
        small_files = [f for f in files if f.stat().st_size < 1024 * 1024]  # < 1MB

        if len(small_files) > 10:
            recommendations.append(f"📦 {schema_dir.name}: {len(small_files)} small files (<1MB) could be consolidated")

    # Check for missing recent data
    cutoff_date = datetime.now().date() - timedelta(days=7)
    for schema_dir in data_path.iterdir():
        if not schema_dir.is_dir():
            continue

        files = list(schema_dir.glob("*.parquet"))
        if not files:
            continue

        # Get latest file date
        latest_date = None
        for file in files:
            date_str = file.stem.split("_")[-1] if "_" in file.stem else file.stem
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if latest_date is None or date > latest_date:
                    latest_date = date
            except ValueError:
                continue

        if latest_date and latest_date < cutoff_date:
            days_behind = (cutoff_date - latest_date).days
            recommendations.append(f"📅 {schema_dir.name}: {days_behind} days behind (latest: {latest_date})")

    if recommendations:
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("   ✅ No major consolidation issues found")


def main():
    """Run data analysis with CLI interface."""
    parser = argparse.ArgumentParser(description="Consolidated data analysis tool")
    parser.add_argument("--data-dir", default="data/tier1", help="Data directory to analyze")
    parser.add_argument("--analysis", choices=["gaps", "completeness", "patterns", "recommendations", "all"],
                       default="all", help="Type of analysis to run")

    args = parser.parse_args()

    print("🔬 Nautilus Trader Data Analysis")
    print("=" * 50)
    print(f"Analyzing data directory: {args.data_dir}")

    if args.analysis in ["gaps", "all"]:
        analyze_data_gaps(args.data_dir)

    if args.analysis in ["completeness", "all"]:
        analyze_data_completeness(args.data_dir)

    if args.analysis in ["patterns", "all"]:
        analyze_coverage_patterns(args.data_dir)

    if args.analysis in ["recommendations", "all"]:
        consolidation_recommendations(args.data_dir)

    print("\n" + "=" * 50)
    print("🎉 Analysis complete!")

    # Usage examples
    print("\nUsage examples:")
    print("  python tools/data_analysis.py --analysis gaps")
    print("  python tools/data_analysis.py --data-dir data/tier2")
    print("  python tools/data_analysis.py --analysis recommendations")


if __name__ == "__main__":
    main()
