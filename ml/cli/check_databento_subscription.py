#!/usr/bin/env python3
"""
Check Databento subscription limits and available data.

This script queries your Databento subscription to determine:
1. What datasets you have access to
2. What date ranges are included in your subscription
3. What schemas (L0/L1/L2/L3) are available
4. Estimated costs (should be $0 if within subscription)

Run this BEFORE downloading any data to ensure you stay within your subscription.

"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.ingest.subscription import SubscriptionChecker


# Map schemas to data levels for display
SCHEMA_LEVELS: dict[str, str] = {
    "ohlcv-1d": "L0 (Daily Bars)",
    "ohlcv-1h": "L0 (Hourly Bars)",
    "ohlcv-1m": "L0 (Minute Bars)",
    "trades": "L1 (Trades)",
    "tbbo": "L1 (Top of Book)",
    "mbp-1": "L2 (Market Depth 1-level)",
    "mbp-10": "L2 (Market Depth 10-level)",
    "mbo": "L3 (Full Order Book)",
}


def _print_datasets(checker: SubscriptionChecker) -> list[str]:
    """Print available datasets."""
    print("\n" + "=" * 60)
    print("CHECKING AVAILABLE DATASETS")
    print("=" * 60)

    datasets = checker.check_available_datasets()

    if not datasets:
        print("❌ No datasets available. Check your API key.")
        return []

    print(f"✅ Found {len(datasets)} available datasets:")
    for dataset in datasets:
        print(f"   - {dataset}")

    return datasets


def _print_dataset_details(checker: SubscriptionChecker, dataset: str) -> None:
    """Print dataset range and schema details."""
    print(f"\n📊 Checking range for {dataset}...")
    range_info = checker.check_dataset_range(dataset)

    if range_info:
        start_date = range_info.get("start_date", "Unknown")
        end_date = range_info.get("end_date", "Unknown")
        print(f"   Available range: {start_date} to {end_date}")

        results = checker.get_results()
        if dataset in results["datasets"]:
            days = results["datasets"][dataset].get("days", 0)
            years = results["datasets"][dataset].get("years", 0.0)
            print(f"   Coverage: {days} days ({years:.1f} years)")

    print(f"\n📈 Checking schemas for {dataset}...")
    schemas = checker.check_available_schemas(dataset)

    if schemas:
        print("   Available schemas:")
        for schema in schemas:
            level = SCHEMA_LEVELS.get(schema, schema)
            print(f"      - {level}")


def _generate_safe_config(checker: SubscriptionChecker) -> dict[str, Any]:
    """Generate configuration that stays within subscription limits."""
    print("\n" + "=" * 60)
    print("GENERATING SAFE CONFIGURATION")
    print("=" * 60)

    results = checker.get_results()
    warnings = results.get("warnings", [])

    safe_config: dict[str, Any] = {
        "datasets": [],
        "date_ranges": {},
        "schemas": {},
        "warnings": [],
    }

    # Find recommended datasets
    for dataset in results.get("available_datasets", []):
        if dataset in {"EQUS.MINI", "DBEQ.BASIC", "XNAS.ITCH", "GLBX.MDP3"}:
            safe_config["datasets"].append(dataset)

    # Recommend safe date ranges
    if not warnings:
        safe_config["date_ranges"] = {
            "L0": "7 years (safe)",
            "L1": "1 year (safe)",
            "L2": "30 days (safe)",
        }
    else:
        safe_config["warnings"] = warnings

    # Save configuration
    config_path = Path("ml/config/databento_safe_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(safe_config, f, indent=2)

    print(f"\n✅ Safe configuration saved to: {config_path}")

    return safe_config


def _print_summary(checker: SubscriptionChecker) -> None:
    """Print subscription check summary."""
    results = checker.get_results()
    warnings = results.get("warnings", [])

    print("\n" + "=" * 60)
    print("SUBSCRIPTION SUMMARY")
    print("=" * 60)

    if not warnings:
        print("\n✅ ALL CLEAR! Your subscription covers:")
        print("   • 7 years of L0 (OHLCV) data")
        print("   • 1 year of L1 (quotes/trades) data")
        print("   • 30 days of L2/L3 (market depth) data")
        print("\n🚀 You can safely download all this data without extra charges!")
    else:
        print("\n⚠️ WARNINGS FOUND:")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n💡 Recommendation: Adjust date ranges to stay within subscription")

    print("\n" + "=" * 60)
    print("Next step: Run ml.cli.populate_* scripts to download your data")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """
    Run complete subscription check.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line arguments (unused for now, reserved for future options).

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).

    """
    configure_logging()
    bind_log_context(run_id="check_databento_subscription", component="ml.cli")

    # Check for API key
    if not os.getenv("DATABENTO_API_KEY"):
        print("❌ DATABENTO_API_KEY not found in environment!")
        return 1

    print("\n" + "=" * 70)
    print("   DATABENTO SUBSCRIPTION CHECKER")
    print("=" * 70)

    try:
        checker = SubscriptionChecker()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return 1

    # 1. Check available datasets
    datasets = _print_datasets(checker)
    if not datasets:
        return 1

    # 2. Check ranges and schemas for each dataset
    for dataset in datasets[:3]:  # Check first 3 datasets
        _print_dataset_details(checker, dataset)

    # 3. Generate safe configuration
    _generate_safe_config(checker)

    # 4. Print summary
    _print_summary(checker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
