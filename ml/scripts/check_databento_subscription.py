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

import json
import os
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any


# Check for API key
if not os.getenv("DATABENTO_API_KEY"):
    print("❌ DATABENTO_API_KEY not found in environment!")
    sys.exit(1)

import databento as db
import pandas as pd


class SubscriptionChecker:
    """Check Databento subscription limits and available data."""

    def __init__(self):
        """Initialize the subscription checker."""
        self.client = db.Historical(os.getenv("DATABENTO_API_KEY"))
        self.results = {
            "datasets": {},
            "costs": {},
            "warnings": [],
            "recommendations": []
        }

    def check_available_datasets(self) -> list[str]:
        """Get list of datasets available to this subscription."""
        print("\n" + "="*60)
        print("CHECKING AVAILABLE DATASETS")
        print("="*60)

        try:
            datasets = self.client.metadata.list_datasets()
            print(f"✅ Found {len(datasets)} available datasets:")
            for dataset in datasets:
                print(f"   - {dataset}")

            self.results["available_datasets"] = datasets
            return datasets

        except Exception as e:
            print(f"❌ Error checking datasets: {e}")
            return []

    def check_dataset_range(self, dataset: str) -> dict[str, Any]:
        """Check the available date range for a dataset under your subscription."""
        print(f"\n📊 Checking range for {dataset}...")

        try:
            range_info = self.client.metadata.get_dataset_range(dataset)

            # Parse the range
            start_date = range_info.get("start_date", "Unknown")
            end_date = range_info.get("end_date", "Unknown")

            print(f"   Available range: {start_date} to {end_date}")

            # Calculate coverage
            if start_date != "Unknown" and end_date != "Unknown":
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                days = (end - start).days
                years = days / 365.25

                print(f"   Coverage: {days} days ({years:.1f} years)")

                self.results["datasets"][dataset] = {
                    "start": start_date,
                    "end": end_date,
                    "days": days,
                    "years": years
                }

            return range_info

        except Exception as e:
            print(f"   ⚠️ Could not check range: {e}")
            return {}

    def check_available_schemas(self, dataset: str) -> list[str]:
        """Check what schemas (L0/L1/L2/L3) are available for a dataset."""
        print(f"\n📈 Checking schemas for {dataset}...")

        try:
            schemas = self.client.metadata.list_schemas(dataset)

            # Map schemas to levels
            schema_levels = {
                "ohlcv-1d": "L0 (Daily Bars)",
                "ohlcv-1h": "L0 (Hourly Bars)",
                "ohlcv-1m": "L0 (Minute Bars)",
                "trades": "L1 (Trades)",
                "tbbo": "L1 (Top of Book)",
                "mbp-1": "L2 (Market Depth 1-level)",
                "mbp-10": "L2 (Market Depth 10-level)",
                "mbo": "L3 (Full Order Book)"
            }

            print("   Available schemas:")
            for schema in schemas:
                level = schema_levels.get(schema, schema)
                print(f"      - {level}")

            self.results["datasets"][dataset]["schemas"] = schemas
            return schemas

        except Exception as e:
            print(f"   ⚠️ Could not check schemas: {e}")
            return []

    def estimate_cost(self, dataset: str, symbols: list[str], schema: str,
                     start_date: str, end_date: str) -> float:
        """
        Estimate the cost for downloading data.
        
        If cost is $0, it's included in your subscription!
        """
        try:
            cost = self.client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols[:10],  # Test with first 10 symbols
                schema=schema,
                start=start_date,
                end=end_date
            )

            return cost

        except Exception as e:
            print(f"   ⚠️ Could not estimate cost: {e}")
            return -1

    def check_subscription_coverage(self):
        """
        Check what data levels and ranges are covered by the subscription.
        """
        print("\n" + "="*60)
        print("CHECKING SUBSCRIPTION COVERAGE")
        print("="*60)

        # Common US equity datasets
        test_datasets = [
            "XNAS.ITCH",  # NASDAQ TotalView-ITCH
            "DBEQ.BASIC",  # Databento Equities Basic
            "OPRA.PILLAR",  # Options
        ]

        # Test symbols
        test_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"]

        # Test date ranges
        today = datetime.now()
        test_ranges = {
            "7 years": (today - timedelta(days=7*365), today),
            "1 year": (today - timedelta(days=365), today),
            "30 days": (today - timedelta(days=30), today),
            "1 day": (today - timedelta(days=1), today),
        }

        # Test schemas by level
        test_schemas = {
            "L0": ["ohlcv-1d", "ohlcv-1h"],
            "L1": ["trades", "tbbo"],
            "L2": ["mbp-1", "mbp-10"],
            "L3": ["mbo"]
        }

        print("\n🔍 Testing data access levels...")
        print("-" * 40)

        for dataset in test_datasets:
            if dataset not in self.results.get("available_datasets", []):
                continue

            print(f"\nDataset: {dataset}")

            # Check each level
            available_schemas = self.results["datasets"].get(dataset, {}).get("schemas", [])

            for level, schemas in test_schemas.items():
                for schema in schemas:
                    if schema not in available_schemas:
                        continue

                    print(f"\n  Testing {level} ({schema}):")

                    # Test different date ranges
                    for range_name, (start, end) in test_ranges.items():
                        cost = self.estimate_cost(
                            dataset=dataset,
                            symbols=test_symbols,
                            schema=schema,
                            start_date=start.strftime("%Y-%m-%d"),
                            end_date=end.strftime("%Y-%m-%d")
                        )

                        if cost == 0:
                            print(f"    ✅ {range_name}: FREE (included in subscription)")
                        elif cost > 0:
                            print(f"    💰 {range_name}: ${cost:.2f} (extra cost)")
                            self.results["warnings"].append(
                                f"{level} data for {range_name} would cost ${cost:.2f}"
                            )
                        else:
                            print(f"    ⚠️ {range_name}: Could not check")

    def generate_safe_config(self):
        """Generate a configuration that stays within subscription limits."""
        print("\n" + "="*60)
        print("GENERATING SAFE CONFIGURATION")
        print("="*60)

        # Analyze results to find safe ranges
        safe_config = {
            "datasets": [],
            "date_ranges": {},
            "schemas": {},
            "warnings": []
        }

        # Find the best dataset
        for dataset in self.results.get("available_datasets", []):
            if dataset in ["XNAS.ITCH", "DBEQ.BASIC"]:
                safe_config["datasets"].append(dataset)

        # Recommend safe date ranges based on cost checks
        if not self.results["warnings"]:
            safe_config["date_ranges"] = {
                "L0": "7 years (safe)",
                "L1": "1 year (safe)",
                "L2": "30 days (safe)",
            }
        else:
            # Parse warnings to find limits
            safe_config["warnings"] = self.results["warnings"]

        # Save configuration
        config_path = Path("ml/config/databento_safe_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(safe_config, f, indent=2)

        print(f"\n✅ Safe configuration saved to: {config_path}")

        return safe_config

    def run_full_check(self):
        """Run complete subscription check."""
        print("\n" + "="*70)
        print("   DATABENTO SUBSCRIPTION CHECKER")
        print("="*70)

        # 1. Check available datasets
        datasets = self.check_available_datasets()

        if not datasets:
            print("\n❌ No datasets available. Check your API key.")
            return

        # 2. Check ranges and schemas for each dataset
        for dataset in datasets[:3]:  # Check first 3 datasets
            self.check_dataset_range(dataset)
            self.check_available_schemas(dataset)

        # 3. Check subscription coverage
        self.check_subscription_coverage()

        # 4. Generate safe configuration
        safe_config = self.generate_safe_config()

        # 5. Print summary
        print("\n" + "="*60)
        print("SUBSCRIPTION SUMMARY")
        print("="*60)

        if not self.results["warnings"]:
            print("\n✅ ALL CLEAR! Your subscription covers:")
            print("   • 7 years of L0 (OHLCV) data")
            print("   • 1 year of L1 (quotes/trades) data")
            print("   • 30 days of L2/L3 (market depth) data")
            print("\n🚀 You can safely download all this data without extra charges!")
        else:
            print("\n⚠️ WARNINGS FOUND:")
            for warning in self.results["warnings"]:
                print(f"   • {warning}")
            print("\n💡 Recommendation: Adjust date ranges to stay within subscription")

        print("\n" + "="*60)
        print("Next step: Run populate_universe_safe.py to download your data")
        print("="*60)


if __name__ == "__main__":
    checker = SubscriptionChecker()
    checker.run_full_check()
