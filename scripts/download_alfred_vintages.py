#!/usr/bin/env python3
"""
Download ALFRED (vintage FRED) data for all 30 macro series.

This ensures point-in-time correctness by downloading historical releases
of macro indicators, avoiding lookahead bias in training.

Usage:
    python scripts/download_alfred_vintages.py

Requires:
    - FRED_API_KEY environment variable set
    - ~15-30 minutes runtime (API rate limits)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.common.env import load_project_dotenv
from ml.common.logging_config import configure_logging
from ml.config.macro_universe import MARKET_BASED_MACRO_SERIES
from ml.config.macro_universe import TIER1_MACRO_SERIES_UNIVERSE
from ml.data.loaders.alfred_loader import ALFREDConfig
from ml.data.loaders.alfred_loader import ALFREDDataLoader


def main() -> int:
    configure_logging(level="INFO")
    logger = logging.getLogger(__name__)
    load_project_dotenv()

    # Expanded macro universe with depth features organised by category
    series_by_category = dict(TIER1_MACRO_SERIES_UNIVERSE.categories)
    all_series = list(TIER1_MACRO_SERIES_UNIVERSE.all_series())

    print(f"{'='*80}")
    print("ALFRED Vintage Data Download")
    print(f"{'='*80}")
    print(f"Series to download: {len(all_series)}")
    print()

    for category, series_list in series_by_category.items():
        print(f"  {category:20s}: {', '.join(series_list)}")
    print()

    # Check what already exists
    vintage_dir = Path("data/features/macro/fred/vintages")
    existing = set()
    if vintage_dir.exists():
        existing = {d.name for d in vintage_dir.iterdir() if d.is_dir()}
        print(f"Already downloaded: {len(existing)} series")
        if existing:
            print(f"  Existing: {', '.join(sorted(existing))}")
    print()

    missing = set(all_series) - existing
    if not missing:
        print("✅ All 30 series already downloaded!")
        return 0

    print(f"To download: {len(missing)} series")
    print(f"  Missing: {', '.join(sorted(missing))}")
    print()

    # Confirm
    response = input(f"Download {len(missing)} series? This will take ~15-30 minutes. (y/N): ")
    if response.lower() not in ("y", "yes"):
        print("Cancelled")
        return 1

    print()
    print(f"{'='*80}")
    print("Starting Download")
    print(f"{'='*80}")
    print()

    # Configure ALFRED loader
    config = ALFREDConfig(
        series_ids=tuple(sorted(missing)),  # Only download missing
        out_dir=vintage_dir,
        start_date="2015-01-01",  # Last 10 years of vintages
        window_days=365,           # Download in yearly chunks to respect API limits
        max_retries=2,
        fallback_to_fred_series=MARKET_BASED_MACRO_SERIES,
    )

    try:
        loader = ALFREDDataLoader(config)
        stats = loader.refresh()

        print()
        print(f"{'='*80}")
        print("Download Complete!")
        print(f"{'='*80}")
        print()

        # Summary
        total_releases = 0
        for series_id, series_stats in stats.items():
            releases = series_stats.get("releases", 0)
            total_releases += releases
            print(f"  {series_id:20s}: {releases:4d} vintage releases")

        print()
        print(f"Total: {len(stats)} series, {total_releases:,} vintage releases")
        print()
        print("✅ Ready for point-in-time training!")
        print()
        print("Next steps:")
        print("  1. Update config: vintage_policy = 'real_time'")
        print("  2. Update config: fred_vintage_dir = 'data/features/macro/fred/vintages'")
        print("  3. Run: python -m ml.cli.pipeline_orchestrator --config <config> --stage dataset")
        print()

        return 0

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        print()
        print(f"❌ Error: {e}")
        print()
        if "FRED_API_KEY" in str(e):
            print("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            print("Then set it: export FRED_API_KEY='your_key_here'")
            print()
        return 1

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        print()
        print(f"❌ Error: {e}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
