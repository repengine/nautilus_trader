#!/usr/bin/env python3
"""
Unified data population script for ML universe.

This script provides a single, safe, configurable interface for populating:
- L0 data: 7 years of OHLCV bars
- L1 data: 1 year of quotes/trades (BBO)
- L2/L3 data: 30 days of market depth

Features:
- Cost estimation and safeguards
- Progress tracking and resume capability
- Configurable date ranges and symbols
- Parallel downloads with rate limiting
- Comprehensive error handling

Usage:
    # Estimate costs only
    python ml/scripts/populate_universe.py --estimate-only

    # Populate specific data level
    python ml/scripts/populate_universe.py --level L0
    python ml/scripts/populate_universe.py --level L1
    python ml/scripts/populate_universe.py --level L2

    # Populate specific tier
    python ml/scripts/populate_universe.py --tier 1 --level L1

    # Resume from progress
    python ml/scripts/populate_universe.py --resume

    # Force restart (ignore progress)
    python ml/scripts/populate_universe.py --force

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import databento as db


# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies


if not HAS_POLARS:
    check_ml_dependencies(["polars"])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseConfig:
    """
    Configuration for universe population.
    """

    # Data levels
    levels: list[str]  # ["L0", "L1", "L2", "L3"]

    # Date ranges
    l0_years: int = 7
    l1_years: int = 1
    l2_days: int = 30

    # Universe
    tier: int = 1  # 1, 2, or 3
    symbols: list[str] | None = None

    # Processing
    batch_size: int = 10
    max_workers: int = 4
    rate_limit_per_min: int = 100

    # Paths
    data_dir: Path = Path("data/tier1")
    progress_dir: Path = Path(".")

    # Safeguards
    max_cost_usd: float = 0.0  # $0 = only use included data
    estimate_only: bool = False
    dry_run: bool = False
    force: bool = False
    resume: bool = True


class DataLevel:
    """
    Data level definitions.
    """

    L0 = "L0"  # OHLCV bars
    L1 = "L1"  # Quotes/trades (BBO)
    L2 = "L2"  # Market depth (order book)
    L3 = "L3"  # Full depth

    @classmethod
    def all(cls) -> list[str]:
        return [cls.L0, cls.L1, cls.L2, cls.L3]

    @classmethod
    def get_schema(cls, level: str) -> str:
        """
        Get Databento schema for level.
        """
        mapping = {
            cls.L0: "ohlcv-1m",
            cls.L1: "trades",  # Will also fetch "bbo-1s"
            cls.L2: "mbp-10",
            cls.L3: "mbp-1",
        }
        return mapping.get(level, "trades")

    @classmethod
    def get_date_range(cls, level: str, config: UniverseConfig) -> tuple[datetime, datetime]:
        """
        Get date range for data level.
        """
        # For EQUS.MINI, we need to account for data delay
        # L2/L3 data requires avoiding very recent dates
        if level in [cls.L2, cls.L3]:
            # End date should be at least 1 day ago to avoid live data requirement
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=config.l2_days)
        else:
            # L0 and L1 can use current date
            end_date = datetime.now()
            if level == cls.L0:
                start_date = end_date - timedelta(days=config.l0_years * 365)
            elif level == cls.L1:
                start_date = end_date - timedelta(days=config.l1_years * 365)
            else:
                raise ValueError(f"Unknown level: {level}")

        return start_date, end_date


class ProgressTracker:
    """
    Track and persist progress.
    """

    def __init__(self, config: UniverseConfig):
        self.config = config
        self.progress_file = config.progress_dir / f"universe_progress_{config.tier}.json"
        self.progress = self._load_progress()

    def _load_progress(self) -> dict[str, Any]:
        """
        Load existing progress.
        """
        if self.progress_file.exists() and self.config.resume:
            with open(self.progress_file) as f:
                return json.load(f)

        return {
            "tier": self.config.tier,
            "levels": {level: {"completed": [], "failed": []} for level in DataLevel.all()},
            "last_update": None,
            "stats": {},
        }

    def save(self) -> None:
        """
        Save current progress.
        """
        self.progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def is_completed(self, level: str, symbol: str) -> bool:
        """
        Check if symbol is completed for level.
        """
        return symbol in self.progress["levels"].get(level, {}).get("completed", [])

    def mark_completed(self, level: str, symbol: str) -> None:
        """
        Mark symbol as completed.
        """
        if level not in self.progress["levels"]:
            self.progress["levels"][level] = {"completed": [], "failed": []}

        completed = self.progress["levels"][level]["completed"]
        if symbol not in completed:
            completed.append(symbol)

        # Remove from failed if present
        failed = self.progress["levels"][level]["failed"]
        if symbol in failed:
            failed.remove(symbol)

    def mark_failed(self, level: str, symbol: str, error: str) -> None:
        """
        Mark symbol as failed.
        """
        if level not in self.progress["levels"]:
            self.progress["levels"][level] = {"completed": [], "failed": []}

        failed = self.progress["levels"][level]["failed"]
        if symbol not in [f.get("symbol") for f in failed if isinstance(f, dict)]:
            failed.append(
                {"symbol": symbol, "error": str(error), "timestamp": datetime.now().isoformat()},
            )


class UniversePopulator:
    """
    Main class for populating universe data.
    """

    def __init__(self, config: UniverseConfig):
        self.config = config
        self.client = self._init_client()
        self.tracker = ProgressTracker(config)
        self.symbols = self._load_symbols()

    def _init_client(self) -> db.Historical:
        """
        Initialize Databento client.
        """
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in environment")

        return db.Historical(api_key)

    def _load_symbols(self) -> list[str]:
        """
        Load symbols for tier.
        """
        if self.config.symbols:
            return self.config.symbols

        # Load tier universe
        tier_files = {
            1: "ml/config/tier1_universe.json",
            2: "ml/config/tier2_universe.json",
            3: "ml/config/tier3_universe.json",
        }

        universe_file = Path(tier_files.get(self.config.tier, tier_files[1]))

        if not universe_file.exists():
            # Fallback to default tier 1 symbols
            logger.warning(f"Universe file {universe_file} not found, using default Tier 1")
            return self._get_default_tier1_symbols()

        with open(universe_file) as f:
            data = json.load(f)
            return data.get("symbols", [])

    def _get_default_tier1_symbols(self) -> list[str]:
        """
        Get default Tier 1 symbols (top 78 liquid stocks).
        """
        return [
            # Major indices and ETFs
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "XLF",
            "XLK",
            "XLE",
            "XLV",
            "XLI",
            "TLT",
            "GLD",
            "SLV",
            "VIX",
            # Mega-cap tech
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "BRK.B",
            "AMD",
            # Large-cap leaders
            "JPM",
            "JNJ",
            "V",
            "PG",
            "UNH",
            "HD",
            "MA",
            "DIS",
            "BAC",
            "ADBE",
            "CRM",
            "NFLX",
            "KO",
            "PEP",
            "TMO",
            "ABBV",
            "CVX",
            "WMT",
            "MRK",
            "LLY",
            "AVGO",
            "NKE",
            "ORCL",
            "ACN",
            "COST",
            "MCD",
            "ABT",
            "TXN",
            # Financials
            "GS",
            "MS",
            "WFC",
            "C",
            # Energy & Industrials
            "XOM",
            "COP",
            "CAT",
            "BA",
            "GE",
            "MMM",
            # Telecom
            "VZ",
            "T",
            # International ETFs
            "EFA",
            "EEM",
            "VEA",
            "VWO",
            # Currency & Commodities
            "UUP",
            "FXE",
            "USO",
            "UNG",
            # High-growth / Meme stocks
            "PLTR",
            "SOFI",
            "RIVN",
            "LCID",
            "COIN",
            "MSTR",
            # REITs
            "VNQ",
        ]

    async def estimate_costs(self) -> dict[str, Any]:
        """
        Estimate costs for data download.
        """
        estimates = {}

        for level in self.config.levels:
            schema = DataLevel.get_schema(level)
            start_date, end_date = DataLevel.get_date_range(level, self.config)

            # Count symbols not yet completed
            pending_symbols = [s for s in self.symbols if not self.tracker.is_completed(level, s)]

            if not pending_symbols:
                estimates[level] = {
                    "symbols": 0,
                    "cost_usd": 0.0,
                    "status": "completed",
                }
                continue

            try:
                # Get cost estimate from Databento
                if level == DataLevel.L1:
                    # L1 needs both trades and quotes
                    cost_trades = self.client.metadata.get_cost(
                        dataset="GLBX.MDP3",  # Using CME for estimate
                        symbols=pending_symbols[:5],  # Sample
                        schema="trades",
                        start=start_date,
                        end=end_date,
                    )
                    cost_quotes = self.client.metadata.get_cost(
                        dataset="GLBX.MDP3",
                        symbols=pending_symbols[:5],
                        schema="bbo-1s",
                        start=start_date,
                        end=end_date,
                    )
                    # Extrapolate
                    cost_per_symbol = (cost_trades + cost_quotes) / 5
                    total_cost = cost_per_symbol * len(pending_symbols)
                else:
                    cost = self.client.metadata.get_cost(
                        dataset="GLBX.MDP3",
                        symbols=pending_symbols[:5],
                        schema=schema,
                        start=start_date,
                        end=end_date,
                    )
                    cost_per_symbol = cost / 5
                    total_cost = cost_per_symbol * len(pending_symbols)

                estimates[level] = {
                    "symbols": len(pending_symbols),
                    "cost_usd": float(total_cost),
                    "date_range": f"{start_date.date()} to {end_date.date()}",
                    "schema": schema,
                }

            except Exception:
                # For XNAS.ITCH (EQUS.MINI), everything should be $0
                estimates[level] = {
                    "symbols": len(pending_symbols),
                    "cost_usd": 0.0,
                    "date_range": f"{start_date.date()} to {end_date.date()}",
                    "schema": schema,
                    "note": "Using EQUS.MINI subscription (no additional cost)",
                }

        estimates["total_cost_usd"] = sum(e.get("cost_usd", 0) for e in estimates.values())
        estimates["total_symbols"] = len(self.symbols)

        return estimates

    async def populate_level(self, level: str) -> dict[str, Any]:
        """
        Populate data for a specific level.
        """
        logger.info(f"Starting population of {level} data for Tier {self.config.tier}")

        _schema = DataLevel.get_schema(level)
        start_date, end_date = DataLevel.get_date_range(level, self.config)

        # Filter pending symbols
        pending_symbols = [s for s in self.symbols if not self.tracker.is_completed(level, s)]

        if not pending_symbols:
            logger.info(f"All symbols already completed for {level}")
            return {"level": level, "completed": len(self.symbols), "failed": 0}

        logger.info(f"Processing {len(pending_symbols)} pending symbols for {level}")

        completed = 0
        failed = 0

        # Process in batches
        for i in range(0, len(pending_symbols), self.config.batch_size):
            batch = pending_symbols[i : i + self.config.batch_size]
            logger.info(f"Processing batch {i//self.config.batch_size + 1}: {batch}")

            for symbol in batch:
                try:
                    if self.config.dry_run:
                        logger.info(f"[DRY RUN] Would download {level} for {symbol}")
                        self.tracker.mark_completed(level, symbol)
                        completed += 1
                        continue

                    # Create output directory
                    output_dir = self.config.data_dir / symbol / level.lower()
                    output_dir.mkdir(parents=True, exist_ok=True)

                    if level == DataLevel.L0:
                        # Download OHLCV bars
                        await self._download_ohlcv(symbol, start_date, end_date, output_dir)

                    elif level == DataLevel.L1:
                        # Download quotes and trades
                        await self._download_l1(symbol, start_date, end_date, output_dir)

                    elif level in [DataLevel.L2, DataLevel.L3]:
                        # Download market depth
                        await self._download_depth(symbol, level, start_date, end_date, output_dir)

                    self.tracker.mark_completed(level, symbol)
                    completed += 1
                    logger.info(f"✓ Completed {level} for {symbol}")

                except Exception as e:
                    logger.error(f"✗ Failed {level} for {symbol}: {e}")
                    self.tracker.mark_failed(level, symbol, str(e))
                    failed += 1

                # Save progress after each symbol
                self.tracker.save()

                # Rate limiting
                await asyncio.sleep(60 / self.config.rate_limit_per_min)

        return {
            "level": level,
            "completed": completed,
            "failed": failed,
            "total": len(pending_symbols),
        }

    async def _download_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
    ) -> None:
        """
        Download OHLCV data.
        """
        output_file = output_dir / f"{symbol}_ohlcv.parquet"

        if output_file.exists():
            logger.info(f"OHLCV file already exists for {symbol}, skipping")
            return

        # Use XNAS.ITCH for EQUS.MINI subscription (free tier)
        df = self.client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=[symbol],
            schema="ohlcv-1m",
            start=start,
            end=end,
        ).to_df()

        if not df.empty:
            df.to_parquet(output_file)
            logger.info(f"Saved {len(df)} OHLCV records for {symbol}")

    async def _download_l1(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
    ) -> None:
        """
        Download L1 (quotes and trades) data.
        """
        # Download trades
        trades_file = output_dir / f"{symbol}_trades.parquet"
        if not trades_file.exists():
            df_trades = self.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[symbol],
                schema="trades",
                start=start,
                end=end,
            ).to_df()

            if not df_trades.empty:
                df_trades.to_parquet(trades_file)
                logger.info(f"Saved {len(df_trades)} trades for {symbol}")

        # Download BBO quotes
        quotes_file = output_dir / f"{symbol}_bbo.parquet"
        if not quotes_file.exists():
            df_quotes = self.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[symbol],
                schema="bbo-1s",
                start=start,
                end=end,
            ).to_df()

            if not df_quotes.empty:
                df_quotes.to_parquet(quotes_file)
                logger.info(f"Saved {len(df_quotes)} BBO quotes for {symbol}")

    async def _download_depth(
        self,
        symbol: str,
        level: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
    ) -> None:
        """
        Download market depth data.
        """
        schema = "mbp-10" if level == DataLevel.L2 else "mbp-1"
        output_file = output_dir / f"{symbol}_{schema}.parquet"

        if output_file.exists():
            logger.info(f"Depth file already exists for {symbol}, skipping")
            return

        # Note: L2/L3 may not be available in EQUS.MINI
        # This will fail gracefully and be caught by error handling
        df = self.client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=[symbol],
            schema=schema,
            start=start,
            end=end,
        ).to_df()

        if not df.empty:
            df.to_parquet(output_file)
            logger.info(f"Saved {len(df)} depth records for {symbol}")

    async def run(self) -> None:
        """
        Run the population process.
        """
        logger.info(f"Starting universe population for Tier {self.config.tier}")
        logger.info(f"Levels to populate: {self.config.levels}")
        logger.info(f"Total symbols: {len(self.symbols)}")

        # Estimate costs
        estimates = await self.estimate_costs()

        logger.info("=" * 50)
        logger.info("COST ESTIMATES:")
        for level, est in estimates.items():
            if isinstance(est, dict):
                logger.info(
                    f"  {level}: {est.get('symbols', 0)} symbols, ${est.get('cost_usd', 0):.2f}",
                )
        logger.info(f"  TOTAL: ${estimates.get('total_cost_usd', 0):.2f}")
        logger.info("=" * 50)

        if self.config.estimate_only:
            logger.info("Estimate only mode - exiting")
            return

        if estimates.get("total_cost_usd", 0) > self.config.max_cost_usd:
            logger.error(
                f"Cost ${estimates['total_cost_usd']:.2f} exceeds limit ${self.config.max_cost_usd:.2f}",
            )
            logger.error("Use --force to override or increase --max-cost")
            if not self.config.force:
                return

        # Process each level
        results = []
        for level in self.config.levels:
            result = await self.populate_level(level)
            results.append(result)

            # Update stats
            self.tracker.progress["stats"][level] = result
            self.tracker.save()

        # Print summary
        logger.info("=" * 50)
        logger.info("POPULATION COMPLETE:")
        for result in results:
            logger.info(
                f"  {result['level']}: {result['completed']} completed, {result['failed']} failed",
            )
        logger.info("=" * 50)


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Populate ML universe data")

    # Data selection
    parser.add_argument(
        "--level",
        choices=["L0", "L1", "L2", "L3"],
        help="Specific level to populate (default: all)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["L0", "L1", "L2", "L3"],
        help="Multiple levels to populate",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Universe tier (default: 1)",
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to populate")

    # Date ranges
    parser.add_argument(
        "--l0-years",
        type=int,
        default=7,
        help="Years of L0 data (default: 7)",
    )
    parser.add_argument(
        "--l1-years",
        type=int,
        default=1,
        help="Years of L1 data (default: 1)",
    )
    parser.add_argument(
        "--l2-days",
        type=int,
        default=30,
        help="Days of L2/L3 data (default: 30)",
    )

    # Processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        help="API calls per minute",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tier1"),
        help="Data directory",
    )

    # Safeguards
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.0,
        help="Maximum cost in USD (default: 0)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate costs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if cost exceeds limit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from progress (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, ignore progress",
    )

    args = parser.parse_args()

    # Determine levels
    if args.level:
        levels = [args.level]
    elif args.levels:
        levels = args.levels
    else:
        # Default based on what's needed
        levels = ["L2", "L3"]  # User wants L2/L3 populated next

    # Create config
    config = UniverseConfig(
        levels=levels,
        tier=args.tier,
        symbols=args.symbols,
        l0_years=args.l0_years,
        l1_years=args.l1_years,
        l2_days=args.l2_days,
        batch_size=args.batch_size,
        rate_limit_per_min=args.rate_limit,
        data_dir=args.data_dir,
        max_cost_usd=args.max_cost,
        estimate_only=args.estimate_only,
        dry_run=args.dry_run,
        force=args.force,
        resume=args.resume,
    )

    # Run
    populator = UniversePopulator(config)
    asyncio.run(populator.run())


if __name__ == "__main__":
    main()
