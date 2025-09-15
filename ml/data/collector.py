#!/usr/bin/env python
"""
Data Collector for Nautilus Trader ML.

Maximizes Databento subscription value by collecting:
- L2 market depth (30 days) for all symbols
- L1 trades (multi-year) for key symbols
- TBBO quotes for spread dynamics
- Optimized for TFT teacher model training

Storage limit: 500GB

"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

from ml.config.base import DataCollectorConfig
from ml.data.ingest.policy import DatabentoCoveragePolicy


warnings.filterwarnings("ignore")

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class CategoryStats(TypedDict):
    count: int
    size_gb: float


class CollectorStats(TypedDict):
    l2_depth: CategoryStats
    l1_trades: CategoryStats
    tbbo_quotes: CategoryStats
    minute_bars: CategoryStats
    total_symbols: int
    total_size_gb: float


class DataCollector:
    """
    Collector for rich market microstructure data from Databento.

    Collects:
    1. L2 depth (mbp-1) - 30 days for all symbols
    2. L1 trades - 1-7 years for key symbols
    3. TBBO quotes - 30 days for spread dynamics
    4. Minute bars - 1 year for all symbols

    Storage budget: 500GB

    """

    def __init__(
        self,
        storage_limit_gb: float | None = None,
        data_dir: Path | None = None,
        *,
        config: DataCollectorConfig | None = None,
        end_date: datetime | None = None,
    ):
        """
        Initialize enhanced collector.

        Parameters
        ----------
        storage_limit_gb : float
            Maximum storage to use in GB

        """
        self.api_key = os.getenv("DATABENTO_API_KEY")
        # Degrade gracefully in environments without an API key (e.g., tests)
        if not self.api_key:
            self.client = None
        else:
            # Import Databento lazily to avoid asyncio loop creation at module import time
            import databento as db  # local import

            self.client = db.Historical(self.api_key)
        # Config-driven defaults with env overrides
        self._config = config or DataCollectorConfig()
        default_dir = Path(self._config.data_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else default_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Storage management
        self.storage_limit_gb = (
            float(storage_limit_gb)
            if storage_limit_gb is not None
            else float(self._config.storage_limit_gb)
        )
        self.storage_used_gb = 0.0

        # End date (last available data) – configurable; defaults to now
        if end_date is not None:
            self.end_date = end_date
        elif self._config.end_date_iso:
            try:
                self.end_date = datetime.fromisoformat(self._config.end_date_iso)
            except Exception:
                self.end_date = datetime.now()
        else:
            self.end_date = datetime.now()

        # Load existing symbols
        self.existing_symbols = self._load_existing_symbols()
        # Optional coverage guard (env-driven; no-ops if unset)
        self._policy = DatabentoCoveragePolicy.from_env()

        # Priority symbols for deep historical data
        self.PRIORITY_SYMBOLS = [
            # Core indices (essential)
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            # Mega caps (high liquidity)
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "META",
            "GOOGL",
            "TSLA",
            # Key sectors
            "XLF",
            "XLK",
            "XLE",
            "XLV",
            # Volatility
            "VXX",
            "UVXY",
            # Bonds/Commodities
            "TLT",
            "GLD",
        ]

        # Track collection stats with precise typing
        self.stats: CollectorStats = {
            "l2_depth": {"count": 0, "size_gb": 0.0},
            "l1_trades": {"count": 0, "size_gb": 0.0},
            "tbbo_quotes": {"count": 0, "size_gb": 0.0},
            "minute_bars": {"count": 0, "size_gb": 0.0},
            "total_symbols": 0,
            "total_size_gb": 0.0,
        }

    def _load_existing_symbols(self) -> list[str]:
        """
        Load list of symbols we already have basic data for.
        """
        symbols: list[str] = []
        # Check if data_dir exists first
        if not Path(self.data_dir).exists():
            return symbols

        for symbol_dir in Path(self.data_dir).iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.isupper():
                if list(symbol_dir.glob("*.parquet")):
                    symbols.append(symbol_dir.name)
        return sorted(symbols)

    def _get_current_storage_gb(self) -> float:
        """
        Get current storage usage in GB.
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024**3)

    def _estimate_data_size_gb(self, schema: str, symbols: list[str], days: int) -> float:
        """
        Estimate data size for collection.

        Parameters
        ----------
        schema : str
            Data schema (trades, mbp-1, tbbo, etc)
        symbols : List[str]
            List of symbols
        days : int
            Number of days

        Returns
        -------
        float
            Estimated size in GB

        """
        # Rough estimates based on schema and liquidity
        estimates_per_symbol_per_day_mb = {
            "trades": 50.0,  # ~50MB per day for liquid stocks
            "mbp-1": 200.0,  # ~200MB per day for L2 depth
            "tbbo": 30.0,  # ~30MB per day for quotes
            "ohlcv-1m": 0.5,  # ~0.5MB per day for minute bars
        }

        mb_per_day = estimates_per_symbol_per_day_mb.get(schema, 10.0)
        total_mb = mb_per_day * len(symbols) * days

        # Adjust for symbol liquidity
        if schema in ["trades", "mbp-1"]:
            # High liquidity symbols have more data
            if any(s in ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"] for s in symbols):
                total_mb *= 1.5

        return total_mb / 1024

    def collect_l2_depth(self, symbols: list[str] | None = None, days: int = 30) -> None:
        """
        Collect L2 market depth (mbp-1) data.

        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to collect. If None, uses all existing symbols.
        days : int
            Number of days to collect (max 30)

        """
        if self.client is None:
            print("Databento client not configured; skipping L2 depth collection.")
            return

        if symbols is None:
            symbols = self.existing_symbols

        print(f"\n{'='*80}")
        print("L2 MARKET DEPTH COLLECTION (MBP-1)")
        print(f"{'='*80}")
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {days} days")

        # Estimate size
        estimated_gb = self._estimate_data_size_gb("mbp-1", symbols, days)
        print(f"Estimated size: {estimated_gb:.1f} GB")

        if self.storage_used_gb + estimated_gb > self.storage_limit_gb:
            print("⚠️  Would exceed storage limit! Skipping.")
            return

        collected = 0
        total_size_gb = 0.0

        # Enforce symbol policy early
        symbols = self._policy.filter_symbols(symbols)
        for i, symbol in enumerate(symbols, 1):
            # Check storage before each symbol
            current_storage = self._get_current_storage_gb()
            if current_storage >= self.storage_limit_gb * 0.95:  # 95% full
                print(
                    f"\n⚠️  Storage limit approaching ({current_storage:.1f}/{self.storage_limit_gb} GB)",
                )
                break

            print(f"\n[{i}/{len(symbols)}] {symbol}:")

            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            # Skip if already collected
            depth_file = symbol_dir / f"l2_depth_{days}d.parquet"
            if depth_file.exists():
                print("  ✓ Already collected")
                continue

            try:
                # Collect L2 depth
                start_date = self.end_date - timedelta(days=days)
                # Clamp by policy
                start_date, _end = self._policy.clamp_range(
                    start=start_date,
                    end=self.end_date,
                    dataset="EQUS.MINI",
                    schema="mbp-1",
                )

                print("  Collecting L2 depth (mbp-1)...")
                data = self.client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_date,
                    end=_end,
                    schema="mbp-1",
                    limit=1000000,  # 1M rows max per symbol
                )

                df = data.to_df()
                if not df.empty:
                    # Save to parquet
                    df.to_parquet(depth_file)

                    size_gb = depth_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    collected += 1

                    print(f"  ✓ L2 depth: {len(df):,} rows, {size_gb:.3f} GB")

                    # Calculate key microstructure features
                    if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
                        spread = (df["ask_px_00"] - df["bid_px_00"]).mean()
                        print(f"    Avg spread: ${spread:.4f}")
                else:
                    print("  ✗ No L2 data available")

            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")

            # Rate limit
            time.sleep(0.5)

            # Progress update every 10 symbols
            if i % 10 == 0:
                print(f"\nProgress: {collected} collected, {total_size_gb:.2f} GB")

        # Update stats
        self.stats["l2_depth"]["count"] = collected
        self.stats["l2_depth"]["size_gb"] = total_size_gb
        self.storage_used_gb += total_size_gb

        print(f"\n{'='*60}")
        print(f"L2 Depth Complete: {collected} symbols, {total_size_gb:.2f} GB")

    def collect_l1_trades(self, symbols: list[str] | None = None, years: int = 1) -> None:
        """
        Collect L1 trade data (multi-year).

        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to collect. If None, uses priority symbols.
        years : int
            Number of years to collect (1-7)

        """
        if self.client is None:
            print("Databento client not configured; skipping L1 trades collection.")
            return

        if symbols is None:
            symbols = self.PRIORITY_SYMBOLS
        # Enforce symbol policy early
        symbols = self._policy.filter_symbols(symbols)

        print(f"\n{'='*80}")
        print("L1 TRADES COLLECTION")
        print(f"{'='*80}")
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {years} year(s)")

        # Estimate size
        estimated_gb = self._estimate_data_size_gb("trades", symbols, years * 365)
        print(f"Estimated size: {estimated_gb:.1f} GB")

        if self.storage_used_gb + estimated_gb > self.storage_limit_gb:
            print("⚠️  Would exceed storage limit! Reducing scope...")
            # Reduce to top 10 symbols or 1 year
            if years > 1:
                years = 1
            elif len(symbols) > 10:
                symbols = symbols[:10]
            estimated_gb = self._estimate_data_size_gb("trades", symbols, years * 365)
            print(f"Adjusted: {len(symbols)} symbols, {years} year(s), ~{estimated_gb:.1f} GB")

        collected = 0
        total_size_gb = 0.0

        for i, symbol in enumerate(symbols, 1):
            # Check storage
            current_storage = self._get_current_storage_gb()
            if current_storage >= self.storage_limit_gb * 0.95:
                print("\n⚠️  Storage limit reached")
                break

            print(f"\n[{i}/{len(symbols)}] {symbol}:")

            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            # Collect year by year to manage size
            for year_offset in range(years):
                year_end = self.end_date - timedelta(days=365 * year_offset)
                year_start = year_end - timedelta(days=365)
                # Clamp by policy
                year_start, year_end = self._policy.clamp_range(
                    start=year_start,
                    end=year_end,
                    dataset="EQUS.MINI",
                    schema="trades",
                )
                year_label = year_end.year

                trades_file = symbol_dir / f"trades_{year_label}.parquet"
                if trades_file.exists():
                    print(f"  ✓ {year_label} already collected")
                    continue

                try:
                    print(f"  Collecting trades for {year_label}...")
                    data = self.client.timeseries.get_range(
                        dataset="EQUS.MINI",
                        symbols=[symbol],
                        start=year_start,
                        end=year_end,
                        schema="trades",
                        limit=10000000,  # 10M rows max per year
                    )

                    df = data.to_df()
                    if not df.empty:
                        # Save to parquet
                        df.to_parquet(trades_file)

                        size_gb = trades_file.stat().st_size / (1024**3)
                        total_size_gb += size_gb

                        # Calculate trade statistics
                        avg_price = df["price"].mean() if "price" in df.columns else 0
                        total_volume = df["size"].sum() if "size" in df.columns else 0

                        print(f"  ✓ {year_label}: {len(df):,} trades, {size_gb:.3f} GB")
                        print(f"    Avg price: ${avg_price:.2f}, Volume: {total_volume:,}")

                        collected += 1
                    else:
                        print(f"  ✗ No trades for {year_label}")

                except Exception as e:
                    print(f"  ✗ Error for {year_label}: {str(e)[:100]}")

                # Rate limit
                time.sleep(1.0)

        # Update stats
        self.stats["l1_trades"]["count"] = collected
        self.stats["l1_trades"]["size_gb"] = total_size_gb
        self.storage_used_gb += total_size_gb

        print(f"\n{'='*60}")
        print(f"L1 Trades Complete: {collected} symbol-years, {total_size_gb:.2f} GB")

    def collect_tbbo_quotes(self, symbols: list[str] | None = None, days: int = 30) -> None:
        """
        Collect top-of-book quotes (TBBO).

        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to collect
        days : int
            Number of days

        """
        if self.client is None:
            print("Databento client not configured; skipping TBBO quotes collection.")
            return

        if symbols is None:
            symbols = self.existing_symbols
        # Enforce symbol policy early
        symbols = self._policy.filter_symbols(symbols)

        print(f"\n{'='*80}")
        print("TBBO QUOTES COLLECTION")
        print(f"{'='*80}")
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {days} days")

        collected = 0
        total_size_gb = 0.0

        for i, symbol in enumerate(symbols, 1):
            # Check storage
            if self._get_current_storage_gb() >= self.storage_limit_gb * 0.95:
                print("\n⚠️  Storage limit reached")
                break

            print(f"\n[{i}/{len(symbols)}] {symbol}:")

            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            quotes_file = symbol_dir / f"tbbo_{days}d.parquet"
            if quotes_file.exists():
                print("  ✓ Already collected")
                continue

            try:
                start_date = self.end_date - timedelta(days=days)
                start_date, _end = self._policy.clamp_range(
                    start=start_date,
                    end=self.end_date,
                    dataset="EQUS.MINI",
                    schema="tbbo",
                )

                print("  Collecting TBBO quotes...")
                data = self.client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_date,
                    end=_end,
                    schema="tbbo",
                    limit=1000000,
                )

                df = data.to_df()
                if not df.empty:
                    df.to_parquet(quotes_file)

                    size_gb = quotes_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    collected += 1

                    # Calculate spread statistics
                    if "ask_px" in df.columns and "bid_px" in df.columns:
                        spreads = df["ask_px"] - df["bid_px"]
                        avg_spread = spreads.mean()
                        print(f"  ✓ TBBO: {len(df):,} quotes, {size_gb:.3f} GB")
                        print(f"    Avg spread: ${avg_spread:.4f}")
                else:
                    print("  ✗ No quotes available")

            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")

            time.sleep(0.5)

        # Update stats
        self.stats["tbbo_quotes"]["count"] = collected
        self.stats["tbbo_quotes"]["size_gb"] = total_size_gb
        self.storage_used_gb += total_size_gb

        print(f"\n{'='*60}")
        print(f"TBBO Complete: {collected} symbols, {total_size_gb:.2f} GB")

    def collect_minute_bars(self, symbols: list[str] | None = None, days: int = 365) -> None:
        """
        Collect minute-level OHLCV bars.

        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to collect
        days : int
            Number of days (max 365)

        """
        if self.client is None:
            print("Databento client not configured; skipping minute bars collection.")
            return

        if symbols is None:
            symbols = self.existing_symbols
        # Enforce symbol policy early
        symbols = self._policy.filter_symbols(symbols)

        print(f"\n{'='*80}")
        print("MINUTE BARS COLLECTION")
        print(f"{'='*80}")
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {days} days")

        collected = 0
        total_size_gb = 0.0

        for i, symbol in enumerate(symbols, 1):
            if self._get_current_storage_gb() >= self.storage_limit_gb * 0.98:
                print("\n⚠️  Storage limit reached")
                break

            print(f"[{i}/{len(symbols)}] {symbol}: ", end="")

            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            bars_file = symbol_dir / f"bars_1m_{days}d.parquet"
            if bars_file.exists():
                print("✓ exists")
                continue

            try:
                start_date = self.end_date - timedelta(days=days)
                start_date, _end = self._policy.clamp_range(
                    start=start_date,
                    end=self.end_date,
                    dataset="EQUS.MINI",
                    schema="ohlcv-1m",
                )

                data = self.client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                    start=start_date,
                    end=_end,
                    schema="ohlcv-1m",
                    limit=500000,
                )

                df = data.to_df()
                if not df.empty:
                    df.to_parquet(bars_file)
                    size_mb = bars_file.stat().st_size / (1024**2)
                    total_size_gb += size_mb / 1024
                    collected += 1
                    print(f"✓ {len(df):,} bars, {size_mb:.1f} MB")
                else:
                    print("✗ no data")

            except Exception:
                print("✗ error")

            time.sleep(0.2)

        # Update stats
        self.stats["minute_bars"]["count"] = collected
        self.stats["minute_bars"]["size_gb"] = total_size_gb
        self.storage_used_gb += total_size_gb

        print(f"\n{'='*60}")
        print(f"Minute bars: {collected} symbols, {total_size_gb:.2f} GB")

    def run_collection(self) -> None:
        """
        Run the complete enhanced collection pipeline.

        Priority order (within 1TB limit):
        1. L2 depth for ALL 106 symbols (30 days) - ~20-40GB
        2. L1 trades for top 30 symbols (2 years) - ~300-400GB
        3. TBBO quotes for ALL symbols (30 days) - ~20-30GB
        4. Minute bars for ALL symbols (1 year) - ~10-20GB
        5. Extended L1 trades for more symbols if space permits

        """
        print(f"\n{'='*80}")
        print("ENHANCED DATA COLLECTION PIPELINE")
        print(f"{'='*80}")
        print(f"Storage limit: {self.storage_limit_gb} GB")
        print(f"Symbols available: {len(self.existing_symbols)}")
        print(f"Priority symbols: {len(self.PRIORITY_SYMBOLS)}")
        print(f"{'='*80}\n")

        # Phase 1: L2 Market Depth (highest priority for microstructure)
        print("\n📊 PHASE 1: L2 MARKET DEPTH")
        print("-" * 40)
        # L2 depth for TOP 50 most liquid symbols only (quality over quantity)
        top_liquid = self.PRIORITY_SYMBOLS + [
            s for s in self.existing_symbols if s not in self.PRIORITY_SYMBOLS
        ]
        self.collect_l2_depth(symbols=top_liquid[:50], days=30)

        # Check storage
        current_gb = self._get_current_storage_gb()
        print(f"\nStorage used: {current_gb:.1f}/{self.storage_limit_gb} GB")

        # Phase 2: L1 Trades (critical for regime learning)
        if current_gb < self.storage_limit_gb * 0.5:  # If under 50% capacity
            print("\n📈 PHASE 2: L1 TRADES (HISTORICAL)")
            print("-" * 40)
            # Collect 2 years for top symbols
            self.collect_l1_trades(symbols=self.PRIORITY_SYMBOLS[:20], years=2)

        # Check storage
        current_gb = self._get_current_storage_gb()
        print(f"\nStorage used: {current_gb:.1f}/{self.storage_limit_gb} GB")

        # Phase 3: TBBO Quotes (spread dynamics)
        if current_gb < self.storage_limit_gb * 0.7:  # If under 70% capacity
            print("\n💹 PHASE 3: TBBO QUOTES")
            print("-" * 40)
            # TBBO for top 75 symbols (good coverage, manageable size)
            self.collect_tbbo_quotes(symbols=top_liquid[:75], days=30)

        # Check storage
        current_gb = self._get_current_storage_gb()
        print(f"\nStorage used: {current_gb:.1f}/{self.storage_limit_gb} GB")

        # Phase 4: Minute bars (fill remaining space)
        if current_gb < self.storage_limit_gb * 0.85:  # If under 85% capacity
            print("\n⏱️ PHASE 4: MINUTE BARS")
            print("-" * 40)
            self.collect_minute_bars(symbols=self.existing_symbols, days=365)

        # Phase 5: Extended L1 trades for more symbols
        current_gb = self._get_current_storage_gb()
        if current_gb < self.storage_limit_gb * 0.9:  # If under 90% capacity
            print("\n📊 PHASE 5: EXTENDED L1 TRADES")
            print("-" * 40)
            # Get next 30 most liquid symbols
            extended_symbols = [
                "HD",
                "PFE",
                "CVX",
                "MRK",
                "ABBV",
                "DIS",
                "PEP",
                "KO",
                "NKE",
                "MCD",
                "TMO",
                "LLY",
                "CAT",
                "BA",
                "HON",
                "UNP",
                "AMD",
                "INTC",
                "QCOM",
                "CRM",
                "ADBE",
                "NFLX",
                "AVGO",
                "BAC",
                "WFC",
                "GS",
                "MS",
                "C",
                "BLK",
                "SCHW",
            ][:20]
            self.collect_l1_trades(symbols=extended_symbols, years=1)

        # Final summary
        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """
        Print comprehensive collection summary.
        """
        current_gb = self._get_current_storage_gb()

        print(f"\n{'='*80}")
        print("ENHANCED COLLECTION COMPLETE")
        print(f"{'='*80}")
        print("\nData Collected:")
        print(
            f"  • L2 Depth (mbp-1):  {self.stats['l2_depth']['count']} symbols, {self.stats['l2_depth']['size_gb']:.2f} GB",
        )
        print(
            f"  • L1 Trades:         {self.stats['l1_trades']['count']} symbol-years, {self.stats['l1_trades']['size_gb']:.2f} GB",
        )
        print(
            f"  • TBBO Quotes:       {self.stats['tbbo_quotes']['count']} symbols, {self.stats['tbbo_quotes']['size_gb']:.2f} GB",
        )
        print(
            f"  • Minute Bars:       {self.stats['minute_bars']['count']} symbols, {self.stats['minute_bars']['size_gb']:.2f} GB",
        )
        print(
            f"\nTotal Storage Used: {current_gb:.2f}/{self.storage_limit_gb} GB ({current_gb/self.storage_limit_gb*100:.1f}%)",
        )

        # Save metadata
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "storage_limit_gb": self.storage_limit_gb,
            "storage_used_gb": current_gb,
            "stats": self.stats,
            "symbols": self.existing_symbols,
            "priority_symbols": self.PRIORITY_SYMBOLS,
        }

        metadata_file = self.data_dir / "collection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {metadata_file}")

        print("\n✅ Data ready for TFT teacher model training!")
        print("✅ Rich microstructure features available")
        print("✅ Multi-year market regimes captured")


def main() -> None:
    """
    Run enhanced data collection entry point.
    """
    print("ENHANCED DATA COLLECTION (1TB)")
    print("=" * 50)
    print("\nOptimized collection strategy:")
    print("• L2 depth: Top 50 most liquid symbols (30 days)")
    print("• L1 trades: Top 20 symbols (2 years history)")
    print("• TBBO quotes: Top 75 symbols (30 days)")
    print("• Minute bars: All 106 symbols (1 year)")
    print("• Extended L1: Next 20 symbols (1 year)")
    print("\nEstimated data breakdown:")
    print("• ~100GB - L2 market depth")
    print("• ~400GB - L1 trades (multi-year)")
    print("• ~50GB - TBBO quotes")
    print("• ~20GB - Minute bars")
    print("• ~200GB - Extended L1 trades")
    print("\nTotal: ~750-850GB of 1000GB limit")
    print("Estimated time: 3-5 hours")
    print()

    # Auto-proceed in background mode
    import sys

    if not sys.stdin.isatty():
        print("Auto-proceeding in background mode...")
    else:
        response = input("Proceed? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled")
            return

    collector = DataCollector(storage_limit_gb=1000.0)
    collector.run_collection()

    print("\n🎉 Enhanced collection complete!")
    print("Your TFT model now has access to rich microstructure data!")


if __name__ == "__main__":
    main()
