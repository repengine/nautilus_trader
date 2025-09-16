#!/usr/bin/env python3
"""
Build production-grade dataset with 20M+ samples.
This script orchestrates the complete data collection pipeline:
1. Fetches 7 years of L0/L1 data for core symbols
2. Collects rolling L2/L3 microstructure data
3. Computes cross-sectional features
4. Adds market regime indicators
5. Creates TFT-ready training dataset

Usage:
    python ml/cli/build_production_dataset.py --phase 1  # Historical L0/L1
    python ml/cli/build_production_dataset.py --phase 2  # Microstructure
    python ml/cli/build_production_dataset.py --phase 3  # Cross-sectional
    python ml/cli/build_production_dataset.py --full     # Complete pipeline

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import cast


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from ml.data.collectors.production_collector import ProductionDataCollector

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


configure_logging()
_run_id: str = f"cli_build_production_dataset_{_uuid.uuid4().hex[:8]}"
bind_log_context(run_id=_run_id, component="ml.cli.build_production_dataset")
logger = logging.getLogger(__name__)


def phase1_historical(collector: ProductionDataCollector) -> dict[str, pl.DataFrame]:
    """
    Phase 1: Collect 7 years of L0/L1 historical data.

    Target: ~20M samples from 30 core symbols
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Historical L0/L1 Data Collection")
    logger.info("=" * 60)

    # Priority symbols for initial collection
    priority_symbols = [
        # Core indices (highest priority)
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        # Mega-cap tech
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        # Additional liquidity
        "TSLA",
        "AMD",
        "GOOGL",
    ]

    # Collect 7 years of minute bars + quotes
    results_any = collector.collect_historical_l0_l1(
        symbols=priority_symbols,
        start_date="2018-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        batch_size=3,  # Process 3 symbols at a time
    )
    results: dict[str, pl.DataFrame] = cast(dict[str, pl.DataFrame], results_any)

    # Summary statistics
    total_samples = sum(len(df) for df in results.values())
    logger.info("\nPhase 1 Complete:")
    logger.info(f"  Symbols collected: {len(results)}")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Average per symbol: {total_samples // len(results):,}")

    # Save checkpoint
    checkpoint_dir = collector.data_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for symbol, df in results.items():
        path = checkpoint_dir / f"{symbol}_l0_l1_7y.parquet"
        df.write_parquet(path)

    logger.info(f"  Checkpoint saved to: {checkpoint_dir}")

    return results


def phase2_microstructure(collector: ProductionDataCollector) -> dict[str, pl.DataFrame]:
    """
    Phase 2: Collect L2/L3 microstructure data.

    Target: 30-day rolling window for top 10 symbols
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: L2/L3 Microstructure Collection")
    logger.info("=" * 60)

    # Top symbols by liquidity
    top_symbols = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

    results_any = collector.collect_microstructure_l2_l3(
        symbols=top_symbols,
        lookback_days=30,
        depth_levels=10,
    )
    results: dict[str, pl.DataFrame] = cast(dict[str, pl.DataFrame], results_any)

    # Summary
    total_samples = sum(len(df) for df in results.values())
    logger.info("\nPhase 2 Complete:")
    logger.info(f"  Symbols with microstructure: {len(results)}")
    logger.info(f"  Total microstructure samples: {total_samples:,}")

    return results


def phase3_cross_sectional(
    collector: ProductionDataCollector,
    universe_data: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Phase 3: Compute cross-sectional features.

    Target: Relative metrics across universe
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Cross-Sectional Feature Engineering")
    logger.info("=" * 60)

    # Compute cross-sectional features
    cross_features_any = collector.compute_cross_sectional_features(
        universe_data=universe_data,
        window_sizes=[20, 60, 120, 390],  # 20min, 1hr, 2hr, 1day
    )
    cross_features = cast(pl.DataFrame, cross_features_any)

    logger.info("\nPhase 3 Complete:")
    logger.info(f"  Cross-sectional features: {len(cross_features.columns)}")
    logger.info(f"  Total rows: {len(cross_features):,}")

    # Save
    path = collector.data_dir / "features" / "cross_sectional.parquet"
    cross_features.write_parquet(path)
    logger.info(f"  Saved to: {path}")

    return cross_features


def phase4_regime_indicators(collector: ProductionDataCollector) -> pl.DataFrame:
    """
    Phase 4: Collect market regime indicators.

    Target: VIX, breadth, sentiment indicators
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Market Regime Indicators")
    logger.info("=" * 60)

    regime_any = collector.collect_regime_indicators()
    regime = cast(pl.DataFrame, regime_any)

    logger.info("\nPhase 4 Complete:")
    logger.info(f"  Regime indicators: {len(regime.columns)}")
    ts_min = str(regime["timestamp"].min())
    ts_max = str(regime["timestamp"].max())
    logger.info(f"  Time range: {ts_min} to {ts_max}")

    return regime


def build_tft_dataset(data_dir: Path) -> pl.DataFrame:
    """
    Phase 5: Build TFT-ready training dataset.
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: TFT Dataset Construction")
    logger.info("=" * 60)

    # Load collected data
    checkpoint_dir = data_dir / "checkpoints"

    # Get available symbols
    available_files = list(checkpoint_dir.glob("*_l0_l1_7y.parquet"))
    symbols = [f.stem.split("_")[0] for f in available_files]

    logger.info(f"Found {len(symbols)} symbols with historical data")

    # Initialize TFT builder
    catalog = ParquetDataCatalog(path=str(checkpoint_dir))
    builder = TFTDatasetBuilder(catalog=catalog, symbols=symbols[:10])

    # Build dataset
    df = builder.build_training_dataset(
        horizon_minutes=15,
        min_return_threshold=0.002,
        lookback_periods=50,
        use_polars=True,
    )

    logger.info("\nTFT Dataset Complete:")
    # Ensure type for subsequent operations
    if not isinstance(df, pl.DataFrame):
        raise TypeError("Expected Polars DataFrame from builder when use_polars=True")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.estimated_size() / 1e9:.2f} GB")

    # Save
    output_path = data_dir / "production" / "tft_dataset_20m.parquet"
    output_path.parent.mkdir(exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"  Saved to: {output_path}")

    return df


def estimate_requirements(collector: ProductionDataCollector) -> None:
    """
    Estimate storage and compute requirements.
    """
    logger.info("=" * 60)
    logger.info("RESOURCE REQUIREMENTS ESTIMATION")
    logger.info("=" * 60)

    symbols = collector.get_all_symbols()
    n_symbols = len(symbols)

    # Storage estimates
    estimates = {
        "L0_OHLCV": {
            "rows_per_symbol": 7 * 252 * 390,  # 7 years × trading days × minutes
            "bytes_per_row": 48,  # timestamp(8) + OHLCV(5×8)
            "total_gb": (n_symbols * 7 * 252 * 390 * 48) / 1e9,
        },
        "L1_Quotes": {
            "rows_per_symbol": 7 * 252 * 390,
            "bytes_per_row": 40,  # timestamp + bid/ask/sizes
            "total_gb": (n_symbols * 7 * 252 * 390 * 40) / 1e9,
        },
        "L2_Depth": {
            "rows_per_symbol": 30 * 390 * 60,  # 30 days × minutes × snapshots
            "bytes_per_row": 200,  # 10 levels × bid/ask × price/size
            "total_gb": (10 * 30 * 390 * 60 * 200) / 1e9,
        },
        "Features": {
            "rows_total": n_symbols * 7 * 252 * 390,
            "features": 200,  # Estimated feature count
            "bytes_per_feature": 4,  # float32
            "total_gb": (n_symbols * 7 * 252 * 390 * 200 * 4) / 1e9,
        },
    }

    total_storage = sum(est["total_gb"] for est in estimates.values())

    logger.info("\nStorage Requirements:")
    logger.info(f"  Symbols: {n_symbols}")
    logger.info(f"  L0 OHLCV: {estimates['L0_OHLCV']['total_gb']:.1f} GB")
    logger.info(f"  L1 Quotes: {estimates['L1_Quotes']['total_gb']:.1f} GB")
    logger.info(f"  L2 Depth: {estimates['L2_Depth']['total_gb']:.1f} GB")
    logger.info(f"  Features: {estimates['Features']['total_gb']:.1f} GB")
    logger.info(f"  Total Raw: {total_storage:.1f} GB")
    logger.info(f"  With Compression (50%): {total_storage * 0.5:.1f} GB")

    # Compute estimates
    logger.info("\nCompute Requirements:")
    logger.info(f"  Initial download: ~{total_storage * 0.1:.1f} GB from Databento")
    logger.info(f"  Feature computation: {n_symbols * 7 * 252:.0f} symbol-days")
    logger.info("  RAM needed: 32-64 GB for processing")
    logger.info("  CPU cores: 8-16 for parallel processing")

    # Time estimates
    logger.info("\nTime Estimates:")
    logger.info("  Data download: 2-4 hours (depends on connection)")
    logger.info("  Feature engineering: 4-8 hours")
    logger.info("  Cross-sectional: 1-2 hours")
    logger.info("  Total pipeline: 8-16 hours")

    # Cost estimates (Databento)
    logger.info("\nDatabento Cost Estimates:")
    logger.info(f"  L0 Historical: ~${n_symbols * 0.10:.0f} (assuming $0.10/symbol/year)")
    logger.info(f"  L1 Quotes: ~${n_symbols * 0.50:.0f}")
    logger.info("  L2 Depth (10 symbols): ~$50")
    logger.info(f"  Total: ~${n_symbols * 0.60 + 50:.0f}")


def main() -> int:
    """
    Execute main.
    """
    parser = argparse.ArgumentParser(description="Build production ML dataset")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific phase",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete pipeline",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate requirements only",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/nate/projects/nautilus_trader/data",
        help="Data directory",
    )

    args = parser.parse_args()

    # Initialize collector
    api_key = os.getenv("DATABENTO_API_KEY")
    collector = ProductionDataCollector(
        data_dir=args.data_dir,
        databento_api_key=api_key,
    )

    # Estimate only
    if args.estimate:
        estimate_requirements(collector)
        return 0

    # Run specific phase or full pipeline
    if args.phase == 1 or args.full:
        universe_data = phase1_historical(collector)

    if args.phase == 2 or args.full:
        _microstructure = phase2_microstructure(collector)

    if args.phase == 3 or args.full:
        # Load universe data if not already loaded
        if not args.full:
            # Load from checkpoint
            checkpoint_dir = Path(args.data_dir) / "checkpoints"
            universe_data = {}
            for f in checkpoint_dir.glob("*_l0_l1_7y.parquet"):
                symbol = f.stem.split("_")[0]
                universe_data[symbol] = pl.read_parquet(f)

        _cross_features = phase3_cross_sectional(collector, universe_data)

    if args.phase == 4 or args.full:
        _regime = phase4_regime_indicators(collector)

    if args.phase == 5 or args.full:
        _tft_dataset = build_tft_dataset(Path(args.data_dir))

    if args.full:
        logger.info("=" * 60)
        logger.info("COMPLETE PIPELINE FINISHED")
        logger.info("=" * 60)
        logger.info("Production dataset ready for TFT training!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
