"""
Production dataset build tasks for large-scale pipelines.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from ml.data.collectors.production_collector import ProductionDataCollector
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ProductionDatasetConfig:
    """
    Configuration for building the production dataset.
    """

    data_dir: Path
    phase: int | None = None
    run_full: bool = False
    estimate_only: bool = False
    databento_api_key: str | None = None


def _phase1_historical(collector: ProductionDataCollector) -> dict[str, pl.DataFrame]:
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 1: Historical L0/L1 Data Collection")
    LOGGER.info("=" * 60)

    priority_symbols = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "TSLA",
        "AMD",
        "GOOGL",
    ]

    results_any = collector.collect_historical_l0_l1(
        symbols=priority_symbols,
        start_date="2018-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        batch_size=3,
    )
    results = cast(dict[str, pl.DataFrame], results_any)

    total_samples = sum(len(df) for df in results.values())
    if results:
        LOGGER.info("\nPhase 1 Complete:")
        LOGGER.info(
            "Phase 1 Complete: symbols=%d total_samples=%d average=%d",
            len(results),
            total_samples,
            total_samples // max(len(results), 1),
        )

    checkpoint_dir = collector.data_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for symbol, df in results.items():
        path = checkpoint_dir / f"{symbol}_l0_l1_7y.parquet"
        df.write_parquet(path)

    LOGGER.info("Checkpoint saved to %s", checkpoint_dir)
    return results


def _phase2_microstructure(collector: ProductionDataCollector) -> dict[str, pl.DataFrame]:
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 2: L2/L3 Microstructure Collection")
    LOGGER.info("=" * 60)

    top_symbols = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

    results_any = collector.collect_microstructure_l2_l3(
        symbols=top_symbols,
        lookback_days=30,
        depth_levels=10,
    )
    results = cast(dict[str, pl.DataFrame], results_any)

    total_samples = sum(len(df) for df in results.values())
    LOGGER.info("Phase 2 Complete: symbols=%d total_samples=%d", len(results), total_samples)
    return results


def _phase3_cross_sectional(
    collector: ProductionDataCollector,
    universe_data: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 3: Cross-Sectional Feature Engineering")
    LOGGER.info("=" * 60)

    cross_features_any = collector.compute_cross_sectional_features(
        universe_data=universe_data,
        window_sizes=[20, 60, 120, 390],
    )
    cross_features = cast(pl.DataFrame, cross_features_any)

    LOGGER.info(
        "Phase 3 Complete: features=%d rows=%d",
        len(cross_features.columns),
        len(cross_features),
    )

    path = collector.data_dir / "features" / "cross_sectional.parquet"
    path.parent.mkdir(exist_ok=True)
    cross_features.write_parquet(path)
    LOGGER.info("Cross-sectional features saved to %s", path)
    return cross_features


def _phase4_regime_indicators(collector: ProductionDataCollector) -> pl.DataFrame:
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 4: Market Regime Indicators")
    LOGGER.info("=" * 60)

    regime_any = collector.collect_regime_indicators()
    regime = cast(pl.DataFrame, regime_any)

    LOGGER.info(
        "Phase 4 Complete: columns=%d range=%s..%s",
        len(regime.columns),
        regime["timestamp"].min(),
        regime["timestamp"].max(),
    )
    return regime


def _phase5_build_tft_dataset(data_dir: Path) -> pl.DataFrame:
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 5: TFT Dataset Construction")
    LOGGER.info("=" * 60)

    checkpoint_dir = data_dir / "checkpoints"
    available_files = list(checkpoint_dir.glob("*_l0_l1_7y.parquet"))
    symbols = [f.stem.split("_")[0] for f in available_files]

    LOGGER.info("Historical symbols discovered: %d", len(symbols))

    catalog = ParquetDataCatalog(path=str(checkpoint_dir))
    builder = TFTDatasetBuilder(catalog=catalog, symbols=symbols[:10])

    df = builder.build_training_dataset(
        horizon_minutes=15,
        min_return_threshold=0.002,
        lookback_periods=50,
        use_polars=True,
    )

    if not isinstance(df, pl.DataFrame):  # Defensive
        raise TypeError("Expected Polars DataFrame from builder when use_polars=True")

    LOGGER.info(
        "TFT dataset built: rows=%d columns=%d size_gb=%.2f",
        df.height,
        len(df.columns),
        df.estimated_size() / 1e9,
    )

    output_path = data_dir / "production" / "tft_dataset_20m.parquet"
    output_path.parent.mkdir(exist_ok=True)
    df.write_parquet(output_path)
    LOGGER.info("TFT dataset saved to %s", output_path)
    return df


def _estimate_requirements(collector: ProductionDataCollector) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("RESOURCE REQUIREMENTS ESTIMATION")
    LOGGER.info("=" * 60)

    symbols = collector.get_all_symbols()
    n_symbols = len(symbols)

    estimates = {
        "L0_OHLCV": {
            "rows_per_symbol": 7 * 252 * 390,
            "bytes_per_row": 48,
            "total_gb": (n_symbols * 7 * 252 * 390 * 48) / 1e9,
        },
        "L1_Quotes": {
            "rows_per_symbol": 7 * 252 * 390,
            "bytes_per_row": 40,
            "total_gb": (n_symbols * 7 * 252 * 390 * 40) / 1e9,
        },
        "L2_Depth": {
            "rows_per_symbol": 30 * 390 * 60,
            "bytes_per_row": 200,
            "total_gb": (10 * 30 * 390 * 60 * 200) / 1e9,
        },
        "Features": {
            "rows_total": n_symbols * 7 * 252 * 390,
            "features": 200,
            "bytes_per_feature": 4,
            "total_gb": (n_symbols * 7 * 252 * 390 * 200 * 4) / 1e9,
        },
    }

    total_storage = sum(est["total_gb"] for est in estimates.values())

    LOGGER.info("Storage requirements: symbols=%d total_gb=%.2f", n_symbols, total_storage)
    LOGGER.info("Compute requirements: symbol_days=%d", n_symbols * 7 * 252)
    LOGGER.info("Databento cost estimate: usd=%.2f", n_symbols * 0.60 + 50)


def build_production_dataset(config: ProductionDatasetConfig) -> None:
    """
    Execute the production dataset pipeline according to ``config``.
    """
    api_key = config.databento_api_key or os.getenv("DATABENTO_API_KEY")
    collector = ProductionDataCollector(
        data_dir=str(config.data_dir),
        databento_api_key=api_key,
    )

    if config.estimate_only:
        _estimate_requirements(collector)
        return

    selected_phase = config.phase
    run_full = config.run_full or selected_phase is None

    universe_data: dict[str, pl.DataFrame] | None = None

    if run_full or selected_phase == 1:
        universe_data = _phase1_historical(collector)

    if run_full or selected_phase == 2:
        _phase2_microstructure(collector)

    if run_full or selected_phase == 3:
        if universe_data is None:
            checkpoint_dir = config.data_dir / "checkpoints"
            universe_data = {
                path.stem.split("_")[0]: pl.read_parquet(path)
                for path in checkpoint_dir.glob("*_l0_l1_7y.parquet")
            }
        _phase3_cross_sectional(collector, universe_data)

    if run_full or selected_phase == 4:
        _phase4_regime_indicators(collector)

    if run_full or selected_phase == 5:
        _phase5_build_tft_dataset(config.data_dir)

    if run_full:
        LOGGER.info("=" * 60)
        LOGGER.info("COMPLETE PIPELINE FINISHED")
        LOGGER.info("=" * 60)
        LOGGER.info("Production dataset ready for TFT training!")
