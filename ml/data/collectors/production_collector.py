"""
Production dataset collection and assembly helpers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path

import polars as pl

from ml.config.targets import BinaryTargetConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import decimal_to_bps
from ml.data.collector import DataCollector
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ProductionDatasetConfig:
    """
    Configuration for production dataset builds.

    Attributes:
        data_dir: Root directory for dataset artifacts and checkpoints.
        phase: Optional pipeline phase selector.
        run_full: Whether to run all phases.
        estimate_only: Whether to emit requirement estimates only.
        databento_api_key: Optional Databento API key override.
    """

    data_dir: Path
    phase: int | None = None
    run_full: bool = False
    estimate_only: bool = False
    databento_api_key: str | None = None


class ProductionDataCollector:
    """
    Adapter that provides production-oriented collection primitives.
    """

    def __init__(self, data_dir: str, databento_api_key: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if databento_api_key:
            os.environ["DATABENTO_API_KEY"] = databento_api_key
        self._collector = DataCollector(data_dir=self.data_dir)

    @staticmethod
    def _parse_iso_date(value: str, *, fallback: datetime) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return fallback

    def _load_symbol_artifacts(
        self,
        *,
        symbols: list[str],
        pattern: str,
    ) -> dict[str, pl.DataFrame]:
        frames_by_symbol: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            symbol_dir = self.data_dir / symbol
            if not symbol_dir.exists():
                continue
            frames: list[pl.DataFrame] = []
            for artifact in sorted(symbol_dir.glob(pattern)):
                try:
                    frame = pl.read_parquet(artifact)
                except Exception:
                    LOGGER.debug(
                        "Failed reading symbol artifact symbol=%s path=%s",
                        symbol,
                        str(artifact),
                        exc_info=True,
                    )
                    continue
                if not frame.is_empty():
                    frames.append(frame)
            if frames:
                frames_by_symbol[symbol] = (
                    frames[0]
                    if len(frames) == 1
                    else pl.concat(frames, how="vertical_relaxed")
                )
        return frames_by_symbol

    def _load_phase1_checkpoints(self, symbols: list[str]) -> dict[str, pl.DataFrame]:
        checkpoint_dir = self.data_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return {}
        frames_by_symbol: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            checkpoint_path = checkpoint_dir / f"{symbol}_l0_l1_7y.parquet"
            if not checkpoint_path.exists():
                continue
            try:
                frame = pl.read_parquet(checkpoint_path)
            except Exception:
                LOGGER.debug(
                    "Failed reading phase1 checkpoint symbol=%s path=%s",
                    symbol,
                    str(checkpoint_path),
                    exc_info=True,
                )
                continue
            if not frame.is_empty():
                frames_by_symbol[symbol] = frame
        return frames_by_symbol

    @staticmethod
    def _resolve_timestamp_column(columns: list[str]) -> str | None:
        for candidate in ("timestamp", "ts_event", "ts_recv", "ts_init"):
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _resolve_price_column(columns: list[str]) -> str | None:
        for candidate in ("close", "price", "last", "bid_px_00", "mid"):
            if candidate in columns:
                return candidate
        return None

    def collect_historical_l0_l1(
        self,
        *,
        symbols: list[str],
        start_date: str,
        end_date: str,
        batch_size: int,
    ) -> dict[str, pl.DataFrame]:
        """
        Collect historical L0/L1 artifacts and return per-symbol frames.
        """
        del batch_size
        start_dt = self._parse_iso_date(start_date, fallback=datetime(2018, 1, 1))
        end_dt = self._parse_iso_date(end_date, fallback=datetime.now())
        span_days = max((end_dt - start_dt).days, 365)
        years = max(1, min(7, ceil(span_days / 365)))

        self._collector.collect_l1_trades(symbols=symbols, years=years)

        frames = self._load_symbol_artifacts(symbols=symbols, pattern="trades_*.parquet")
        if frames:
            return frames
        return self._load_phase1_checkpoints(symbols)

    def collect_microstructure_l2_l3(
        self,
        *,
        symbols: list[str],
        lookback_days: int,
        depth_levels: int,
    ) -> dict[str, pl.DataFrame]:
        """
        Collect L2 microstructure artifacts and return per-symbol frames.
        """
        del depth_levels
        self._collector.collect_l2_depth(symbols=symbols, days=lookback_days)

        exact_pattern = f"l2_depth_{lookback_days}d.parquet"
        frames = self._load_symbol_artifacts(symbols=symbols, pattern=exact_pattern)
        if frames:
            return frames
        return self._load_symbol_artifacts(symbols=symbols, pattern="l2_depth_*.parquet")

    def compute_cross_sectional_features(
        self,
        *,
        universe_data: dict[str, pl.DataFrame],
        window_sizes: list[int],
    ) -> pl.DataFrame:
        """
        Build coarse cross-sectional features from available symbol frames.
        """
        if not universe_data:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="us"),
                    "instrument_id": pl.Utf8,
                    "reference_price": pl.Float64,
                    "cross_sectional_rank": pl.Float64,
                },
            )

        normalized_frames: list[pl.DataFrame] = []
        for symbol, frame in universe_data.items():
            if frame.is_empty():
                continue
            columns = list(frame.columns)
            timestamp_col = self._resolve_timestamp_column(columns)
            price_col = self._resolve_price_column(columns)
            if timestamp_col is None or price_col is None:
                LOGGER.debug(
                    "Skipping symbol without timestamp/price columns symbol=%s columns=%s",
                    symbol,
                    columns[:12],
                )
                continue

            normalized = (
                frame.select(
                    pl.col(timestamp_col)
                    .cast(pl.Datetime(time_unit="us"), strict=False)
                    .alias("timestamp"),
                    pl.col(price_col).cast(pl.Float64, strict=False).alias("reference_price"),
                )
                .drop_nulls(subset=["timestamp", "reference_price"])
                .with_columns(pl.lit(symbol).alias("instrument_id"))
            )
            if not normalized.is_empty():
                normalized_frames.append(normalized)

        if not normalized_frames:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="us"),
                    "instrument_id": pl.Utf8,
                    "reference_price": pl.Float64,
                    "cross_sectional_rank": pl.Float64,
                },
            )

        combined = pl.concat(normalized_frames, how="vertical_relaxed").sort(
            by=["timestamp", "instrument_id"],
        )
        features = combined.with_columns(
            pl.col("reference_price").rank("average").over("timestamp").alias("cross_sectional_rank"),
        )

        for window_size in window_sizes:
            normalized_window = max(int(window_size), 1)
            features = features.with_columns(
                pl.col("reference_price")
                .rolling_mean(window_size=normalized_window)
                .over("instrument_id")
                .alias(f"reference_price_mean_{normalized_window}"),
            )

        return features

    def collect_regime_indicators(self) -> pl.DataFrame:
        """
        Return coarse regime indicators suitable for phase-level reporting.
        """
        now = datetime.now()
        return pl.DataFrame(
            {
                "timestamp": [now],
                "volatility_regime": [0.0],
                "liquidity_regime": [0.0],
            },
        )

    def get_all_symbols(self) -> list[str]:
        """Return discovered symbols from collector state and checkpoints."""
        symbols = {str(symbol) for symbol in self._collector.existing_symbols}
        symbols.update(str(symbol) for symbol in getattr(self._collector, "PRIORITY_SYMBOLS", []))
        checkpoint_dir = self.data_dir / "checkpoints"
        for checkpoint_file in checkpoint_dir.glob("*_l0_l1_7y.parquet"):
            symbols.add(checkpoint_file.stem.split("_")[0])
        return sorted(symbols)


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

    results = collector.collect_historical_l0_l1(
        symbols=priority_symbols,
        start_date="2018-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        batch_size=3,
    )

    total_samples = sum(len(df) for df in results.values())
    if results:
        LOGGER.info(
            "Phase 1 Complete: symbols=%d total_samples=%d average=%d",
            len(results),
            total_samples,
            total_samples // max(len(results), 1),
        )
    else:
        LOGGER.info("Phase 1 produced no symbol frames.")

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

    results = collector.collect_microstructure_l2_l3(
        symbols=top_symbols,
        lookback_days=30,
        depth_levels=10,
    )

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

    cross_features = collector.compute_cross_sectional_features(
        universe_data=universe_data,
        window_sizes=[20, 60, 120, 390],
    )

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

    regime = collector.collect_regime_indicators()

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

    if not symbols:
        LOGGER.warning("Skipping phase 5; no phase1 checkpoints found in %s", checkpoint_dir)
        return pl.DataFrame()

    catalog = ParquetDataCatalog(path=str(checkpoint_dir))
    builder = TFTDatasetBuilder(catalog=catalog, symbols=symbols[:10])

    target_semantics = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=15),),
        binary=BinaryTargetConfig(
            enabled=True,
            threshold_bps=decimal_to_bps(0.002),
            return_basis="raw",
        ),
    )

    df_any = builder.build_training_dataset(
        target_semantics=target_semantics,
        lookback_periods=50,
        use_polars=True,
    )

    if not isinstance(df_any, pl.DataFrame):  # Defensive
        raise TypeError("Expected Polars DataFrame from builder when use_polars=True")
    df = df_any

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

    total_storage = sum(float(est["total_gb"]) for est in estimates.values())

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


__all__ = [
    "ProductionDataCollector",
    "ProductionDatasetConfig",
    "build_production_dataset",
]
