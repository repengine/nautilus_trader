"""
Utility for building a quick TFT training dataset and optional teacher run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pandas import DataFrame

_DEFAULT_DATA_DIRS: tuple[Path, ...] = (
    Path("data/tier1"),
    Path("/home/nate/projects/nautilus_trader/data/tier1"),
    Path("data"),
)
_DEFAULT_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")


@dataclass(slots=True, frozen=True)
class QuickTFTTrainConfig:
    """
    Configuration for `train_tft_quick` task.
    """

    data_dirs: Sequence[Path] = _DEFAULT_DATA_DIRS
    output_dir: Path = Path("data/tft_training")
    symbols: Sequence[str] = _DEFAULT_SYMBOLS
    horizon_minutes: int = 15
    min_return_threshold: float = 0.002
    lookback_periods: int = 50
    sample_prediction_count: int = 10


@dataclass(slots=True, frozen=True)
class QuickTFTTrainResult:
    """
    Summary of the quick TFT training pipeline.
    """

    dataset_parquet: Path
    dataset_csv: Path
    dataset_shape: tuple[int, int]
    target_distribution_json: str
    trained: bool
    sample_predictions: tuple[float, ...] | None


def _select_data_dir(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            LOGGER.info("Using data directory: %s", path)
            return path
    raise FileNotFoundError("No valid data directory found among provided candidates")


def _to_pandas_frame(obj: object) -> DataFrame:
    if pd is None:
        check_ml_dependencies(["pandas"])
        if pd is None:
            raise RuntimeError("pandas remains unavailable after dependency check")
    if pl is not None and isinstance(obj, pl.DataFrame):
        return cast(DataFrame, obj.to_pandas())
    if isinstance(obj, pd.DataFrame):
        return cast(DataFrame, obj)
    raise TypeError("Expected Polars or Pandas DataFrame from dataset builder")


def _target_distribution(df: DataFrame) -> dict[str, int]:
    if "y" not in df.columns:
        return {}
    counts = df["y"].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def train_tft_quick(config: QuickTFTTrainConfig) -> QuickTFTTrainResult:
    """
    Build a quick TFT dataset and optionally train a small teacher model.
    """
    data_dir = _select_data_dir(config.data_dirs)

    catalog = ParquetDataCatalog(path=str(data_dir))
    builder = TFTDatasetBuilder(catalog=catalog, symbols=list(config.symbols))

    dataset = builder.build_training_dataset(
        horizon_minutes=config.horizon_minutes,
        min_return_threshold=config.min_return_threshold,
        lookback_periods=config.lookback_periods,
        use_polars=True,
    )

    df_pd = _to_pandas_frame(dataset)
    if df_pd.empty:
        raise ValueError("No data was processed successfully; dataset is empty")

    LOGGER.info("Dataset shape: rows=%d columns=%d", df_pd.shape[0], df_pd.shape[1])
    LOGGER.info("Dataset columns: %s", list(df_pd.columns))

    target_distribution = _target_distribution(df_pd)
    if target_distribution:
        LOGGER.info("Target distribution: %s", target_distribution)
        minority = min(target_distribution.values()) / max(sum(target_distribution.values()), 1)
        LOGGER.info("Minority class percentage: %.4f", minority * 100.0)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tft_training_data.csv"
    parquet_path = output_dir / "tft_training_data.parquet"
    df_pd.to_csv(csv_path, index=False)
    df_pd.to_parquet(parquet_path, index=False)
    LOGGER.info("Saved training data to CSV=%s, Parquet=%s", csv_path, parquet_path)

    trained = False
    sample_predictions: tuple[float, ...] | None = None

    try:
        from ml.training.teacher.tft_teacher import TFTTeacher
        from ml.training.teacher.tft_teacher import TFTTeacherConfig

        LOGGER.info("Starting TFT teacher training")
        teacher = TFTTeacher(
            config=TFTTeacherConfig(),
            max_encoder_length=20,
            max_prediction_length=1,
            hidden_size=32,
            lstm_layers=2,
            dropout=0.1,
            static_categoricals=["asset_class", "exchange"],
            static_reals=["tick_size"],
            time_varying_known_reals=[
                "tod_sin",
                "tod_cos",
                "dow_sin",
                "dow_cos",
                "is_market_open",
                "is_premarket",
                "is_aftermarket",
            ],
            time_varying_unknown_reals=[
                col
                for col in df_pd.select_dtypes(include=[np.number]).columns
                if col
                not in {
                    "time_index",
                    "y",
                    "tick_size",
                    "tod_sin",
                    "tod_cos",
                    "dow_sin",
                    "dow_cos",
                    "is_market_open",
                    "is_premarket",
                    "is_aftermarket",
                }
            ],
        )

        train_size = int(len(df_pd) * 0.8)
        train_df = df_pd.iloc[:train_size].copy()
        val_df = df_pd.iloc[train_size:].copy()
        LOGGER.info("Training rows=%d validation rows=%d", len(train_df), len(val_df))

        teacher.fit(train_df)
        predictions = teacher.predict_logits(val_df.head(config.sample_prediction_count))
        if predictions is not None:
            logits = predictions[: config.sample_prediction_count]
            probabilities = 1 / (1 + np.exp(-logits))
            sample_predictions = tuple(float(x) for x in probabilities)
            trained = True
            LOGGER.info("Sample predictions: %s", sample_predictions)
        else:
            LOGGER.warning("Teacher returned no logits; skipping sample predictions")
    except ImportError as exc:
        LOGGER.warning("TFT teacher module not available: %s", exc, exc_info=True)
    except Exception as exc:
        LOGGER.error("TFT training failed", exc_info=True)
        raise RuntimeError("TFT training failed") from exc

    return QuickTFTTrainResult(
        dataset_parquet=parquet_path,
        dataset_csv=csv_path,
        dataset_shape=(df_pd.shape[0], df_pd.shape[1]),
        target_distribution_json=json.dumps(target_distribution, sort_keys=True),
        trained=trained,
        sample_predictions=sample_predictions,
    )


__all__ = [
    "QuickTFTTrainConfig",
    "QuickTFTTrainResult",
    "train_tft_quick",
]
