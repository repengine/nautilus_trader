"""
Dataset task helpers consumed by CLI entry points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ml.data import BuildResult
from ml.data import DatasetBuildConfig
from ml.data import build_tft_dataset as _build_tft_dataset


FeatureRoleName = Literal["teacher", "student", "inference_support"]


@dataclass(slots=True, frozen=True)
class TFTDatasetTaskConfig:
    """
    Configuration options for building a TFT dataset.
    """

    data_dir: Path
    out_dir: Path
    symbols: Sequence[str]
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 30
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    chunk_days: int = 0
    start: datetime | None = None
    end: datetime | None = None
    register_features: bool = False
    feature_registry_dir: Path | None = None
    feature_role: FeatureRoleName = "teacher"
    emit_dataset_events: bool = False
    fred_vintage_dir: Path | None = None
    events_base_dir: Path | None = None
    student_mode: bool = False


def build_tft_dataset(cfg: TFTDatasetTaskConfig) -> BuildResult:
    """
    Build a TFT dataset using :mod:`ml.data` helpers.
    """
    dataset_cfg = DatasetBuildConfig(
        data_dir=cfg.data_dir,
        out_dir=cfg.out_dir,
        symbols=[symbol.upper() for symbol in cfg.symbols],
        include_macro=cfg.include_macro,
        macro_lag_days=cfg.macro_lag_days,
        include_micro=cfg.include_micro,
        include_l2=cfg.include_l2,
        include_events=cfg.include_events,
        include_calendar=cfg.include_calendar,
        horizon_minutes=cfg.horizon_minutes,
        threshold=cfg.threshold,
        lookback_periods=cfg.lookback_periods,
        start=cfg.start,
        end=cfg.end,
        chunk_days=cfg.chunk_days,
        register_features=cfg.register_features,
        feature_registry_dir=cfg.feature_registry_dir,
        feature_role=cfg.feature_role,
        emit_dataset_events=cfg.emit_dataset_events,
        fred_vintage_dir=cfg.fred_vintage_dir,
        events_base_dir=cfg.events_base_dir,
        student_mode=cfg.student_mode,
    )
    return _build_tft_dataset(dataset_cfg)


__all__ = [
    "FeatureRoleName",
    "TFTDatasetTaskConfig",
    "build_tft_dataset",
]
