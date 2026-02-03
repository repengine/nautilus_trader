"""
Dataset task helpers consumed by CLI entry points.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from ml.config.market_data import MarketDatasetInput
from ml.data import BuildResult
from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset as _build_tft_dataset
from ml.data.vintage import VintagePolicy
from ml.config.targets import TargetSemanticsConfig
from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
from ml.preprocessing.vintage_age import update_metadata_with_vintage_age
from ml.preprocessing.vintage_age import write_metadata
from ml.stores.protocols import DataStoreFacadeProtocol


FeatureRoleName = Literal["teacher", "student", "inference_support"]


@dataclass(slots=True, frozen=True)
class TFTDatasetTaskConfig:
    """
    Configuration options for building a TFT dataset.
    """

    data_dir: Path
    out_dir: Path
    symbols: Sequence[str]
    target_semantics: TargetSemanticsConfig
    instrument_ids: Sequence[str] | None = None
    lookback_periods: int = 30
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    include_earnings: bool = False
    earnings_lag_days: int = 1
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    micro_base_dir: Path | None = None
    l2_base_dir: Path | None = None
    chunk_days: int = 0
    start: datetime | None = None
    end: datetime | None = None
    write_csv: bool | None = None
    csv_max_rows: int = 1_000_000
    csv_sample_rows: int = 0
    register_features: bool = False
    feature_registry_dir: Path | None = None
    feature_role: FeatureRoleName = "teacher"
    emit_dataset_events: bool = False
    fred_vintage_dir: Path | None = None
    events_base_dir: Path | None = None
    student_mode: bool = False
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: Path | None = None
    validation: DatasetValidationConfig | None = None
    market_dataset_id: str | None = None
    market_inputs: tuple[MarketDatasetInput, ...] | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: datetime | None = None
    convert_vintage_to_age: bool = False
    include_macro_revisions: bool = False
    macro_revision_mode: Literal["minimal", "core", "full"] = "core"
    macro_revision_windows: tuple[int, ...] | None = None


def build_tft_dataset(
    cfg: TFTDatasetTaskConfig,
    *,
    data_store: DataStoreFacadeProtocol | None = None,
) -> BuildResult:
    """
    Build a TFT dataset using :mod:`ml.data` helpers.
    """
    dataset_cfg = DatasetBuildConfig(
        data_dir=cfg.data_dir,
        out_dir=cfg.out_dir,
        symbols=[symbol.upper() for symbol in cfg.symbols],
        instrument_ids=[inst for inst in cfg.instrument_ids] if cfg.instrument_ids else None,
        target_semantics=cfg.target_semantics,
        include_macro=cfg.include_macro,
        macro_lag_days=cfg.macro_lag_days,
        include_micro=cfg.include_micro,
        include_l2=cfg.include_l2,
        include_events=cfg.include_events,
        include_calendar=cfg.include_calendar,
        include_macro_deltas=cfg.include_macro_deltas,
        include_calendar_lags=cfg.include_calendar_lags,
        include_clustering_tags=cfg.include_clustering_tags,
        include_context_features=cfg.include_context_features,
        include_earnings=cfg.include_earnings,
        earnings_lag_days=cfg.earnings_lag_days,
        micro_base_dir=cfg.micro_base_dir,
        l2_base_dir=cfg.l2_base_dir,
        lookback_periods=cfg.lookback_periods,
        start=cfg.start,
        end=cfg.end,
        chunk_days=cfg.chunk_days,
        write_csv=cfg.write_csv,
        csv_max_rows=cfg.csv_max_rows,
        csv_sample_rows=cfg.csv_sample_rows,
        register_features=cfg.register_features,
        feature_registry_dir=cfg.feature_registry_dir,
        feature_role=cfg.feature_role,
        emit_dataset_events=cfg.emit_dataset_events,
        fred_vintage_dir=cfg.fred_vintage_dir,
        events_base_dir=cfg.events_base_dir,
        student_mode=cfg.student_mode,
        auto_refresh_macro=cfg.auto_refresh_macro,
        macro_staleness_hours=cfg.macro_staleness_hours,
        macro_series_ids=cfg.macro_series_ids,
        macro_fred_path=cfg.macro_fred_path,
        validation=cfg.validation,
        market_dataset_id=cfg.market_dataset_id,
        market_inputs=cfg.market_inputs,
        vintage_policy=cfg.vintage_policy,
        vintage_as_of=cfg.vintage_as_of,
        include_macro_revisions=cfg.include_macro_revisions,
        macro_revision_mode=cfg.macro_revision_mode,
        macro_revision_windows=cfg.macro_revision_windows,
    )
    result = _build_tft_dataset(dataset_cfg, data_store=data_store)
    if not cfg.convert_vintage_to_age:
        return result

    destination = result.dataset_parquet.with_name("dataset_with_vintage_age.parquet")
    conversion = convert_vintage_timestamps_to_age(result.dataset_parquet, destination)
    metadata_path = result.dataset_parquet.with_name("dataset_metadata.json")
    if not metadata_path.exists():
        msg = f"Metadata file not found for dataset: {metadata_path}"
        raise FileNotFoundError(msg)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    updated_metadata = update_metadata_with_vintage_age(
        metadata,
        vintage_columns=conversion.vintage_columns,
        age_columns=conversion.age_columns,
    )
    write_metadata(metadata_path, updated_metadata)
    return replace(result, dataset_parquet=destination)


__all__ = [
    "FeatureRoleName",
    "TFTDatasetTaskConfig",
    "build_tft_dataset",
]
