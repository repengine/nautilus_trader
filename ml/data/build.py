"""
Dataset build orchestration for TFT datasets.

Centralizes dataset build configuration, feature materialization, and artifact
persistence for TFT dataset pipelines.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pyarrow.parquet as pq
import structlog
from numpy.typing import NDArray

from ml.common import current_rss_mb
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.config.feature_cache import FeatureCachePolicy
from ml.config.feature_cache import normalize_feature_cache_policy
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import load_market_feed_descriptors
from ml.config.targets import TargetSemanticsConfig
from ml.data.common.capability_flags import capability_flags_from_builder
from ml.data.common.dataset_csv import resolve_write_csv as _resolve_write_csv
from ml.data.common.dataset_csv import write_dataset_csv as _write_dataset_csv
from ml.data.common.dataset_events import emit_dataset_build_event
from ml.data.common.macro_vintage import derive_alfred_range
from ml.data.common.macro_vintage import normalize_vintage_as_of
from ml.data.common.macro_vintage import prepare_validation_config
from ml.data.common.macro_vintage import refresh_macro_artifacts_if_needed
from ml.data.common.macro_vintage import resolve_alfred_window_days
from ml.data.common.macro_vintage import resolve_fred_parquet_path
from ml.data.common.macro_vintage import resolve_macro_presence
from ml.data.common.target_semantics import resolve_binary_target_column
from ml.data.common.target_semantics import resolve_target_semantics
from ml.data.ingest.market_bindings import resolve_market_dataset_bindings
from ml.data.metadata import DatasetMetadata
from ml.data.metadata import MarketBindingMetadata
from ml.data.metadata import _binding_stats_to_metadata
from ml.data.metadata import _compute_dataset_metadata
from ml.data.metadata import _ensure_datetime
from ml.data.metadata import _validate_dataset_metadata
from ml.data.metadata import build_dataset_metadata_from_windows
from ml.data.metadata import write_dataset_metadata
from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import DatasetValidationResult
from ml.data.validation import validate_dataset
from ml.data.vintage import VintagePolicy
from ml.ml_types import PolarsDF
from ml.stores.protocols import DataStoreFacadeProtocol


if TYPE_CHECKING:
    from ml.data.tft_dataset_builder import TFTDatasetBuilder


FloatArray = NDArray[np.float32]

logger = structlog.get_logger(__name__)

_DATASET_BUILD_RSS_GAUGE = get_gauge(
    "ml_dataset_build_rss_mb",
    "Observed RSS (MB) during dataset build chunk stages.",
    labelnames=("stage",),
)
_DATASET_BUILD_CHUNK_SECONDS = get_histogram(
    "ml_dataset_build_chunk_seconds",
    "Elapsed seconds for dataset build chunk stages.",
    labelnames=("stage",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)


@dataclass(slots=True)
class _ChunkMeta:
    path: Path
    rows: int
    positives: float
    macro_counts: dict[str, int]
    ts_start: datetime | None
    ts_end: datetime | None


@dataclass(frozen=True)
class DatasetBuildConfig:
    """
    Configuration for TFT dataset builds.

    Args:
        data_dir: Base directory for input data.
        out_dir: Output directory for dataset artifacts.
        symbols: Symbols to include.
        target_semantics: Target generation configuration.
        dataset_id: Dataset identifier.
        market_dataset_id: Optional dataset id for market bindings.
        market_inputs: Optional market input overrides for bindings.
        instrument_ids: Optional instrument identifiers.
        include_macro: Whether to include macro joins.
        macro_lag_days: Lag days for macro joins.
        include_micro: Whether to include microstructure features.
        include_l2: Whether to include L2 features.
        micro_cache_policy: Cache policy for microstructure features.
        l2_cache_policy: Cache policy for L2 features.
        include_events: Whether to include event features.
        include_calendar: Whether to include calendar features.
        include_earnings: Whether to include earnings features.
        earnings_lag_days: Lag days for earnings joins.
        include_macro_deltas: Whether to include macro delta features.
        include_calendar_lags: Whether to include calendar lag features.
        include_clustering_tags: Whether to include clustering tags.
        include_context_features: Whether to include contextual features.
        fred_vintage_dir: Directory for ALFRED vintage data.
        events_base_dir: Directory for event data.
        student_mode: Whether to enable student-mode shortcuts.
        micro_base_dir: Override directory for micro data.
        l2_base_dir: Override directory for L2 data.
        lookback_periods: Lookback window length.
        start: Optional dataset start datetime.
        end: Optional dataset end datetime.
        chunk_days: Chunk size in days for chunked builds.
        write_csv: Override for CSV emission.
        csv_max_rows: Maximum CSV rows to emit.
        csv_sample_rows: Sample row count for dataset_sample.csv.
        register_features: Whether to register features.
        feature_registry_dir: Feature registry output directory.
        feature_role: Feature registry role.
        emit_dataset_events: Whether to emit dataset events.
        auto_refresh_macro: Whether to auto-refresh macro data.
        macro_staleness_hours: Max macro staleness before refresh.
        macro_series_ids: Macro series identifiers.
        macro_fred_path: Override for FRED parquet path.
        vintage_policy: Vintage policy for macro joins.
        vintage_as_of: Vintage cutoff timestamp.
        include_macro_revisions: Whether to include macro revisions.
        macro_revision_mode: Macro revision mode.
        macro_revision_windows: Macro revision windows.
        validation: Optional dataset validation configuration.
    """

    data_dir: Path
    out_dir: Path
    symbols: list[str]
    # Builder params
    target_semantics: TargetSemanticsConfig
    dataset_id: str = "tft_dataset"
    market_dataset_id: str | None = None
    market_inputs: tuple[MarketDatasetInput, ...] | None = None
    instrument_ids: list[str] | None = None
    # Feature options
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    micro_cache_policy: FeatureCachePolicy = "cache_first"
    l2_cache_policy: FeatureCachePolicy = "cache_first"
    include_events: bool = False
    include_calendar: bool = False
    include_earnings: bool = False
    earnings_lag_days: int = 1
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    fred_vintage_dir: Path | None = None
    events_base_dir: Path | None = None
    student_mode: bool = False
    micro_base_dir: Path | None = None
    l2_base_dir: Path | None = None
    lookback_periods: int = 30
    # Optional window
    start: datetime | None = None
    end: datetime | None = None
    chunk_days: int = 0
    # Output controls
    write_csv: bool | None = None
    csv_max_rows: int = 1_000_000
    csv_sample_rows: int = 0
    # Optional feature registration
    register_features: bool = False
    feature_registry_dir: Path | None = None
    feature_role: str = "teacher"  # teacher|student|inference_support
    # Optional lineage/events (cold path only)
    emit_dataset_events: bool = False
    # Macro refresh controls
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: Path | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: datetime | None = None
    include_macro_revisions: bool = False
    macro_revision_mode: Literal["minimal", "core", "full"] = "core"
    macro_revision_windows: tuple[int, ...] | None = None
    # Validation
    validation: DatasetValidationConfig | None = None

    def __post_init__(self) -> None:
        """
        Normalize cache policy tokens.
        """
        object.__setattr__(
            self,
            "micro_cache_policy",
            normalize_feature_cache_policy(
                self.micro_cache_policy,
                label="micro_cache_policy",
            ),
        )
        object.__setattr__(
            self,
            "l2_cache_policy",
            normalize_feature_cache_policy(
                self.l2_cache_policy,
                label="l2_cache_policy",
            ),
        )


@dataclass(frozen=True)
class BuildResult:
    """
    Output artifacts produced by a dataset build.

    Args:
        dataset_parquet: Parquet dataset path.
        dataset_csv: CSV dataset path.
        features_npz: Feature matrix NPZ path.
        feature_names: Feature column names.
        feature_set_id: Optional feature registry id.
        metadata: Optional dataset metadata payload.
    """

    dataset_parquet: Path
    dataset_csv: Path
    features_npz: Path
    feature_names: list[str]
    feature_set_id: str | None = None
    metadata: DatasetMetadata | None = None


def _resolve_target_semantics(cfg: DatasetBuildConfig) -> TargetSemanticsConfig:
    """
    Resolve target semantics for a dataset build configuration.

    Args:
        cfg: Dataset build configuration.

    Returns:
        TargetSemanticsConfig instance.
    """
    target_semantics = cast(
        TargetSemanticsConfig | None,
        getattr(cfg, "target_semantics", None),
    )
    return resolve_target_semantics(
        target_semantics,
        error_message=(
            "target_semantics must be provided in DatasetBuildConfig; legacy defaults are disabled"
        ),
    )


def _resolve_binary_target_column(target_semantics: TargetSemanticsConfig) -> str | None:
    """
    Resolve the binary target column name for positive-rate checks.

    Args:
        target_semantics: Target semantics configuration.

    Returns:
        Binary target column name if available.
    """
    return resolve_binary_target_column(target_semantics)


def _sorted_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(sorted(str(item) for item in values))


def _sorted_market_bindings(
    bindings: Sequence[MarketBindingMetadata] | None,
) -> tuple[tuple[str, ...], ...]:
    if not bindings:
        return ()
    summary: list[tuple[str, ...]] = []
    for binding in bindings:
        summary.append(
            (
                binding.dataset_id,
                binding.descriptor_id or "",
                ",".join(sorted(binding.symbols)),
                ",".join(sorted(binding.instrument_ids)),
                binding.source,
                binding.storage_kind or "",
            ),
        )
    summary.sort()
    return tuple(summary)


def compute_dataset_pipeline_signature(
    *,
    dataset_id: str | None,
    symbols: Sequence[str],
    instrument_ids: Sequence[str] | None,
    macro_series_ids: Sequence[str] | None,
    include_macro: bool,
    macro_lag_days: int,
    vintage_policy: VintagePolicy,
    vintage_cutoff: str | None,
    ts_event_start: str | None,
    ts_event_end: str | None,
    market_bindings: Sequence[MarketBindingMetadata] | None = None,
) -> str:
    """
    Compute a deterministic pipeline signature describing dataset lineage.

    Args:
        dataset_id: Dataset identifier.
        symbols: Symbols used in the build.
        instrument_ids: Optional instrument ids.
        macro_series_ids: Optional macro series identifiers.
        include_macro: Whether macro joins were enabled.
        macro_lag_days: Macro lag days.
        vintage_policy: Vintage policy used for macro joins.
        vintage_cutoff: Vintage cutoff timestamp.
        ts_event_start: Dataset window start timestamp.
        ts_event_end: Dataset window end timestamp.
        market_bindings: Optional market binding metadata.

    Returns:
        Deterministic pipeline signature string.
    """
    payload = {
        "dataset_id": dataset_id,
        "symbols": _sorted_tuple(symbols),
        "instrument_ids": _sorted_tuple(instrument_ids),
        "macro_series_ids": _sorted_tuple(macro_series_ids),
        "include_macro": bool(include_macro),
        "macro_lag_days": int(macro_lag_days),
        "vintage_policy": vintage_policy.value,
        "vintage_cutoff": vintage_cutoff,
        "ts_event_start": ts_event_start,
        "ts_event_end": ts_event_end,
        "market_bindings": _sorted_market_bindings(market_bindings),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"tft_pipeline:{digest[:16]}"


def _infer_feature_columns(df: Any) -> list[str]:
    """
    Infer numeric feature columns, excluding label/index/meta fields.

    Args:
        df: Dataset dataframe (polars or pandas).

    Returns:
        List of inferred feature column names.
    """
    from ml.data.feature_columns import infer_numeric_feature_columns

    return infer_numeric_feature_columns(df)


def _write_feature_npz_from_polars(
    df_sorted: PolarsDF,
    feature_names: Sequence[str],
    *,
    out_path: Path,
    cutoff: int,
    chunk_size: int = 200_000,
) -> None:
    """
    Persist feature matrices to an ``.npz`` without loading all rows into memory.

    Args:
        df_sorted: Sorted polars dataframe.
        feature_names: Feature column names to include.
        out_path: Destination path for the npz file.
        cutoff: Train/validation split index.
        chunk_size: Chunk size for streaming writes.
    """
    from ml._imports import pl

    if pl is None:
        msg = "Polars is required to write feature matrices"
        raise RuntimeError(msg)
    if not isinstance(df_sorted, pl.DataFrame):
        msg = "df_sorted must be a Polars DataFrame"
        raise TypeError(msg)

    num_rows = df_sorted.height
    num_features = len(feature_names)
    if num_rows == 0 or num_features == 0:
        np.savez(
            out_path,
            X_train=np.empty((0, num_features), dtype=np.float32),
            X_val=np.empty((0, num_features), dtype=np.float32),
            feature_names=np.array(feature_names, dtype=np.str_),
        )
        return

    chunk_size = max(int(chunk_size), 1)
    train_rows = int(min(max(cutoff, 0), num_rows))
    val_rows = int(max(num_rows - train_rows, 0))

    train_tmp = out_path.with_name(out_path.name + ".train.tmp") if train_rows > 0 else None
    val_tmp = out_path.with_name(out_path.name + ".val.tmp") if val_rows > 0 else None

    train_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    val_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    if train_tmp is not None and num_features > 0:
        train_mem = np.memmap(
            str(train_tmp),
            dtype=np.float32,
            mode="w+",
            shape=(train_rows, num_features),
        )
    if val_tmp is not None and num_features > 0:
        val_mem = np.memmap(
            str(val_tmp),
            dtype=np.float32,
            mode="w+",
            shape=(val_rows, num_features),
        )

    train_pos = 0
    val_pos = 0
    feature_expr = [pl.col(name).cast(pl.Float32) for name in feature_names]

    row_start = 0
    while row_start < num_rows:
        length = min(chunk_size, num_rows - row_start)
        if feature_expr:
            chunk = df_sorted.slice(row_start, length).select(feature_expr)
            chunk_np = chunk.to_numpy()
        else:
            chunk_np = np.empty((length, 0), dtype=np.float32)

        row_end = row_start + length
        if train_mem is not None and row_start < train_rows:
            train_end = min(row_end, train_rows)
            train_len = train_end - row_start
            if train_len > 0:
                train_mem[train_pos : train_pos + train_len] = chunk_np[:train_len]
                train_pos += train_len
        if val_mem is not None and row_end > train_rows:
            val_chunk_start = max(train_rows - row_start, 0)
            val_len = row_end - max(row_start, train_rows)
            if val_len > 0:
                val_mem[val_pos : val_pos + val_len] = chunk_np[val_chunk_start : val_chunk_start + val_len]
                val_pos += val_len

        row_start = row_end

    X_train: NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]]
    X_val: NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]]
    if train_mem is not None:
        assert train_tmp is not None
        train_mem.flush()
        X_train = np.memmap(
            str(train_tmp),
            dtype=np.float32,
            mode="r",
            shape=(train_rows, num_features),
        )
    else:
        X_train = np.empty((0, num_features), dtype=np.float32)
    if val_mem is not None:
        assert val_tmp is not None
        val_mem.flush()
        X_val = np.memmap(
            str(val_tmp),
            dtype=np.float32,
            mode="r",
            shape=(val_rows, num_features),
        )
    else:
        X_val = np.empty((0, num_features), dtype=np.float32)

    try:
        np.savez(
            out_path,
            X_train=X_train,
            X_val=X_val,
            feature_names=np.array(feature_names, dtype=np.str_),
        )
    finally:
        if isinstance(X_train, np.memmap):
            del X_train
        if isinstance(X_val, np.memmap):
            del X_val
        if train_tmp is not None and train_tmp.exists():
            train_tmp.unlink()
        if val_tmp is not None and val_tmp.exists():
            val_tmp.unlink()


@dataclass(slots=True)
class _StreamingFeatureWriter:
    """Incrementally write feature matrices to NPZ using memory-mapped buffers."""

    out_path: Path
    feature_names: list[str]
    total_rows: int
    cutoff: int
    _train_tmp: Path | None = None
    _val_tmp: Path | None = None
    _train_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    _val_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    _train_rows: int = 0
    _val_rows: int = 0
    _train_cursor: int = 0
    _val_cursor: int = 0

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            msg = "total_rows must be non-negative"
            raise ValueError(msg)
        feature_dim = len(self.feature_names)
        self._train_rows = max(min(self.cutoff, self.total_rows), 0)
        self._val_rows = max(self.total_rows - self._train_rows, 0)
        if feature_dim == 0:
            return
        if self._train_rows > 0:
            self._train_tmp = self.out_path.with_name(self.out_path.name + ".train.tmp")
            self._train_mem = np.memmap(
                str(self._train_tmp),
                dtype=np.float32,
                mode="w+",
                shape=(self._train_rows, feature_dim),
            )
        if self._val_rows > 0:
            self._val_tmp = self.out_path.with_name(self.out_path.name + ".val.tmp")
            self._val_mem = np.memmap(
                str(self._val_tmp),
                dtype=np.float32,
                mode="w+",
                shape=(self._val_rows, feature_dim),
            )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def append(self, values: FloatArray, *, global_offset: int) -> None:
        if self.feature_dim == 0:
            return
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            msg = "Chunk feature matrix has unexpected shape"
            raise ValueError(msg)
        rows = values.shape[0]
        if rows == 0:
            return
        train_remaining = max(self._train_rows - global_offset, 0)
        train_take = min(train_remaining, rows)
        if train_take > 0 and self._train_mem is not None:
            self._train_mem[self._train_cursor : self._train_cursor + train_take] = values[:train_take]
            self._train_cursor += train_take
        val_take = rows - train_take
        if val_take > 0 and self._val_mem is not None:
            self._val_mem[self._val_cursor : self._val_cursor + val_take] = values[train_take:]
            self._val_cursor += val_take

    def finalize(self) -> None:
        if self.feature_dim == 0:
            empty = cast(FloatArray, np.empty((0, 0), dtype=np.float32))
            np.savez(
                self.out_path,
                X_train=empty,
                X_val=empty,
                feature_names=np.array(self.feature_names, dtype=np.str_),
            )
            return
        if self._train_mem is not None:
            self._train_mem.flush()
        if self._val_mem is not None:
            self._val_mem.flush()
        X_train: FloatArray
        X_val: FloatArray
        if self._train_tmp is not None and self._train_tmp.exists():
            X_train = cast(
                FloatArray,
                np.memmap(
                    str(self._train_tmp),
                    dtype=np.float32,
                    mode="r",
                    shape=(self._train_rows, self.feature_dim),
                ),
            )
        else:
            X_train = cast(FloatArray, np.empty((0, self.feature_dim), dtype=np.float32))
        if self._val_tmp is not None and self._val_tmp.exists():
            X_val = cast(
                FloatArray,
                np.memmap(
                    str(self._val_tmp),
                    dtype=np.float32,
                    mode="r",
                    shape=(self._val_rows, self.feature_dim),
                ),
            )
        else:
            X_val = cast(FloatArray, np.empty((0, self.feature_dim), dtype=np.float32))
        try:
            np.savez(
                self.out_path,
                X_train=X_train,
                X_val=X_val,
                feature_names=np.array(self.feature_names, dtype=np.str_),
            )
        finally:
            if isinstance(X_train, np.memmap):
                del X_train
            if isinstance(X_val, np.memmap):
                del X_val
            if self._train_tmp is not None and self._train_tmp.exists():
                self._train_tmp.unlink()
            if self._val_tmp is not None and self._val_tmp.exists():
                self._val_tmp.unlink()


def _validate_aggregated_dataset(
    *,
    config: DatasetValidationConfig,
    total_rows: int,
    positive_rate: float | None,
    feature_coverage: dict[str, float],
    macro_counts: dict[str, int],
) -> DatasetValidationResult:
    if total_rows < config.min_rows:
        msg = f"Dataset has {total_rows} rows; minimum required is {config.min_rows}"
        raise DatasetValidationError(msg)

    if positive_rate is not None and config.min_positive_rate is not None:
        if positive_rate < config.min_positive_rate:
            msg = (
                f"Target positive rate {positive_rate:.4f} below minimum "
                f"{config.min_positive_rate:.4f}"
            )
            raise DatasetValidationError(msg)
    if positive_rate is not None and config.max_positive_rate is not None:
        if positive_rate > config.max_positive_rate:
            msg = (
                f"Target positive rate {positive_rate:.4f} above maximum "
                f"{config.max_positive_rate:.4f}"
            )
            raise DatasetValidationError(msg)

    if config.min_feature_coverage is not None:
        low_coverage = [
            (name, ratio)
            for name, ratio in feature_coverage.items()
            if ratio < config.min_feature_coverage
        ]
        if low_coverage:
            worst_low_cov = min(low_coverage, key=lambda item: item[1])
            msg = (
                "Feature coverage below acceptance threshold; "
                f"example: {worst_low_cov[0]}={worst_low_cov[1]:.3f} < {config.min_feature_coverage:.3f}"
            )
            raise DatasetValidationError(msg)

    macro_present = resolve_macro_presence(
        config=config,
        macro_counts=macro_counts,
    )

    return DatasetValidationResult(
        row_count=total_rows,
        positive_rate=positive_rate,
        feature_coverage=feature_coverage,
        macro_columns_present=macro_present,
        macro_observation_counts=macro_counts,
    )


def _build_dataset_chunked(
    *,
    builder: TFTDatasetBuilder,
    cfg: DatasetBuildConfig,
    vintage_as_of: datetime | None,
    build_ts: datetime,
    target_semantics: TargetSemanticsConfig,
) -> tuple[BuildResult, DatasetValidationResult]:
    from ml._imports import HAS_POLARS
    from ml._imports import check_ml_dependencies
    from ml._imports import pl

    polars_module = pl
    if not HAS_POLARS or polars_module is None:
        check_ml_dependencies(["polars"])
        import polars as polars_module_import

        polars_module = polars_module_import
    assert polars_module is not None

    def _update_rss_peak(current_peak: float | None) -> float | None:
        rss_mb = current_rss_mb()
        if rss_mb is None:
            return current_peak
        if current_peak is None or rss_mb > current_peak:
            return rss_mb
        return current_peak

    chunk_dir = cfg.out_dir / ".chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    capability_flags = capability_flags_from_builder(builder)

    chunk_metas: list[_ChunkMeta] = []
    binary_target_col = _resolve_binary_target_column(target_semantics)
    from ml.training.datasets.target_generator import build_target_semantics_metadata

    target_semantics_metadata = build_target_semantics_metadata(target_semantics)

    cursor = cast(datetime, cfg.start)
    end = cast(datetime, cfg.end)
    chunk_index = 0
    chunk_delta = timedelta(days=cfg.chunk_days)

    while cursor < end:
        chunk_started = time.perf_counter()
        chunk_peak_rss: float | None = None
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)
        chunk_end = min(cursor + chunk_delta, end)
        df_any = builder.build_training_dataset(
            target_semantics=target_semantics,
            lookback_periods=cfg.lookback_periods,
            use_polars=True,
            start=cursor,
            end=chunk_end,
        )
        if isinstance(df_any, polars_module.DataFrame):
            df_chunk = df_any
        else:
            from ml._imports import HAS_PANDAS
            from ml._imports import pd

            if not HAS_PANDAS:
                check_ml_dependencies(["pandas"])  # pragma: no cover
            assert pd is not None
            df_chunk = polars_module.from_pandas(df_any)
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)

        if not df_chunk.is_empty():
            chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.parquet"
            if "timestamp" in df_chunk.columns:
                df_chunk = df_chunk.sort(
                    ["timestamp", "instrument_id"] if "instrument_id" in df_chunk.columns else ["timestamp"],
                )
            elif "time_index" in df_chunk.columns:
                df_chunk = df_chunk.sort("time_index")
            df_chunk.write_parquet(str(chunk_path))

            positives = 0.0
            if binary_target_col and binary_target_col in df_chunk.columns:
                positives = float(df_chunk[binary_target_col].sum())

            macro_counts: dict[str, int] = {}
            if cfg.macro_series_ids:
                for macro in cfg.macro_series_ids:
                    value_col = f"{macro}__value_vintage_ts"
                    if value_col in df_chunk.columns:
                        count_val = int(df_chunk[value_col].is_not_null().sum())
                        macro_counts[macro] = count_val
            ts_start = None
            ts_end = None
            if "timestamp" in df_chunk.columns and df_chunk.height > 0:
                ts_series = df_chunk["timestamp"]
                ts_start = _ensure_datetime(
                    cast(datetime | float | None, ts_series.min()),
                )
                ts_end = _ensure_datetime(
                    cast(datetime | float | None, ts_series.max()),
                )

            chunk_metas.append(
                _ChunkMeta(
                    path=chunk_path,
                    rows=df_chunk.height,
                    positives=positives,
                    macro_counts=macro_counts,
                    ts_start=ts_start,
                    ts_end=ts_end,
                ),
            )

        cursor = chunk_end
        chunk_index += 1
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)
        _DATASET_BUILD_CHUNK_SECONDS.labels(stage="build").observe(
            time.perf_counter() - chunk_started,
        )
        if chunk_peak_rss is not None:
            _DATASET_BUILD_RSS_GAUGE.labels(stage="build").set(chunk_peak_rss)

    binding_metadata = _binding_stats_to_metadata(builder.get_binding_stats())

    if not chunk_metas:
        dataset_parquet = cfg.out_dir / "dataset.parquet"
        dataset_csv = cfg.out_dir / "dataset.csv"
        features_npz = cfg.out_dir / "features_npz.npz"
        if dataset_parquet.exists():
            dataset_parquet.unlink()
        if dataset_csv.exists():
            dataset_csv.unlink()
        empty_df = polars_module.DataFrame()
        empty_df.write_parquet(str(dataset_parquet))
        _write_dataset_csv(empty_df, cfg, dataset_csv=dataset_csv)
        np.savez(
            features_npz,
            X_train=np.empty((0, 0), dtype=np.float32),
            X_val=np.empty((0, 0), dtype=np.float32),
            feature_names=np.array([], dtype=np.str_),
        )
        metadata = build_dataset_metadata_from_windows(
            dataset_id=cfg.dataset_id,
            vintage_policy=cfg.vintage_policy,
            vintage_as_of=vintage_as_of,
            build_ts=build_ts,
            overall_start=None,
            overall_end=None,
            train_window_end=None,
            validation_window_start=None,
            macro_observation_counts={},
            capability_flags=capability_flags,
            market_bindings=binding_metadata,
            target_semantics=target_semantics_metadata,
        )
        write_dataset_metadata(metadata, out_dir=cfg.out_dir)
        validation_result = DatasetValidationResult(
            row_count=0,
            positive_rate=None,
            feature_coverage={},
            macro_columns_present=(),
            macro_observation_counts={},
        )
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return (
            BuildResult(
                dataset_parquet=dataset_parquet,
                dataset_csv=dataset_csv,
                features_npz=features_npz,
                feature_names=[],
                feature_set_id=None,
                metadata=metadata,
            ),
            validation_result,
        )

    total_rows = sum(meta.rows for meta in chunk_metas)
    positive_sum = sum(meta.positives for meta in chunk_metas)
    macro_totals: dict[str, int] = {}
    for meta in chunk_metas:
        for macro, count in meta.macro_counts.items():
            macro_totals[macro] = macro_totals.get(macro, 0) + count
    if cfg.macro_series_ids:
        for macro in cfg.macro_series_ids:
            macro_totals.setdefault(macro, 0)
    overall_start = min((meta.ts_start for meta in chunk_metas if meta.ts_start is not None), default=None)
    overall_end = max((meta.ts_end for meta in chunk_metas if meta.ts_end is not None), default=None)

    dataset_parquet = cfg.out_dir / "dataset.parquet"
    dataset_csv = cfg.out_dir / "dataset.csv"
    features_npz = cfg.out_dir / "features_npz.npz"

    if dataset_parquet.exists():
        dataset_parquet.unlink()
    if dataset_csv.exists():
        dataset_csv.unlink()
    if features_npz.exists():
        features_npz.unlink()
    write_csv = _resolve_write_csv(cfg, total_rows)
    sample_rows = max(int(cfg.csv_sample_rows), 0)
    sample_path = dataset_csv.with_name("dataset_sample.csv")
    if not write_csv and sample_path.exists():
        sample_path.unlink()

    validation_cfg = prepare_validation_config(
        cfg=cfg,
        validation_cfg=cfg.validation,
    )

    pq_writer: pq.ParquetWriter | None = None
    csv_header_written = False
    sample_header_written = False
    sample_remaining = sample_rows
    feature_names: list[str] | None = None
    non_null_counts: dict[str, int] | None = None
    feature_writer: _StreamingFeatureWriter | None = None
    offset = 0
    train_rows = int(total_rows * 0.8)
    train_window_end: datetime | None = None
    validation_window_start: datetime | None = None

    def _column_to_float32(column: Any) -> FloatArray:
        values = column.to_numpy(zero_copy_only=False)
        if isinstance(values, np.ma.MaskedArray):
            filled = np.where(values.mask, np.nan, values.data)
            return cast(FloatArray, np.asarray(filled, dtype=np.float32))
        return cast(FloatArray, np.asarray(values, dtype=np.float32))

    for meta in chunk_metas:
        merge_started = time.perf_counter()
        merge_peak_rss: float | None = None
        merge_peak_rss = _update_rss_peak(merge_peak_rss)
        parquet_file = pq.ParquetFile(str(meta.path))
        for batch in parquet_file.iter_batches():
            batch_rows = batch.num_rows
            if batch_rows == 0:
                continue
            df_batch = polars_module.from_arrow(batch)
            if df_batch.is_empty():
                continue
            df_batch = df_batch.with_columns(
                polars_module.arange(offset, offset + df_batch.height, eager=True).alias("time_index"),
            )
            df_batch = builder._add_known_future_features_polars(df_batch)
            merge_peak_rss = _update_rss_peak(merge_peak_rss)

            if feature_names is None:
                feature_names = _infer_feature_columns(df_batch)
                feature_writer = _StreamingFeatureWriter(
                    out_path=features_npz,
                    feature_names=feature_names,
                    total_rows=total_rows,
                    cutoff=train_rows,
                )
                non_null_counts = dict.fromkeys(feature_names, 0)

            if train_rows > 0 and train_window_end is None and offset <= train_rows - 1 < offset + df_batch.height:
                idx = train_rows - 1 - offset
                if 0 <= idx < df_batch.height and "timestamp" in df_batch.columns:
                    train_window_end = cast(datetime, df_batch[idx, "timestamp"])
            if validation_window_start is None and offset <= train_rows < offset + df_batch.height:
                idx_val = train_rows - offset
                if 0 <= idx_val < df_batch.height and "timestamp" in df_batch.columns:
                    validation_window_start = cast(datetime, df_batch[idx_val, "timestamp"])

            table = df_batch.to_arrow()
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(dataset_parquet), table.schema, compression="zstd")

            if feature_names:
                assert non_null_counts is not None
                batch_values: list[FloatArray] = []
                for name in feature_names:
                    try:
                        column = table.column(name)
                    except Exception:
                        column = None
                    if column is None:
                        batch_values.append(cast(FloatArray, np.zeros(batch_rows, dtype=np.float32)))
                        continue
                    non_null_counts[name] += int(batch_rows - column.null_count)
                    batch_values.append(_column_to_float32(column))
                if feature_writer is not None:
                    values = cast(FloatArray, np.column_stack(batch_values))
                    feature_writer.append(
                        values,
                        global_offset=offset,
                    )

            pq_writer.write_table(table)

            if write_csv:
                mode = "w" if not csv_header_written else "a"
                with open(dataset_csv, mode, newline="") as csv_handle:
                    df_batch.write_csv(csv_handle, include_header=not csv_header_written)
                csv_header_written = True
            elif sample_remaining > 0:
                sample_df = df_batch.head(sample_remaining)
                if not sample_df.is_empty():
                    mode = "w" if not sample_header_written else "a"
                    with open(sample_path, mode, newline="") as csv_handle:
                        sample_df.write_csv(csv_handle, include_header=not sample_header_written)
                    sample_remaining -= sample_df.height
                    sample_header_written = True

            offset += df_batch.height
            merge_peak_rss = _update_rss_peak(merge_peak_rss)

        meta.path.unlink(missing_ok=True)
        _DATASET_BUILD_CHUNK_SECONDS.labels(stage="merge").observe(
            time.perf_counter() - merge_started,
        )
        if merge_peak_rss is not None:
            _DATASET_BUILD_RSS_GAUGE.labels(stage="merge").set(merge_peak_rss)

    if pq_writer is not None:
        pq_writer.close()
    if feature_writer is not None:
        feature_writer.finalize()

    shutil.rmtree(chunk_dir, ignore_errors=True)

    positive_rate = (
        positive_sum / total_rows if total_rows and binary_target_col is not None else None
    )
    if feature_names and non_null_counts is not None and total_rows > 0:
        feature_coverage = {
            name: non_null_counts[name] / total_rows
            for name in feature_names
        }
    elif feature_names and non_null_counts is not None:
        feature_coverage = dict.fromkeys(feature_names, 0.0)
    else:
        feature_coverage = {}

    validation_result = _validate_aggregated_dataset(
        config=validation_cfg,
        total_rows=total_rows,
        positive_rate=positive_rate,
        feature_coverage=feature_coverage,
        macro_counts=macro_totals,
    )

    logger.info(
        "Dataset validation succeeded",
        rows=validation_result.row_count,
        positive_rate=validation_result.positive_rate,
    )

    dataset_id = cfg.dataset_id

    metadata = build_dataset_metadata_from_windows(
        dataset_id=dataset_id,
        vintage_policy=cfg.vintage_policy,
        vintage_as_of=vintage_as_of,
        build_ts=build_ts,
        overall_start=overall_start,
        overall_end=overall_end,
        train_window_end=train_window_end,
        validation_window_start=validation_window_start,
        macro_observation_counts=macro_totals,
        capability_flags=capability_flags,
        market_bindings=binding_metadata,
        target_semantics=target_semantics_metadata,
    )

    write_dataset_metadata(metadata, out_dir=cfg.out_dir)

    return (
        BuildResult(
            dataset_parquet=dataset_parquet,
            dataset_csv=dataset_csv,
            features_npz=features_npz,
            feature_names=list(feature_names or []),
            feature_set_id=None,
            metadata=metadata,
        ),
        validation_result,
    )


def build_tft_dataset(
    cfg: DatasetBuildConfig,
    *,
    data_store: DataStoreFacadeProtocol | None = None,
) -> BuildResult:
    """
    Build a TFT dataset and persist artifacts under ``cfg.out_dir``.

    Parameters
    ----------
    cfg : DatasetBuildConfig
        Dataset configuration describing the output location and build options.
    data_store : DataStoreFacadeProtocol, optional
        Canonical DataStore for loading raw market data. When provided, the
        builder reads OHLCV data via the store and falls back to the catalog
        only if the store returns no rows.

    Returns
    -------
    BuildResult
        Artifact bundle with dataset paths and metadata.
    """
    from ml._imports import check_ml_dependencies
    from ml._imports import pl

    if pl is None:
        check_ml_dependencies(["polars"])  # Guard for cold-path environment

    # Defer ParquetDataCatalog import to avoid import-time heavy deps here
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = ParquetDataCatalog(path=str(cfg.data_dir))

    build_ts = datetime.now(tz=UTC)

    fred_parquet_path = resolve_fred_parquet_path(cfg)
    alfred_start_str, alfred_end_str = derive_alfred_range(cfg)
    alfred_window_days = resolve_alfred_window_days(cfg)
    refresh_macro_artifacts_if_needed(
        cfg=cfg,
        data_store=data_store,
        fred_parquet_path=fred_parquet_path,
        alfred_start=alfred_start_str,
        alfred_end=alfred_end_str,
        alfred_window_days=alfred_window_days,
        logger=logger,
    )

    fred_path_str = str(fred_parquet_path) if cfg.include_macro else None

    vintage_as_of = normalize_vintage_as_of(cfg.vintage_as_of)

    descriptor_map = load_market_feed_descriptors().as_mapping()
    resolved_bindings = resolve_market_dataset_bindings(
        symbols=cfg.symbols,
        instrument_ids=cfg.instrument_ids,
        market_dataset_id=cfg.market_dataset_id,
        market_inputs=cfg.market_inputs,
        descriptors=descriptor_map,
    )

    from ml.data.tft_dataset_builder import TFTDatasetBuilder

    builder = TFTDatasetBuilder(
        catalog=catalog,
        symbols=cfg.symbols,
        instrument_ids=cfg.instrument_ids,
        include_macro=cfg.include_macro,
        macro_lag_days=cfg.macro_lag_days,
        include_micro=cfg.include_micro,
        include_l2=cfg.include_l2,
        micro_cache_policy=cfg.micro_cache_policy,
        l2_cache_policy=cfg.l2_cache_policy,
        include_events=cfg.include_events,
        include_calendar=cfg.include_calendar,
        include_earnings=cfg.include_earnings,
        earnings_lag_days=cfg.earnings_lag_days,
        fred_path=fred_path_str,
        vintage_base_dir=cfg.fred_vintage_dir,
        events_base_dir=cfg.events_base_dir,
        student_mode=cfg.student_mode,
        micro_base_dir=str(cfg.micro_base_dir) if cfg.micro_base_dir is not None else None,
        l2_base_dir=str(cfg.l2_base_dir) if cfg.l2_base_dir is not None else None,
        macro_series_ids=cfg.macro_series_ids,
        vintage_policy=cfg.vintage_policy,
        vintage_as_of=vintage_as_of,
        data_store=data_store,
        market_dataset_id=cfg.market_dataset_id,
        market_bindings=resolved_bindings,
        include_macro_revisions=cfg.include_macro_revisions,
        macro_revision_mode=cfg.macro_revision_mode,
        macro_revision_windows=cfg.macro_revision_windows,
    )
    capability_flags = capability_flags_from_builder(builder)
    target_semantics = _resolve_target_semantics(cfg)
    from ml.training.datasets.target_generator import build_target_semantics_metadata

    target_semantics_metadata = build_target_semantics_metadata(target_semantics)
    primary_horizon_minutes = (
        int(target_semantics.horizons[0].minutes)
        if target_semantics.horizons
        else None
    )

    chunk_mode = bool(cfg.chunk_days > 0 and cfg.start and cfg.end)
    if chunk_mode:
        build_result, _ = _build_dataset_chunked(
            builder=builder,
            cfg=cfg,
            vintage_as_of=vintage_as_of,
            build_ts=build_ts,
            target_semantics=target_semantics,
        )
        return build_result

    from ml._imports import HAS_POLARS
    from ml._imports import pl

    assert HAS_POLARS and pl is not None

    df_any = builder.build_training_dataset(
        target_semantics=target_semantics,
        lookback_periods=cfg.lookback_periods,
        use_polars=True,
        start=cfg.start,
        end=cfg.end,
    )
    if isinstance(df_any, pl.DataFrame):
        dataset_df = df_any
    else:  # pragma: no cover - fallback path
        from ml._imports import HAS_PANDAS
        from ml._imports import pd

        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover
        assert pd is not None
        dataset_df = pl.from_pandas(df_any)

    # Validate dataset before persisting artifacts
    validation_cfg = prepare_validation_config(
        cfg=cfg,
        validation_cfg=cfg.validation,
    )
    validation_result = validate_dataset(dataset_df, config=validation_cfg)
    logger.info(
        "Dataset validation succeeded",
        rows=validation_result.row_count,
        positive_rate=validation_result.positive_rate,
    )

    # Persist dataset artifacts
    dataset_parquet = cfg.out_dir / "dataset.parquet"
    dataset_csv = cfg.out_dir / "dataset.csv"
    dataset_df.write_parquet(str(dataset_parquet))
    _write_dataset_csv(dataset_df, cfg, dataset_csv=dataset_csv)

    # Build feature matrix artifacts without materialising the entire dataset in memory
    feature_names = _infer_feature_columns(dataset_df)
    df_sorted = dataset_df.sort("time_index") if "time_index" in dataset_df.columns else dataset_df.clone()
    cutoff = int(df_sorted.height * 0.8) if df_sorted.height > 0 else 0
    features_npz = cfg.out_dir / "features_npz.npz"
    _write_feature_npz_from_polars(
        df_sorted,
        feature_names,
        out_path=features_npz,
        cutoff=cutoff,
    )

    metadata = _compute_dataset_metadata(
        df_sorted,
        cutoff,
        cfg.vintage_policy,
        vintage_as_of,
        build_ts,
        getattr(cfg, "dataset_id", None),
        getattr(validation_result, "macro_observation_counts", {}),
        target_semantics_metadata,
    )
    binding_metadata = _binding_stats_to_metadata(builder.get_binding_stats())
    metadata = replace(metadata, market_bindings=binding_metadata, capability_flags=capability_flags)
    _validate_dataset_metadata(metadata)
    write_dataset_metadata(metadata, out_dir=cfg.out_dir)

    # Optional feature registration
    feature_set_id: str | None = None
    if cfg.register_features:
        if cfg.feature_registry_dir is None:
            raise ValueError("feature_registry_dir is required when register_features=True")
        from ml.data.feature_manifest_export import FeatureExportConfig
        from ml.data.feature_manifest_export import export_feature_manifest
        from ml.registry.base import DataRequirements
        from ml.registry.feature_registry import FeatureRole

        role_map = {
            "teacher": FeatureRole.TEACHER,
            "student": FeatureRole.STUDENT,
            "inference_support": FeatureRole.INFERENCE_SUPPORT,
        }
        data_req = DataRequirements.L1_ONLY if not cfg.include_l2 else DataRequirements.L1_L2
        reg_cfg = FeatureExportConfig(
            registry_path=Path(cfg.feature_registry_dir),
            role=role_map.get(cfg.feature_role, FeatureRole.TEACHER),
            data_requirements=data_req,
        )
        flags = {
            "include_macro": capability_flags["include_macro"],
            "include_micro": capability_flags["include_micro"],
            "include_l2": capability_flags["include_l2"],
            "include_earnings": capability_flags["include_earnings"],
            "horizon_minutes": primary_horizon_minutes,
            "lookback_periods": cfg.lookback_periods,
        }
        feature_set_id = export_feature_manifest(
            feature_names=feature_names,
            feature_dtypes=["float32"] * len(feature_names),
            flags=flags,
            cfg=reg_cfg,
        )

    # Optional dataset event emission (best-effort; cold path only)
    if cfg.emit_dataset_events:
        emit_dataset_build_event(
            df=df_sorted,
            dataset_id=cfg.dataset_id,
            symbols=cfg.symbols,
            include_macro=cfg.include_macro,
            include_micro=cfg.include_micro,
            include_l2=cfg.include_l2,
            lookback_periods=cfg.lookback_periods,
            primary_horizon_minutes=primary_horizon_minutes,
        )

    return BuildResult(
        dataset_parquet=dataset_parquet,
        dataset_csv=dataset_csv,
        features_npz=features_npz,
        feature_names=feature_names,
        feature_set_id=feature_set_id,
        metadata=metadata,
    )


__all__ = [
    "BuildResult",
    "DatasetBuildConfig",
    "build_tft_dataset",
    "compute_dataset_pipeline_signature",
]
