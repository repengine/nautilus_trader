"""
Streaming data utilities for TFT teacher training.

This module exposes helpers to derive metadata from large parquet datasets and
yield Lightning-compatible batches without materialising the entire dataset into
memory. It operates in two passes:

1. `collect_streaming_metadata` performs a low-memory scan to gather feature
   statistics, categorical vocabularies, and shard boundaries.
2. `TFTStreamingDataset` replays shards lazily, emitting batches that follow the
   structure produced by ``TimeSeriesDataSet.to_dataloader``.

The implementation focuses on minimising peak RSS by loading one shard at a time
and immediately converting sequences to torch tensors. Categorical lookups and
numeric scaling rely on metadata statistics captured during the first pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import DTypeLike

from ml import _imports as _ml_imports
from ml._imports import check_ml_dependencies
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


HAS_PANDAS: bool = getattr(_ml_imports, "HAS_PANDAS", False)
HAS_PYARROW: bool = getattr(_ml_imports, "HAS_PYARROW", False)
HAS_TORCH: bool = getattr(_ml_imports, "HAS_TORCH", False)
pa = cast(Any, getattr(_ml_imports, "pa", None))
torch = cast(Any, getattr(_ml_imports, "torch", None))

if not HAS_PYARROW:
    try:  # pragma: no cover - optional dependency resolution
        import pyarrow as _pa_runtime

        pa = _pa_runtime
        HAS_PYARROW = True
    except ImportError:
        pa = None
        HAS_PYARROW = False


try:  # pragma: no cover - platform dependent
    import resource as _resource
except ImportError:  # pragma: no cover - windows fallback
    _resource = cast(Any, None)


if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:  # pragma: no cover - torch is optional
    try:
        from torch import Tensor as TorchTensor  # type: ignore[attr-defined,assignment]
    except Exception:  # pragma: no cover - fallback when torch missing
        TorchTensor = Any  # type: ignore[assignment]

BatchItem = tuple[dict[str, TorchTensor], tuple[TorchTensor, None]]


if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import IterableDataset as TorchIterableDataset

    StreamDataLoader = TorchDataLoader[BatchItem]
    StreamIterableDatasetBase = TorchIterableDataset[BatchItem]
else:
    StreamDataLoader = Any
    StreamIterableDatasetBase = (
        torch.utils.data.IterableDataset if torch is not None else object
    )


if HAS_PYARROW:
    from pyarrow import compute as pa_compute  # pragma: no cover - heavy import
    from pyarrow import dataset as pa_dataset  # pragma: no cover - heavy import
else:  # pragma: no cover - handled at runtime via dependency guard
    pa_compute = cast(Any, None)
    pa_dataset = cast(Any, None)

if TYPE_CHECKING:
    try:
        from pyarrow.dataset import Dataset as ArrowDataset
    except Exception:  # pragma: no cover - typing fallback
        ArrowDataset = Any
else:
    ArrowDataset = Any


__all__ = [
    "PhaseOneFeatureSignals",
    "RunningStats",
    "StreamingLimitSummary",
    "TFTShardIndex",
    "TFTStreamingConfig",
    "TFTStreamingDataModule",
    "TFTStreamingDataset",
    "TFTStreamingMetadata",
    "TFTStreamingPreprocessor",
    "TFTStreamingSummary",
    "apply_streaming_limits",
    "build_streaming_dataloader",
    "collect_streaming_metadata",
    "count_sequences",
    "filter_metadata_by_instruments",
    "instrument_row_counts",
    "is_within_shard_budget",
    "split_metadata_by_row_fraction",
    "split_metadata_by_time",
    "summarize_metadata",
]


_DEFAULT_SHARD_ROW_BUDGET: Final[int] = 2_000_000

_METADATA_SHARDS_COUNTER = get_counter(
    "ml_tft_streaming_metadata_shards_total",
    "Total number of shards discovered during TFT streaming metadata scans.",
)
_METADATA_ROWS_COUNTER = get_counter(
    "ml_tft_streaming_metadata_rows_total",
    "Total number of rows discovered during TFT streaming metadata scans.",
)
_METADATA_MAX_ROWS_HIST = get_histogram(
    "ml_tft_streaming_metadata_max_shard_rows",
    "Distribution of maximum rows per shard discovered during metadata scans.",
    buckets=(
        1_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
    ),
)
_ITERATION_SHARDS_COUNTER = get_counter(
    "ml_tft_streaming_iterated_shards_total",
    "Total number of TFT streaming shards iterated during training/inference.",
)
_ITERATION_ROWS_HIST = get_histogram(
    "ml_tft_streaming_shard_rows",
    "Rows per shard during TFT streaming iteration.",
    buckets=(
        1_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
    ),
)
_RSS_GAUGE = get_gauge(
    "ml_tft_streaming_rss_mb",
    "Observed RSS (MB) during TFT streaming metadata scans and iteration.",
    labelnames=("stage",),
)
_SKIPPED_SHARDS_COUNTER = get_counter(
    "ml_tft_streaming_skipped_shards_total",
    "Number of shards skipped because of streaming limits.",
)
_SKIPPED_ROWS_COUNTER = get_counter(
    "ml_tft_streaming_skipped_rows_total",
    "Number of rows skipped because of streaming limits.",
)
_SKIPPED_SEQUENCES_COUNTER = get_counter(
    "ml_tft_streaming_skipped_sequences_total",
    "Estimated number of sequences skipped due to streaming limits.",
)

logger = logging.getLogger(__name__)


def _combine_chunks(array: Any) -> Any:
    if hasattr(array, "combine_chunks"):
        return array.combine_chunks()
    return array


def _cast_arrow_array(array: Any, dtype: DTypeLike) -> Any:
    if pa is None:
        return array
    if pa_compute is None:
        return array
    try:
        numpy_dtype = np.dtype(dtype)
        if np.issubdtype(numpy_dtype, np.float32):
            target = pa.float32()
        elif np.issubdtype(numpy_dtype, np.floating):
            target = pa.float64()
        elif np.issubdtype(numpy_dtype, np.integer):
            target = pa.int64()
        else:
            return array
        return pa_compute.cast(array, target, safe=False)
    except Exception:
        return array


def _arrow_to_numpy(
    array: Any,
    *,
    dtype: DTypeLike,
    fill_value: float | None = None,
) -> np.ndarray:
    numpy_dtype = np.dtype(dtype)
    if array is None:
        return np.empty(0, dtype=numpy_dtype)
    combined = _combine_chunks(array)
    casted = _cast_arrow_array(combined, dtype)
    np_array = casted.to_numpy(zero_copy_only=False)
    if isinstance(np_array, np.ma.MaskedArray):
        if fill_value is None and np.issubdtype(numpy_dtype, np.floating):
            fill = np.nan
        elif fill_value is None:
            fill = 0.0
        else:
            fill = float(fill_value)
        filled = np_array.filled(fill)
        return np.asarray(filled, dtype=numpy_dtype)
    if not isinstance(np_array, np.ndarray):
        return np.asarray(np_array, dtype=numpy_dtype)
    if np_array.dtype != numpy_dtype:
        return np_array.astype(numpy_dtype, copy=False)
    return np_array


def _arrow_strings(array: Any) -> list[str]:
    if array is None:
        return []
    combined = _combine_chunks(array)
    try:
        values = combined.to_pylist()
    except Exception:
        values = []
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        result.append(str(value))
    return result


def _arrow_strings_preserve_length(array: Any) -> list[str]:
    if array is None:
        return []
    combined = _combine_chunks(array)
    try:
        values = combined.to_pylist()
    except Exception:
        values = []
    result: list[str] = []
    for value in values:
        result.append("" if value is None else str(value))
    return result


def _update_running_stats(stats: RunningStats, values: np.ndarray) -> RunningStats:
    if values.ndim == 0:
        return stats
    if values.size == 0:
        return stats
    return stats.update(values.astype(np.float64, copy=False))


def _get_column(batch: Any, name: str) -> Any:
    schema = getattr(batch, "schema", None)
    if schema is None:
        return None
    try:
        index = schema.get_field_index(name)
    except Exception:
        index = -1
    if index is None or index < 0:
        return None
    return batch.column(index)


def _get_table_column(table: Any, name: str) -> Any:
    if table is None:
        return None
    try:
        return table.column(name)
    except Exception:
        return None


@dataclass(slots=True, frozen=True)
class RunningStats:
    """Numerically stable running mean/variance tracker (Welford's algorithm)."""

    count: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def update(self, values: np.ndarray) -> RunningStats:
        """Return a new ``RunningStats`` updated with ``values``."""
        if values.size == 0:
            return self
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return self

        chunk_count = int(finite.size)
        chunk_mean = float(finite.mean())
        chunk_m2 = float(np.sum((finite - chunk_mean) ** 2))

        if self.count == 0:
            return RunningStats(count=chunk_count, mean=chunk_mean, m2=chunk_m2)

        total_count = self.count + chunk_count
        if total_count <= 0:
            return RunningStats(count=0, mean=0.0, m2=0.0)

        delta = chunk_mean - self.mean
        mean = self.mean + delta * chunk_count / total_count
        m2 = self.m2 + chunk_m2 + (delta**2) * self.count * chunk_count / total_count
        return RunningStats(count=total_count, mean=mean, m2=m2)


@dataclass(slots=True, frozen=True)
class TFTShardIndex:
    """Descriptor for a contiguous shard of rows belonging to one instrument."""

    shard_id: str
    instrument_id: str
    row_start: int
    row_end: int
    row_count: int
    time_start: int
    time_end: int


@dataclass(slots=True)
class _InstrumentShardState:
    """Accumulator tracking the current shard bounds for an instrument."""

    row_start: int
    row_count: int
    time_start: int
    time_end: int


@dataclass(slots=True, frozen=True)
class PhaseOneFeatureSignals:
    """Categorised Phase 1 feature families derived from metadata annotations."""

    macro_delta_columns: tuple[str, ...] = ()
    calendar_lag_columns: tuple[str, ...] = ()
    clustering_tag_columns: tuple[str, ...] = ()
    context_feature_columns: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, list[str]]:
        """Return a JSON-serialisable representation of the feature signals."""
        return {
            "macro_delta_columns": [str(value) for value in self.macro_delta_columns],
            "calendar_lag_columns": [str(value) for value in self.calendar_lag_columns],
            "clustering_tag_columns": [str(value) for value in self.clustering_tag_columns],
            "context_feature_columns": [str(value) for value in self.context_feature_columns],
        }

    def is_empty(self) -> bool:
        """Return ``True`` when no feature families were supplied."""
        return not (
            self.macro_delta_columns
            or self.calendar_lag_columns
            or self.clustering_tag_columns
            or self.context_feature_columns
        )


@dataclass(slots=True, frozen=True)
class TFTStreamingMetadata:
    """Aggregated metadata derived from the parquet scan."""

    shard_indices: tuple[TFTShardIndex, ...]
    numeric_stats: dict[str, RunningStats]
    categorical_vocab: dict[str, tuple[str, ...]]
    instrument_row_counts: dict[str, int]
    phase_one_signals: PhaseOneFeatureSignals = field(default_factory=PhaseOneFeatureSignals)


@dataclass(slots=True, frozen=True)
class TFTStreamingSummary:
    """Summary statistics for streaming metadata."""

    total_shards: int
    total_rows: int
    max_shard_rows: int


def summarize_metadata(metadata: TFTStreamingMetadata) -> TFTStreamingSummary:
    """Return aggregate statistics for logging and metrics."""
    total_shards = len(metadata.shard_indices)
    total_rows = sum(metadata.instrument_row_counts.values())
    max_shard_rows = max((shard.row_count for shard in metadata.shard_indices), default=0)
    return TFTStreamingSummary(
        total_shards=total_shards,
        total_rows=total_rows,
        max_shard_rows=max_shard_rows,
    )


def is_within_shard_budget(
    summary: TFTStreamingSummary,
    shard_row_budget: int,
    *,
    tolerance_pct: float = 0.05,
) -> bool:
    """Return True when the maximum shard size is within the tolerated budget."""
    budget = max(1, shard_row_budget)
    allowed = int(budget * (1.0 + max(0.0, tolerance_pct)))
    return summary.max_shard_rows <= allowed


def _current_rss_mb() -> float | None:
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
    except Exception:  # pragma: no cover - platform dependent
        return None
    rss_value = float(getattr(usage, "ru_maxrss", 0.0))
    if rss_value <= 0.0:
        return None
    # Linux reports kilobytes; macOS reports bytes. Heuristic threshold.
    if rss_value > 1e8:
        return rss_value / (1024.0 * 1024.0)
    return rss_value / 1024.0


class TFTStreamingPreprocessor:
    """First-stage parquet scanner collecting metadata for streaming loader."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        feature_names: Iterable[str],
        categorical_columns: Iterable[str],
        numeric_columns: Iterable[str],
        group_id_col: str,
        time_index_col: str,
        shard_row_budget: int = _DEFAULT_SHARD_ROW_BUDGET,
    ) -> None:
        if not HAS_PYARROW or pa_dataset is None:
            check_ml_dependencies(["pyarrow"])
            raise ImportError("PyArrow dataset module unavailable")
        self._parquet_path = Path(parquet_path)
        self._feature_names = tuple(feature_names)
        self._categorical_columns = tuple(categorical_columns)
        self._numeric_columns = tuple(numeric_columns)
        self._group_id_col = group_id_col
        self._time_index_col = time_index_col
        self._shard_row_budget = max(1, int(shard_row_budget))

    def build_metadata(self) -> TFTStreamingMetadata:
        if pa_dataset is None:
            check_ml_dependencies(["pyarrow"])
            raise ImportError("PyArrow dataset module unavailable")

        dataset = pa_dataset.dataset(str(self._parquet_path), format="parquet")
        selected_columns = set(self._feature_names)
        selected_columns.add(self._group_id_col)
        selected_columns.add(self._time_index_col)
        scanner = dataset.scanner(columns=sorted(selected_columns))

        numeric_stats: dict[str, RunningStats] = {
            name: RunningStats(count=0, mean=0.0, m2=0.0) for name in self._numeric_columns
        }
        categorical_vocab: dict[str, set[str]] = {
            name: set() for name in self._categorical_columns
        }

        shard_indices: list[TFTShardIndex] = []
        instrument_row_counts: dict[str, int] = {}

        shard_counter = 0
        shard_states: dict[str, _InstrumentShardState] = {}

        def _finalize_shard(instrument: str, state: _InstrumentShardState) -> None:
            nonlocal shard_counter
            if state.row_count <= 0:
                return
            shard_id = f"shard_{shard_counter:05d}"
            shard_indices.append(
                TFTShardIndex(
                    shard_id=shard_id,
                    instrument_id=instrument,
                    row_start=state.row_start,
                    row_end=state.row_start + state.row_count,
                    row_count=state.row_count,
                    time_start=state.time_start,
                    time_end=state.time_end,
                ),
            )
            shard_counter += 1

        for batch in scanner.to_batches():
            group_column = _get_column(batch, self._group_id_col)
            time_column = _get_column(batch, self._time_index_col)
            if group_column is None or time_column is None:
                continue

            group_values = _arrow_strings_preserve_length(group_column)
            time_values = _arrow_to_numpy(time_column, dtype=np.int64, fill_value=0)

            for column in self._categorical_columns:
                cat_column = _get_column(batch, column)
                if cat_column is None:
                    continue
                categorical_vocab[column].update(_arrow_strings(cat_column))

            for column in self._numeric_columns:
                num_column = _get_column(batch, column)
                if num_column is None:
                    continue
                values = _arrow_to_numpy(num_column, dtype=np.float64)
                numeric_stats[column] = _update_running_stats(numeric_stats[column], values)

            for instrument, time_value in zip(group_values, time_values.tolist()):
                if not instrument:
                    continue
                previous_total = instrument_row_counts.get(instrument, 0)
                current_total = previous_total + 1
                instrument_row_counts[instrument] = current_total
                time_int = int(time_value)

                state = shard_states.get(instrument)
                if state is None:
                    state = _InstrumentShardState(
                        row_start=previous_total,
                        row_count=0,
                        time_start=time_int,
                        time_end=time_int,
                    )
                    shard_states[instrument] = state

                state.row_count += 1
                state.time_end = time_int

                if state.row_count >= self._shard_row_budget:
                    _finalize_shard(instrument, state)
                    shard_states.pop(instrument, None)

        for instrument, state in shard_states.items():
            _finalize_shard(instrument, state)

        vocab_final: dict[str, tuple[str, ...]] = {
            column: tuple(sorted(values)) for column, values in categorical_vocab.items()
        }
        metadata = TFTStreamingMetadata(
            shard_indices=tuple(shard_indices),
            numeric_stats=numeric_stats,
            categorical_vocab=vocab_final,
            instrument_row_counts=instrument_row_counts,
            phase_one_signals=PhaseOneFeatureSignals(),
        )
        summary = summarize_metadata(metadata)
        _METADATA_SHARDS_COUNTER.inc(summary.total_shards)
        _METADATA_ROWS_COUNTER.inc(summary.total_rows)
        _METADATA_MAX_ROWS_HIST.observe(float(summary.max_shard_rows))
        rss_mb = _current_rss_mb()
        if rss_mb is not None:
            _RSS_GAUGE.labels(stage="metadata").set(rss_mb)
        logger.info(
            "tft streaming metadata collected",
            extra={
                "total_shards": summary.total_shards,
                "total_rows": summary.total_rows,
                "max_shard_rows": summary.max_shard_rows,
            },
        )
        return metadata


def collect_streaming_metadata(
    parquet_path: Path,
    *,
    feature_names: Iterable[str],
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str],
    group_id_col: str,
    time_index_col: str,
    shard_row_budget: int = _DEFAULT_SHARD_ROW_BUDGET,
    phase_one_signals: PhaseOneFeatureSignals | None = None,
) -> TFTStreamingMetadata:
    """Return metadata derived from a parquet dataset scan."""
    preprocessor = TFTStreamingPreprocessor(
        parquet_path,
        feature_names=feature_names,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        group_id_col=group_id_col,
        time_index_col=time_index_col,
        shard_row_budget=shard_row_budget,
    )
    metadata = preprocessor.build_metadata()
    if phase_one_signals is None:
        return metadata
    return replace(metadata, phase_one_signals=phase_one_signals)


@dataclass(slots=True, frozen=True)
class TFTStreamingConfig:
    """Configuration governing streaming dataloader behaviour."""

    time_idx_col: str
    group_id_col: str
    target_col: str
    static_categoricals: tuple[str, ...]
    static_reals: tuple[str, ...]
    time_varying_known_reals: tuple[str, ...]
    time_varying_unknown_reals: tuple[str, ...]
    max_encoder_length: int
    max_prediction_length: int
    batch_size: int
    drop_last: bool = False
    shuffle_shards: bool = False
    seed: int | None = None
    num_workers: int = 0
    max_total_rows: int | None = None
    max_total_sequences: int | None = None
    max_shards: int | None = None
    include_macro: bool = False
    include_calendar: bool = False
    include_events: bool = False
    include_earnings: bool = False
    include_micro: bool = False
    include_l2: bool = False
    include_macro_revisions: bool = False
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    macro_lag_days: int = 1
    earnings_lag_days: int = 1
    events_notice_minutes: int = 0
    phase_one_signals: PhaseOneFeatureSignals = field(default_factory=PhaseOneFeatureSignals)

    def __post_init__(self) -> None:
        for name, value in (
            ("macro_lag_days", self.macro_lag_days),
            ("earnings_lag_days", self.earnings_lag_days),
            ("events_notice_minutes", self.events_notice_minutes),
        ):
            if int(value) < 0:
                msg = f"{name} must be non-negative (received {value})"
                raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class StreamingLimitSummary:
    """Counts describing how much work was skipped when enforcing streaming limits."""

    skipped_shards: int = 0
    skipped_rows: int = 0
    skipped_sequences: int = 0
    total_instrument_rows: dict[str, int] = field(default_factory=dict)
    selected_instrument_rows: dict[str, int] = field(default_factory=dict)
    skipped_instrument_rows: dict[str, int] = field(default_factory=dict)
    total_instrument_sequences: dict[str, int] = field(default_factory=dict)
    selected_instrument_sequences: dict[str, int] = field(default_factory=dict)
    skipped_instrument_sequences: dict[str, int] = field(default_factory=dict)


def _estimate_sequences_for_shard(
    shard: TFTShardIndex,
    encoder_length: int,
    decoder_length: int,
) -> int:
    """Return the number of encoder/decoder sequences available within a shard."""
    available = shard.row_count - encoder_length - decoder_length + 1
    return available if available > 0 else 0


def count_sequences(
    metadata: TFTStreamingMetadata,
    config: TFTStreamingConfig,
) -> int:
    """Return the number of encoder/decoder sequences available across metadata shards."""
    encoder_length = config.max_encoder_length
    decoder_length = config.max_prediction_length
    total = 0
    for shard in metadata.shard_indices:
        total += _estimate_sequences_for_shard(shard, encoder_length, decoder_length)
    return total


def apply_streaming_limits(
    metadata: TFTStreamingMetadata,
    config: TFTStreamingConfig,
) -> tuple[TFTStreamingMetadata, StreamingLimitSummary]:
    """Return metadata trimmed to respect global limits on shard/row/sequence counts."""
    return _limit_metadata_for_streaming(metadata, config)


def _limit_metadata_for_streaming(
    metadata: TFTStreamingMetadata,
    config: TFTStreamingConfig,
) -> tuple[TFTStreamingMetadata, StreamingLimitSummary]:
    """Return metadata trimmed to respect global limits on shard/row/sequence counts."""
    encoder_len = config.max_encoder_length
    decoder_len = config.max_prediction_length
    total_row_counts: dict[str, int] = {}
    total_sequence_counts: dict[str, int] = {}
    instruments_in_metadata: set[str] = set()
    for shard in metadata.shard_indices:
        instrument = shard.instrument_id
        instruments_in_metadata.add(instrument)
        total_row_counts[instrument] = total_row_counts.get(instrument, 0) + shard.row_count
        if instrument not in total_sequence_counts:
            total_sequence_counts[instrument] = 0
        total_sequence_counts[instrument] += _estimate_sequences_for_shard(
            shard,
            encoder_len,
            decoder_len,
        )
    for instrument, count in metadata.instrument_row_counts.items():
        instruments_in_metadata.add(instrument)
        total_row_counts.setdefault(instrument, int(count))
        total_sequence_counts.setdefault(instrument, 0)

    limits_active = any(
        value is not None
        for value in (config.max_total_rows, config.max_total_sequences, config.max_shards)
    )
    if not limits_active:
        selected_rows_map = dict(metadata.instrument_row_counts)
        if not selected_rows_map and total_row_counts:
            selected_rows_map = dict(total_row_counts)
        selected_rows_filtered = {
            instrument: count for instrument, count in selected_rows_map.items() if count > 0
        }
        selected_sequences_filtered = {
            instrument: count for instrument, count in total_sequence_counts.items() if count > 0
        }
        summary = StreamingLimitSummary(
            skipped_shards=0,
            skipped_rows=0,
            skipped_sequences=0,
            total_instrument_rows=dict(sorted(total_row_counts.items())),
            selected_instrument_rows=dict(sorted(selected_rows_filtered.items())),
            skipped_instrument_rows={},
            total_instrument_sequences=dict(sorted(total_sequence_counts.items())),
            selected_instrument_sequences=dict(sorted(selected_sequences_filtered.items())),
            skipped_instrument_sequences={},
        )
        return metadata, summary

    instrument_ids = sorted(instruments_in_metadata)
    max_rows = config.max_total_rows
    max_sequences = config.max_total_sequences
    max_shards = config.max_shards

    shards_by_instrument: dict[str, deque[TFTShardIndex]] = {}
    for instrument in instrument_ids:
        shards_by_instrument[instrument] = deque()
    for shard in metadata.shard_indices:
        instrument = shard.instrument_id
        shards_by_instrument.setdefault(instrument, deque()).append(shard)

    for instrument, shard_queue in list(shards_by_instrument.items()):
        sorted_queue = deque(sorted(shard_queue, key=lambda s: (s.time_start, s.row_start, s.shard_id)))
        shards_by_instrument[instrument] = sorted_queue

    instrument_order = sorted(
        shards_by_instrument.keys(),
        key=lambda inst: (
            shards_by_instrument[inst][0].time_start if shards_by_instrument[inst] else float("inf"),
            inst,
        ),
    )

    selected: list[TFTShardIndex] = []
    rows_accum = 0
    sequences_accum = 0
    shards_accum = 0
    skipped_rows = 0
    skipped_sequences = 0
    skipped_shards = 0

    selected_rows_by_instrument = dict.fromkeys(instrument_ids, 0)
    skipped_rows_by_instrument = dict.fromkeys(instrument_ids, 0)
    selected_sequences_by_instrument = dict.fromkeys(instrument_ids, 0)
    skipped_sequences_by_instrument = dict.fromkeys(instrument_ids, 0)

    def _pop_and_skip(instrument: str, shard: TFTShardIndex, shard_sequences: int) -> None:
        nonlocal skipped_shards, skipped_rows, skipped_sequences
        skipped_shards += 1
        skipped_rows += shard.row_count
        skipped_sequences += shard_sequences
        skipped_rows_by_instrument[instrument] += shard.row_count
        skipped_sequences_by_instrument[instrument] += shard_sequences

    while True:
        progress = False
        for instrument in instrument_order:
            queue = shards_by_instrument.get(instrument)
            if not queue:
                continue

            while queue:
                shard = queue[0]
                shard_sequences = _estimate_sequences_for_shard(shard, encoder_len, decoder_len)
                if shard_sequences <= 0:
                    queue.popleft()
                    _pop_and_skip(instrument, shard, shard_sequences)
                    continue
                projected_shards = shards_accum + 1
                projected_rows = rows_accum + shard.row_count
                projected_sequences = sequences_accum + shard_sequences
                limit_exceeded = (
                    (max_shards is not None and projected_shards > max_shards)
                    or (max_rows is not None and projected_rows > max_rows)
                    or (max_sequences is not None and projected_sequences > max_sequences)
                )
                if limit_exceeded:
                    queue.popleft()
                    _pop_and_skip(instrument, shard, shard_sequences)
                    continue

                queue.popleft()
                selected.append(shard)
                shards_accum = projected_shards
                rows_accum = projected_rows
                sequences_accum = projected_sequences
                selected_rows_by_instrument[instrument] += shard.row_count
                selected_sequences_by_instrument[instrument] += shard_sequences
                progress = True
                break

        if all(not queue for queue in shards_by_instrument.values()):
            break
        if not progress:
            break

    for instrument, queue in shards_by_instrument.items():
        while queue:
            shard = queue.popleft()
            shard_sequences = _estimate_sequences_for_shard(shard, encoder_len, decoder_len)
            _pop_and_skip(instrument, shard, shard_sequences)

    selected_rows_filtered = {
        instrument: count for instrument, count in selected_rows_by_instrument.items() if count > 0
    }
    selected_sequences_filtered = {
        instrument: count for instrument, count in selected_sequences_by_instrument.items() if count > 0
    }
    skipped_rows_filtered = {
        instrument: total_row_counts.get(instrument, 0) - selected_rows_by_instrument.get(instrument, 0)
        for instrument in instrument_ids
        if (total_row_counts.get(instrument, 0) - selected_rows_by_instrument.get(instrument, 0)) > 0
    }
    skipped_sequences_filtered = {
        instrument: total_sequence_counts.get(instrument, 0)
        - selected_sequences_by_instrument.get(instrument, 0)
        for instrument in instrument_ids
        if (
            total_sequence_counts.get(instrument, 0)
            - selected_sequences_by_instrument.get(instrument, 0)
        )
        > 0
    }

    limited_metadata = TFTStreamingMetadata(
        shard_indices=tuple(selected),
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=selected_rows_filtered,
        phase_one_signals=metadata.phase_one_signals,
    )
    summary = StreamingLimitSummary(
        skipped_shards=skipped_shards,
        skipped_rows=skipped_rows,
        skipped_sequences=skipped_sequences,
        total_instrument_rows=dict(sorted(total_row_counts.items())),
        selected_instrument_rows=dict(sorted(selected_rows_filtered.items())),
        skipped_instrument_rows=dict(sorted(skipped_rows_filtered.items())),
        total_instrument_sequences=dict(sorted(total_sequence_counts.items())),
        selected_instrument_sequences=dict(sorted(selected_sequences_filtered.items())),
        skipped_instrument_sequences=dict(sorted(skipped_sequences_filtered.items())),
    )
    return limited_metadata, summary


class TFTStreamingDataset(StreamIterableDatasetBase):
    """Iterable dataset yielding TFT-compatible batches from parquet shards."""

    def __init__(
        self,
        parquet_path: Path,
        metadata: TFTStreamingMetadata,
        config: TFTStreamingConfig,
    ) -> None:
        if not HAS_TORCH:
            check_ml_dependencies(["torch"])
            raise ImportError("PyTorch is required for streaming dataset")
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])
            raise ImportError("pandas is required for streaming dataset")
        self._parquet_path = Path(parquet_path)
        self._metadata = metadata
        self._config = config
        self._shard_log_counter = 0

        self._categorical_maps: dict[str, dict[str, int]] = {}
        for name, vocab in metadata.categorical_vocab.items():
            mapping = {value: idx for idx, value in enumerate(vocab)}
            mapping.setdefault("__UNK__", len(mapping))
            self._categorical_maps[name] = mapping
        if config.group_id_col in self._categorical_maps:
            sample_mapping = [(str(key), int(value)) for key, value in list(self._categorical_maps[config.group_id_col].items())[:5]]
            logger.info(
                "streaming dataset group mapping sample (column=%s, entries=%s)",
                config.group_id_col,
                sample_mapping,
                extra={
                    "group_column": config.group_id_col,
                    "mapping_sample": sample_mapping,
                },
            )

        self._numeric_stats = metadata.numeric_stats
        target_stats = metadata.numeric_stats.get(config.target_col)
        if target_stats is None:
            raise ValueError(f"Missing numeric statistics for target column '{config.target_col}'")
        self._target_stats = target_stats

        self._encoder_cont_columns: tuple[str, ...] = (
            config.static_reals
            + config.time_varying_known_reals
            + config.time_varying_unknown_reals
        )
        self._decoder_cont_columns: tuple[str, ...] = (
            config.static_reals
            + config.time_varying_known_reals
            + config.time_varying_unknown_reals
        )
        self._static_cat_columns = config.static_categoricals

        self._columns = tuple(
            sorted(
                {
                    config.time_idx_col,
                    config.group_id_col,
                    config.target_col,
                    *config.static_categoricals,
                    *config.static_reals,
                    *config.time_varying_known_reals,
                    *config.time_varying_unknown_reals,
                },
            ),
        )

        self._num_batches = self._estimate_batches()

    def _estimate_batches(self) -> int:
        encoder_len = self._config.max_encoder_length
        decoder_len = self._config.max_prediction_length
        effective_sequences = 0
        for shard in self._metadata.shard_indices:
            available = shard.row_count - encoder_len - decoder_len + 1
            if available > 0:
                effective_sequences += available
        if effective_sequences <= 0 or self._config.batch_size <= 0:
            return 0
        if self._config.drop_last:
            return effective_sequences // self._config.batch_size
        return (effective_sequences + self._config.batch_size - 1) // self._config.batch_size

    def __len__(self) -> int:  # pragma: no cover - lightweight helper
        return self._num_batches

    def _resolve_shards(
        self,
        *,
        worker_id: int,
        num_workers: int,
        order: np.ndarray,
    ) -> list[TFTShardIndex]:
        shard_count = len(self._metadata.shard_indices)
        if shard_count == 0:
            return []

        shard_indices = order
        if num_workers > 1:
            shard_indices = shard_indices[worker_id::num_workers]
        return [self._metadata.shard_indices[index] for index in shard_indices.tolist()]

    def __iter__(self) -> Iterator[BatchItem]:
        worker_info = torch.utils.data.get_worker_info()
        base_seed = self._config.seed or 0
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = max(1, worker_info.num_workers) if worker_info is not None else 1

        if not HAS_PYARROW or pa_dataset is None:
            check_ml_dependencies(["pyarrow"])
            raise ImportError("PyArrow dataset module unavailable")
        dataset_module = pa_dataset
        if dataset_module is None:
            check_ml_dependencies(["pyarrow"])
            raise ImportError("PyArrow dataset module unavailable")
        dataset_obj = cast(ArrowDataset, dataset_module.dataset(str(self._parquet_path), format="parquet"))

        shard_count = len(self._metadata.shard_indices)
        if shard_count == 0:
            return

        if self._config.shuffle_shards:
            shard_order = np.random.default_rng(base_seed).permutation(shard_count)
        else:
            shard_order = np.arange(shard_count)

        shards = self._resolve_shards(
            worker_id=worker_id,
            num_workers=num_workers,
            order=shard_order,
        )
        if not shards:
            return

        for shard in shards:
            if self._shard_log_counter < 5:
                logger.info(
                    "streaming dataset iterating shard (instrument=%s, time_start=%s, time_end=%s, rows=%s)",
                    shard.instrument_id,
                    shard.time_start,
                    shard.time_end,
                    shard.row_count,
                    extra={
                        "instrument_id": shard.instrument_id,
                        "time_start": shard.time_start,
                        "time_end": shard.time_end,
                        "row_count": shard.row_count,
                    },
                )
                self._shard_log_counter += 1
            yield from self._iter_shard_batches(dataset_obj, shard)

    def _iter_shard_batches(
        self,
        dataset: ArrowDataset,
        shard: TFTShardIndex,
    ) -> Iterator[BatchItem]:
        dataset_module = pa_dataset
        pa_module = pa
        if dataset_module is None or pa_module is None:
            check_ml_dependencies(["pyarrow"])
            raise ImportError("PyArrow dataset module unavailable")
        scanner = dataset.scanner(
            columns=list(self._columns),
            filter=(
                (dataset_module.field(self._config.group_id_col) == pa_module.scalar(shard.instrument_id))
                & (dataset_module.field(self._config.time_idx_col) >= pa_module.scalar(shard.time_start))
                & (dataset_module.field(self._config.time_idx_col) <= pa_module.scalar(shard.time_end))
            ),
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return

        if pa_compute is not None:
            try:
                sort_indices = pa_compute.sort_indices(
                    table,
                    sort_keys=[(self._config.time_idx_col, "ascending")],
                )
                table = table.take(sort_indices)
            except Exception as sort_exc:
                logger.debug(
                    "Failed to sort shard rows for instrument %s",
                    shard.instrument_id,
                    exc_info=True,
                    extra={"error": repr(sort_exc)},
                )

        target_column = _get_table_column(table, self._config.target_col)
        time_column = _get_table_column(table, self._config.time_idx_col)
        group_column = _get_table_column(table, self._config.group_id_col)
        if target_column is None or time_column is None or group_column is None:
            return

        target_array = _arrow_to_numpy(target_column, dtype=np.float32, fill_value=0.0)
        time_array = _arrow_to_numpy(time_column, dtype=np.int64, fill_value=0)
        total_rows = int(target_array.shape[0])

        encoder_len = self._config.max_encoder_length
        decoder_len = self._config.max_prediction_length
        if total_rows < encoder_len + decoder_len:
            return
        _ITERATION_SHARDS_COUNTER.inc()
        _ITERATION_ROWS_HIST.observe(float(total_rows))
        rss_mb = _current_rss_mb()
        if rss_mb is not None:
            _RSS_GAUGE.labels(stage="iteration").set(rss_mb)
        logger.debug(
            "tft streaming shard materialised",
            extra={
                "instrument_id": shard.instrument_id,
                "rows": total_rows,
                "time_start": shard.time_start,
                "time_end": shard.time_end,
            },
        )

        numeric_arrays: dict[str, np.ndarray] = {}
        for column in self._encoder_cont_columns:
            column_data = _get_table_column(table, column)
            if column_data is None:
                numeric_arrays[column] = np.zeros(total_rows, dtype=np.float32)
                continue
            array = _arrow_to_numpy(column_data, dtype=np.float32, fill_value=0.0)
            stats = self._numeric_stats.get(column)
            if stats is not None and stats.count > 0:
                std = (stats.variance**0.5) if stats.variance > 0 else 0.0
                mean = stats.mean
                if std > 0:
                    array = (array - mean) / std
                else:
                    array = array - mean
            array = np.nan_to_num(array, nan=0.0)
            numeric_arrays[column] = array

        static_reals_values: dict[str, float] = {}
        for column in self._config.static_reals:
            column_data = _get_table_column(table, column)
            if column_data is not None and len(column_data) > 0:
                try:
                    raw_value = column_data[0].as_py()
                except Exception:
                    raw_value = None
                stats = self._numeric_stats.get(column)
                value = float(raw_value) if raw_value is not None else 0.0
                if stats is not None and stats.count > 0:
                    std = (stats.variance**0.5) if stats.variance > 0 else 0.0
                    mean = stats.mean
                    if std > 0:
                        value = (value - mean) / std
                    else:
                        value = value - mean
                static_reals_values[column] = float(value)
            else:
                static_reals_values[column] = 0.0

        static_cat_codes: dict[str, int] = {}
        for column in self._static_cat_columns:
            mapping = self._categorical_maps.get(column)
            if mapping is None:
                continue
            column_data = _get_table_column(table, column)
            raw_value = "__UNK__"
            if column_data is not None and len(column_data) > 0:
                try:
                    value = column_data[0].as_py()
                    raw_value = "__UNK__" if value is None else str(value)
                except Exception:
                    raw_value = "__UNK__"
            static_cat_codes[column] = mapping.get(raw_value, mapping.get("__UNK__", 0))

        cat_vector: np.ndarray | None = None
        if self._static_cat_columns:
            cat_vector = np.asarray(
                [static_cat_codes.get(column, 0) for column in self._static_cat_columns],
                dtype=np.int64,
            )

        group_mapping = self._categorical_maps.get(self._config.group_id_col, {})
        group_code = group_mapping.get(shard.instrument_id, group_mapping.get("__UNK__", 0))
        if shard.instrument_id not in group_mapping:
            logger.info(
                "streaming dataset instrument missing from categorical mapping; using fallback",
                extra={
                    "instrument_id": shard.instrument_id,
                    "fallback_code": int(group_code),
                },
            )

        batch_encoder_cont: list[np.ndarray] = []
        batch_decoder_cont: list[np.ndarray] = []
        batch_encoder_cat: list[np.ndarray] = []
        batch_decoder_cat: list[np.ndarray] = []
        batch_encoder_target: list[np.ndarray] = []
        batch_decoder_target: list[np.ndarray] = []
        batch_encoder_lengths: list[int] = []
        batch_decoder_lengths: list[int] = []
        batch_groups: list[np.ndarray] = []
        batch_target_scale: list[np.ndarray] = []
        batch_decoder_time: list[np.ndarray] = []
        batch_decoder_group_ids: list[np.ndarray] = []

        batch_size = max(1, self._config.batch_size)
        encoder_len = self._config.max_encoder_length
        decoder_len = self._config.max_prediction_length

        target_mean = self._target_stats.mean
        target_std = (self._target_stats.variance**0.5) if self._target_stats.variance > 0 else 1.0
        if target_std == 0.0:
            target_std = 1.0

        for current in range(encoder_len, total_rows - decoder_len + 1):
            enc_start = current - encoder_len
            enc_end = current
            dec_start = current
            dec_end = current + decoder_len

            encoder_target = target_array[enc_start:enc_end]
            decoder_target = target_array[dec_start:dec_end]
            if encoder_target.shape[0] != encoder_len or decoder_target.shape[0] != decoder_len:
                continue

            encoder_cont = np.empty(
                (encoder_len, len(self._encoder_cont_columns)),
                dtype=np.float32,
            )
            decoder_cont = np.empty(
                (decoder_len, len(self._decoder_cont_columns)),
                dtype=np.float32,
            )

            idx = 0
            for column in self._config.static_reals:
                value = static_reals_values.get(column, 0.0)
                encoder_cont[:, idx] = value
                decoder_cont[:, idx] = value
                idx += 1
            for column in self._config.time_varying_known_reals:
                values = numeric_arrays[column]
                encoder_cont[:, idx] = values[enc_start:enc_end]
                decoder_cont[:, idx] = values[dec_start:dec_end]
                idx += 1
            for column in self._config.time_varying_unknown_reals:
                values = numeric_arrays[column]
                encoder_cont[:, idx] = values[enc_start:enc_end]
                decoder_cont[:, idx] = values[dec_start:dec_end]
                idx += 1

            if cat_vector is not None:
                encoder_cat = np.tile(cat_vector, (encoder_len, 1))
                decoder_cat = np.tile(cat_vector, (decoder_len, 1))
            else:
                encoder_cat = np.empty((encoder_len, 0), dtype=np.int64)
                decoder_cat = np.empty((decoder_len, 0), dtype=np.int64)

            batch_encoder_cont.append(encoder_cont)
            batch_decoder_cont.append(decoder_cont)
            batch_encoder_cat.append(encoder_cat)
            batch_decoder_cat.append(decoder_cat)
            batch_encoder_target.append(encoder_target.astype(np.float32, copy=False))
            batch_decoder_target.append(decoder_target.astype(np.float32, copy=False))
            batch_encoder_lengths.append(encoder_len)
            batch_decoder_lengths.append(decoder_len)
            batch_groups.append(np.array([[group_code]], dtype=np.int64))
            batch_target_scale.append(np.array([target_mean, target_std], dtype=np.float32))
            batch_decoder_time.append(time_array[dec_start:dec_end].astype(np.int64, copy=False))
            batch_decoder_group_ids.append(
                np.full((decoder_len,), group_code, dtype=np.int64)
            )

            if len(batch_encoder_cont) == batch_size:
                yield self._build_batch(
                    batch_encoder_cont,
                    batch_decoder_cont,
                    batch_encoder_cat,
                    batch_decoder_cat,
                    batch_encoder_target,
                    batch_decoder_target,
                    batch_encoder_lengths,
                    batch_decoder_lengths,
                    batch_groups,
                    batch_target_scale,
                    batch_decoder_time,
                    batch_decoder_group_ids,
                )
                batch_encoder_cont.clear()
                batch_decoder_cont.clear()
                batch_encoder_cat.clear()
                batch_decoder_cat.clear()
                batch_encoder_target.clear()
                batch_decoder_target.clear()
                batch_encoder_lengths.clear()
                batch_decoder_lengths.clear()
                batch_groups.clear()
                batch_target_scale.clear()
                batch_decoder_time.clear()
                batch_decoder_group_ids.clear()

        if not self._config.drop_last and batch_encoder_cont:
            yield self._build_batch(
                batch_encoder_cont,
                batch_decoder_cont,
                batch_encoder_cat,
                batch_decoder_cat,
                batch_encoder_target,
                batch_decoder_target,
                batch_encoder_lengths,
                batch_decoder_lengths,
                batch_groups,
                batch_target_scale,
                batch_decoder_time,
                batch_decoder_group_ids,
            )

    def _build_batch(
        self,
        encoder_cont: list[np.ndarray],
        decoder_cont: list[np.ndarray],
        encoder_cat: list[np.ndarray],
        decoder_cat: list[np.ndarray],
        encoder_target: list[np.ndarray],
        decoder_target: list[np.ndarray],
        encoder_lengths: list[int],
        decoder_lengths: list[int],
        groups: list[np.ndarray],
        target_scales: list[np.ndarray],
        decoder_time: list[np.ndarray],
        decoder_group_ids: list[np.ndarray],
    ) -> BatchItem:
        batch_inputs: dict[str, TorchTensor] = {}
        batch_inputs["encoder_cont"] = torch.from_numpy(np.stack(encoder_cont, axis=0))
        batch_inputs["decoder_cont"] = torch.from_numpy(np.stack(decoder_cont, axis=0))

        if encoder_cat and encoder_cat[0].size > 0:
            batch_inputs["encoder_cat"] = torch.from_numpy(np.stack(encoder_cat, axis=0))
            batch_inputs["decoder_cat"] = torch.from_numpy(np.stack(decoder_cat, axis=0))
        else:
            batch_inputs["encoder_cat"] = torch.empty(
                len(encoder_cont),
                self._config.max_encoder_length,
                0,
                dtype=torch.int64,
            )
            batch_inputs["decoder_cat"] = torch.empty(
                len(encoder_cont),
                self._config.max_prediction_length,
                0,
                dtype=torch.int64,
            )

        batch_inputs["encoder_target"] = torch.from_numpy(np.stack(encoder_target, axis=0))
        decoder_target_tensor = torch.from_numpy(np.stack(decoder_target, axis=0))
        batch_inputs["decoder_target"] = decoder_target_tensor
        batch_inputs["encoder_lengths"] = torch.tensor(encoder_lengths, dtype=torch.int64)
        batch_inputs["decoder_lengths"] = torch.tensor(decoder_lengths, dtype=torch.int64)
        batch_inputs["groups"] = torch.from_numpy(np.vstack(groups))
        batch_inputs["target_scale"] = torch.from_numpy(np.stack(target_scales, axis=0))
        batch_inputs["decoder_time_idx"] = torch.from_numpy(
            np.stack(decoder_time, axis=0).astype(np.int64, copy=False)
        )
        batch_inputs["decoder_group_ids"] = torch.from_numpy(np.stack(decoder_group_ids, axis=0))

        outputs = (decoder_target_tensor, None)
        return batch_inputs, outputs


def build_streaming_dataloader(
    parquet_path: Path,
    metadata: TFTStreamingMetadata,
    config: TFTStreamingConfig,
    *,
    metadata_is_limited: bool = False,
    limit_summary: StreamingLimitSummary | None = None,
) -> StreamDataLoader:
    """Return a DataLoader that replays parquet shards lazily."""
    if torch is None:
        check_ml_dependencies(["torch"])
        raise ImportError("PyTorch is required to build streaming dataloaders")
    if metadata_is_limited:
        limited_metadata = metadata
        summary = limit_summary or StreamingLimitSummary()
    else:
        limited_metadata, summary = apply_streaming_limits(metadata, config)
    if summary.skipped_shards > 0:
        _SKIPPED_SHARDS_COUNTER.inc(summary.skipped_shards)
        if summary.skipped_rows > 0:
            _SKIPPED_ROWS_COUNTER.inc(summary.skipped_rows)
        if summary.skipped_sequences > 0:
            _SKIPPED_SEQUENCES_COUNTER.inc(summary.skipped_sequences)
        logger.warning(
            "tft streaming metadata limited",
            extra={
                "skipped_shards": summary.skipped_shards,
                "skipped_rows": summary.skipped_rows,
                "skipped_sequences": summary.skipped_sequences,
                "max_shards": config.max_shards,
                "max_rows": config.max_total_rows,
                "max_sequences": config.max_total_sequences,
            },
        )
    if not limited_metadata.shard_indices:
        raise RuntimeError(
            "Streaming limits filtered out all shards. Relax constraints or adjust dataset window.",
        )

    dataset = TFTStreamingDataset(parquet_path, limited_metadata, config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
    )
    return cast(StreamDataLoader, loader)


def instrument_row_counts(metadata: TFTStreamingMetadata) -> dict[str, int]:
    """Aggregate total row counts per instrument from shard metadata."""
    return dict(metadata.instrument_row_counts)


def split_metadata_by_time(
    metadata: TFTStreamingMetadata,
    cutoff_time: int,
) -> tuple[TFTStreamingMetadata, TFTStreamingMetadata]:
    """Split metadata into train/validation shards at ``cutoff_time`` (inclusive)."""
    train_shards: list[TFTShardIndex] = []
    val_shards: list[TFTShardIndex] = []
    for shard in metadata.shard_indices:
        if shard.time_end <= cutoff_time:
            train_shards.append(shard)
        elif shard.time_start >= cutoff_time:
            val_shards.append(shard)
        else:
            val_shards.append(shard)

    train_metadata = TFTStreamingMetadata(
        shard_indices=tuple(train_shards),
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=metadata.instrument_row_counts,
    )
    val_metadata = TFTStreamingMetadata(
        shard_indices=tuple(val_shards),
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=metadata.instrument_row_counts,
    )
    return train_metadata, val_metadata


def filter_metadata_by_instruments(
    metadata: TFTStreamingMetadata,
    instruments: Iterable[str],
) -> TFTStreamingMetadata:
    """Return a metadata copy restricted to the provided instruments."""
    allowed = {instrument for instrument in instruments}
    filtered_shards = tuple(
        shard for shard in metadata.shard_indices if shard.instrument_id in allowed
    )
    filtered_counts = {
        instrument: metadata.instrument_row_counts[instrument]
        for instrument in allowed
        if instrument in metadata.instrument_row_counts
    }
    return TFTStreamingMetadata(
        shard_indices=filtered_shards,
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=filtered_counts,
    )


def split_metadata_by_row_fraction(
    metadata: TFTStreamingMetadata,
    train_fraction: float,
) -> tuple[TFTStreamingMetadata, TFTStreamingMetadata]:
    """
    Split shards by cumulative row count fraction while preserving per-instrument coverage.

    Shards are grouped by instrument and sorted chronologically. Each instrument
    receives training shards up to the requested fraction (clipped to [0, 1]) and
    retains at least one validation shard when possible. Instruments with a single
    shard are duplicated across splits so the worker observes them during both
    training and validation.
    """
    if not metadata.shard_indices:
        empty = TFTStreamingMetadata(
            shard_indices=(),
            numeric_stats=metadata.numeric_stats,
            categorical_vocab=metadata.categorical_vocab,
            instrument_row_counts=metadata.instrument_row_counts,
        )
        return empty, empty

    grouped_shards: dict[str, list[TFTShardIndex]] = defaultdict(list)
    for shard in metadata.shard_indices:
        grouped_shards[shard.instrument_id].append(shard)

    train_shards: list[TFTShardIndex] = []
    val_shards: list[TFTShardIndex] = []
    clamped_fraction = max(0.0, min(1.0, float(train_fraction)))

    for instrument, shards in grouped_shards.items():
        instrument_shards = sorted(shards, key=lambda shard: shard.time_start)
        instrument_rows = sum(shard.row_count for shard in instrument_shards)
        if instrument_rows <= 0:
            val_shards.extend(instrument_shards)
            continue
        target_rows = round(clamped_fraction * instrument_rows)
        target_rows = max(0, min(target_rows, instrument_rows))

        cumulative = 0
        instrument_train: list[TFTShardIndex] = []
        instrument_val: list[TFTShardIndex] = []

        for shard in instrument_shards:
            if cumulative < target_rows:
                instrument_train.append(shard)
            else:
                instrument_val.append(shard)
            cumulative += shard.row_count

        if not instrument_train and instrument_val:
            instrument_train.append(instrument_val[0])
        if not instrument_val and instrument_train:
            if len(instrument_train) > 1:
                instrument_val.append(instrument_train.pop())
            else:
                instrument_val.append(instrument_train[0])

        train_shards.extend(instrument_train)
        val_shards.extend(instrument_val)

    if not train_shards and val_shards:
        train_shards.append(val_shards[0])
    if not val_shards and train_shards:
        val_shards.append(train_shards[-1])

    train_metadata = TFTStreamingMetadata(
        shard_indices=tuple(train_shards),
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=metadata.instrument_row_counts,
    )
    val_metadata = TFTStreamingMetadata(
        shard_indices=tuple(val_shards),
        numeric_stats=metadata.numeric_stats,
        categorical_vocab=metadata.categorical_vocab,
        instrument_row_counts=metadata.instrument_row_counts,
    )
    return train_metadata, val_metadata


class TFTStreamingDataModule:
    """Expose train/validation/test streaming dataloaders for TFT training."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        config: TFTStreamingConfig,
        train_metadata: TFTStreamingMetadata,
        val_metadata: TFTStreamingMetadata | None = None,
        test_metadata: TFTStreamingMetadata | None = None,
        shuffle_train: bool = False,
        drop_last_train: bool = False,
    ) -> None:
        self._parquet_path = Path(parquet_path)
        self._base_config = config
        self._train_metadata = train_metadata
        self._val_metadata = val_metadata
        self._test_metadata = test_metadata
        self._shuffle_train = shuffle_train
        self._drop_last_train = drop_last_train

        self._train_loader: StreamDataLoader | None = None
        self._val_loader: StreamDataLoader | None = None
        self._test_loader: StreamDataLoader | None = None

    @property
    def train_metadata(self) -> TFTStreamingMetadata:
        return self._train_metadata

    @property
    def val_metadata(self) -> TFTStreamingMetadata | None:
        return self._val_metadata

    @property
    def test_metadata(self) -> TFTStreamingMetadata | None:
        return self._test_metadata

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self._train_metadata.shard_indices:
                train_config = replace(
                    self._base_config,
                    shuffle_shards=self._shuffle_train,
                    drop_last=self._drop_last_train,
                )
                self._train_loader = build_streaming_dataloader(
                    self._parquet_path,
                    self._train_metadata,
                    train_config,
                )
            if self._val_metadata is not None and self._val_metadata.shard_indices:
                val_config = replace(
                    self._base_config,
                    shuffle_shards=False,
                    drop_last=False,
                )
                self._val_loader = build_streaming_dataloader(
                    self._parquet_path,
                    self._val_metadata,
                    val_config,
                )

        if stage in (None, "test") and self._test_metadata is not None:
            if self._test_metadata.shard_indices:
                test_config = replace(
                    self._base_config,
                    shuffle_shards=False,
                    drop_last=False,
                )
                self._test_loader = build_streaming_dataloader(
                    self._parquet_path,
                    self._test_metadata,
                    test_config,
                )

    def train_dataloader(self) -> StreamDataLoader:
        if self._train_loader is None:
            raise RuntimeError("Streaming data module not initialised for training")
        return self._train_loader

    def val_dataloader(self) -> StreamDataLoader | None:
        return self._val_loader

    def test_dataloader(self) -> StreamDataLoader | None:
        return self._test_loader
