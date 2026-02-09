"""
Feature dataset restoration from parquet mirrors.

This module replays Tier-1 feature datasets (earnings, calendar releases, etc.)
from their parquet mirrors back into the SQL stores whenever coverage
classification marks buckets as ``RESTORE_FROM_CATALOG``. It keeps behaviour
config-driven and type-safe per AGENTS.md: restoration routes through the
standard :class:`ml.stores.data_store.DataStore` APIs so registry watermarks,
events, and validation continue to function exactly as the ingestion path.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from ml._imports import HAS_PANDAS
from ml._imports import HAS_PYARROW
from ml._imports import pa
from ml._imports import pd
from ml._imports import pq
from ml.common.metrics_bootstrap import get_counter
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.dataset_ids import FEATURE_VALUES_DATASET_ID
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID
from ml.config.events import Source
from ml.data.coverage.manager import BucketSpec
from ml.data.coverage.types import DAY_NS
from ml.data.coverage.types import GLOBAL_ENTITY_ID
from ml.ml_types import DataFrameLike
from ml.stores.base import FeatureData
from ml.stores.providers import ParquetCoverageSpec


if TYPE_CHECKING:
    from typing import Protocol

    from pandas import DataFrame as _PandasDataFrame
    from pandas import Series as _PandasSeries

    class _FeatureDatasetWriter(Protocol):
        def write_earnings_actual(
            self,
            *,
            ticker: str,
            period_end: str,
            filing_date: str,
            eps_diluted: float | None,
            revenue: float | None,
            ts_event: int,
            ts_init: int,
            eps_basic: float | None = ...,
            net_income: float | None = ...,
            operating_income: float | None = ...,
            shares_outstanding: int | None = ...,
            filing_type: str | None = ...,
            fiscal_year: int | None = ...,
            fiscal_quarter: int | None = ...,
            source: str = ...,
            run_id: str | None = ...,
        ) -> object: ...

        def write_earnings_estimate(
            self,
            *,
            ticker: str,
            estimate_date: str,
            period_end: str,
            eps_consensus: float | None,
            ts_event: int,
            ts_init: int,
            revenue_consensus: float | None = ...,
            num_analysts: int | None = ...,
            source: str = ...,
            run_id: str | None = ...,
        ) -> object: ...

        def write_ingestion(
            self,
            dataset_id: str,
            records: DataFrameLike | list[dict[str, object]],
            source: str,
            run_id: str,
            instrument_id: str | None = ...,
        ) -> object: ...

        def write_features(
            self,
            instrument_id: str,
            features: list[FeatureData],
            source: str = ...,
            run_id: str | None = ...,
        ) -> object: ...
else:
    _PandasDataFrame = Any  # type: ignore[assignment]
    _PandasSeries = Any  # type: ignore[assignment]
    _FeatureDatasetWriter = object

PANDAS = cast(Any, pd)


logger = logging.getLogger(__name__)

_fallback_counter = get_counter(
    "ml_fallback_activations_total",
    "Fallback activations",
    labelnames=("component", "level"),
)


def _is_global_entity(instrument_id: str) -> bool:
    return instrument_id.strip().upper() == GLOBAL_ENTITY_ID

_GENERAL_DATASETS: frozenset[str] = frozenset(
    {
        MACRO_RELEASES_DATASET_ID,
        MACRO_OBSERVATIONS_DATASET_ID,
        EVENTS_CALENDAR_DATASET_ID,
        MICRO_MINUTE_DATASET_ID,
        L2_MINUTE_DATASET_ID,
    },
)
SUPPORTED_FEATURE_DATASET_IDS: Final[frozenset[str]] = frozenset(
    {
        EARNINGS_ACTUALS_DATASET_ID,
        EARNINGS_ESTIMATES_DATASET_ID,
        FEATURE_VALUES_DATASET_ID,
        *tuple(_GENERAL_DATASETS),
    },
)
_SUPPORTED_DATASETS = SUPPORTED_FEATURE_DATASET_IDS

_MICRO_FEATURE_COLUMNS: tuple[str, ...] = (
    "midprice",
    "spread_bps",
    "quote_imbalance",
    "trade_imbalance",
    "realized_vol",
)
_L2_METRIC_PREFIXES: tuple[str, ...] = ("depth_imbalance", "dwp_bps", "bid_slope", "ask_slope")
_L2_FEATURE_COLUMNS: tuple[str, ...] = ("midprice", "spread_bps", "microprice_bps") + tuple(
    f"{prefix}_top{level}"
    for prefix in _L2_METRIC_PREFIXES
    for level in (1, 3, 5, 10)
)


@dataclass(frozen=True, slots=True)
class FeatureRestoreResult:
    """
    Summary of a feature dataset restoration run.
    """

    datasets_processed: int
    instruments_processed: int
    buckets_requested: int
    buckets_restored: int
    rows_written: int
    failures: dict[str, str]

    @property
    def success(self) -> bool:
        """Return True when no failures were recorded."""
        return not self.failures


@dataclass(frozen=True, slots=True)
class _InstrumentRestoreResult:
    rows: int
    buckets: int
    missing: set[int]


class FeatureCoverageRestorer:
    """
    Replay feature datasets from parquet mirrors into SQL stores.
    """

    def __init__(
        self,
        *,
        db_connection: str,
        parquet_specs: Mapping[str, ParquetCoverageSpec],
        writer_factory: Callable[[str], _FeatureDatasetWriter] | None = None,
    ) -> None:
        if not db_connection:
            msg = "db_connection must be provided for feature restoration"
            raise ValueError(msg)
        if not parquet_specs:
            msg = "parquet_specs cannot be empty for feature restoration"
            raise ValueError(msg)
        if not (HAS_PANDAS and PANDAS is not None):
            msg = "pandas is required for feature restoration because parquet frames need filtering"
            raise RuntimeError(msg)
        self._db_connection = db_connection
        self._parquet_specs = dict(parquet_specs)
        self._writer_factory = writer_factory
        self._writer: _FeatureDatasetWriter | None = None

    def restore(self, bucket_specs: Sequence[BucketSpec]) -> FeatureRestoreResult:
        """
        Replay missing buckets from parquet mirrors.
        """
        if not bucket_specs:
            return FeatureRestoreResult(
                datasets_processed=0,
                instruments_processed=0,
                buckets_requested=0,
                buckets_restored=0,
                rows_written=0,
                failures={},
            )
        dataset_groups: dict[str, list[BucketSpec]] = defaultdict(list)
        for spec in bucket_specs:
            dataset_groups[spec.dataset_id].append(spec)

        buckets_requested = len(bucket_specs)
        datasets_processed = 0
        instruments_processed = 0
        buckets_restored = 0
        rows_written = 0
        failures: dict[str, str] = {}
        activated_metrics: set[str] = set()

        for dataset_id, specs in dataset_groups.items():
            parquet_spec = self._parquet_specs.get(dataset_id)
            if parquet_spec is None:
                logger.debug(
                    "feature_restore.spec_missing",
                    extra={"dataset_id": dataset_id},
                )
                failures[dataset_id] = "parquet_spec_missing"
                continue
            writer = self._resolve_writer(dataset_id)
            if writer is None:
                failures[dataset_id] = "writer_unsupported"
                continue

            instrument_map: dict[str, list[BucketSpec]] = defaultdict(list)
            for spec in specs:
                instrument_id = spec.instrument_id.strip()
                if instrument_id:
                    instrument_map[instrument_id].append(spec)
            if not instrument_map:
                continue

            datasets_processed += 1
            for instrument_id, instrument_specs in instrument_map.items():
                instruments_processed += 1
                try:
                    result = self._restore_instrument(
                        dataset_id=dataset_id,
                        parquet_spec=parquet_spec,
                        writer=writer,
                        instrument_id=instrument_id,
                        specs=instrument_specs,
                    )
                except Exception as exc:
                    logger.warning(
                        "feature_restore.instrument_failed",
                        exc_info=True,
                        extra={
                            "dataset_id": dataset_id,
                            "instrument_id": instrument_id,
                        },
                    )
                    failures[f"{dataset_id}:{instrument_id}"] = str(exc)
                    continue

                if result.rows > 0 and dataset_id not in activated_metrics:
                    _fallback_counter.labels(
                        component="feature_coverage_restorer",
                        level=dataset_id,
                    ).inc()
                    activated_metrics.add(dataset_id)

                buckets_restored += result.buckets
                rows_written += result.rows
                missing = result.missing
                if missing:
                    failures[f"{dataset_id}:{instrument_id}"] = f"missing_buckets:{sorted(missing)}"

        return FeatureRestoreResult(
            datasets_processed=datasets_processed,
            instruments_processed=instruments_processed,
            buckets_requested=buckets_requested,
            buckets_restored=buckets_restored,
            rows_written=rows_written,
            failures=failures,
        )

    def _restore_instrument(
        self,
        *,
        dataset_id: str,
        parquet_spec: ParquetCoverageSpec,
        writer: _FeatureDatasetWriter,
        instrument_id: str,
        specs: Sequence[BucketSpec],
    ) -> _InstrumentRestoreResult:
        bucket_targets: set[int] = {spec.bucket_index for spec in specs}
        restored_buckets: set[int] = set()
        rows_written = 0
        timestamp_field = parquet_spec.timestamp_field or "ts_event"
        global_entity = _is_global_entity(instrument_id)
        partition_files = parquet_spec.files_for_instrument(instrument_id)
        if not partition_files:
            return _InstrumentRestoreResult(rows=0, buckets=0, missing=bucket_targets)

        for path in partition_files:
            partition_path = path if isinstance(path, Path) else Path(path)
            frame = self._read_parquet(partition_path, timestamp_field=timestamp_field)
            if frame is None:
                continue
            filtered_frame = self._filter_partition_field(
                frame=frame,
                partition_field=parquet_spec.partition_field,
                instrument_id=instrument_id,
                global_entity=global_entity,
            )
            if filtered_frame.empty:
                continue
            ts_series: _PandasSeries = filtered_frame[timestamp_field].dropna()
            if ts_series.empty:
                continue
            bucket_series = (ts_series.astype("int64") // DAY_NS).astype("int64")
            mask = bucket_series.isin(bucket_targets)
            if not mask.any():
                continue
            filtered = filtered_frame.loc[mask].copy()
            filtered["_bucket_index"] = bucket_series[mask].to_numpy()
            raw_records = filtered.to_dict("records")
            records: list[dict[str, object]] = []
            for raw in raw_records:
                typed_record: dict[str, object] = {}
                for key, value in raw.items():
                    typed_record[str(key)] = value
                records.append(typed_record)
            if not records:
                continue
            general_bucket_indices: set[int] = set()
            if "_bucket_index" in filtered:
                bucket_values = filtered["_bucket_index"].tolist()
                for raw_bucket in bucket_values:
                    bucket_value = self._coerce_int(raw_bucket)
                    if bucket_value is not None:
                        general_bucket_indices.add(bucket_value)
            if dataset_id in _GENERAL_DATASETS:
                written_rows = self._write_general_dataset(
                    dataset_id=dataset_id,
                    writer=writer,
                    instrument_id=None if global_entity else instrument_id,
                    frame=filtered,
                )
                if written_rows > 0:
                    rows_written += written_rows
                    restored_buckets.update(general_bucket_indices)
                    if restored_buckets >= bucket_targets:
                        break
                continue
            if dataset_id == FEATURE_VALUES_DATASET_ID:
                written_rows = self._write_feature_values(
                    writer=writer,
                    instrument_id=instrument_id,
                    frame=filtered,
                    timestamp_field=timestamp_field,
                )
                if written_rows > 0:
                    rows_written += written_rows
                    restored_buckets.update(general_bucket_indices)
                    if restored_buckets >= bucket_targets:
                        break
                continue

            stop_after_buckets = dataset_id not in {
                EARNINGS_ACTUALS_DATASET_ID,
                EARNINGS_ESTIMATES_DATASET_ID,
            }
            for record in records:
                bucket_index_value = record.pop("_bucket_index", None)
                bucket_index = self._coerce_int(bucket_index_value)
                if bucket_index is None:
                    continue
                ts_event = self._coerce_int(record.get(timestamp_field))
                ts_init = self._coerce_int(record.get("ts_init"), default=ts_event)
                if ts_event is None or ts_init is None:
                    continue
                if dataset_id == EARNINGS_ACTUALS_DATASET_ID:
                    self._write_earnings_actual(writer, record, ts_event, ts_init)
                elif dataset_id == EARNINGS_ESTIMATES_DATASET_ID:
                    self._write_earnings_estimate(writer, record, ts_event, ts_init)
                else:  # pragma: no cover - guarded by writer resolution
                    continue
                restored_buckets.add(bucket_index)
                rows_written += 1
                if stop_after_buckets and restored_buckets >= bucket_targets:
                    break
            if stop_after_buckets and restored_buckets >= bucket_targets:
                break

        missing = bucket_targets - restored_buckets
        return _InstrumentRestoreResult(
            rows=rows_written,
            buckets=len(restored_buckets),
            missing=missing,
        )

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        scalar: float | int | str | None = None
        if isinstance(value, (int, float)):
            scalar = value
        elif isinstance(value, str):
            scalar = value.strip()
            if not scalar:
                return None
        else:
            return None
        try:
            number = float(scalar)
        except ValueError:
            return None
        if math.isnan(number):
            return None
        return number

    @staticmethod
    def _coerce_int(value: object, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if math.isnan(value):
                return default
            return int(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return default
            try:
                return int(float(trimmed))
            except ValueError:
                return default
        return default

    def _write_earnings_actual(
        self,
        writer: _FeatureDatasetWriter,
        record: dict[str, object],
        ts_event: int,
        ts_init: int,
    ) -> None:
        writer.write_earnings_actual(
            ticker=str(record.get("ticker", "")).strip(),
            period_end=str(record.get("period_end", "")).strip(),
            filing_date=str(record.get("filing_date", "")).strip(),
            eps_diluted=self._coerce_float(record.get("eps_diluted")),
            revenue=self._coerce_float(record.get("revenue")),
            ts_event=ts_event,
            ts_init=ts_init,
            eps_basic=self._coerce_float(record.get("eps_basic")),
            net_income=self._coerce_float(record.get("net_income")),
            operating_income=self._coerce_float(record.get("operating_income")),
            shares_outstanding=self._coerce_int(record.get("shares_outstanding")),
            filing_type=self._coerce_str(record.get("filing_type")),
            fiscal_year=self._coerce_int(record.get("fiscal_year")),
            fiscal_quarter=self._coerce_int(record.get("fiscal_quarter")),
            source=Source.BACKFILL.value,
            run_id="feature_catalog_restore",
        )

    def _write_earnings_estimate(
        self,
        writer: _FeatureDatasetWriter,
        record: dict[str, object],
        ts_event: int,
        ts_init: int,
    ) -> None:
        writer.write_earnings_estimate(
            ticker=str(record.get("ticker", "")).strip(),
            estimate_date=str(record.get("estimate_date", "")).strip(),
            period_end=str(record.get("period_end", "")).strip(),
            eps_consensus=self._coerce_float(record.get("eps_consensus")),
            ts_event=ts_event,
            ts_init=ts_init,
            revenue_consensus=self._coerce_float(record.get("revenue_consensus")),
            num_analysts=self._coerce_int(record.get("num_analysts")),
            source=Source.BACKFILL.value,
            run_id="feature_catalog_restore",
        )

    def _write_feature_values(
        self,
        *,
        writer: _FeatureDatasetWriter,
        instrument_id: str,
        frame: _PandasDataFrame,
        timestamp_field: str,
    ) -> int:
        """
        Restore computed FeatureStore values from a parquet mirror.

        Args:
            writer: DataStore-compatible writer instance.
            instrument_id: Instrument being restored.
            frame: Filtered parquet frame for the bucket range.
            timestamp_field: Column containing ts_event values.

        Returns:
            Number of rows written.
        """
        if frame.empty:
            return 0
        values_column = "values"
        rows: list[FeatureData] = []
        for raw in frame.to_dict("records"):
            feature_set_id = self._coerce_str(raw.get("feature_set_id"))
            row_instrument = self._coerce_str(raw.get("instrument_id")) or instrument_id
            ts_event = self._coerce_int(raw.get(timestamp_field))
            ts_init = self._coerce_int(raw.get("ts_init"), default=ts_event)
            if ts_event is None or ts_init is None:
                continue
            values = self._coerce_feature_values(raw.get(values_column))
            if values is None:
                continue
            payload = FeatureData(
                feature_set_id=feature_set_id,
                instrument_id=row_instrument,
                values=values,
                ts_event=ts_event,
                ts_init=ts_init,
                quality_flags=self._coerce_int(raw.get("quality_flags"), default=0) or 0,
            )
            rows.append(payload)
        if not rows:
            return 0
        writer.write_features(
            instrument_id=instrument_id,
            features=rows,
            source=Source.BACKFILL.value,
            run_id="feature_catalog_restore",
        )
        return len(rows)

    def _write_general_dataset(
        self,
        *,
        dataset_id: str,
        writer: _FeatureDatasetWriter,
        instrument_id: str | None,
        frame: _PandasDataFrame,
    ) -> int:
        if frame.empty:
            return 0
        if dataset_id != EVENTS_CALENDAR_DATASET_ID and not instrument_id:
            return 0
        prepared: _PandasDataFrame | None
        if dataset_id == MACRO_RELEASES_DATASET_ID:
            prepared = self._prepare_macro_release_frame(frame, instrument_id or "")
        elif dataset_id == MACRO_OBSERVATIONS_DATASET_ID:
            prepared = self._prepare_macro_observation_frame(frame, instrument_id or "")
        elif dataset_id == EVENTS_CALENDAR_DATASET_ID:
            prepared = self._prepare_events_frame(frame)
        elif dataset_id == MICRO_MINUTE_DATASET_ID:
            prepared = self._prepare_micro_frame(frame, instrument_id or "")
        elif dataset_id == L2_MINUTE_DATASET_ID:
            prepared = self._prepare_l2_frame(frame, instrument_id or "")
        else:  # pragma: no cover - guarded by _GENERAL_DATASETS
            return 0
        return self._write_prepared_frame(
            dataset_id=dataset_id,
            writer=writer,
            instrument_id=instrument_id,
            frame=prepared,
        )

    def _write_prepared_frame(
        self,
        *,
        dataset_id: str,
        writer: _FeatureDatasetWriter,
        instrument_id: str | None,
        frame: _PandasDataFrame | None,
    ) -> int:
        if frame is None:
            return 0
        row_count = len(frame)
        if row_count <= 0:
            return 0
        writer.write_ingestion(
            dataset_id=dataset_id,
            records=frame,
            source=Source.BACKFILL.value,
            run_id="feature_catalog_restore",
            instrument_id=instrument_id,
        )
        return row_count

    def _prepare_macro_release_frame(
        self,
        frame: _PandasDataFrame,
        instrument_id: str,
    ) -> _PandasDataFrame | None:
        df = frame.copy()
        df["series_id"] = instrument_id
        df["observation_ts"] = self._to_datetime_ns_series(df.get("observation_ts"), df)
        df["release_ts"] = self._to_datetime_ns_series(df.get("release_ts"), df)
        df["release_end_ts"] = self._to_datetime_ns_series(df.get("release_end_ts"), df)
        df["value"] = PANDAS.to_numeric(df.get("value"), errors="coerce")
        df["ts_event"] = df["release_ts"]
        df["ts_init"] = sanitize_timestamp_ns(time.time_ns(), context="macro_release_restore")
        df["source"] = Source.BACKFILL.value
        df["run_id"] = "feature_catalog_restore"
        df = df.dropna(subset=["release_ts", "observation_ts"])
        columns = [
            "series_id",
            "observation_ts",
            "release_ts",
            "release_end_ts",
            "value",
            "ts_event",
            "ts_init",
            "source",
            "run_id",
        ]
        available = self._select_existing_columns(df, columns)
        if not available:
            return None
        return df.loc[:, available]

    def _prepare_macro_observation_frame(
        self,
        frame: _PandasDataFrame,
        instrument_id: str,
    ) -> _PandasDataFrame | None:
        df = frame.copy()
        if "observation_ts" not in df.columns and "timestamp" in df.columns:
            df["observation_ts"] = df["timestamp"]
        df["series_id"] = instrument_id
        df["observation_ts"] = self._to_datetime_ns_series(df.get("observation_ts"), df)
        df["value"] = PANDAS.to_numeric(df.get("value"), errors="coerce")
        df["ts_event"] = df["observation_ts"]
        df["ts_init"] = sanitize_timestamp_ns(time.time_ns(), context="macro_observations_restore")
        df["source"] = Source.BACKFILL.value
        df["run_id"] = "feature_catalog_restore"
        df = df.dropna(subset=["observation_ts"])
        columns = [
            "series_id",
            "observation_ts",
            "value",
            "ts_event",
            "ts_init",
            "source",
            "run_id",
        ]
        available = self._select_existing_columns(df, columns)
        if not available:
            return None
        return df.loc[:, available]

    def _prepare_events_frame(self, frame: _PandasDataFrame) -> _PandasDataFrame | None:
        df = frame.copy()
        df["event_timestamp"] = self._to_datetime_ns_series(df.get("event_timestamp"), df)
        df["ts_event"] = df["event_timestamp"]
        df["ts_init"] = sanitize_timestamp_ns(time.time_ns(), context="events_restore")
        df["source"] = df.get("source", Source.BACKFILL.value)
        df = df.dropna(subset=["event_timestamp"])
        columns = [
            "event_timestamp",
            "event_type",
            "name",
            "instrument_id",
            "importance",
            "source",
            "metadata",
            "ts_event",
            "ts_init",
        ]
        available = self._select_existing_columns(df, columns)
        if not available:
            return None
        return df.loc[:, available]

    def _prepare_micro_frame(
        self,
        frame: _PandasDataFrame,
        instrument_id: str,
    ) -> _PandasDataFrame | None:
        df = frame.copy()
        df["instrument_id"] = instrument_id
        df["timestamp"] = self._to_datetime_ns_series(df.get("timestamp"), df)
        df["ts_event"] = df["timestamp"]
        df["ts_init"] = sanitize_timestamp_ns(time.time_ns(), context="micro_restore")
        df = df.dropna(subset=["timestamp"])
        columns = ["instrument_id", "timestamp", "ts_event", "ts_init", *_MICRO_FEATURE_COLUMNS]
        available = self._select_existing_columns(df, columns)
        if not available:
            return None
        return df.loc[:, available]

    def _prepare_l2_frame(
        self,
        frame: _PandasDataFrame,
        instrument_id: str,
    ) -> _PandasDataFrame | None:
        df = frame.copy()
        df["instrument_id"] = instrument_id
        df["timestamp"] = self._to_datetime_ns_series(df.get("timestamp"), df)
        df["ts_event"] = df["timestamp"]
        df["ts_init"] = sanitize_timestamp_ns(time.time_ns(), context="l2_restore")
        df = df.dropna(subset=["timestamp"])
        columns = ["instrument_id", "timestamp", "ts_event", "ts_init", *_L2_FEATURE_COLUMNS]
        available = self._select_existing_columns(df, columns)
        if not available:
            return None
        return df.loc[:, available]

    @staticmethod
    def _filter_partition_field(
        frame: _PandasDataFrame,
        *,
        partition_field: str,
        instrument_id: str,
        global_entity: bool = False,
    ) -> _PandasDataFrame:
        if global_entity or not partition_field or partition_field not in frame.columns:
            return frame
        column = frame[partition_field].astype(str).str.strip()
        mask = column == instrument_id.strip()
        return frame.loc[mask].copy()

    @staticmethod
    def _to_datetime_ns_series(series: _PandasSeries | None, frame: _PandasDataFrame) -> _PandasSeries:
        if series is None:
            filler = PANDAS.Series([PANDAS.NA] * len(frame), index=frame.index, dtype="Int64")
            return cast(_PandasSeries, filler)
        converted = PANDAS.to_datetime(series, utc=True, errors="coerce")
        return cast(_PandasSeries, converted.astype("int64", copy=False))

    @staticmethod
    def _select_existing_columns(df: _PandasDataFrame, columns: Sequence[str]) -> list[str]:
        return [column for column in columns if column in df.columns]

    @staticmethod
    def _coerce_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_feature_values(value: object) -> dict[str, float] | None:
        if value is None:
            return None
        raw = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                raw = json.loads(text)
            except Exception:
                return None
        if not isinstance(raw, dict):
            return None
        converted: dict[str, float] = {}
        for key, item in raw.items():
            try:
                converted[str(key)] = float(item)
            except (TypeError, ValueError):
                continue
        return converted if converted else None

    def _resolve_writer(self, dataset_id: str) -> _FeatureDatasetWriter | None:
        if dataset_id not in _SUPPORTED_DATASETS:
            msg = f"Unsupported dataset_id for feature restoration: {dataset_id}"
            raise ValueError(msg)
        return self._get_writer()

    def _get_writer(self) -> _FeatureDatasetWriter:
        if self._writer is None:
            if self._writer_factory is not None:
                self._writer = self._writer_factory(self._db_connection)
            else:
                self._writer = self._build_default_writer(self._db_connection)
        return self._writer

    @staticmethod
    def _build_default_writer(connection_string: str) -> _FeatureDatasetWriter:
        """
        Build a fully wired DataStore writer for feature restoration.

        Ensures registries and stores are initialized so write validation and
        data registry updates occur as in the normal ingestion path.
        """
        try:
            from ml.core.common.registry_initialization import build_data_store

            data_store = build_data_store(db_connection=connection_string)
        except Exception:
            logger.warning(
                "feature_restore.writer_init_failed",
                exc_info=True,
                extra={"db_connection": connection_string},
            )
            raise
        return cast(_FeatureDatasetWriter, data_store)

    @staticmethod
    def _read_parquet(path: Path, *, timestamp_field: str) -> _PandasDataFrame | None:
        if PANDAS is None:
            return None
        try:
            frame = PANDAS.read_parquet(path)
        except Exception:
            if not HAS_PYARROW or pa is None or pq is None:
                logger.warning(
                    "feature_restore.parquet_read_failed",
                    exc_info=True,
                    extra={"path": str(path)},
                )
                return None
            try:
                parquet_file = pq.ParquetFile(path)
                tables = []
                for idx in range(parquet_file.num_row_groups):
                    try:
                        tables.append(parquet_file.read_row_group(idx))
                    except Exception:
                        logger.debug(
                            "feature_restore.row_group_read_failed",
                            exc_info=True,
                            extra={"path": str(path), "row_group": idx},
                        )
                if not tables:
                    return None
                frame = pa.concat_tables(tables, promote=True).to_pandas()
            except Exception:
                logger.warning(
                    "feature_restore.parquet_read_failed",
                    exc_info=True,
                    extra={"path": str(path)},
                )
                return None
        if timestamp_field not in frame.columns:
            logger.debug(
                "feature_restore.missing_timestamp_field",
                extra={"path": str(path), "timestamp_field": timestamp_field},
            )
            return None
        return cast(_PandasDataFrame, frame)


__all__ = sorted(
    [
        "FeatureCoverageRestorer",
        "FeatureRestoreResult",
        "SUPPORTED_FEATURE_DATASET_IDS",
    ],
)
