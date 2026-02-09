"""
Raw writer plumbing for feature datasets (events, microstructure, L2).

This module provides a :class:`RawIngestionWriterProtocol` implementation that
mirrors feature datasets to their canonical parquet caches so coverage tooling
always has a recoverable mirror. It also exposes a composite writer helper so
callers can fan out to multiple raw writers (e.g., catalog + feature mirrors).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.data.cache_common import day_partition_path
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.stores.raw_protocols import RawIngestionWriterProtocol


if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame
else:  # pragma: no cover - typing helper
    PandasDataFrame = object


logger = logging.getLogger(__name__)


def _ensure_pandas() -> None:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])


def _to_pandas_frame(data: object) -> PandasDataFrame:
    _ensure_pandas()
    assert pd is not None  # for mypy
    if isinstance(data, pd.DataFrame):
        return cast(PandasDataFrame, data.copy(deep=False))
    if HAS_POLARS and pl is not None:
        if isinstance(data, pl.DataFrame):
            return cast(PandasDataFrame, data.to_pandas())
    if isinstance(data, list):
        if not data:
            return cast(PandasDataFrame, pd.DataFrame())
        first = data[0]
        if isinstance(first, dict):
            return cast(PandasDataFrame, pd.DataFrame(data))
        # fall back to string conversion
        return cast(PandasDataFrame, pd.DataFrame(data))
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if isinstance(frame, pd.DataFrame):
            return cast(PandasDataFrame, frame)
    raise TypeError(f"Unsupported data payload for feature raw writer: {type(data)}")


def _strip_venue(symbol: str) -> str:
    token = symbol.strip().upper()
    if not token:
        return ""
    head = token.split(".", 1)[0]
    return head or token


@dataclass(slots=True)
class FeatureDatasetParquetRawWriter(RawIngestionWriterProtocol):
    """
    Raw writer that mirrors feature datasets to their parquet caches.
    """

    events_path: Path
    micro_base_dir: Path
    l2_base_dir: Path

    def __init__(
        self,
        *,
        events_path: Path | None = None,
        micro_base_dir: Path | None = None,
        l2_base_dir: Path | None = None,
    ) -> None:
        events_default = Path(
            os.getenv("FEATURE_EVENTS_PARQUET_PATH", "data/features/events/events.parquet"),
        )
        micro_default = Path(
            os.getenv("FEATURE_MICRO_CACHE_DIR", "data/features/micro_minute"),
        )
        l2_default = Path(
            os.getenv("FEATURE_L2_CACHE_DIR", "data/features/l2_minute"),
        )
        self.events_path = (events_path or events_default).resolve()
        self.micro_base_dir = (micro_base_dir or micro_default).resolve()
        self.l2_base_dir = (l2_base_dir or l2_default).resolve()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.micro_base_dir.mkdir(parents=True, exist_ok=True)
        self.l2_base_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: object,
    ) -> int:
        frame = _to_pandas_frame(data)
        if frame.empty:
            return 0
        if dataset_type == DatasetType.EVENTS_CALENDAR:
            return self._write_events(frame)
        if dataset_type == DatasetType.MICRO_MINUTE_FEATURES:
            return self._write_time_series(frame, base_dir=self.micro_base_dir)
        if dataset_type == DatasetType.L2_MINUTE_FEATURES:
            return self._write_time_series(frame, base_dir=self.l2_base_dir)
        msg = f"{type(self).__name__} does not support dataset {dataset_type.value}"
        raise ValueError(msg)

    def _write_events(self, frame: PandasDataFrame) -> int:
        assert pd is not None
        work = frame.copy(deep=False)
        if "event_timestamp" not in work.columns:
            logger.warning("feature_raw_writer.events_missing_timestamp")
            return 0
        if "instrument_id" not in work.columns:
            work["instrument_id"] = ""
        work["instrument_id"] = work["instrument_id"].fillna("").astype(str)
        work["event_timestamp"] = pd.to_datetime(work["event_timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["event_timestamp"])
        if work.empty:
            return 0
        work["event_timestamp"] = work["event_timestamp"].astype("int64", copy=False)
        if "ts_event" not in work.columns:
            work["ts_event"] = work["event_timestamp"]
        else:
            work["ts_event"] = (
                pd.to_numeric(work["ts_event"], errors="coerce")
                .fillna(work["event_timestamp"])
                .astype("int64")
            )
        if "ts_init" not in work.columns:
            work["ts_init"] = work["ts_event"]
        else:
            work["ts_init"] = (
                pd.to_numeric(work["ts_init"], errors="coerce")
                .fillna(work["ts_event"])
                .astype("int64")
            )
        dedup_keys: list[str] = [
            column
            for column in ("event_type", "event_timestamp", "instrument_id", "name")
            if column in work.columns
        ]
        combined = work
        if self.events_path.exists():
            try:
                existing = pd.read_parquet(self.events_path)
                if "event_timestamp" in existing.columns:
                    existing["event_timestamp"] = pd.to_datetime(
                        existing["event_timestamp"],
                        utc=True,
                        errors="coerce",
                    )
                    existing = existing.dropna(subset=["event_timestamp"])
                    if not existing.empty:
                        existing["event_timestamp"] = existing["event_timestamp"].astype(
                            "int64",
                            copy=False,
                        )
                combined = pd.concat([existing, work], ignore_index=True)
            except Exception:
                logger.warning(
                    "feature_raw_writer.events_read_failed",
                    exc_info=True,
                    extra={"path": str(self.events_path)},
                )
                combined = work
        if dedup_keys:
            combined = combined.drop_duplicates(subset=dedup_keys, keep="last", ignore_index=True)
        combined = combined.sort_values("event_timestamp")
        combined.to_parquet(self.events_path, index=False)
        return len(work)

    def _write_time_series(self, frame: PandasDataFrame, *, base_dir: Path) -> int:
        assert pd is not None
        required = {"instrument_id", "timestamp"}
        if not required.issubset(frame.columns):
            logger.warning(
                "feature_raw_writer.timeseries_missing_columns",
                extra={"required": sorted(required)},
            )
            return 0
        work = frame.copy(deep=False)
        work["instrument_id"] = work["instrument_id"].fillna("").astype(str)
        numeric_ts = pd.to_numeric(work["timestamp"], errors="coerce")
        work = work.loc[numeric_ts.notna()].copy()
        if work.empty:
            return 0
        work["timestamp_ns"] = pd.to_datetime(numeric_ts.loc[work.index], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp_ns"])
        if work.empty:
            return 0
        work["day"] = work["timestamp_ns"].dt.date
        if "ts_event" not in work.columns:
            work["ts_event"] = numeric_ts.loc[work.index].astype("int64")
        else:
            work["ts_event"] = (
                pd.to_numeric(work["ts_event"], errors="coerce")
                .fillna(numeric_ts.loc[work.index])
                .astype("int64")
            )
        if "ts_init" not in work.columns:
            work["ts_init"] = work["ts_event"]
        else:
            work["ts_init"] = (
                pd.to_numeric(work["ts_init"], errors="coerce")
                .fillna(work["ts_event"])
                .astype("int64")
            )
        total_rows = 0
        for (instrument, day), group in work.groupby(["instrument_id", "day"], sort=False):
            normalized = _strip_venue(instrument)
            if not normalized:
                continue
            path = day_partition_path(base_dir, normalized, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            to_write = group.drop(columns=["timestamp_ns", "day"])
            try:
                to_write.to_parquet(path, index=False)
            except Exception:
                logger.warning(
                    "feature_raw_writer.partition_write_failed",
                    exc_info=True,
                    extra={"path": str(path), "instrument": normalized},
                )
                continue
            total_rows += len(to_write)
        return total_rows


@dataclass(slots=True)
class FeatureValuesParquetMirrorWriter:
    """
    Mirror writer for computed FeatureStore values (ml_feature_values).
    """

    base_dir: Path
    partition_field: str = "instrument_id"
    timestamp_field: str = "ts_event"
    values_field: str = "values"

    def __init__(
        self,
        *,
        base_dir: Path,
        partition_field: str = "instrument_id",
        timestamp_field: str = "ts_event",
        values_field: str = "values",
    ) -> None:
        self.base_dir = base_dir.resolve()
        self.partition_field = partition_field
        self.timestamp_field = timestamp_field
        self.values_field = values_field
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_rows(self, rows: Sequence[Mapping[str, object]]) -> int:
        """
        Persist feature value rows into partitioned parquet mirrors.

        Args:
            rows: Sequence of row-like mappings for feature values.

        Returns:
            Number of rows written.
        """
        if not rows:
            return 0
        frame = _to_pandas_frame(list(rows))
        if frame.empty:
            return 0
        if self.partition_field not in frame.columns or self.timestamp_field not in frame.columns:
            logger.warning(
                "feature_values_mirror.missing_columns",
                extra={
                    "partition_field": self.partition_field,
                    "timestamp_field": self.timestamp_field,
                },
            )
            return 0
        assert pd is not None
        work = frame.copy(deep=False)
        work[self.partition_field] = work[self.partition_field].fillna("").astype(str)
        ts_series = work[self.timestamp_field]
        if pd.api.types.is_datetime64_any_dtype(ts_series):
            numeric_ts = ts_series.astype("int64", copy=False)
        else:
            numeric_ts = pd.to_numeric(ts_series, errors="coerce")
        work = work.loc[numeric_ts.notna()].copy()
        if work.empty:
            return 0
        numeric_ts = pd.to_numeric(numeric_ts.loc[work.index], errors="coerce").astype("int64")
        work[self.timestamp_field] = numeric_ts
        if "ts_init" not in work.columns:
            work["ts_init"] = work[self.timestamp_field]
        else:
            work["ts_init"] = (
                pd.to_numeric(work["ts_init"], errors="coerce")
                .fillna(work[self.timestamp_field])
                .astype("int64")
            )
        work["_day"] = pd.to_datetime(work[self.timestamp_field], unit="ns", utc=True).dt.date
        if self.values_field in work.columns:
            values_series = work[self.values_field].tolist()
            serialized: list[str] = []
            for value in values_series:
                if isinstance(value, str):
                    serialized.append(value)
                else:
                    try:
                        serialized.append(json.dumps(value, sort_keys=True))
                    except TypeError:
                        serialized.append(json.dumps(str(value)))
            work[self.values_field] = serialized
        total_rows = 0
        for (instrument, day), group in work.groupby(
            [self.partition_field, "_day"],
            sort=False,
        ):
            instrument_id = str(instrument).strip()
            if not instrument_id:
                continue
            path = day_partition_path(self.base_dir, instrument_id, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            to_write = group.drop(columns=["_day"])
            combined = to_write
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                    combined = pd.concat([existing, to_write], ignore_index=True)
                except Exception:
                    logger.warning(
                        "feature_values_mirror.read_failed",
                        exc_info=True,
                        extra={"path": str(path)},
                    )
                    combined = to_write
            dedup_keys = [
                key
                for key in ("feature_set_id", "instrument_id", "ts_event")
                if key in combined.columns
            ]
            if dedup_keys:
                combined = combined.drop_duplicates(
                    subset=dedup_keys,
                    keep="last",
                    ignore_index=True,
                )
            try:
                combined.to_parquet(path, index=False)
            except Exception:
                logger.warning(
                    "feature_values_mirror.write_failed",
                    exc_info=True,
                    extra={"path": str(path), "instrument": instrument_id},
                )
                continue
            total_rows += len(to_write)
        return total_rows

    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData-style objects.

        Args:
            data: List of objects to mirror.
        """
        if not data:
            return
        rows: list[dict[str, object]] = []
        for item in data:
            if isinstance(item, Mapping):
                rows.append(dict(item))
                continue
            raw = getattr(item, "__dict__", None)
            if isinstance(raw, dict):
                rows.append(dict(raw))
                continue
            row: dict[str, object] = {}
            for field in ("feature_set_id", "instrument_id", "values", "ts_event", "ts_init", "quality_flags"):
                if hasattr(item, field):
                    row[field] = getattr(item, field)
            if row:
                rows.append(row)
        self.write_rows(rows)

    def store_features(self, *args: object, **kwargs: object) -> None:
        """
        Backward-compatible alias for mirror writes.
        """
        data = kwargs.get("data")
        if data is None and args:
            data = args[0]
        if isinstance(data, list):
            self.write_batch(data)
        elif data is not None:
            self.write_batch([data])


@dataclass(slots=True)
class CompositeRawIngestionWriter(RawIngestionWriterProtocol):
    """
    Fan-out writer that forwards writes to multiple raw writers.
    """

    writers: tuple[RawIngestionWriterProtocol, ...]

    def __init__(self, writers: Iterable[RawIngestionWriterProtocol]) -> None:
        pool = tuple(writers)
        if not pool:
            msg = "CompositeRawIngestionWriter requires at least one writer"
            raise ValueError(msg)
        self.writers = pool

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        total = 0
        for writer in self.writers:
            try:
                total += writer.write(dataset_type=dataset_type, data=data)
            except ValueError:
                logger.debug(
                    "feature_raw_writer.writer_unsupported",
                    extra={
                        "writer": writer.__class__.__name__,
                        "dataset": dataset_type.value,
                    },
                )
                continue
            except Exception:
                logger.warning(
                    "feature_raw_writer.writer_failed",
                    exc_info=True,
                    extra={
                        "writer": writer.__class__.__name__,
                        "dataset": dataset_type.value,
                    },
                )
        return total


__all__ = [
    "CompositeRawIngestionWriter",
    "FeatureDatasetParquetRawWriter",
    "FeatureValuesParquetMirrorWriter",
]
