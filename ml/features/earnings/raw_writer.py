#!/usr/bin/env python3
"""
Raw writer implementation for earnings datasets.

The :class:`EarningsParquetRawWriter` mirrors earnings actuals and estimates to
partitioned Parquet files so that operators can restore coverage quickly after
database outages. The writer implements :class:`ml.stores.raw_protocols.RawIngestionWriterProtocol`
and therefore integrates with the :class:`ml.stores.data_store.DataStore`
facade transparently.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import pd
from ml._imports import pl
from ml.registry.dataclasses import DatasetType
from ml.stores.raw_protocols import RawIngestionWriterProtocol


logger = logging.getLogger(__name__)


def _now_ts_suffix() -> str:
    """Return a timestamp-based suffix suitable for filenames."""
    now = datetime.now(tz=UTC)
    return now.strftime("%Y%m%dT%H%M%S%fZ")


class EarningsParquetRawWriter(RawIngestionWriterProtocol):
    """
    Raw writer that stores earnings datasets as partitioned Parquet files.

    Parameters
    ----------
    base_path:
        Root directory for raw mirrors. A sub-directory per dataset type will
        be created automatically.
    partition_keys:
        Columns used to build partition directories (e.g., ``("ticker",)``).
    file_prefix:
        Prefix applied to generated Parquet filenames.
    """

    SUPPORTED_DATASETS: Final[tuple[DatasetType, ...]] = (
        DatasetType.EARNINGS_ACTUALS,
        DatasetType.EARNINGS_ESTIMATES,
    )

    def __init__(
        self,
        base_path: Path,
        *,
        partition_keys: tuple[str, ...] = ("ticker",),
        file_prefix: str = "earnings",
    ) -> None:
        if not base_path:
            msg = "base_path cannot be empty"
            raise ValueError(msg)
        self._base_path = base_path
        self._partition_keys = partition_keys
        self._file_prefix = file_prefix
        self._base_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: object,
    ) -> int:
        """
        Write earnings records to partitioned Parquet files.

        Unsupported dataset types raise ``ValueError``. Failures are logged and
        emitted as warnings rather than exceptions so that the primary database
        write path remains unaffected.
        """
        if dataset_type not in self.SUPPORTED_DATASETS:
            msg = f"{type(self).__name__} only supports earnings datasets"
            raise ValueError(msg)

        try:
            frames = self._materialize_partitions(data)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "EarningsParquetRawWriter failed to materialize data for %s: %s",
                dataset_type.value,
                exc,
                exc_info=True,
            )
            return 0

        if not frames:
            return 0

        dataset_dir = self._base_path / dataset_type.value
        dataset_dir.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        for partition_path, frame in frames:
            filename = f"{self._file_prefix}_{_now_ts_suffix()}_{uuid.uuid4().hex}.parquet"
            full_path = dataset_dir.joinpath(*partition_path, filename)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._write_parquet(frame, full_path)
            except Exception as exc:  # pragma: no cover - IO backend issues
                logger.warning(
                    "Failed to write parquet mirror for %s partition=%s: %s",
                    dataset_type.value,
                    partition_path,
                    exc,
                    exc_info=True,
                )
                continue
            total_rows += self._frame_len(frame)
        return total_rows

    # ------------------------------------------------------------------#
    # Private helpers
    # ------------------------------------------------------------------#

    def _materialize_partitions(self, data: object) -> list[tuple[tuple[str, ...], object]]:
        if HAS_POLARS and pl is not None:
            frame = self._to_polars(data)
            if frame.is_empty():
                return []
            partitions = frame.partition_by(self._partition_keys, maintain_order=True) if self._partition_keys else [frame]
            result: list[tuple[tuple[str, ...], object]] = []
            for part in partitions:
                partition_dirs = self._build_partition_dirs_polars(part)
                result.append((partition_dirs, part))
            return result

        if HAS_PANDAS and pd is not None:
            frame_pd = self._to_pandas(data)
            if frame_pd.empty:
                return []
            if self._partition_keys:
                grouped = frame_pd.groupby(list(self._partition_keys), dropna=False)
                return [
                    (self._build_partition_dirs_pandas(keys), group.copy(deep=False))
                    for keys, group in grouped
                ]
            return [(tuple(), frame_pd)]

        raise RuntimeError("Neither polars nor pandas is available for earnings parquet writing")

    @staticmethod
    def _frame_len(frame: object) -> int:
        height = getattr(frame, "height", None)
        if isinstance(height, int):
            return height
        shape = getattr(frame, "shape", None)
        if isinstance(shape, tuple) and shape:
            first = shape[0]
            if isinstance(first, int):
                return first
            if isinstance(first, float):
                return int(first)
        length_fn = getattr(frame, "__len__", None)
        if callable(length_fn):
            try:
                return int(length_fn())
            except Exception:  # pragma: no cover - defensive
                return 0
        return 0

    @staticmethod
    def _write_parquet(frame: object, path: Path) -> None:
        write_parquet = getattr(frame, "write_parquet", None)
        if callable(write_parquet):
            write_parquet(path)
            return
        to_parquet = getattr(frame, "to_parquet", None)
        if callable(to_parquet):
            to_parquet(path, index=False)
            return
        raise TypeError(f"Unsupported frame type {type(frame)} for Parquet serialization")

    @staticmethod
    def _to_polars(data: object) -> Any:
        if not (HAS_POLARS and pl is not None):
            raise TypeError("Polars is not available")
        frame_cls: Any = getattr(pl, "DataFrame", tuple())
        if isinstance(data, frame_cls):
            return data
        if HAS_PANDAS and pd is not None and isinstance(data, getattr(pd, "DataFrame", tuple())):
            return pl.from_pandas(data, include_index=False)
        if isinstance(data, list):
            return pl.DataFrame(data)
        to_dicts = getattr(data, "to_dicts", None)
        if callable(to_dicts):
            return pl.DataFrame(to_dicts())
        raise TypeError(f"Unsupported data type {type(data)} for polars conversion")

    @staticmethod
    def _to_pandas(data: object) -> Any:
        if not (HAS_PANDAS and pd is not None):
            raise TypeError("pandas is not available")
        frame_cls: Any = getattr(pd, "DataFrame", tuple())
        if isinstance(data, frame_cls):
            return data
        if HAS_POLARS and pl is not None and isinstance(data, getattr(pl, "DataFrame", tuple())):
            to_pandas = getattr(data, "to_pandas", None)
            if callable(to_pandas):
                return to_pandas()
        if isinstance(data, list):
            return pd.DataFrame(data)
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            return pd.DataFrame([to_dict()])
        to_dicts = getattr(data, "to_dicts", None)
        if callable(to_dicts):
            return pd.DataFrame(to_dicts())
        raise TypeError(f"Unsupported data type {type(data)} for pandas conversion")

    def _build_partition_dirs_polars(self, frame: Any) -> tuple[str, ...]:
        if not self._partition_keys:
            return tuple()
        values = []
        head = getattr(frame, "head", None)
        if callable(head):
            first_row = head(1)
        else:
            return tuple()
        for key in self._partition_keys:
            value: Any = None
            if hasattr(first_row, "item"):
                try:
                    value = first_row.item(0, key)
                except Exception:
                    value = None
            values.append(f"{key}={self._normalize_partition_value(value)}")
        return tuple(values)

    def _build_partition_dirs_pandas(self, keys: object) -> tuple[str, ...]:
        if not self._partition_keys:
            return tuple()
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = [
            f"{key}={self._normalize_partition_value(value)}"
            for key, value in zip(self._partition_keys, keys, strict=False)
        ]
        return tuple(values)

    @staticmethod
    def _normalize_partition_value(value: object) -> str:
        if value is None:
            return "missing"
        if isinstance(value, str):
            return value.strip().upper() or "missing"
        try:
            return str(value)
        except Exception:  # pragma: no cover - defensive
            return "unknown"


__all__ = ["EarningsParquetRawWriter"]
