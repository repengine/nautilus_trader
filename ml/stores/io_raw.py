#!/usr/bin/env python3

"""
Raw dataset IO protocols and parquet-backed adapters.

Consolidates the raw IO Protocols and the ParquetDataCatalog reader/writer.

Migrated from:
- raw_io.py
- raw_io_parquet.py

Existing modules re-export these symbols with deprecation warnings.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast, runtime_checkable

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType


@runtime_checkable
class RawIngestionWriterProtocol(Protocol):
    """Protocol for writing raw datasets (bars/quotes/trades/mbp1/tbbo)."""

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int: ...


@runtime_checkable
class RawReaderProtocol(Protocol):
    """Protocol for reading raw datasets over a time range."""

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike: ...


class ParquetCatalogRawReader(RawReaderProtocol):
    """Raw reader backed by Nautilus ParquetDataCatalog."""

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike:
        start: Any = start_ns
        end: Any = end_ns
        if dataset_type == DatasetType.BARS:
            return cast(DataFrameLike, bars_to_dataframe(self._catalog, [instrument_id], start, end))
        if dataset_type in (DatasetType.QUOTES, DatasetType.TBBO):
            return cast(DataFrameLike, quotes_to_dataframe(self._catalog, [instrument_id], start, end))
        if dataset_type == DatasetType.TRADES:
            return cast(DataFrameLike, trades_to_dataframe(self._catalog, [instrument_id], start, end))
        try:
            from ml._imports import HAS_POLARS
            from ml._imports import pl as _pl

            if HAS_POLARS:
                return cast(DataFrameLike, _pl.DataFrame({}))
        except Exception:
            pass
        return cast(DataFrameLike, [])


class ParquetCatalogRawWriter(RawIngestionWriterProtocol):
    """
    Pass-through writer to ParquetDataCatalog for domain objects.
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        if isinstance(data, list) and data and not isinstance(data[0], dict):
            items = cast(Iterable[object], data)
            self._catalog.write_data(items)
            return len(data)
        return 0


__all__ = [
    "ParquetCatalogRawReader",
    "ParquetCatalogRawWriter",
    "RawIngestionWriterProtocol",
    "RawReaderProtocol",
]

