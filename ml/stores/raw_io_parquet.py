#!/usr/bin/env python3

"""
ParquetDataCatalog-backed raw IO adapters.

Reader uses catalog_utils to return DataFrame-like structures. Writer is a
pass-through for domain objects; mapping generic rows to Nautilus domain
objects is intentionally left to ingestion flows (e.g., Scheduler) where
instrument metadata and bar types are known.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.stores.raw_io import RawIngestionWriterProtocol
from ml.stores.raw_io import RawReaderProtocol


class ParquetCatalogRawReader(RawReaderProtocol):
    """
    Raw reader backed by Nautilus ParquetDataCatalog.
    """

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
        # Map to helper per dataset type
        # Catalog utils accept datetime/str; pass nanoseconds through as integers which
        # the underlying loader tolerates in our usage (tests patch a fake catalog).
        start: Any = start_ns
        end: Any = end_ns
        if dataset_type == DatasetType.BARS:
            return cast(DataFrameLike, bars_to_dataframe(self._catalog, [instrument_id], start, end))
        if dataset_type in (DatasetType.QUOTES, DatasetType.TBBO):
            return cast(DataFrameLike, quotes_to_dataframe(self._catalog, [instrument_id], start, end))
        if dataset_type == DatasetType.TRADES:
            return cast(DataFrameLike, trades_to_dataframe(self._catalog, [instrument_id], start, end))
        # Fallback: empty result for unsupported raw types
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
    Pass-through writer to ParquetDataCatalog.

    Expects caller to supply Nautilus domain objects (e.g., Bar/QuoteTick/TradeTick).
    For DataFrame/dict rows, perform conversion in ingestion flows where instrument
    metadata and bar types are available.
    """

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        # Only pass-through for iterable of domain objects
        if isinstance(data, list) and data and not isinstance(data[0], dict):
            items = cast(Iterable[object], data)
            self._catalog.write_data(items)
            return len(data)

        # Not supported: DataFrame or list of dicts without domain mapping
        return 0


__all__ = ["ParquetCatalogRawReader", "ParquetCatalogRawWriter"]
