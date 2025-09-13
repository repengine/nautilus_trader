#!/usr/bin/env python3

"""
Raw dataset IO protocols and optional adapters.

These protocols let DataStore delegate raw dataset persistence and reads to a
pluggable component (e.g., a ParquetDataCatalog-backed writer/reader). Keeping
this optional preserves the current design where ingestion occurs via
Scheduler/CLI, while enabling a single in-process facade when configured.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType


@runtime_checkable
class RawIngestionWriterProtocol(Protocol):
    """
    Protocol for writing raw datasets (bars/quotes/trades/mbp1/tbbo).

    Implementations should persist the provided records and return the number of
    records written. Implementations must be safe to call off the hot path.
    """

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int: ...


@runtime_checkable
class RawReaderProtocol(Protocol):
    """
    Protocol for reading raw datasets over a time range.
    """

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike: ...


__all__ = ["RawIngestionWriterProtocol", "RawReaderProtocol"]

