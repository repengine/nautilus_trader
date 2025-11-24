"""
Protocols for raw ingestion IO (reader/writer) used across stores and data orchestrators.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType


@runtime_checkable
class RawIngestionWriterProtocol(Protocol):
    """
    Protocol for writing raw dataset frames or domain objects.
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
    Protocol for reading raw datasets over a bounded time range.
    """

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike: ...

