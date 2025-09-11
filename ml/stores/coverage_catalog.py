from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

from ml.stores.protocols import CoverageProviderProtocol
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


DAY_NS: Final[int] = 86_400_000_000_000


def _schema_to_dataclass(schema: str) -> type[Any]:
    s = schema.lower()
    if "bar" in s or "ohlcv" in s:
        return cast(type[Any], Bar)
    if "tbbo" in s or "quote" in s:
        return cast(type[Any], QuoteTick)
    if "trade" in s:
        return cast(type[Any], TradeTick)
    # Default to Bar coverage when unspecified
    return cast(type[Any], Bar)


@dataclass(slots=True)
class CatalogCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider backed by Nautilus ParquetDataCatalog.

    Computes day-bucket coverage by reading available file intervals for the requested
    data class and instrument.

    """

    catalog_path: str

    def __post_init__(self) -> None:
        # Lazy-load catalog to avoid import-time overhead in some contexts
        self._catalog = ParquetDataCatalog(self.catalog_path)

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        data_cls = _schema_to_dataclass(schema)
        intervals = self._catalog.get_intervals(data_cls=data_cls, identifier=instrument_id)
        if not intervals:
            return set()

        buckets: set[int] = set()
        window_start = int(start_ns)
        window_end = int(end_ns)

        for s, e in intervals:
            # Overlap with requested window
            s_clamped = max(int(s), window_start)
            e_clamped = min(int(e), window_end)
            if s_clamped >= e_clamped:
                continue
            start_bucket = s_clamped // DAY_NS
            # Subtract 1ns to avoid counting the end boundary as next-day bucket
            end_bucket = (e_clamped - 1) // DAY_NS
            for b in range(int(start_bucket), int(end_bucket) + 1):
                buckets.add(b)
        return buckets
