"""
Convenience helpers for Databento ingestion consumers.

This module provides small, typed helpers built around
:class:`ml.data.ingest.service.DatabentoIngestionService` so that CLIs and other cold
path tools can issue historical downloads without re-implementing request plumbing.

"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import SymbolIngestionSummary


FetchCallback = Callable[[IngestionChunk], None]


@dataclass(frozen=True)
class IngestionJob:
    """
    Description of a single historical ingestion request.
    """

    dataset: str
    schema: str
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    chunk_days: int | None = None
    allow_cost: bool = False
    rate_limit_per_min: int | None = None
    reason: str | None = None


def ensure_service(service: DatabentoIngestionService | None = None) -> DatabentoIngestionService:
    """
    Return the provided service instance or construct one from the environment.
    """
    return service if service is not None else DatabentoIngestionService.from_env()


def fetch_symbol_data(
    *,
    service: DatabentoIngestionService,
    dataset: str,
    schema: str,
    symbol: str,
    start: datetime,
    end: datetime,
    chunk_days: int | None = None,
    allow_cost: bool = False,
    rate_limit_per_min: int | None = None,
    reason: str | None = None,
) -> pd.DataFrame:
    """
    Download a symbol range into a single pandas DataFrame.

    The helper uses :meth:`DatabentoIngestionService.ingest` to stream data chunk-by-chunk
    and concatenates non-empty frames. When no data is returned an empty frame is
    produced.

    """
    frames: list[pd.DataFrame] = []

    def _collect(chunk: IngestionChunk) -> None:
        if chunk.frame.empty:
            return
        frames.append(chunk.frame)

    request = IngestionRequest(
        dataset=dataset,
        schema=schema,
        symbols=(symbol,),
        start=start,
        end=end,
        chunk_days=chunk_days,
        allow_cost=allow_cost,
        rate_limit_per_min=rate_limit_per_min,
        reason=reason,
    )
    service.ingest(request, on_chunk=_collect)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def run_jobs(
    jobs: Sequence[IngestionJob],
    *,
    service: DatabentoIngestionService,
    on_chunk: FetchCallback | None = None,
) -> list[SymbolIngestionSummary]:
    """
    Execute a sequence of ingestion jobs and return their summaries.
    """
    summaries: list[SymbolIngestionSummary] = []
    for job in jobs:
        request = IngestionRequest(
            dataset=job.dataset,
            schema=job.schema,
            symbols=job.symbols,
            start=job.start,
            end=job.end,
            chunk_days=job.chunk_days,
            allow_cost=job.allow_cost,
            rate_limit_per_min=job.rate_limit_per_min,
            reason=job.reason,
        )
        summaries.extend(service.ingest(request, on_chunk=on_chunk))
    return summaries


__all__ = [
    "FetchCallback",
    "IngestionJob",
    "ensure_service",
    "fetch_symbol_data",
    "run_jobs",
]
