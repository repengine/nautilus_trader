"""
Ingestion orchestrator: live start hook + gap detection and backfill.

Detects day-bucket gaps within a lookback window via a CoverageProvider and
backfills them using a DatabentoIngestor. Successful batches are persisted via a
MarketDataWriter, with DataRegistry events emitted and watermarks advanced.

Live streaming hook is provided for integration (attach a streaming client that
writes to the same canonical storage via the writer) and is intentionally not
exercised in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any, Final, Protocol

import pandas as pd

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import IngestState
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.registry.dataclasses import DatasetType
from ml.registry.protocols import RegistryProtocol
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


DAY_NS: Final[int] = 86_400_000_000_000


def _utc_now_ns() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1e9)


@dataclass(slots=True)
class IngestionOrchestrator:
    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    registry: RegistryProtocol
    ingestor: DatabentoIngestor
    # Optional dual-write to ParquetDataCatalog using domain objects
    raw_writer: RawIngestionWriterProtocol | None = None
    domain_loader: DomainWindowLoaderProtocol | None = None
    service: DatabentoIngestionService | None = None

    def backfill_gaps(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
        state: IngestState | None = None,
    ) -> list[tuple[int, int]]:
        """
        Detect day-bucket gaps within lookback window, backfill them, then emit registry
        events and update watermarks.

        Returns list of requested window ranges.

        """
        now_ns = _utc_now_ns()
        start_ns = now_ns - int(lookback_days) * DAY_NS
        covered = self.coverage.read_bucket_coverage(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            start_ns=start_ns,
            end_ns=now_ns,
        )
        start_bucket = start_ns // DAY_NS
        end_bucket = now_ns // DAY_NS
        gaps: list[tuple[int, int]] = []
        for b in range(int(start_bucket), int(end_bucket) + 1):
            if b not in covered:
                gaps.append((b * DAY_NS, (b + 1) * DAY_NS))

        for ws, we in gaps:
            frames: list[pd.DataFrame] = []

            def _persist_frame(df: pd.DataFrame) -> None:
                if df.empty:
                    return
                self.writer.write(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    df=df,
                )
                if self.raw_writer is not None:
                    try:
                        dataset_type = _schema_to_dataset_type(schema)
                        if self.domain_loader is not None:
                            items = self.domain_loader.load(
                                dataset_id=dataset_id,
                                schema=schema,
                                instrument_id=instrument_id,
                                start_ns=ws,
                                end_ns=we,
                            )
                            if items:
                                self.raw_writer.write(dataset_type=dataset_type, data=items)
                        else:
                            self.raw_writer.write(dataset_type=dataset_type, data=df)
                    except Exception:
                        pass

            if self.service is not None:
                start_dt = datetime.fromtimestamp(ws / 1_000_000_000, tz=UTC)
                end_dt = datetime.fromtimestamp(we / 1_000_000_000, tz=UTC)

                def _handle_chunk(chunk: IngestionChunk) -> None:
                    frames.append(chunk.frame)
                    _persist_frame(chunk.frame)

                self.service.ingest(
                    IngestionRequest(
                        dataset=dataset_id,
                        schema=schema,
                        symbols=(instrument_id,),
                        start=start_dt,
                        end=end_dt,
                        chunk_days=1,
                        allow_cost=False,
                        reason="orchestrator_backfill",
                    ),
                    on_chunk=_handle_chunk,
                )
            else:
                df = self.ingestor.ingest_time_window(
                    dataset=dataset_id,
                    schema=schema,
                    instrument=instrument_id,
                    start_ns=ws,
                    end_ns=we,
                    source=Source.HISTORICAL.value,
                    state=state,
                )
                if df.empty:
                    continue
                frames.append(df)
                _persist_frame(df)

            if not frames:
                continue

            df_combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            if df_combined.empty or "ts_event" not in df_combined.columns:
                continue
            ts_max = int(df_combined["ts_event"].max())
            ts_min = int(df_combined["ts_event"].min())
            count = len(df_combined.index)
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=Stage.DATA_INGESTED,
                source=Source.BACKFILL,
                run_id="auto_backfill",
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.SUCCESS,
            )
            self.registry.update_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=Source.BACKFILL,
                last_success_ns=ts_max,
                count=count,
                completeness_pct=1.0,
            )

        return gaps

    def start_live(self) -> None:  # pragma: no cover - integration hook
        """
        Attach live streaming (implementation-specific).
        """
        return None


class DomainWindowLoaderProtocol(Protocol):
    """
    Protocol to load Nautilus domain objects for a time window.

    Implementations should return a list of Nautilus data objects (e.g., Bars, Quotes,
    Trades) for the given dataset/schema/instrument and time range.

    """

    def load(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> list[Any]: ...


def _schema_to_dataset_type(schema: str) -> DatasetType:
    s = schema.lower()
    if "bar" in s or "ohlcv" in s:
        return DatasetType.BARS
    if "tbbo" in s or "quote" in s:
        return DatasetType.TBBO
    if "trade" in s:
        return DatasetType.TRADES
    # Default to BARS when ambiguous
    return DatasetType.BARS
