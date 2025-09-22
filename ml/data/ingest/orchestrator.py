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

import logging
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any, Final, Protocol

import pandas as pd
from sqlalchemy.exc import IntegrityError

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


LOGGER = logging.getLogger(__name__)


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
        end_bucket_candidate = ((now_ns - DAY_NS) // DAY_NS) - 1
        if end_bucket_candidate < start_bucket:
            end_bucket = int(start_bucket)
        else:
            end_bucket = int(end_bucket_candidate)

        gaps: list[tuple[int, int]] = []
        for b in range(int(start_bucket), int(end_bucket) + 1):
            if b not in covered:
                gaps.append((b * DAY_NS, (b + 1) * DAY_NS))

        requested: list[tuple[int, int]] = []
        for ws, we in gaps:
            clamped = self._clamp_window_to_available_range(
                dataset_id=dataset_id,
                schema=schema,
                start_ns=ws,
                end_ns=we,
            )
            if clamped is None:
                continue
            start_ns, end_ns = clamped
            requested.append((start_ns, end_ns))
            frames: list[pd.DataFrame] = []
            ingest_symbol = instrument_id.split(".")[0]

            def _persist_frame(df: pd.DataFrame) -> None:
                if df.empty:
                    return
                normalized_df = self._normalize_time_columns(df)
                frames.append(normalized_df)
                self.writer.write(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    df=normalized_df,
                )
                if self.raw_writer is not None:
                    try:
                        dataset_type = _schema_to_dataset_type(schema)
                        if self.domain_loader is not None:
                            items = self.domain_loader.load(
                                dataset_id=dataset_id,
                                schema=schema,
                                instrument_id=instrument_id,
                                start_ns=start_ns,
                                end_ns=end_ns,
                            )
                            if items:
                                self.raw_writer.write(dataset_type=dataset_type, data=items)
                        else:
                            self.raw_writer.write(dataset_type=dataset_type, data=normalized_df)
                    except Exception:
                        pass

            if self.service is not None:
                start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
                end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)

                def _handle_chunk(chunk: IngestionChunk) -> None:
                    _persist_frame(chunk.frame)

                self.service.ingest(
                    IngestionRequest(
                        dataset=dataset_id,
                        schema=schema,
                        symbols=(ingest_symbol,),
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
                    instrument=ingest_symbol,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    source=Source.HISTORICAL.value,
                    state=state,
                )
                if df.empty:
                    continue
                _persist_frame(df)

            if not frames:
                continue

            df_combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            if df_combined.empty or "ts_event" not in df_combined.columns:
                continue
            ts_series = df_combined["ts_event"]
            if pd.api.types.is_datetime64_any_dtype(ts_series):
                ts_max = int(ts_series.max().value)
                ts_min = int(ts_series.min().value)
            else:
                ts_max = int(ts_series.max())
                ts_min = int(ts_series.min())
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
            try:
                self.registry.update_watermark(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    source=Source.BACKFILL,
                    last_success_ns=ts_max,
                    count=count,
                    completeness_pct=1.0,
                )
            except IntegrityError as exc:
                LOGGER.warning(
                    "Watermark update skipped (unregistered dataset)",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "schema": schema,
                        "reason": str(exc),
                    },
                )
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.debug(
                    "Watermark update failed",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "schema": schema,
                        "reason": str(exc),
                    },
                )

        return requested

    def _clamp_window_to_available_range(
        self,
        *,
        dataset_id: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> tuple[int, int] | None:
        """Clamp ingestion window to provider metadata bounds."""
        service = self.service
        if service is None:
            return (start_ns, end_ns)

        try:
            meta_start, meta_end = service.get_available_range_ns(
                dataset=dataset_id,
                schema=schema,
            )
        except AttributeError:
            return (start_ns, end_ns)

        clamped_start = start_ns
        clamped_end = end_ns
        if meta_start is not None:
            clamped_start = max(clamped_start, meta_start)
        if meta_end is not None:
            end_limit = meta_end - 1 if meta_end > 0 else 0
            clamped_end = min(clamped_end, end_limit)

        if clamped_end <= clamped_start:
            LOGGER.debug(
                "Skipping ingestion window outside metadata range",
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "meta_start": meta_start,
                    "meta_end": meta_end,
                },
            )
            return None

        if clamped_start != start_ns or clamped_end != end_ns:
            LOGGER.debug(
                "Trimmed ingestion window to metadata range",
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "trimmed_start": clamped_start,
                    "trimmed_end": clamped_end,
                },
            )

        return (clamped_start, clamped_end)

    @staticmethod
    def _normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ts_event/ts_init columns are nanosecond integers."""
        if "ts_event" in df.columns:
            event_series = df["ts_event"]
            if not pd.api.types.is_integer_dtype(event_series):
                converted = pd.to_datetime(event_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted) and converted.notna().all():
                    df.loc[:, "ts_event"] = converted.astype("int64")
                else:
                    try:
                        numeric = pd.to_numeric(event_series, errors="raise")
                    except Exception:
                        numeric = None
                    if numeric is not None and not numeric.isna().any():
                        df.loc[:, "ts_event"] = numeric.astype("int64")
        if "ts_init" in df.columns:
            init_series = df["ts_init"]
            if not pd.api.types.is_integer_dtype(init_series):
                converted_init = pd.to_datetime(init_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted_init) and converted_init.notna().all():
                    df.loc[:, "ts_init"] = converted_init.astype("int64")
                else:
                    try:
                        numeric_init = pd.to_numeric(init_series, errors="raise")
                    except Exception:
                        numeric_init = None
                    if numeric_init is not None and not numeric_init.isna().any():
                        df.loc[:, "ts_init"] = numeric_init.astype("int64")
        elif "ts_event" in df.columns:
            df.loc[:, "ts_init"] = df["ts_event"]
        return df

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
