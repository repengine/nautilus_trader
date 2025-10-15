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
import math
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any, Final, Protocol

import pandas as pd
from sqlalchemy.exc import IntegrityError

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import load_market_feed_descriptors
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.market_bindings import resolve_market_dataset_bindings
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import IngestState
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.protocols import RegistryProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import FanoutMarketDataWriter


DAY_NS: Final[int] = 86_400_000_000_000


class BackfillWindowList(list[tuple[int, int]]):
    """
    List of persisted backfill windows enriched with ingestion metadata.
    """

    __slots__ = ("frames_written", "requested_windows", "rows_written")

    requested_windows: tuple[tuple[int, int], ...]
    frames_written: int
    rows_written: int

    def __init__(
        self,
        persisted: Iterable[tuple[int, int]] = (),
        *,
        requested: Iterable[tuple[int, int]] = (),
        frames_written: int = 0,
        rows_written: int = 0,
    ) -> None:
        super().__init__(persisted)
        self.requested_windows = tuple(requested)
        self.frames_written = int(frames_written)
        self.rows_written = int(rows_written)

    @property
    def attempted_window_count(self) -> int:
        """
        Number of windows we attempted to ingest regardless of persistence.
        """
        return len(self.requested_windows)

    @property
    def persisted_window_count(self) -> int:
        """
        Number of windows that produced persisted frames.
        """
        return len(self)

    def __repr__(self) -> str:  # pragma: no cover - debug representation
        return (
            "BackfillWindowList("  # nosec - repr only
            f"persisted={list(self)!r}, "
            f"requested={self.requested_windows!r}, "
            f"frames_written={self.frames_written}, "
            f"rows_written={self.rows_written})"
        )


def _coalesce_gap_windows(
    gaps: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """
    Merge contiguous daily gap windows into larger ranges.
    """
    if not gaps:
        return ()
    ordered = sorted(gaps, key=lambda window: window[0])
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        if start == current_end:
            current_end = end
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return tuple(merged)


def _max_chunk_days_for_schema(schema: str) -> int:
    """
    Derive an upper bound for chunk sizes based on schema type.
    """
    normalized = schema.lower()
    if "mbp" in normalized or "mbo" in normalized or normalized.startswith("l2"):
        return 31
    if "tbbo" in normalized or "bbo" in normalized or "quote" in normalized:
        return 365
    if "trade" in normalized:
        return 365
    if "ohlcv" in normalized or "bar" in normalized:
        return 1_095
    return 365


def _split_into_chunks(
    *,
    start_ns: int,
    end_ns: int,
    max_days: int,
) -> tuple[tuple[int, int], ...]:
    if start_ns >= end_ns:
        return ()
    chunks: list[tuple[int, int]] = []
    cursor = start_ns
    step_ns = max_days * DAY_NS
    while cursor < end_ns:
        chunk_end = min(end_ns, cursor + step_ns)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return tuple(chunks)


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

    @staticmethod
    def resolve_market_bindings(
        *,
        symbols: Sequence[str],
        instrument_ids: Sequence[str] | None,
        market_dataset_id: str | None,
        market_inputs: Sequence[MarketDatasetInput] | None,
    ) -> tuple[ResolvedMarketBinding, ...]:
        descriptors = load_market_feed_descriptors().as_mapping()
        return resolve_market_dataset_bindings(
            symbols=symbols,
            instrument_ids=instrument_ids,
            market_dataset_id=market_dataset_id,
            market_inputs=market_inputs,
            descriptors=descriptors,
        )

    def backfill_binding(
        self,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
        state: IngestState | None = None,
    ) -> dict[str, BackfillWindowList]:
        """
        Backfill a resolved binding across all mapped instruments.
        """
        self._log_binding(binding)
        schema = binding.schema
        if not schema:
            msg = (
                "Resolved binding missing schema; update feed descriptor to include schema for"
                f" {binding.descriptor_id or binding.dataset_id}"
            )
            raise ValueError(msg)

        instruments = binding.instrument_ids or (binding.symbol,)
        results: dict[str, BackfillWindowList] = {}
        for instrument_id in instruments:
            gaps = self.backfill_gaps(
                dataset_id=binding.dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                lookback_days=lookback_days,
                state=state,
                symbol_hint=binding.symbol,
            )
            results[instrument_id] = gaps
        return results

    def _log_binding(self, binding: ResolvedMarketBinding) -> None:
        extras = {
            "binding_id": binding.binding_id,
            "dataset_id": binding.dataset_id,
            "descriptor_id": binding.descriptor_id,
            "storage_kind": binding.storage_kind.value if binding.storage_kind else None,
            "source": binding.source,
        }

        if binding.source != "descriptor":
            LOGGER.warning(
                "Using fallback market binding (non-descriptor source)",
                extra=extras,
            )

        if binding.storage_kind is StorageKind.POSTGRES and not self._writer_supports_sql():
            LOGGER.warning(
                "SQL storage binding detected but writer is not SQL-backed",
                extra=extras,
            )

    def _writer_supports_sql(self) -> bool:
        writer = self.writer
        if isinstance(writer, FanoutMarketDataWriter):
            writer = writer.primary
        return isinstance(writer, SqlMarketDataWriter | DataStoreMarketDataWriter)

    def backfill_gaps(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
        state: IngestState | None = None,
        symbol_hint: str | None = None,
    ) -> BackfillWindowList:
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

        missing_windows: list[tuple[int, int]] = []
        for bucket in range(int(start_bucket), int(end_bucket) + 1):
            if bucket not in covered:
                missing_windows.append((bucket * DAY_NS, (bucket + 1) * DAY_NS))

        coalesced = _coalesce_gap_windows(missing_windows)
        window_slices: list[tuple[int, int]] = []
        max_chunk_days = _max_chunk_days_for_schema(schema)
        for window_start, window_end in coalesced:
            segments = _split_into_chunks(
                start_ns=window_start,
                end_ns=window_end,
                max_days=max_chunk_days,
            )
            window_slices.extend(segments)

        requested_windows: list[tuple[int, int]] = []
        persisted_windows: list[tuple[int, int]] = []
        frames_written = 0
        rows_written = 0
        for ws, we in window_slices:
            clamped = self._clamp_window_to_available_range(
                dataset_id=dataset_id,
                schema=schema,
                start_ns=ws,
                end_ns=we,
            )
            if clamped is None:
                continue
            start_ns, end_ns = clamped
            requested_windows.append((start_ns, end_ns))
            frames: list[pd.DataFrame] = []
            ingest_symbol = symbol_hint or instrument_id.split(".")[0]
            seen_source_datasets: set[str] = set()

            def _unique_str(frame: pd.DataFrame, column: str) -> str | None:
                if column not in frame.columns:
                    return None
                series = frame[column].dropna().astype(str).unique()
                if len(series) == 1 and series[0]:
                    return str(series[0])
                return None

            def _persist_frame(frame: pd.DataFrame) -> None:
                nonlocal frames_written
                nonlocal rows_written
                if frame.empty:
                    return
                normalized_frame = self._normalize_time_columns(frame)
                coerced_df = self._coerce_frame_to_manifest(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    frame=normalized_frame,
                )
                frames.append(coerced_df)
                frames_written += 1
                window_rows = len(coerced_df.index)
                rows_written += window_rows
                source_dataset = _unique_str(coerced_df, "source_dataset")
                if source_dataset:
                    seen_source_datasets.add(source_dataset)
                self.writer.write(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    df=coerced_df,
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
                            self.raw_writer.write(dataset_type=dataset_type, data=coerced_df)
                    except Exception:
                        pass
            if self.service is not None:
                start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
                end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)

                def _handle_chunk(chunk: IngestionChunk) -> None:
                    _persist_frame(chunk.frame)

                span_days = max(1, math.ceil((end_ns - start_ns) / DAY_NS))
                chunk_days = min(span_days, max_chunk_days)
                try:
                    self.service.ingest(
                        IngestionRequest(
                            dataset=dataset_id,
                            schema=schema,
                            symbols=(ingest_symbol,),
                            start=start_dt,
                            end=end_dt,
                            chunk_days=chunk_days,
                            allow_cost=False,
                            reason="orchestrator_backfill",
                        ),
                        on_chunk=_handle_chunk,
                    )
                except Exception:
                    LOGGER.error(
                        "Ingestion service failed",
                        exc_info=True,
                        extra={
                            "dataset_id": dataset_id,
                            "schema": schema,
                            "instrument_id": instrument_id,
                            "symbol": ingest_symbol,
                        },
                    )
                    raise
            else:
                try:
                    ingested_frame = self.ingestor.ingest_time_window(
                        dataset=dataset_id,
                        schema=schema,
                        instrument=ingest_symbol,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        source=Source.HISTORICAL.value,
                        state=state,
                    )
                except Exception:
                    LOGGER.error(
                        "Ingestor backfill failed",
                        exc_info=True,
                        extra={
                            "dataset_id": dataset_id,
                            "schema": schema,
                            "instrument_id": instrument_id,
                            "symbol": ingest_symbol,
                        },
                    )
                    raise
                if ingested_frame.empty:
                    continue
                _persist_frame(ingested_frame)

            if not frames:
                LOGGER.warning(
                    "Ingestion returned no frames",
                    extra={
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "instrument_id": instrument_id,
                        "symbol": ingest_symbol,
                        "lookback_days": lookback_days,
                    },
                )
                continue

            df_combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            if df_combined.empty or "ts_event" not in df_combined.columns:
                LOGGER.warning(
                    "Ingestion returned empty frame",
                    extra={
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "instrument_id": instrument_id,
                        "symbol": ingest_symbol,
                        "lookback_days": lookback_days,
                    },
                )
                continue
            persisted_windows.append((start_ns, end_ns))
            ts_series = df_combined["ts_event"]
            if pd.api.types.is_datetime64_any_dtype(ts_series):
                ts_max = int(ts_series.max().value)
                ts_min = int(ts_series.min().value)
            else:
                ts_max = int(ts_series.max())
                ts_min = int(ts_series.min())
            count = len(df_combined.index)
            event_metadata: dict[str, object] = {}
            if "source_dataset" in df_combined.columns:
                sources = df_combined["source_dataset"].dropna().astype(str).unique().tolist()
                seen_source_datasets.update(str(value) for value in sources if value)
            if seen_source_datasets:
                event_metadata["source_datasets"] = sorted(seen_source_datasets)
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
                metadata=event_metadata or None,
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

        return BackfillWindowList(
            persisted=tuple(persisted_windows),
            requested=tuple(requested_windows),
            frames_written=frames_written,
            rows_written=rows_written,
        )

    def _clamp_window_to_available_range(
        self,
        *,
        dataset_id: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> tuple[int, int] | None:
        """
        Clamp ingestion window to provider metadata bounds.
        """
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

    def _coerce_frame_to_manifest(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align incoming frames with the registered schema before persistence.
        """
        aligned_frame = frame.copy()
        aligned_frame.loc[:, "instrument_id"] = instrument_id

        registry = self.registry
        try:
            manifest = registry.get_manifest(dataset_id)
        except Exception:
            return aligned_frame

        schema = getattr(manifest, "schema", {}) or {}
        for column, expected_type in schema.items():
            if column not in aligned_frame.columns:
                continue
            try:
                normalized_expected = expected_type.lower()
            except AttributeError:
                normalized_expected = str(expected_type).lower()

            series = aligned_frame[column]
            try:
                if normalized_expected in {"str", "string", "object"}:
                    aligned_frame = aligned_frame.assign(**{column: series.astype(str)})
                elif normalized_expected in {"float", "float64"}:
                    numeric = pd.to_numeric(series, errors="coerce")
                    aligned_frame = aligned_frame.assign(**{column: numeric.astype("float64")})
                elif normalized_expected in {"int", "int64"}:
                    numeric = pd.to_numeric(series, errors="coerce")
                    if numeric.isna().any():
                        aligned_frame = aligned_frame.assign(**{column: numeric.astype("Int64")})
                    else:
                        aligned_frame = aligned_frame.assign(**{column: numeric.astype("int64")})
                elif normalized_expected in {"bool", "boolean"}:
                    aligned_frame = aligned_frame.assign(**{column: series.astype("bool")})
            except Exception as exc:  # pragma: no cover - defensive typing guard
                LOGGER.debug(
                    "Type coercion skipped for column %s on dataset %s: %s",
                    column,
                    dataset_id,
                    exc,
                    exc_info=True,
                )
        LOGGER.debug(
            "Coerced frame dtypes | dataset=%s instrument=%s dtypes=%s",
            dataset_id,
            instrument_id,
            {col: str(dtype) for col, dtype in aligned_frame.dtypes.items()},
        )
        return aligned_frame


    @staticmethod
    def _normalize_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure ts_event/ts_init columns are nanosecond integers.
        """
        working = frame.copy()
        if "ts_event" in working.columns:
            event_series = working["ts_event"]
            if not pd.api.types.is_integer_dtype(event_series):
                converted = pd.to_datetime(event_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted) and converted.notna().all():
                    working.loc[:, "ts_event"] = converted.astype("int64")
                else:
                    numeric = pd.to_numeric(event_series, errors="coerce")
                    if numeric.notna().all():
                        working.loc[:, "ts_event"] = numeric.astype("int64")
        if "ts_init" in working.columns:
            init_series = working["ts_init"]
            if not pd.api.types.is_integer_dtype(init_series):
                converted_init = pd.to_datetime(init_series, utc=True, errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted_init) and converted_init.notna().all():
                    working.loc[:, "ts_init"] = converted_init.astype("int64")
                else:
                    numeric_init = pd.to_numeric(init_series, errors="coerce")
                    if numeric_init.notna().all():
                        working.loc[:, "ts_init"] = numeric_init.astype("int64")
        elif "ts_event" in working.columns:
            working.loc[:, "ts_init"] = working["ts_event"]
        return working


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
