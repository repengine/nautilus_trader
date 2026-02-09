"""
Offline Databento DBN archive ingestion utilities.

Cold-path helper for decoding Databento ``.dbn.zst`` archives that have been
downloaded ahead of time. The ingestor keeps the workflow fully typed,
provenance-aware, and aligned with the standard market-data writer interfaces.

Key responsibilities:
    * stream ``.dbn.zst`` members from a zip archive without holding the
      complete payloads in memory
    * decode each DBN file into pandas DataFrames using ``databento.DBNStore``
      (falling back to the Nautilus loader is impractical for the current DBN
      frames)
    * add ``source_dataset`` tags and normalized timestamps (nanoseconds)
    * fan-out writes to the canonical SQL writer and, optionally, a DataStore
      mirror via ``MarketDataWriterProtocol`` implementations

The module is intentionally cold-path only. All heavy I/O is performed inside
context managers and the ingestion surface is designed for CLI orchestration
under strict typing and metrics/observability conventions.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Protocol
from zipfile import ZipFile

import pandas as pd
import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.schema import schema_spec_for
from ml.stores.protocols import MarketDataWriterProtocol


try:  # Import lazily to keep tooling environments lightweight
    import zstandard
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guards
    raise RuntimeError(
        "The 'zstandard' package is required for DBN archive ingestion. Install the"
        " project dependencies before running the offline ingestor.",
    ) from exc


logger = structlog.get_logger(__name__)


_FILES_COUNTER = get_counter(
    "ml_offline_dbn_files_total",
    "Number of DBN files processed from offline archives",
    ["dataset"],
)
_ROWS_COUNTER = get_counter(
    "ml_offline_dbn_rows_total",
    "Total rows ingested from offline DBN archives",
    ["dataset"],
)
_DECODE_SECONDS = get_histogram(
    "ml_offline_dbn_decode_seconds",
    "Time spent decoding offline DBN files",
    ["dataset"],
)


class DBNDecoderProtocol(Protocol):
    """Protocol for decoding a decompressed DBN file into a pandas DataFrame."""

    def decode(self, path: Path, *, schema: str) -> pd.DataFrame: ...


@dataclass(slots=True)
class DatabentoDBNDecoder(DBNDecoderProtocol):
    """Decode DBN files via ``databento.DBNStore`` into pandas DataFrames."""

    def decode(self, path: Path, *, schema: str) -> pd.DataFrame:
        from databento import DBNStore  # Lazy import keeps module import cheap

        store = DBNStore.from_file(str(path))
        frame = store.to_df().reset_index()
        if "ts_event" not in frame.columns:
            raise ValueError("Decoded DBN frame missing 'ts_event' column")
        # Normalize timestamps to integer nanoseconds (UTC)
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True).astype("int64", copy=False)
        if "ts_init" in frame.columns:
            frame["ts_init"] = pd.to_datetime(frame["ts_init"], utc=True).astype("int64", copy=False)
        else:
            frame["ts_init"] = frame["ts_event"]
        return frame


@dataclass(slots=True, frozen=True)
class DBNArchiveIngestionConfig:
    """Configuration for ingesting a single offline DBN archive."""

    archive_path: Path
    dataset: str | None = None
    schema: str | None = None
    source_dataset: str | None = None
    instrument_suffix: str | None = None


@dataclass(slots=True, frozen=True)
class InstrumentIngestionSummary:
    """Per-instrument ingestion summary."""

    instrument_id: str
    frames_written: int
    rows_written: int


@dataclass(slots=True, frozen=True)
class DBNArchiveIngestionResult:
    """Aggregated ingestion outcome for a DBN archive."""

    dataset: str
    schema: str
    source_dataset: str
    instruments: tuple[InstrumentIngestionSummary, ...]
    total_frames: int
    total_rows: int


@dataclass(slots=True)
class DBNArchiveIngestor:
    """
    Stream and ingest Databento DBN archives into MarketData writers.

    Parameters
    ----------
    writer : MarketDataWriterProtocol
        Primary writer (typically ``SqlMarketDataWriter``).
    mirror_writer : MarketDataWriterProtocol | None
        Optional secondary writer (e.g., ``DataStoreMarketDataWriter``).
    decoder : DBNDecoderProtocol | None
        Decoder implementation. Defaults to ``DatabentoDBNDecoder``.

    Notes
    -----
    * No canonicalisation is applied beyond timestamp normalization and
      provenance tagging. Hot-path semantics remain untouched.
    * Archives are processed sequentially to minimise peak memory usage.
    """

    writer: MarketDataWriterProtocol
    mirror_writer: MarketDataWriterProtocol | None = None
    decoder: DBNDecoderProtocol | None = None

    def ingest_archive(self, config: DBNArchiveIngestionConfig) -> DBNArchiveIngestionResult:
        """Ingest a single zip archive containing ``.dbn.zst`` members."""
        archive_path = config.archive_path
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        with ZipFile(archive_path) as archive:
            metadata = _load_metadata(archive)
            dataset = config.dataset or metadata.dataset
            schema = config.schema or metadata.schema
            source_dataset = config.source_dataset or dataset

            schema_spec_for(schema)
            decoder = self.decoder or DatabentoDBNDecoder()
            summaries: list[InstrumentIngestionSummary] = []
            total_frames = 0
            total_rows = 0

            for entry in _iter_dbn_members(archive):
                instrument_id = _instrument_from_member(
                    filename=entry,
                    schema=schema,
                    suffix=config.instrument_suffix,
                )
                frame = _decode_member(
                    archive=archive,
                    member=entry,
                    dataset=dataset,
                    schema=schema,
                    instrument_id=instrument_id,
                    decoder=decoder,
                )
                if frame.empty:
                    logger.info(
                        "offline.ingest.member.empty",
                        archive=str(archive_path),
                        member=entry,
                        dataset=dataset,
                        schema=schema,
                        instrument_id=instrument_id,
                    )
                    summaries.append(
                        InstrumentIngestionSummary(
                            instrument_id=instrument_id,
                            frames_written=0,
                            rows_written=0,
                        ),
                    )
                    continue

                _FILES_COUNTER.labels(dataset=dataset).inc()
                _ROWS_COUNTER.labels(dataset=dataset).inc(int(frame.shape[0]))

                normalized = _prepare_frame(
                    frame=frame,
                    source_dataset=source_dataset,
                    instrument_id=instrument_id,
                )

                rows_written = int(normalized.shape[0])
                if rows_written == 0:
                    summaries.append(
                        InstrumentIngestionSummary(
                            instrument_id=instrument_id,
                            frames_written=0,
                            rows_written=0,
                        ),
                    )
                    continue

                written_primary = self.writer.write(
                    dataset_id=dataset,
                    schema=schema,
                    instrument_id=instrument_id,
                    df=normalized,
                )
                if self.mirror_writer is not None:
                    self.mirror_writer.write(
                        dataset_id=dataset,
                        schema=schema,
                        instrument_id=instrument_id,
                        df=normalized,
                    )

                summaries.append(
                    InstrumentIngestionSummary(
                        instrument_id=instrument_id,
                        frames_written=1,
                        rows_written=written_primary,
                    ),
                )
                total_frames += 1
                total_rows += written_primary

                logger.info(
                    "offline.ingest.member.completed",
                    archive=str(archive_path),
                    dataset=dataset,
                    schema=schema,
                    instrument_id=instrument_id,
                    rows=written_primary,
                )

        return DBNArchiveIngestionResult(
            dataset=dataset,
            schema=schema,
            source_dataset=source_dataset,
            instruments=tuple(summaries),
            total_frames=total_frames,
            total_rows=total_rows,
        )


@dataclass(slots=True, frozen=True)
class _ArchiveMetadata:
    dataset: str
    schema: str
    available_symbols: tuple[str, ...]


def _load_metadata(archive: ZipFile) -> _ArchiveMetadata:
    """Parse archive metadata from ``metadata.json`` and ``symbology.json``."""
    import json

    try:
        metadata_raw = json.loads(archive.read("metadata.json"))
    except KeyError as exc:  # pragma: no cover - archive quality ensured in tests
        raise ValueError("Archive missing metadata.json") from exc
    dataset = metadata_raw.get("query", {}).get("dataset")
    schema = metadata_raw.get("query", {}).get("schema")
    if not isinstance(dataset, str) or not isinstance(schema, str):
        raise ValueError("Archive metadata missing dataset/schema information")

    try:
        symbology_raw = json.loads(archive.read("symbology.json"))
        symbols = symbology_raw.get("symbols", [])
    except KeyError:
        symbols = []
    if not isinstance(symbols, Sequence):
        symbols = []

    return _ArchiveMetadata(
        dataset=dataset,
        schema=schema,
        available_symbols=tuple(str(token) for token in symbols),
    )


def _iter_dbn_members(archive: ZipFile) -> Iterable[str]:
    """Yield ``.dbn.zst`` members from the archive (sorted by name)."""
    entries = [info.filename for info in archive.infolist() if info.filename.endswith(".dbn.zst")]
    return tuple(sorted(entries))


def _instrument_from_member(*, filename: str, schema: str, suffix: str | None) -> str:
    """Derive the instrument identifier from the archive member name."""
    base_name = Path(filename).name
    if not base_name.endswith(".dbn.zst"):
        raise ValueError(f"Unexpected DBN member layout: {filename}")
    symbol_part = base_name[: -len(".dbn.zst")]
    schema_token = f".{schema}."
    if schema_token in symbol_part:
        symbol = symbol_part.split(schema_token, 1)[1]
    else:
        symbol = symbol_part.split(".")[-1]
    return f"{symbol}{suffix}" if suffix else symbol


def _decode_member(
    *,
    archive: ZipFile,
    member: str,
    dataset: str,
    schema: str,
    instrument_id: str,
    decoder: DBNDecoderProtocol,
) -> pd.DataFrame:
    """Decode a single archive member into a pandas DataFrame."""
    start = perf_counter()
    with archive.open(member) as compressed:
        with TemporaryDirectory() as tmp_dir:
            dbn_path = Path(tmp_dir, f"{instrument_id}.dbn")
            with dbn_path.open("wb") as target:
                with closing(zstandard.ZstdDecompressor().stream_reader(compressed)) as reader:
                    shutil.copyfileobj(reader, target, length=32_768)
            frame = decoder.decode(dbn_path, schema=schema)
    duration = perf_counter() - start
    _DECODE_SECONDS.labels(dataset=dataset).observe(duration)
    return frame


def _prepare_frame(
    *,
    frame: pd.DataFrame,
    source_dataset: str,
    instrument_id: str,
) -> pd.DataFrame:
    """Apply provenance tagging and enforce nanosecond timestamps."""
    working = frame.copy()
    if "ts_event" not in working.columns:
        raise ValueError("Frame missing 'ts_event' column post decode")
    working["ts_event"] = working["ts_event"].astype("int64", copy=False)
    working["ts_init"] = working.get("ts_init", working["ts_event"])
    working["source_dataset"] = source_dataset
    working["instrument_id"] = instrument_id
    working = working.sort_values(
        ["ts_init", "ts_event"],
        kind="mergesort",
    )
    ordered_cols = [
        "ts_event",
        "ts_init",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "last",
        "trade_count",
        "vwap",
        "source_dataset",
        "instrument_id",
    ]
    final_columns: list[str] = [col for col in ordered_cols if col in working.columns]
    # Preserve additional columns (e.g., publisher metadata) to keep lineage intact.
    for column in working.columns:
        if column not in final_columns:
            final_columns.append(column)
    return working[final_columns]


__all__ = [
    "DBNArchiveIngestionConfig",
    "DBNArchiveIngestionResult",
    "DBNArchiveIngestor",
    "DatabentoDBNDecoder",
    "InstrumentIngestionSummary",
]
