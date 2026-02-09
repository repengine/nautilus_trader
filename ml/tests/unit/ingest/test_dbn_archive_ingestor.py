from __future__ import annotations

import json
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import zstandard

from ml.data.ingest import dbn_archive
from ml.data.ingest.dbn_archive import DBNArchiveIngestionConfig
from ml.data.ingest.dbn_archive import DBNArchiveIngestionResult
from ml.data.ingest.dbn_archive import DBNArchiveIngestor
from ml.data.ingest.dbn_archive import DBNDecoderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


@dataclass(slots=True)
class _FakeDecoder(DBNDecoderProtocol):
    frame: pd.DataFrame

    def decode(self, path: Path, *, schema: str) -> pd.DataFrame:  # noqa: ARG002
        return self.frame.copy()


@dataclass(slots=True)
class _RoutingDecoder(DBNDecoderProtocol):
    frames_by_symbol: dict[str, pd.DataFrame]

    def decode(self, path: Path, *, schema: str) -> pd.DataFrame:  # noqa: ARG002
        return self.frames_by_symbol[path.stem].copy()


@dataclass(slots=True)
class _RecordingWriter(MarketDataWriterProtocol):
    writes: list[dict[str, Any]]

    def __init__(self) -> None:
        self.writes = []

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        self.writes.append(
            {
                "dataset_id": dataset_id,
                "schema": schema,
                "instrument_id": instrument_id,
                "frame": df.copy(),
            },
        )
        return int(df.shape[0])


def _build_archive(
    tmp_path: Path,
    symbols: list[str],
    *,
    metadata: dict[str, Any] | None = None,
    include_symbology: bool = True,
    symbology_symbols: Any | None = None,
) -> Path:
    archive_path = tmp_path / "bundle.zip"
    metadata_payload = metadata or {
        "query": {
            "dataset": "EQUS.MINI",
            "schema": "ohlcv-1m",
        },
    }
    symbology = {"symbols": symbols if symbology_symbols is None else symbology_symbols}
    compressor = zstandard.ZstdCompressor()
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("metadata.json", json.dumps(metadata_payload))
        if include_symbology:
            zf.writestr("symbology.json", json.dumps(symbology))
        for symbol in symbols:
            payload = f"dbn-{symbol}".encode()
            compressed = compressor.compress(payload)
            member_name = f"bundle.ohlcv-1m.{symbol}.dbn.zst"
            zf.writestr(member_name, compressed)
    return archive_path


def test_ingest_archive_writes_frames(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path, symbols=["FOO", "BAR"])
    frame = pd.DataFrame(
        {
            "ts_event": [1, 2],
            "open": [1.0, 2.0],
        },
    )
    frame["ts_init"] = frame["ts_event"]

    writer = _RecordingWriter()
    decoder = _FakeDecoder(frame=frame)
    ingestor = DBNArchiveIngestor(writer=writer, mirror_writer=None, decoder=decoder)

    result = ingestor.ingest_archive(
        DBNArchiveIngestionConfig(archive_path=archive, instrument_suffix=".XNAS"),
    )

    assert isinstance(result, DBNArchiveIngestionResult)
    assert result.dataset == "EQUS.MINI"
    assert result.schema == "ohlcv-1m"
    assert result.source_dataset == "EQUS.MINI"
    assert len(result.instruments) == 2
    assert result.total_frames == 2
    assert result.total_rows == 4

    assert [entry["instrument_id"] for entry in writer.writes] == ["BAR.XNAS", "FOO.XNAS"]
    for call in writer.writes:
        frame_written = call["frame"]
        assert "source_dataset" in frame_written.columns
        assert frame_written["source_dataset"].dropna().unique().tolist() == ["EQUS.MINI"]
        assert frame_written["instrument_id"].dropna().unique().tolist() == [call["instrument_id"]]


def test_ingest_archive_raises_for_missing_archive(tmp_path: Path) -> None:
    writer = _RecordingWriter()
    ingestor = DBNArchiveIngestor(writer=writer)

    with pytest.raises(FileNotFoundError, match="Archive not found"):
        ingestor.ingest_archive(
            DBNArchiveIngestionConfig(archive_path=tmp_path / "missing.zip"),
        )


def test_ingest_archive_records_empty_member_and_mirror_write(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path, symbols=["BAR", "FOO"])
    decoder = _RoutingDecoder(
        frames_by_symbol={
            "BAR": pd.DataFrame(columns=["ts_event"]),
            "FOO": pd.DataFrame({"ts_event": [1], "open": [1.5]}),
        },
    )
    writer = _RecordingWriter()
    mirror_writer = _RecordingWriter()
    ingestor = DBNArchiveIngestor(writer=writer, mirror_writer=mirror_writer, decoder=decoder)

    result = ingestor.ingest_archive(DBNArchiveIngestionConfig(archive_path=archive))

    assert result.total_frames == 1
    assert result.total_rows == 1
    assert tuple(summary.instrument_id for summary in result.instruments) == ("BAR", "FOO")
    assert result.instruments[0].rows_written == 0
    assert result.instruments[1].rows_written == 1
    assert [entry["instrument_id"] for entry in writer.writes] == ["FOO"]
    assert [entry["instrument_id"] for entry in mirror_writer.writes] == ["FOO"]


def test_ingest_archive_skips_when_prepare_frame_returns_zero_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _build_archive(tmp_path, symbols=["SPY"])
    writer = _RecordingWriter()
    decoder = _FakeDecoder(frame=pd.DataFrame({"ts_event": [1], "open": [1.0]}))
    ingestor = DBNArchiveIngestor(writer=writer, decoder=decoder)
    monkeypatch.setattr(
        dbn_archive,
        "_prepare_frame",
        lambda **_: pd.DataFrame(columns=["ts_event", "ts_init"]),
    )

    result = ingestor.ingest_archive(DBNArchiveIngestionConfig(archive_path=archive))

    assert result.total_frames == 0
    assert result.total_rows == 0
    assert result.instruments[0].frames_written == 0
    assert writer.writes == []


def test_ingest_archive_raises_when_metadata_missing_dataset(tmp_path: Path) -> None:
    archive = _build_archive(
        tmp_path,
        symbols=["SPY"],
        metadata={"query": {"schema": "ohlcv-1m"}},
    )
    writer = _RecordingWriter()
    ingestor = DBNArchiveIngestor(writer=writer, decoder=_FakeDecoder(pd.DataFrame()))

    with pytest.raises(ValueError, match="dataset/schema"):
        ingestor.ingest_archive(DBNArchiveIngestionConfig(archive_path=archive))


def test_load_metadata_handles_missing_or_non_sequence_symbology(tmp_path: Path) -> None:
    missing_symbology_archive = _build_archive(tmp_path, symbols=["SPY"], include_symbology=False)
    with zipfile.ZipFile(missing_symbology_archive) as archive:
        metadata = dbn_archive._load_metadata(archive)
    assert metadata.available_symbols == ()

    nonscalar_symbology_archive = _build_archive(
        tmp_path,
        symbols=["SPY"],
        symbology_symbols={"SPY": "mapped"},
    )
    with zipfile.ZipFile(nonscalar_symbology_archive) as archive:
        metadata = dbn_archive._load_metadata(archive)
    assert metadata.available_symbols == ()


def test_instrument_from_member_validates_layout() -> None:
    assert (
        dbn_archive._instrument_from_member(
            filename="bundle.ohlcv-1m.SPY.dbn.zst",
            schema="ohlcv-1m",
            suffix=".XNAS",
        )
        == "SPY.XNAS"
    )
    assert (
        dbn_archive._instrument_from_member(
            filename="bundle.SPY.dbn.zst",
            schema="ohlcv-1m",
            suffix=None,
        )
        == "SPY"
    )
    with pytest.raises(ValueError, match="Unexpected DBN member layout"):
        dbn_archive._instrument_from_member(
            filename="bundle.SPY.txt",
            schema="ohlcv-1m",
            suffix=None,
        )


def test_prepare_frame_requires_ts_event_and_preserves_extra_columns() -> None:
    with pytest.raises(ValueError, match="ts_event"):
        dbn_archive._prepare_frame(
            frame=pd.DataFrame({"open": [1.0]}),
            source_dataset="EQUS.MINI",
            instrument_id="SPY",
        )

    prepared = dbn_archive._prepare_frame(
        frame=pd.DataFrame(
            {
                "ts_event": [20, 10],
                "ts_init": [20, 10],
                "open": [2.0, 1.0],
                "publisher_id": [7, 7],
            },
        ),
        source_dataset="EQUS.MINI",
        instrument_id="SPY.XNAS",
    )
    assert prepared.columns.tolist()[:4] == ["ts_event", "ts_init", "open", "source_dataset"]
    assert prepared.columns.tolist()[-1] == "publisher_id"
    assert prepared["ts_event"].tolist() == [10, 20]
    assert prepared["instrument_id"].dropna().unique().tolist() == ["SPY.XNAS"]


def test_databento_decoder_normalizes_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeStore:
        def to_df(self) -> pd.DataFrame:
            return pd.DataFrame({"ts_event": pd.to_datetime(["2024-01-02T14:30:00Z"])})

    class _FakeDBNStore:
        @staticmethod
        def from_file(path: str) -> _FakeStore:  # noqa: ARG004
            return _FakeStore()

    monkeypatch.setitem(sys.modules, "databento", types.SimpleNamespace(DBNStore=_FakeDBNStore))
    frame = dbn_archive.DatabentoDBNDecoder().decode(tmp_path / "dummy.dbn", schema="ohlcv-1m")

    assert str(frame["ts_event"].dtype) == "int64"
    assert frame["ts_init"].tolist() == frame["ts_event"].tolist()


def test_databento_decoder_requires_ts_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeStore:
        def to_df(self) -> pd.DataFrame:
            return pd.DataFrame({"open": [1.0]})

    class _FakeDBNStore:
        @staticmethod
        def from_file(path: str) -> _FakeStore:  # noqa: ARG004
            return _FakeStore()

    monkeypatch.setitem(sys.modules, "databento", types.SimpleNamespace(DBNStore=_FakeDBNStore))
    decoder = dbn_archive.DatabentoDBNDecoder()

    with pytest.raises(ValueError, match="ts_event"):
        decoder.decode(tmp_path / "dummy.dbn", schema="ohlcv-1m")
