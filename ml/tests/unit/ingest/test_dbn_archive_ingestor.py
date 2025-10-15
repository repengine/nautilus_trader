from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import zstandard

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


def _build_archive(tmp_path: Path, symbols: list[str]) -> Path:
    archive_path = tmp_path / "bundle.zip"
    metadata = {
        "query": {
            "dataset": "EQUS.MINI",
            "schema": "ohlcv-1m",
        },
    }
    symbology = {
        "symbols": symbols,
    }
    compressor = zstandard.ZstdCompressor()
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("metadata.json", json.dumps(metadata))
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
