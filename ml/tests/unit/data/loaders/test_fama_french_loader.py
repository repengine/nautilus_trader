"""Tests for the Fama/French loader."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ml.data.loaders.fama_french_loader import FamaFrenchDatasetSpec
from ml.data.loaders.fama_french_loader import FamaFrenchLoader


@pytest.fixture()
def sample_archive_bytes() -> bytes:
    """Create a small in-memory Fama/French style archive."""

    header = """This file was created with a fictitious generator
Please refer to the accompanying documentation
Date,MKT_RF,SMB,HML,RF
"""
    body = """202401,1.23,0.15,0.05,0.30
202402,-99.99,0.00,0.10,0.28
202403,0.45,-0.05,0.02,0.27
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("test-dataset.txt", header + body)
    return buffer.getvalue()


def test_parse_converts_percentages_and_dates(sample_archive_bytes: bytes) -> None:
    spec = FamaFrenchDatasetSpec(
        name="test",
        url="https://example.com/test.zip",
        columns=("date", "mkt_rf", "smb", "hml", "rf"),
        frequency="monthly",
        skip_rows=2,
        file_pattern="test-dataset.txt",
        value_scale=0.01,
    )

    loader = FamaFrenchLoader()
    frame = loader.parse(spec, sample_archive_bytes)

    assert frame.columns == ["date", "mkt_rf", "smb", "hml", "rf"]
    assert frame.height == 3

    first = frame.slice(0, 1)
    assert first["date"][0].year == 2024
    assert first["date"][0].month == 1
    assert first["mkt_rf"][0] == pytest.approx(0.0123)

    second = frame.slice(1, 1)
    assert second["mkt_rf"][0] is None


def test_load_writes_parquet(tmp_path: Path, sample_archive_bytes: bytes) -> None:
    spec = FamaFrenchDatasetSpec(
        name="test-load",
        url="https://example.com/test.zip",
        columns=("date", "mkt_rf", "smb", "hml", "rf"),
        frequency="monthly",
        skip_rows=3,
        file_pattern="test-dataset.txt",
    )

    class _Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def get(self, *_args: Any, **_kwargs: Any) -> _Response:
            return _Response(self.payload)

    loader = FamaFrenchLoader(session=_Session(sample_archive_bytes))
    output_path = tmp_path / "fama" / "test.parquet"
    frame = loader.load(spec, output_path)

    assert output_path.exists()
    reloaded = pl.read_parquet(output_path)
    assert reloaded.height == frame.height
