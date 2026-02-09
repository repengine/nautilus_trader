"""Tests for the Fama/French loader."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ml.data.loaders.fama_french_loader import FamaFrenchDatasetSpec
from ml.data.loaders.fama_french_loader import FamaFrenchLoader
from ml.data.loaders.fama_french_loader import download_fama_french_dataset


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


def test_fetch_invokes_session_get_with_timeout(sample_archive_bytes: bytes) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, url: str, timeout: int) -> _Response:
            captured["url"] = url
            captured["timeout"] = timeout
            return _Response(sample_archive_bytes)

    spec = FamaFrenchDatasetSpec(
        name="fetch-spec",
        url="https://example.com/fetch.zip",
        columns=("date", "mkt_rf"),
        frequency="monthly",
        timeout_seconds=17,
    )
    loader = FamaFrenchLoader(session=_Session())
    payload = loader.fetch(spec)

    assert payload == sample_archive_bytes
    assert captured == {"url": spec.url, "timeout": 17}


def test_select_member_errors_for_missing_pattern_and_empty_archive() -> None:
    loader = FamaFrenchLoader()
    spec = FamaFrenchDatasetSpec(
        name="missing-pattern",
        url="https://example.com/missing.zip",
        columns=("date", "value"),
        frequency="monthly",
        file_pattern="needle.csv",
    )

    with pytest.raises(FileNotFoundError, match="matching pattern"):
        loader._select_member(spec, ["a.txt", "b.csv"])

    without_pattern = FamaFrenchDatasetSpec(
        name="empty",
        url="https://example.com/empty.zip",
        columns=("date", "value"),
        frequency="monthly",
        file_pattern=None,
    )
    with pytest.raises(FileNotFoundError, match="contained no data files"):
        loader._select_member(without_pattern, ["folder/"])


def test_parse_text_payload_raises_value_error_when_polars_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FamaFrenchDatasetSpec(
        name="broken",
        url="https://example.com/broken.zip",
        columns=("date", "mkt_rf"),
        frequency="monthly",
    )
    loader = FamaFrenchLoader()

    def _raise(*_args: object, **_kwargs: object) -> pl.DataFrame:
        raise RuntimeError("bad csv")

    monkeypatch.setattr(pl, "read_csv", _raise)

    with pytest.raises(ValueError, match="Failed to parse Fama/French dataset broken"):
        loader._parse_text_payload(spec, b"any-bytes")


def test_parse_date_value_and_strip_helpers_cover_edge_cases() -> None:
    assert FamaFrenchLoader._strip_strings(None) == ""
    assert FamaFrenchLoader._strip_strings("  abc  ") == "abc"

    daily = FamaFrenchLoader._parse_date_value("20240130", "daily")
    monthly = FamaFrenchLoader._parse_date_value("202402", "monthly")
    annual = FamaFrenchLoader._parse_date_value("2024", "annual")
    invalid_text = FamaFrenchLoader._parse_date_value("not-a-date", "daily")
    invalid_date = FamaFrenchLoader._parse_date_value("20241301", "daily")

    assert daily == datetime(2024, 1, 30, tzinfo=UTC)
    assert monthly == datetime(2024, 2, 1, tzinfo=UTC)
    assert annual == datetime(2024, 1, 1, tzinfo=UTC)
    assert invalid_text is None
    assert invalid_date is None


def test_download_helper_delegates_to_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FamaFrenchDatasetSpec(
        name="delegate",
        url="https://example.com/delegate.zip",
        columns=("date", "value"),
        frequency="monthly",
    )
    captured: dict[str, object] = {}
    expected = pl.DataFrame({"date": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [0.1]})

    def _load(self: FamaFrenchLoader, incoming_spec: FamaFrenchDatasetSpec, output: Path) -> pl.DataFrame:
        captured["spec"] = incoming_spec
        captured["output"] = output
        return expected

    monkeypatch.setattr(FamaFrenchLoader, "load", _load)
    output_path = tmp_path / "fama.parquet"
    frame = download_fama_french_dataset(spec, output_path)

    assert frame.equals(expected)
    assert captured["spec"] is spec
    assert captured["output"] == output_path
