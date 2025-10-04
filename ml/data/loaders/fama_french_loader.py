"""Fama/French dataset loader."""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
import requests

from ml.ml_types import PolarsDF


LOGGER = logging.getLogger(__name__)


Frequency = Literal["daily", "monthly", "annual"]


@dataclass(frozen=True)
class FamaFrenchDatasetSpec:
    """Specification describing a Fama/French dataset."""

    name: str
    url: str
    columns: Sequence[str]
    frequency: Frequency
    skip_rows: int = 4  # Metadata lines before the header row
    file_pattern: str | None = None
    value_scale: float = 0.01
    missing_sentinel: float | None = -99.99
    timeout_seconds: int = 30


class FamaFrenchLoader:
    """Loader capable of downloading and parsing Fama/French datasets."""

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, spec: FamaFrenchDatasetSpec) -> bytes:
        """Download the dataset archive for the supplied specification."""
        LOGGER.info("Downloading Fama/French dataset: %s", spec.name)
        response = self._session.get(spec.url, timeout=spec.timeout_seconds)
        response.raise_for_status()
        return response.content

    def parse(self, spec: FamaFrenchDatasetSpec, archive_bytes: bytes) -> PolarsDF:
        """Parse an archive downloaded from the Fama/French library."""
        buffer = io.BytesIO(archive_bytes)
        with zipfile.ZipFile(buffer) as archive:
            member_name = self._select_member(spec, archive.namelist())
            with archive.open(member_name) as member:
                raw_bytes = member.read()

        return self._parse_text_payload(spec, raw_bytes)

    def load(self, spec: FamaFrenchDatasetSpec, output_path: Path) -> PolarsDF:
        """Download, parse, and persist the dataset to disk."""
        archive = self.fetch(spec)
        frame = self.parse(spec, archive)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path)
        return frame

    def _select_member(self, spec: FamaFrenchDatasetSpec, members: Iterable[str]) -> str:
        candidates: list[str] = [name for name in members if not name.endswith("/")]
        if spec.file_pattern is not None:
            for name in candidates:
                if spec.file_pattern in name:
                    return name
            msg = f"No archive member matching pattern '{spec.file_pattern}' for {spec.name}"
            raise FileNotFoundError(msg)
        if not candidates:
            raise FileNotFoundError(f"Archive for {spec.name} contained no data files")
        return candidates[0]

    def _parse_text_payload(self, spec: FamaFrenchDatasetSpec, payload: bytes) -> PolarsDF:
        text = io.BytesIO(payload)
        try:
            df = pl.read_csv(
                text,
                skip_rows=spec.skip_rows,
                has_header=True,
                ignore_errors=True,
                infer_schema_length=50,
            )
            rename_map = {
                old: spec.columns[idx]
                for idx, old in enumerate(df.columns[: len(spec.columns)])
            }
            df = df.rename(rename_map)
        except Exception as exc:  # pragma: no cover - dependency guard
            msg = f"Failed to parse Fama/French dataset {spec.name}: {exc}"
            raise ValueError(msg) from exc

        df = df.with_columns(pl.all().map_elements(self._strip_strings, return_dtype=pl.Utf8))
        date_col = spec.columns[0]
        df = df.rename({date_col: "date_raw"})
        date_col = "date_raw"
        df = df.filter(pl.col(date_col).str.len_chars() > 0)
        df = df.with_columns(pl.col(date_col).alias("date_str"))
        df = df.with_columns(
            pl.col("date_str")
            .map_elements(
                lambda value: self._parse_date_value(value, spec.frequency),
                return_dtype=pl.Datetime(time_zone="UTC"),
            )
            .alias("date"),
        )
        df = df.drop_nulls("date")
        df = df.drop([date_col, "date_str"])

        numeric_cols = [col for col in spec.columns[1:]]
        for column in numeric_cols:
            df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False))
            if spec.missing_sentinel is not None:
                df = df.with_columns(
                    pl.when(pl.col(column) == spec.missing_sentinel)
                    .then(None)
                    .otherwise(pl.col(column))
                    .alias(column),
                )
            if spec.value_scale != 1.0:
                df = df.with_columns((pl.col(column) * spec.value_scale).alias(column))

        df = df.sort("date")
        df = df.select(["date", *numeric_cols])
        return df

    @staticmethod
    def _strip_strings(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_date_value(value: object, frequency: Frequency) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or not text.replace("-", "").isdigit():
            return None

        try:
            if frequency == "daily" and len(text) >= 8:
                year = int(text[0:4])
                month = int(text[4:6])
                day = int(text[6:8])
            elif frequency == "monthly" and len(text) >= 6:
                year = int(text[0:4])
                month = int(text[4:6])
                day = 1
            else:
                year = int(text[0:4])
                month = 1
                day = 1
            return datetime(year, month, day, tzinfo=UTC)
        except ValueError:
            return None


def download_fama_french_dataset(spec: FamaFrenchDatasetSpec, output_path: Path) -> PolarsDF:
    """Helper to download and store a Fama/French dataset using default loader."""
    loader = FamaFrenchLoader()
    return loader.load(spec, output_path)
