from __future__ import annotations

import hashlib
from dataclasses import asdict
from dataclasses import dataclass
from typing import Final

import pandas as pd


def compute_schema_hash(df: pd.DataFrame) -> str:
    """
    Compute a stable schema hash from column names and dtypes.

    Uses SHA256 over "name:dtype" joined by commas in column order.

    """
    parts = [f"{name}:{dtype!s}" for name, dtype in zip(df.columns, df.dtypes, strict=False)]
    joined = ",".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """
    Return SHA256 hex digest for raw bytes.
    """
    return hashlib.sha256(data).hexdigest()


@dataclass(slots=True)
class FixtureManifest:
    """
    Minimal manifest describing a deterministic fixture snapshot.
    """

    dataset: str
    instrument_id: str
    start_ns: int
    end_ns: int
    rows: int
    schema_hash: str
    content_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


__all__: Final[list[str]] = [
    "FixtureManifest",
    "compute_bytes_sha256",
    "compute_schema_hash",
]
