"""
Observability persistence adapters (off hot-path).

Provides a small, typed sink for writing observability DataFrames to disk in
JSONL or CSV formats. Intended for background tasks; hot loops should not call
this directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(slots=True)
class ObservabilityPersistor:
    """
    Persist observability tables to disk.

    Parameters
    ----------
    base_path : Path
        Directory where output files will be written.
    file_format : str
        One of {"jsonl", "csv"}. Defaults to "jsonl".
    """

    base_path: Path
    file_format: str = "jsonl"

    def persist(self, tables: Mapping[str, pd.DataFrame | None]) -> dict[str, Path]:
        """
        Persist non-empty DataFrames to disk.

        Returns mapping of table name to written file path.
        """
        self.base_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            if self.file_format == "csv":
                out = self.base_path / f"{name}.csv"
                df.to_csv(out, index=False)
            else:
                out = self.base_path / f"{name}.jsonl"
                df.to_json(out, orient="records", lines=True)
            written[name] = out
        return written


__all__ = ["ObservabilityPersistor"]

