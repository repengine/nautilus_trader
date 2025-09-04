"""
Observability persistence adapters (off hot-path).

Provides a small, typed sink for writing observability DataFrames to disk in JSONL or
CSV formats. Intended for background tasks; hot loops should not call this directly.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime
from pathlib import Path

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
    # Optional rotation/compaction knobs (off hot-path)
    rotate_daily: bool = False
    max_file_bytes: int | None = None

    # Mapping of table -> time column (ns) used for daily rotation
    _time_cols: Mapping[str, str] = field(
        default_factory=lambda: {
            "latency": "ts_stage_end",
            "metrics": "timestamp",
            "correlation": "ts_event",
            "health": "timestamp",
        },
    )

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
            # Decide output path with optional rotation
            out_dir = self.base_path
            suffix = self.file_format
            if self.rotate_daily:
                col = self._time_cols.get(name)
                try:
                    ts_ns = int(df[col].iloc[0]) if col in df.columns else 0
                except Exception:
                    ts_ns = 0
                if ts_ns <= 0:
                    day_str = datetime.now(UTC).strftime("%Y-%m-%d")
                else:
                    day_str = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).strftime("%Y-%m-%d")
                out_dir = self.base_path / day_str
                out_dir.mkdir(parents=True, exist_ok=True)

            # Size-based rotation: if current file exists and exceeds threshold, write a unique file
            unique_tag = None
            base_name = f"{name}.{suffix}"
            out = out_dir / base_name
            if (
                self.max_file_bytes is not None
                and out.exists()
                and out.stat().st_size >= self.max_file_bytes
            ):
                unique_tag = datetime.now(UTC).strftime("%H%M%S%f")
            if self.rotate_daily and unique_tag is None:
                # Avoid overwriting when rotating by day: include a time tag
                unique_tag = datetime.now(UTC).strftime("%H%M%S%f")
            if unique_tag is not None:
                out = out_dir / f"{name}-{unique_tag}.{suffix}"

            if self.file_format == "csv":
                df.to_csv(out, index=False)
            else:
                df.to_json(out, orient="records", lines=True)
            written[name] = out
        return written

    def compact_daily(self, day: str) -> dict[str, Path]:
        """
        Compact per-table JSONL shards for a given day into single files.

        Writes compacted files under `<base_path>/<day>/compacted/<name>.jsonl` and
        returns a mapping of table name to compacted file path. Only applicable to
        JSONL; CSV users should rely on external compaction.

        """
        out: dict[str, Path] = {}
        day_dir = self.base_path / day
        if not day_dir.exists() or self.file_format != "jsonl":
            return out
        compact_dir = day_dir / "compacted"
        compact_dir.mkdir(parents=True, exist_ok=True)
        for table in ("latency", "metrics", "correlation", "health"):
            shards = sorted(day_dir.glob(f"{table}-*.jsonl"))
            if not shards:
                # Also handle case where a single file was written as {table}.jsonl
                single = day_dir / f"{table}.jsonl"
                if single.exists():
                    (compact_dir / f"{table}.jsonl").write_text(single.read_text())
                    out[table] = compact_dir / f"{table}.jsonl"
                continue
            compact_path = compact_dir / f"{table}.jsonl"
            with compact_path.open("w", encoding="utf-8") as w:
                for shard in shards:
                    with shard.open("r", encoding="utf-8") as r:
                        for line in r:
                            w.write(line)
            out[table] = compact_path
        return out


__all__ = ["ObservabilityPersistor"]
