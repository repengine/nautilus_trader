from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ml.observability.db_persistence import ObservabilityDBPersistor


"""
Backfill observability tables from persisted JSONL files.

Usage:
  uv run -m ml.scripts.observability_backfill --src ./observability --db-url postgresql://...

Notes:
- Expects files under `<src>/<YYYY-MM-DD>/*` or `<src>/*` with names matching
  latency.jsonl, metrics.jsonl, correlation.jsonl, health.jsonl (or suffixed shards).
- Designed for small to moderate volumes; for large backfills prefer COPY.
"""


def _load_jsonl_files(base: Path, name: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    # Day-partitioned shards
    for day_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        frames.extend(_load_name_shards(day_dir, name))
    # Flat files at root (non-rotated)
    frames.extend(_load_name_shards(base, name))
    return frames


def _load_name_shards(dir_path: Path, name: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for p in sorted(dir_path.glob(f"{name}*.jsonl")):
        try:
            frames.append(pd.read_json(p, orient="records", lines=True))
        except Exception as exc:  # pragma: no cover - best-effort backfill
            logging.getLogger(__name__).debug("Failed reading %s: %s", p, exc)
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill observability JSONL into DB")
    ap.add_argument("--src", required=True, help="Base directory of observability files")
    ap.add_argument("--db-url", required=True, help="SQLAlchemy DB URL")
    args = ap.parse_args(argv)

    per = ObservabilityDBPersistor(connection_string=str(args.db_url))
    base = Path(args.src)
    tables: dict[str, pd.DataFrame] = {}
    for name in ("latency", "metrics", "correlation", "health"):
        frames = _load_jsonl_files(base, name)
        if frames:
            tables[name] = pd.concat(frames, ignore_index=True)
    if not tables:
        print("No observability files found")
        return 0
    out = per.persist(tables)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
