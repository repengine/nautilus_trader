#!/usr/bin/env python3
"""
Backfill recent OHLCV-1m bars for Tier 1 (or specified) symbols via Databento.

This script downloads only the required recent window and merges into a single
per-symbol file at `data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet` which the
TFT builder now reads as a fallback.

Usage examples
--------------

1) Backfill last 14 days for all symbols present under data/tier1
   python -m ml.scripts.backfill_ohlcv_recent --days 14

2) Backfill explicit dates for Tier 1 universe
   python -m ml.scripts.backfill_ohlcv_recent --start 2025-08-30 --end 2025-09-10 --tier 1

3) Backfill explicit symbols
   python -m ml.scripts.backfill_ohlcv_recent --symbols SPY QQQ AAPL --days 10

Requires DATABENTO_API_KEY in environment and the `databento` package.

"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Any

import numpy as np

from ml._imports import HAS_POLARS, check_ml_dependencies, pl

if not HAS_POLARS:
    check_ml_dependencies(["polars"])  # pragma: no cover


def _discover_symbols(data_dir: Path) -> list[str]:
    syms: list[str] = []
    if not data_dir.exists():
        return syms
    for p in data_dir.iterdir():
        if p.is_dir() and p.name.isupper():
            syms.append(p.name)
    return sorted(syms)


def _symbols_from_universe_file(path: Path) -> list[str]:
    """
    Parse a universe JSON file collecting all symbol entries.

    Supports two shapes:
    - {"symbols": ["SPY", "QQQ", ...]}
    - Nested lists of dicts with {"symbol": "SPY", ...} entries (current format).

    """
    try:
        import json

        with path.open() as f:
            data: Any = json.load(f)
    except Exception:
        return []

    # Direct string list at key "symbols"
    if isinstance(data, dict):
        direct = data.get("symbols")
        if isinstance(direct, list) and all(isinstance(x, str) for x in direct):
            return sorted({s.upper() for s in direct})

        # Otherwise, walk nested lists of dicts and collect any {"symbol": str}
        out: set[str] = set()
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        sym = item.get("symbol")
                        if isinstance(sym, str) and sym:
                            out.add(sym.upper())
        if out:
            return sorted(out)

    return []


def _last_bar_timestamp(base: Path, symbol: str) -> datetime | None:
    # Check l0 file first
    l0 = base / symbol / "l0" / f"{symbol}_ohlcv.parquet"
    frames: list[pl.DataFrame] = []
    if l0.exists():
        try:
            df = pl.read_parquet(str(l0))
            if not df.is_empty():
                col = (
                    "timestamp"
                    if "timestamp" in df.columns
                    else ("ts_event" if "ts_event" in df.columns else None)
                )
                if col:
                    frames.append(df.select(col).rename({col: "timestamp"}))
        except Exception:
            pass
    # Also consider historical/recent fallback files
    for name in ("ohlcv-1m_historical.parquet", "ohlcv-1m_recent.parquet"):
        f = base / symbol / name
        if f.exists():
            try:
                df = pl.read_parquet(str(f)).select([pl.col("timestamp").alias("timestamp")])
                if not df.is_empty():
                    frames.append(df)
            except Exception:
                pass
    if not frames:
        return None
    dfc = pl.concat(frames, how="vertical").drop_nulls()
    if dfc.is_empty():
        return None
    # Databento returns tz-aware; ensure UTC
    ts = dfc.select(pl.col("timestamp").max())[0, 0]
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime()
    return datetime.fromtimestamp(np.datetime64(ts, "ns").astype("int64") / 1e9)


def _merge_save(base: Path, symbol: str, df_new) -> None:
    out_dir = base / symbol / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_ohlcv.parquet"
    # Convert incoming pandas DataFrame with index (Databento often uses time index)
    if hasattr(df_new, "to_dict"):
        new_pl = pl.from_pandas(df_new, include_index=True)
    else:
        new_pl = pl.DataFrame(df_new)

    # Normalize potential time column names to "timestamp"
    time_aliases = ("timestamp", "ts_event", "ts", "time", "index")
    has_timestamp = any(c in new_pl.columns for c in time_aliases)
    if not has_timestamp:
        # Nothing to do; rely on selection below to fail explicitly
        pass
    else:
        # Choose the first present alias as the source
        source = next(c for c in time_aliases if c in new_pl.columns)
        if source != "timestamp":
            new_pl = new_pl.rename({source: "timestamp"})
    # Ensure dtype is datetime[ns] if possible
    if "timestamp" in new_pl.columns:
        try:
            new_pl = new_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        except Exception:
            pass
    keep = [
        c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in new_pl.columns
    ]
    new_pl = new_pl.select(keep)
    if out_path.exists():
        try:
            old = pl.read_parquet(str(out_path)).select(keep)
            merged = (
                pl.concat([old, new_pl], how="vertical")
                .unique(subset=["timestamp"])
                .sort("timestamp")
            )
        except Exception:
            merged = new_pl.sort("timestamp")
    else:
        merged = new_pl.sort("timestamp")
    merged.write_parquet(str(out_path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill recent OHLCV-1m bars via Databento")
    ap.add_argument("--data_dir", type=Path, default=Path("data/tier1"))
    ap.add_argument(
        "--symbols", nargs="*", help="Symbols to backfill (default: directories in data_dir)"
    )
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument(
        "--days", type=int, default=14, help="Backfill window when start/end not provided"
    )
    args = ap.parse_args(argv)

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not set")

    import databento as db

    client = db.Historical(api_key)
    base: Path = args.data_dir

    symbols: Iterable[str]
    if args.symbols:
        symbols = args.symbols
    elif args.tier in (1, 2, 3):
        # Attempt to read configured universe file, else directories
        uni_file = Path(f"ml/config/universe_tier{args.tier}.json")
        if uni_file.exists():
            syms = _symbols_from_universe_file(uni_file)
            symbols = syms if syms else _discover_symbols(base)
        else:
            symbols = _discover_symbols(base)
    else:
        symbols = _discover_symbols(base)

    # Parse start/end
    end_dt: datetime
    start_dt: datetime
    if args.start and args.end:
        start_dt = datetime.fromisoformat(args.start)
        end_dt = datetime.fromisoformat(args.end)
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=max(1, int(args.days)))

    for sym in symbols:
        # If no explicit start, try continue from last ts
        begin = start_dt
        if not args.start:
            last_ts = _last_bar_timestamp(base, sym)
            if last_ts is not None:
                # Add one minute margin
                begin = max(last_ts + timedelta(minutes=1), start_dt)
        if begin >= end_dt:
            continue
        try:
            df = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[sym],
                schema="ohlcv-1m",
                start=begin,
                end=end_dt,
            ).to_df()
            if not df.empty:
                _merge_save(base, sym, df)
                print(
                    f"{sym}: downloaded {len(df)} rows from {begin:%Y-%m-%d} to {end_dt:%Y-%m-%d}"
                )
            else:
                print(f"{sym}: no rows in requested window")
        except Exception as exc:
            print(f"{sym}: error {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
