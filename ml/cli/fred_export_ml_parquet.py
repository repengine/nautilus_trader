#!/usr/bin/env python3
"""
Export FRED indicators to ML-format parquet for dataset builder.

Writes a long-format parquet file with columns: timestamp, series_id, value.
Uses FREDDataLoader with configured API key and default indicators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.loaders.fred_loader import FREDConfig
from ml.data.loaders.fred_loader import FREDDataLoader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export FRED ML-format parquet for joins.")
    ap.add_argument(
        "--out",
        dest="out",
        default="data/fred/fred_indicators_ml_format.parquet",
        help="Output parquet path (default: data/fred/fred_indicators_ml_format.parquet)",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass loader cache when fetching indicators",
    )
    args = ap.parse_args(argv)

    cfg = FREDConfig()
    loader = FREDDataLoader(cfg)
    data = loader.fetch_all_indicators(use_cache=not args.no_cache)
    out_path = loader.export_ml_parquet(data=data, out_path=Path(args.out))
    print(f"Wrote ML-format parquet: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

