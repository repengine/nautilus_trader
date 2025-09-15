#!/usr/bin/env python3
"""
Build TFT training dataset from tier1 data with optional macro and micro features.

Outputs:
- dataset.parquet / dataset.csv
- features_npz.npz with {X_train, X_val, feature_names}

This CLI avoids heavy training dependencies and focuses on dataset preparation.

"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Any as _Any
from typing import cast as _cast

import numpy as np

from ml._imports import HAS_PANDAS
from ml._imports import pd
from ml._imports import pl
from ml.data import DatasetBuildConfig as APICfg
from ml.data import build_tft_dataset as api_build


pl = _cast(_Any, pl)


def _infer_feature_columns(df: Any) -> list[str]:
    if pl is not None:
        PLDF = getattr(pl, "DataFrame", None)
        if PLDF is not None and isinstance(df, PLDF):
            numeric = [c for c in df.columns if df[c].dtype.is_numeric()]
            exclude = {"y", "time_index"}
            # Keep timestamp and instrument_id out of feature matrix
            exclude |= {"timestamp", "instrument_id", "ts_event"}
            return [c for c in numeric if c not in exclude]
    if HAS_PANDAS:
        PDDF = getattr(pd, "DataFrame", None)
        if PDDF is not None and isinstance(df, PDDF):  # pragma: no cover
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
            return [c for c in numeric if c not in exclude]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tier1")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., SPY,QQQ,AAPL")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--horizon_minutes", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--lookback_periods", type=int, default=30)
    ap.add_argument("--start", required=False, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=False, help="End date (YYYY-MM-DD)")
    ap.add_argument(
        "--chunk_days",
        type=int,
        default=0,
        help="If >0, build in date chunks of this many days and concatenate",
    )
    # Macro now on by default; --no_macro disables it
    ap.add_argument("--include_macro", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--no_macro",
        action="store_true",
        help="Disable FRED macro join (enabled by default)",
    )
    ap.add_argument("--macro_lag_days", type=int, default=1)
    ap.add_argument("--include_micro", action="store_true")
    ap.add_argument("--include_l2", action="store_true")
    # Optional FeatureRegistry export
    ap.add_argument("--register_features", action="store_true")
    ap.add_argument("--feature_registry_dir", required=False)
    ap.add_argument(
        "--feature_role",
        choices=["teacher", "student", "inference_support"],
        default="teacher",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = ap.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose or os.environ.get("ML_DEBUG") else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    include_macro = not bool(getattr(args, "no_macro", False))
    # Parse optional start/end
    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    logging.info("Initializing dataset API build at %s", data_dir)
    cfg = APICfg(
        data_dir=data_dir,
        out_dir=out_dir,
        symbols=symbols,
        include_macro=include_macro,
        macro_lag_days=args.macro_lag_days,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        horizon_minutes=args.horizon_minutes,
        threshold=args.threshold,
        lookback_periods=args.lookback_periods,
        start=start_dt,
        end=end_dt,
        chunk_days=args.chunk_days,
        register_features=bool(args.register_features),
        feature_registry_dir=Path(args.feature_registry_dir) if args.feature_registry_dir else None,
        feature_role=str(args.feature_role),
    )

    result = api_build(cfg)
    print(
        f"Saved dataset to {result.dataset_parquet} and {result.dataset_csv}\nSaved features to {result.features_npz}",
    )
    if result.feature_set_id:
        print(f"Registered feature set: {result.feature_set_id} in {args.feature_registry_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
