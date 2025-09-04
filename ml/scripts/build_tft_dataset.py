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
from pathlib import Path
from typing import Any

import numpy as np

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def _infer_feature_columns(df: Any) -> list[str]:
    if pl is not None and isinstance(df, pl.DataFrame):
        numeric = [c for c in df.columns if df[c].dtype.is_numeric()]
        exclude = {"y", "time_index"}
        # Keep timestamp and instrument_id out of feature matrix
        exclude |= {"timestamp", "instrument_id", "ts_event"}
        return [c for c in numeric if c not in exclude]
    if HAS_PANDAS and isinstance(df, pd.DataFrame):  # pragma: no cover
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
    ap.add_argument("--include_macro", action="store_true")
    ap.add_argument("--macro_lag_days", type=int, default=1)
    ap.add_argument("--include_micro", action="store_true")
    ap.add_argument("--include_l2", action="store_true")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = ParquetDataCatalog(path=str(data_dir))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    builder = TFTDatasetBuilder(
        catalog,
        symbols,
        include_macro=args.include_macro,
        macro_lag_days=args.macro_lag_days,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        micro_base_dir=str(data_dir),
    )

    df = builder.build_training_dataset(
        horizon_minutes=args.horizon_minutes,
        min_return_threshold=args.threshold,
        lookback_periods=args.lookback_periods,
        use_polars=True,
    )

    # Persist dataset
    dataset_parquet = out_dir / "dataset.parquet"
    dataset_csv = out_dir / "dataset.csv"
    if isinstance(df, pl.DataFrame):
        df.write_parquet(str(dataset_parquet))
        df.write_csv(str(dataset_csv))
    else:  # pragma: no cover
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # import guard
        assert pd is not None
        df.to_parquet(str(dataset_parquet))
        df.to_csv(str(dataset_csv), index=False)

    # Build feature matrix for student pipeline
    if not HAS_PANDAS:
        check_ml_dependencies(["pandas"])  # pragma: no cover
    assert pd is not None
    df_pd = df.to_pandas() if pl is not None and isinstance(df, pl.DataFrame) else df
    feature_names = _infer_feature_columns(df_pd)
    df_pd_sorted = df_pd.sort_values("time_index")
    cutoff = int(len(df_pd_sorted) * 0.8) if len(df_pd_sorted) > 0 else 0
    X = df_pd_sorted[feature_names].to_numpy(dtype=np.float32)
    X_train = X[:cutoff]
    X_val = X[cutoff:]

    np.savez_compressed(
        out_dir / "features_npz.npz",
        X_train=X_train,
        X_val=X_val,
        feature_names=np.array(feature_names),
    )

    print(
        f"Saved dataset to {dataset_parquet} and {dataset_csv}\nSaved features to {out_dir / 'features_npz.npz'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
