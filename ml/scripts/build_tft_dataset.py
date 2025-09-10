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
    ap.add_argument("--start", required=False, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=False, help="End date (YYYY-MM-DD)")
    ap.add_argument(
        "--chunk_days",
        type=int,
        default=0,
        help="If >0, build in date chunks of this many days and concatenate",
    )
    ap.add_argument("--include_macro", action="store_true")
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

    logging.info("Initializing ParquetDataCatalog at %s", data_dir)
    catalog = ParquetDataCatalog(path=str(data_dir))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logging.info("Building dataset for symbols: %s", ",".join(symbols))

    builder = TFTDatasetBuilder(
        catalog,
        symbols,
        include_macro=args.include_macro,
        macro_lag_days=args.macro_lag_days,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        micro_base_dir=str(data_dir),
    )

    # Parse optional start/end
    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    logging.info(
        "Build params: horizon=%s, threshold=%s, lookback=%s, start=%s, end=%s, include_l2=%s, include_micro=%s, include_macro=%s",
        args.horizon_minutes,
        args.threshold,
        args.lookback_periods,
        start_dt,
        end_dt,
        args.include_l2,
        args.include_micro,
        args.include_macro,
    )

    # Optional chunked build to constrain memory
    if args.chunk_days and start_dt and end_dt:
        from datetime import timedelta

        logging.info("Chunked build enabled: %d-day chunks", args.chunk_days)
        dfs: list[pl.DataFrame] = []  # type: ignore[name-defined]
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=args.chunk_days), end_dt)
            logging.info("Building chunk %s -> %s", cursor, chunk_end)
            df_chunk = builder.build_training_dataset(
                horizon_minutes=args.horizon_minutes,
                min_return_threshold=args.threshold,
                lookback_periods=args.lookback_periods,
                use_polars=True,
                start=cursor,
                end=chunk_end,
            )
            if isinstance(df_chunk, pl.DataFrame) and not df_chunk.is_empty():  # type: ignore[name-defined]
                logging.info("Chunk rows: %d", len(df_chunk))
                dfs.append(df_chunk)
            cursor = chunk_end
        if not dfs:
            logging.warning("No data produced in any chunk; writing empty dataset")
            df = pl.DataFrame()  # type: ignore[name-defined]
        else:
            df = pl.concat(dfs, how="vertical")  # type: ignore[name-defined]
            logging.info("Concatenated dataset rows: %d", len(df))
    else:
        df = builder.build_training_dataset(
            horizon_minutes=args.horizon_minutes,
            min_return_threshold=args.threshold,
            lookback_periods=args.lookback_periods,
            use_polars=True,
            start=start_dt,
            end=end_dt,
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

    # Optionally export feature manifest
    if args.register_features:
        if not args.feature_registry_dir:
            raise SystemExit("--feature_registry_dir is required when --register_features is set")
        from ml.data.feature_manifest_export import FeatureExportConfig
        from ml.data.feature_manifest_export import export_feature_manifest
        from ml.registry.base import DataRequirements
        from ml.registry.feature_registry import FeatureRole

        role_map = {
            "teacher": FeatureRole.TEACHER,
            "student": FeatureRole.STUDENT,
            "inference_support": FeatureRole.INFERENCE_SUPPORT,
        }
        data_req = DataRequirements.L1_ONLY if not args.include_l2 else DataRequirements.L1_L2
        cfg = FeatureExportConfig(
            registry_path=Path(args.feature_registry_dir),
            role=role_map[args.feature_role],
            data_requirements=data_req,
        )
        flags = {
            "include_macro": args.include_macro,
            "include_micro": args.include_micro,
            "include_l2": args.include_l2,
            "horizon_minutes": args.horizon_minutes,
            "lookback_periods": args.lookback_periods,
        }
        fid = export_feature_manifest(
            feature_names=feature_names,
            feature_dtypes=["float32"] * len(feature_names),
            flags=flags,
            cfg=cfg,
        )
        print(f"Registered feature set: {fid} in {args.feature_registry_dir}")

    print(
        f"Saved dataset to {dataset_parquet} and {dataset_csv}\nSaved features to {out_dir / 'features_npz.npz'}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
