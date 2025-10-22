"""
Materialize feature set CSV in manifest order.

Two modes:
- Reorder mode (default): read an input CSV that already contains feature columns
  named as in the manifest; write an output CSV with columns ordered exactly as
  the manifest (plus time_index, instrument_id, and optional target).
- From-OHLCV mode (best-effort): compute features from OHLCV bars using the
  FeatureEngineer, then select manifest columns if present. This path depends on
  Nautilus indicators and is optional; tests cover reorder mode only.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.registry.feature_registry import FeatureRegistry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--target_col", required=False, default=None)
    ap.add_argument(
        "--from_ohlcv",
        action="store_true",
        help="Compute features from OHLCV using FeatureEngineer (best-effort)",
    )
    args = ap.parse_args(argv)

    if not HAS_PANDAS:
        check_ml_dependencies(["pandas"])  # pragma: no cover - raises if missing
    pandas_module = pd
    if pandas_module is None:
        # If HAS_PANDAS was False we already raised above; guard for defensive coding
        check_ml_dependencies(["pandas"])  # pragma: no cover - raises if missing
        pandas_module = pd
    if pandas_module is None:
        raise RuntimeError("Pandas dependency 'pandas' is required for feature materialization")

    freg = FeatureRegistry(Path(args.feature_registry_dir))
    finfo = freg.get_feature_set(args.feature_set_id)
    if finfo is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")
    manifest = finfo.manifest
    feature_names = list(manifest.feature_names)

    df = pandas_module.read_csv(args.input_csv)

    if args.from_ohlcv:
        # Best-effort compute via FeatureEngineer
        try:  # pragma: no cover - heavy path not covered by unit tests
            from ml.features.engineering import FeatureConfig
            from ml.features.engineering import FeatureEngineer

            fe = FeatureEngineer(FeatureConfig())
            # Expect OHLCV and time_index columns
            required = ["open", "high", "low", "close"]
            for col in required:
                if col not in df.columns:
                    raise SystemExit(
                        f"from_ohlcv requires columns {required}; missing {col}",
                    )
            features_df, _ = fe.calculate_features(df, mode="batch", fit_scaler=False)
            # Select manifest columns if present
            missing = [c for c in feature_names if c not in features_df.columns]
            if missing:
                raise SystemExit(
                    f"Computed features missing required manifest columns: {missing}",
                )
            from typing import cast

            out_df = cast(Any, features_df[feature_names]).copy()
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"Feature computation failed: {exc}")
    else:
        # Reorder mode: ensure columns exist and write in manifest order
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise SystemExit(
                f"Input CSV missing required feature columns: {missing}",
            )
        from typing import cast

        out_df = cast(Any, df[feature_names]).copy()

    # Prepend index/group columns if present in input
    columns_out: list[str] = []
    if "time_index" in df.columns:
        out_df_typed = cast(Any, out_df)
        out_df_typed.insert(0, "time_index", df["time_index"])  # pandas indexing
        columns_out.append("time_index")
    if "instrument_id" in df.columns:
        out_df_typed = cast(Any, out_df)
        out_df_typed.insert(
            len(columns_out),
            "instrument_id",
            df["instrument_id"],
        )  # pandas indexing
        columns_out.append("instrument_id")
    # Append target if requested and present
    if args.target_col and args.target_col in df.columns:
        out_df[args.target_col] = df[args.target_col]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote materialized features: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
