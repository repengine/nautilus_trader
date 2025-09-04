#!/usr/bin/env python3
"""
End-to-end pipeline to build a TFT dataset, train/calibrate a TFT teacher, and distill a
student.

This composes existing CLIs to keep responsibilities single-purpose.

Steps
-----
1. Build dataset: ml/scripts/build_tft_dataset.py
2. Train teacher: ml/training/teacher/tft_cli.py (optional if --train is set)
3. Distill student: ml/training/distillation/cli.py

"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tier1")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--include_macro", action="store_true")
    ap.add_argument("--macro_lag_days", type=int, default=1)
    ap.add_argument("--include_micro", action="store_true")
    ap.add_argument("--include_l2", action="store_true")
    ap.add_argument("--horizon_minutes", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--lookback_periods", type=int, default=30)
    # Teacher config
    ap.add_argument("--train_teacher", action="store_true")
    ap.add_argument("--teacher_model_id", required=True)
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    ap.add_argument("--model_registry_dir", required=True)
    ap.add_argument("--student_model_id", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build dataset
    from ml.scripts.build_tft_dataset import main as build_main

    build_args = [
        "--data_dir",
        str(args.data_dir),
        "--symbols",
        args.symbols,
        "--out_dir",
        str(out_dir),
        "--horizon_minutes",
        str(args.horizon_minutes),
        "--threshold",
        str(args.threshold),
        "--lookback_periods",
        str(args.lookback_periods),
    ]
    if args.include_macro:
        build_args += ["--include_macro", "--macro_lag_days", str(args.macro_lag_days)]
    if args.include_micro:
        build_args += ["--include_micro"]
    if args.include_l2:
        build_args += ["--include_l2"]
    rc = build_main(build_args)
    if rc != 0:
        return rc

    # 2) Train teacher and calibrate
    teacher_npz = out_dir / "teacher_preds.npz"
    if args.train_teacher:
        from ml.training.teacher.tft_cli import main as tft_main

        train_csv = out_dir / "dataset.csv"
        tft_args = [
            "--train_data_csv",
            str(train_csv),
            "--out_dir",
            str(out_dir),
            "--model_id",
            args.teacher_model_id,
            "--feature_registry_dir",
            str(args.feature_registry_dir),
            "--feature_set_id",
            str(args.feature_set_id),
            "--max_epochs",
            "5",
        ]
        rc = tft_main(tft_args)
        if rc != 0:
            return rc
    else:
        # Expect teacher_preds.npz precomputed
        if not teacher_npz.exists():
            raise SystemExit("Teacher preds missing and --train_teacher not set")

    # 3) Distill student and register
    from ml.training.distillation.cli import main as distill_main

    features_npz = out_dir / "features_npz.npz"
    distill_args = [
        "--features_npz",
        str(features_npz),
        "--teacher_npz",
        str(teacher_npz),
        "--out_dir",
        str(out_dir),
        "--model_id",
        args.student_model_id,
        "--parent_id",
        args.teacher_model_id,
        "--registry_dir",
        str(args.model_registry_dir),
        "--feature_registry_dir",
        str(args.feature_registry_dir),
        "--feature_set_id",
        str(args.feature_set_id),
    ]
    rc = distill_main(distill_args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
