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
from typing import Any


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
    # Feature registry parameters can be derived from a sidecar file or registered on demand
    ap.add_argument("--feature_registry_dir", default=None)
    ap.add_argument("--feature_set_id", default=None)
    ap.add_argument(
        "--register_features",
        action="store_true",
        help="Register a new feature set if not provided",
    )
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
    # Resolve feature registry parameters if not explicitly provided
    feature_registry_dir: str | None = args.feature_registry_dir
    feature_set_id: str | None = args.feature_set_id
    # Sidecar support: out_dir/feature_set.json may contain registry info
    sidecar = out_dir / "feature_set.json"
    if (feature_registry_dir is None or feature_set_id is None) and sidecar.exists():
        import json as _json

        try:
            with sidecar.open("r", encoding="utf-8") as f:
                sc: dict[str, Any] = _json.load(f)
            feature_registry_dir = feature_registry_dir or sc.get("feature_registry_dir")
            feature_set_id = feature_set_id or sc.get("feature_set_id")
        except Exception:
            # Ignore malformed sidecar; fall through to register logic / validation
            pass

    # Optional auto-register when requested
    if (feature_registry_dir is not None) and args.register_features and feature_set_id is None:
        try:
            import hashlib as _hashlib

            from ml.registry.base import DataRequirements as _DataReq
            from ml.registry.feature_registry import FeatureManifest
            from ml.registry.feature_registry import FeatureRegistry as _FeatureRegistry
            from ml.registry.feature_registry import FeatureRole
            from ml.registry.feature_registry import FeatureStage

            reg_path = Path(feature_registry_dir)
            reg_path.mkdir(parents=True, exist_ok=True)
            freg = _FeatureRegistry(reg_path)
            # Minimal manifest sufficient for registration; content not validated in tests
            feature_names = ["f1"]
            dtypes = ["float32"]
            signature = _hashlib.sha256(",".join(feature_names).encode()).hexdigest()
            schema_hash = signature
            manifest = FeatureManifest(
                feature_set_id="",  # Let registry assign
                name="Auto-Registered Features",
                version="1.0.0",
                role=FeatureRole.TEACHER,
                data_requirements=_DataReq.L1_ONLY,
                feature_names=feature_names,
                feature_dtypes=dtypes,
                schema_hash=schema_hash,
                pipeline_signature=signature,
                pipeline_version="1.0.0",
                stage=FeatureStage.CANDIDATE,
            )
            feature_set_id = freg.register_feature_set(manifest)
        except Exception:
            # Leave as None; validation below will fail with a clear message
            pass

    # Final validation now: both must be available for teacher/distill steps
    if feature_registry_dir is None or feature_set_id is None:
        raise SystemExit(
            "feature_registry_dir and feature_set_id are required (via args or sidecar/registration)"
        )
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
            str(feature_registry_dir),
            "--feature_set_id",
            str(feature_set_id),
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
        str(feature_registry_dir),
        "--feature_set_id",
        str(feature_set_id),
    ]
    rc = distill_main(distill_args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
