from __future__ import annotations


"""
CLI to train and register a LightGBM student from teacher outputs.

Cold-path only: uses numpy arrays saved in .npz files.

Inputs
------
- --features_npz: .npz with X_train, X_val, feature_names
- --teacher_npz:  .npz with q_train (probs) and optionally y_val_true (0/1)
- --out_dir:      directory to write ONNX and sidecar
- --model_id:     identifier for the student
- --parent_id:    teacher model_id for lineage
- --registry_dir: path to local model registry

Outputs
-------
- student.onnx
- student.meta.json
- registry entry (LocalModelRegistry)
"""

import argparse
from pathlib import Path

import numpy as np

from ml.distillation.lightgbm_student import LightGBMStudentDistiller
from ml.distillation.lightgbm_student import schema_hash
from ml.registry.model_registry import LocalModelRegistry
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest


def main(argv: list[str] | None = None) -> int:
    """
    Run LightGBM student distillation from command line.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments.

    Returns
    -------
    int
        Exit code.

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True)
    ap.add_argument("--teacher_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--parent_id", required=True)
    ap.add_argument("--registry_dir", required=True)
    ap.add_argument("--objective", default="logit_mse", choices=["logit_mse", "soft_ce", "hybrid"])
    ap.add_argument("--kd_lambda", type=float, default=0.5)
    ap.add_argument("--early_stopping", type=int, default=200)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    X = np.load(args.features_npz, allow_pickle=True)
    X_train = X["X_train"].astype(np.float32)
    X_val = X["X_val"].astype(np.float32)
    feature_names = list(X["feature_names"].tolist())

    T = np.load(args.teacher_npz, allow_pickle=True)
    q_train = T["q_train"].astype(np.float32)
    y_val_true = T["y_val_true"].astype(np.float32) if "y_val_true" in T else None

    distiller = LightGBMStudentDistiller(
        objective=args.objective,
        kd_lambda=args.kd_lambda,
        early_stopping=args.early_stopping,
        opset=args.opset,
    )
    distiller.fit(X_train, q_train, X_val, y_val_true)
    onnx_path, meta_path = distiller.export_onnx(
        feature_names=feature_names,
        out_dir=args.out_dir,
        model_id=args.model_id,
        flags={"distilled_from": "teacher", "objective": args.objective},
    )

    # Register in local registry
    registry = LocalModelRegistry(Path(args.registry_dir))
    dtypes = ["float32"] * len(feature_names)
    fschema = build_feature_schema(feature_names, dtypes)
    manifest = build_student_manifest(
        model_id=args.model_id,
        architecture="LightGBM",
        feature_schema=fschema,
        feature_schema_hash=schema_hash(feature_names, dtypes),
        parent_id=args.parent_id,
        performance_metrics={"inference_latency_ms": 1.0},  # Placeholder; measure in prod
    )
    registry.register_model(Path(onnx_path), manifest, auto_deploy=True)

    print(f"Saved: {onnx_path}\nMeta: {meta_path}\nRegistered in: {args.registry_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
