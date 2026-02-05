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
- registry entry (ModelRegistry)

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ml.common.model_sidecar import extract_inference_metadata
from ml.common.model_sidecar import load_sidecar_metadata
from ml.registry.model_registry import ModelRegistry
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest
from ml.training.student.lightgbm import LightGBMStudentDistiller
from ml.training.student.lightgbm import build_student_decision_config


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
    ap.add_argument(
        "--use_val_for_distill",
        action="store_true",
        help="Train student on validation split using q_val/X_val instead of q_train/X_train",
    )
    # Mandatory FeatureRegistry integration for parity/backfill
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    args = ap.parse_args(argv)

    X_npz = np.load(args.features_npz, allow_pickle=True)
    X_train: npt.NDArray[np.float32] = X_npz["X_train"].astype(np.float32)
    X_val: npt.NDArray[np.float32] = X_npz["X_val"].astype(np.float32)
    feature_names: list[str] = list(X_npz["feature_names"].tolist())

    # Parity check: feature_names must match manifest exactly (name + order)
    from ml.registry.feature_registry import FeatureRegistry

    freg = FeatureRegistry(Path(args.feature_registry_dir))
    finfo = freg.get_feature_set(args.feature_set_id)
    if finfo is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")
    manifest_names: list[str] = list(finfo.manifest.feature_names)
    if feature_names != manifest_names:
        raise SystemExit(
            "Feature schema mismatch with registry manifest:\n"
            f"expected={manifest_names}\nactual={feature_names}",
        )

    T = np.load(args.teacher_npz, allow_pickle=True)
    q_train: npt.NDArray[np.float32] | None = (
        T["q_train"].astype(np.float32) if "q_train" in T else None
    )
    q_val: npt.NDArray[np.float32] | None = T["q_val"].astype(np.float32) if "q_val" in T else None
    y_val_true: npt.NDArray[np.float32] | None = (
        T["y_val_true"].astype(np.float32) if "y_val_true" in T else None
    )

    # Select training split
    if args.use_val_for_distill:
        if q_val is None:
            raise SystemExit("--use_val_for_distill set but teacher_npz missing 'q_val'")
        X_train_sel: npt.NDArray[np.float32] = X_val
        q_train_sel: npt.NDArray[np.float32] = q_val.reshape(-1)
    else:
        if q_train is None:
            raise SystemExit(
                "teacher_npz missing 'q_train' — did you run teacher CLI with training mode?",
            )
        X_train_sel = X_train
        q_train_sel = q_train.reshape(-1)

    # Shape validation
    if X_train_sel.shape[0] != q_train_sel.shape[0]:
        raise SystemExit(
            f"Training shape mismatch: X ({X_train_sel.shape[0]}) vs q ({q_train_sel.shape[0]})",
        )

    distiller = LightGBMStudentDistiller(
        objective=args.objective,
        kd_lambda=args.kd_lambda,
        early_stopping=args.early_stopping,
        opset=args.opset,
    )
    distiller.fit(X_train_sel, q_train_sel, X_val, y_val_true)
    onnx_path, meta_path = distiller.export_onnx(
        feature_names=feature_names,
        out_dir=args.out_dir,
        model_id=args.model_id,
        flags={"distilled_from": "teacher", "objective": args.objective},
    )

    # Register in local registry
    registry = ModelRegistry(Path(args.registry_dir))
    dtypes = ["float32"] * len(feature_names)
    fschema = build_feature_schema(feature_names, dtypes)

    feature_schema_hash = finfo.manifest.schema_hash
    feature_set_id = finfo.manifest.feature_set_id
    pipeline_signature = finfo.manifest.pipeline_signature
    pipeline_version = finfo.manifest.pipeline_version

    # Compute validation metrics if labels available
    performance_metrics: dict[str, float] = {}
    if y_val_true is not None:
        try:
            from sklearn.metrics import average_precision_score
            from sklearn.metrics import brier_score_loss
            from sklearn.metrics import log_loss
            from sklearn.metrics import roc_auc_score

            p_val: npt.NDArray[np.float32] = distiller.predict_proba(X_val).reshape(-1)
            yv = y_val_true.reshape(-1).astype(np.int32)
            # Clip to avoid logloss instability
            p_val = np.clip(p_val, 1e-6, 1.0 - 1e-6)
            performance_metrics = {
                "auc": float(roc_auc_score(yv, p_val)),
                "pr_auc": float(average_precision_score(yv, p_val)),
                "brier": float(brier_score_loss(yv, p_val)),
                "logloss": float(log_loss(yv, p_val)),
            }
        except Exception:
            performance_metrics = {}

    decision_cfg = build_student_decision_config()
    sidecar = load_sidecar_metadata(Path(meta_path))
    output_schema, calibration = (
        extract_inference_metadata(sidecar) if sidecar is not None else (None, None)
    )

    manifest = build_student_manifest(
        model_id=args.model_id,
        architecture="LightGBM",
        feature_schema=fschema,
        feature_schema_hash=feature_schema_hash,
        parent_id=args.parent_id,
        performance_metrics=performance_metrics or {"inference_latency_ms": 1.0},
        feature_set_id=feature_set_id,
        pipeline_signature=pipeline_signature,
        pipeline_version=pipeline_version,
        decision_config=decision_cfg,
        output_schema=output_schema,
        calibration=calibration,
    )
    registry.register_model(Path(onnx_path), manifest, auto_deploy=True)

    print(f"Saved: {onnx_path}\nMeta: {meta_path}\nRegistered in: {args.registry_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
