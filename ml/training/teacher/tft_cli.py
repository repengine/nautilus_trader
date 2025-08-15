from __future__ import annotations

"""
CLI to calibrate a TFT teacher and emit soft labels for distillation, with registry integration.

This CLI integrates with:
- Local Feature Registry: resolve feature schema and enforce feature parity
- Local Model Registry: optionally load a teacher ONNX to produce raw logits

Inputs (NPZ conventions)
- If passing precomputed logits: keys {"z_val", "y_val_true"}
- If using a teacher ONNX: keys {"X_val", "y_val_true"}; ensure X_val columns
  are ordered to match the feature manifest (or provide the same order in file)

Outputs
- teacher_preds.npz: contains q_train (calibrated probabilities) and y_val_true
- teacher_meta.json: includes model_id, feature_set_id, teacher_model_id, schema hash
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ml.config.names import ONNX_INPUT_NAME
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.model_registry import ModelRegistry
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


class CalibratingTeacher(BaseTeacher):
    def fit(self, dataset: object) -> CalibratingTeacher:
        self._is_fitted = True
        return self

    def predict_logits(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return X.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # Data/outputs
    ap.add_argument(
        "--student_window_npz",
        required=True,
        help="NPZ with either {z_val,y_val_true} or {X_val,y_val_true}",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    # Feature registry integration
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    # Model registry integration (optional ONNX teacher)
    ap.add_argument("--model_registry_dir", required=False)
    ap.add_argument("--teacher_model_id", required=False)
    ap.add_argument(
        "--onnx_output_is_logits",
        action="store_true",
        help="Interpret ONNX model output as logits (else as probabilities)",
    )
    args = ap.parse_args(argv)

    # Resolve feature manifest and enforce schema
    freg = FeatureRegistry(Path(args.feature_registry_dir))
    finfo = freg.get_feature_set(args.feature_set_id)
    if finfo is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")
    fman = finfo.manifest
    feature_names = list(fman.feature_names)
    n_features = len(feature_names)

    # Load arrays
    npz = np.load(args.student_window_npz, allow_pickle=True)
    y_val_true = None
    if "y_val_true" in npz:
        y_val_true = npz["y_val_true"].astype(np.float64)
    else:
        raise SystemExit("NPZ missing required key 'y_val_true'")

    z_val: npt.NDArray[np.float64] | None = None
    X_val: npt.NDArray[np.float32] | None = None
    if "z_val" in npz:
        z_val = npz["z_val"].astype(np.float64)
    elif "X_val" in npz:
        X_val = np.asarray(npz["X_val"], dtype=np.float32)
        if X_val.ndim != 2 or X_val.shape[1] != n_features:
            raise SystemExit(
                f"X_val shape {X_val.shape} does not match feature manifest width {n_features}",
            )
    else:
        raise SystemExit("NPZ must contain either 'z_val' or 'X_val' with 'y_val_true'")

    # Optionally load teacher ONNX to produce logits if X_val provided
    if X_val is not None:
        if not args.model_registry_dir or not args.teacher_model_id:
            raise SystemExit(
                "Provide --model_registry_dir and --teacher_model_id to run ONNX teacher on X_val",
            )
        mreg = ModelRegistry(Path(args.model_registry_dir))
        session = mreg.load_model(args.teacher_model_id)
        if session is None:
            raise SystemExit(f"Failed to load teacher model {args.teacher_model_id} from registry")
        # Run inference
        try:
            input_name = (
                session.get_inputs()[0].name if hasattr(session, "get_inputs") else ONNX_INPUT_NAME
            )
            outputs: list[Any] = session.run(None, {input_name: X_val})
            raw_out = outputs[0]
            raw = np.asarray(raw_out, dtype=np.float64).reshape(-1)
            if args.onnx_output_is_logits:
                z_val = raw
            else:
                # Assume probabilities; convert to logits for calibration pipeline
                eps = 1e-6
                p = np.clip(raw, eps, 1.0 - eps)
                z_val = np.log(p / (1.0 - p))
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise SystemExit(f"ONNX inference failed: {exc}")

    assert z_val is not None

    # Calibrate and produce calibrated probabilities
    teacher = CalibratingTeacher(TeacherConfig(architecture="TFT"))
    teacher.calibrate(z_val.reshape(-1, 1), y_val_true)
    q_cal = teacher.predict_proba(z_val.reshape(-1, 1)).astype(np.float32)

    # Persist outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "teacher_preds.npz"
    # Save both q_train and y_val_true for student calibration convenience
    np.savez_compressed(
        preds_path,
        q_train=q_cal.squeeze(),
        y_val_true=y_val_true.astype(np.float32),
    )
    meta_path = out_dir / "teacher_meta.json"
    meta = {
        "model_id": args.model_id,
        "feature_set_id": args.feature_set_id,
        "feature_schema_hash": fman.schema_hash,
        "teacher_model_id": args.teacher_model_id,
        "calibrator": True,
        "onnx_output_is_logits": bool(args.onnx_output_is_logits),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {preds_path}\nMeta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
