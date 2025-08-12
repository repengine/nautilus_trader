"""
CLI to train a LightGBM student by distilling a teacher (e.g., TFT).

Inputs
------
- Features (X) for train/val: numpy .npz with arrays X_train, X_val, feature_names
- Teacher outputs: .npz with q_train (probs), and optionally y_val_true (0/1)
  (align timestamps before saving these arrays)

Outputs
-------
- ONNX: student.onnx
- Sidecar: student.meta.json
- (Optional) acceptance.json (framework vs ORT parity) if onnxruntime is available

"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .lightgbm_student_model import LightGBMStudentDistiller


# Optional ONNXRuntime for acceptance test
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


def _acceptance_test(
    onnx_path: str,
    X_val: np.ndarray,
    p_val_framework: np.ndarray,
    tol: float = 1e-5,
) -> dict:
    if ort is None:
        return {"skipped": True, "reason": "onnxruntime not installed"}
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # keep simple
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_val.astype(np.float32)})
    p_onnx = outputs[0].astype(np.float32)
    diff = np.max(np.abs(p_onnx - p_val_framework))
    return {"skipped": False, "max_abs_diff": float(diff), "pass": bool(diff <= tol)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features_npz",
        required=True,
        help=".npz with X_train, X_val, feature_names (list)",
    )
    ap.add_argument("--teacher_npz", required=True, help=".npz with q_train, (optional) y_val_true")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--model_id", required=True, help="e.g., es_dir_3s_student_v1")
    ap.add_argument("--objective", default="logit_mse", choices=["logit_mse", "soft_ce", "hybrid"])
    ap.add_argument("--kd_lambda", type=float, default=0.5)
    ap.add_argument("--early_stopping", type=int, default=200)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

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
    # Framework proba on validation
    p_val = distiller.predict_proba(X_val)

    onnx_path, meta_path = distiller.export_onnx(
        feature_names=feature_names,
        out_dir=args.out_dir,
        model_id=args.model_id,
        train_date_range=None,
        flags={"distilled_from": "TFT"},
    )

    acc = _acceptance_test(onnx_path, X_val, p_val)
    with open(os.path.join(args.out_dir, "acceptance.json"), "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=2)

    print("Saved:", onnx_path, meta_path)
    if not acc.get("skipped"):
        print("ONNX parity:", acc)


if __name__ == "__main__":
    main()
