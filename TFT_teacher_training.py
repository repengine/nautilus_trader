"""
CLI to train a TFT teacher and dump predictions for student distillation.

This is a simplified scaffold. In practice you'll build a TimeSeriesDataSet from your
parquet store and your feature engineering pipeline; here we assume you've already
prepared tensors/arrays for demonstration.

Inputs
------
- teacher_features.npz: contains arrays needed by your TFT (left abstract here)
- student_window.npz:   contains `p_raw_val` (raw prob outputs on student window) and `y_val_true` for calibration

Outputs
-------
- teacher_meta.json
- teacher_preds.npz  (q_train for student training; optionally logits z_T)

"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .TFT_teacher_model import TFTTeacher
from .TFT_teacher_model import TFTTeacherConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_features_npz", required=False, help="placeholder in this scaffold")
    ap.add_argument(
        "--student_window_npz",
        required=True,
        help=".npz with p_raw_val and y_val_true",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    args = ap.parse_args()

    # In a real pipeline, you'd build a TimeSeriesDataSet and fit the model.
    # For now, we only perform calibration over provided raw probabilities on the student window.
    npz = np.load(args.student_window_npz, allow_pickle=True)
    p_raw_val = npz["p_raw_val"].astype(np.float32)
    y_val_true = npz["y_val_true"].astype(np.float32)

    teacher = TFTTeacher(TFTTeacherConfig())
    teacher.calibrate(p_raw_val, y_val_true)  # fit Platt/Isotonic on provided window
    q_cal = teacher.predict_proba(p_raw_val)  # calibrated probs to use as distillation targets

    os.makedirs(args.out_dir, exist_ok=True)
    preds_path = os.path.join(args.out_dir, "teacher_preds.npz")
    np.savez_compressed(preds_path, q_train=q_cal.squeeze())
    meta_path = os.path.join(args.out_dir, "teacher_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "calibrator": True}, f, indent=2)

    print("Saved:", preds_path, meta_path)


if __name__ == "__main__":
    main()
