from __future__ import annotations

"""
CLI scaffold to calibrate a teacher on a validation window and emit soft labels
for student distillation. This is a cold-path utility.

Inputs
------
- --student_window_npz: .npz with z_val (raw logits) and y_val_true (0/1)
- --out_dir:            directory to write outputs
- --model_id:           teacher model id

Outputs
-------
- teacher_preds.npz with q_train (calibrated probabilities from z_val)
- teacher_meta.json with minimal metadata
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.models.teacher import BaseTeacher
from ml.models.teacher import TeacherConfig


class CalibratingTeacher(BaseTeacher):
    """
    Teacher model that calibrates raw logits.
    """

    def fit(self, dataset: object) -> CalibratingTeacher:
        """
        Fit the calibrating teacher (no-op for this scaffold).

        Parameters
        ----------
        dataset : object
            Dataset (unused in this scaffold).

        Returns
        -------
        CalibratingTeacher
            Self for chaining.

        """
        # No-op: this scaffold expects raw logits provided externally
        self._is_fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Pass through raw logits.

        Parameters
        ----------
        X : np.ndarray
            Raw logits.

        Returns
        -------
        np.ndarray
            Logits as float64.

        """
        # Identity passthrough: X is already z
        return X.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        """
        Get feature schema.

        Returns
        -------
        dict[str, str]
            Empty schema.

        """
        return {}


def main(argv: list[str] | None = None) -> int:
    """
    Run teacher calibration CLI.

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
    ap.add_argument("--student_window_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    args = ap.parse_args(argv)

    npz = np.load(args.student_window_npz, allow_pickle=True)
    if "z_val" not in npz or "y_val_true" not in npz:
        raise ValueError("student_window_npz must contain z_val and y_val_true arrays")
    z_val = npz["z_val"].astype(np.float64)
    y_val_true = npz["y_val_true"].astype(np.float64)

    teacher = CalibratingTeacher(TeacherConfig(architecture="TFT"))
    teacher.calibrate(z_val.reshape(-1, 1), y_val_true)

    # Produce calibrated soft labels on the provided window
    q_cal = teacher.predict_proba(z_val.reshape(-1, 1)).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "teacher_preds.npz"
    np.savez_compressed(preds_path, q_train=q_cal.squeeze())
    meta_path = out_dir / "teacher_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "calibrator": True}, f, indent=2)

    print(f"Saved: {preds_path}\nMeta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
