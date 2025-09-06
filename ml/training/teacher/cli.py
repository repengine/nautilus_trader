from __future__ import annotations


# ruff: noqa: E402 - allow module docstring before imports in CLI script

"""
Teacher calibration CLI (compat shim) — forwards to tft_cli when registry args provided.

Two modes:
1) Legacy simple mode (no registries):
   - Inputs: --student_window_npz with keys {z_val, y_val_true}
   - Action: Platt-calibrate and emit q_train + meta

2) Registry-integrated mode (preferred):
   - Provide --feature_registry_dir, --feature_set_id, and optionally --model_registry_dir,
     --teacher_model_id, --onnx_output_is_logits. Arguments are forwarded to tft_cli.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


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

    def predict_logits(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
    ap = argparse.ArgumentParser(add_help=False)
    # Common
    ap.add_argument("--student_window_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    # Registry (optional to forward to tft_cli)
    ap.add_argument("--feature_registry_dir", required=False)
    ap.add_argument("--feature_set_id", required=False)
    ap.add_argument("--model_registry_dir", required=False)
    ap.add_argument("--teacher_model_id", required=False)
    ap.add_argument("--onnx_output_is_logits", action="store_true")
    # Parse only known; allow tft_cli to reparse if forwarded
    args, _ = ap.parse_known_args(argv)

    # If feature registry args are present, forward to tft_cli for full flow
    if args.feature_registry_dir and args.feature_set_id:
        from ml.training.teacher.tft_cli import main as tft_main

        return tft_main(argv)

    # Legacy simple mode (no registry)
    npz = np.load(args.student_window_npz, allow_pickle=True)
    if "z_val" not in npz or "y_val_true" not in npz:
        raise ValueError("student_window_npz must contain z_val and y_val_true arrays")
    z_val = npz["z_val"].astype(np.float64)
    y_val_true = npz["y_val_true"].astype(np.float64)

    teacher = CalibratingTeacher(TeacherConfig(architecture="TFT"))
    teacher.calibrate(z_val.reshape(-1, 1), y_val_true)
    q_cal = teacher.predict_proba(z_val.reshape(-1, 1)).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "teacher_preds.npz"
    np.savez_compressed(preds_path, q_train=q_cal.squeeze())
    meta_path = out_dir / "teacher_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "calibrator": True}, f, indent=2)

    print(
        f"[compat] Using legacy teacher CLI without registries\nSaved: {preds_path}\nMeta: {meta_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
