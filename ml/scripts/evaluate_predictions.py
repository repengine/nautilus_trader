#!/usr/bin/env python3
"""
Evaluate binary classifier predictions and print metrics JSON.

Inputs (NPZ or CSV):
- NPZ with keys {y_true, scores} or {y_true, probs}
- CSV with columns y_true and scores/probs

Outputs:
- Prints JSON with logloss, roc_auc, pr_auc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation.metrics import binary_logloss
from ml.evaluation.metrics import pr_auc
from ml.evaluation.metrics import roc_auc


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        y_true = data["y_true"].astype(np.float64)
        if "scores" in data:
            scores = data["scores"].astype(np.float64)
            return y_true, scores, "scores"
        if "probs" in data:
            probs = data["probs"].astype(np.float64)
            return y_true, probs, "probs"
        raise SystemExit("NPZ must contain either 'scores' or 'probs'")
    # CSV
    import pandas as pd  # local to avoid import at module import time

    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise SystemExit("CSV must contain y_true column")
    y_true = df["y_true"].to_numpy(dtype=np.float64)
    col = "scores" if "scores" in df.columns else ("probs" if "probs" in df.columns else None)
    if col is None:
        raise SystemExit("CSV must contain 'scores' or 'probs' column")
    arr = df[col].to_numpy(dtype=np.float64)
    return y_true, arr, col


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to NPZ/CSV with y_true and scores/probs")
    args = ap.parse_args(argv)

    y_true, arr, kind = _load_arrays(Path(args.preds))
    # Convert probs to scores (logits) for AUC/PR consistency if needed
    if kind == "probs":
        scores = np.log(np.clip(arr, 1e-6, 1 - 1e-6) / np.clip(1 - arr, 1e-6, 1 - 1e-6))
        probs = arr
    else:
        scores = arr
        probs = 1.0 / (1.0 + np.exp(-arr))

    metrics: dict[str, Any] = {
        "roc_auc": roc_auc(y_true, scores),
        "pr_auc": pr_auc(y_true, probs),
        "logloss": binary_logloss(y_true, probs),
    }
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

