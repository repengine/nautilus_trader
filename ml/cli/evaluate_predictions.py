#!/usr/bin/env python3
"""
Compute evaluation metrics (ROC AUC, PR AUC, logloss) from predictions.

This CLI reads a NumPy NPZ file containing either calibrated probabilities or
raw logits together with true labels, computes metrics, and writes a JSON
summary. Intended to feed `promote_features.py` via `--metrics_json`.

Inputs (NPZ conventions)
- Probabilities: keys {"q_val", "y_val_true"}
- Logits: keys {"z_val", "y_val_true"} (will apply sigmoid)

Examples
--------
1) Evaluate from probabilities:
   python -m ml.scripts.evaluate_predictions \
       --preds /tmp/teacher_preds.npz \
       --out_json /tmp/metrics.json

2) Evaluate from logits with custom keys:
   python -m ml.scripts.evaluate_predictions \
       --preds /tmp/logits.npz \
       --logits_key z \
       --y_key y \
       --out_json /tmp/metrics.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from ml._imports import HAS_SKLEARN
from ml._imports import check_ml_dependencies


if not HAS_SKLEARN:  # pragma: no cover - env guard, tests assume sklearn available
    check_ml_dependencies(["sklearn"])  # ensure clear message

from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def _flatten(arr: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(-1).astype(np.float64)
    return a.reshape(-1).astype(np.float64)


def _compute_metrics(y_true: NDArray[np.float64], p_pred: NDArray[np.float64]) -> dict[str, float]:
    # Cast to explicit dtypes
    from typing import cast as _cast
    y = y_true.astype(np.int32).reshape(-1)
    p = _cast(NDArray[np.float64], p_pred.astype(np.float64).reshape(-1))
    # Clip probabilities for logloss stability
    eps: Final[float] = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    metrics = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
    }
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate predictions and write metrics JSON")
    ap.add_argument("--preds", required=True, help="Path to NPZ with predictions and labels")
    ap.add_argument("--probs_key", default="q_val", help="Key for probabilities in NPZ")
    ap.add_argument("--logits_key", default="z_val", help="Key for logits in NPZ")
    ap.add_argument("--y_key", default="y_val_true", help="Key for true labels in NPZ")
    ap.add_argument("--out_json", required=True, help="Where to write metrics JSON")
    args = ap.parse_args(argv)

    data = np.load(args.preds)
    y_raw = data.get(args.y_key)
    if y_raw is None:
        raise SystemExit(f"Missing labels in NPZ: {args.y_key}")
    y_arr = np.asarray(y_raw)

    q_raw = data.get(args.probs_key)
    z_raw = data.get(args.logits_key)
    if q_raw is None and z_raw is None:
        raise SystemExit("Provide either probabilities (q_val) or logits (z_val) in NPZ")

    if q_raw is None:
        p_arr = _sigmoid(_flatten(np.asarray(z_raw)))
    else:
        p_arr = _flatten(np.asarray(q_raw))

    y = _flatten(np.asarray(y_arr))
    if y.shape[0] != p_arr.shape[0]:
        raise SystemExit(f"Length mismatch y={y.shape[0]} vs p={p_arr.shape[0]}")

    metrics = _compute_metrics(y, p_arr)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_json": str(out)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
