#!/usr/bin/env python3
"""
Simple promotion gate for teacher predictions.

Reads a ``teacher_preds.npz`` file (as produced by the TFT teacher CLI) and
evaluates validation metrics (AUC, PR-AUC, LogLoss, Brier). If the metrics pass
the configured gates, exits with code 0; otherwise exits with code 2.

Example:
    $ python -m ml.scripts.promote_model_if_metrics_pass \
        --teacher_npz /tmp/run/teacher_preds.npz \
        --min_auc 0.56 \
        --min_pr_auc_multiple 1.5

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promotion gate for TFT teacher metrics")
    ap.add_argument("--teacher_npz", required=True, help="Path to teacher_preds.npz")
    ap.add_argument("--min_auc", type=float, default=0.56)
    ap.add_argument(
        "--min_pr_auc_multiple",
        type=float,
        default=1.5,
        help="PR-AUC must be at least this multiple of prevalence baseline",
    )
    args = ap.parse_args(argv)

    npz = _load_npz(Path(args.teacher_npz))
    q_val = npz.get("q_val")
    y_val = npz.get("y_val_true")
    if q_val is None or y_val is None:
        print("teacher_npz missing q_val or y_val_true", file=sys.stderr)
        return 2

    p = np.asarray(q_val, dtype=float).reshape(-1)
    y = np.asarray(y_val, dtype=int).reshape(-1)
    if p.size == 0 or y.size == 0:
        print("empty validation arrays", file=sys.stderr)
        return 2

    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    auc = float(roc_auc_score(y, p))
    pr = float(average_precision_score(y, p))
    ll = float(log_loss(y, p))
    br = float(brier_score_loss(y, p))
    prev = float(y.mean())
    pr_multiple = pr / prev if prev > 0 else 0.0

    metrics = {
        "AUC": auc,
        "PR_AUC": pr,
        "PR_AUC_multiple": pr_multiple,
        "LogLoss": ll,
        "Brier": br,
        "Prevalence": prev,
    }
    print(metrics)

    if auc >= args.min_auc and pr_multiple >= args.min_pr_auc_multiple:
        print("PROMOTE: pass gates")
        return 0
    print("PROMOTE: failed gates", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
