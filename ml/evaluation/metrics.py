"""
Lightweight binary classification metrics without heavy dependencies.

Provides ROC AUC, PR AUC (approx), and log loss using numpy only.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def binary_logloss(
    y_true: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    eps: float = 1e-12,
) -> float:
    y = y_true.astype(np.float64).reshape(-1)
    p = np.clip(p.reshape(-1), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y_true: npt.NDArray[np.float64], scores: npt.NDArray[np.float64]) -> float:
    y = y_true.astype(np.float64).reshape(-1)
    s = scores.astype(np.float64).reshape(-1)
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_sorted) - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.0
    # Mann-Whitney U statistic
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    u = ranks[y_sorted == 1.0].sum() - n_pos * (n_pos + 1.0) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def pr_auc(
    y_true: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    num_thresholds: int = 200,
) -> float:
    y = y_true.astype(np.float64).reshape(-1)
    s = scores.astype(np.float64).reshape(-1)
    if y.sum() == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precisions = []
    recalls = []
    for t in thresholds:
        preds = (s >= t).astype(np.float64)
        tp = float((preds * y).sum())
        fp = float((preds * (1 - y)).sum())
        fn = float(((1 - preds) * y).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    # Sort by recall
    order = np.argsort(recalls)
    r = np.asarray(recalls, dtype=np.float64)[order]
    p = np.asarray(precisions, dtype=np.float64)[order]
    # Compute area via trapezoidal rule
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:  # NumPy >= 2.0
        area = trapezoid(p, r)
    else:  # NumPy < 2.0
        area = np.trapz(p, r)  # noqa: NPY201
    return float(area)


def expected_calibration_error(
    probabilities: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    *,
    bins: int = 20,
) -> float:
    """
    Compute expected calibration error (ECE) using equal-width bins.

    Args:
        probabilities: Model probabilities in the range [0, 1].
        targets: Binary labels aligned with ``probabilities``.
        bins: Number of histogram bins to use when comparing confidence vs. accuracy.

    Returns:
        Expected calibration error in the range [0, 1].
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    probs = np.clip(probabilities.astype(np.float64).reshape(-1), 0.0, 1.0)
    y = targets.astype(np.float64).reshape(-1)
    if probs.size == 0 or y.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probs, bin_edges[1:-1], right=True)
    total = float(probs.size)
    ece = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        count = float(np.count_nonzero(mask))
        if count == 0.0:
            continue
        bin_probs = probs[mask]
        bin_targets = y[mask]
        confidence = float(bin_probs.mean())
        accuracy = float(bin_targets.mean())
        ece += abs(accuracy - confidence) * (count / total)
    return float(ece)
