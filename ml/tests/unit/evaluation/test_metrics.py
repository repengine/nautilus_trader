from __future__ import annotations

import numpy as np

from ml.evaluation.metrics import binary_logloss, pr_auc, roc_auc


def test_binary_logloss_simple() -> None:
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    p = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float64)
    ll = binary_logloss(y, p)
    assert ll > 0
    # Perfect predictions should be close to zero
    ll_perfect = binary_logloss(y, np.array([0.0, 1.0, 1.0, 0.0]))
    assert ll_perfect < 1e-6


def test_roc_auc_monotonic_scores() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    auc = roc_auc(y, s)
    assert 0.99 < auc <= 1.0


def test_pr_auc_reasonable() -> None:
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float64)
    s = np.array([0.1, 0.8, 0.7, 0.2, 0.6, 0.3, 0.4, 0.9], dtype=np.float64)
    area = pr_auc(y, s)
    assert 0.0 <= area <= 1.0
