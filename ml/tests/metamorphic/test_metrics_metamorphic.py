from __future__ import annotations

import numpy as np

from ml.evaluation.metrics import binary_logloss, roc_auc


def test_roc_auc_monotonic_transform_invariant() -> None:
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=200).astype(np.float64)
    s = rng.normal(size=200).astype(np.float64)
    auc1 = roc_auc(y, s)
    auc2 = roc_auc(y, 2.0 * s + 3.0)  # strictly increasing transform
    assert abs(auc1 - auc2) < 1e-9


def test_logloss_label_prob_complement_symmetry() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=100).astype(np.float64)
    p = np.clip(rng.random(size=100), 1e-6, 1 - 1e-6)
    ll = binary_logloss(y, p)
    ll_comp = binary_logloss(1.0 - y, 1.0 - p)
    assert abs(ll - ll_comp) < 1e-9

