from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from ml.distillation.lightgbm_student import LightGBMStudentDistiller


class FakeBooster:
    def __init__(self, w: float = 1.0) -> None:
        self.best_iteration = 3
        self._w = np.float32(w)

    def predict(
        self,
        X: np.ndarray,
        num_iteration: int | None = None,
        raw_score: bool | None = None,
    ) -> np.ndarray:
        # Linear score z = w * sum(features)
        return (self._w * X.sum(axis=1)).astype(np.float32)


@given(
    n=st.integers(min_value=1, max_value=32),
    d=st.integers(min_value=1, max_value=16),
)
def test_student_predict_proba_bounds_and_shape(n: int, d: int) -> None:
    dist = LightGBMStudentDistiller()
    dist.model = FakeBooster()
    X = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    p = dist.predict_proba(X)
    assert p.shape == (n, 1)
    assert p.dtype == np.float32
    assert np.all(p > 0.0) and np.all(p < 1.0)


@given(
    a=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_platt_monotonicity_sign(a: float, b: float) -> None:
    # For a>0, probability increases with z; for a<0, decreases
    dist = LightGBMStudentDistiller()
    dist.model = FakeBooster(w=1.0)
    dist._calibrator_kind = "platt"
    dist._platt_coef = np.float32(a)
    dist._platt_intercept = np.float32(b)
    X = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    p = dist.predict_proba(X).ravel()
    if a > 0:
        assert p[0] <= p[1] <= p[2]
    elif a < 0:
        assert p[0] >= p[1] >= p[2]
    else:
        # a == 0 -> constant probability regardless of z (sigmoid(b))
        assert np.allclose(p[0], p[1]) and np.allclose(p[1], p[2])
