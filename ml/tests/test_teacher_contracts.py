from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from ml.models.teacher import BaseTeacher
from ml.models.teacher import TeacherConfig


class FakeTeacher(BaseTeacher):
    def __init__(self) -> None:
        super().__init__(TeacherConfig(architecture="FAKE"))

    def fit(self, dataset: object) -> FakeTeacher:
        self._is_fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        # Simple linear score for tests: z = sum(features)
        return X.sum(axis=1).astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        return {"f1": "float32", "f2": "float32"}


@given(
    st.lists(
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
)
def test_teacher_predict_proba_monotonic(xs: list[float]) -> None:
    X = np.array(xs, dtype=np.float64).reshape(-1, 1)
    t = FakeTeacher().fit(object())
    z = t.predict_logits(X)
    p = t.predict_proba(X)
    # Sigmoid(z) monotonic: sort z, p follows
    order = np.argsort(z)
    assert np.all(np.diff(p[order].ravel()) >= 0)
