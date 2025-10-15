from __future__ import annotations

import pytest

from ml.deployment.security import assert_allowed_model_path


def test_onnx_model_allowed() -> None:
    # Should not raise
    assert_allowed_model_path("/models/model.ONNX")


@pytest.mark.parametrize("bad", [
    "/models/model.pkl",
    "/models/model.joblib",
    "/models/model.bin",
])
def test_non_onnx_rejected(bad: str) -> None:
    with pytest.raises(ValueError) as ei:
        assert_allowed_model_path(bad)
    assert "Only ONNX" in str(ei.value)
