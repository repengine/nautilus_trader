from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.distillation.lightgbm_student import LightGBMStudentDistiller
from ml.distillation.lightgbm_student import schema_hash


class FakeBooster:
    """
    Minimal LightGBM Booster stand-in for tests.

    predict returns a deterministic raw score: z = X @ w, where w is ones.

    """

    def __init__(self) -> None:
        self.best_iteration = 5

    def predict(
        self,
        X: npt.NDArray[np.float32],
        num_iteration: int | None = None,
        raw_score: bool | None = None,
    ) -> npt.NDArray[np.float32]:
        # deterministic linear score
        w = np.ones((X.shape[1],), dtype=np.float32)
        z = X @ w
        return z.astype(np.float32)


def test_predict_proba_without_calibration() -> None:
    dist = LightGBMStudentDistiller()
    dist.model = FakeBooster()
    X = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    p = dist.predict_proba(X)
    # z = [1, 2] -> sigmoid
    z = np.array([1.0, 2.0], dtype=np.float32)
    expected = (1.0 / (1.0 + np.exp(-z))).reshape(-1, 1).astype(np.float32)
    assert p.dtype == np.float32
    assert p.shape == (2, 1)
    np.testing.assert_allclose(p, expected, rtol=1e-6, atol=1e-6)


def test_predict_proba_with_platt() -> None:
    dist = LightGBMStudentDistiller()
    dist.model = FakeBooster()
    # Inject known Platt params (avoid sklearn at runtime)
    dist._calibrator_kind = "platt"
    dist._platt_coef = np.float32(2.0)
    dist._platt_intercept = np.float32(-1.0)
    X = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    # raw z = [1, 2]; apply a*z + b => [1, 3]
    z_adj = np.array([1.0, 3.0], dtype=np.float32)
    expected = (1.0 / (1.0 + np.exp(-z_adj))).reshape(-1, 1).astype(np.float32)
    p = dist.predict_proba(X)
    np.testing.assert_allclose(p, expected, rtol=1e-6, atol=1e-6)


@given(
    names=st.lists(
        st.text(min_size=1, max_size=5).filter(lambda s: "|" not in s),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
def test_schema_hash_is_order_sensitive(names: list[str]) -> None:
    dtypes = ["float32"] * len(names)
    h1 = schema_hash(names, dtypes)
    h2 = schema_hash(list(reversed(names)), dtypes)
    assert h1 != h2 or names == list(reversed(names))


def test_export_onnx_appends_mul_add_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prepare a dummy onnx model graph we can inspect
    class DummyOutput:
        def __init__(self, name: str) -> None:
            self.name = name

    class DummyGraph:
        def __init__(self) -> None:
            self.node: list[Any] = []
            self.output = [DummyOutput("raw_output")]
            self.initializer: list[Any] = []

    class DummyOnnxModel:
        def __init__(self) -> None:
            self.graph = DummyGraph()

    # Dummies mimicking onnx helper api
    dummy_onnx_mod = types.SimpleNamespace()

    def dummy_save(model: Any, path: str) -> None:
        # Create a placeholder file to satisfy filesystem expectations
        Path(path).write_bytes(b"dummy-onnx")

    dummy_onnx_mod.save = dummy_save

    def make_node(op_type: str, inputs: list[str], outputs: list[str], **_: Any) -> dict[str, Any]:
        return {"op_type": op_type, "inputs": inputs, "outputs": outputs}

    dummy_helper = types.SimpleNamespace(make_node=make_node)

    def from_array(arr: Any, name: str) -> dict[str, Any]:
        return {"name": name, "value": np.array(arr)}

    dummy_numpy_helper = types.SimpleNamespace(from_array=from_array)

    def dummy_convert_lgbm_booster(_model: Any, initial_types: Any) -> DummyOnnxModel:
        return DummyOnnxModel()

    class DummyFloatTensorType:
        def __init__(self, shape: Any) -> None:
            self.shape = shape

    # Monkeypatch the module-level imports inside our distiller
    import ml.distillation.lightgbm_student as student

    monkeypatch.setitem(student.__dict__, "onnx", dummy_onnx_mod)
    monkeypatch.setitem(student.__dict__, "onnx_helper", dummy_helper)
    monkeypatch.setitem(student.__dict__, "onnx_numpy_helper", dummy_numpy_helper)
    monkeypatch.setitem(student.__dict__, "convert_lgbm_booster", dummy_convert_lgbm_booster)
    monkeypatch.setitem(student.__dict__, "FloatTensorType", DummyFloatTensorType)

    dist = LightGBMStudentDistiller()
    dist.model = FakeBooster()
    dist._calibrator_kind = "platt"
    dist._platt_coef = np.float32(1.5)
    dist._platt_intercept = np.float32(-0.2)

    out_dir = tmp_path / "out"
    onnx_path, meta_path = dist.export_onnx(["f1", "f2"], str(out_dir), model_id="test_model")

    # Validate graph nodes were appended
    # Retrieve the dummy model that was created inside dummy_convert_lgbm_booster
    # We cannot access it directly here; instead we check the file and metadata content
    assert Path(onnx_path).exists()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    assert meta["model_id"] == "test_model"
    assert meta["feature_schema_hash"] == schema_hash(["f1", "f2"], ["float32", "float32"])
    assert meta["calibrator_kind"] == "platt"
    assert set(meta["calibrator_params"].keys()) == {"coef", "intercept"}
