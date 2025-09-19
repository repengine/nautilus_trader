from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig


def _require_onnx_runtime() -> tuple[Any, Any]:
    try:
        import onnx  # type: ignore
        import onnx.helper as oh  # type: ignore
        import onnx.numpy_helper as onh  # type: ignore
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"onnx/onnxruntime not available: {exc}")
        raise
    return (onnx, ort)


def _build_minimal_sum_model(feature_dim: int, model_path: Path) -> None:
    import onnx  # type: ignore
    import onnx.helper as oh  # type: ignore
    import onnx.numpy_helper as onh  # type: ignore

    # Inputs/Outputs
    features = oh.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [None, feature_dim])
    pred_out = oh.make_tensor_value_info("pred", onnx.TensorProto.FLOAT, [None])
    conf_out = oh.make_tensor_value_info("conf", onnx.TensorProto.FLOAT, [None])

    # Nodes: pred = ReduceSum(features, axis=1); conf = (pred * 0.0) + 0.9 (broadcast)
    axis_tensor = onh.from_array(np.array([1], dtype=np.int64), name="axis")
    zero = onh.from_array(np.array(0.0, dtype=np.float32), name="zero")
    nine = onh.from_array(np.array(0.9, dtype=np.float32), name="nine")

    node_reduce = oh.make_node("ReduceSum", inputs=["features", "axis"], outputs=["pred"], keepdims=0)
    node_mul = oh.make_node("Mul", inputs=["pred", "zero"], outputs=["zero_vec"])  # broadcast 0 over pred shape
    node_add = oh.make_node("Add", inputs=["zero_vec", "nine"], outputs=["conf"])  # broadcast 0.9 over shape

    graph = oh.make_graph(
        nodes=[node_reduce, node_mul, node_add],
        name="SumThenConstConf",
        inputs=[features],
        outputs=[pred_out, conf_out],
        initializer=[axis_tensor, zero, nine],
        value_info=[oh.make_tensor_value_info("zero_vec", onnx.TensorProto.FLOAT, [None])],
    )

    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 13)])
    # Adjust IR version for broad onnxruntime compatibility
    model.ir_version = 10
    onnx.checker.check_model(model)
    model_path.write_bytes(model.SerializeToString())


@pytest.mark.integration
def test_vectorized_infer_with_real_onnxruntime(tmp_path: Path) -> None:
    _onnx, ort = _require_onnx_runtime()

    feature_dim = 4
    model_file = tmp_path / "mini_sum.onnx"
    _build_minimal_sum_model(feature_dim, model_file)

    # Prepare actor (we will assign the ORT session directly)
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="onnx-integration",
        max_batch_size=8,
        feature_dim=feature_dim,
        use_dummy_stores=True,
        model_path=str(model_file),
        model_id="mini_sum",
        instrument_id=None,  # not used here
        bar_type=None,  # not used here
    )
    actor = MultiInstrumentSignalActor(cfg)  # type: ignore[arg-type]

    # Load ORT session and wire metadata
    session = ort.InferenceSession(str(model_file))
    meta = {
        "input_names": [i.name for i in session.get_inputs()],
        "output_names": [o.name for o in session.get_outputs()],
    }
    setattr(actor, "_model", session)
    setattr(actor, "_model_metadata", meta)

    batch: npt.NDArray[np.float32] = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # sum = 10
            [0.5, 0.5, 0.0, 0.0],  # sum = 1.0
        ],
        dtype=np.float32,
    )
    preds, confs = actor._infer_batch(batch)
    np.testing.assert_allclose(preds, np.array([10.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(confs, np.array([0.9, 0.9], dtype=np.float32))
