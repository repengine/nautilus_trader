#!/usr/bin/env python3
"""Create a dummy ONNX model for exercising the ML pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np

from ml._imports import HAS_ONNX
from ml._imports import HAS_ONNX_CORE
from ml._imports import check_ml_dependencies
from ml._imports import onnx
from ml._imports import ort


if TYPE_CHECKING:  # pragma: no cover - type-only hints
    from onnx import ModelProto as OnnxModel


logger = logging.getLogger(__name__)

_FEATURE_COUNT: int = 26


def _ensure_onnx_dependencies() -> tuple[ModuleType, ModuleType]:
    """Ensure ONNX and ONNX Runtime are available."""
    if onnx is None or ort is None or not (HAS_ONNX and HAS_ONNX_CORE):
        check_ml_dependencies(["onnx"])
    if onnx is None or ort is None:
        raise ImportError("ONNX runtime dependencies failed to import")
    return onnx, ort


def create_dummy_model() -> OnnxModel:
    """
    Create a dummy ONNX model that produces probabilistic predictions.

    Returns
    -------
    onnx.ModelProto
        In-memory ONNX model definition.
    """
    onnx_module, _ = _ensure_onnx_dependencies()

    helper = onnx_module.helper
    tensor_proto = onnx_module.TensorProto

    w_pred = np.random.randn(_FEATURE_COUNT, 1).astype(np.float32) * 0.05
    b_pred = np.array([0.0], dtype=np.float32)
    w_conf = np.random.randn(_FEATURE_COUNT, 1).astype(np.float32) * 0.05
    b_conf = np.array([0.5], dtype=np.float32)

    input_tensor = helper.make_tensor_value_info("features", tensor_proto.FLOAT, [1, _FEATURE_COUNT])
    out_pred = helper.make_tensor_value_info("pred", tensor_proto.FLOAT, [1])
    out_conf = helper.make_tensor_value_info("conf", tensor_proto.FLOAT, [1])

    t_w_pred = helper.make_tensor("W_pred", tensor_proto.FLOAT, [_FEATURE_COUNT, 1], w_pred.flatten().tolist())
    t_b_pred = helper.make_tensor("b_pred", tensor_proto.FLOAT, [1], b_pred.flatten().tolist())
    t_w_conf = helper.make_tensor("W_conf", tensor_proto.FLOAT, [_FEATURE_COUNT, 1], w_conf.flatten().tolist())
    t_b_conf = helper.make_tensor("b_conf", tensor_proto.FLOAT, [1], b_conf.flatten().tolist())

    n_mm_p = helper.make_node("MatMul", inputs=["features", "W_pred"], outputs=["mm_p"])
    n_add_p = helper.make_node("Add", inputs=["mm_p", "b_pred"], outputs=["add_p"])
    n_sq_p = helper.make_node("Squeeze", inputs=["add_p"], outputs=["sq_p"], axes=[1])
    n_sig_p = helper.make_node("Sigmoid", inputs=["sq_p"], outputs=["pred"])

    n_mm_c = helper.make_node("MatMul", inputs=["features", "W_conf"], outputs=["mm_c"])
    n_add_c = helper.make_node("Add", inputs=["mm_c", "b_conf"], outputs=["add_c"])
    n_sq_c = helper.make_node("Squeeze", inputs=["add_c"], outputs=["sq_c"], axes=[1])
    n_sig_c = helper.make_node("Sigmoid", inputs=["sq_c"], outputs=["conf"])

    graph_def = helper.make_graph(
        [n_mm_p, n_add_p, n_sq_p, n_sig_p, n_mm_c, n_add_c, n_sq_c, n_sig_c],
        "dummy_ml_model_26",
        [input_tensor],
        [out_pred, out_conf],
        [t_w_pred, t_b_pred, t_w_conf, t_b_conf],
    )

    model_def = helper.make_model(graph_def, producer_name="ml_pipeline_test")
    model_def.opset_import[0].version = 11
    model_def.ir_version = 7

    model_def.metadata_props.append(
        onnx_module.StringStringEntryProto(key="model_type", value="dummy_classifier_v26"),
    )
    model_def.metadata_props.append(
        onnx_module.StringStringEntryProto(
            key="description",
            value="Dummy 26-feature model for ML pipeline testing",
        ),
    )

    return cast(OnnxModel, model_def)


def verify_model(model_path: str) -> bool:
    """
    Verify the generated ONNX model using ONNX Runtime.

    Parameters
    ----------
    model_path : str
        Filesystem path to the model artifact.

    Returns
    -------
    bool
        True when inference succeeds.
    """
    _, ort_runtime = _ensure_onnx_dependencies()
    session = ort_runtime.InferenceSession(model_path)

    test_input = np.random.randn(1, _FEATURE_COUNT).astype(np.float32)
    outputs = session.run(None, {"features": test_input})

    print(f"Model input shape: {test_input.shape}")
    print(f"Model output shape: {outputs[0].shape}")
    print(f"Sample output: {outputs[0]}")

    return True


if __name__ == "__main__":
    model_dir = Path("ml/models")
    model_dir.mkdir(exist_ok=True, parents=True)

    model_path = model_dir / "dummy_bullish_model.onnx"
    onnx_module, _ = _ensure_onnx_dependencies()
    model = create_dummy_model()
    onnx_module.save(model, str(model_path))
    try:
        verify_model(str(model_path))
    except Exception as exc:  # pragma: no cover - defensive CLI logging
        logger.warning(
            "Dummy model verification failed; continuing without runtime validation",
            exc_info=True,
            extra={"model_path": str(model_path), "reason": str(exc)},
        )
    print(f"Dummy 26-feature model saved to: {model_path}")
