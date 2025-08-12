#!/usr/bin/env python3

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_ONNX


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX not installed")
def test_onnx_runtime_session_load_and_run() -> None:
    import onnx
    from onnx import TensorProto
    from onnx import helper

    from ml.actors.base import ProductionModelLoader
    from ml.training.export import DEFAULT_ONNX_OPSET

    # Build a minimal ONNX model: y = Identity(x)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    node = helper.make_node("ReduceSum", inputs=["input"], outputs=["output"], keepdims=1, axes=[1])
    graph = helper.make_graph([node], "sum_model", [input_tensor], [output_tensor])
    # Use opset that's compatible with our ONNX runtime
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", min(DEFAULT_ONNX_OPSET, 10))],
    )
    model.ir_version = 10  # Explicitly set IR version for compatibility with older ONNX runtime

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        onnx.save(model, str(onnx_path))

        loader = ProductionModelLoader()
        session, metadata = loader.load_model(str(onnx_path))

        X = np.random.randn(3, 4).astype(np.float32)
        outputs = session.run(None, {metadata["input_names"][0]: X})
        assert outputs[0].shape == (3, 1)
