#!/usr/bin/env python3
"""
Create a dummy ONNX model for testing the ML pipeline.
This model generates random predictions to test the full system.
"""

import os
from pathlib import Path

import numpy as np


# Check if onnx is available
try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto
    from onnx import helper
except ImportError:
    print("Installing required packages...")
    os.system("pip install onnx onnxruntime")  # noqa: S605
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto
    from onnx import helper

def create_dummy_model() -> "onnx.ModelProto":
    """
    Create a dummy ONNX model that:
    - Takes 30 features as input
    - Outputs 3 values: [probability, direction, confidence]
    - Uses random weights for testing
    """
    # Define inputs/outputs (26-feature model with separate pred/conf outputs)
    num_features = 26

    # Create random weights for simple linear outputs (pred, conf)
    w_pred = np.random.randn(num_features, 1).astype(np.float32) * 0.05
    b_pred = np.array([0.0], dtype=np.float32)
    w_conf = np.random.randn(num_features, 1).astype(np.float32) * 0.05
    b_conf = np.array([0.5], dtype=np.float32)

    # IO tensors
    input_tensor = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, num_features])
    out_pred = helper.make_tensor_value_info("pred", TensorProto.FLOAT, [1])
    out_conf = helper.make_tensor_value_info("conf", TensorProto.FLOAT, [1])

    # Initializers
    t_w_pred = helper.make_tensor("W_pred", TensorProto.FLOAT, [num_features, 1], w_pred.flatten().tolist())
    t_b_pred = helper.make_tensor("b_pred", TensorProto.FLOAT, [1], b_pred.flatten().tolist())
    t_w_conf = helper.make_tensor("W_conf", TensorProto.FLOAT, [num_features, 1], w_conf.flatten().tolist())
    t_b_conf = helper.make_tensor("b_conf", TensorProto.FLOAT, [1], b_conf.flatten().tolist())

    # Nodes: prediction path (sigmoid)
    n_mm_p = helper.make_node("MatMul", inputs=["features", "W_pred"], outputs=["mm_p"])
    n_add_p = helper.make_node("Add", inputs=["mm_p", "b_pred"], outputs=["add_p"])
    n_sq_p = helper.make_node("Squeeze", inputs=["add_p"], outputs=["sq_p"], axes=[1])
    n_sig_p = helper.make_node("Sigmoid", inputs=["sq_p"], outputs=["pred"])

    # Nodes: confidence path (sigmoid)
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
    # Compatibility for onnxruntime versions shipped in images
    model_def.opset_import[0].version = 11  # opset 11
    model_def.ir_version = 7               # IR v7

    # Metadata hints
    model_def.metadata_props.append(onnx.StringStringEntryProto(key="model_type", value="dummy_classifier_v26"))
    model_def.metadata_props.append(onnx.StringStringEntryProto(key="description", value="Dummy 26-feature model for ML pipeline testing"))

    return model_def

def verify_model(model_path: str) -> bool:
    """Verify the model works with ONNX Runtime."""
    session = ort.InferenceSession(model_path)

    # Test with random input
    test_input = np.random.randn(1, 30).astype(np.float32)
    outputs = session.run(None, {"features": test_input})

    print(f"Model input shape: {test_input.shape}")
    print(f"Model output shape: {outputs[0].shape}")
    print(f"Sample output: {outputs[0]}")

    return True

if __name__ == "__main__":
    # Repository-relative output path used by Docker compose mounts
    model_dir = Path("ml/models")
    model_dir.mkdir(exist_ok=True, parents=True)

    model_path = model_dir / "dummy_bullish_model.onnx"
    model = create_dummy_model()
    onnx.save(model, str(model_path))
    # Verify with onnxruntime if available
    try:
        verify_model(str(model_path))
    except Exception:
        pass
    print(f"Dummy 26-feature model saved to: {model_path}")
