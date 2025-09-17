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

def create_dummy_model():
    """
    Create a dummy ONNX model that:
    - Takes 30 features as input
    - Outputs 3 values: [probability, direction, confidence]
    - Uses random weights for testing
    """
    # Define input/output
    num_features = 30

    # Create random weights for a simple linear model
    weights = np.random.randn(num_features, 3).astype(np.float32) * 0.1
    bias = np.array([0.5, 0.0, 0.5], dtype=np.float32)  # Default to 50% probability, neutral, 50% confidence

    # Create ONNX graph
    input_tensor = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, [1, num_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "predictions", TensorProto.FLOAT, [1, 3]
    )

    # Create weight and bias initializers
    weight_initializer = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [num_features, 3],
        weights.flatten().tolist()
    )
    bias_initializer = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [3],
        bias.tolist()
    )

    # Create MatMul and Add nodes
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["features", "weights"],
        outputs=["matmul_output"]
    )

    add_node = helper.make_node(
        "Add",
        inputs=["matmul_output", "bias"],
        outputs=["add_output"]
    )

    # Apply sigmoid to get probabilities
    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=["add_output"],
        outputs=["predictions"]
    )

    # Create the graph
    graph_def = helper.make_graph(
        [matmul_node, add_node, sigmoid_node],
        "dummy_ml_model",
        [input_tensor],
        [output_tensor],
        [weight_initializer, bias_initializer]
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="ml_pipeline_test")
    model_def.opset_import[0].version = 11  # Use version 11 for compatibility

    # Set IR version for compatibility with older ONNX Runtime
    model_def.ir_version = 7  # Use IR version 7 for compatibility

    # Add metadata
    model_def.metadata_props.append(
        onnx.StringStringEntryProto(key="model_type", value="dummy_classifier")
    )
    model_def.metadata_props.append(
        onnx.StringStringEntryProto(key="description", value="Dummy model for ML pipeline testing")
    )

    return model_def

def verify_model(model_path):
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
    # Create model directory
    model_dir = Path("/home/nate/projects/nautilus_trader/ml/models")
    model_dir.mkdir(exist_ok=True)

    # Create and save the model
    model_path = model_dir / "dummy_model.onnx"
    model = create_dummy_model()

    # Save model
    onnx.save(model, str(model_path))
    print(f"Dummy model saved to: {model_path}")

    # Verify it works
    if verify_model(str(model_path)):
        print("✓ Model verification successful!")
        print("\nModel details:")
        print("- Input: 30 features (float32)")
        print("- Output: 3 values [probability, direction, confidence]")
        print("- Type: Simple linear classifier with sigmoid activation")
