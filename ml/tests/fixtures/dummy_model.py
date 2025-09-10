#!/usr/bin/env python3

"""
Dummy model fixtures for ML module testing.

This module provides utilities to create dummy models for testing purposes, avoiding the
need for real model files during unit tests.

"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    import onnx
    import xgboost as xgb


def create_dummy_onnx_model(output_path: str | Path | None = None) -> Path:
    """
    Create a dummy ONNX model file for testing.

    Parameters
    ----------
    output_path : str | Path, optional
        Path where the model should be saved. If None, creates in temp directory.

    Returns
    -------
    Path
        Path to the created dummy model file

    """
    try:
        import onnx
        from onnx import TensorProto
        from onnx import helper

        # Create a simple ONNX model (identity function)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])

        # Create identity node
        identity = helper.make_node("Identity", ["X"], ["Y"], name="identity")

        # Create graph
        graph = helper.make_graph(
            [identity],
            "dummy_model",
            [X],
            [Y],
        )

        # Create model
        model = helper.make_model(graph)

        # Save model
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = Path(temp_dir) / "dummy_model.onnx"
        else:
            output_path = Path(output_path)

        onnx.save(model, str(output_path))
        return output_path

    except ImportError:
        # If ONNX is not installed, create a fake file
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        else:
            output_path = Path(output_path)

        # Write some dummy bytes to make it look like a model file
        output_path.write_bytes(b"ONNX_DUMMY_MODEL")
        return output_path


def create_dummy_xgboost_model(output_path: str | Path | None = None) -> Path:
    """
    Create a dummy XGBoost model file for testing.

    Parameters
    ----------
    output_path : str | Path, optional
        Path where the model should be saved. If None, creates in temp directory.

    Returns
    -------
    Path
        Path to the created dummy model file

    """
    try:
        import xgboost as xgb

        # Create simple training data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        # Train a minimal model
        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3}
        model = xgb.train(params, dtrain, num_boost_round=1)

        # Save model
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = Path(temp_dir) / "dummy_model.json"
        else:
            output_path = Path(output_path)

        model.save_model(str(output_path))
        return output_path

    except ImportError:
        # If XGBoost is not installed, create a fake JSON file
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        else:
            output_path = Path(output_path)

        # Write minimal valid JSON
        dummy_model = {
            "learner": {
                "gradient_booster": {
                    "name": "gbtree",
                    "model": {"trees": []},
                },
            },
        }
        output_path.write_text(json.dumps(dummy_model))
        return output_path


def create_dummy_model_metadata(model_type: str = "onnx") -> dict[str, Any]:
    """
    Create dummy model metadata for testing.

    Parameters
    ----------
    model_type : str, default="onnx"
        Type of model ("onnx", "xgboost", etc.)

    Returns
    -------
    dict[str, Any]
        Model metadata dictionary

    """
    return {
        "type": model_type,
        "version": "v1.0.0-test",
        "size_bytes": 1024,
        "format": "onnx" if model_type == "onnx" else "json",
        "created_at": "2024-01-01T00:00:00Z",
        "features": ["feature_1", "feature_2", "feature_3"],
        "target": "signal",
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
        },
    }
