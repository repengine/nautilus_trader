#!/usr/bin/env python3

"""
ONNX model implementation for Nautilus ML.

This module provides the ONNXModel class that wraps ONNX Runtime sessions
for high-performance inference in the hot path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml.models.base import BaseModel


class ONNXModel(BaseModel):
    """
    ONNX model wrapper for production inference.

    Uses ONNX Runtime for optimized inference with support for
    hardware acceleration and graph optimization.
    """

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        """
        Initialize ONNX model.

        Parameters
        ----------
        model : onnxruntime.InferenceSession
            The ONNX Runtime inference session
        metadata : dict[str, Any]
            Model metadata including input/output names
        """
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        super().__init__(model, metadata)

        # Cache input/output names for performance
        if hasattr(model, "get_inputs"):
            self._input_names = [inp.name for inp in model.get_inputs()]
            self._output_names = [out.name for out in model.get_outputs()]
        else:
            self._input_names = metadata.get("input_names", ["input"])
            self._output_names = metadata.get("output_names", ["output"])

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Make prediction with ONNX model.

        Parameters
        ----------
        features : NDArray[np.float32]
            Input features

        Returns
        -------
        NDArray[np.float32]
            Model predictions
        """
        self.validate_input(features)

        # Ensure 2D input for batch prediction
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Ensure float32 type for ONNX
        features = features.astype(np.float32)

        # Run inference
        input_name = self._input_names[0] if self._input_names else "input"
        outputs = self._model.run(None, {input_name: features})

        # Return first output as numpy array
        result = outputs[0]
        if isinstance(result, np.ndarray):
            # Handle both regression and classification outputs
            if result.shape[-1] == 1:
                # Regression or binary classification - return 1D array
                return result.squeeze(-1).astype(np.float32)
            else:
                # Multi-class - return probabilities or take argmax
                # For now, return raw output
                return result.astype(np.float32)
        else:
            # Scalar output
            return np.array([result], dtype=np.float32)
