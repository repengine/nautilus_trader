#!/usr/bin/env python
"""
Create and save secure dummy models for testing.

Security Note: This script creates ONNX models instead of pickle files
to maintain production security standards and prevent arbitrary code execution.

"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from ml._imports import HAS_SKLEARN
from ml._imports import check_ml_dependencies


if not HAS_SKLEARN:
    check_ml_dependencies(["scikit-learn"])

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    HAS_ONNX_EXPORT = True
except ImportError:
    HAS_ONNX_EXPORT = False


class DummyModel:
    """
    Legacy dummy model for reference (not saved).

    This class shows the interface that the old pickle models had, but is not used for
    actual model saving anymore.

    """

    def __init__(self, n_features: int = 10) -> None:
        self.n_features: int = n_features
        self.bias: float = 0.5  # Default neutral
        rng = default_rng(42)
        self.weights = rng.standard_normal(n_features).astype(np.float64) * 0.1

    def predict(
        self,
        features: npt.NDArray[np.float64] | Sequence[float],
    ) -> npt.NDArray[np.float64]:
        """
        Generate predictions based on simple linear combination.
        """
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Simple linear model with bias
        predictions = np.dot(arr, self.weights) + self.bias

        # Apply sigmoid to get probability
        from typing import cast

        return cast(npt.NDArray[np.float64], 1 / (1 + np.exp(-predictions)))

    def predict_proba(
        self,
        features: npt.NDArray[np.float64] | Sequence[float],
    ) -> npt.NDArray[np.float64]:
        """
        Return probability estimates.
        """
        probs = self.predict(features)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        # Return both negative and positive class probabilities
        return np.hstack([1 - probs, probs])


def create_secure_sklearn_model(
    random_state: int = 42,
    class_weight: dict[int, float] | None = None,
) -> Pipeline:
    """
    Create a secure sklearn model for ONNX export.

    Parameters
    ----------
    random_state : int, default 42
        Random state for reproducibility.
    class_weight : dict[int, float] | None
        Class weights for bias control.

    Returns
    -------
    Pipeline
        Trained sklearn pipeline.

    """
    # Create a simple pipeline
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10,
                    max_depth=3,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ],
    )

    # Generate dummy training data
    rng = default_rng(random_state)
    X = rng.standard_normal((1000, 10)).astype(np.float32)

    # Create slightly biased targets based on class_weight
    if class_weight and 1 in class_weight:
        # Bias toward positive class if weight > 1
        bias = (class_weight[1] - 1.0) * 0.2
        y_prob = 1 / (1 + np.exp(-(X.sum(axis=1) * 0.1 + bias)))
        y = (y_prob > 0.5).astype(int)
    else:
        y = (rng.random(1000) > 0.5).astype(int)

    # Fit the model
    model.fit(X, y)
    return model


def export_to_onnx(model: Pipeline, output_path: Path) -> None:
    """
    Export sklearn model to ONNX format.

    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline.
    output_path : Path
        Output path for ONNX model.

    """
    if not HAS_ONNX_EXPORT:
        raise ImportError(
            "ONNX export dependencies not available. " "Install with: pip install onnx skl2onnx",
        )

    # Define input schema for 10 features
    initial_type = [("float_input", FloatTensorType([None, 10]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save ONNX model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main() -> None:
    """
    Create and save secure dummy models in ONNX format.
    """
    if not HAS_ONNX_EXPORT:
        print("Error: ONNX export dependencies not available.")
        print("Install with: pip install scikit-learn onnx skl2onnx")
        return

    # Create bullish model
    print("Creating bullish model...")
    bullish_model = create_secure_sklearn_model(
        random_state=42,
        class_weight={0: 0.8, 1: 1.2},  # Bias toward positive class
    )
    export_to_onnx(bullish_model, Path("dummy_bullish_model.onnx"))
    print("Saved dummy_bullish_model.onnx")

    # Create bearish model
    print("Creating bearish model...")
    bearish_model = create_secure_sklearn_model(
        random_state=43,
        class_weight={0: 1.2, 1: 0.8},  # Bias toward negative class
    )
    export_to_onnx(bearish_model, Path("dummy_bearish_model.onnx"))
    print("Saved dummy_bearish_model.onnx")

    # Create neutral model
    print("Creating neutral model...")
    neutral_model = create_secure_sklearn_model(
        random_state=44,
        class_weight=None,  # Balanced
    )
    export_to_onnx(neutral_model, Path("dummy_neutral_model.onnx"))
    print("Saved dummy_neutral_model.onnx")

    print("\nSecure ONNX dummy models created successfully!")
    print("Security Note: These models use ONNX format for production safety.")
    print("Legacy pickle models are no longer supported.")


if __name__ == "__main__":
    main()
