"""
Create a dummy model for dry run testing.

This script creates a simple model that can be used for testing the ML pipeline without
requiring actual training.

"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from ml._imports import HAS_SKLEARN, check_ml_dependencies
if not HAS_SKLEARN:
    check_ml_dependencies(["scikit-learn"])

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX_EXPORT = True
except ImportError:
    HAS_ONNX_EXPORT = False


class DummyModel:
    """
    Simple dummy model for testing.

    Generates random predictions with slight bias based on feature values.

    """

    def __init__(self, n_features: int = 10) -> None:
        self.n_features: int = n_features
        self.feature_names: list[str] = [f"feature_{i}" for i in range(n_features)]
        # Random weights for linear combination (deterministic RNG)
        rng = default_rng(42)
        self.weights = rng.standard_normal(n_features).astype(np.float64) * 0.1
        self.bias: float = 0.5

    def predict(self, X: npt.NDArray[np.float64] | Sequence[float]) -> npt.NDArray[np.float64]:
        """
        Generate predictions.

        Returns values between 0 and 1 (for binary classification).

        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Simple linear combination with sigmoid
        logits = np.dot(X_arr, self.weights) + self.bias
        predictions = 1 / (1 + np.exp(-logits))

        # Add some noise for variety (deterministic RNG)
        rng = default_rng(123)
        noise = rng.standard_normal(len(predictions)).astype(np.float64) * 0.05
        predictions = np.clip(predictions + noise, 0, 1)

        from typing import cast

        return cast(npt.NDArray[np.float64], predictions)

    def predict_proba(
        self,
        X: npt.NDArray[np.float64] | Sequence[float],
    ) -> npt.NDArray[np.float64]:
        """
        Generate probability predictions (for compatibility).
        """
        preds = self.predict(X)
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
        # Return probabilities for binary classification
        from typing import cast

        return cast(npt.NDArray[np.float64], np.column_stack([1 - preds, preds]))


def create_dummy_sklearn_model(random_state: int = 42, class_weight: dict[int, float] | None = None) -> Pipeline:
    """
    Create a dummy sklearn model for ONNX export.

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
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=random_state,
            class_weight=class_weight
        ))
    ])

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


def export_to_onnx(model: Pipeline, output_path: Path, feature_names: list[str]) -> None:
    """
    Export sklearn model to ONNX format.

    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline.
    output_path : Path
        Output path for ONNX model.
    feature_names : list[str]
        Names of input features.
    """
    if not HAS_ONNX_EXPORT:
        raise ImportError(
            "ONNX export dependencies not available. "
            "Install with: pip install onnx skl2onnx"
        )

    # Define input schema
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save ONNX model
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def create_dummy_models() -> Path:
    """
    Create several dummy ONNX models for secure testing.

    Security Note: This function creates ONNX models instead of pickle files
    to maintain production security standards.
    """
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f"feature_{i}" for i in range(10)]

    # Create a trend-following model (bullish bias)
    print("Creating bullish model...")
    bullish_model = create_dummy_sklearn_model(
        random_state=42,
        class_weight={0: 0.8, 1: 1.2}  # Bias toward positive class
    )
    export_to_onnx(bullish_model, models_dir / "dummy_bullish_model.onnx", feature_names)
    print(f"Created: {models_dir}/dummy_bullish_model.onnx")

    # Create a mean-reversion model (bearish bias)
    print("Creating bearish model...")
    bearish_model = create_dummy_sklearn_model(
        random_state=43,
        class_weight={0: 1.2, 1: 0.8}  # Bias toward negative class
    )
    export_to_onnx(bearish_model, models_dir / "dummy_bearish_model.onnx", feature_names)
    print(f"Created: {models_dir}/dummy_bearish_model.onnx")

    # Create a neutral model
    print("Creating neutral model...")
    neutral_model = create_dummy_sklearn_model(
        random_state=44,
        class_weight=None  # Balanced
    )
    export_to_onnx(neutral_model, models_dir / "dummy_neutral_model.onnx", feature_names)
    print(f"Created: {models_dir}/dummy_neutral_model.onnx")

    print("\nModel feature names (all models use the same):")
    print(feature_names)

    return models_dir


if __name__ == "__main__":
    try:
        models_dir = create_dummy_models()
        print(f"\nSecure ONNX dummy models created in: {models_dir}")
        print("\nYou can now use these models for dry run testing:")
        print("- dummy_bullish_model.onnx (tends to generate BUY signals)")
        print("- dummy_bearish_model.onnx (tends to generate SELL signals)")
        print("- dummy_neutral_model.onnx (balanced signals)")
        print("\nSecurity Note: These models use ONNX format for production safety.")
        print("Legacy pickle models are no longer supported.")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo create ONNX models, install required dependencies:")
        print("pip install scikit-learn onnx skl2onnx")
