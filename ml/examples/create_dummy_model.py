"""
Create a dummy model for dry run testing.

This script creates a simple model that can be used for testing the ML pipeline without
requiring actual training.

"""

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng


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


def create_dummy_models() -> Path:
    """
    Create several dummy models for testing different scenarios.
    """
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create a trend-following model (bullish bias)
    bullish_model = DummyModel(n_features=10)
    bullish_model.bias = 0.6  # Slight bullish bias

    with open(models_dir / "dummy_bullish_model.pkl", "wb") as f:
        pickle.dump(bullish_model, f)
    print(f"Created: {models_dir}/dummy_bullish_model.pkl")

    # Create a mean-reversion model (bearish bias)
    bearish_model = DummyModel(n_features=10)
    bearish_model.bias = 0.4  # Slight bearish bias

    with open(models_dir / "dummy_bearish_model.pkl", "wb") as f:
        pickle.dump(bearish_model, f)
    print(f"Created: {models_dir}/dummy_bearish_model.pkl")

    # Create a neutral model
    neutral_model = DummyModel(n_features=10)
    neutral_model.bias = 0.5  # Neutral

    with open(models_dir / "dummy_neutral_model.pkl", "wb") as f:
        pickle.dump(neutral_model, f)
    print(f"Created: {models_dir}/dummy_neutral_model.pkl")

    print("\nModel feature names (all models use the same):")
    print(neutral_model.feature_names)

    return models_dir


if __name__ == "__main__":
    models_dir = create_dummy_models()
    print(f"\nDummy models created in: {models_dir}")
    print("\nYou can now use these models for dry run testing:")
    print("- dummy_bullish_model.pkl (tends to generate BUY signals)")
    print("- dummy_bearish_model.pkl (tends to generate SELL signals)")
    print("- dummy_neutral_model.pkl (balanced signals)")
