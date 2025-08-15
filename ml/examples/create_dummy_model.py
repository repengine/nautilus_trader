"""
Create a dummy model for dry run testing.

This script creates a simple model that can be used for testing
the ML pipeline without requiring actual training.
"""

import pickle
import numpy as np
from pathlib import Path


class DummyModel:
    """
    Simple dummy model for testing.
    
    Generates random predictions with slight bias based on feature values.
    """
    
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
        # Random weights for linear combination
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.5
        
    def predict(self, X):
        """
        Generate predictions.
        
        Returns values between 0 and 1 (for binary classification).
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Simple linear combination with sigmoid
        logits = np.dot(X, self.weights) + self.bias
        predictions = 1 / (1 + np.exp(-logits))
        
        # Add some noise for variety
        noise = np.random.randn(len(predictions)) * 0.05
        predictions = np.clip(predictions + noise, 0, 1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Generate probability predictions (for compatibility).
        """
        preds = self.predict(X)
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
        # Return probabilities for binary classification
        return np.column_stack([1 - preds, preds])


def create_dummy_models():
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