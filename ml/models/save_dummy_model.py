#!/usr/bin/env python
"""
Create and save a simple dummy model for testing.
"""

import pickle
import numpy as np


class DummyModel:
    """
    A simple dummy model for testing ML infrastructure.
    """
    
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.bias = 0.5  # Default neutral
        self.weights = np.random.randn(n_features) * 0.1
        
    def predict(self, features):
        """
        Generate predictions based on simple linear combination.
        """
        if isinstance(features, list):
            features = np.array(features)
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Simple linear model with bias
        predictions = np.dot(features, self.weights) + self.bias
        
        # Apply sigmoid to get probability
        return 1 / (1 + np.exp(-predictions))
    
    def predict_proba(self, features):
        """
        Return probability estimates.
        """
        probs = self.predict(features)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        # Return both negative and positive class probabilities
        return np.hstack([1 - probs, probs])


def main():
    """
    Create and save dummy models.
    """
    # Create bullish model
    bullish_model = DummyModel(n_features=10)
    bullish_model.bias = 0.55  # Slightly bullish
    
    with open("dummy_bullish_model.pkl", "wb") as f:
        pickle.dump(bullish_model, f)
    print("Saved dummy_bullish_model.pkl")
    
    # Create bearish model
    bearish_model = DummyModel(n_features=10)
    bearish_model.bias = 0.45  # Slightly bearish
    
    with open("dummy_bearish_model.pkl", "wb") as f:
        pickle.dump(bearish_model, f)
    print("Saved dummy_bearish_model.pkl")
    
    # Create neutral model
    neutral_model = DummyModel(n_features=10)
    neutral_model.bias = 0.5  # Neutral
    
    with open("dummy_neutral_model.pkl", "wb") as f:
        pickle.dump(neutral_model, f)
    print("Saved dummy_neutral_model.pkl")


if __name__ == "__main__":
    main()