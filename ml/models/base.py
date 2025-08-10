#!/usr/bin/env python3

"""
Abstract base class for ML models in Nautilus.

This module provides the BaseModel interface that all model implementations
must follow to ensure consistent prediction behavior across different
model types (ONNX, XGBoost, LightGBM, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    This provides a consistent interface regardless of the underlying
    model implementation. All models must expose:
    - predict() method returning numpy array
    - metadata property for model information
    - validate_input() for shape validation
    """
    
    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        """
        Initialize base model.
        
        Parameters
        ----------
        model : Any
            The underlying model object (ONNX session, XGBoost booster, etc.)
        metadata : dict[str, Any]
            Model metadata including type, version, input_shape, etc.
        """
        self._model = model
        self._metadata = metadata
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Get model metadata."""
        return self._metadata
    
    @property
    def model_id(self) -> str:
        """Get model identifier for versioning."""
        model_id = self._metadata.get("model_id", self._metadata.get("version", "unknown"))
        return str(model_id)
    
    @abstractmethod
    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Make predictions on input features.
        
        Parameters
        ----------
        features : NDArray[np.float32]
            Input features with shape (n_samples, n_features) or (n_features,)
            
        Returns
        -------
        NDArray[np.float32]
            Predictions with shape (n_samples,) or scalar for single sample
            
        Raises
        ------
        ValueError
            If input shape doesn't match expected shape
        """
        ...
    
    def validate_input(self, features: NDArray[np.float32]) -> None:
        """
        Validate input features shape.
        
        Parameters
        ----------
        features : NDArray[np.float32]
            Input features to validate
            
        Raises
        ------
        ValueError
            If input shape doesn't match expected shape
        """
        if "input_shape" in self._metadata and self._metadata["input_shape"] is not None:
            expected_shape = self._metadata["input_shape"]
            if isinstance(expected_shape, (list, tuple)) and len(expected_shape) > 0:
                # Check feature dimension (last dimension)
                expected_features = expected_shape[-1]
                actual_features = features.shape[-1] if features.ndim > 0 else 0
                
                if actual_features != expected_features:
                    raise ValueError(
                        f"Input validation failed: expected {expected_features} features, "
                        f"got {actual_features}"
                    )