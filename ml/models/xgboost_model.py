#!/usr/bin/env python3

"""
XGBoost model implementation for Nautilus ML.

This module provides the XGBoostModel class that wraps XGBoost models
for inference, supporting both Booster and sklearn-style API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import xgb
from ml.models.base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model wrapper for production inference.

    Supports both raw Booster objects and sklearn-style XGBClassifier/XGBRegressor.
    """

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        """
        Initialize XGBoost model.

        Parameters
        ----------
        model : xgboost.Booster or xgboost.XGBClassifier/XGBRegressor
            The XGBoost model object
        metadata : dict[str, Any]
            Model metadata
        """
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        super().__init__(model, metadata)

        # Determine if it's a Booster or sklearn-style model
        self._is_booster = hasattr(model, "predict") and not hasattr(model, "predict_proba")

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Make prediction with XGBoost model.

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

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._is_booster:
            # Raw Booster object
            if not HAS_XGBOOST:
                check_ml_dependencies(["xgboost"])

            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(features)
            predictions = self._model.predict(dmatrix)
        else:
            # Sklearn-style model
            if hasattr(self._model, "predict_proba"):
                # Classification model - return class probabilities
                predictions = self._model.predict_proba(features)
                # For binary classification, return positive class probability
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
            else:
                # Regression model
                predictions = self._model.predict(features)

        return np.asarray(predictions, dtype=np.float32)
