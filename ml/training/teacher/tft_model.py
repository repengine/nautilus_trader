from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


@dataclass(frozen=True)
class TFTTeacherConfig(TeacherConfig):
    """
    Configuration for a TFT teacher model.
    """

    architecture: str = "TFT"


class TFTTeacher(BaseTeacher):
    """
    Temporal Fusion Transformer (TFT) teacher model.

    This is a placeholder implementation. In a real-world scenario, this class
    would wrap a proper TFT model from a library like PyTorch Forecasting.
    The `fit` method would train the transformer on time-series data, and the
    `predict_logits` method would perform inference.

    For this scaffold, we assume the model is pre-trained, and we only
    focus on the calibration aspect.

    """

    def __init__(self, config: TFTTeacherConfig):
        super().__init__(config)
        self.config: TFTTeacherConfig = config

    def fit(self, dataset: object) -> TFTTeacher:
        """
        Fits the TFT model to the dataset.

        Note: This is a placeholder. A real implementation would train the model.

        """
        # In a real implementation, you would train the underlying TFT model here.
        # For now, we'll just mark it as fitted to allow calibration.
        self._is_fitted = True
        return self

    def predict_logits(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Generates logits from the input features.

        Note: This is a placeholder. It currently returns the input, assuming
        it contains pre-computed raw logits or probabilities.

        """
        # This assumes X contains the raw model outputs (logits) to be calibrated.
        return X.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        """
        Returns the feature schema for the model.
        """
        # In a real implementation, this would be derived from the TimeSeriesDataSet.
        return {}
