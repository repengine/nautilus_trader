"""
Base teacher interface and minimal TFT teacher placeholder.

The teacher runs in the cold path to produce high-quality soft labels
(logits/probabilities) for student distillation. Heavy dependencies are intentionally
optional; tests may use fakes implementing this interface.

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TeacherConfig:
    """
    Configuration for teacher models.
    """

    architecture: str = "TFT"
    version: str = "0.1.0"


class BaseTeacher(ABC):
    """
    Abstract interface for teacher models.
    """

    def __init__(self, config: TeacherConfig) -> None:
        """
        Initialize base teacher with configuration.

        Parameters
        ----------
        config : TeacherConfig
            Teacher model configuration.

        """
        self.config = config
        self._is_fitted: bool = False
        self._platt_coef: float | None = None
        self._platt_intercept: float | None = None

    @abstractmethod
    def fit(self, dataset: Any) -> BaseTeacher:
        """
        Fit the teacher on a dataset (cold path).
        """

    @abstractmethod
    def predict_logits(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Return raw scores (logits) for given features.
        """

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Return calibrated probabilities for given features.
        """
        z = self.predict_logits(X)
        if self._platt_coef is not None and self._platt_intercept is not None:
            z = self._platt_coef * z + self._platt_intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return p.astype(np.float64)

    def calibrate(self, z_val: npt.NDArray[np.float64], y_val: npt.NDArray[np.float64]) -> None:
        """
        Fit Platt calibration on raw scores vs true labels.

        If scikit-learn is not available, leaves the model uncalibrated.

        """
        try:
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(solver="lbfgs")
            lr.fit(z_val.reshape(-1, 1), y_val.astype(int))
            self._platt_coef = float(lr.coef_.ravel()[0])
            self._platt_intercept = float(lr.intercept_.ravel()[0])
        except Exception:
            self._platt_coef = None
            self._platt_intercept = None

    @abstractmethod
    def feature_schema(self) -> dict[str, str]:
        """
        Return a mapping of feature name -> dtype as strings.
        """


class TFTTeacher(BaseTeacher):
    """
    Minimal placeholder for a Temporal Fusion Transformer teacher.

    In production, this would wrap PyTorch Forecasting/Lightning code. Here we only
    define the interface; fit/predict raise if not implemented.

    """

    def fit(self, dataset: Any) -> TFTTeacher:
        """
        Fit the TFT teacher model.

        Parameters
        ----------
        dataset : Any
            Training dataset.

        Returns
        -------
        TFTTeacher
            Fitted teacher instance.

        Raises
        ------
        NotImplementedError
            This is a placeholder implementation.

        """
        raise NotImplementedError("TFTTeacher.fit is a placeholder in this prototype.")

    def predict_logits(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predict raw logits for distillation.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Input features.

        Returns
        -------
        npt.NDArray[np.float64]
            Raw logits for soft label generation.

        Raises
        ------
        NotImplementedError
            This is a placeholder implementation.

        """
        raise NotImplementedError("TFTTeacher.predict_logits is a placeholder in this prototype.")

    def feature_schema(self) -> dict[str, str]:
        """
        Get expected feature schema.

        Returns
        -------
        dict[str, str]
            Mapping of feature names to types.

        """
        # In a real implementation this is derived from dataset configuration
        return {}
