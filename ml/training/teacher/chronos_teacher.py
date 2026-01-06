"""
Chronos teacher for knowledge distillation.

This module provides a BaseTeacher implementation using Chronos foundation
models via AutoGluon TimeSeries. The teacher produces soft labels for
distillation to faster student models (Chronos-Bolt or LightGBM).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_AUTOGLUON
from ml._imports import check_ml_dependencies
from ml.training.autogluon.chronos_trainer import ChronosTrainer
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


if TYPE_CHECKING:
    import polars as _pl



__all__ = [
    "ChronosTeacher",
    "ChronosTeacherConfig",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronosTeacherConfig(TeacherConfig):
    """
    Configuration for Chronos teacher model.

    Parameters
    ----------
    architecture : str, default "Chronos-2"
        Model architecture name.
    version : str, default "1.0.0"
        Model version.
    preset : str, default "chronos2"
        AutoGluon Chronos preset.
    prediction_length : int, default 15
        Forecast horizon in time steps.
    freq : str, default "min"
        Time series frequency.
    time_limit : int, default 3600
        Training time budget in seconds.
    target_column : str, default "forward_return"
        Target column for regression.
    enable_gpu : bool, default True
        Whether to use GPU acceleration.

    """

    architecture: str = "Chronos-2"
    version: str = "1.0.0"
    preset: str = "chronos2"
    prediction_length: int = 15
    freq: str = "min"
    time_limit: int = 3600
    target_column: str = "forward_return"
    enable_gpu: bool = True


class ChronosTeacher(BaseTeacher):
    """
    Teacher model using Chronos foundation models for distillation.

    This teacher wraps a Chronos model (via AutoGluon) to produce
    soft labels for training faster student models. The Chronos-2
    teacher provides best accuracy, while distillation to Chronos-Bolt
    or LightGBM enables 250x+ faster inference.

    Parameters
    ----------
    config : ChronosTeacherConfig
        Teacher configuration.

    Examples
    --------
    >>> from ml.training.teacher.chronos_teacher import ChronosTeacher, ChronosTeacherConfig
    >>> config = ChronosTeacherConfig(preset="chronos2", time_limit=1800)
    >>> teacher = ChronosTeacher(config)
    >>> teacher.fit(dataset)
    >>> soft_labels = teacher.predict_proba(X)

    """

    def __init__(self, config: ChronosTeacherConfig) -> None:
        """
        Initialize the Chronos teacher.

        Parameters
        ----------
        config : ChronosTeacherConfig
            Teacher configuration.

        """
        super().__init__(config)

        if not HAS_AUTOGLUON:
            check_ml_dependencies(["autogluon"])

        self._chronos_config = config
        self._trainer: Any = None  # ChronosTrainer
        self._feature_names: list[str] = []

    def fit(self, dataset: Any) -> ChronosTeacher:
        """
        Fit the Chronos teacher on a dataset.

        This method trains the Chronos model on the provided dataset
        using the configured preset and time limit.

        Parameters
        ----------
        dataset : pl.DataFrame | TimeSeriesDataFrame
            Training dataset with features and target.

        Returns
        -------
        ChronosTeacher
            Self for method chaining.

        """
        from ml.config.autogluon import AutoGluonDataConfig
        from ml.config.autogluon import ChronosTrainingConfig

        logger.info(
            f"Fitting Chronos teacher with preset='{self._chronos_config.preset}', "
            f"time_limit={self._chronos_config.time_limit}s"
        )

        # Create Chronos training config from teacher config
        training_config = ChronosTrainingConfig(
            prediction_length=self._chronos_config.prediction_length,
            freq=self._chronos_config.freq,
            target_column=self._chronos_config.target_column,
            preset=self._chronos_config.preset,
            time_limit=self._chronos_config.time_limit,
            enable_gpu=self._chronos_config.enable_gpu,
            data_config=AutoGluonDataConfig(
                item_id_column="instrument_id",
                timestamp_column="ts_event",
                target_column=self._chronos_config.target_column,
            ),
        )

        # Train the model
        self._trainer = ChronosTrainer(training_config)
        result = self._trainer.train(dataset)

        self._feature_names = result.get("feature_names", [])
        self._is_fitted = True

        logger.info("Chronos teacher training complete")
        return self

    def predict_logits(self, X: Any) -> npt.NDArray[np.float64]:
        """
        Return raw predictions (logits) for given features.

        For Chronos regression models, this returns the mean forecast
        values which can be treated as logits for distillation.

        Parameters
        ----------
        X : Any
            DataFrame or TimeSeriesDataFrame input. Numpy inputs are not supported.

        Returns
        -------
        np.ndarray
            Raw predictions as logits.

        Raises
        ------
        ValueError
            If teacher has not been fitted.

        """
        if not self._is_fitted or self._trainer is None:
            raise ValueError("Teacher must be fitted before prediction")

        if isinstance(X, np.ndarray):
            raise ValueError(
                "ChronosTeacher requires DataFrame input; use predict_from_dataframe()."
            )

        predictions = self._trainer.predict(X)
        result: npt.NDArray[np.float64] = predictions.astype(np.float64)
        return result

    def predict_from_dataframe(
        self,
        df: _pl.DataFrame | Any,
    ) -> npt.NDArray[np.float64]:
        """
        Generate predictions directly from DataFrame input.

        This is the preferred prediction method as it avoids the
        numpy-to-DataFrame conversion overhead.

        Parameters
        ----------
        df : pl.DataFrame | TimeSeriesDataFrame
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predictions as float64 array.

        """
        if not self._is_fitted or self._trainer is None:
            raise ValueError("Teacher must be fitted before prediction")

        result: npt.NDArray[np.float64] = self._trainer.generate_soft_labels(df)
        return result

    def predict_proba(self, X: Any) -> npt.NDArray[np.float64]:
        """
        Return calibrated probabilities for given features.

        For regression targets (forward returns), this applies a sigmoid
        transformation to convert predictions to pseudo-probabilities
        suitable for distillation.

        Parameters
        ----------
        X : Any
            DataFrame or TimeSeriesDataFrame input. Numpy inputs are not supported.

        Returns
        -------
        np.ndarray
            Calibrated pseudo-probabilities in [0, 1].

        """
        logits = self.predict_logits(X)

        # Apply Platt calibration if available
        if self._platt_coef is not None and self._platt_intercept is not None:
            logits = self._platt_coef * logits + self._platt_intercept

        # Sigmoid for probability conversion
        proba = 1.0 / (1.0 + np.exp(-logits))
        return proba.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        """
        Return a mapping of feature name to dtype as strings.

        Returns
        -------
        dict[str, str]
            Feature schema mapping.

        """
        return dict.fromkeys(self._feature_names, "float32")

    def save(self, path: str | Path) -> None:
        """
        Save the trained teacher to disk.

        Parameters
        ----------
        path : str | Path
            Path to save the teacher.

        """
        if not self._is_fitted or self._trainer is None:
            raise ValueError("Teacher must be fitted before saving")

        self._trainer.save(path)
        logger.info(f"Chronos teacher saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ChronosTeacher:
        """
        Load a trained teacher from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved teacher.

        Returns
        -------
        ChronosTeacher
            Loaded teacher.

        """
        from ml.training.autogluon.chronos_trainer import ChronosTrainer

        trainer = ChronosTrainer.load(path)

        # Create config from loaded trainer
        config = ChronosTeacherConfig(
            preset=trainer.config.preset,
            prediction_length=trainer.config.prediction_length,
            freq=trainer.config.freq,
            time_limit=trainer.config.time_limit,
        )

        teacher = cls(config)
        teacher._trainer = trainer
        teacher._is_fitted = True
        teacher._feature_names = trainer._feature_names

        logger.info(f"Chronos teacher loaded from {path}")
        return teacher

    def get_soft_labels(
        self,
        data: _pl.DataFrame | Any,
        *,
        temperature: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """
        Generate soft labels for student distillation.

        Parameters
        ----------
        data : pl.DataFrame | TimeSeriesDataFrame
            Data to generate soft labels for.
        temperature : float, default 1.0
            Temperature for softening predictions.
            Higher values produce softer (more uniform) labels.

        Returns
        -------
        np.ndarray
            Soft labels for distillation.

        """
        predictions = self.predict_from_dataframe(data)

        if temperature != 1.0:
            # Apply temperature scaling
            predictions = predictions / temperature

        return predictions
