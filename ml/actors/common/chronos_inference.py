"""
Chronos inference adapter for AutoGluon TimeSeries predictors.

This module provides a thin wrapper around AutoGluon/Chronos predictors so
Nautilus ML pipelines can reuse Chronos outputs without coupling to the
training stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_AUTOGLUON
from ml._imports import TimeSeriesDataFrame
from ml._imports import check_ml_dependencies
from ml.config.autogluon import AutoGluonDataConfig
from ml.data.autogluon_adapter import convert_to_timeseries_dataframe


logger = logging.getLogger(__name__)


class ChronosPredictorProtocol(Protocol):
    """Protocol for Chronos/AutoGluon predictor objects."""

    def predict(self, data: Any, **kwargs: Any) -> Any:
        """Return predictions for the supplied TimeSeriesDataFrame."""
        ...


def _extract_prediction_array(predictions: Any) -> npt.NDArray[np.float32]:
    """
    Extract a float32 prediction array from AutoGluon predictor output.

    Handles DataFrame-like results with a "mean" column and falls back to
    .values or array conversion for other outputs.
    """
    if hasattr(predictions, "columns") and "mean" in getattr(predictions, "columns", ()):
        series = predictions["mean"]
        values = getattr(series, "values", series)
        return np.asarray(values, dtype=np.float32)

    if hasattr(predictions, "values"):
        return np.asarray(getattr(predictions, "values"), dtype=np.float32)

    return np.asarray(predictions, dtype=np.float32)


@dataclass(slots=True, frozen=True)
class ChronosInferenceAdapter:
    """
    Adapter that normalizes Chronos predictor output for inference consumers.

    Parameters
    ----------
    predictor : ChronosPredictorProtocol
        AutoGluon TimeSeries predictor or Chronos trainer wrapper.
    data_config : AutoGluonDataConfig
        Configuration used to convert raw DataFrames into TimeSeriesDataFrame.
    """

    predictor: ChronosPredictorProtocol
    data_config: AutoGluonDataConfig

    def predict_from_timeseries(self, tsdf: Any, **kwargs: Any) -> npt.NDArray[np.float32]:
        """
        Generate predictions from a pre-built TimeSeriesDataFrame.

        Parameters
        ----------
        tsdf : Any
            TimeSeriesDataFrame-compatible input.
        **kwargs : Any
            Additional keyword arguments forwarded to the predictor.

        Returns
        -------
        npt.NDArray[np.float32]
            Prediction array (mean forecast values).
        """
        predictions = self.predictor.predict(tsdf, **kwargs)
        return _extract_prediction_array(predictions)

    def predict(self, data: Any, **kwargs: Any) -> npt.NDArray[np.float32]:
        """
        Generate predictions from a raw DataFrame or TimeSeriesDataFrame.

        Parameters
        ----------
        data : Any
            Raw data frame or TimeSeriesDataFrame input.
        **kwargs : Any
            Additional keyword arguments forwarded to the predictor.

        Returns
        -------
        npt.NDArray[np.float32]
            Prediction array (mean forecast values).
        """
        tsdf = self._ensure_timeseries_dataframe(data)
        return self.predict_from_timeseries(tsdf, **kwargs)

    def _ensure_timeseries_dataframe(self, data: Any) -> Any:
        """
        Convert raw input data into TimeSeriesDataFrame when needed.
        """
        if TimeSeriesDataFrame is not None and isinstance(data, TimeSeriesDataFrame):
            return data
        if not HAS_AUTOGLUON:
            check_ml_dependencies(["autogluon"])
            raise ImportError("AutoGluon TimeSeries is required for Chronos inference")
        return convert_to_timeseries_dataframe(data, self.data_config)


__all__ = [
    "ChronosInferenceAdapter",
    "ChronosPredictorProtocol",
]
