"""
Base classes for ML store implementations.

This module provides abstract base classes and data structures for storing ML-related
data including features, predictions, and signals.

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nautilus_trader.core.data import Data


if TYPE_CHECKING:
    import pandas as pd


@dataclass
class FeatureData(Data):
    """
    Nautilus-compatible feature data class.

    Attributes
    ----------
    feature_set_id : str
        Identifier for the feature set
    instrument_id : str
        Instrument identifier
    values : dict[str, float]
        Feature name to value mapping
    _ts_event : int
        Event timestamp in nanoseconds
    _ts_init : int
        Initialization timestamp in nanoseconds

    """

    feature_set_id: str
    instrument_id: str
    values: dict[str, float]
    _ts_event: int  # nanoseconds
    _ts_init: int  # nanoseconds

    @property
    def ts_event(self) -> int:
        """
        Event timestamp in nanoseconds.
        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        Initialization timestamp in nanoseconds.
        """
        return self._ts_init


@dataclass
class ModelPrediction(Data):
    """
    Store model predictions and inference metadata.

    Attributes
    ----------
    model_id : str
        Model identifier
    instrument_id : str
        Instrument identifier
    prediction : float
        Model prediction value
    confidence : float
        Prediction confidence score
    features_used : dict[str, float]
        Feature values used for prediction
    inference_time_ms : float
        Inference latency in milliseconds
    _ts_event : int
        Event timestamp in nanoseconds
    _ts_init : int
        Initialization timestamp in nanoseconds

    """

    model_id: str
    instrument_id: str
    prediction: float
    confidence: float
    features_used: dict[str, float]
    inference_time_ms: float
    _ts_event: int
    _ts_init: int

    @property
    def ts_event(self) -> int:
        """
        Event timestamp in nanoseconds.
        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        Initialization timestamp in nanoseconds.
        """
        return self._ts_init


@dataclass
class StrategySignal(Data):
    """
    Store strategy decisions and execution signals.

    Attributes
    ----------
    strategy_id : str
        Strategy identifier
    instrument_id : str
        Instrument identifier
    signal_type : str
        Signal type ('BUY', 'SELL', 'HOLD')
    strength : float
        Signal strength/confidence
    model_predictions : dict[str, float]
        Model ID to prediction mapping
    risk_metrics : dict[str, float]
        Risk metrics at decision time
    execution_params : dict[str, Any]
        Execution parameters (stop loss, take profit, etc.)
    _ts_event : int
        Event timestamp in nanoseconds
    _ts_init : int
        Initialization timestamp in nanoseconds

    """

    strategy_id: str
    instrument_id: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float
    model_predictions: dict[str, float]
    risk_metrics: dict[str, float]
    execution_params: dict[str, Any]
    _ts_event: int
    _ts_init: int

    @property
    def ts_event(self) -> int:
        """
        Event timestamp in nanoseconds.
        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        Initialization timestamp in nanoseconds.
        """
        return self._ts_init


class BaseStore(ABC):
    """
    Abstract base class for all store implementations.

    Provides common interface for reading and writing time-series data with batching
    support.

    """

    @abstractmethod
    def write_batch(self, data: list[Any]) -> None:
        """
        Write batch of data to storage.

        Parameters
        ----------
        data : list[Any]
            List of data objects to write

        """
        ...

    @abstractmethod
    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read data in time range.

        Parameters
        ----------
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        instrument_id : str | None
            Optional instrument filter

        Returns
        -------
        pd.DataFrame
            Data within the specified range

        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """
        Flush any pending writes to storage.
        """
        ...

    @abstractmethod
    def get_latest(
        self,
        instrument_id: str,
        limit: int = 1,
    ) -> pd.DataFrame:
        """
        Get latest entries for an instrument.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        limit : int
            Maximum number of entries to return

        Returns
        -------
        pd.DataFrame
            Latest entries

        """
        ...

    @abstractmethod
    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Get storage statistics.

        Parameters
        ----------
        start_ns : int | None
            Optional start timestamp filter
        end_ns : int | None
            Optional end timestamp filter

        Returns
        -------
        dict[str, Any]
            Statistics including count, date range, etc.

        """
        ...
