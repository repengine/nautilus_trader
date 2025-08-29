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

if TYPE_CHECKING:  # pragma: no cover - typing stub for mypy
    class NautilusData:
        """Typing stub for Nautilus `Data` base class."""
        pass
else:  # Use real runtime base to preserve integration
    from nautilus_trader.core.data import Data as NautilusData


if TYPE_CHECKING:
    import pandas as pd


@dataclass
class FeatureData(NautilusData):
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
    def feature_values(self) -> dict[str, float]:
        """
        Safe accessor for the feature values dict.

        Avoids collisions with any inherited `values()` method on base classes by
        reading the underlying mapping directly and normalizing to a plain dict.

        Returns
        -------
        dict[str, float]
            Mapping of feature name to value.
        """
        # Prefer raw attribute dict to bypass potential properties/methods
        raw: dict[str, Any] = object.__getattribute__(self, "__dict__")
        data = raw.get("values")
        if data is None:
            candidate = getattr(self, "values", {})
            data = candidate() if callable(candidate) else candidate
        # Defensive copy as a plain dict[str, float]
        try:
            return {str(k): float(v) for k, v in dict(data or {}).items()}
        except Exception:
            # Last-resort fallback
            return {}

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
class ModelPrediction(NautilusData):
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
class StrategySignal(NautilusData):
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


class DummyStore:
    """
    Dummy store for testing when database is not available.

    This store accepts all method calls but doesn't persist anything. Used only for unit
    testing when PostgreSQL is not available.

    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """
        Initialize dummy store (accepts any arguments).
        """

    def write_features(self, *args: object, **kwargs: object) -> None:
        """
        Write features (dummy).
        """

    def write_prediction(self, *args: object, **kwargs: object) -> None:
        """
        Write prediction (dummy).
        """

    def write_signal(self, *args: object, **kwargs: object) -> None:
        """
        Write signal (dummy).
        """

    def write_batch(self, data: list[Any], emit_events: bool = True) -> None:
        """Batch write (dummy)."""
        return None

    def flush(self, *args: object, **kwargs: object) -> None:
        """
        Flush buffered state (dummy).
        """

    # Backward-compatible alias used in some tests
    def get_stats(self, *args: object, **kwargs: object) -> dict[str, Any]:
        """Deprecated alias for get_statistics (retained for compatibility)."""
        return self.get_statistics()

    def get_statistics(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]:
        """
        Get storage statistics (dummy implementation).

        This method conforms to the BaseStore interface. For backward compatibility,
        `get_stats` remains as a deprecated alias.
        """
        return self.get_stats(start_ns=start_ns, end_ns=end_ns)

    def is_healthy(self) -> bool:
        """
        Perform health check (always True in dummy).
        """
        return True

    def get_latest(self, *args: object, **kwargs: object) -> None:
        """
        Get latest item (dummy).
        """
        return None

    # Model store protocol methods (dummy implementations)
    def read_predictions(self, model_id: str, instrument_id: str, start_ns: int, end_ns: int) -> Any:
        return None

    def get_model_performance(
        self, model_id: str, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, Any]:
        return {}

    # Strategy store protocol methods (dummy implementations)
    def read_signals(self, strategy_id: str, instrument_id: str, start_ns: int, end_ns: int) -> Any:
        return None

    def get_strategy_performance(
        self, strategy_id: str, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, Any]:
        return {}

    def get_signal_distribution(
        self, strategy_id: str | None = None, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, int]:
        return {}

    # Feature store extension (dummy realtime computation)
    def compute_realtime(self, bar: Any, store: bool = True, indicator_manager: Any | None = None) -> Any:
        return None

    def __getattr__(self, name: str) -> object:
        """
        Handle any other method calls.
        """

        def dummy_method(*args: object, **kwargs: object) -> None:
            return None

        return dummy_method
