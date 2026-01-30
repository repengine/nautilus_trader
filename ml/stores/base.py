"""
Base classes for ML store implementations.

This module provides abstract base classes and data structures for storing ML-related
data including features, predictions, and signals.

"""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ml.common.protocols import MLComponentMixin


if TYPE_CHECKING:  # pragma: no cover - typing stub for mypy

    class NautilusData:
        """
        Typing stub for Nautilus `Data` base class.
        """

else:  # Use real runtime base to preserve integration
    from nautilus_trader.core.data import Data as NautilusData


if TYPE_CHECKING:
    import pandas as pd


@dataclass(init=False)
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
    # Optional quality flags for compatibility with tests expecting them on the payload
    quality_flags: int = 0

    def __init__(
        self,
        feature_set_id: str | None = None,
        instrument_id: str = "",
        values: dict[str, float] | None = None,
        _ts_event: int | None = None,
        _ts_init: int | None = None,
        *,
        # Backward-compatible aliases used by tests
        ts_event: int | None = None,
        ts_init: int | None = None,
        features: dict[str, float] | None = None,
        feature_values: dict[str, float] | None = None,
        quality_flags: int = 0,
    ) -> None:
        """
        Create a FeatureData record.

        Accepts both internal field names and test-friendly aliases:
        - `values` or `features`
        - `feature_values` (legacy alias)
        - `_ts_event`/`_ts_init` or `ts_event`/`ts_init`
        - `feature_set_id` optional (defaults to "default")

        """
        self.feature_set_id = feature_set_id or "default"
        self.instrument_id = instrument_id
        resolved_values = (
            features
            if features is not None
            else feature_values
            if feature_values is not None
            else values
        )
        self.values = dict(resolved_values or {})
        # Event/init timestamps: prefer public aliases when provided
        evt = ts_event if ts_event is not None else _ts_event
        init = ts_init if ts_init is not None else _ts_init
        self._ts_event = int(evt) if evt is not None else 0
        self._ts_init = int(init) if init is not None else self._ts_event
        self.quality_flags = int(quality_flags)

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

    # Backwards-compat: some tests refer to `.features` instead of `.values`
    @property
    def features(self) -> dict[str, float]:  # pragma: no cover - alias for compatibility
        return self.feature_values

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
    is_live: bool = False

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
    run_id : str | None
        Optional run identifier for replay/audit correlation.
    ingested_at_ns : int | None
        Ingestion timestamp in nanoseconds.
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
    run_id: str | None = None
    ingested_at_ns: int | None = None

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
class StrategyOrderEvent(NautilusData):
    """
    Store order events for strategy execution audit.

    Attributes
    ----------
    event_id : str
        Unique event identifier.
    strategy_id : str
        Strategy identifier.
    instrument_id : str
        Instrument identifier.
    client_order_id : str
        Client order identifier.
    venue_order_id : str | None
        Venue order identifier when available.
    event_type : str
        Order event type (e.g., OrderSubmitted).
    payload : dict[str, Any]
        Full order event payload for audit.
    run_id : str | None
        Optional run identifier for replay/audit correlation.
    ingested_at_ns : int | None
        Ingestion timestamp in nanoseconds.
    _ts_event : int
        Event timestamp in nanoseconds.
    _ts_init : int
        Initialization timestamp in nanoseconds.
    is_live : bool
        Whether this event occurred in live trading.

    """

    event_id: str
    strategy_id: str
    instrument_id: str
    client_order_id: str
    venue_order_id: str | None
    event_type: str
    payload: dict[str, Any]
    _ts_event: int
    _ts_init: int
    is_live: bool = False
    run_id: str | None = None
    ingested_at_ns: int | None = None

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

    @staticmethod
    def _coerce_str(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        attr = getattr(value, "value", None)
        if attr is not None:
            return str(attr)
        return str(value)

    @classmethod
    def from_event(
        cls,
        event: object,
        *,
        is_live: bool = False,
        run_id: str | None = None,
        ingested_at_ns: int | None = None,
        logger: logging.Logger | None = None,
        context: str = "StrategyOrderEvent.from_event",
    ) -> StrategyOrderEvent | None:
        """
        Build a StrategyOrderEvent from a Nautilus order event.

        Parameters
        ----------
        event : object
            Order event instance with ``to_dict`` and order metadata.
        is_live : bool, optional
            Whether the event is from live trading.
        logger : logging.Logger | None, optional
            Logger for debug output on extraction failure.
        context : str, optional
            Context label for timestamp sanitization.

        Returns
        -------
        StrategyOrderEvent | None
            Parsed order event, or ``None`` when required fields are missing.

        """
        from ml.common.timestamps import sanitize_timestamp_ns

        raw: dict[str, Any] = {}
        try:
            to_dict = getattr(event, "to_dict", None)
            if callable(to_dict):
                try:
                    raw = to_dict()
                except TypeError:
                    raw = to_dict(event)
        except Exception as exc:
            if logger is not None:
                logger.debug(
                    "strategy_order_event_to_dict_failed",
                    exc_info=True,
                    extra={"context": context, "error": str(exc)},
                )
            return None

        if not isinstance(raw, dict):
            raw = {}

        event_type = cls._coerce_str(raw.get("type")) or type(event).__name__
        event_id = cls._coerce_str(raw.get("event_id") or getattr(event, "id", None))
        strategy_id = cls._coerce_str(raw.get("strategy_id") or getattr(event, "strategy_id", None))
        instrument_id = cls._coerce_str(raw.get("instrument_id") or getattr(event, "instrument_id", None))
        client_order_id = cls._coerce_str(
            raw.get("client_order_id") or getattr(event, "client_order_id", None),
        )
        venue_order_id = cls._coerce_str(
            raw.get("venue_order_id") or getattr(event, "venue_order_id", None),
        )
        ts_event_raw = raw.get("ts_event", getattr(event, "ts_event", None))
        ts_init_raw = raw.get("ts_init", getattr(event, "ts_init", None))

        if None in (event_id, strategy_id, instrument_id, client_order_id):
            if logger is not None:
                logger.debug(
                    "strategy_order_event_missing_fields",
                    extra={
                        "context": context,
                        "event_id": event_id,
                        "strategy_id": strategy_id,
                        "instrument_id": instrument_id,
                        "client_order_id": client_order_id,
                        "ts_event": ts_event_raw,
                    },
                )
            return None
        if ts_event_raw is None:
            if logger is not None:
                logger.debug(
                    "strategy_order_event_missing_ts_event",
                    extra={"context": context, "event_id": event_id},
                )
            return None

        ts_event = sanitize_timestamp_ns(
            int(ts_event_raw),
            logger=logger,
            context=f"{context}:ts_event",
        )
        ts_init_base = ts_init_raw if ts_init_raw is not None else ts_event
        ts_init = sanitize_timestamp_ns(
            int(ts_init_base),
            logger=logger,
            context=f"{context}:ts_init",
        )
        if ts_init < ts_event:
            ts_init = ts_event

        payload = dict(raw) if raw else {}
        if ingested_at_ns is None:
            ingested_at_ns = time.time_ns()
        return cls(
            event_id=str(event_id),
            strategy_id=str(strategy_id),
            instrument_id=str(instrument_id),
            client_order_id=str(client_order_id),
            venue_order_id=venue_order_id,
            event_type=str(event_type),
            payload=payload,
            _ts_event=ts_event,
            _ts_init=ts_init,
            is_live=is_live,
            run_id=run_id,
            ingested_at_ns=ingested_at_ns,
        )


@dataclass
class StrategyRiskHaltEvent(NautilusData):
    """
    Store risk-halt transitions for strategy audit.

    Attributes
    ----------
    event_id : str
        Unique event identifier.
    strategy_id : str
        Strategy identifier.
    instrument_id : str
        Instrument identifier.
    event_type : str
        Halt transition type (e.g., "halted", "resumed").
    reason : str
        Halt reason label.
    detail : str | None
        Optional detail for the halt reason.
    run_id : str | None
        Optional run identifier for replay/audit correlation.
    ingested_at_ns : int | None
        Ingestion timestamp in nanoseconds.
    _ts_event : int
        Event timestamp in nanoseconds.
    _ts_init : int
        Initialization timestamp in nanoseconds.
    is_live : bool
        Whether this event occurred in live trading.

    """

    event_id: str
    strategy_id: str
    instrument_id: str
    event_type: str
    reason: str
    detail: str | None
    _ts_event: int
    _ts_init: int
    is_live: bool = False
    run_id: str | None = None
    ingested_at_ns: int | None = None

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
class StrategyReplaySummary(NautilusData):
    """
    Store replay summary statistics for fast audits.

    Attributes
    ----------
    run_id : str
        Replay run identifier.
    instrument_ids : list[str]
        Instruments included in the replay.
    started_ns : int | None
        Run start timestamp in nanoseconds when available.
    finished_ns : int | None
        Run end timestamp in nanoseconds when available.
    total_orders : int
        Total orders submitted during the replay.
    total_fills : int
        Total fills recorded during the replay.
    total_halts : int
        Total risk-halt transitions recorded.
    total_sizing_rejects : int
        Total sizing rejections recorded.
    total_positions : int
        Total positions opened during the replay.
    _ts_event : int
        Event timestamp in nanoseconds.
    _ts_init : int
        Initialization timestamp in nanoseconds.
    ingested_at_ns : int | None
        Ingestion timestamp in nanoseconds.

    """

    run_id: str
    instrument_ids: list[str]
    started_ns: int | None
    finished_ns: int | None
    total_orders: int
    total_fills: int
    total_halts: int
    total_sizing_rejects: int
    total_positions: int
    _ts_event: int
    _ts_init: int
    ingested_at_ns: int | None = None

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


class BaseStore(MLComponentMixin, ABC):
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

    def write_order_event(self, *args: object, **kwargs: object) -> None:
        """
        Write order event (dummy).
        """

    def write_batch(self, data: list[Any], emit_events: bool = True) -> None:
        """
        Batch write (dummy).
        """
        return None

    def flush(self, *args: object, **kwargs: object) -> None:
        """
        Flush buffered state (dummy).
        """

    # Backward-compatible alias used in some tests
    def get_stats(self, *args: object, **kwargs: object) -> dict[str, Any]:
        """
        Deprecated alias for get_statistics (retained for compatibility).
        """
        return self.get_statistics()

    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
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
    def read_predictions(
        self,
        model_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object | None:
        return None

    def get_model_performance(
        self,
        model_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        return {}

    # Strategy store protocol methods (dummy implementations)
    def read_signals(
        self,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object | None:
        return None

    def get_strategy_performance(
        self,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        return {}

    def get_signal_distribution(
        self,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]:
        return {}

    # Feature store extension (dummy realtime computation)
    def compute_realtime(
        self,
        bar: object,
        store: bool = True,
        indicator_manager: object | None = None,
    ) -> object | None:
        return None

    def __getattr__(self, name: str) -> object:
        """
        Handle any other method calls.
        """

        def dummy_method(*args: object, **kwargs: object) -> None:
            return None

        return dummy_method
