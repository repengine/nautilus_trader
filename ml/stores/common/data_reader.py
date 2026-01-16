#!/usr/bin/env python3

"""
Data reader component for DataStore.

Extracted from DataStore (Phase 2.4.3). Provides data reading operations
with time-travel queries for features, predictions, signals, and earnings data.

CRITICAL: get_features_at_or_before() is HOT PATH (P99 < 5ms)
All other methods are COLD PATH (async acceptable).

"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from ml._imports import HAS_PROMETHEUS
from ml.common.metrics_bootstrap import get_histogram
from ml.registry.dataclasses import DatasetType


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.earnings_store import EarningsStore
    from ml.stores.feature_store_facade import FeatureStore
    from ml.stores.io_raw import RawReaderProtocol
    from ml.stores.model_store import ModelStore
    from ml.stores.strategy_store import StrategyStore

logger = logging.getLogger(__name__)


class PredictionRecord:
    """
    Lightweight prediction record for point-in-time queries.

    Attributes
    ----------
    model_id : str
        Model identifier
    ts_event : int
        Event timestamp in nanoseconds
    prediction : float
        Prediction value
    confidence : float
        Confidence score (0-1)

    """

    def __init__(
        self,
        model_id: str,
        ts_event: int,
        prediction: float,
        confidence: float,
    ) -> None:
        """
        Initialize prediction record.

        Args:
            model_id: Model identifier
            ts_event: Event timestamp in nanoseconds
            prediction: Prediction value
            confidence: Confidence score

        """
        self.model_id = model_id
        self.ts_event = ts_event
        self.prediction = prediction
        self.confidence = confidence


class SignalRecord:
    """
    Lightweight signal record for point-in-time queries.

    Attributes
    ----------
    strategy_id : str
        Strategy identifier
    ts_event : int
        Event timestamp in nanoseconds
    signal : float
        Signal value
    strength : float
        Signal strength

    """

    def __init__(
        self,
        strategy_id: str,
        ts_event: int,
        signal: float,
        strength: float,
    ) -> None:
        """
        Initialize signal record.

        Args:
            strategy_id: Strategy identifier
            ts_event: Event timestamp in nanoseconds
            signal: Signal value
            strength: Signal strength

        """
        self.strategy_id = strategy_id
        self.ts_event = ts_event
        self.signal = signal
        self.strength = strength


# =========================================================================
# Prometheus Metrics (using centralized bootstrap - CLAUDE.md Pattern 5)
# =========================================================================


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
read_latency_histogram = get_histogram(
    "ml_datastore_read_latency_seconds",
    "Data read operation latency in seconds",
    labelnames=["operation", "store"],
)


# =========================================================================
# DataReaderComponent
# =========================================================================


class DataReaderComponent:
    """
    Data reading operations for DataStore.

    Extracted from DataStore (Phase 2.4.3).

    HOT PATH: get_features_at_or_before() must achieve P99 < 5ms
    COLD PATH: All bulk read methods (async acceptable)

    Provides:
    - Point-in-time feature retrieval (HOT PATH)
    - Point-in-time prediction/signal queries
    - Range queries for bulk data access
    - Earnings data queries

    Example
    -------
    >>> from ml.stores.common.data_reader import DataReaderComponent
    >>> reader = DataReaderComponent(
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     earnings_store=earnings_store,
    ...     registry=registry,
    ... )
    >>> features = reader.get_features_at_or_before(
    ...     instrument_id="EURUSD.SIM",
    ...     ts_event=1699999990000000000,
    ... )

    """

    def __init__(
        self,
        feature_store: FeatureStore,
        model_store: ModelStore,
        strategy_store: StrategyStore,
        earnings_store: EarningsStore,
        registry: RegistryProtocol,
        *,
        raw_reader: RawReaderProtocol | None = None,
    ) -> None:
        """
        Initialize data reader with store dependencies.

        Args:
            feature_store: FeatureStore for feature data
            model_store: ModelStore for prediction data
            strategy_store: StrategyStore for signal data
            earnings_store: EarningsStore for earnings data
            registry: Data registry for manifest lookup

        """
        self._feature_store = feature_store
        self._model_store = model_store
        self._strategy_store = strategy_store
        self._earnings_store = earnings_store
        self._registry = registry
        self._raw_reader = raw_reader

    # =========================================================================
    # Public API - HOT PATH
    # =========================================================================

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before the given timestamp.

        EXTRACTED FROM: ml/stores/data_store_facade.py:536
        HOT PATH: P99 < 5ms requirement

        This is a thin facade over FeatureStore.get_latest_at_or_before.
        Returns ALL features as dict[str, float], not a filtered subset.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier (e.g., "EURUSD.SIM")
        ts_event : int
            Timestamp in nanoseconds (point-in-time query)

        Returns
        -------
        dict[str, float] | None
            Dictionary mapping feature names to values (all features),
            or None if no features exist before timestamp.

        Examples
        --------
        >>> features = reader.get_features_at_or_before(
        ...     instrument_id="EURUSD.SIM",
        ...     ts_event=1699999990000000000,
        ... )
        >>> if features:
        ...     print(f"Close price: {features['close']}")

        """
        import time

        start_time = time.perf_counter()

        try:
            # Delegate to FeatureStore (pre-optimized for hot path)
            result = self._feature_store.get_latest_at_or_before(instrument_id, int(ts_event))

            # Record latency metric
            if HAS_PROMETHEUS:
                latency_seconds = time.perf_counter() - start_time
                read_latency_histogram.labels(
                    operation="get_features_at_or_before",
                    store="feature",
                ).observe(latency_seconds)

            return result
        except Exception as exc:
            logger.debug(
                "get_features_at_or_before failed for %s at %d: %s",
                instrument_id,
                ts_event,
                exc,
                exc_info=True,
            )
            return None

    # =========================================================================
    # Public API - COLD PATH
    # =========================================================================

    def read_ingestion_data(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        dataset_type: DatasetType | None = None,
    ) -> pl.DataFrame:
        """
        Read ingestion data for a specific time range.

        COLD PATH: Bulk read operation (async acceptable)

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start_ts : int
            Start timestamp in nanoseconds (inclusive)
        end_ts : int
            End timestamp in nanoseconds (exclusive)
        dataset_type : DatasetType | None
            Optional dataset type filter (BARS, QUOTES, TRADES)

        Returns
        -------
        pl.DataFrame
            Ingestion data for the specified range

        Raises
        ------
        ValueError
            If invalid time range (start >= end)

        Examples
        --------
        >>> df = reader.read_ingestion_data(
        ...     instrument_id="EURUSD.SIM",
        ...     start_ts=1699999900000000000,
        ...     end_ts=1699999990000000000,
        ... )
        >>> print(f"Loaded {len(df)} rows")

        """
        import time

        start_time = time.perf_counter()

        # Validate time range
        if start_ts >= end_ts:
            raise ValueError(f"Invalid time range: start={start_ts} >= end={end_ts}")

        try:
            raw_result: object
            if self._raw_reader is not None and dataset_type is not None:
                raw_result = self._raw_reader.read_range(
                    dataset_type=dataset_type,
                    instrument_id=instrument_id,
                    start_ns=start_ts,
                    end_ns=end_ts,
                )
            else:
                # Convert to datetime for store queries
                start_dt = datetime.fromtimestamp(start_ts / 1e9)
                end_dt = datetime.fromtimestamp(end_ts / 1e9)

                # For ingestion data, use feature store's training data method
                raw_result = self._feature_store.get_training_data(
                    instrument_id=instrument_id,
                    start=start_dt,
                    end=end_dt,
                )

            # Convert to Polars DataFrame if not already
            if isinstance(raw_result, pl.DataFrame):
                result = raw_result
            elif isinstance(raw_result, tuple):
                # Handle tuple format (values, timestamps, columns)
                result = pl.DataFrame()
            else:
                # Try to convert other formats
                result = pl.DataFrame(raw_result)

            # Record latency metric
            if HAS_PROMETHEUS:
                latency_seconds = time.perf_counter() - start_time
                read_latency_histogram.labels(
                    operation="read_ingestion_data",
                    store="feature",
                ).observe(latency_seconds)

            logger.debug(
                "Read %d ingestion rows for %s (%d - %d)",
                len(result),
                instrument_id,
                start_ts,
                end_ts,
            )

            return result
        except Exception as exc:
            logger.error(
                "read_ingestion_data failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            # Return empty DataFrame on error
            return pl.DataFrame()

    def read_features(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Read feature data for a specific time range.

        COLD PATH: Bulk read operation (async acceptable)

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start_ts : int
            Start timestamp in nanoseconds (inclusive)
        end_ts : int
            End timestamp in nanoseconds (exclusive)
        feature_names : list[str] | None
            Optional list of feature names to retrieve (returns all if None)

        Returns
        -------
        pl.DataFrame
            Feature data for the specified range

        Raises
        ------
        ValueError
            If invalid time range (start >= end)

        Examples
        --------
        >>> df = reader.read_features(
        ...     instrument_id="EURUSD.SIM",
        ...     start_ts=1699999900000000000,
        ...     end_ts=1699999990000000000,
        ...     feature_names=["close", "volume"],
        ... )

        """
        import time

        start_time = time.perf_counter()

        # Validate time range
        if start_ts >= end_ts:
            raise ValueError(f"Invalid time range: start={start_ts} >= end={end_ts}")

        try:
            # Convert to datetime for FeatureStore
            start_dt = datetime.fromtimestamp(start_ts / 1e9)
            end_dt = datetime.fromtimestamp(end_ts / 1e9)

            # Use feature store's training data method (may return tuple or DataFrame)
            raw_result = self._feature_store.get_training_data(
                instrument_id=instrument_id,
                start=start_dt,
                end=end_dt,
            )

            # Convert to Polars DataFrame if not already
            if isinstance(raw_result, pl.DataFrame):
                result = raw_result
            elif isinstance(raw_result, tuple):
                # Handle tuple format (values, timestamps, columns)
                result = pl.DataFrame()
            else:
                # Try to convert other formats
                result = pl.DataFrame(raw_result)

            # Filter columns if feature_names specified
            if feature_names is not None and len(result) > 0:
                available_columns = set(result.columns)
                requested_columns = set(feature_names)
                valid_columns = list(requested_columns.intersection(available_columns))
                if valid_columns:
                    result = result.select(valid_columns)

            # Record latency metric
            if HAS_PROMETHEUS:
                latency_seconds = time.perf_counter() - start_time
                read_latency_histogram.labels(
                    operation="read_features",
                    store="feature",
                ).observe(latency_seconds)

            logger.debug(
                "Read %d feature rows for %s (%d - %d)",
                len(result),
                instrument_id,
                start_ts,
                end_ts,
            )

            return result
        except Exception as exc:
            logger.error(
                "read_features failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            # Return empty DataFrame on error
            return pl.DataFrame()

    def read_predictions(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        model_id: str | None = None,
    ) -> pl.DataFrame:
        """
        Read model predictions for a specific time range.

        COLD PATH: Bulk read operation (async acceptable)

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start_ts : int
            Start timestamp in nanoseconds (inclusive)
        end_ts : int
            End timestamp in nanoseconds (exclusive)
        model_id : str | None
            Optional model identifier filter

        Returns
        -------
        pl.DataFrame
            Prediction data for the specified range

        Raises
        ------
        ValueError
            If invalid time range (start >= end)

        Examples
        --------
        >>> df = reader.read_predictions(
        ...     instrument_id="EURUSD.SIM",
        ...     start_ts=1699999900000000000,
        ...     end_ts=1699999990000000000,
        ...     model_id="xgb_v1",
        ... )

        """
        import time

        start_time = time.perf_counter()

        # Validate time range
        if start_ts >= end_ts:
            raise ValueError(f"Invalid time range: start={start_ts} >= end={end_ts}")

        try:
            # Delegate to ModelStore (expects method read_predictions)
            raw_result = self._model_store.read_predictions(
                model_id=model_id or "default",
                instrument_id=instrument_id,
                start_ns=start_ts,
                end_ns=end_ts,
            )

            # Convert to Polars DataFrame if not already
            if isinstance(raw_result, pl.DataFrame):
                result = raw_result
            else:
                # Handle pandas DataFrame or list
                import pandas as pd

                if isinstance(raw_result, pd.DataFrame):
                    result = pl.from_pandas(raw_result)
                else:
                    result = pl.DataFrame(raw_result)

            # Record latency metric
            if HAS_PROMETHEUS:
                latency_seconds = time.perf_counter() - start_time
                read_latency_histogram.labels(
                    operation="read_predictions",
                    store="model",
                ).observe(latency_seconds)

            logger.debug(
                "Read %d prediction rows for %s (%d - %d)",
                len(result),
                instrument_id,
                start_ts,
                end_ts,
            )

            return result
        except Exception as exc:
            logger.error(
                "read_predictions failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            # Return empty DataFrame on error
            return pl.DataFrame()

    def read_signals(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        strategy_id: str | None = None,
    ) -> pl.DataFrame:
        """
        Read strategy signals for a specific time range.

        COLD PATH: Bulk read operation (async acceptable)

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start_ts : int
            Start timestamp in nanoseconds (inclusive)
        end_ts : int
            End timestamp in nanoseconds (exclusive)
        strategy_id : str | None
            Optional strategy identifier filter

        Returns
        -------
        pl.DataFrame
            Signal data for the specified range

        Raises
        ------
        ValueError
            If invalid time range (start >= end)

        Examples
        --------
        >>> df = reader.read_signals(
        ...     instrument_id="EURUSD.SIM",
        ...     start_ts=1699999900000000000,
        ...     end_ts=1699999990000000000,
        ...     strategy_id="rsi_v1",
        ... )

        """
        import time

        start_time = time.perf_counter()

        # Validate time range
        if start_ts >= end_ts:
            raise ValueError(f"Invalid time range: start={start_ts} >= end={end_ts}")

        try:
            # Delegate to StrategyStore (expects method read_signals)
            raw_result = self._strategy_store.read_signals(
                strategy_id=strategy_id or "default",
                instrument_id=instrument_id,
                start_ns=start_ts,
                end_ns=end_ts,
            )

            # Convert to Polars DataFrame if not already
            if isinstance(raw_result, pl.DataFrame):
                result = raw_result
            else:
                # Handle pandas DataFrame or list
                import pandas as pd

                if isinstance(raw_result, pd.DataFrame):
                    result = pl.from_pandas(raw_result)
                else:
                    result = pl.DataFrame(raw_result)

            # Record latency metric
            if HAS_PROMETHEUS:
                latency_seconds = time.perf_counter() - start_time
                read_latency_histogram.labels(
                    operation="read_signals",
                    store="strategy",
                ).observe(latency_seconds)

            logger.debug(
                "Read %d signal rows for %s (%d - %d)",
                len(result),
                instrument_id,
                start_ts,
                end_ts,
            )

            return result
        except Exception as exc:
            logger.error(
                "read_signals failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            # Return empty DataFrame on error
            return pl.DataFrame()

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None:
        """
        Return latest prediction at or before ts_event (optionally filtered by model_id).

        EXTRACTED FROM: ml/stores/data_store_facade.py:552

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Timestamp in nanoseconds (point-in-time query)
        model_id : str | None
            Optional model identifier filter

        Returns
        -------
        PredictionRecord | None
            Minimal typed record or None when not found.

        Examples
        --------
        >>> pred = reader.get_latest_prediction_at_or_before(
        ...     instrument_id="EURUSD.SIM",
        ...     ts_event=1699999990000000000,
        ...     model_id="xgb_v1",
        ... )
        >>> if pred:
        ...     print(f"Prediction: {pred.prediction}, Confidence: {pred.confidence}")

        """
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self._model_store, "model_predictions_table", None)
        engine = getattr(self._model_store, "engine", None)
        if table is None or engine is None:
            logger.debug("ModelStore missing prediction table or engine")
            return None

        where = [table.c.instrument_id == instrument_id, table.c.ts_event <= int(ts_event)]
        if model_id is not None:
            where.append(table.c.model_id == model_id)

        stmt = (
            _select(
                table.c.model_id,
                table.c.ts_event,
                table.c.prediction,
                table.c.confidence,
            )
            .where(_and(*where))
            .order_by(_desc(table.c.ts_event))
            .limit(1)
        )

        try:
            with engine.connect() as conn:
                row = conn.execute(stmt).fetchone()
            if row is None:
                return None
            return PredictionRecord(
                model_id=str(row[0]),
                ts_event=int(row[1]),
                prediction=float(row[2]) if row[2] is not None else 0.0,
                confidence=float(row[3]) if row[3] is not None else 0.0,
            )
        except Exception as exc:
            logger.error(
                "get_latest_prediction_at_or_before failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            return None

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> SignalRecord | None:
        """
        Return latest strategy signal at or before ts_event (optionally by strategy_id).

        EXTRACTED FROM: ml/stores/data_store_facade.py:606

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Timestamp in nanoseconds (point-in-time query)
        strategy_id : str | None
            Optional strategy identifier filter

        Returns
        -------
        SignalRecord | None
            Minimal typed record or None when not found.

        Examples
        --------
        >>> signal = reader.get_latest_signal_at_or_before(
        ...     instrument_id="EURUSD.SIM",
        ...     ts_event=1699999990000000000,
        ...     strategy_id="rsi_v1",
        ... )
        >>> if signal:
        ...     print(f"Signal: {signal.signal}, Strength: {signal.strength}")

        """
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self._strategy_store, "strategy_signals_table", None)
        engine = getattr(self._strategy_store, "engine", None)
        if table is None or engine is None:
            logger.debug("StrategyStore missing signal table or engine")
            return None

        where = [table.c.instrument_id == instrument_id, table.c.ts_event <= int(ts_event)]
        if strategy_id is not None:
            where.append(table.c.strategy_id == strategy_id)

        stmt = (
            _select(
                table.c.strategy_id,
                table.c.ts_event,
                table.c.signal,
                table.c.strength,
            )
            .where(_and(*where))
            .order_by(_desc(table.c.ts_event))
            .limit(1)
        )

        try:
            with engine.connect() as conn:
                row = conn.execute(stmt).fetchone()
            if row is None:
                return None
            return SignalRecord(
                strategy_id=str(row[0]),
                ts_event=int(row[1]),
                signal=float(row[2]) if row[2] is not None else 0.0,
                strength=float(row[3]) if row[3] is not None else 0.0,
            )
        except Exception as exc:
            logger.error(
                "get_latest_signal_at_or_before failed for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            return None

    def read_earnings_actual(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> pl.DataFrame:
        """
        Read earnings actuals for a ticker within date range.

        COLD PATH: Bulk read operation (async acceptable)
        EXTRACTED FROM: ml/stores/data_store_facade.py:658

        Parameters
        ----------
        symbol : str
            Stock ticker symbol
        start_date : str | None
            Start date (YYYY-MM-DD format) or None for all dates
        end_date : str | None
            End date (YYYY-MM-DD format) or None for all dates
        as_of_ts : int | None
            Optional point-in-time timestamp (nanoseconds) for time-travel query

        Returns
        -------
        pl.DataFrame
            Earnings actuals data

        Examples
        --------
        >>> df = reader.read_earnings_actual(
        ...     symbol="AAPL",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )

        """
        try:
            # Delegate to EarningsStore
            records = self._earnings_store.get_actuals(
                ticker=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_ts=as_of_ts,
            )

            # Convert to Polars DataFrame
            if isinstance(records, list):
                return pl.DataFrame(records)
            elif isinstance(records, pl.DataFrame):
                return records
            else:
                return pl.DataFrame()
        except Exception as exc:
            logger.error(
                "read_earnings_actual failed for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )
            return pl.DataFrame()

    def read_earnings_estimate(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> pl.DataFrame:
        """
        Read earnings estimates for a ticker within date range.

        COLD PATH: Bulk read operation (async acceptable)
        EXTRACTED FROM: ml/stores/data_store_facade.py:688

        Parameters
        ----------
        symbol : str
            Stock ticker symbol
        start_date : str | None
            Start date (YYYY-MM-DD format) or None for all dates
        end_date : str | None
            End date (YYYY-MM-DD format) or None for all dates
        as_of_ts : int | None
            Optional point-in-time timestamp (nanoseconds) for time-travel query

        Returns
        -------
        pl.DataFrame
            Earnings estimates data

        Examples
        --------
        >>> df = reader.read_earnings_estimate(
        ...     symbol="AAPL",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )

        """
        try:
            # Delegate to EarningsStore
            # Note: get_estimates requires period_end, so we query with start_date if available
            if start_date:
                records = self._earnings_store.get_estimates(
                    ticker=symbol,
                    period_end=start_date,  # Use start_date as period_end filter
                    as_of_ts=as_of_ts,
                )
            else:
                # Return empty if no date filter provided
                return pl.DataFrame()

            # Convert to Polars DataFrame
            if isinstance(records, list):
                return pl.DataFrame(records)
            elif isinstance(records, dict):
                # Single record - wrap in list
                return pl.DataFrame([records])
            elif isinstance(records, pl.DataFrame):
                return records
            else:
                return pl.DataFrame()
        except Exception as exc:
            logger.error(
                "read_earnings_estimate failed for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )
            return pl.DataFrame()
