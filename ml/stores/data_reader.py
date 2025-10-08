#!/usr/bin/env python3

"""
Read operations for ML data stores.

This module provides focused read operations across FeatureStore, ModelStore,
StrategyStore, and EarningsStore for cold-path queries. Extracted from the
monolithic DataStore class for better maintainability and testability.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import PredictionRecord
from ml.stores.protocols import SignalRecord


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class DataReaderProtocol(Protocol):
    """Protocol for data read operations."""

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds

        Returns
        -------
        dict[str, float] | None
            Feature values or None if not found
        """
        ...

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None:
        """
        Return latest prediction at or before timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds
        model_id : str | None
            Optional model identifier filter

        Returns
        -------
        PredictionRecord | None
            Prediction record or None if not found
        """
        ...

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> SignalRecord | None:
        """
        Return latest signal at or before timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds
        strategy_id : str | None
            Optional strategy identifier filter

        Returns
        -------
        SignalRecord | None
            Signal record or None if not found
        """
        ...

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return earnings actuals visible at timestamp.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        ts_event : int
            Event timestamp in nanoseconds
        limit : int
            Maximum number of records to return
        start_date : str | None
            Start date filter (ISO format)
        end_date : str | None
            End date filter (ISO format)

        Returns
        -------
        list[dict[str, Any]]
            List of earnings actual records
        """
        ...

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """
        Return latest consensus estimate for period at timestamp.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format)
        ts_event : int
            Event timestamp in nanoseconds

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found
        """
        ...


# ========================================================================
# DataReader Implementation
# ========================================================================


class DataReader:
    """
    Performs read operations across ML stores.

    Provides typed read facades over FeatureStore, ModelStore, StrategyStore,
    and EarningsStore for cold-path queries. This component is stateless and
    delegates to underlying stores.

    This component is extracted from the DataStore god class to provide focused,
    testable read functionality following the Strangler Fig pattern.

    Parameters
    ----------
    feature_store : Any
        Feature store instance (FeatureStoreProtocol)
    model_store : Any
        Model store instance (ModelStoreProtocol)
    strategy_store : Any
        Strategy store instance (StrategyStoreProtocol)
    earnings_store : EarningsStoreProtocol
        Earnings store instance

    Examples
    --------
    >>> reader = DataReader(
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     earnings_store=earnings_store,
    ... )
    >>> features = reader.get_features_at_or_before(
    ...     instrument_id="EURUSD.SIM",
    ...     ts_event=1234567890000000000,
    ... )
    """

    def __init__(
        self,
        *,
        feature_store: Any,
        model_store: Any,
        strategy_store: Any,
        earnings_store: EarningsStoreProtocol,
    ) -> None:
        """
        Initialize data reader with store dependencies.

        Parameters
        ----------
        feature_store : Any
            Feature store instance (FeatureStoreProtocol)
        model_store : Any
            Model store instance (ModelStoreProtocol)
        strategy_store : Any
            Strategy store instance (StrategyStoreProtocol)
        earnings_store : EarningsStoreProtocol
            Earnings store instance
        """
        self.feature_store = feature_store
        self.model_store = model_store
        self.strategy_store = strategy_store
        self.earnings_store = earnings_store
        logger.debug("Initialized DataReader")

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before the given timestamp.

        This is a thin facade over FeatureStore.get_latest_at_or_before.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds

        Returns
        -------
        dict[str, float] | None
            Feature values or None if not found

        Notes
        -----
        Cold-path only. Performance target: <5ms P99.
        """
        return self.feature_store.get_latest_at_or_before(instrument_id, int(ts_event))

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None:
        """
        Return latest prediction at or before ts_event (optionally filtered by model_id).

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds
        model_id : str | None
            Optional model identifier filter

        Returns
        -------
        PredictionRecord | None
            Minimal typed record or None when not found

        Notes
        -----
        Cold-path only. Queries model_predictions table directly.
        """
        # Lazy imports to keep import-time overhead minimal
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self.model_store, "model_predictions_table", None)
        engine = getattr(self.model_store, "engine", None)
        if table is None or engine is None:
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

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> SignalRecord | None:
        """
        Return latest strategy signal at or before ts_event (optionally by strategy_id).

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds
        strategy_id : str | None
            Optional strategy identifier filter

        Returns
        -------
        SignalRecord | None
            Minimal typed record or None when not found

        Notes
        -----
        Cold-path only. Queries strategy_signals table directly.
        """
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self.strategy_store, "strategy_signals_table", None)
        engine = getattr(self.strategy_store, "engine", None)
        if table is None or engine is None:
            return None

        where = [table.c.instrument_id == instrument_id, table.c.ts_event <= int(ts_event)]
        if strategy_id is not None:
            where.append(table.c.strategy_id == strategy_id)

        stmt = (
            _select(
                table.c.strategy_id,
                table.c.ts_event,
                table.c.signal_type,
                table.c.strength,
            )
            .where(_and(*where))
            .order_by(_desc(table.c.ts_event))
            .limit(1)
        )

        with engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        if row is None:
            return None
        return SignalRecord(
            strategy_id=str(row[0]),
            ts_event=int(row[1]),
            signal_type=str(row[2]),
            strength=float(row[3]) if row[3] is not None else 0.0,
        )

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return earnings actuals visible at the specified timestamp.

        Records are filtered point-in-time and truncated to limit entries.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        ts_event : int
            Event timestamp in nanoseconds
        limit : int
            Maximum number of records to return (default: 5)
        start_date : str | None
            Start date filter (ISO format)
        end_date : str | None
            End date filter (ISO format)

        Returns
        -------
        list[dict[str, Any]]
            List of earnings actual records (empty list if none found)

        Notes
        -----
        Cold-path only. Uses point-in-time filtering for backtest correctness.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        if limit <= 0:
            return []

        as_of_ts = _sanitize_ts(int(ts_event), context="data_reader.get_earnings_actuals_at_or_before:ts_event")
        records = self.earnings_store.get_actuals(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )
        if len(records) <= limit:
            return list(records)
        return list(records[:limit])

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """
        Return the latest consensus estimate for the specified period at ts_event.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format)
        ts_event : int
            Event timestamp in nanoseconds

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found

        Notes
        -----
        Cold-path only. Uses point-in-time filtering for backtest correctness.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        as_of_ts = _sanitize_ts(int(ts_event), context="data_reader.get_earnings_estimate_at_or_before:ts_event")
        return self.earnings_store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        )
