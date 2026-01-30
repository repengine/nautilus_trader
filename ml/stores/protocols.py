"""
Protocols for ML store interfaces.

These provide structural contracts for store implementations so mypy can detect
interface drift across implementations and tests.

"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict, runtime_checkable

import pandas as pd


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.validation_types import DataEvent
else:  # pragma: no cover - typing fallback
    DataEvent = Any  # type: ignore[assignment]


# Phase 1: introduce aliases for read/write frames to retain flexibility
ReadFrame: TypeAlias = pd.DataFrame
WriteRecords: TypeAlias = list[dict[str, Any]]


class BaseStoreProtocol(Protocol):
    def write_batch(self, data: list[Any]) -> None: ...
    def read_range(self, start_ns: int, end_ns: int, instrument_id: str | None = None) -> Any: ...
    def flush(self) -> None: ...
    def get_latest(self, instrument_id: str, limit: int = 1) -> Any: ...
    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]: ...


class FeatureStoreProtocol(Protocol):
    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: dict[str, float] | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Any | None = None,
    ) -> None: ...
    def flush(self) -> None: ...
    def compute_realtime(
        self,
        bar: Any,
        store: bool = ...,
        indicator_manager: Any | None = ...,
    ) -> Any: ...


class ModelStoreProtocol(Protocol):
    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None: ...
    def write_batch(self, data: list[Any], emit_events: bool = True) -> None: ...
    def read_predictions(
        self,
        model_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> Any: ...
    def get_model_performance(
        self,
        model_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]: ...
    def flush(self) -> None: ...


class StrategyStoreProtocol(Protocol):
    def write_signal(
        self,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        ts_event: int,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_order_event(
        self,
        event: object,
        *,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_risk_halt_event(
        self,
        *,
        strategy_id: str,
        instrument_id: str,
        event_type: str,
        reason: str,
        detail: str | None,
        ts_event: int,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_replay_summary(self, summary: Any) -> None: ...
    def write_batch(self, data: list[Any]) -> None: ...
    def read_signals(
        self,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> Any: ...
    def get_strategy_performance(
        self,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]: ...
    def get_signal_distribution(
        self,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]: ...
    def flush(self) -> None: ...


class CoverageProviderProtocol(Protocol):
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = ...,
        start_ns: int,
        end_ns: int,
    ) -> set[int]: ...


class MarketDataWriterProtocol(Protocol):
    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int: ...


# Optional stricter protocols for new components (adopt incrementally)


@runtime_checkable
class FeatureStoreStrictProtocol(Protocol):
    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: Mapping[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None: ...
    def flush(self) -> None: ...


@runtime_checkable
class ModelStoreStrictProtocol(Protocol):
    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: Mapping[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None: ...
    def write_batch(self, data: Sequence[Any], emit_events: bool = True) -> None: ...
    def flush(self) -> None: ...


@runtime_checkable
class StrategyStoreStrictProtocol(Protocol):
    def write_signal(
        self,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: Mapping[str, float],
        risk_metrics: Mapping[str, float],
        execution_params: Mapping[str, Any],
        ts_event: int,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_order_event(
        self,
        event: object,
        *,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_risk_halt_event(
        self,
        *,
        strategy_id: str,
        instrument_id: str,
        event_type: str,
        reason: str,
        detail: str | None,
        ts_event: int,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None: ...
    def write_replay_summary(self, summary: Any) -> None: ...
    def write_batch(self, data: Sequence[Any]) -> None: ...
    def flush(self) -> None: ...


class PredictionRecord(TypedDict):
    """
    Typed view over a model prediction row.
    """

    model_id: str
    ts_event: int
    prediction: float
    confidence: float


class SignalRecord(TypedDict):
    """
    Typed view over a strategy signal row.
    """

    strategy_id: str
    ts_event: int
    signal_type: str
    strength: float


@runtime_checkable
class DataStoreFacadeProtocol(Protocol):
    """
    Minimal facade for actor-attached data store.

    Only the methods exercised by actors are included to keep the protocol narrow.

    """

    def read_range(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object: ...

    def flush(self) -> None: ...

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> Mapping[str, float] | None: ...

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = ...,
    ) -> PredictionRecord | None: ...

    def write_ingestion(
        self,
        dataset_id: str,
        records: object,
        source: str,
        run_id: str,
        instrument_id: str | None = ...,
    ) -> object: ...

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = ...,
    ) -> SignalRecord | None: ...

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = ...,
        net_income: float | None = ...,
        operating_income: float | None = ...,
        shares_outstanding: int | None = ...,
        filing_type: str | None = ...,
        fiscal_year: int | None = ...,
        fiscal_quarter: int | None = ...,
        source: str = ...,
        run_id: str | None = ...,
    ) -> DataEvent: ...

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = ...,
        num_analysts: int | None = ...,
        source: str = ...,
        run_id: str | None = ...,
    ) -> DataEvent: ...

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = ...,
        start_date: str | None = ...,
        end_date: str | None = ...,
    ) -> list[dict[str, Any]]: ...

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None: ...


class CircuitBreakerProtocol(Protocol):
    """
    Minimal protocol for circuit breaker integration in stores.

    Stores use this protocol to gate potentially unstable operations and to record
    success/failure outcomes without importing actor modules. This keeps typing explicit
    and avoids concrete coupling.

    """

    def can_execute(self) -> bool: ...

    def record_success(self) -> None: ...

    def record_failure(self) -> None: ...


# ============================================================================
# Strict Service Dependency Protocols (Phase 1)
# ============================================================================


@runtime_checkable
class TableLike(Protocol):
    """
    Minimal table interface used by services.

    Avoid importing SQLAlchemy types in hot modules; keep the surface small.

    """

    # Column namespace (e.g., table.c.<column>) — kept loose on purpose
    c: Any

    def delete(self) -> Any: ...


@runtime_checkable
class LoggerLike(Protocol):
    """
    Logger protocol to avoid Any in services.

    Accepts "object" for msg to match stdlib logging.Logger signature.

    """

    def debug(
        self,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | BaseException
            | None
        ) = ...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Mapping[str, object] | None = ...,
    ) -> None: ...

    def info(
        self,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | BaseException
            | None
        ) = ...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Mapping[str, object] | None = ...,
    ) -> None: ...

    def warning(
        self,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | BaseException
            | None
        ) = ...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Mapping[str, object] | None = ...,
    ) -> None: ...

    def error(
        self,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | BaseException
            | None
        ) = ...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Mapping[str, object] | None = ...,
    ) -> None: ...


# Model service deps ---------------------------------------------------------


class ModelWriteDepsStrict(Protocol):
    """
    Strict dependency surface for model write service.
    """

    model_predictions_table: TableLike

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, object]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: TableLike,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str],
        dataset_id: str,
        stage: object,
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: LoggerLike,
        publish_bus: bool = True,
    ) -> None: ...


class ModelReadDepsStrict(Protocol):
    """
    Strict dependency surface for model read/stats services.
    """

    def _qualified_table(self, base: str) -> str: ...

    def _execute_read(
        self,
        sql: object,
        params: Mapping[str, object],
        *,
        columns: Sequence[str],
    ) -> object: ...

    def _fetch_one(
        self,
        sql: object,
        params: Mapping[str, object],
    ) -> tuple[object, ...] | None: ...


class ModelEventDepsStrict(Protocol):
    def _get_data_registry(self) -> RegistryProtocol | None: ...


class ModelClearDepsStrict(Protocol):
    engine: Any
    model_predictions_table: TableLike


# Strategy service deps ------------------------------------------------------


class StrategyWriteDepsStrict(Protocol):
    strategy_signals_table: TableLike
    strategy_order_events_table: TableLike
    strategy_risk_halt_events_table: TableLike
    strategy_replay_summary_table: TableLike

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, object]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: TableLike,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str],
        dataset_id: str,
        stage: object,
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: LoggerLike,
        publish_bus: bool = True,
    ) -> None: ...


class StrategyReadDepsStrict(Protocol):
    def _safe_table(self, base: str, allowed: set[str]) -> str: ...

    def _execute_read(
        self,
        sql: object,
        params: Mapping[str, object],
        *,
        columns: Sequence[str],
    ) -> object: ...

    def _fetch_one(
        self,
        sql: object,
        params: Mapping[str, object],
    ) -> tuple[object, ...] | None: ...

    def _fetch_all(self, sql: object, params: Mapping[str, object]) -> list[tuple[object, ...]]: ...


class StrategyEventDepsStrict(Protocol):
    def _get_data_registry(self) -> RegistryProtocol | None: ...


class StrategyClearDepsStrict(Protocol):
    engine: Any
    strategy_signals_table: TableLike


@runtime_checkable
class EarningsStoreProtocol(Protocol):
    """
    Protocol for earnings data store implementations.

    Provides earnings actuals (SEC EDGAR), estimates (Yahoo Finance), and calendar
    with point-in-time correctness for backtesting.
    """

    def write_actuals(
        self,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
    ) -> None:
        """
        Write actual earnings data from SEC EDGAR.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL')
        period_end : str
            Quarter end date (ISO format: 'YYYY-MM-DD')
        filing_date : str
            10-Q/10-K filing date (ISO format: 'YYYY-MM-DD')
        eps_diluted : float | None
            Diluted earnings per share
        revenue : float | None
            Total revenue in dollars
        ts_event : int
            Filing date in nanoseconds
        ts_init : int
            Record creation timestamp in nanoseconds
        eps_basic : float | None
            Basic earnings per share
        net_income : float | None
            Net income in dollars
        operating_income : float | None
            Operating income in dollars
        shares_outstanding : int | None
            Weighted average shares outstanding
        filing_type : str | None
            '10-Q' or '10-K'
        fiscal_year : int | None
            Fiscal year
        fiscal_quarter : int | None
            Fiscal quarter (1-4)
        """
        ...

    def write_estimates(
        self,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
    ) -> None:
        """
        Write consensus earnings estimates from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        estimate_date : str
            Date estimate was recorded (ISO format)
        period_end : str
            Quarter being estimated (ISO format)
        eps_consensus : float | None
            Consensus EPS estimate
        ts_event : int
            Estimate date in nanoseconds
        ts_init : int
            Record creation timestamp in nanoseconds
        revenue_consensus : float | None
            Consensus revenue estimate
        num_analysts : int | None
            Number of analysts contributing
        """
        ...

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get actual earnings for a ticker with point-in-time filtering.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str | None
            Start date filter (ISO format)
        end_date : str | None
            End date filter (ISO format)
        as_of_ts : int | None
            Point-in-time timestamp (only include filings before this time)

        Returns
        -------
        list[dict[str, Any]]
            List of actual earnings records
        """
        ...

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get consensus estimate for a specific period.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format)
        as_of_ts : int | None
            Point-in-time timestamp (get estimate as of this time)

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found
        """
        ...

    def flush(self) -> None:
        """Flush any pending writes to persistent storage."""
        ...


@runtime_checkable
class InstrumentMetadataStoreProtocol(Protocol):
    """
    Protocol for instrument metadata store implementations.

    Provides temporal instrument metadata for factor-based portfolio construction,
    supporting dynamic assignment of instruments to duration buckets, issuer types,
    and liquidity tiers.
    """

    def write_metadata(
        self,
        instrument_id: str,
        duration_bucket: int,
        issuer_type: int,
        liquidity_tier: int,
        ts_event: int,
        ts_init: int,
        region: str | None = None,
        sector: str | None = None,
        rating: str | None = None,
        valid_from_ns: int | None = None,
        valid_until_ns: int | None = None,
    ) -> None:
        """
        Write instrument metadata to the store.

        Parameters
        ----------
        instrument_id : str
            Nautilus InstrumentId (e.g., "US10Y.BOND", "AAPL.NASDAQ")
        duration_bucket : int
            Duration classification: 0=Short (0-2Y), 1=Medium (2-7Y), 2=Long (7Y+)
        issuer_type : int
            Issuer classification: 0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY
        liquidity_tier : int
            Liquidity classification: 1=High, 2=Medium, 3=Low
        ts_event : int
            Event timestamp in nanoseconds
        ts_init : int
            Initialization timestamp in nanoseconds
        region : str | None
            Geographic region (e.g., 'US', 'EU', 'ASIA')
        sector : str | None
            Market sector (e.g., 'TREASURY', 'AGENCY', 'CORPORATE')
        rating : str | None
            Credit rating if applicable
        valid_from_ns : int | None
            Start of validity period (defaults to ts_event)
        valid_until_ns : int | None
            End of validity period (None = currently valid)
        """
        ...

    def get_metadata(
        self,
        instrument_id: str,
        ts_event: int | None = None,
    ) -> Mapping[str, Any] | None:
        """
        Get metadata for an instrument at a specific point in time.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int | None
            Query timestamp in nanoseconds (None = get current/latest)

        Returns
        -------
        Mapping[str, Any] | None
            Metadata dictionary or None if not found
        """
        ...

    def get_instruments_by_factors(
        self,
        duration_bucket: int | None = None,
        issuer_type: int | None = None,
        liquidity_tier: int | None = None,
        ts_event: int | None = None,
    ) -> list[str]:
        """
        Get instruments matching factor criteria.

        Parameters
        ----------
        duration_bucket : int | None
            Filter by duration bucket
        issuer_type : int | None
            Filter by issuer type
        liquidity_tier : int | None
            Filter by liquidity tier
        ts_event : int | None
            Query timestamp (None = current)

        Returns
        -------
        list[str]
            List of matching instrument IDs
        """
        ...

    def flush(self) -> None:
        """Flush any pending writes to persistent storage."""
        ...


__all__ = [
    "BaseStoreProtocol",
    "CircuitBreakerProtocol",
    "CoverageProviderProtocol",
    "DataStoreFacadeProtocol",
    "EarningsStoreProtocol",
    "FeatureStoreProtocol",
    "FeatureStoreStrictProtocol",
    "InstrumentMetadataStoreProtocol",
    "LoggerLike",
    "MarketDataWriterProtocol",
    "ModelClearDepsStrict",
    "ModelEventDepsStrict",
    "ModelReadDepsStrict",
    "ModelStoreProtocol",
    "ModelStoreStrictProtocol",
    "ModelWriteDepsStrict",
    "PredictionRecord",
    "SignalRecord",
    "StrategyClearDepsStrict",
    "StrategyEventDepsStrict",
    "StrategyReadDepsStrict",
    "StrategyStoreProtocol",
    "StrategyStoreStrictProtocol",
    "StrategyWriteDepsStrict",
    "TableLike",
]
