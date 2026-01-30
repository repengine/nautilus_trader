"""
Strict protocol adapters for legacy stores.

These thin wrappers present the strict protocol surfaces while delegating to the
existing store implementations. They avoid widening public types at actor
boundaries and enable incremental migration without touching hot paths.

Conversions are intentionally minimized. Where the underlying store accepts a
`dict` but the protocol uses `Mapping`, we forward the mapping directly to avoid
allocations on the hot path. Python call sites accept any mapping for JSON
serialization and persistence.

"""

from __future__ import annotations

import time
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from ml.stores.data_store import DataStore
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import FeatureStoreStrictProtocol
from ml.stores.protocols import ModelStoreStrictProtocol
from ml.stores.protocols import StrategyStoreStrictProtocol


class FeatureStoreStrictAdapter(FeatureStoreStrictProtocol):
    def __init__(self, store: object) -> None:
        self._store = store

    @property
    def connection_string(self) -> str | None:
        """
        Expose underlying store connection string when available.
        """
        return getattr(self._store, "connection_string", None)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        # Delegate attribute access for non-protocol conveniences used in tests
        return getattr(self._store, name)

    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: Mapping[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        # Delegate without copying; underlying store tolerates Mapping
        getattr(self._store, "write_features")(
            feature_set_id=feature_set_id,
            instrument_id=instrument_id,
            features=features,
            ts_event=ts_event,
            ts_init=ts_init,
        )

    def flush(self) -> None:
        getattr(self._store, "flush")()


class ModelStoreStrictAdapter(ModelStoreStrictProtocol):
    def __init__(self, store: object) -> None:
        self._store = store

    @property
    def connection_string(self) -> str | None:
        return getattr(self._store, "connection_string", None)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._store, name)

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
    ) -> None:
        # Delegate without copying to avoid hot-path allocations
        getattr(self._store, "write_prediction")(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            features=features,
            inference_time_ms=inference_time_ms,
            ts_event=ts_event,
            is_live=is_live,
        )

    def write_batch(self, data: Sequence[Any], emit_events: bool = True) -> None:
        # Underlying store expects list; pass through sequence reference when already a list
        getattr(self._store, "write_batch")(list(data), emit_events=emit_events)

    def flush(self) -> None:
        getattr(self._store, "flush")()


class StrategyStoreStrictAdapter(StrategyStoreStrictProtocol):
    def __init__(self, store: object) -> None:
        self._store = store

    @property
    def connection_string(self) -> str | None:
        return getattr(self._store, "connection_string", None)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._store, name)

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
    ) -> None:
        # Delegate directly; underlying store tolerates Mapping
        getattr(self._store, "write_signal")(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=strength,
            model_predictions=model_predictions,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            ts_event=ts_event,
            is_live=is_live,
            run_id=run_id,
        )

    def write_order_event(
        self,
        event: object,
        *,
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None:
        getattr(self._store, "write_order_event")(event, is_live=is_live, run_id=run_id)

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
    ) -> None:
        writer = getattr(self._store, "write_risk_halt_event", None)
        if callable(writer):
            writer(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                event_type=event_type,
                reason=reason,
                detail=detail,
                ts_event=ts_event,
                is_live=is_live,
                run_id=run_id,
            )

    def write_replay_summary(self, summary: Any) -> None:
        writer = getattr(self._store, "write_replay_summary", None)
        if callable(writer):
            writer(summary)

    def write_batch(self, data: Sequence[Any]) -> None:
        getattr(self._store, "write_batch")(list(data))

    def flush(self) -> None:
        getattr(self._store, "flush")()

    def write_signals(self, data: Sequence[Any]) -> None:
        """
        Write a batch of strategy signals if the underlying store exposes the helper.
        """
        writer = getattr(self._store, "write_signals", None)
        if callable(writer):
            writer(list(data))
            return
        self.write_batch(data)


class DataStoreEarningsAdapter(EarningsStoreProtocol):
    """Adapter exposing DataStore earnings methods via the historical protocol."""

    def __init__(self, store: DataStore, *, default_limit: int = 1000) -> None:
        self._store = store
        self._default_limit = default_limit

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
        self._store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=revenue,
            ts_event=ts_event,
            ts_init=ts_init,
            eps_basic=eps_basic,
            net_income=net_income,
            operating_income=operating_income,
            shares_outstanding=shares_outstanding,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )

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
        self._store.write_earnings_estimate(
            ticker=ticker,
            estimate_date=estimate_date,
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=ts_event,
            ts_init=ts_init,
            revenue_consensus=revenue_consensus,
            num_analysts=num_analysts,
        )

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        ts_query = as_of_ts if as_of_ts is not None else int(time.time_ns())
        return self._store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_query,
            limit=self._default_limit,
            start_date=start_date,
            end_date=end_date,
        )

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        ts_query = as_of_ts if as_of_ts is not None else int(time.time_ns())
        return self._store.get_earnings_estimate_at_or_before(
            ticker=ticker,
            period_end=period_end,
            ts_event=ts_query,
        )

    def flush(self) -> None:
        # DataStore is synchronous; nothing to flush for earnings-specific writes.
        return None


__all__ = [
    "DataStoreEarningsAdapter",
    "FeatureStoreStrictAdapter",
    "ModelStoreStrictAdapter",
    "StrategyStoreStrictAdapter",
]
