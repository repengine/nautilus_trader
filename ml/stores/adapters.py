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

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

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
        )

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


__all__ = [
    "FeatureStoreStrictAdapter",
    "ModelStoreStrictAdapter",
    "StrategyStoreStrictAdapter",
]
