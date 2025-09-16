"""
Protocols for ML store interfaces.

These provide structural contracts for store implementations so mypy can detect
interface drift across implementations and tests.

"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

import pandas as pd


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
    ) -> None: ...
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
    ) -> None: ...
    def write_batch(self, data: Sequence[Any]) -> None: ...
    def flush(self) -> None: ...


class DataStoreFacadeProtocol(Protocol):
    """
    Minimal facade for actor-attached data store.

    Only the methods exercised by actors are included to keep the protocol narrow.

    """

    def flush(self) -> None: ...
