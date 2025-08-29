"""
Protocols for ML store interfaces.

These provide structural contracts for store implementations so mypy can detect
interface drift across implementations and tests.
"""

from __future__ import annotations

from typing import Any, Protocol


class BaseStoreProtocol(Protocol):
    def write_batch(self, data: list[Any]) -> None: ...
    def read_range(self, start_ns: int, end_ns: int, instrument_id: str | None = None) -> Any: ...
    def flush(self) -> None: ...
    def get_latest(self, instrument_id: str, limit: int = 1) -> Any: ...
    def get_statistics(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]: ...


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
    def read_predictions(self, model_id: str, instrument_id: str, start_ns: int, end_ns: int) -> Any: ...
    def get_model_performance(self, model_id: str, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]: ...


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
    def read_signals(self, strategy_id: str, instrument_id: str, start_ns: int, end_ns: int) -> Any: ...
    def get_strategy_performance(self, strategy_id: str, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]: ...
    def get_signal_distribution(self, strategy_id: str | None = None, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, int]: ...
    def flush(self) -> None: ...
