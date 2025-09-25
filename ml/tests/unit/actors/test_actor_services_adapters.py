from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ml.actors.actor_services import ActorServices, init_actor_services

from ml.stores.protocols import (
    DataStoreFacadeProtocol,
    FeatureStoreStrictProtocol,
    ModelStoreStrictProtocol,
    StrategyStoreStrictProtocol,
    PredictionRecord,
    SignalRecord,
)


class _DummyFeature(FeatureStoreStrictProtocol):  # runtime structural
    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: object,
        ts_event: int,
        ts_init: int,
    ) -> None:
        return None

    def flush(self) -> None:  # noqa: D401
        return None


class _DummyModel(ModelStoreStrictProtocol):  # runtime structural
    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: object,
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        return None

    def write_batch(self, data: object, emit_events: bool = True) -> None:
        return None

    def flush(self) -> None:  # noqa: D401
        return None


class _DummyStrategy(StrategyStoreStrictProtocol):  # runtime structural
    def write_signal(
        self,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: object,
        risk_metrics: object,
        execution_params: object,
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        return None

    def write_batch(self, data: object) -> None:
        return None

    def flush(self) -> None:  # noqa: D401
        return None


class _DummyDataStore(DataStoreFacadeProtocol):
    def read_range(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object:
        return []

    def flush(self) -> None:  # noqa: D401
        return None

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> Mapping[str, float] | None:
        return None

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None:
        return None

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> SignalRecord | None:
        return None


@dataclass
class _InitResult:
    feature_store: object
    model_store: object
    strategy_store: object
    data_store: DataStoreFacadeProtocol = field(default_factory=_DummyDataStore)
    feature_registry: object = object()
    model_registry: object = object()
    strategy_registry: object = object()
    data_registry: object = object()


def test_init_actor_services_skips_adapters_when_protocols_conform(monkeypatch: Any) -> None:
    def _fake_init(_config: Any) -> _InitResult:
        return _InitResult(_DummyFeature(), _DummyModel(), _DummyStrategy(), _DummyDataStore())

    monkeypatch.setattr("ml.core.integration.init_ml_stores_and_registries", _fake_init)

    services = init_actor_services(config={})
    assert isinstance(services, ActorServices)
    assert isinstance(services.feature_store, FeatureStoreStrictProtocol)
    assert isinstance(services.model_store, ModelStoreStrictProtocol)
    assert isinstance(services.strategy_store, StrategyStoreStrictProtocol)
    assert isinstance(services.data_store, DataStoreFacadeProtocol)
