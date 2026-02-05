from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.actors.actor_services import ActorServices
from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig
from ml.stores.base import DummyStore


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@dataclass(slots=True)
class _ModelStore:
    calls: list[dict[str, object]]

    def write_prediction(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _TestActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)
        self._next_features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self._prediction = 0.6
        self._confidence = 0.7

    def _load_model(self) -> None:
        return None

    def _initialize_features(self) -> None:
        return None

    def _compute_features(self, bar: object) -> npt.NDArray[np.float32]:
        return self._next_features

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        return self._prediction, self._confidence


def _make_services() -> ActorServices:
    dummy = DummyStore()
    registries = SimpleNamespace(
        get_feature_manifest=lambda _feature_set_id: None,
        get_model=lambda _model_id: None,
    )
    return ActorServices(
        feature_store=dummy,
        model_store=dummy,
        strategy_store=dummy,
        data_store=dummy,
        feature_registry=registries,
        model_registry=registries,
        strategy_registry=registries,
        data_registry=registries,
    )


def _make_actor(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> _TestActor:
    services = _make_services()
    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        lambda _config: services,
    )

    config = MLActorConfig(
        component_id="base-actor",
        model_path="model.onnx",
        model_id="model-v1",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        enable_async_persistence=False,
        enable_health_monitoring=False,
        publish_signals=False,
        **overrides,
    )
    return _TestActor(config)


def _make_bar() -> SimpleNamespace:
    return SimpleNamespace(
        bar_type=SimpleNamespace(instrument_id="EUR/USD.SIM"),
        ts_event=1,
        ts_init=1,
    )


def test_on_bar_skips_when_circuit_breaker_open(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _make_actor(monkeypatch, warm_up_period=1)
    actor._circuit_breaker = SimpleNamespace(can_execute=lambda: False)
    actor._features_component.compute_features = lambda _bar: pytest.fail("compute called")

    actor.on_bar(_make_bar())

    assert actor._bars_processed == 0


def test_on_bar_warmup_triggers_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _make_actor(monkeypatch, warm_up_period=2)
    calls: list[tuple[object, np.ndarray]] = []
    actor._generate_prediction_protected = lambda bar, features: calls.append((bar, features))

    actor.on_bar(_make_bar())
    assert calls == []
    assert actor._is_warmed_up is False

    actor.on_bar(_make_bar())
    assert len(calls) == 1
    assert actor._is_warmed_up is True
    assert len(actor._feature_window) == 2


def test_persist_prediction_queue_full_records_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _make_actor(monkeypatch)

    class _Worker:
        def enqueue_prediction(self, **_kwargs: object) -> bool:
            return False

    actor._BaseMLInferenceActor__persistence_worker_instance = _Worker()
    drops: list[tuple[str, str]] = []

    def _record(*, kind: str, reason: str) -> None:
        drops.append((kind, reason))

    actor._record_persistence_drop = _record

    ok = actor._persist_prediction_async(
        instrument_id="EUR/USD.SIM",
        prediction=0.1,
        confidence=0.2,
        features={"f1": 1.0},
        inference_time_ms=1.0,
        ts_event=1,
    )

    assert ok is False
    assert drops == [("prediction", "queue_full")]


def test_persist_prediction_sync_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _make_actor(monkeypatch, allow_sync_persistence_fallback=False)
    actor._BaseMLInferenceActor__persistence_worker_instance = None
    drops: list[tuple[str, str]] = []

    def _record(*, kind: str, reason: str) -> None:
        drops.append((kind, reason))

    actor._record_persistence_drop = _record

    ok1 = actor._persist_prediction_async(
        instrument_id="EUR/USD.SIM",
        prediction=0.1,
        confidence=0.2,
        features={"f1": 1.0},
        inference_time_ms=1.0,
        ts_event=1,
    )
    ok2 = actor._persist_prediction_async(
        instrument_id="EUR/USD.SIM",
        prediction=0.2,
        confidence=0.3,
        features={"f1": 2.0},
        inference_time_ms=1.0,
        ts_event=2,
    )

    assert ok1 is False
    assert ok2 is False
    assert actor._sync_prediction_fallback_disabled_logged is True
    assert drops == [("prediction", "sync_disabled"), ("prediction", "sync_disabled")]


def test_persist_prediction_sync_fallback_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _make_actor(monkeypatch, allow_sync_persistence_fallback=True)
    actor._BaseMLInferenceActor__persistence_worker_instance = None

    model_store = _ModelStore(calls=[])
    actor._BaseMLInferenceActor__model_store_instance = model_store

    ok = actor._persist_prediction_async(
        instrument_id="EUR/USD.SIM",
        prediction=0.1,
        confidence=0.2,
        features={"f1": 1.0},
        inference_time_ms=1.0,
        ts_event=1,
    )

    assert ok is True
    assert len(model_store.calls) == 1
