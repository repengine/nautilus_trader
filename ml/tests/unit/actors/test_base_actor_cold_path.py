from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.actor_services import ActorServices
from ml.actors.base import BaseMLInferenceActor
from ml.config.actors import OptimizationConfig
from ml.config.base import CircuitBreakerConfig
from ml.config.base import MLActorConfig
from ml.config.constants import TimeConstants
from ml.stores.base import DummyStore


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


class _ConcreteActor(BaseMLInferenceActor):
    def _load_model(self) -> None:
        return None

    def _initialize_features(self) -> None:
        return None

    def _compute_features(self, bar: object) -> npt.NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        return 0.1, 0.2


def _make_services(
    *,
    feature_store: object,
    model_store: object,
    strategy_store: object,
    data_store: object,
) -> ActorServices:
    registries = SimpleNamespace(
        get_feature_manifest=lambda _feature_set_id: None,
        get_model=lambda _model_id: None,
    )
    return ActorServices(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_store=data_store,
        feature_registry=registries,
        model_registry=registries,
        strategy_registry=registries,
        data_registry=registries,
    )


class _BreakerTarget:
    def __init__(self, fail: bool = False) -> None:
        super().__setattr__("_fail", fail)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_circuit_breaker" and self._fail:
            raise RuntimeError("breaker propagation failed")
        super().__setattr__(name, value)


@dataclass(slots=True)
class _Adapter:
    _store: object | None
    _circuit_breaker: object | None = None


@dataclass(slots=True)
class _Manifest:
    model_id: str
    version: str
    architecture: str
    role: SimpleNamespace
    data_requirements: SimpleNamespace
    feature_schema: dict[str, str]
    feature_schema_hash: str
    parent_id: str | None
    performance_metrics: dict[str, float]
    deployment_constraints: dict[str, float]
    training_config: dict[str, object]
    decision_policy: str | None
    decision_config: dict[str, object]
    output_schema: dict[str, object] | None
    calibration: dict[str, object] | None
    artifact_sha256_digest: str | None


@dataclass(slots=True)
class _ModelInfo:
    manifest: _Manifest


class _Registry:
    def __init__(self, *, model_info: _ModelInfo | None, model: object) -> None:
        self._model_info = model_info
        self._model = model
        self.loaded_ids: list[str] = []

    def get_model(self, model_id: str) -> _ModelInfo | None:
        return self._model_info

    def load_model(self, model_id: str) -> object:
        self.loaded_ids.append(model_id)
        return self._model


class _PersistenceWorker:
    def __init__(self, *, enqueued: bool) -> None:
        self.enqueued = enqueued
        self.calls: list[dict[str, object]] = []

    def enqueue_prediction(
        self,
        *,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
    ) -> bool:
        self.calls.append(
            {
                "model_id": model_id,
                "instrument_id": instrument_id,
                "prediction": prediction,
                "confidence": confidence,
                "features": features,
                "inference_time_ms": inference_time_ms,
                "ts_event": ts_event,
            },
        )
        return self.enqueued


class _ModelStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write_prediction(
        self,
        *,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool,
    ) -> None:
        self.calls.append(
            {
                "model_id": model_id,
                "instrument_id": instrument_id,
                "prediction": prediction,
                "confidence": confidence,
                "features": features,
                "inference_time_ms": inference_time_ms,
                "ts_event": ts_event,
                "is_live": is_live,
            },
        )


def _make_actor(
    *,
    base_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: object,
) -> _ConcreteActor:
    services = _make_services(
        feature_store=DummyStore(),
        model_store=DummyStore(),
        strategy_store=DummyStore(),
        data_store=DummyStore(),
    )
    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        lambda _config: services,
    )
    config_kwargs = {"model_path": str(dummy_onnx_model)}
    config_kwargs.update(overrides)
    config = msgspec.structs.replace(base_config, **config_kwargs)
    return _ConcreteActor(config)


def test_init_stores_and_registries_attaches_services_and_breaker(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_workers: list[object] = []

    class _StubWorker:
        def __init__(
            self,
            *,
            feature_store: object,
            model_store: object,
            queue_maxsize: int,
            flush_interval_seconds: float,
            batch_size: int,
        ) -> None:
            self.feature_store = feature_store
            self.model_store = model_store
            self.queue_maxsize = queue_maxsize
            self.flush_interval_seconds = flush_interval_seconds
            self.batch_size = batch_size
            created_workers.append(self)

    monkeypatch.setattr(
        "ml.observability.ml_async_persistence.MLPersistenceWorker",
        _StubWorker,
    )

    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        enable_async_persistence=True,
    )

    raw_ok = _BreakerTarget()
    raw_fail = _BreakerTarget()
    feature_store = _Adapter(raw_ok)
    model_store = _Adapter(None)
    strategy_store = _Adapter(raw_fail)
    data_store = _Adapter(raw_fail)

    runtime_services = _make_services(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_store=data_store,
    )
    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        lambda _config: runtime_services,
    )

    breaker = object()
    actor._circuit_breaker = breaker

    actor._init_stores_and_registries()

    assert actor._BaseMLInferenceActor__feature_store_instance is feature_store
    assert actor._BaseMLInferenceActor__model_store_instance is model_store
    assert actor._BaseMLInferenceActor__strategy_store_instance is strategy_store
    assert actor._BaseMLInferenceActor__data_store_instance is data_store

    assert isinstance(actor._BaseMLInferenceActor__persistence_worker_instance, _StubWorker)
    worker = actor._BaseMLInferenceActor__persistence_worker_instance
    assert worker is created_workers[-1]
    assert worker.feature_store is feature_store
    assert worker.model_store is model_store

    assert raw_ok._circuit_breaker is breaker
    assert getattr(model_store, "_circuit_breaker", None) is breaker


@pytest.mark.parametrize(
    ("metadata", "model_version", "expected"),
    [
        ({"model_id": "model-1"}, "v2", "model-1"),
        ({"training_metadata": {"model_id": "train-1"}}, "v3", "train-1"),
        ({}, "abcdefgh1234", "model_abcdefgh"),
    ],
)
def test_determine_model_id_uses_metadata_or_fallback(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, object],
    model_version: str,
    expected: str,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        model_path="model.onnx",
        enable_async_persistence=False,
    )
    actor._model_metadata = metadata
    actor._model_version = model_version

    actor._determine_model_id()

    assert actor._model_id == expected


def test_check_model_updates_triggers_reload_and_state_restore(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        enable_async_persistence=False,
    )

    calls: list[str] = []
    actor._model_version = "v1"
    actor._model_loader = SimpleNamespace(get_model_version=lambda _path: "v2")

    actor._backup_indicator_state = lambda: calls.append("backup")

    def _reload() -> None:
        calls.append("reload")
        actor._model_version = "v2"

    actor._reload_model = _reload
    actor._restore_indicator_state = lambda: calls.append("restore")

    actor._check_model_updates(object())

    assert calls == ["backup", "reload", "restore"]
    assert actor._last_model_check > 0.0


def test_check_model_updates_handles_errors(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        enable_async_persistence=False,
    )

    def _raise(_path: str) -> str:
        raise RuntimeError("version lookup failed")

    actor._model_loader = SimpleNamespace(get_model_version=_raise)
    actor._model_version = "v1"

    actor._check_model_updates(object())

    assert actor._last_model_check == 0.0


def test_try_load_from_registry_when_model_found_populates_metadata(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        max_inference_latency_ms=10.0,
    )

    manifest = _Manifest(
        model_id="registry-model",
        version="v1",
        architecture="onnx",
        role=SimpleNamespace(value="signal"),
        data_requirements=SimpleNamespace(value="bars"),
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="hash123",
        parent_id=None,
        performance_metrics={"accuracy": 0.9},
        deployment_constraints={"max_latency_ms": 1.0},
        training_config={"seed": 7},
        decision_policy="threshold",
        decision_config={"threshold": 0.5},
        output_schema={"prediction": "float32"},
        calibration=None,
        artifact_sha256_digest="abc123",
    )
    model_info = _ModelInfo(manifest=manifest)
    model_obj = object()
    registry = _Registry(model_info=model_info, model=model_obj)
    actor._BaseMLInferenceActor__model_registry_instance = registry

    loaded = actor._try_load_from_registry()

    assert loaded is True
    assert actor._model is model_obj
    assert registry.loaded_ids == [base_ml_config.model_id]
    assert actor._model_metadata["model_id"] == "registry-model"
    assert actor._model_metadata["version"] == "v1"
    assert actor._model_metadata["feature_schema_hash"] == "hash123"
    assert actor._manifest_feature_names == ["f1", "f2"]
    assert actor._manifest_feature_dtypes == ["float32", "float32"]


def test_try_load_from_registry_when_missing_model_returns_false(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )

    registry = _Registry(model_info=None, model=object())
    actor._BaseMLInferenceActor__model_registry_instance = registry

    loaded = actor._try_load_from_registry()

    assert loaded is False
    assert registry.loaded_ids == []


def test_load_model_with_metadata_when_fallback_warms_model(
    base_signal_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_signal_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        model_path="model.onnx",
        optimization_config=OptimizationConfig(enable_model_warm_up=True),
    )

    dummy_model = object()
    metadata = {
        "version": "v1",
        "model_id": "meta-model",
        "feature_schema": {"f1": "float32", "f2": "float32"},
        "size_bytes": 128,
        "type": "onnx",
    }
    calls: dict[str, object] = {}

    def _load(path: str) -> tuple[object, dict[str, object]]:
        calls["path"] = path
        return dummy_model, metadata

    actor._model_loader = SimpleNamespace(load_model=_load)
    monkeypatch.setattr(actor, "_try_load_from_registry", lambda: False)

    warm_calls: list[tuple[object, bool, int]] = []

    def _warm(model: object, enable: bool, input_dim: int) -> None:
        warm_calls.append((model, enable, input_dim))

    monkeypatch.setattr("ml.actors.model_loader_utils.maybe_warm_up_model", _warm)

    actor._load_model_with_metadata()

    assert calls["path"] == "model.onnx"
    assert actor._model is dummy_model
    assert actor._model_id == "meta-model"
    assert actor._model_version == "v1"
    assert warm_calls == [(dummy_model, True, 2)]
    assert actor._decision_metadata_payload is not None


def test_load_model_with_metadata_when_non_onnx_and_not_test_env_raises(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        model_path="model.joblib",
    )

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("ML_TEST_ALLOW_NON_ONNX", "0")
    monkeypatch.setenv("ML_ALLOW_NON_ONNX_IN_TESTS", "0")
    monkeypatch.setattr(actor, "_try_load_from_registry", lambda: False)

    with pytest.raises(ValueError, match="Non-ONNX model format disallowed"):
        actor._load_model_with_metadata()


def test_reload_model_when_success_updates_metadata_and_health(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )

    new_model = object()
    actor._model_version = "v1"
    actor._model_loader = SimpleNamespace(
        load_model=lambda _path: (new_model, {"version": "v2"}),
    )
    refreshed: list[str] = []
    actor._refresh_decision_metadata_payload = lambda: refreshed.append("ok")

    actor._reload_model()

    assert actor._model is new_model
    assert actor._model_version == "v2"
    assert refreshed == ["ok"]
    assert actor._health_monitor is not None
    assert actor._health_monitor.model_loaded is True


def test_reload_model_when_load_fails_marks_unhealthy(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )

    def _raise(_path: str) -> tuple[object, dict[str, object]]:
        raise RuntimeError("load failed")

    actor._model_loader = SimpleNamespace(load_model=_raise)
    assert actor._health_monitor is not None
    actor._health_monitor.set_model_loaded(True)

    with pytest.raises(RuntimeError, match="load failed"):
        actor._reload_model()

    assert actor._health_monitor.model_loaded is False


def test_refresh_decision_metadata_payload_builds_surface_metadata(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )
    actor._model_metadata = {"decision_policy": "threshold"}
    actor._model_id = "meta-model"
    actor._model_version = "v1"

    actor._refresh_decision_metadata_payload()

    assert actor._decision_metadata_payload is not None
    assert actor._signal_metadata_extra is not None
    assert actor._signal_metadata_extra["neutral_band"] == base_ml_config.prediction_neutral_band


def test_schedule_model_checks_when_enabled_registers_timer(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = msgspec.structs.replace(
        base_ml_config,
        enable_hot_reload=True,
        model_check_interval=123,
    )
    actor = _make_actor(
        base_config=config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )

    calls: list[dict[str, object]] = []

    class _Clock:
        def timestamp_ns(self) -> int:
            return 1000

        def set_timer_ns(
            self,
            *,
            name: str,
            interval_ns: int,
            start_time_ns: int,
            stop_time_ns: int,
            callback: object,
        ) -> None:
            calls.append(
                {
                    "name": name,
                    "interval_ns": interval_ns,
                    "start_time_ns": start_time_ns,
                    "stop_time_ns": stop_time_ns,
                    "callback": callback,
                },
            )

    clock = _Clock()
    monkeypatch.setattr(type(actor), "clock", property(lambda _self: clock))

    actor._schedule_model_checks()

    assert len(calls) == 1
    call = calls[0]
    interval = 123 * TimeConstants.NS_IN_SECOND
    assert call["name"] == "model_version_check"
    assert call["interval_ns"] == interval
    assert call["start_time_ns"] == 1000 + interval
    assert call["stop_time_ns"] == 0
    assert call["callback"] == actor._check_model_updates


def test_get_health_status_includes_monitor_and_breaker(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = msgspec.structs.replace(
        base_ml_config,
        circuit_breaker_config=CircuitBreakerConfig(),
    )
    actor = _make_actor(
        base_config=config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )
    actor._model_version = "v2"
    actor._prediction_count = 2
    actor._total_inference_time = 10.0
    actor._bars_processed = 4
    actor._total_feature_time = 8.0
    actor._is_warmed_up = True

    status = actor.get_health_status()

    assert status["model_version"] == "v2"
    assert status["predictions_made"] == 2
    assert status["avg_inference_time_ms"] == 5.0
    assert status["avg_feature_time_ms"] == 2.0
    assert "status" in status
    assert "circuit_breaker" in status


def test_reset_health_status_replaces_monitor(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )
    assert actor._health_monitor is not None
    actor._health_monitor.update_prediction_failure()
    old_monitor = actor._health_monitor

    actor.reset_health_status()

    assert actor._health_monitor is not old_monitor
    assert actor._health_monitor.total_predictions == 0


def test_persist_prediction_async_enqueues_with_worker(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )
    worker = _PersistenceWorker(enqueued=True)
    actor._BaseMLInferenceActor__persistence_worker_instance = worker
    actor._model_id = "model-1"

    result = actor._persist_prediction_async(
        instrument_id="EURUSD",
        prediction=0.4,
        confidence=0.6,
        features={"f1": 1.0},
        inference_time_ms=1.2,
        ts_event=123,
        is_live=False,
    )

    assert result is True
    assert worker.calls == [
        {
            "model_id": "model-1",
            "instrument_id": "EURUSD",
            "prediction": 0.4,
            "confidence": 0.6,
            "features": {"f1": 1.0},
            "inference_time_ms": 1.2,
            "ts_event": 123,
        },
    ]


def test_persist_prediction_async_records_drop_when_queue_full(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
    )
    worker = _PersistenceWorker(enqueued=False)
    actor._BaseMLInferenceActor__persistence_worker_instance = worker
    actor._model_id = "model-2"

    result = actor._persist_prediction_async(
        instrument_id="GBPUSD",
        prediction=0.2,
        confidence=0.3,
        features={"f2": 2.0},
        inference_time_ms=2.4,
        ts_event=456,
        is_live=None,
    )

    assert result is False
    assert worker.calls


def test_persist_prediction_async_sync_fallback_writes_when_allowed(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        allow_sync_persistence_fallback=True,
        enable_async_persistence=False,
    )
    store = _ModelStore()
    actor._BaseMLInferenceActor__model_store_instance = store
    actor._model_id = "model-3"

    result = actor._persist_prediction_async(
        instrument_id="USDJPY",
        prediction=0.7,
        confidence=0.8,
        features={"f3": 3.0},
        inference_time_ms=3.0,
        ts_event=789,
        is_live=True,
    )

    assert result is True
    assert store.calls == [
        {
            "model_id": "model-3",
            "instrument_id": "USDJPY",
            "prediction": 0.7,
            "confidence": 0.8,
            "features": {"f3": 3.0},
            "inference_time_ms": 3.0,
            "ts_event": 789,
            "is_live": True,
        },
    ]


def test_persist_prediction_async_sync_fallback_disabled_drops(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _make_actor(
        base_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        allow_sync_persistence_fallback=False,
        enable_async_persistence=False,
    )
    actor._model_id = "model-4"

    result = actor._persist_prediction_async(
        instrument_id="AUDUSD",
        prediction=0.1,
        confidence=0.2,
        features={"f4": 4.0},
        inference_time_ms=0.4,
        ts_event=321,
        is_live=False,
    )

    assert result is False
    assert actor._sync_prediction_fallback_disabled_logged is True
