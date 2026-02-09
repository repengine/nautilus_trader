from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import Mock

import msgspec
import numpy as np
import numpy.typing as npt
import pytest

import ml.actors.base as base_module
from ml.actors.actor_services import ActorServices
from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import HealthStatus
from ml.config.base import MLActorConfig
from ml.config.policy import ActorRemediationPolicyConfig
from ml.config.policy import InferenceTimeoutAction
from ml.config.policy import MLFailureAction
from ml.stores.base import DummyStore
from ml.tests.utils.stubs import make_stub_bar


pytestmark = [
    pytest.mark.unit,
    pytest.mark.runtime_correctness,
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


class _MetricStub:
    def __init__(self) -> None:
        self.labels_calls: list[dict[str, str]] = []
        self.observe_calls: list[float] = []
        self.inc_calls = 0

    def labels(self, **kwargs: str) -> _MetricStub:
        self.labels_calls.append({k: str(v) for k, v in kwargs.items()})
        return self

    def observe(self, value: float) -> None:
        self.observe_calls.append(float(value))

    def inc(self) -> None:
        self.inc_calls += 1


class _RiskHaltStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write_risk_halt_event(self, **kwargs: Any) -> None:
        self.calls.append(dict(kwargs))


class _GuardActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)
        self._prediction_response: tuple[float, float] = (0.8, 0.9)
        self._predict_error: Exception | None = None

    def _load_model(self) -> None:
        return None

    def _initialize_features(self) -> None:
        return None

    def _compute_features(self, _bar: object) -> npt.NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    def _predict(self, _features: npt.NDArray[np.float32]) -> tuple[float, float]:
        if self._predict_error is not None:
            raise self._predict_error
        return self._prediction_response


def _make_services(*, strategy_store: object | None = None) -> ActorServices:
    dummy = DummyStore()
    registries = SimpleNamespace(
        get_feature_manifest=lambda _feature_set_id: None,
        get_model=lambda _model_id: None,
    )
    return ActorServices(
        feature_store=cast(Any, dummy),
        model_store=cast(Any, dummy),
        strategy_store=cast(Any, strategy_store or dummy),
        data_store=cast(Any, dummy),
        feature_registry=registries,
        model_registry=registries,
        strategy_registry=registries,
        data_registry=registries,
    )


@dataclass(frozen=True, slots=True)
class _ActorHarness:
    actor: _GuardActor
    strategy_store: _RiskHaltStore | None
    deadline_metric: _MetricStub
    failure_metric: _MetricStub


def _make_actor(
    *,
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    remediation_policy: ActorRemediationPolicyConfig,
) -> _ActorHarness:
    strategy_store = _RiskHaltStore()
    services = _make_services(strategy_store=strategy_store)
    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        lambda _config: services,
    )

    config = msgspec.structs.replace(
        base_ml_config,
        model_path=str(dummy_onnx_model),
        enable_async_persistence=False,
        publish_signals=False,
        warm_up_period=1,
        max_inference_latency_ms=0.0001,
        remediation_policy=remediation_policy,
    )
    actor = _GuardActor(config)
    actor._features_component = SimpleNamespace(
        persist_features_async=Mock(),
        compute_features=Mock(return_value=np.ones(1, dtype=np.float32)),
    )
    actor._persist_prediction_async = Mock(return_value=True)
    actor._publish_signal = Mock()
    actor._inference_latency_metric = _MetricStub()
    actor._inference_count_metric = _MetricStub()
    actor._inference_confidence_metric = _MetricStub()

    deadline_metric = _MetricStub()
    failure_metric = _MetricStub()
    monkeypatch.setattr(base_module, "inference_deadline_timeouts_total", deadline_metric)
    monkeypatch.setattr(base_module, "ml_failure_actions_total", failure_metric)

    return _ActorHarness(
        actor=actor,
        strategy_store=strategy_store,
        deadline_metric=deadline_metric,
        failure_metric=failure_metric,
    )


def _make_bar(instrument_id: Any) -> Any:
    bar = make_stub_bar(instrument_id, ts_event=1, close=1.0)
    setattr(bar, "ts_init", 1)
    return bar


def test_deadline_guard_permissive_default_preserves_flow(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=False,
            inference_timeout_action=InferenceTimeoutAction.HALT,
        ),
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    harness.actor._features_component.persist_features_async.assert_called_once()
    harness.actor._persist_prediction_async.assert_called_once()
    assert harness.actor._ml_inference_halted is False
    assert harness.deadline_metric.inc_calls == 0
    assert harness.failure_metric.inc_calls == 0


@pytest.mark.parametrize(
    ("timeout_action", "expected_halt"),
    [
        (InferenceTimeoutAction.DROP, False),
        (InferenceTimeoutAction.HALT, True),
    ],
)
def test_deadline_timeout_does_not_publish_signal(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    timeout_action: InferenceTimeoutAction,
    expected_halt: bool,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=timeout_action,
        ),
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    harness.actor._features_component.persist_features_async.assert_not_called()
    harness.actor._persist_prediction_async.assert_not_called()
    harness.actor._publish_signal.assert_not_called()
    assert harness.actor._ml_inference_halted is expected_halt


def test_deadline_guard_drop_skips_persist_and_publish(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.DROP,
        ),
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    harness.actor._features_component.persist_features_async.assert_not_called()
    harness.actor._persist_prediction_async.assert_not_called()
    harness.actor._publish_signal.assert_not_called()
    assert harness.actor._ml_inference_halted is False
    assert harness.deadline_metric.inc_calls == 1
    assert harness.deadline_metric.labels_calls[0]["action"] == "drop"
    assert harness.failure_metric.inc_calls == 0


def test_deadline_guard_halt_sets_failure_state_and_transition_hook(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.HALT,
        ),
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    harness.actor._features_component.persist_features_async.assert_not_called()
    harness.actor._persist_prediction_async.assert_not_called()
    harness.actor._publish_signal.assert_not_called()
    assert harness.actor._ml_inference_halted is True
    assert harness.actor._ml_failure_reason == "inference_deadline_timeout"
    assert harness.deadline_metric.inc_calls == 1
    assert harness.deadline_metric.labels_calls[0]["action"] == "halt"
    assert harness.failure_metric.inc_calls == 1
    assert harness.failure_metric.labels_calls[0]["action"] == "halt"
    assert harness.failure_metric.labels_calls[0]["reason"] == "inference_deadline_timeout"
    assert harness.strategy_store is not None
    assert len(harness.strategy_store.calls) == 1
    assert harness.strategy_store.calls[0]["reason"] == "inference_deadline_timeout"
    assert harness.strategy_store.calls[0]["is_live"] is True

    publish_calls = harness.actor._publish_signal.call_count
    timeout_metric_calls = harness.deadline_metric.inc_calls
    failure_metric_calls = harness.failure_metric.inc_calls
    harness.actor._features_component.compute_features = Mock(
        side_effect=AssertionError("feature computation should be skipped when halted"),
    )
    harness.actor._generate_prediction_protected = Mock(
        side_effect=AssertionError("prediction path should be skipped when halted"),
    )
    harness.actor.on_bar(bar)
    assert harness.actor._ml_inference_halted is True
    assert harness.actor._ml_failure_reason == "inference_deadline_timeout"
    assert harness.actor._publish_signal.call_count == publish_calls
    assert harness.deadline_metric.inc_calls == timeout_metric_calls
    assert harness.failure_metric.inc_calls == failure_metric_calls
    assert harness.strategy_store is not None
    assert len(harness.strategy_store.calls) == 1


def test_deadline_guard_halt_marks_missing_risk_transition_when_hook_unavailable(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.HALT,
        ),
    )
    missing_store = object()
    monkeypatch.setattr(
        _GuardActor,
        "_strategy_store",
        property(lambda _self: cast(Any, missing_store)),
        raising=False,
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    assert harness.actor._ml_inference_halted is True
    assert harness.actor._ml_failure_reason == "risk_state_transition_unavailable"
    assert harness.deadline_metric.inc_calls == 1
    assert harness.failure_metric.inc_calls == 1
    assert harness.failure_metric.labels_calls[0]["reason"] == "inference_deadline_timeout"


def test_deadline_guard_halt_risk_transition_uses_replay_safe_live_flag(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.HALT,
        ),
    )
    monkeypatch.setattr(
        _GuardActor,
        "cache",
        property(lambda _self: SimpleNamespace(is_backtesting=True)),
        raising=False,
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    assert harness.strategy_store is not None
    assert len(harness.strategy_store.calls) == 1
    assert harness.strategy_store.calls[0]["is_live"] is False


def test_deadline_guard_halt_marks_missing_risk_transition_when_hook_write_fails(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RaisingRiskStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def write_risk_halt_event(self, **kwargs: Any) -> None:
            self.calls.append(dict(kwargs))
            raise RuntimeError("boom")

    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.HALT,
        ),
    )
    raising_store = _RaisingRiskStore()
    monkeypatch.setattr(
        _GuardActor,
        "_strategy_store",
        property(lambda _self: cast(Any, raising_store)),
        raising=False,
    )
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    assert len(raising_store.calls) == 1
    assert harness.actor._ml_inference_halted is True
    assert harness.actor._ml_failure_reason == "risk_state_transition_unavailable"


@pytest.mark.parametrize(
    ("failure_action", "expected_halt", "expected_status"),
    [
        (MLFailureAction.LOG_ONLY, False, None),
        (MLFailureAction.DEGRADED, False, HealthStatus.DEGRADED),
        (MLFailureAction.HALT, True, HealthStatus.UNHEALTHY),
    ],
)
def test_prediction_failure_applies_configured_failure_action(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    failure_action: MLFailureAction,
    expected_halt: bool,
    expected_status: HealthStatus | None,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=False,
            inference_timeout_action=InferenceTimeoutAction.DROP,
            ml_failure_action=failure_action,
        ),
    )
    harness.actor._predict_error = RuntimeError("boom")
    bar = _make_bar(base_ml_config.instrument_id)

    harness.actor._generate_prediction_protected(
        bar,
        np.ones(1, dtype=np.float32),
    )

    assert harness.failure_metric.inc_calls == 1
    assert harness.failure_metric.labels_calls[0]["action"] == str(failure_action.value)
    assert harness.failure_metric.labels_calls[0]["reason"] == "prediction_exception"
    assert harness.actor._ml_inference_halted is expected_halt
    if expected_status is not None:
        assert harness.actor._health_monitor is not None
        assert harness.actor._health_monitor.status == expected_status
    if failure_action == MLFailureAction.HALT:
        assert harness.strategy_store is not None
        assert len(harness.strategy_store.calls) == 1
        assert harness.strategy_store.calls[0]["reason"] == "prediction_exception"
    else:
        assert harness.strategy_store is not None
        assert not harness.strategy_store.calls


@pytest.mark.parametrize(
    "failure_action",
    [
        MLFailureAction.LOG_ONLY,
        MLFailureAction.DEGRADED,
        MLFailureAction.HALT,
    ],
)
def test_configured_failure_action_is_noop_when_actor_already_halted(
    base_ml_config: MLActorConfig,
    dummy_onnx_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    failure_action: MLFailureAction,
) -> None:
    harness = _make_actor(
        base_ml_config=base_ml_config,
        dummy_onnx_model=dummy_onnx_model,
        monkeypatch=monkeypatch,
        remediation_policy=ActorRemediationPolicyConfig(
            enable_inference_deadline_guard=True,
            inference_timeout_action=InferenceTimeoutAction.HALT,
            ml_failure_action=failure_action,
        ),
    )
    harness.actor._ml_inference_halted = True
    harness.actor._ml_failure_reason = "prior_halt"
    if harness.actor._health_monitor is not None:
        harness.actor._health_monitor.status = HealthStatus.UNHEALTHY

    harness.actor._apply_configured_ml_failure_action(
        reason="prediction_exception",
        ts_event=1,
        detail="boom",
    )

    assert harness.actor._ml_failure_reason == "prior_halt"
    assert harness.failure_metric.inc_calls == 0
    assert harness.strategy_store is not None
    assert len(harness.strategy_store.calls) == 0
    if harness.actor._health_monitor is not None:
        assert harness.actor._health_monitor.status == HealthStatus.UNHEALTHY
