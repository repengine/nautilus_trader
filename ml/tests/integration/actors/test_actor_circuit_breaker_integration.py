#!/usr/bin/env python3
from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import os
import time as _time
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, Iterator

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.base import CircuitBreakerState, MLSignal
from ml.actors.signal import MLSignalActor, MLSignalActorConfig
from ml.config.base import CircuitBreakerConfig
from nautilus_trader.model.data import Bar


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@contextmanager
def env(vars: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _configure_onnx_stub(
    mock_onnx_runtime: Any,
    onnx_session_stub_factory: Callable[..., object],
) -> None:
    """
    Ensure every test uses the deterministic ONNX harness.
    """

    mock_onnx_runtime.ort.InferenceSession.return_value = onnx_session_stub_factory()


@pytest.mark.integration
def test_circuit_breaker_transitions(
    monkeypatch: pytest.MonkeyPatch,
    base_signal_config: MLSignalActorConfig,
    generate_test_bars: list[Bar],
) -> None:
    """
    Verify circuit breaker transitions CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """
    # Immediate warmup; small thresholds for quick transitions
    cb = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0, success_threshold=1)

    cfg2 = MLSignalActorConfig(
        model_id=base_signal_config.model_id,
        model_path=base_signal_config.model_path,
        bar_type=base_signal_config.bar_type,
        instrument_id=base_signal_config.instrument_id,
        feature_config=base_signal_config.feature_config,
        batch_size=base_signal_config.batch_size,
        warm_up_period=0,
        prediction_threshold=base_signal_config.prediction_threshold,
        use_dummy_stores=True,
        signal_strategy=base_signal_config.signal_strategy,
        circuit_breaker_config=cb,
    )

    actor = MLSignalActor(cfg2)

    # Make feature computation trivial and allocation-free
    def fake_compute_features(_bar: Bar) -> npt.NDArray[np.float32]:
        return np.array([0.0, 1.0], dtype=np.float32)

    # Monkeypatch the component's method directly as it captures the callback at init
    monkeypatch.setattr(actor._features_component, "compute_features", fake_compute_features)

    # Prepare predict to fail twice, then succeed
    calls = {"n": 0}

    def fail_then_succeed(_features: npt.NDArray[np.float32]) -> tuple[float, float]:
        if calls["n"] < 2:
            calls["n"] += 1
            raise RuntimeError("predict failure")
        return (0.8, 0.9)

    monkeypatch.setattr(actor, "_predict", fail_then_succeed)

    # Feed two bars: should OPEN
    actor.on_bar(generate_test_bars[0])
    actor.on_bar(generate_test_bars[1])

    cb_inst = getattr(actor, "_circuit_breaker", None)
    assert cb_inst is not None
    assert cb_inst.state.value == CircuitBreakerState.OPEN.value

    # Ensure next attempt is allowed immediately and transition to HALF_OPEN
    setattr(cb_inst, "_next_attempt", 0.0)
    assert cb_inst.can_execute() is True
    assert cb_inst.state.value == CircuitBreakerState.HALF_OPEN.value

    # Manually record a success to reach CLOSED
    cb_inst.record_success()
    assert cb_inst.state.value == CircuitBreakerState.CLOSED.value


@pytest.mark.integration
def test_actor_bus_scheme_prefix_integration(
    monkeypatch: pytest.MonkeyPatch,
    base_signal_config: MLSignalActorConfig,
) -> None:
    """
    Verify actor-side bus honors scheme/prefix from env and publishes stage-first topic.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Pub:
        def publish(self, topic: str, payload: dict[str, Any]) -> bool:  # noqa: D401
            calls.append((topic, payload))
            return True

    def fake_factory(_cfg: Any) -> Any:
        return _Pub()

    with env(
        {
            "ML_BUS_FROM_ACTOR": "1",
            "ML_BUS_ENABLE": "1",
            "ML_BUS_SCHEME": "stage_first",
            "ML_BUS_TOPIC_PREFIX": "events.ml.qa",
        },
    ):
        monkeypatch.setattr("ml.actors.ml_domain_events.publisher_from_config", fake_factory)
        # Avoid double publish to Nautilus path
        monkeypatch.setattr(
            "ml.actors.base.BaseMLInferenceActor._publish_signal",
            lambda self, s: None,
        )

        actor = MLSignalActor(base_signal_config)

        # Bypass Nautilus actor registration requirements for publish_data.
        def _noop_publish_data(self: MLSignalActor, data_type: Any, data: Any) -> None:
            return None

        monkeypatch.setattr(
            actor,
            "publish_data",
            MethodType(_noop_publish_data, actor),
        )
        # Publish one signal
        sig = MLSignal(
            instrument_id=base_signal_config.instrument_id,
            model_id="demo",
            prediction=0.5,
            confidence=0.6,
            features=np.array([0.0], dtype=np.float32),
            ts_event=1,
            ts_init=2,
        )
        actor._publish_signal(sig)
        _time.sleep(0.05)
        bridge = getattr(actor, "_actor_bus_bridge", None)
        if bridge is not None:
            bridge.stop(drain=True, timeout=1.0)

    assert calls, "Actor bus should publish when enabled"
    topic, payload = calls[-1]
    assert topic.startswith("events.ml.qa.SIGNAL_EMITTED."), topic
    assert payload.get("stage") == "SIGNAL_EMITTED"
