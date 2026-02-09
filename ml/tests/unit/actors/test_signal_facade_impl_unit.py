"""
Targeted unit tests for MLSignalActorFacade hot-path guards and publishing branches.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

from ml.actors.base import BaseMLInferenceActor, MLSignal
from ml.actors.common.drift_monitoring import DriftObservation
from ml.actors.common.drift_monitoring import DriftPolicyConfig
from ml.actors.common.drift_monitoring import DriftPolicyOutcome
from ml.actors.common.drift_monitoring import DriftThresholds
from ml.actors.common.signal_strategy import ThresholdSignalStrategy
from ml.actors import signal_facade_impl as facade_impl
from ml.actors.signal_facade_impl import MLSignalActorFacade
from ml.actors.signal_facade_impl import _record_feature_time_metric
from ml.config.policy import DriftActionPolicy
from ml.tests.utils.stubs import make_stub_bar


pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


class _MetricStub:
    def __init__(self) -> None:
        self.labels_calls: list[dict[str, str]] = []
        self.inc_calls = 0

    def labels(self, **kwargs: str) -> _MetricStub:
        self.labels_calls.append(kwargs)
        return self

    def inc(self) -> None:
        self.inc_calls += 1


class _BufferStub:
    def __init__(self) -> None:
        self._metadata = {
            "_prediction_ring": np.zeros(1, dtype=np.float32),
            "_prediction_ring_index": 0,
            "_prediction_ring_count": 0,
        }

    def get_ring_metadata(self) -> dict[str, object]:
        return dict(self._metadata)


class _HistoryBufferStub:
    def __init__(self) -> None:
        self.window_count = 2
        self.volatility_window = np.array([0.2, 0.3], dtype=np.float32)

    def get_history(self, *, lookback: int) -> tuple[list[float], list[float]]:
        return [0.1, 0.2][:lookback], [0.4, 0.5][:lookback]


class _CounterStub:
    def __init__(self) -> None:
        self.inc_calls = 0

    def labels(self, **_: str) -> _CounterStub:
        return self

    def inc(self) -> None:
        self.inc_calls += 1


def _drift_policy_config(action: DriftActionPolicy = DriftActionPolicy.LOG_ONLY) -> DriftPolicyConfig:
    return DriftPolicyConfig(
        action_policy=action,
        thresholds=DriftThresholds(
            log_only=0.2,
            degraded=0.4,
            fail_closed=0.8,
        ),
        min_baseline_samples=2,
        min_observed_samples=3,
    )


def _drift_observation(score: float = 1.0, policy_ready: bool = True) -> DriftObservation:
    return DriftObservation(
        drift_score=score,
        sample_count=3,
        baseline_samples=2,
        baseline_ready=True,
        policy_ready=policy_ready,
    )


class _StrategyStub:
    def __init__(self, signal: MLSignal | None) -> None:
        self.signal = signal
        self.calls: list[tuple[object, float, float, object, dict[str, Any]]] = []

    def generate_signal(
        self,
        bar: object,
        prediction: float,
        confidence: float,
        features: object,
        context: dict[str, Any],
    ) -> MLSignal | None:
        self.calls.append((bar, prediction, confidence, features, context))
        return self.signal


class _StrategyStoreStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write_signal(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return make_stub_bar(inst)


def _stub_bar_with_ts_init(ts_init: int) -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    bar_type = SimpleNamespace(instrument_id=inst, spec="stub")
    return SimpleNamespace(bar_type=bar_type, ts_event=1, ts_init=ts_init)


def _stub_bar_with_ts_event(ts_event: int) -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    bar_type = SimpleNamespace(instrument_id=inst, spec="stub")
    return SimpleNamespace(bar_type=bar_type, ts_event=ts_event, ts_init=ts_event)


def _metric_stub() -> _MetricStub:
    return _MetricStub()


def test_try_generate_signal_returns_early_when_min_separation_not_met() -> None:
    strategy = Mock()
    strategy.generate_signal = Mock()
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)

    actor = SimpleNamespace(
        _signal_strategy=None,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=3,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=True, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=5,
        _last_signal_bar=4,
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=Mock(),
        _signals_generated_metric=_metric_stub(),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        _stub_bar(),
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    assert strategy_component.apply_pending_swap.called
    assert not strategy.generate_signal.called
    assert not actor._publish_signal.called


def test_try_generate_signal_returns_early_when_strategy_missing() -> None:
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=None)
    buffer_component = _BufferStub()

    actor = SimpleNamespace(
        _signal_strategy=None,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=True, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=1,
        _last_signal_bar=0,
        _prediction_buffer_component=buffer_component,
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=Mock(),
        _signals_generated_metric=_metric_stub(),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        _stub_bar(),
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    assert strategy_component.apply_pending_swap.called
    assert not actor._publish_signal.called


def test_try_generate_signal_skips_publish_when_disabled() -> None:
    bar = _stub_bar()
    signal = MLSignal(
        instrument_id=bar.bar_type.instrument_id,
        model_id="model-1",
        prediction=0.9,
        confidence=0.95,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=bar.ts_event,
        ts_init=bar.ts_event,
    )
    strategy = _StrategyStub(signal)
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)

    actor = SimpleNamespace(
        _signal_strategy=strategy,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=False, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=5,
        _last_signal_bar=0,
        _prediction_buffer_component=_BufferStub(),
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=None,
        _signals_generated_metric=_metric_stub(),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        bar,
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    assert actor._last_signal_bar == actor._bars_processed
    assert not actor._publish_signal.called


def test_try_generate_signal_records_error_when_decision_metadata_missing() -> None:
    bar = _stub_bar()
    signal = MLSignal(
        instrument_id=bar.bar_type.instrument_id,
        model_id="model-1",
        prediction=0.9,
        confidence=0.95,
        features=np.zeros(1, dtype=np.float32),
        metadata={},
        ts_event=bar.ts_event,
        ts_init=bar.ts_event,
    )
    strategy = _StrategyStub(signal)
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)
    strategy_store = _StrategyStoreStub()
    monitor = Mock()

    actor = SimpleNamespace(
        _signal_strategy=strategy,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=False, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=2,
        _last_signal_bar=0,
        _prediction_buffer_component=_BufferStub(),
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        log=SimpleNamespace(error=lambda *a, **k: None, debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=strategy_store,
        _performance_monitoring_component=monitor,
        _signals_generated_metric=_metric_stub(),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        bar,
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    assert not strategy_store.calls
    monitor.record_error.assert_called_once()


def test_generate_prediction_protected_forces_signal_mode(monkeypatch) -> None:
    monkeypatch.setenv("FORCE_SIGNAL_MODE", "true")

    buffer_component = SimpleNamespace(
        update=Mock(),
        volatility_window=np.array([0.1], dtype=np.float32),
        window_count=1,
    )
    adaptive_component = SimpleNamespace(detect_regime=Mock())
    performance_component = SimpleNamespace(record_timing=Mock(), record_error=Mock())

    actor = SimpleNamespace(
        _should_hot_reload=lambda: False,
        _execute_hot_reload=lambda: None,
        _predict=Mock(side_effect=AssertionError("predict should not be called")),
        _prediction_count=0,
        _apply_inference_deadline_guard=Mock(return_value=False),
        _calculate_volatility=lambda _bar: 0.05,
        _prediction_buffer_component=buffer_component,
        _adaptive_threshold_component=adaptive_component,
        _persist_prediction=Mock(),
        _try_generate_signal=Mock(),
        _performance_monitoring_component=performance_component,
        _last_feature_time_ns=1_000,
        _record_success=Mock(),
        _record_failure=Mock(),
        _apply_configured_ml_failure_action=Mock(),
        log=SimpleNamespace(exception=lambda *a, **k: None),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._generate_prediction_protected(
        actor_proxy,
        _stub_bar(),
        np.zeros(1, dtype=np.float32),
    )

    buffer_component.update.assert_called_once_with(1.0, 1.0, 0.05)
    adaptive_component.detect_regime.assert_called_once_with(
        buffer_component.volatility_window,
        buffer_component.window_count,
    )
    assert actor._prediction_count == 1
    actor._persist_prediction.assert_called_once()
    actor._try_generate_signal.assert_called_once()
    performance_component.record_timing.assert_called_once()
    actor._record_success.assert_called_once()


def test_generate_prediction_protected_stops_pipeline_on_deadline_guard() -> None:
    buffer_component = SimpleNamespace(
        update=Mock(),
        volatility_window=np.array([0.1], dtype=np.float32),
        window_count=1,
    )
    adaptive_component = SimpleNamespace(detect_regime=Mock())
    performance_component = SimpleNamespace(record_timing=Mock(), record_error=Mock())

    actor = SimpleNamespace(
        _should_hot_reload=lambda: False,
        _execute_hot_reload=lambda: None,
        _predict=Mock(return_value=(0.9, 0.8)),
        _prediction_count=0,
        _apply_inference_deadline_guard=Mock(return_value=True),
        _calculate_volatility=lambda _bar: 0.05,
        _prediction_buffer_component=buffer_component,
        _adaptive_threshold_component=adaptive_component,
        _persist_prediction=Mock(),
        _try_generate_signal=Mock(),
        _performance_monitoring_component=performance_component,
        _last_feature_time_ns=2_000,
        _record_success=Mock(),
        _record_failure=Mock(),
        _apply_configured_ml_failure_action=Mock(),
        log=SimpleNamespace(exception=lambda *a, **k: None),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._generate_prediction_protected(
        actor_proxy,
        _stub_bar(),
        np.zeros(1, dtype=np.float32),
    )

    assert actor._prediction_count == 1
    actor._apply_inference_deadline_guard.assert_called_once()
    assert not buffer_component.update.called
    assert not adaptive_component.detect_regime.called
    assert not actor._persist_prediction.called
    assert not actor._try_generate_signal.called
    performance_component.record_timing.assert_called_once()
    assert not actor._record_success.called
    assert not actor._record_failure.called


def test_generate_prediction_protected_stops_pipeline_on_drift_halt() -> None:
    buffer_component = SimpleNamespace(
        update=Mock(),
        volatility_window=np.array([0.1], dtype=np.float32),
        window_count=1,
    )
    adaptive_component = SimpleNamespace(detect_regime=Mock())
    performance_component = SimpleNamespace(record_timing=Mock(), record_error=Mock())

    actor = SimpleNamespace(
        _should_hot_reload=lambda: False,
        _execute_hot_reload=lambda: None,
        _predict=Mock(return_value=(0.9, 0.8)),
        _prediction_count=0,
        _apply_inference_deadline_guard=Mock(return_value=False),
        _handle_runtime_drift_policy=Mock(return_value=True),
        _calculate_volatility=lambda _bar: 0.05,
        _prediction_buffer_component=buffer_component,
        _adaptive_threshold_component=adaptive_component,
        _persist_prediction=Mock(),
        _try_generate_signal=Mock(),
        _performance_monitoring_component=performance_component,
        _last_feature_time_ns=2_000,
        _record_success=Mock(),
        _record_failure=Mock(),
        _apply_configured_ml_failure_action=Mock(),
        log=SimpleNamespace(exception=lambda *a, **k: None),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._generate_prediction_protected(
        actor_proxy,
        _stub_bar(),
        np.zeros(1, dtype=np.float32),
    )

    actor._handle_runtime_drift_policy.assert_called_once()
    assert actor._prediction_count == 1
    assert buffer_component.update.called
    assert adaptive_component.detect_regime.called
    assert not actor._persist_prediction.called
    assert not actor._try_generate_signal.called
    performance_component.record_timing.assert_called_once()
    assert not actor._record_success.called
    assert not actor._record_failure.called


def test_generate_prediction_protected_applies_ml_failure_action_on_exception() -> None:
    performance_component = SimpleNamespace(record_timing=Mock(), record_error=Mock())

    actor = SimpleNamespace(
        _should_hot_reload=lambda: False,
        _execute_hot_reload=lambda: None,
        _predict=Mock(side_effect=RuntimeError("boom")),
        _prediction_count=0,
        _apply_inference_deadline_guard=Mock(return_value=False),
        _calculate_volatility=lambda _bar: 0.05,
        _prediction_buffer_component=SimpleNamespace(
            update=Mock(),
            volatility_window=np.array([0.1], dtype=np.float32),
            window_count=1,
        ),
        _adaptive_threshold_component=SimpleNamespace(detect_regime=Mock()),
        _persist_prediction=Mock(),
        _try_generate_signal=Mock(),
        _performance_monitoring_component=performance_component,
        _last_feature_time_ns=1_000,
        _record_success=Mock(),
        _record_failure=Mock(),
        _apply_configured_ml_failure_action=Mock(),
        log=SimpleNamespace(exception=lambda *a, **k: None),
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    bar = _stub_bar()
    MLSignalActorFacade._generate_prediction_protected(
        actor_proxy,
        bar,
        np.zeros(1, dtype=np.float32),
    )

    performance_component.record_error.assert_called_once()
    actor._record_failure.assert_called_once()
    actor._apply_configured_ml_failure_action.assert_called_once_with(
        reason="prediction_exception",
        ts_event=int(bar.ts_event),
        detail="RuntimeError('boom')",
    )


@pytest.mark.runtime_correctness
@pytest.mark.parametrize(
    ("action", "expected_abort"),
    [
        (DriftActionPolicy.LOG_ONLY, False),
        (DriftActionPolicy.DEGRADED, False),
        (DriftActionPolicy.FAIL_CLOSED, True),
    ],
)
def test_handle_runtime_drift_policy_applies_configured_action(
    action: DriftActionPolicy,
    expected_abort: bool,
) -> None:
    observation = _drift_observation(score=1.0, policy_ready=True)
    outcome = DriftPolicyOutcome(
        action=action,
        configured_action=action,
        reason="runtime_feature_drift",
        drift_score=1.0,
        threshold=0.2,
    )
    monitor = SimpleNamespace(
        record_inference=Mock(return_value=observation),
        evaluate_policy=Mock(return_value=outcome),
        policy_config=_drift_policy_config(action=action),
    )
    warmup = SimpleNamespace(is_drift_policy_ready=Mock(return_value=True))

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _drift_monitoring_component=monitor,
            _model_warmup_component=warmup,
            cache=SimpleNamespace(is_backtesting=False),
            _apply_drift_policy_outcome=Mock(return_value=expected_abort),
            log=SimpleNamespace(debug=Mock()),
        ),
    )
    bar = _stub_bar()

    should_abort = MLSignalActorFacade._handle_runtime_drift_policy(
        actor,
        bar=bar,
        features=np.array([1.0], dtype=np.float32),
    )

    assert should_abort is expected_abort
    warmup.is_drift_policy_ready.assert_called_once()
    monitor.evaluate_policy.assert_called_once_with(
        observation=observation,
        policy_ready=True,
    )
    actor._apply_drift_policy_outcome.assert_called_once()
    kwargs = actor._apply_drift_policy_outcome.call_args.kwargs
    assert kwargs["action"] == action
    assert kwargs["reason"] == "runtime_feature_drift"
    assert kwargs["drift_score"] == pytest.approx(1.0)
    assert kwargs["threshold"] == pytest.approx(0.2)
    assert f"configured_action={action.value}" in kwargs["detail"]
    assert kwargs["ts_event"] == int(bar.ts_event)


@pytest.mark.runtime_correctness
def test_handle_runtime_drift_policy_replay_forces_log_only() -> None:
    observation = _drift_observation(score=1.0, policy_ready=True)
    outcome = DriftPolicyOutcome(
        action=DriftActionPolicy.FAIL_CLOSED,
        configured_action=DriftActionPolicy.FAIL_CLOSED,
        reason="runtime_feature_drift",
        drift_score=1.0,
        threshold=0.2,
    )
    monitor = SimpleNamespace(
        record_inference=Mock(return_value=observation),
        evaluate_policy=Mock(return_value=outcome),
        policy_config=_drift_policy_config(action=DriftActionPolicy.FAIL_CLOSED),
    )

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _drift_monitoring_component=monitor,
            _model_warmup_component=SimpleNamespace(is_drift_policy_ready=Mock(return_value=True)),
            cache=SimpleNamespace(is_backtesting=True),
            _apply_drift_policy_outcome=Mock(return_value=False),
            log=SimpleNamespace(debug=Mock()),
        ),
    )

    should_abort = MLSignalActorFacade._handle_runtime_drift_policy(
        actor,
        bar=_stub_bar(),
        features=np.array([1.0], dtype=np.float32),
    )

    assert should_abort is False
    actor._model_warmup_component.is_drift_policy_ready.assert_called_once()
    monitor.evaluate_policy.assert_called_once_with(
        observation=observation,
        policy_ready=True,
    )
    kwargs = actor._apply_drift_policy_outcome.call_args.kwargs
    assert kwargs["action"] == DriftActionPolicy.LOG_ONLY
    assert kwargs["reason"] == "runtime_feature_drift_replay_safe"
    assert kwargs["drift_score"] == pytest.approx(1.0)
    assert kwargs["threshold"] == pytest.approx(0.2)
    assert "configured_action=fail_closed" in kwargs["detail"]


@pytest.mark.runtime_correctness
def test_handle_runtime_drift_policy_respects_warmup_gate() -> None:
    observation = _drift_observation(score=1.0, policy_ready=True)
    monitor = SimpleNamespace(
        record_inference=Mock(return_value=observation),
        evaluate_policy=Mock(return_value=None),
        policy_config=_drift_policy_config(action=DriftActionPolicy.FAIL_CLOSED),
    )
    warmup = SimpleNamespace(is_drift_policy_ready=Mock(return_value=False))

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _drift_monitoring_component=monitor,
            _model_warmup_component=warmup,
            cache=SimpleNamespace(is_backtesting=False),
            _apply_drift_policy_outcome=Mock(return_value=True),
            log=SimpleNamespace(debug=Mock()),
        ),
    )

    should_abort = MLSignalActorFacade._handle_runtime_drift_policy(
        actor,
        bar=_stub_bar(),
        features=np.array([1.0], dtype=np.float32),
    )

    assert should_abort is False
    warmup.is_drift_policy_ready.assert_called_once()
    monitor.evaluate_policy.assert_called_once_with(
        observation=observation,
        policy_ready=False,
    )
    actor._apply_drift_policy_outcome.assert_not_called()


def test_publish_signal_returns_when_bridge_missing(
    base_signal_config,
    default_instrument_id,
    mock_onnx_runtime,
    monkeypatch,
) -> None:
    _ = mock_onnx_runtime
    actor = MLSignalActorFacade(base_signal_config)
    actor._actor_bus_bridge = None

    calls: list[MLSignal] = []

    def _noop_publish(self: BaseMLInferenceActor, signal: MLSignal) -> None:
        calls.append(signal)

    signal = MLSignal(
        instrument_id=default_instrument_id,
        model_id="model-1",
        prediction=0.8,
        confidence=0.9,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=1,
        ts_init=1,
    )

    monkeypatch.setattr(BaseMLInferenceActor, "_publish_signal", _noop_publish)
    MLSignalActorFacade._publish_signal(actor, signal)

    assert calls == [signal]


def test_publish_signal_logs_when_bridge_publish_fails(
    base_signal_config,
    default_instrument_id,
    mock_onnx_runtime,
    monkeypatch,
) -> None:
    _ = mock_onnx_runtime
    actor = MLSignalActorFacade(base_signal_config)
    actor._topic_scheme = "domain_op"
    actor._topic_prefix = "events.ml"

    bridge = Mock()
    bridge.publish.side_effect = RuntimeError("boom")
    actor._actor_bus_bridge = bridge

    def _noop_publish(self: BaseMLInferenceActor, signal: MLSignal) -> None:
        return None

    signal = MLSignal(
        instrument_id=default_instrument_id,
        model_id="model-1",
        prediction=0.8,
        confidence=0.9,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=1,
        ts_init=1,
    )

    monkeypatch.setattr(BaseMLInferenceActor, "_publish_signal", _noop_publish)
    MLSignalActorFacade._publish_signal(actor, signal)

    assert bridge.publish.called


def test_record_feature_time_metric_prefers_module_override_and_logs(monkeypatch) -> None:
    class _MetricRaises:
        def labels(self, **_: str) -> _MetricRaises:
            raise RuntimeError("boom")

    module_stub = SimpleNamespace(_feature_time_by_feature_set_metric=_MetricRaises())
    monkeypatch.setitem(sys.modules, "ml.actors.signal", module_stub)

    log = SimpleNamespace(debug=Mock())
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(_feature_set_id="fs-1", id="actor-1", log=log),
    )

    _record_feature_time_metric(actor, 5.0)


def test_record_feature_time_metric_returns_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(facade_impl, "_resolve_feature_time_metric", lambda: None)
    actor = cast(MLSignalActorFacade, SimpleNamespace())
    _record_feature_time_metric(actor, 1.0)


def test_try_generate_signal_uses_ts_init_when_clock_missing(monkeypatch) -> None:
    bar = _stub_bar_with_ts_init(123)
    signal = MLSignal(
        instrument_id=bar.bar_type.instrument_id,
        model_id="model-1",
        prediction=0.9,
        confidence=0.95,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=bar.ts_event,
        ts_init=bar.ts_init,
    )
    strategy = _StrategyStub(signal)
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)

    actor = SimpleNamespace(
        _signal_strategy=strategy,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=False, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=3,
        _last_signal_bar=0,
        _prediction_buffer_component=_BufferStub(),
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=None,
        _signals_generated_metric=None,
    )

    monkeypatch.setattr(facade_impl, "_signals_generated_metric", None)

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        bar,
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    context = strategy.calls[0][4]
    assert context["timestamp_ns"] == 123


def test_try_generate_signal_falls_back_when_clock_fails() -> None:
    bar = _stub_bar_with_ts_init(222)
    signal = MLSignal(
        instrument_id=bar.bar_type.instrument_id,
        model_id="model-1",
        prediction=0.9,
        confidence=0.95,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=bar.ts_event,
        ts_init=bar.ts_init,
    )
    strategy = _StrategyStub(signal)
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)

    def _raise() -> int:
        raise RuntimeError("boom")

    actor = SimpleNamespace(
        _signal_strategy=strategy,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=False, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _decision_metadata_payload={"version": "v1"},
        _bars_processed=3,
        _last_signal_bar=0,
        _prediction_buffer_component=_BufferStub(),
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        clock=SimpleNamespace(timestamp_ns=_raise),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=None,
        _signals_generated_metric=None,
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        bar,
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    context = strategy.calls[0][4]
    assert context["timestamp_ns"] == 222


def test_try_generate_signal_populates_decision_metadata(monkeypatch) -> None:
    sentinel = {"version": "v2"}

    monkeypatch.setattr(
        "ml.common.decision_metadata_from_model_metadata",
        lambda *_args, **_kwargs: sentinel,
    )

    bar = _stub_bar_with_ts_init(100)
    signal = MLSignal(
        instrument_id=bar.bar_type.instrument_id,
        model_id="model-1",
        prediction=0.9,
        confidence=0.95,
        features=np.zeros(1, dtype=np.float32),
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=bar.ts_event,
        ts_init=bar.ts_init,
    )
    strategy = _StrategyStub(signal)
    strategy_component = SimpleNamespace(apply_pending_swap=Mock(), current_strategy=strategy)

    actor = SimpleNamespace(
        _signal_strategy=strategy,
        _signal_strategy_component=strategy_component,
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            prediction_neutral_band=0.0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False, publish_signals=False, use_dummy_stores=True),
        id="actor-1",
        _model_id="m1",
        _model_metadata={"calibration": {"foo": "bar"}},
        _model_version="v1",
        _decision_metadata_payload=None,
        _bars_processed=3,
        _last_signal_bar=0,
        _prediction_buffer_component=_BufferStub(),
        _prediction_history=[],
        _confidence_history=[],
        _adaptive_threshold_component=SimpleNamespace(
            current_threshold=0.2,
            current_regime="neutral",
        ),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=Mock(),
        _strategy_store=None,
        _signals_generated_metric=None,
    )

    actor_proxy = cast(MLSignalActorFacade, actor)
    MLSignalActorFacade._try_generate_signal(
        actor_proxy,
        bar,
        0.25,
        0.6,
        np.zeros(1, dtype=np.float32),
    )

    assert actor._decision_metadata_payload == sentinel
    context = strategy.calls[0][4]
    assert context["signal_metadata"]["decision_metadata"]["version"] == "v2"


def test_reset_signal_state_resets_components() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _prediction_buffer_component=SimpleNamespace(reset=Mock()),
            _adaptive_threshold_component=SimpleNamespace(update_threshold=Mock()),
            _signal_config=SimpleNamespace(min_signal_separation_bars=3),
            _last_signal_bar=10,
            _last_close_price=1.0,
            log=SimpleNamespace(info=Mock()),
        ),
    )

    MLSignalActorFacade.reset_signal_state(actor)

    assert actor._last_signal_bar == -3
    assert actor._last_close_price is None
    actor._prediction_buffer_component.reset.assert_called_once()
    actor._adaptive_threshold_component.update_threshold.assert_called_once_with(0.0)


def test_prepare_bar_runtime_state_resets_on_backstep() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _last_processed_ts_event=200,
            _reset_inference_runtime_state=Mock(),
            log=SimpleNamespace(info=Mock()),
        ),
    )

    MLSignalActorFacade._prepare_bar_runtime_state(actor, _stub_bar_with_ts_event(150))

    actor._reset_inference_runtime_state.assert_called_once_with(
        reason="replay_rewind_backstep",
        ts_event=150,
    )
    assert actor._last_processed_ts_event == 150


def test_prepare_bar_runtime_state_accepts_monotonic_timestamps() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _last_processed_ts_event=100,
            _reset_inference_runtime_state=Mock(),
            log=SimpleNamespace(info=Mock()),
        ),
    )

    MLSignalActorFacade._prepare_bar_runtime_state(actor, _stub_bar_with_ts_event(100))
    MLSignalActorFacade._prepare_bar_runtime_state(actor, _stub_bar_with_ts_event(101))

    actor._reset_inference_runtime_state.assert_not_called()
    assert actor._last_processed_ts_event == 101


def test_reset_inference_runtime_state_components_resets_facade_runtime() -> None:
    indicator_manager = SimpleNamespace(reset=Mock())
    feature_engineer = SimpleNamespace(reset=Mock())
    drift_monitor = SimpleNamespace(reset_runtime_state=Mock())
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            reset_signal_state=Mock(),
            _indicator_manager=indicator_manager,
            _feature_engineer=feature_engineer,
            _drift_monitoring_component=drift_monitor,
            _last_feature_time_ns=123,
            _last_processed_ts_event=999,
        ),
    )

    MLSignalActorFacade._reset_inference_runtime_state_components(actor)

    actor.reset_signal_state.assert_called_once()
    indicator_manager.reset.assert_called_once()
    feature_engineer.reset.assert_called_once()
    drift_monitor.reset_runtime_state.assert_called_once()
    assert actor._indicator_manager is None
    assert actor._last_feature_time_ns == 0
    assert actor._last_processed_ts_event is None


def test_history_and_threshold_overrides() -> None:
    buffer = _HistoryBufferStub()
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _prediction_buffer_component=buffer,
            _prediction_history_override=None,
            _confidence_history_override=None,
            _adaptive_threshold_component=SimpleNamespace(current_threshold=0.2, current_regime="bull"),
            _adaptive_threshold_override=None,
            _market_regime_override=None,
            _window_index_override=None,
            _window_count_override=None,
            _volatility_window_override=None,
        ),
    )

    pred_history = MLSignalActorFacade._prediction_history.fget(actor)
    conf_history = MLSignalActorFacade._confidence_history.fget(actor)
    assert pred_history == [0.1, 0.2]
    assert conf_history == [0.4, 0.5]

    MLSignalActorFacade._prediction_history.fset(actor, [0.9])
    MLSignalActorFacade._confidence_history.fset(actor, [0.8])
    assert MLSignalActorFacade._prediction_history.fget(actor) == [0.9]
    assert MLSignalActorFacade._confidence_history.fget(actor) == [0.8]

    MLSignalActorFacade._adaptive_threshold.fset(actor, 0.7)
    assert MLSignalActorFacade._adaptive_threshold.fget(actor) == 0.7
    MLSignalActorFacade._market_regime.fset(actor, "bear")
    assert MLSignalActorFacade._market_regime.fget(actor) == "bear"

    MLSignalActorFacade._window_index.fset(actor, 5)
    MLSignalActorFacade._window_count.fset(actor, 9)
    assert MLSignalActorFacade._window_index.fget(actor) == 5
    assert MLSignalActorFacade._window_count.fget(actor) == 9

    vol = np.array([0.1, 0.2], dtype=np.float32)
    MLSignalActorFacade._volatility_window.fset(actor, vol)
    np.testing.assert_allclose(MLSignalActorFacade._volatility_window.fget(actor), vol)

    actor._prediction_buffer_component = None
    actor._prediction_history_override = None
    actor._confidence_history_override = None
    actor._volatility_window_override = None

    assert MLSignalActorFacade._prediction_history.fget(actor) == []
    assert MLSignalActorFacade._confidence_history.fget(actor) == []
    assert MLSignalActorFacade._volatility_window.fget(actor).size == 0


def test_update_prediction_history_and_detect_market_regime() -> None:
    buffer = SimpleNamespace(update=Mock(), volatility_window=np.array([0.2]), window_count=1)
    threshold = SimpleNamespace(detect_regime=Mock(return_value="bull"))
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _prediction_buffer_component=buffer,
            _adaptive_threshold_component=threshold,
            _calculate_volatility=lambda _bar: 0.5,
        ),
    )

    MLSignalActorFacade._update_prediction_history(actor, 0.1, 0.2, _stub_bar_with_ts_init(1))

    buffer.update.assert_called_once_with(0.1, 0.2, 0.5)
    threshold.detect_regime.assert_called_once_with(buffer.volatility_window, buffer.window_count)

    actor._prediction_buffer_component = None
    actor._adaptive_threshold_component = None
    assert MLSignalActorFacade._detect_market_regime(actor, _stub_bar_with_ts_init(1)) == "unknown"


def test_calculate_volatility_updates_last_close_price() -> None:
    bar = SimpleNamespace(close=SimpleNamespace(as_double=lambda: 10.0))
    actor = cast(MLSignalActorFacade, SimpleNamespace(_last_close_price=None))
    assert MLSignalActorFacade._calculate_volatility(actor, bar) == 0.0

    bar2 = SimpleNamespace(close=SimpleNamespace(as_double=lambda: 12.5))
    assert MLSignalActorFacade._calculate_volatility(actor, bar2) == 2.5


def test_persist_prediction_handles_feature_name_errors() -> None:
    bar = SimpleNamespace(bar_type=SimpleNamespace(instrument_id="EUR/USD.SIM"), ts_event=10)
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _active_feature_names=[],
            _feature_engineer=SimpleNamespace(
                config=SimpleNamespace(get_feature_names=lambda: (_ for _ in ()).throw(RuntimeError())),
            ),
            _persist_prediction_async=Mock(),
            log=SimpleNamespace(exception=Mock()),
        ),
    )

    MLSignalActorFacade._persist_prediction(
        actor,
        bar,
        np.array([1.0, 2.0], dtype=np.float32),
        0.5,
        0.6,
        0.0,
    )

    assert actor.log.exception.called
    assert actor._persist_prediction_async.called


def test_persist_prediction_logs_on_failure() -> None:
    bar = SimpleNamespace(bar_type=SimpleNamespace(instrument_id="EUR/USD.SIM"), ts_event=10)

    def _raise(**_: Any) -> None:
        raise RuntimeError("boom")

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _active_feature_names=["f1"],
            _feature_engineer=SimpleNamespace(config=SimpleNamespace(get_feature_names=lambda: ["f1"])),
            _persist_prediction_async=_raise,
            log=SimpleNamespace(exception=Mock()),
        ),
    )

    MLSignalActorFacade._persist_prediction(
        actor,
        bar,
        np.array([1.0], dtype=np.float32),
        0.5,
        0.6,
        0.0,
    )

    assert actor.log.exception.called


def test_hot_reload_checks_interval_and_updates() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _config=SimpleNamespace(enable_hot_reload=False),
            _signal_config=SimpleNamespace(hot_reload_interval=10.0),
            _last_model_check=90.0,
        ),
    )

    assert not MLSignalActorFacade._should_hot_reload(actor)

    actor._config = SimpleNamespace(enable_hot_reload=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(facade_impl.time, "time", lambda: 95.0)
    try:
        assert not MLSignalActorFacade._should_hot_reload(actor)
        monkeypatch.setattr(facade_impl.time, "time", lambda: 105.0)
        assert MLSignalActorFacade._should_hot_reload(actor)
        assert actor._last_model_check == 105.0
    finally:
        monkeypatch.undo()


def test_execute_hot_reload_with_component_updates_model() -> None:
    class _Warmup:
        def should_hot_reload(self) -> bool:
            return True

        def load_model(self) -> tuple[object, dict[str, object]]:
            return "fallback", {"meta": "fallback"}

        def execute_hot_reload(
            self,
            *,
            model_path: str,
            load_model_fn: Any,
        ) -> tuple[object, dict[str, object]] | None:
            return "model", {"meta": "hot"}

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _model_warmup_component=_Warmup(),
            _config=SimpleNamespace(model_path="model.onnx"),
            _model=None,
            _model_metadata={},
        ),
    )

    MLSignalActorFacade._execute_hot_reload(actor)

    assert actor._model == "model"
    assert actor._model_metadata == {"meta": "hot"}


def test_execute_hot_reload_falls_back_to_legacy(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test")

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _model_warmup_component=None,
            _config=SimpleNamespace(model_path=str(model_path)),
            _model_mtime=None,
            _load_model_with_metadata=Mock(),
            log=SimpleNamespace(info=Mock(), exception=Mock()),
        ),
    )

    MLSignalActorFacade._execute_hot_reload(actor)

    assert actor._load_model_with_metadata.called
    assert actor._model_mtime == model_path.stat().st_mtime


def test_record_success_and_failure_updates_monitors() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _health_monitor=SimpleNamespace(
                update_prediction_success=Mock(),
                update_prediction_failure=Mock(),
            ),
            _circuit_breaker=SimpleNamespace(record_success=Mock(), record_failure=Mock()),
        ),
    )

    MLSignalActorFacade._record_success(actor)
    MLSignalActorFacade._record_failure(actor)

    actor._health_monitor.update_prediction_success.assert_called_once()
    actor._health_monitor.update_prediction_failure.assert_called_once()
    actor._circuit_breaker.record_success.assert_called_once()
    actor._circuit_breaker.record_failure.assert_called_once()


def test_load_model_initialize_features_and_create_strategy() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _model_warmup_component=SimpleNamespace(
                load_model=lambda: ("model", {"meta": "v1"}),
            ),
            _feature_engineer=SimpleNamespace(n_features=3),
            _feature_buffer=np.zeros(2, dtype=np.float32),
            _signal_strategy_component=SimpleNamespace(
                create_strategy=Mock(side_effect=RuntimeError("boom")),
            ),
            _signal_config=SimpleNamespace(prediction_threshold=0.4),
            log=SimpleNamespace(debug=lambda *a, **k: None),
        ),
    )

    MLSignalActorFacade._load_model(actor)
    assert actor._model == "model"
    assert actor._model_metadata == {"meta": "v1"}

    MLSignalActorFacade._initialize_features(actor)
    assert actor._feature_buffer.size == 3

    strategy = MLSignalActorFacade._create_strategy(actor)
    assert isinstance(strategy, ThresholdSignalStrategy)
    assert actor._signal_strategy is strategy


def test_prepare_onnx_input_aligns_features(monkeypatch) -> None:
    counter = _CounterStub()
    monkeypatch.setattr(facade_impl, "_inference_fallback_counter", counter)

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _predict_input_buf=np.zeros((1, 1), dtype=np.float32),
            _input_dim_mismatch_logged=False,
            _resolve_model_input_dim=lambda: 3,
            log=SimpleNamespace(warning=Mock()),
            id="actor-1",
        ),
    )

    out = MLSignalActorFacade._prepare_onnx_input(actor, np.array([1.0, 2.0], dtype=np.float32))

    assert out.shape == (1, 3)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(2.0)
    assert actor._input_dim_mismatch_logged
    assert counter.inc_calls == 1


def test_extract_output_scalar_handles_empty_and_mismatch(monkeypatch) -> None:
    counter = _CounterStub()
    monkeypatch.setattr(facade_impl, "_inference_fallback_counter", counter)

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(_output_shape_mismatch_logged=False, log=SimpleNamespace(warning=Mock()), id="a1"),
    )

    assert MLSignalActorFacade._extract_output_scalar(actor, [], label="pred") == 0.0
    assert actor._output_shape_mismatch_logged

    actor._output_shape_mismatch_logged = False
    assert MLSignalActorFacade._extract_output_scalar(actor, [0.2, 0.3], label="pred") == 0.2


def test_resolve_model_input_dim_handles_metadata_and_session() -> None:
    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(_model_metadata={"input_shape": [None, 4]}, _model=SimpleNamespace(), log=Mock()),
    )
    assert MLSignalActorFacade._resolve_model_input_dim(actor) == 4

    class _Model:
        def get_inputs(self) -> list[object]:
            return [SimpleNamespace(shape=[None, 6])]

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(_model_metadata={"input_shape": ["x", "y"]}, _model=_Model(), log=Mock()),
    )
    assert MLSignalActorFacade._resolve_model_input_dim(actor) == 6

    def _raise() -> list[object]:
        raise RuntimeError("boom")

    actor = cast(
        MLSignalActorFacade,
        SimpleNamespace(
            _model_metadata={"input_shape": ["x", "y"]},
            _model=SimpleNamespace(get_inputs=_raise),
            log=SimpleNamespace(debug=Mock()),
        ),
    )
    assert MLSignalActorFacade._resolve_model_input_dim(actor) == 0
    actor.log.debug.assert_called_once()
