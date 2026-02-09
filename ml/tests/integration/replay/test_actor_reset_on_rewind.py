from __future__ import annotations

from types import MethodType
from unittest.mock import Mock

import msgspec
import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.signal_facade_impl import MLSignalActorFacade
from ml.config.actors import MLSignalActorConfig
from ml.tests.utils.stubs import make_stub_bar


pytestmark = [
    pytest.mark.integration,
    pytest.mark.serial,
    pytest.mark.runtime_correctness,
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


def test_actor_resets_runtime_state_on_timestamp_rewind(
    base_signal_config: MLSignalActorConfig,
    mock_onnx_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mock_onnx_runtime
    config = msgspec.structs.replace(
        base_signal_config,
        warm_up_period=1,
        publish_signals=False,
        use_dummy_stores=True,
    )
    actor = MLSignalActorFacade(config)

    features = np.array([0.1], dtype=np.float32)
    monkeypatch.setattr(actor._features_component, "compute_features", lambda _bar: features)
    generate_prediction = Mock()
    monkeypatch.setattr(actor, "_generate_prediction_protected", generate_prediction)

    reset_calls: list[tuple[str, int | None]] = []
    original_reset = actor._reset_inference_runtime_state

    def _wrapped_reset(
        self: MLSignalActorFacade,
        *,
        reason: str,
        ts_event: int | None = None,
    ) -> None:
        reset_calls.append((reason, ts_event))
        original_reset(reason=reason, ts_event=ts_event)

    actor._reset_inference_runtime_state = MethodType(_wrapped_reset, actor)

    actor._is_warmed_up = True
    actor._bars_processed = 10

    forward_bar = make_stub_bar(config.instrument_id, ts_event=1_000, close=1.1000)
    actor.on_bar(forward_bar)
    assert actor._last_processed_ts_event == 1_000
    generate_prediction.assert_called_once()
    generate_prediction.reset_mock()

    actor._ml_inference_halted = True
    actor._ml_failure_reason = "manual_halt"
    actor._prediction_buffer_component.update(0.8, 0.9, 0.1)
    actor._drift_monitoring_component.record_inference(
        np.zeros_like(actor._feature_buffer, dtype=np.float32),
    )

    rewind_bar = make_stub_bar(config.instrument_id, ts_event=900, close=1.1001)
    actor.on_bar(rewind_bar)

    assert reset_calls == [("replay_rewind_backstep", 900)]
    assert actor._ml_inference_halted is False
    assert actor._ml_failure_reason is None
    assert actor._bars_processed == 1
    assert actor._last_processed_ts_event == 900
    assert actor._prediction_buffer_component.window_count == 0
    assert actor._drift_monitoring_component._sample_count == 0
    generate_prediction.assert_called_once()


def test_actor_rewind_rejects_stale_state_and_uses_rewind_features(
    base_signal_config: MLSignalActorConfig,
    mock_onnx_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mock_onnx_runtime
    config = msgspec.structs.replace(
        base_signal_config,
        warm_up_period=1,
        publish_signals=False,
        use_dummy_stores=True,
    )
    actor = MLSignalActorFacade(config)

    def _features_for_bar(bar: object) -> npt.NDArray[np.float32]:
        return np.array([float(getattr(bar, "ts_event", 0))], dtype=np.float32)

    monkeypatch.setattr(actor._features_component, "compute_features", _features_for_bar)

    observed_predictions: list[tuple[int, float]] = []

    def _capture_prediction(
        self: MLSignalActorFacade,
        bar: object,
        features: npt.NDArray[np.float32],
    ) -> None:
        observed_predictions.append((int(getattr(bar, "ts_event", 0)), float(features[0])))

    actor._generate_prediction_protected = MethodType(_capture_prediction, actor)
    actor._is_warmed_up = True
    actor._bars_processed = 5

    forward_bar = make_stub_bar(config.instrument_id, ts_event=1_000, close=1.1000)
    actor.on_bar(forward_bar)
    assert observed_predictions == [(1_000, 1_000.0)]

    actor._ml_inference_halted = True
    actor._ml_failure_reason = "stale_halt"
    actor._prediction_buffer_component.update(0.8, 0.9, 0.1)
    actor._drift_monitoring_component.record_inference(
        np.zeros_like(actor._feature_buffer, dtype=np.float32),
    )
    assert actor._prediction_buffer_component.window_count > 0
    assert actor._drift_monitoring_component._sample_count > 0

    rewind_bar = make_stub_bar(config.instrument_id, ts_event=900, close=1.1001)
    actor.on_bar(rewind_bar)

    assert observed_predictions == [(1_000, 1_000.0), (900, 900.0)]
    assert actor._ml_inference_halted is False
    assert actor._ml_failure_reason is None
    assert actor._prediction_buffer_component.window_count == 0
    assert actor._drift_monitoring_component._sample_count == 0
    assert actor._bars_processed == 1
    assert actor._last_processed_ts_event == 900
