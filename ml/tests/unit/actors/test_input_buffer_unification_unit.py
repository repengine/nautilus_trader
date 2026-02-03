"""
Unified input buffer preparation tests.

Contracts:
- Actor exposes a reusable 2D float32 buffer of shape [1, n_features]
- The buffer is reused across calls (same object / shared memory)
- Works with mock models exposing predict_proba, predict, and run using the 2D buffer
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.signal import MLSignalActor
from ml.tests.builders import MLConfigBuilder


@pytest.mark.unit
def test_unified_input_buffer_and_model_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure dummy mode and no metrics server
    monkeypatch.setenv("ML_ALLOW_DUMMY", "1")
    monkeypatch.setenv("ML_DISABLE_METRICS_SERVER", "1")

    # Small feature size for clarity
    cfg = MLConfigBuilder.signal_config()
    actor = MLSignalActor(cfg)

    n_features = actor._feature_engineer.n_features

    # Buffer basics
    buf = actor._predict_input_buf
    assert buf.shape == (1, n_features)
    assert buf.dtype == np.float32

    x = np.arange(n_features, dtype=np.float32)

    # predict_proba path
    mock_proba = MagicMock()
    mock_proba.predict_proba.return_value = np.array([[0.1, 0.9]], dtype=np.float32)
    mock_proba.classes_ = np.array([0, 1], dtype=np.int64)
    actor._model = mock_proba
    actor._model_metadata = {"decision_config": {"positive_class_index": 1}}
    p1, c1 = actor._predict(x)
    # Called with the preallocated buffer
    arg_array = mock_proba.predict_proba.call_args[0][0]
    assert arg_array is actor._predict_input_buf
    assert p1 == pytest.approx(0.9)
    assert c1 == pytest.approx(0.9, rel=1e-6)

    # predict path (use a concrete stub so hasattr checks don't fabricate attributes)
    pred_model = Mock(spec=["predict"])  # Only exposes predict
    pred_model.predict.return_value = np.array([0.42], dtype=np.float32)
    actor._model = pred_model
    p2, c2 = actor._predict(x)
    # Ensure predict() received the preallocated buffer
    arg_array2 = pred_model.predict.call_args[0][0]
    assert arg_array2 is actor._predict_input_buf
    assert p2 == pytest.approx(0.42)
    assert c2 == pytest.approx(0.58)

    # run (ONNX) path
    mock_onnx = Mock(spec=["run"])  # Only exposes run
    # Return [prediction], [confidence]
    mock_onnx.run.return_value = [np.array([[0.7]], dtype=np.float32), np.array([[0.8]], dtype=np.float32)]
    actor._model = mock_onnx
    actor._model_metadata = {
        "input_names": ["input"],
        "decision_config": {"positive_class_index": 1},
    }
    p3, c3 = actor._predict(x)
    # Ensure the same buffer object was provided to run
    kwargs = mock_onnx.run.call_args[0][1]
    assert isinstance(kwargs, dict)
    assert kwargs.get("input") is actor._predict_input_buf
    assert p3 == pytest.approx(0.7)
    assert c3 == pytest.approx(0.8)

    # Buffer reused across calls
    assert actor._predict_input_buf is buf


@pytest.mark.unit
def test_predict_onnx_single_output_derives_confidence() -> None:
    """
    Regression: single-output ONNX inference derives confidence from probability.
    """
    cfg = MLConfigBuilder.signal_config()
    actor = MLSignalActor(cfg)

    mock_onnx = Mock(spec=["run"])  # Only exposes run
    mock_onnx.run.return_value = [np.array([[0.7]], dtype=np.float32)]
    actor._model = mock_onnx
    actor._model_metadata = {
        "input_names": ["input"],
        "decision_config": {"positive_class_index": 1},
    }

    features = np.zeros(actor._feature_engineer.n_features, dtype=np.float32)
    pred, conf = actor._predict(features)

    assert pred == pytest.approx(0.7)
    assert conf == pytest.approx(0.7)
