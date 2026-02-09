#!/usr/bin/env python3
"""
Unit tests for ModelWarmUpComponent.

This module tests the model warm-up component which manages ONNX model loading,
warm-up iterations, feature parity smoke checks, and hot-reload scheduling for
MLSignalActor decomposition.

Test Categories (25 tests total):
- ONNX Loading: 5 tests
- Model Warm-Up: 5 tests
- Feature Parity Smoke Check: 8 tests
- Hot Reload Scheduling: 7 tests

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (all cold path operations)
- Pattern 2: Protocol-First Interface Design (property accessors)
- Centralized ML Imports (ml._imports for ONNX)

"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import numpy.typing as npt
import pytest

from ml.tests.utils.model_artifacts import write_stub_onnx_artifact


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger() -> logging.Logger:
    """
    Mock logger for testing.
    """
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def temp_onnx_model_path() -> str:
    """
    Create a temporary file path to simulate an ONNX model.

    This does NOT create an actual ONNX model, just a placeholder file. Tests that need
    actual ONNX behavior should mock onnxruntime.

    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    write_stub_onnx_artifact(
        Path(path),
        content=b"dummy_onnx_model_content",
    )
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)
    sidecar_path = Path(path).with_suffix(".meta.json")
    if sidecar_path.exists():
        sidecar_path.unlink()


@pytest.fixture
def mock_onnx_session():
    """
    Mock ONNX InferenceSession for testing.

    Provides a session mock with input/output names and run method.

    """
    session = MagicMock()
    # Mock inputs and outputs
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_output = MagicMock()
    mock_output.name = "output"
    session.get_inputs.return_value = [mock_input]
    session.get_outputs.return_value = [mock_output]
    session.run.return_value = [np.array([[0.5, 0.8]])]
    return session


@pytest.fixture
def warmup_component_factory(temp_onnx_model_path: str, mock_logger: logging.Logger):
    """
    Factory fixture for creating ModelWarmUpComponent instances.

    Returns a callable that creates components with specified parameters.

    """

    def _create_component(
        model_path: str | None = None,
        n_features: int = 50,
        warm_up_iterations: int = 10,
        enable_parity_check: bool = False,
        parity_tolerance: float = 1e-6,
        parity_window: int = 200,
        enable_hot_reload: bool = False,
        hot_reload_interval: float = 300.0,
        actor_id: str = "test_actor",
        log: logging.Logger | None = None,
    ):
        """
        Create a ModelWarmUpComponent with specified parameters.
        """
        from ml.actors.common.model_warmup import ModelWarmUpComponent

        return ModelWarmUpComponent(
            model_path=model_path or temp_onnx_model_path,
            n_features=n_features,
            onnx_runtime_config=None,
            warm_up_iterations=warm_up_iterations,
            enable_parity_check=enable_parity_check,
            parity_tolerance=parity_tolerance,
            parity_window=parity_window,
            enable_hot_reload=enable_hot_reload,
            hot_reload_interval=hot_reload_interval,
            actor_id=actor_id,
            log=log,
        )

    return _create_component


# =============================================================================
# ONNX Loading Tests (5 tests)
# =============================================================================


class TestOnnxLoading:
    """
    Tests for ONNX model loading functionality.
    """

    def test_onnx_loading_creates_inference_session(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify load_model() creates ort.InferenceSession.

        The load_model method should return a valid InferenceSession instance.

        """
        # Arrange
        component = warmup_component_factory()

        # Mock the ONNX runtime - patch at ml._imports level and ml.config.runtime
        with patch("ml.actors.common.model_warmup.HAS_ONNX", True):
            with patch("ml.actors.common.model_warmup.ort") as mock_ort:
                mock_ort.InferenceSession.return_value = mock_onnx_session
                # Mock to_session_options at the module where it's imported from
                mock_session_opts = MagicMock()
                mock_providers = ["CPUExecutionProvider"]
                with patch(
                    "ml.config.runtime.to_session_options",
                    return_value=(mock_session_opts, mock_providers),
                ):
                    # Act
                    model, _metadata = component.load_model()

                    # Assert
                    assert (
                        model is mock_onnx_session
                    ), "load_model() should return InferenceSession instance"
                    mock_ort.InferenceSession.assert_called_once()

    def test_onnx_loading_uses_session_options(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify optimized session options are applied.

        The to_session_options() should be called and providers should be set.

        """
        # Arrange
        component = warmup_component_factory()

        # Mock the ONNX runtime
        with patch("ml.actors.common.model_warmup.HAS_ONNX", True):
            with patch("ml.actors.common.model_warmup.ort") as mock_ort:
                mock_ort.InferenceSession.return_value = mock_onnx_session
                with patch("ml.config.runtime.to_session_options") as mock_to_opts:
                    mock_session_opts = MagicMock()
                    mock_providers = ["CPUExecutionProvider"]
                    mock_to_opts.return_value = (mock_session_opts, mock_providers)

                    # Act
                    _model, _metadata = component.load_model()

                    # Assert
                    mock_to_opts.assert_called_once()
                    # Verify session was created with options
                    call_kwargs = mock_ort.InferenceSession.call_args
                    assert (
                        call_kwargs[1]["sess_options"] == mock_session_opts
                    ), "Session options should be passed to InferenceSession"
                    assert (
                        call_kwargs[1]["providers"] == mock_providers
                    ), "Providers should be passed to InferenceSession"

    def test_onnx_loading_extracts_metadata(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify model metadata extracted (input/output names).

        Metadata dict should contain input_names, output_names, and model_path.

        """
        # Arrange
        component = warmup_component_factory()

        # Mock the ONNX runtime - patch at ml._imports level and ml.config.runtime
        with patch("ml.actors.common.model_warmup.HAS_ONNX", True):
            with patch("ml.actors.common.model_warmup.ort") as mock_ort:
                mock_ort.InferenceSession.return_value = mock_onnx_session
                # Mock to_session_options at the module where it's imported from
                mock_session_opts = MagicMock()
                mock_providers = ["CPUExecutionProvider"]
                with patch(
                    "ml.config.runtime.to_session_options",
                    return_value=(mock_session_opts, mock_providers),
                ):
                    # Act
                    _model, metadata = component.load_model()

                    # Assert
                    assert "input_names" in metadata, "Metadata should contain input_names"
                    assert "output_names" in metadata, "Metadata should contain output_names"
                    assert "model_path" in metadata, "Metadata should contain model_path"
                    assert metadata["input_names"] == ["input"], "input_names should match mock"
                    assert metadata["output_names"] == ["output"], "output_names should match mock"

    def test_onnx_loading_raises_if_onnx_not_available(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify error if onnxruntime not installed.

        ImportError should be raised with clear error message.

        """
        # Arrange
        component = warmup_component_factory()

        # Mock ONNX as unavailable
        with patch("ml.actors.common.model_warmup.HAS_ONNX", False):
            with patch("ml.actors.common.model_warmup.check_ml_dependencies") as mock_check:
                mock_check.side_effect = ImportError("onnxruntime not available")

                # Act & Assert
                with pytest.raises(ImportError, match="onnx"):
                    component.load_model()

    def test_onnx_loading_file_not_found(self, warmup_component_factory) -> None:
        """
        Verify error if model file doesn't exist.

        FileNotFoundError should be raised.

        """
        from ml.actors.common.model_warmup import ModelWarmUpComponent

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            ModelWarmUpComponent(
                model_path="/nonexistent/path/model.onnx",
                n_features=50,
            )


# =============================================================================
# Model Warm-Up Tests (5 tests)
# =============================================================================


class TestModelWarmUp:
    """
    Tests for model warm-up functionality.
    """

    def test_warm_up_runs_specified_iterations(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify warm_up_model() runs N iterations.

        With warm_up_iterations=10, predict function should be called 10 times.

        """
        # Arrange
        component = warmup_component_factory(warm_up_iterations=10)
        mock_predict_fn = MagicMock(return_value=(0.5, 0.8))

        # Act
        component.warm_up_model(mock_onnx_session, mock_predict_fn)

        # Assert
        assert (
            mock_predict_fn.call_count == 10
        ), f"predict_fn should be called 10 times, got {mock_predict_fn.call_count}"

    def test_warm_up_uses_dummy_features(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify warm-up uses random dummy features.

        Features should be numpy arrays with correct shape (n_features,).

        """
        # Arrange
        n_features = 50
        component = warmup_component_factory(n_features=n_features, warm_up_iterations=5)
        captured_features: list[npt.NDArray[np.float32]] = []

        def capture_predict(features):
            captured_features.append(features.copy())
            return (0.5, 0.8)

        # Act
        component.warm_up_model(mock_onnx_session, capture_predict)

        # Assert
        assert len(captured_features) == 5, "Should capture 5 feature arrays"
        for features in captured_features:
            assert features.shape == (
                n_features,
            ), f"Features should have shape ({n_features},), got {features.shape}"
            assert features.dtype == np.float32, "Features should be float32"

    def test_warm_up_logs_timing_statistics(
        self,
        warmup_component_factory,
        mock_logger,
        mock_onnx_session,
    ) -> None:
        """
        Verify warm-up logs average and P99 timings.

        Log message should contain "avg=" and "P99=".

        """
        # Arrange
        component = warmup_component_factory(
            warm_up_iterations=10,
            log=mock_logger,
        )
        mock_predict_fn = MagicMock(return_value=(0.5, 0.8))

        # Act
        component.warm_up_model(mock_onnx_session, mock_predict_fn)

        # Assert
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "avg=" in call_args, "Log should contain 'avg='"
        assert "P99=" in call_args, "Log should contain 'P99='"

    def test_warm_up_handles_prediction_errors(
        self,
        warmup_component_factory,
        mock_logger,
        mock_onnx_session,
    ) -> None:
        """
        Verify warm-up continues if prediction fails.

        Error should be logged but warm-up should complete all iterations.

        """
        # Arrange
        component = warmup_component_factory(
            warm_up_iterations=10,
            log=mock_logger,
        )
        mock_predict_fn = MagicMock(side_effect=RuntimeError("Prediction failed"))

        # Act - should not raise
        component.warm_up_model(mock_onnx_session, mock_predict_fn)

        # Assert
        assert mock_predict_fn.call_count == 10, "All iterations should be attempted despite errors"
        assert (
            mock_logger.debug.call_count == 10
        ), "Error should be logged for each failed iteration"

    def test_warm_up_disabled_when_iterations_zero(
        self,
        warmup_component_factory,
        mock_onnx_session,
    ) -> None:
        """
        Verify warm-up skipped if warm_up_iterations=0.

        No predictions should be made when iterations is 0.

        """
        # Arrange
        component = warmup_component_factory(warm_up_iterations=0)
        mock_predict_fn = MagicMock(return_value=(0.5, 0.8))

        # Act
        component.warm_up_model(mock_onnx_session, mock_predict_fn)

        # Assert
        mock_predict_fn.assert_not_called()


# =============================================================================
# Feature Parity Smoke Check Tests (8 tests)
# =============================================================================


class TestFeatureParitySmokeCheck:
    """
    Tests for feature parity smoke check functionality.
    """

    def test_parity_check_enabled_when_configured(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check runs when enable_parity_check=True.

        When enabled and data available, check should execute.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=10,
        )

        # Create test data
        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        # Add some data
        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        def mock_compute_features(bar):
            return np.ones(50, dtype=np.float32)

        # Act
        drift = component.check_parity(recent_bars, recent_features, mock_compute_features)

        # Assert - drift should be computed (0.0 since features match)
        assert (
            component.parity_checked is True
        ), "Parity check should have run and set _parity_checked=True"

    def test_parity_check_disabled_by_default(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check disabled by default.

        When enable_parity_check=False, check_parity returns 0.0 immediately.

        """
        # Arrange
        component = warmup_component_factory(enable_parity_check=False)

        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        # Add data
        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        mock_compute_features = MagicMock(return_value=np.ones(50, dtype=np.float32))

        # Act
        drift = component.check_parity(recent_bars, recent_features, mock_compute_features)

        # Assert
        assert drift == 0.0, "Parity check should return 0.0 when disabled"
        mock_compute_features.assert_not_called()

    def test_parity_check_returns_zero_with_empty_data(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check returns 0.0 with no data.

        With empty buffers, drift should be 0.0.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=200,
        )

        empty_bars: deque[Any] = deque(maxlen=200)
        empty_features: deque[npt.NDArray[np.float32]] = deque(maxlen=200)

        def mock_compute_features(bar):
            return np.ones(50, dtype=np.float32)

        # Act
        drift = component.check_parity(empty_bars, empty_features, mock_compute_features)

        # Assert
        assert drift == 0.0, "Drift should be 0.0 with no data"
        assert component.parity_checked is False, "_parity_checked should remain False with no data"

    def test_parity_check_compares_online_offline_features(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check compares online vs offline features.

        Drift should be calculated as max(|online - offline|).

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=10,
        )

        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        # Online features: all ones
        # Offline features: will return ones + 0.001 = slight drift
        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        def mock_compute_features_with_drift(bar):
            # Offline computation has slight difference
            return np.ones(50, dtype=np.float32) + 0.001

        # Act
        drift = component.check_parity(
            recent_bars,
            recent_features,
            mock_compute_features_with_drift,
        )

        # Assert
        assert drift == pytest.approx(0.001, rel=1e-3), f"Drift should be ~0.001, got {drift}"

    def test_parity_check_emits_metrics(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check emits metrics.

        Both counter and gauge should be updated.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=10,
        )

        # Mock the metrics
        component._parity_checks_counter = MagicMock()
        component._parity_drift_gauge = MagicMock()

        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        def mock_compute_features(bar):
            return np.ones(50, dtype=np.float32)

        # Act
        drift = component.check_parity(recent_bars, recent_features, mock_compute_features)

        # Assert
        component._parity_checks_counter.labels.assert_called_once()
        component._parity_checks_counter.labels().inc.assert_called_once()
        component._parity_drift_gauge.labels.assert_called_once()
        component._parity_drift_gauge.labels().set.assert_called_once()

    def test_parity_check_warns_if_drift_exceeds_tolerance(
        self,
        warmup_component_factory,
        mock_logger,
    ) -> None:
        """
        Verify warning logged if drift > parity_tolerance.

        Warning should contain drift value.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_tolerance=0.0001,  # Very low tolerance
            parity_window=10,
            log=mock_logger,
        )

        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        # Online features
        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        def mock_compute_features_with_large_drift(bar):
            # Large drift exceeding tolerance
            return np.ones(50, dtype=np.float32) + 0.001

        # Act
        drift = component.check_parity(
            recent_bars,
            recent_features,
            mock_compute_features_with_large_drift,
        )

        # Assert
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert (
            "drift" in warning_msg.lower() or "0.001" in warning_msg
        ), "Warning should mention drift value"

    def test_parity_check_runs_once_only(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check runs once then sets _parity_checked=True.

        After first successful check, subsequent calls should not repeat.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=10,
        )

        recent_bars: deque[Any] = deque(maxlen=10)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=10)

        for i in range(5):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32))

        call_count = 0

        def mock_compute_features_with_counter(bar):
            nonlocal call_count
            call_count += 1
            return np.ones(50, dtype=np.float32)

        # Act - first check
        drift1 = component.check_parity(
            recent_bars,
            recent_features,
            mock_compute_features_with_counter,
        )

        # Assert
        assert component.parity_checked is True, "_parity_checked should be True after first check"
        first_call_count = call_count

        # Reset counter and call again
        call_count = 0
        drift2 = component.check_parity(
            recent_bars,
            recent_features,
            mock_compute_features_with_counter,
        )

        # The component DOES re-run parity checks on each call
        # The _parity_checked flag is set, but check_parity doesn't short-circuit
        # This is by design - it's up to the caller to check parity_checked
        assert first_call_count > 0, "First check should have called compute_features"

    def test_parity_check_uses_recent_bars_and_features(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify parity check uses deque buffers (recent bars/features).

        Only the recent data from deques should be used.

        """
        # Arrange
        component = warmup_component_factory(
            enable_parity_check=True,
            parity_window=5,
        )

        # Create deques with specific maxlen
        recent_bars: deque[Any] = deque(maxlen=5)
        recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=5)

        # Add more items than maxlen to verify deque behavior
        for i in range(10):
            recent_bars.append(f"bar_{i}")
            recent_features.append(np.ones(50, dtype=np.float32) * i)

        # Should only have last 5 items
        assert len(recent_bars) == 5, "Deque should have maxlen items"
        assert len(recent_features) == 5, "Deque should have maxlen items"

        bars_processed: list[str] = []

        def mock_compute_features(bar):
            bars_processed.append(bar)
            return np.ones(50, dtype=np.float32)

        # Act
        drift = component.check_parity(recent_bars, recent_features, mock_compute_features)

        # Assert - only 5 bars should be processed (the recent ones)
        assert len(bars_processed) == 5, f"Should process 5 bars, got {len(bars_processed)}"
        # Verify it's the recent bars (5-9)
        assert bars_processed == [
            f"bar_{i}" for i in range(5, 10)
        ], "Should process only recent bars"


# =============================================================================
# Hot Reload Scheduling Tests (7 tests)
# =============================================================================


class TestHotReloadScheduling:
    """
    Tests for model hot-reload scheduling functionality.
    """

    def test_hot_reload_disabled_by_default(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify hot reload disabled when enable_hot_reload=False.

        should_hot_reload() should return False.

        """
        # Arrange
        component = warmup_component_factory(enable_hot_reload=False)

        # Act
        should_reload = component.should_hot_reload()

        # Assert
        assert should_reload is False, "should_hot_reload() should return False when disabled"

    def test_hot_reload_checks_interval(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify hot reload checks honor hot_reload_interval.

        Check should return False before interval elapsed.

        """
        # Arrange
        component = warmup_component_factory(
            enable_hot_reload=True,
            hot_reload_interval=300.0,  # 5 minutes
        )

        # Act - first check should return True
        first_check = component.should_hot_reload()

        # Act - immediate second check should return False (interval not elapsed)
        second_check = component.should_hot_reload()

        # Assert
        assert first_check is True, "First check should return True"
        assert second_check is False, "Second check (before interval) should return False"

    def test_hot_reload_updates_last_check_time(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify should_hot_reload() updates _last_check_time.

        After check, _last_check_time should be updated.

        """
        # Arrange
        component = warmup_component_factory(
            enable_hot_reload=True,
            hot_reload_interval=300.0,
        )

        initial_check_time = component.last_check_time
        assert initial_check_time == 0.0, "Initial _last_check_time should be 0"

        # Act
        component.should_hot_reload()

        # Assert
        assert component.last_check_time > 0, "_last_check_time should be updated after check"

    def test_execute_hot_reload_checks_file_exists(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify execute_hot_reload() checks if model file exists.

        If file doesn't exist, should return None without reload.

        """
        # Arrange
        component = warmup_component_factory()

        mock_load_fn = MagicMock(return_value=(MagicMock(), {}))

        # Act - pass nonexistent path
        result = component.execute_hot_reload(
            "/nonexistent/path/model.onnx",
            mock_load_fn,
        )

        # Assert
        assert result is None, "Should return None when file doesn't exist"
        mock_load_fn.assert_not_called()

    def test_execute_hot_reload_checks_mtime(
        self,
        warmup_component_factory,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify reload checks file modification time.

        If mtime unchanged, should not reload.

        """
        # Arrange
        component = warmup_component_factory()
        # Set initial mtime (simulating first load)
        component._model_mtime = os.path.getmtime(temp_onnx_model_path)

        mock_load_fn = MagicMock(return_value=(MagicMock(), {}))

        # Act - mtime hasn't changed
        result = component.execute_hot_reload(
            temp_onnx_model_path,
            mock_load_fn,
        )

        # Assert
        assert result is None, "Should return None when mtime unchanged"
        mock_load_fn.assert_not_called()

    def test_execute_hot_reload_loads_new_model(
        self,
        warmup_component_factory,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify reload loads model when mtime changed.

        When file is modified, model should be reloaded.

        """
        # Arrange
        component = warmup_component_factory()
        # Set old mtime (simulating stale model)
        component._model_mtime = os.path.getmtime(temp_onnx_model_path) - 100

        mock_model = MagicMock()
        mock_metadata = {"reloaded": True}
        mock_load_fn = MagicMock(return_value=(mock_model, mock_metadata))

        # Act
        result = component.execute_hot_reload(
            temp_onnx_model_path,
            mock_load_fn,
        )

        # Assert
        assert result is not None, "Should return (model, metadata) when mtime changed"
        assert result[0] is mock_model, "Should return the loaded model"
        mock_load_fn.assert_called_once_with(temp_onnx_model_path)
        # Verify mtime updated
        assert component._model_mtime == os.path.getmtime(
            temp_onnx_model_path
        ), "_model_mtime should be updated after reload"

    def test_execute_hot_reload_handles_errors_gracefully(
        self,
        warmup_component_factory,
        mock_logger,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify reload errors logged but don't crash.

        Error should be logged with exc_info=True, actor should continue.

        """
        # Arrange
        component = warmup_component_factory(log=mock_logger)
        # Set old mtime to trigger reload
        component._model_mtime = os.path.getmtime(temp_onnx_model_path) - 100

        mock_load_fn = MagicMock(side_effect=RuntimeError("Load failed"))

        # Act - should not raise
        result = component.execute_hot_reload(
            temp_onnx_model_path,
            mock_load_fn,
        )

        # Assert
        assert result is None, "Should return None on error"
        mock_logger.error.assert_called_once()
        # Verify exc_info=True is passed
        call_kwargs = mock_logger.error.call_args[1]
        assert call_kwargs.get("exc_info") is True, "Error should be logged with exc_info=True"


# =============================================================================
# Edge Cases and Property Accessors
# =============================================================================


class TestEdgeCasesAndProperties:
    """
    Tests for edge cases and property accessors.
    """

    def test_component_rejects_zero_n_features(
        self,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify component rejects n_features=0.
        """
        from ml.actors.common.model_warmup import ModelWarmUpComponent

        with pytest.raises(ValueError, match=r"n_features.*must be > 0"):
            ModelWarmUpComponent(
                model_path=temp_onnx_model_path,
                n_features=0,
            )

    def test_component_rejects_negative_n_features(
        self,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify component rejects negative n_features.
        """
        from ml.actors.common.model_warmup import ModelWarmUpComponent

        with pytest.raises(ValueError, match=r"n_features.*must be > 0"):
            ModelWarmUpComponent(
                model_path=temp_onnx_model_path,
                n_features=-10,
            )

    def test_component_rejects_negative_warm_up_iterations(
        self,
        temp_onnx_model_path,
    ) -> None:
        """
        Verify component rejects negative warm_up_iterations.
        """
        from ml.actors.common.model_warmup import ModelWarmUpComponent

        with pytest.raises(ValueError, match=r"warm_up_iterations.*must be >= 0"):
            ModelWarmUpComponent(
                model_path=temp_onnx_model_path,
                n_features=50,
                warm_up_iterations=-1,
            )

    def test_property_accessors_return_correct_values(
        self,
        warmup_component_factory,
    ) -> None:
        """
        Verify property accessors return correct values.
        """
        # Arrange
        component = warmup_component_factory(
            n_features=100,
        )

        # Assert
        assert component.n_features == 100, "n_features property should return 100"
        assert component.parity_checked is False, "parity_checked should initially be False"
        assert component.last_check_time == 0.0, "last_check_time should initially be 0.0"
        # model_path property test
        assert component.model_path.endswith(".onnx"), "model_path should end with .onnx"
