"""
Unit tests for MLflowTrackingComponent.

This module tests the MLflow tracking component extracted from BaseMLTrainer
(lines 483-491 and 1071-1123). Tests verify:
- MLflow enablement check (_should_use_mlflow)
- MLflow run lifecycle (_start_mlflow_run, _end_mlflow_run)
- Metrics tracking (_track_with_mlflow)
- Configuration serialization (_config_to_dict)

Following the established test patterns for training components.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml._imports import HAS_MLFLOW
from ml.training.common.mlflow_tracking import (
    MLflowTrackingComponent,
    MLflowTrainerProtocol,
)


# ============================================================================
# Mock Configuration Classes
# ============================================================================


@dataclass
class MockMLflowConfig:
    """Mock MLflow configuration for testing."""

    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None


@dataclass
class MockConfig:
    """Mock training configuration for MLflow testing."""

    mlflow_config: MockMLflowConfig | None = None
    learning_rate: float = 0.01
    max_depth: int = 5
    objective: str = "binary:logistic"


class MinimalConfig:
    """Minimal config without mlflow_config attribute."""

    learning_rate: float = 0.05


# ============================================================================
# Testable Trainer Implementation
# ============================================================================


class TestableTrainer:
    """
    Concrete trainer implementation for testing MLflowTrackingComponent.

    Implements the MLflowTrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | MinimalConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._mlflow_run_id: str | None = None
        self._call_log: list[str] = []

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_info."""
        self._call_log.append(f"info: {message}")
        logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_warning."""
        self._call_log.append(f"warning: {message}")
        logging.warning(message, *args, **kwargs)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create basic TestableTrainer instance without MLflow configured."""
    config = MockConfig(mlflow_config=None)
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_mlflow_fixture() -> TestableTrainer:
    """Create TestableTrainer with MLflow enabled."""
    mlflow_config = MockMLflowConfig(
        enabled=True,
        tracking_uri="http://localhost:5000",
        experiment_name="test_experiment",
        run_name="test_run",
    )
    config = MockConfig(mlflow_config=mlflow_config)
    return TestableTrainer(config)


@pytest.fixture
def trainer_minimal_config_fixture() -> TestableTrainer:
    """Create TestableTrainer with config missing mlflow_config attribute."""
    return TestableTrainer(MinimalConfig())


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    """Sample metrics dictionary for testing."""
    return {
        "accuracy": 0.95,
        "loss": 0.05,
        "epoch_losses": [0.5, 0.3, 0.2, 0.1],
        "feature_names": ["a", "b", "c"],  # non-numeric, should be skipped
        "training_time": 123.45,
    }


# ============================================================================
# Happy Path Tests - _should_use_mlflow
# ============================================================================


class TestShouldUseMlflow:
    """Tests for _should_use_mlflow method."""

    def test_should_use_mlflow_returns_false_when_has_mlflow_is_false(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify MLflow disabled when HAS_MLFLOW is False."""
        # HAS_MLFLOW is False by default (MLflow is deprecated)
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        # Since HAS_MLFLOW=False in ml/_imports.py, this should return False
        result = mlflow_component._should_use_mlflow()

        # HAS_MLFLOW is False by default, so even with config, should be False
        assert result is False or HAS_MLFLOW  # True only if HAS_MLFLOW is actually True

    def test_should_use_mlflow_returns_false_no_config(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify MLflow disabled without config."""
        mlflow_component = MLflowTrackingComponent(trainer_fixture)
        assert mlflow_component._should_use_mlflow() is False

    def test_should_use_mlflow_returns_false_missing_attr(
        self,
        trainer_minimal_config_fixture: TestableTrainer,
    ) -> None:
        """Verify MLflow disabled when mlflow_config attribute is missing."""
        mlflow_component = MLflowTrackingComponent(trainer_minimal_config_fixture)
        assert mlflow_component._should_use_mlflow() is False

    def test_should_use_mlflow_with_has_mlflow_true(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify MLflow enabled when HAS_MLFLOW=True and config present."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            result = mlflow_component._should_use_mlflow()

        assert result is True

    def test_should_use_mlflow_returns_false_config_none(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify MLflow disabled when mlflow_config is None."""
        mlflow_component = MLflowTrackingComponent(trainer_fixture)

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            result = mlflow_component._should_use_mlflow()

        # mlflow_config is None in trainer_fixture
        assert result is False


# ============================================================================
# Happy Path Tests - _start_mlflow_run
# ============================================================================


class TestStartMlflowRun:
    """Tests for _start_mlflow_run method."""

    def test_start_mlflow_run_does_nothing_when_not_available(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify no action when MLflow not available."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        # HAS_MLFLOW is False by default
        mlflow_component._start_mlflow_run()

        # Run ID should remain None
        assert trainer_with_mlflow_fixture._mlflow_run_id is None

    def test_start_mlflow_run_sets_tracking_uri(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify tracking URI is set from config."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._start_mlflow_run()

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_start_mlflow_run_sets_experiment_name(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify experiment name is set from config."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._start_mlflow_run()

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_start_mlflow_run_stores_run_id(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify run ID is stored on trainer."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-456"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._start_mlflow_run()

        assert trainer_with_mlflow_fixture._mlflow_run_id == "test-run-id-456"

    def test_start_mlflow_run_logs_params(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify config params are logged."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._start_mlflow_run()

        # Should log params from config_to_dict
        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert "learning_rate" in logged_params
        assert logged_params["learning_rate"] == 0.01


# ============================================================================
# Happy Path Tests - _track_with_mlflow
# ============================================================================


class TestTrackWithMlflow:
    """Tests for _track_with_mlflow method."""

    def test_track_with_mlflow_does_nothing_when_not_available(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify no action when MLflow not available."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        # No error should be raised
        mlflow_component._track_with_mlflow(sample_metrics)

    def test_track_with_mlflow_does_nothing_without_run_id(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify no action when no run is active."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow(sample_metrics)

        # log_metric should not be called since run_id is None
        mock_mlflow.log_metric.assert_not_called()

    def test_track_with_mlflow_logs_scalar_metrics(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify scalar metrics are logged."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow(sample_metrics)

        # Check scalar metrics were logged
        calls = mock_mlflow.log_metric.call_args_list
        logged_keys = [call[0][0] for call in calls]

        assert "accuracy" in logged_keys
        assert "loss" in logged_keys
        assert "training_time" in logged_keys

    def test_track_with_mlflow_logs_list_metrics_indexed(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify list metrics are logged with indices."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow(sample_metrics)

        # Check indexed list metrics
        calls = mock_mlflow.log_metric.call_args_list
        logged_keys = [call[0][0] for call in calls]

        assert "epoch_losses_0" in logged_keys
        assert "epoch_losses_1" in logged_keys
        assert "epoch_losses_2" in logged_keys
        assert "epoch_losses_3" in logged_keys

    def test_track_with_mlflow_skips_non_numeric(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify non-numeric values are skipped."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow(sample_metrics)

        # Check non-numeric "feature_names" was not logged
        calls = mock_mlflow.log_metric.call_args_list
        logged_keys = [call[0][0] for call in calls]

        assert "feature_names" not in logged_keys
        assert not any(k.startswith("feature_names") for k in logged_keys)


# ============================================================================
# Happy Path Tests - _end_mlflow_run
# ============================================================================


class TestEndMlflowRun:
    """Tests for _end_mlflow_run method."""

    def test_end_mlflow_run_does_nothing_when_not_available(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify no action when MLflow not available."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        # No error should be raised
        mlflow_component._end_mlflow_run()

    def test_end_mlflow_run_does_nothing_without_run_id(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify no action when no run is active."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._end_mlflow_run()

        mock_mlflow.end_run.assert_not_called()

    def test_end_mlflow_run_calls_end_run(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify end_run is called when run is active."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._end_mlflow_run()

        mock_mlflow.end_run.assert_called_once()


# ============================================================================
# Happy Path Tests - _config_to_dict
# ============================================================================


class TestConfigToDict:
    """Tests for _config_to_dict method."""

    def test_config_to_dict_extracts_scalars(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify scalar values are extracted from config."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        config_dict = mlflow_component._config_to_dict()

        assert "learning_rate" in config_dict
        assert config_dict["learning_rate"] == 0.01
        assert "max_depth" in config_dict
        assert config_dict["max_depth"] == 5
        assert "objective" in config_dict
        assert config_dict["objective"] == "binary:logistic"

    def test_config_to_dict_excludes_complex_objects(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify complex objects are excluded."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        config_dict = mlflow_component._config_to_dict()

        # mlflow_config should not be in the dict (it's a complex object)
        assert "mlflow_config" not in config_dict

    def test_config_to_dict_handles_minimal_config(
        self,
        trainer_minimal_config_fixture: TestableTrainer,
    ) -> None:
        """Verify minimal config is handled."""
        mlflow_component = MLflowTrackingComponent(trainer_minimal_config_fixture)

        config_dict = mlflow_component._config_to_dict()

        # MinimalConfig has learning_rate as class attribute
        assert isinstance(config_dict, dict)

    def test_config_to_dict_returns_empty_for_no_scalars(self) -> None:
        """Verify empty dict returned when no scalar attributes."""

        class ComplexOnlyConfig:
            complex_attr: dict[str, Any] = {}
            list_attr: list[int] = []

        trainer = TestableTrainer(ComplexOnlyConfig())  # type: ignore[arg-type]
        mlflow_component = MLflowTrackingComponent(trainer)

        config_dict = mlflow_component._config_to_dict()

        # Should be empty since no scalar attributes
        assert config_dict == {}

    def test_config_to_dict_includes_bool_values(self) -> None:
        """Verify boolean values are included."""

        @dataclass
        class ConfigWithBool:
            use_feature: bool = True
            verbose: bool = False

        trainer = TestableTrainer(ConfigWithBool())  # type: ignore[arg-type]
        mlflow_component = MLflowTrackingComponent(trainer)

        config_dict = mlflow_component._config_to_dict()

        assert "use_feature" in config_dict
        assert config_dict["use_feature"] is True
        assert "verbose" in config_dict
        assert config_dict["verbose"] is False


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestErrorConditions:
    """Tests for error conditions and edge cases."""

    def test_start_mlflow_run_handles_import_error(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify graceful handling of MLflow import error."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            # Patch the import inside the method to raise ImportError
            with patch.dict("sys.modules", {"mlflow": None}):
                # This should not raise but may log a warning
                mlflow_component._start_mlflow_run()

        # Run ID should remain None
        assert trainer_with_mlflow_fixture._mlflow_run_id is None

    def test_track_with_mlflow_handles_empty_metrics(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify empty metrics dict handled gracefully."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow({})

        mock_mlflow.log_metric.assert_not_called()

    def test_track_with_mlflow_handles_empty_list(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify empty list in metrics handled gracefully."""
        trainer_with_mlflow_fixture._mlflow_run_id = "active-run-id"
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        metrics = {"empty_list": [], "scalar": 1.0}

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._track_with_mlflow(metrics)

        # Only scalar should be logged
        calls = mock_mlflow.log_metric.call_args_list
        logged_keys = [call[0][0] for call in calls]

        assert "scalar" in logged_keys
        assert "empty_list" not in logged_keys


# ============================================================================
# Integration Tests - Full Lifecycle
# ============================================================================


class TestFullLifecycle:
    """Tests for complete MLflow tracking lifecycle."""

    def test_full_lifecycle_with_mocked_mlflow(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_metrics: dict[str, Any],
    ) -> None:
        """Verify complete lifecycle: start -> track -> end."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "lifecycle-test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                # Start
                mlflow_component._start_mlflow_run()
                assert trainer_with_mlflow_fixture._mlflow_run_id == "lifecycle-test-run-id"

                # Track
                mlflow_component._track_with_mlflow(sample_metrics)
                assert mock_mlflow.log_metric.called

                # End
                mlflow_component._end_mlflow_run()
                mock_mlflow.end_run.assert_called_once()

    def test_lifecycle_logs_info_messages(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
    ) -> None:
        """Verify info messages logged during lifecycle."""
        mlflow_component = MLflowTrackingComponent(trainer_with_mlflow_fixture)

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "log-test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        with patch("ml.training.common.mlflow_tracking.HAS_MLFLOW", True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                mlflow_component._start_mlflow_run()
                mlflow_component._end_mlflow_run()

        # Check info logs were recorded
        info_logs = [log for log in trainer_with_mlflow_fixture._call_log if log.startswith("info:")]
        assert len(info_logs) >= 2  # Start and end messages


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


class TestProtocolConformance:
    """Tests verifying protocol conformance."""

    def test_testable_trainer_conforms_to_protocol(self) -> None:
        """Verify TestableTrainer implements MLflowTrainerProtocol."""
        trainer = TestableTrainer()

        # Protocol requires _config and _mlflow_run_id attributes
        assert hasattr(trainer, "_config")
        assert hasattr(trainer, "_mlflow_run_id")

        # Protocol requires _log_info and _log_warning methods
        assert hasattr(trainer, "_log_info")
        assert hasattr(trainer, "_log_warning")
        assert callable(trainer._log_info)
        assert callable(trainer._log_warning)

    def test_component_initializes_with_protocol_conforming_trainer(self) -> None:
        """Verify component accepts any protocol-conforming trainer."""
        trainer = TestableTrainer()

        # Should not raise
        component = MLflowTrackingComponent(trainer)

        assert component._trainer is trainer


__all__ = [
    "MinimalConfig",
    "MockConfig",
    "MockMLflowConfig",
    "TestableTrainer",
]
