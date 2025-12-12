"""
Unit tests for PersistenceComponent.

This module tests the persistence component extracted from BaseMLTrainer
(lines 1125-1161, 1265-1494). Tests verify:
- ONNX model export
- Model saving with registry integration
- Model manifest creation
- Model loading from registry or file
- Feature importance extraction
- Logging helpers
- Error handling

Following the test design for Phase 3.8.7 component extraction.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.training.common.persistence import (
    PersistenceComponent,
    PersistenceTrainerProtocol,
)


# ============================================================================
# Mock Model and Trainer Fixtures
# ============================================================================


class MockModel:
    """Mock model for testing with feature importance support."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.feature_importances_ = np.array([0.3, 0.5, 0.2])

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> MockModel:
        return self


class MockModelWithGetScore:
    """Mock model with XGBoost-style get_score method."""

    def __init__(self) -> None:
        self._scores: dict[str, float] = {"f0": 0.3, "f1": 0.5, "f2": 0.2}

    def get_score(self, importance_type: str = "gain") -> dict[str, float]:
        return self._scores


class MockModelNoImportance:
    """Mock model without feature importance support."""



@dataclass
class MockConfig:
    """Mock training configuration for testing."""

    data_source: str = "memory"
    target_column: str = "target"
    train_test_split: float = 0.8
    save_model_path: str | None = None
    registry_path: Path | None = None
    export_onnx: bool = False
    auto_deploy: bool = False
    model_role: str = "inference"
    data_requirements: str = "l1_only"
    model_version: str = "1.0.0"
    feature_dtypes: list[str] | None = None
    pipeline_signature: str = ""
    max_inference_latency_ms: float = 50.0
    memory_limit_mb: float = 1024.0
    feature_set_id: str | None = None
    pipeline_version: str | None = None
    decision_policy: str | None = None
    decision_config: dict[str, Any] | None = None
    parent_model_id: str | None = None


class TestableTrainer:
    """
    Concrete trainer implementation for testing PersistenceComponent.

    Implements the PersistenceTrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._feature_names: list[str] = ["f0", "f1", "f2"]
        self._training_metrics: dict[str, Any] = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "training_time": 10.5,
        }
        self._is_fitted = False
        self._model: Any = None
        self._call_log: list[str] = []

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """Mock ONNX conversion."""
        self._call_log.append("_convert_to_onnx")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock_onnx_model_bytes")

    def _config_to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        self._call_log.append("_config_to_dict")
        result: dict[str, Any] = {}
        for key, value in vars(self._config).items():
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
        return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create a basic TestableTrainer instance."""
    trainer = TestableTrainer(MockConfig())
    trainer._model = MockModel()
    trainer._is_fitted = True
    return trainer


@pytest.fixture
def trainer_not_fitted_fixture() -> TestableTrainer:
    """Create a TestableTrainer that has not been fitted."""
    return TestableTrainer(MockConfig())


@pytest.fixture
def trainer_with_get_score_fixture() -> TestableTrainer:
    """Create TestableTrainer with XGBoost-style model."""
    trainer = TestableTrainer(MockConfig())
    trainer._model = MockModelWithGetScore()
    trainer._is_fitted = True
    return trainer


@pytest.fixture
def trainer_no_importance_fixture() -> TestableTrainer:
    """Create TestableTrainer with model lacking feature importance."""
    trainer = TestableTrainer(MockConfig())
    trainer._model = MockModelNoImportance()
    trainer._is_fitted = True
    return trainer


@pytest.fixture
def persistence_fixture(trainer_fixture: TestableTrainer) -> PersistenceComponent:
    """Create PersistenceComponent with fitted trainer."""
    return PersistenceComponent(trainer_fixture)


# ============================================================================
# ONNX Export Tests
# ============================================================================


class TestExportToOnnx:
    """Tests for ONNX export functionality."""

    def test_export_to_onnx_success(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify successful ONNX export when model is fitted."""
        persistence = PersistenceComponent(trainer_fixture)
        onnx_path = tmp_path / "model.onnx"

        with patch("ml.training.common.persistence.HAS_ONNX", True):
            persistence.export_to_onnx(onnx_path)

        # Verify _convert_to_onnx was called
        assert "_convert_to_onnx" in trainer_fixture._call_log

        # Verify file was created
        assert onnx_path.exists()

    def test_export_to_onnx_creates_parent_dirs(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify parent directories are created if missing."""
        persistence = PersistenceComponent(trainer_fixture)
        onnx_path = tmp_path / "nested" / "dir" / "model.onnx"

        with patch("ml.training.common.persistence.HAS_ONNX", True):
            persistence.export_to_onnx(onnx_path)

        assert onnx_path.parent.exists()
        assert onnx_path.exists()

    def test_export_to_onnx_raises_when_not_fitted(
        self,
        trainer_not_fitted_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify error when model is not fitted."""
        persistence = PersistenceComponent(trainer_not_fitted_fixture)
        onnx_path = tmp_path / "model.onnx"

        with pytest.raises(ValueError, match="Model must be fitted"):
            persistence.export_to_onnx(onnx_path)

    def test_export_to_onnx_checks_onnx_dependency(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify ONNX dependency check when HAS_ONNX is False."""
        persistence = PersistenceComponent(trainer_fixture)
        onnx_path = tmp_path / "model.onnx"

        with patch("ml.training.common.persistence.HAS_ONNX", False):
            with patch(
                "ml.training.common.persistence.check_ml_dependencies"
            ) as mock_check:
                mock_check.side_effect = ImportError("onnx not installed")
                with pytest.raises(ImportError, match="onnx not installed"):
                    persistence.export_to_onnx(onnx_path)

    def test_export_to_onnx_accepts_string_path(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify ONNX export accepts string path."""
        persistence = PersistenceComponent(trainer_fixture)
        onnx_path = str(tmp_path / "model.onnx")

        with patch("ml.training.common.persistence.HAS_ONNX", True):
            persistence.export_to_onnx(onnx_path)

        assert Path(onnx_path).exists()


# ============================================================================
# Model Save Tests
# ============================================================================


class TestSaveModel:
    """Tests for model saving functionality."""

    def test_save_model_raises_when_not_fitted(
        self,
        trainer_not_fitted_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify error when saving unfitted model."""
        persistence = PersistenceComponent(trainer_not_fitted_fixture)
        model_path = tmp_path / "model"

        with pytest.raises(ValueError, match="Model must be fitted"):
            persistence.save_model(model_path)

    def test_save_model_creates_parent_dirs(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify parent directories are created."""
        persistence = PersistenceComponent(trainer_fixture)
        model_path = tmp_path / "nested" / "dir" / "model"

        # Mock the registry and export dependencies at the source module
        with patch("ml.registry.ModelRegistry") as mock_registry:
            with patch("ml.training.export.save_model_with_metadata") as mock_save:
                mock_save.return_value = model_path.with_suffix(".onnx")
                mock_registry.return_value.register_model.return_value = "model_123"

                persistence.save_model(model_path)

        assert model_path.parent.exists()

    def test_save_model_calls_registry(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify model is registered with ModelRegistry."""
        persistence = PersistenceComponent(trainer_fixture)
        model_path = tmp_path / "model"

        with patch("ml.registry.ModelRegistry") as mock_registry:
            with patch("ml.training.export.save_model_with_metadata") as mock_save:
                mock_save.return_value = model_path.with_suffix(".onnx")
                mock_registry_instance = MagicMock()
                mock_registry_instance.register_model.return_value = "model_123"
                mock_registry.return_value = mock_registry_instance

                persistence.save_model(model_path)

        mock_registry_instance.register_model.assert_called_once()

    def test_save_model_uses_config_registry_path(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify registry path from config is used."""
        registry_path = tmp_path / "custom_registry"
        config = MockConfig(registry_path=registry_path)
        trainer = TestableTrainer(config)
        trainer._model = MockModel()
        trainer._is_fitted = True
        persistence = PersistenceComponent(trainer)

        model_path = tmp_path / "model"

        with patch("ml.registry.ModelRegistry") as mock_registry:
            with patch("ml.training.export.save_model_with_metadata") as mock_save:
                mock_save.return_value = model_path.with_suffix(".onnx")
                mock_registry.return_value.register_model.return_value = "model_123"

                persistence.save_model(model_path)

        mock_registry.assert_called_with(registry_path)


# ============================================================================
# Model Manifest Creation Tests
# ============================================================================


class TestCreateModelManifest:
    """Tests for model manifest creation."""

    def test_create_manifest_includes_feature_schema(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify manifest includes feature schema."""
        persistence = PersistenceComponent(trainer_fixture)
        save_path = tmp_path / "model.onnx"

        with patch("ml.registry.ModelManifest") as mock_manifest:
            with patch(
                "ml.registry.feature_registry.compute_schema_hash"
            ) as mock_hash:
                mock_hash.return_value = "hash123"
                mock_manifest.return_value = MagicMock()

                persistence._create_model_manifest(save_path)

        # Verify feature_schema was passed to ModelManifest
        call_kwargs = mock_manifest.call_args[1]
        assert "feature_schema" in call_kwargs
        assert call_kwargs["feature_schema"] == {"f0": "float32", "f1": "float32", "f2": "float32"}

    def test_create_manifest_includes_performance_metrics(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify manifest includes performance metrics."""
        persistence = PersistenceComponent(trainer_fixture)
        save_path = tmp_path / "model.onnx"

        with patch("ml.registry.ModelManifest") as mock_manifest:
            with patch(
                "ml.registry.feature_registry.compute_schema_hash"
            ) as mock_hash:
                mock_hash.return_value = "hash123"
                mock_manifest.return_value = MagicMock()

                persistence._create_model_manifest(save_path)

        call_kwargs = mock_manifest.call_args[1]
        assert "performance_metrics" in call_kwargs
        metrics = call_kwargs["performance_metrics"]
        assert metrics["accuracy"] == 0.85
        assert metrics["f1_score"] == 0.82

    def test_create_manifest_determines_serveable_for_onnx(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify serveable=True for ONNX files."""
        persistence = PersistenceComponent(trainer_fixture)
        save_path = tmp_path / "model.onnx"

        with patch("ml.registry.ModelManifest") as mock_manifest:
            with patch(
                "ml.registry.feature_registry.compute_schema_hash"
            ) as mock_hash:
                mock_hash.return_value = "hash123"
                mock_manifest.return_value = MagicMock()

                persistence._create_model_manifest(save_path)

        call_kwargs = mock_manifest.call_args[1]
        assert call_kwargs["serveable"] is True
        assert call_kwargs["artifact_format"] == "onnx"

    def test_create_manifest_uses_trainer_class_for_architecture(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify architecture is derived from trainer class name."""
        persistence = PersistenceComponent(trainer_fixture)
        save_path = tmp_path / "model.onnx"

        with patch("ml.registry.ModelManifest") as mock_manifest:
            with patch(
                "ml.registry.feature_registry.compute_schema_hash"
            ) as mock_hash:
                mock_hash.return_value = "hash123"
                mock_manifest.return_value = MagicMock()

                persistence._create_model_manifest(save_path)

        call_kwargs = mock_manifest.call_args[1]
        # TestableTrainer -> Testable (removes "Trainer" suffix)
        assert call_kwargs["architecture"] == "Testable"


# ============================================================================
# Model Load Tests
# ============================================================================


class TestLoadModel:
    """Tests for model loading functionality."""

    def test_load_model_from_registry_by_id(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify loading model by registry ID."""
        trainer_fixture._is_fitted = False
        trainer_fixture._model = None
        persistence = PersistenceComponent(trainer_fixture)

        mock_model_info = MagicMock()
        mock_model_info.manifest.feature_schema = {"f0": "float32", "f1": "float32"}
        mock_model_info.manifest.performance_metrics = {"accuracy": 0.9}

        with patch("ml.registry.ModelRegistry") as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model.return_value = mock_model_info
            mock_registry_instance.load_model.return_value = MockModel()
            mock_registry.return_value = mock_registry_instance

            persistence.load_model("model_abc123")

        assert trainer_fixture._is_fitted is True
        assert trainer_fixture._model is not None
        assert trainer_fixture._feature_names == ["f0", "f1"]

    def test_load_model_raises_when_id_not_found(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify error when model ID not in registry."""
        persistence = PersistenceComponent(trainer_fixture)

        with patch("ml.registry.ModelRegistry") as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model.return_value = None
            mock_registry.return_value = mock_registry_instance

            with pytest.raises(ValueError, match="Model ID not found"):
                persistence.load_model("nonexistent_id")

    def test_load_model_raises_when_load_fails(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify error when model loading fails."""
        persistence = PersistenceComponent(trainer_fixture)

        mock_model_info = MagicMock()
        mock_model_info.manifest.feature_schema = {}
        mock_model_info.manifest.performance_metrics = {}

        with patch("ml.registry.ModelRegistry") as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model.return_value = mock_model_info
            mock_registry_instance.load_model.return_value = None
            mock_registry.return_value = mock_registry_instance

            with pytest.raises(RuntimeError, match="Failed to load model"):
                persistence.load_model("model_xyz")

    def test_load_model_from_file_path(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify loading model from file path."""
        trainer_fixture._is_fitted = False
        trainer_fixture._model = None
        persistence = PersistenceComponent(trainer_fixture)

        # Create a mock model file
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"mock_onnx_bytes")

        # Mock both ModelRegistry and ProductionModelLoader
        with patch("ml.registry.model_registry.ModelRegistry") as mock_registry:
            # Ensure ModelRegistry doesn't interfere
            mock_registry.return_value = MagicMock()

            with patch("ml.actors.base.ProductionModelLoader") as mock_loader:
                mock_loader_instance = MagicMock()
                mock_loader_instance.load_model.return_value = (
                    MockModel(),
                    {"feature_names": ["a", "b"], "training_metrics": {"loss": 0.1}},
                )
                mock_loader.return_value = mock_loader_instance

                persistence.load_model(str(model_path))

        assert trainer_fixture._is_fitted is True
        assert trainer_fixture._feature_names == ["a", "b"]

    def test_load_model_raises_when_file_not_found(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
    ) -> None:
        """Verify error when file does not exist."""
        persistence = PersistenceComponent(trainer_fixture)
        model_path = tmp_path / "nonexistent.onnx"

        with patch("ml.registry.ModelRegistry") as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry.return_value = mock_registry_instance

            with pytest.raises(FileNotFoundError, match="Model file not found"):
                persistence.load_model(str(model_path))


# ============================================================================
# Feature Importance Tests
# ============================================================================


class TestGetFeatureImportance:
    """Tests for feature importance extraction."""

    def test_get_feature_importance_with_sklearn_model(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify feature importance from sklearn-style model."""
        persistence = PersistenceComponent(trainer_fixture)

        importance = persistence.get_feature_importance()

        assert importance is not None
        assert len(importance) == 3
        assert importance["f0"] == pytest.approx(0.3)
        assert importance["f1"] == pytest.approx(0.5)
        assert importance["f2"] == pytest.approx(0.2)

    def test_get_feature_importance_with_xgboost_model(
        self,
        trainer_with_get_score_fixture: TestableTrainer,
    ) -> None:
        """Verify feature importance from XGBoost-style model."""
        persistence = PersistenceComponent(trainer_with_get_score_fixture)

        importance = persistence.get_feature_importance()

        assert importance is not None
        assert importance["f0"] == pytest.approx(0.3)
        assert importance["f1"] == pytest.approx(0.5)
        assert importance["f2"] == pytest.approx(0.2)

    def test_get_feature_importance_returns_none_when_not_fitted(
        self,
        trainer_not_fitted_fixture: TestableTrainer,
    ) -> None:
        """Verify None returned when model not fitted."""
        persistence = PersistenceComponent(trainer_not_fitted_fixture)

        importance = persistence.get_feature_importance()

        assert importance is None

    def test_get_feature_importance_returns_none_when_model_is_none(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify None returned when model is None."""
        trainer_fixture._model = None
        persistence = PersistenceComponent(trainer_fixture)

        importance = persistence.get_feature_importance()

        assert importance is None

    def test_get_feature_importance_returns_none_when_unsupported(
        self,
        trainer_no_importance_fixture: TestableTrainer,
    ) -> None:
        """Verify None returned for models without importance support."""
        persistence = PersistenceComponent(trainer_no_importance_fixture)

        importance = persistence.get_feature_importance()

        assert importance is None


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Tests for logging methods."""

    def test_log_info_calls_logger(
        self,
        persistence_fixture: PersistenceComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_info uses logger.info."""
        with caplog.at_level(logging.INFO):
            persistence_fixture._log_info("Test info message")

        assert "Test info message" in caplog.text

    def test_log_warning_calls_logger(
        self,
        persistence_fixture: PersistenceComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_warning uses logger.warning."""
        with caplog.at_level(logging.WARNING):
            persistence_fixture._log_warning("Test warning message")

        assert "Test warning message" in caplog.text

    def test_log_error_calls_logger(
        self,
        persistence_fixture: PersistenceComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_error uses logger.error."""
        with caplog.at_level(logging.ERROR):
            persistence_fixture._log_error("Test error message")

        assert "Test error message" in caplog.text


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_testable_trainer_implements_protocol(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify TestableTrainer implements PersistenceTrainerProtocol."""
        # Check required attributes
        assert hasattr(trainer_fixture, "_config")
        assert hasattr(trainer_fixture, "_feature_names")
        assert hasattr(trainer_fixture, "_training_metrics")
        assert hasattr(trainer_fixture, "_is_fitted")
        assert hasattr(trainer_fixture, "_model")

        # Check required methods
        assert callable(getattr(trainer_fixture, "_convert_to_onnx", None))
        assert callable(getattr(trainer_fixture, "_config_to_dict", None))

    def test_persistence_component_accepts_protocol_compliant_trainer(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify PersistenceComponent works with protocol-compliant trainer."""
        # This should not raise
        persistence = PersistenceComponent(trainer_fixture)

        # Verify component stores trainer reference
        assert persistence._trainer is trainer_fixture


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for persistence workflow."""

    def test_export_and_log_workflow(
        self,
        trainer_fixture: TestableTrainer,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify complete export workflow with logging."""
        persistence = PersistenceComponent(trainer_fixture)
        onnx_path = tmp_path / "model.onnx"

        with caplog.at_level(logging.INFO):
            with patch("ml.training.common.persistence.HAS_ONNX", True):
                persistence.export_to_onnx(onnx_path)

        assert "Model exported to ONNX" in caplog.text
        assert str(onnx_path) in caplog.text


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_feature_importance_with_mismatched_lengths(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify None returned when feature names and importance lengths differ."""
        trainer_fixture._feature_names = ["f0", "f1"]  # 2 features
        # Model has 3 importance values
        persistence = PersistenceComponent(trainer_fixture)

        importance = persistence.get_feature_importance()

        assert importance is None

    def test_create_manifest_handles_missing_optional_config(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify manifest creation with minimal config."""
        config = MockConfig()
        trainer = TestableTrainer(config)
        trainer._model = MockModel()
        trainer._is_fitted = True
        trainer._feature_names = []  # No features
        trainer._training_metrics = {}  # No metrics
        persistence = PersistenceComponent(trainer)

        save_path = tmp_path / "model.pkl"

        with patch("ml.registry.ModelManifest") as mock_manifest:
            with patch(
                "ml.registry.feature_registry.compute_schema_hash"
            ) as mock_hash:
                mock_hash.return_value = ""
                mock_manifest.return_value = MagicMock()

                persistence._create_model_manifest(save_path)

        call_kwargs = mock_manifest.call_args[1]
        assert call_kwargs["feature_schema"] == {}
        assert call_kwargs["performance_metrics"] == {}
        assert call_kwargs["serveable"] is False  # Not ONNX
        assert call_kwargs["artifact_format"] == "native"
