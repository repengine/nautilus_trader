"""
Unit tests for BaseMLTrainerFacade.

This module tests the facade implementation for BaseMLTrainer decomposition,
verifying that it correctly delegates to components and maintains the same
public API as the legacy implementation.

Since BaseMLTrainerFacade is abstract (ABC), we use TestableTrainerFacade
which provides mock implementations for all 7 abstract methods.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.config.base import MLFeatureConfig, MLTrainingConfig
from ml.training.base_facade import BaseMLTrainerFacade


# =============================================================================
# Mock Model for Testing
# =============================================================================


class MockModel:
    """Mock model for testing."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.feature_importances_ = np.array([0.3, 0.5, 0.2])
        self.fitted = False

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        *,
        eval_set: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None,
        verbose: bool = False,
    ) -> MockModel:
        self.fitted = True
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        return np.full(len(X), 0.5, dtype=np.float32)


# =============================================================================
# Testable Facade Implementation (concrete subclass for testing)
# =============================================================================


class TestableTrainerFacade(BaseMLTrainerFacade):
    """
    Concrete subclass of BaseMLTrainerFacade for testing.

    Provides mock implementations for all 7 abstract methods.
    """

    def prepare_data(
        self,
        data: Any,
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """Mock: return X, y from DataFrame or array."""
        if hasattr(data, "drop"):
            # Polars DataFrame style
            X = data.drop(target_col).to_numpy()
            y = data[target_col].to_numpy()
        elif hasattr(data, "values"):
            # Pandas-like
            X = np.array(data.values)[:, :-1]
            y = np.array(data.values)[:, -1]
        else:
            # Numpy array
            X = np.array(data)
            y = np.zeros(len(X))

        X = X.astype(np.float64) if X.ndim > 1 else X.reshape(-1, 1).astype(np.float64)
        y = y.astype(np.float64)

        return (
            X,
            y,
            {"feature_names": [f"feature_{i}" for i in range(X.shape[1])]},
        )

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock: return a simple model object with metrics."""
        model = MockModel(kwargs)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return {
            "model": model,
            "metrics": {"loss": 0.1, "val_loss": 0.15},
        }

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """Mock: return predictions."""
        return_labels = kwargs.get("return_labels", False)
        n = len(X)
        if return_labels:
            return np.zeros(n, dtype=np.int64).astype(np.float32)
        return np.full(n, 0.5, dtype=np.float32)

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Mock: create model instance."""
        return MockModel(params)

    def _get_model_params(self) -> dict[str, Any]:
        """Mock: return default params."""
        return {"mock_param": 1, "learning_rate": 0.1}

    def _suggest_hyperparameters(self, trial: Any) -> dict[str, Any]:
        """Mock: suggest hyperparameters."""
        return {"learning_rate": trial.suggest_float("lr", 0.01, 0.1)}

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """Mock: write dummy ONNX bytes."""
        path.write_bytes(b"mock_onnx_model_bytes")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config() -> MLTrainingConfig:
    """Create a basic training configuration."""
    return MLTrainingConfig(
        data_source="memory",
        target_column="target",
        train_test_split=0.8,
    )


@pytest.fixture
def trainer_facade(basic_config: MLTrainingConfig) -> TestableTrainerFacade:
    """Create a TestableTrainerFacade instance."""
    return TestableTrainerFacade(basic_config)


@pytest.fixture
def sample_data() -> npt.NDArray[np.float64]:
    """Create sample training data."""
    np.random.seed(42)
    return np.random.randn(100, 4).astype(np.float64)


@pytest.fixture
def sample_features() -> npt.NDArray[np.float64]:
    """Create sample features."""
    np.random.seed(42)
    return np.random.randn(100, 3).astype(np.float64)


@pytest.fixture
def sample_labels() -> npt.NDArray[np.float64]:
    """Create sample binary labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100).astype(np.float64)


@pytest.fixture
def continuous_labels() -> npt.NDArray[np.float64]:
    """Create sample continuous labels for regression."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


# =============================================================================
# Test: Facade is ABC (Abstract Base Class)
# =============================================================================


class TestFacadeIsAbstract:
    """Test that BaseMLTrainerFacade cannot be instantiated directly."""

    def test_cannot_instantiate_base_facade_directly(
        self, basic_config: MLTrainingConfig
    ) -> None:
        """BaseMLTrainerFacade is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseMLTrainerFacade(basic_config)  # type: ignore[abstract]

    def test_subclass_without_all_abstract_methods_raises(
        self, basic_config: MLTrainingConfig
    ) -> None:
        """Subclass without all abstract methods cannot be instantiated."""

        class IncompleteTrainer(BaseMLTrainerFacade):
            """Incomplete - missing abstract methods."""


        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteTrainer(basic_config)  # type: ignore[abstract]

    def test_testable_facade_can_be_instantiated(
        self, basic_config: MLTrainingConfig
    ) -> None:
        """TestableTrainerFacade implements all abstract methods and can be instantiated."""
        trainer = TestableTrainerFacade(basic_config)
        assert trainer is not None
        assert isinstance(trainer, BaseMLTrainerFacade)


# =============================================================================
# Test: Initialization
# =============================================================================


class TestFacadeInitialization:
    """Test facade initialization."""

    def test_config_stored(self, trainer_facade: TestableTrainerFacade) -> None:
        """Config is stored on trainer."""
        assert trainer_facade._config is not None
        assert trainer_facade._config.target_column == "target"

    def test_feature_config_defaults(self, trainer_facade: TestableTrainerFacade) -> None:
        """Feature config defaults when not provided."""
        assert trainer_facade._feature_config is not None
        assert isinstance(trainer_facade._feature_config, MLFeatureConfig)

    def test_initial_state(self, trainer_facade: TestableTrainerFacade) -> None:
        """Initial state is unfitted."""
        assert trainer_facade._model is None
        assert trainer_facade._feature_names == []
        assert trainer_facade._training_metrics == {}
        assert trainer_facade._is_fitted is False
        assert trainer_facade._mlflow_run_id is None
        assert trainer_facade._optuna_study is None
        assert trainer_facade._cv_results == []

    def test_components_initialized(self, trainer_facade: TestableTrainerFacade) -> None:
        """All 7 components are initialized."""
        assert trainer_facade._orchestrator is not None
        assert trainer_facade._data_prep is not None
        assert trainer_facade._cv_component is not None
        assert trainer_facade._hyperparameter is not None
        assert trainer_facade._mlflow is not None
        assert trainer_facade._evaluation is not None
        assert trainer_facade._persistence is not None

    def test_custom_feature_config(self) -> None:
        """Custom feature config is used when provided."""
        feature_config = MLFeatureConfig(lookback_window=200)
        config = MLTrainingConfig(
            data_source="memory",
            feature_config=feature_config,
        )
        trainer = TestableTrainerFacade(config)
        assert trainer._feature_config.lookback_window == 200


# =============================================================================
# Test: Abstract Method Implementations (via TestableTrainerFacade)
# =============================================================================


class TestAbstractMethodImplementations:
    """Test that abstract method implementations work correctly."""

    def test_prepare_data_returns_tuple(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_data: npt.NDArray[np.float64],
    ) -> None:
        """prepare_data returns (X, y, metadata) tuple."""
        X, y, metadata = trainer_facade.prepare_data(sample_data)
        assert X.ndim == 2
        assert y.ndim == 1
        assert "feature_names" in metadata

    def test_train_model_returns_model_and_metrics(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """_train_model returns dict with model and metrics."""
        result = trainer_facade._train_model(
            sample_features[:80],
            sample_labels[:80],
            sample_features[80:],
            sample_labels[80:],
        )
        assert "model" in result
        assert "metrics" in result
        assert isinstance(result["model"], MockModel)

    def test_predict_returns_float32_array(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
    ) -> None:
        """predict returns float32 array."""
        model = MockModel()
        predictions = trainer_facade.predict(model, sample_features)
        assert predictions.dtype == np.float32
        assert len(predictions) == len(sample_features)

    def test_predict_with_return_labels(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
    ) -> None:
        """predict with return_labels=True returns labels."""
        model = MockModel()
        predictions = trainer_facade.predict(model, sample_features, return_labels=True)
        # Should be class labels (integers as float32)
        assert len(predictions) == len(sample_features)

    def test_create_model_returns_model_instance(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_create_model returns model instance."""
        params = {"learning_rate": 0.05}
        model = trainer_facade._create_model(params)
        assert isinstance(model, MockModel)
        assert model.params == params

    def test_get_model_params_returns_dict(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_get_model_params returns dict."""
        params = trainer_facade._get_model_params()
        assert isinstance(params, dict)
        assert "learning_rate" in params


# =============================================================================
# Test: Component Delegation
# =============================================================================


class TestComponentDelegation:
    """Test that facade correctly delegates to components."""

    def test_should_use_cv_delegates(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_should_use_cv delegates to CrossValidationComponent."""
        # With default config (no cv_folds), should return False
        assert trainer_facade._should_use_cv() is False

    def test_should_use_optuna_delegates(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_should_use_optuna delegates to HyperparameterComponent."""
        # With default config (no optuna_config), should return False
        assert trainer_facade._should_use_optuna() is False

    def test_should_use_mlflow_delegates(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_should_use_mlflow delegates to MLflowTrackingComponent."""
        # With default config (no mlflow_config), should return False
        assert trainer_facade._should_use_mlflow() is False

    def test_is_classification_problem_delegates(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """_is_classification_problem delegates to EvaluationComponent."""
        # Binary labels should be classification
        assert trainer_facade._is_classification_problem(sample_labels) is True

    def test_is_classification_problem_regression(
        self,
        trainer_facade: TestableTrainerFacade,
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """_is_classification_problem returns False for continuous labels."""
        assert trainer_facade._is_classification_problem(continuous_labels) is False

    def test_split_data_delegates(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_data: npt.NDArray[np.float64],
    ) -> None:
        """_split_data delegates to DataPreparationComponent."""
        train, val = trainer_facade._split_data(sample_data)
        # With 0.8 split, expect 80 train, 20 val
        assert len(train) == 80
        assert len(val) == 20


# =============================================================================
# Test: Evaluation Methods
# =============================================================================


class TestEvaluationMethods:
    """Test evaluation-related methods."""

    def test_evaluate_classification(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """evaluate returns classification metrics for binary targets."""
        model = MockModel()
        metrics = trainer_facade.evaluate(model, sample_features, sample_labels)
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_regression(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """evaluate returns regression metrics for continuous targets."""
        model = MockModel()
        metrics = trainer_facade.evaluate(model, sample_features, continuous_labels)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics

    def test_calculate_trading_metrics(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """calculate_trading_metrics returns trading metrics."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        predictions = np.full(100, 0.6, dtype=np.float32)

        metrics = trainer_facade.calculate_trading_metrics(returns, predictions)
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    def test_calculate_classification_metrics(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_calculate_classification_metrics returns accuracy etc."""
        y_true = np.array([0, 1, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0, 1, 0, 0, 1], dtype=np.float32)
        metrics = trainer_facade._calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(0.8, abs=0.01)

    def test_calculate_regression_metrics(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_calculate_regression_metrics returns mse, rmse, mae, r2."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        y_pred = np.array([1.1, 2.2, 2.9, 4.0], dtype=np.float32)
        metrics = trainer_facade._calculate_regression_metrics(y_true, y_pred)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0


# =============================================================================
# Test: Logging Methods
# =============================================================================


class TestLoggingMethods:
    """Test logging helper methods."""

    def test_log_info_calls_logger(
        self, trainer_facade: TestableTrainerFacade, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_log_info logs at INFO level."""
        import logging

        with caplog.at_level(logging.INFO):
            trainer_facade._log_info("Test info message")
        assert "Test info message" in caplog.text

    def test_log_warning_calls_logger(
        self, trainer_facade: TestableTrainerFacade, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_log_warning logs at WARNING level."""
        import logging

        with caplog.at_level(logging.WARNING):
            trainer_facade._log_warning("Test warning message")
        assert "Test warning message" in caplog.text

    def test_log_error_calls_logger(
        self, trainer_facade: TestableTrainerFacade, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_log_error logs at ERROR level."""
        import logging

        with caplog.at_level(logging.ERROR):
            trainer_facade._log_error("Test error message")
        assert "Test error message" in caplog.text


# =============================================================================
# Test: Persistence Methods
# =============================================================================


class TestPersistenceMethods:
    """Test persistence-related methods."""

    def test_get_feature_importance_unfitted(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """get_feature_importance returns None when not fitted."""
        assert trainer_facade.get_feature_importance() is None

    def test_get_feature_importance_fitted(
        self,
        trainer_facade: TestableTrainerFacade,
        sample_features: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """get_feature_importance returns dict when fitted."""
        # Manually set fitted state
        trainer_facade._model = MockModel()
        trainer_facade._feature_names = ["f0", "f1", "f2"]
        trainer_facade._is_fitted = True

        importance = trainer_facade.get_feature_importance()
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == 3

    def test_export_to_onnx_raises_when_not_fitted(
        self, trainer_facade: TestableTrainerFacade, tmp_path: Path
    ) -> None:
        """export_to_onnx raises ValueError when not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            trainer_facade.export_to_onnx(tmp_path / "model.onnx")

    def test_save_model_raises_when_not_fitted(
        self, trainer_facade: TestableTrainerFacade, tmp_path: Path
    ) -> None:
        """save_model raises ValueError when not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            trainer_facade.save_model(tmp_path / "model")


# =============================================================================
# Test: Config to Dict
# =============================================================================


class TestConfigToDict:
    """Test config serialization."""

    def test_config_to_dict_returns_dict(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_config_to_dict returns a dictionary."""
        result = trainer_facade._config_to_dict()
        assert isinstance(result, dict)

    def test_config_to_dict_includes_scalars(
        self, trainer_facade: TestableTrainerFacade
    ) -> None:
        """_config_to_dict includes scalar config values."""
        result = trainer_facade._config_to_dict()
        # Should include string, int, float, bool values
        assert "data_source" in result
        assert "target_column" in result
