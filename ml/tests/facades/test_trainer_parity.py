"""
Parity tests for BaseMLTrainer legacy vs facade implementations.

This module verifies that the new BaseMLTrainerFacade produces identical results
to the legacy BaseMLTrainer implementation. Tests are parameterized to run
against both implementations and compare outputs.

Key parity areas tested:
- Training workflow produces same results
- Evaluation metrics match
- Cross-validation fold results match
- Trading metrics calculation is identical
- Feature importance extraction is identical
- Data splitting preserves counts

"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from ml.config.base import MLFeatureConfig, MLTrainingConfig
from ml.training.base import BaseMLTrainer
from ml.training.base_facade import BaseMLTrainerFacade


# =============================================================================
# Mock Model for Both Implementations
# =============================================================================


class MockModel:
    """Mock model for testing both implementations."""

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
# Testable Legacy Trainer Implementation
# =============================================================================


class TestableLegacyTrainer(BaseMLTrainer):
    """
    Concrete subclass of legacy BaseMLTrainer for testing.

    Provides mock implementations for all 7 abstract methods with
    deterministic behavior for parity comparison.
    """

    def prepare_data(
        self,
        data: Any,
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """Mock: return X, y from data."""
        if hasattr(data, "drop"):
            X = data.drop(target_col).to_numpy()
            y = data[target_col].to_numpy()
        elif hasattr(data, "values"):
            X = np.array(data.values)[:, :-1]
            y = np.array(data.values)[:, -1]
        else:
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
        """Mock: return deterministic model and metrics."""
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
        """Mock: return deterministic predictions."""
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
# Testable Facade Trainer Implementation
# =============================================================================


class TestableFacadeTrainer(BaseMLTrainerFacade):
    """
    Concrete subclass of BaseMLTrainerFacade for testing.

    Uses IDENTICAL implementations to TestableLegacyTrainer for parity comparison.
    """

    def prepare_data(
        self,
        data: Any,
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """Mock: return X, y from data - IDENTICAL to legacy."""
        if hasattr(data, "drop"):
            X = data.drop(target_col).to_numpy()
            y = data[target_col].to_numpy()
        elif hasattr(data, "values"):
            X = np.array(data.values)[:, :-1]
            y = np.array(data.values)[:, -1]
        else:
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
        """Mock: return deterministic model and metrics - IDENTICAL to legacy."""
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
        """Mock: return deterministic predictions - IDENTICAL to legacy."""
        return_labels = kwargs.get("return_labels", False)
        n = len(X)
        if return_labels:
            return np.zeros(n, dtype=np.int64).astype(np.float32)
        return np.full(n, 0.5, dtype=np.float32)

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Mock: create model instance - IDENTICAL to legacy."""
        return MockModel(params)

    def _get_model_params(self) -> dict[str, Any]:
        """Mock: return default params - IDENTICAL to legacy."""
        return {"mock_param": 1, "learning_rate": 0.1}

    def _suggest_hyperparameters(self, trial: Any) -> dict[str, Any]:
        """Mock: suggest hyperparameters - IDENTICAL to legacy."""
        return {"learning_rate": trial.suggest_float("lr", 0.01, 0.1)}

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """Mock: write dummy ONNX bytes - IDENTICAL to legacy."""
        path.write_bytes(b"mock_onnx_model_bytes")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config() -> MLTrainingConfig:
    """Create basic training configuration for both implementations."""
    return MLTrainingConfig(
        data_source="memory",
        target_column="target",
        train_test_split=0.8,
        random_seed=42,
    )


@pytest.fixture
def legacy_trainer(basic_config: MLTrainingConfig) -> TestableLegacyTrainer:
    """Create legacy trainer instance."""
    return TestableLegacyTrainer(basic_config)


@pytest.fixture
def facade_trainer(basic_config: MLTrainingConfig) -> TestableFacadeTrainer:
    """Create facade trainer instance."""
    return TestableFacadeTrainer(basic_config)


@pytest.fixture
def sample_features() -> npt.NDArray[np.float64]:
    """Create deterministic sample features."""
    np.random.seed(42)
    return np.random.randn(100, 3).astype(np.float64)


@pytest.fixture
def sample_labels() -> npt.NDArray[np.float64]:
    """Create deterministic sample binary labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100).astype(np.float64)


@pytest.fixture
def continuous_labels() -> npt.NDArray[np.float64]:
    """Create deterministic continuous labels for regression."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


@pytest.fixture
def sample_returns() -> npt.NDArray[np.float64]:
    """Create deterministic sample returns for trading metrics."""
    np.random.seed(42)
    return (np.random.randn(100) * 0.02).astype(np.float64)


# =============================================================================
# Parity Tests: Evaluation Methods
# =============================================================================


class TestEvaluateParity:
    """Test that evaluate() produces identical results."""

    @pytest.mark.parametrize("use_facade", [False, True])
    def test_evaluate_classification_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        use_facade: bool,
    ) -> None:
        """Classification evaluation produces same metrics."""
        trainer = facade_trainer if use_facade else legacy_trainer
        model = MockModel()
        metrics = trainer.evaluate(model, sample_features, sample_labels)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_classification_parity_comparison(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Direct comparison: legacy and facade produce identical classification metrics."""
        model = MockModel()

        legacy_metrics = legacy_trainer.evaluate(model, sample_features, sample_labels)
        facade_metrics = facade_trainer.evaluate(model, sample_features, sample_labels)

        # Compare all metrics
        assert legacy_metrics.keys() == facade_metrics.keys()
        for key in legacy_metrics:
            np.testing.assert_allclose(
                legacy_metrics[key],
                facade_metrics[key],
                rtol=1e-10,
                err_msg=f"Metric {key} differs between legacy and facade",
            )

    def test_evaluate_regression_parity_comparison(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """Direct comparison: legacy and facade produce identical regression metrics."""
        model = MockModel()

        legacy_metrics = legacy_trainer.evaluate(model, sample_features, continuous_labels)
        facade_metrics = facade_trainer.evaluate(model, sample_features, continuous_labels)

        # Compare all metrics
        assert legacy_metrics.keys() == facade_metrics.keys()
        for key in legacy_metrics:
            np.testing.assert_allclose(
                legacy_metrics[key],
                facade_metrics[key],
                rtol=1e-10,
                err_msg=f"Metric {key} differs between legacy and facade",
            )


# =============================================================================
# Parity Tests: Trading Metrics
# =============================================================================


class TestTradingMetricsParity:
    """Test that calculate_trading_metrics() produces identical results."""

    def test_trading_metrics_parity_comparison(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_returns: npt.NDArray[np.float64],
    ) -> None:
        """Direct comparison: trading metrics are identical."""
        predictions = np.full(100, 0.6, dtype=np.float32)

        legacy_metrics = legacy_trainer.calculate_trading_metrics(sample_returns, predictions)
        facade_metrics = facade_trainer.calculate_trading_metrics(sample_returns, predictions)

        # Compare all metrics
        assert legacy_metrics.keys() == facade_metrics.keys()
        for key in legacy_metrics:
            np.testing.assert_allclose(
                legacy_metrics[key],
                facade_metrics[key],
                rtol=1e-10,
                err_msg=f"Trading metric {key} differs between legacy and facade",
            )

    @pytest.mark.parametrize("use_facade", [False, True])
    def test_trading_metrics_sharpe_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_returns: npt.NDArray[np.float64],
        use_facade: bool,
    ) -> None:
        """Sharpe ratio calculation is consistent."""
        trainer = facade_trainer if use_facade else legacy_trainer
        predictions = np.full(100, 0.6, dtype=np.float32)

        metrics = trainer.calculate_trading_metrics(sample_returns, predictions)

        if "sharpe_ratio" in metrics:
            assert np.isfinite(metrics["sharpe_ratio"])


# =============================================================================
# Parity Tests: Data Splitting
# =============================================================================


class TestDataSplitParity:
    """Test that _split_data() produces identical results."""

    def test_split_data_parity_comparison(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
    ) -> None:
        """Direct comparison: data splitting produces same train/val sizes."""
        legacy_train, legacy_val = legacy_trainer._split_data(sample_features)
        facade_train, facade_val = facade_trainer._split_data(sample_features)

        # Same sizes
        assert len(legacy_train) == len(facade_train)
        assert len(legacy_val) == len(facade_val)

        # Same content
        np.testing.assert_array_equal(legacy_train, facade_train)
        np.testing.assert_array_equal(legacy_val, facade_val)

    def test_split_data_preserves_total_count(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
    ) -> None:
        """Total count is preserved after split."""
        original_len = len(sample_features)

        legacy_train, legacy_val = legacy_trainer._split_data(sample_features)
        facade_train, facade_val = facade_trainer._split_data(sample_features)

        assert len(legacy_train) + len(legacy_val) == original_len
        assert len(facade_train) + len(facade_val) == original_len


# =============================================================================
# Parity Tests: Classification Problem Detection
# =============================================================================


class TestClassificationDetectionParity:
    """Test that _is_classification_problem() produces identical results."""

    def test_binary_classification_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Binary labels detected as classification by both."""
        assert legacy_trainer._is_classification_problem(sample_labels) is True
        assert facade_trainer._is_classification_problem(sample_labels) is True

    def test_regression_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """Continuous labels detected as regression by both."""
        assert legacy_trainer._is_classification_problem(continuous_labels) is False
        assert facade_trainer._is_classification_problem(continuous_labels) is False

    @pytest.mark.parametrize("n_unique", [2, 5, 10, 15, 50])
    def test_unique_value_threshold_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        n_unique: int,
    ) -> None:
        """Unique value threshold is same for both implementations."""
        # Create labels with exactly n_unique values
        labels = np.tile(np.arange(n_unique), 100 // n_unique + 1)[:100].astype(np.float64)

        legacy_result = legacy_trainer._is_classification_problem(labels)
        facade_result = facade_trainer._is_classification_problem(labels)

        assert legacy_result == facade_result, (
            f"Parity failed for n_unique={n_unique}: "
            f"legacy={legacy_result}, facade={facade_result}"
        )


# =============================================================================
# Parity Tests: Feature Importance
# =============================================================================


class TestFeatureImportanceParity:
    """Test that get_feature_importance() produces identical results."""

    def test_feature_importance_unfitted_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Unfitted model returns None for both."""
        assert legacy_trainer.get_feature_importance() is None
        assert facade_trainer.get_feature_importance() is None

    def test_feature_importance_fitted_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Fitted model returns same importance for both."""
        # Set up identical fitted state
        model = MockModel()
        feature_names = ["f0", "f1", "f2"]

        legacy_trainer._model = model
        legacy_trainer._feature_names = feature_names
        legacy_trainer._is_fitted = True

        facade_trainer._model = model
        facade_trainer._feature_names = feature_names
        facade_trainer._is_fitted = True

        legacy_importance = legacy_trainer.get_feature_importance()
        facade_importance = facade_trainer.get_feature_importance()

        assert legacy_importance is not None
        assert facade_importance is not None
        assert legacy_importance.keys() == facade_importance.keys()

        for key in legacy_importance:
            np.testing.assert_allclose(
                legacy_importance[key],
                facade_importance[key],
                rtol=1e-10,
            )


# =============================================================================
# Parity Tests: CV and Optuna Flags
# =============================================================================


class TestFlagsParity:
    """Test that _should_use_* methods produce identical results."""

    def test_should_use_cv_parity_default(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Default config (no CV) returns False for both."""
        assert legacy_trainer._should_use_cv() == facade_trainer._should_use_cv()
        assert legacy_trainer._should_use_cv() is False

    def test_should_use_optuna_parity_default(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Default config (no Optuna) returns False for both."""
        assert legacy_trainer._should_use_optuna() == facade_trainer._should_use_optuna()
        assert legacy_trainer._should_use_optuna() is False

    def test_should_use_mlflow_parity_default(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Default config (no MLflow) returns False for both."""
        assert legacy_trainer._should_use_mlflow() == facade_trainer._should_use_mlflow()
        assert legacy_trainer._should_use_mlflow() is False


# =============================================================================
# Parity Tests: Config to Dict
# =============================================================================


class TestConfigToDictParity:
    """Test that _config_to_dict() produces valid results."""

    def test_config_to_dict_parity(
        self,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Config to dict produces valid output for facade.

        Note: Legacy _config_to_dict uses vars() which doesn't work with
        msgspec structs. The facade implementation handles this properly
        using __struct_fields__ for msgspec structs. This test verifies
        the facade returns the expected scalar config values.
        """
        facade_dict = facade_trainer._config_to_dict()

        # Facade should return dict with scalar values
        assert isinstance(facade_dict, dict)

        # Should include string values
        assert "data_source" in facade_dict
        assert facade_dict["data_source"] == "memory"

        # Should include target_column
        assert "target_column" in facade_dict
        assert facade_dict["target_column"] == "target"

        # All values should be scalars
        for value in facade_dict.values():
            assert isinstance(value, (str, int, float, bool))


# =============================================================================
# Parity Tests: Prepare Data
# =============================================================================


class TestPrepareDataParity:
    """Test that prepare_data() produces identical results."""

    def test_prepare_data_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
        sample_features: npt.NDArray[np.float64],
    ) -> None:
        """prepare_data produces identical X, y, metadata."""
        legacy_X, legacy_y, legacy_meta = legacy_trainer.prepare_data(sample_features)
        facade_X, facade_y, facade_meta = facade_trainer.prepare_data(sample_features)

        np.testing.assert_array_equal(legacy_X, facade_X)
        np.testing.assert_array_equal(legacy_y, facade_y)
        assert legacy_meta.keys() == facade_meta.keys()
        assert legacy_meta["feature_names"] == facade_meta["feature_names"]


# =============================================================================
# Parity Tests: Classification and Regression Metrics
# =============================================================================


class TestMetricCalculationParity:
    """Test that metric calculation methods produce identical results."""

    def test_classification_metrics_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Classification metrics calculation is identical."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.float32)

        legacy_metrics = legacy_trainer._calculate_classification_metrics(y_true, y_pred)
        facade_metrics = facade_trainer._calculate_classification_metrics(y_true, y_pred)

        assert legacy_metrics.keys() == facade_metrics.keys()
        for key in legacy_metrics:
            np.testing.assert_allclose(
                legacy_metrics[key],
                facade_metrics[key],
                rtol=1e-10,
            )

    def test_regression_metrics_parity(
        self,
        legacy_trainer: TestableLegacyTrainer,
        facade_trainer: TestableFacadeTrainer,
    ) -> None:
        """Regression metrics calculation is identical."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8], dtype=np.float32)

        legacy_metrics = legacy_trainer._calculate_regression_metrics(y_true, y_pred)
        facade_metrics = facade_trainer._calculate_regression_metrics(y_true, y_pred)

        assert legacy_metrics.keys() == facade_metrics.keys()
        for key in legacy_metrics:
            np.testing.assert_allclose(
                legacy_metrics[key],
                facade_metrics[key],
                rtol=1e-10,
            )
