"""
Unit tests for HyperparameterComponent.

This module tests the hyperparameter optimization component extracted from BaseMLTrainer
(lines 493-502 and 514-816). Tests verify:
- Optuna enablement check (_should_use_optuna)
- Hyperparameter optimization orchestration (_optimize_hyperparameters)
- Metric name resolution (_resolve_optuna_metric_name)
- Direction resolution (_resolve_optuna_direction)
- Optuna direction mapping (_optuna_direction_for_metric)
- Metric calculations (_calculate_optuna_metric)
- Probability thresholding (_probabilities_to_labels)
- Sharpe ratio calculation (_calculate_sharpe_metric)
- Sampler factory (_build_optuna_sampler)
- Pruner factory (_build_optuna_pruner)
- Train with params (_train_with_params)
- Classification detection (_is_classification_problem)

Following the test design in reports/tests/phase_3_8_test_design_report.md - Component 3.8.4.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml._imports import HAS_OPTUNA, HAS_SKLEARN
from ml.config.targets import BinaryTargetConfig
from ml.config.targets import MulticlassTargetConfig
from ml.config.targets import RegressionTargetConfig
from ml.config.targets import TargetCostModelConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.training.common.hyperparameter import (
    HyperparameterComponent,
    HyperparameterTrainerProtocol,
)


# ============================================================================
# Skip if Optuna not available
# ============================================================================

pytestmark = pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not available")


# ============================================================================
# Mock Model and Trainer Fixtures
# ============================================================================


class MockModel:
    """Mock model for testing hyperparameter optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self._is_fitted = False

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> MockModel:
        self._is_fitted = True
        return self


@dataclass
class MockOptunaConfig:
    """Mock Optuna configuration for testing."""

    enabled: bool = True
    n_trials: int = 3
    direction: str = "maximize"
    metric: str = "accuracy"
    pruner: str = "median"
    sampler: str = "tpe"
    timeout: int | None = None


@dataclass
class MockConfig:
    """Mock training configuration for hyperparameter testing."""

    optuna_config: MockOptunaConfig | None = None
    target_semantics: TargetSemanticsConfig | None = None
    random_seed: int | None = 42


class TestableTrainer:
    """
    Concrete trainer implementation for testing HyperparameterComponent.

    Implements the HyperparameterTrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._optuna_study: Any = None
        self._call_log: list[str] = []
        self._models_created: list[MockModel] = []

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_info."""
        self._call_log.append(f"info: {message}")
        logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Mock implementation of _log_warning."""
        self._call_log.append(f"warning: {message}")
        logging.warning(message, *args, **kwargs)

    def _create_model(self, params: dict[str, Any]) -> MockModel:
        """Mock implementation of _create_model."""
        self._call_log.append("_create_model")
        model = MockModel(params)
        self._models_created.append(model)
        return model

    def _suggest_hyperparameters(self, trial: Any) -> dict[str, Any]:
        """Mock implementation of _suggest_hyperparameters."""
        self._call_log.append("_suggest_hyperparameters")
        return {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
            "max_depth": trial.suggest_int("depth", 3, 10),
        }

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock implementation of _train_model."""
        self._call_log.append("_train_model")
        model = MockModel(kwargs)
        model._is_fitted = True
        return {"model": model, "metrics": {"loss": 0.1}}

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """Mock implementation of predict."""
        self._call_log.append("predict")
        n = len(X)
        return np.full(n, 0.6, dtype=np.float32)


# ============================================================================
# Helpers
# ============================================================================


def _build_cost_semantics(cost_bps: float) -> TargetSemanticsConfig:
    return TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=15),),
        cost_model=TargetCostModelConfig(cost_bps=cost_bps),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0, return_basis="raw"),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create basic TestableTrainer instance without Optuna configured."""
    config = MockConfig(optuna_config=None)
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_optuna_fixture() -> TestableTrainer:
    """Create TestableTrainer with Optuna enabled."""
    optuna_config = MockOptunaConfig(
        enabled=True,
        n_trials=3,
        metric="accuracy",
    )
    config = MockConfig(optuna_config=optuna_config)
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_optuna_disabled_fixture() -> TestableTrainer:
    """Create TestableTrainer with Optuna disabled."""
    optuna_config = MockOptunaConfig(enabled=False)
    config = MockConfig(optuna_config=optuna_config)
    return TestableTrainer(config)


@pytest.fixture
def sample_feature_array() -> npt.NDArray[np.float64]:
    """Create sample feature array for testing."""
    np.random.seed(42)
    return np.random.randn(50, 5).astype(np.float64)


@pytest.fixture
def sample_labels() -> npt.NDArray[np.float64]:
    """Create sample labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, 50).astype(np.float64)


@pytest.fixture
def binary_labels() -> npt.NDArray[np.float64]:
    """Binary classification labels."""
    return np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)


@pytest.fixture
def continuous_labels() -> npt.NDArray[np.float64]:
    """Continuous regression labels."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


@pytest.fixture
def sample_returns() -> npt.NDArray[np.float64]:
    """Sample returns array for Sharpe calculation."""
    np.random.seed(42)
    return np.random.randn(50).astype(np.float64) * 0.02


@pytest.fixture
def optuna_config_fixture() -> MockOptunaConfig:
    """Create sample Optuna configuration."""
    return MockOptunaConfig()


# ============================================================================
# Happy Path Tests - _should_use_optuna
# ============================================================================


class TestShouldUseOptuna:
    """Tests for _should_use_optuna method."""

    def test_should_use_optuna_returns_true(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
    ) -> None:
        """Verify Optuna enabled check returns True when properly configured."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)
        assert hp_component._should_use_optuna() is True

    def test_should_use_optuna_returns_false_no_config(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify Optuna disabled without config."""
        hp_component = HyperparameterComponent(trainer_fixture)
        assert hp_component._should_use_optuna() is False

    def test_should_use_optuna_returns_false_disabled(
        self,
        trainer_with_optuna_disabled_fixture: TestableTrainer,
    ) -> None:
        """Verify Optuna disabled when enabled=False."""
        hp_component = HyperparameterComponent(trainer_with_optuna_disabled_fixture)
        assert hp_component._should_use_optuna() is False

    def test_should_use_optuna_returns_false_missing_attr(self) -> None:
        """Verify Optuna disabled when optuna_config attribute is missing."""

        class MinimalConfig:
            pass

        trainer = TestableTrainer()
        trainer._config = MinimalConfig()  # type: ignore[assignment]
        hp_component = HyperparameterComponent(trainer)
        assert hp_component._should_use_optuna() is False


# ============================================================================
# Happy Path Tests - _optimize_hyperparameters
# ============================================================================


class TestOptimizeHyperparameters:
    """Tests for _optimize_hyperparameters method."""

    def test_optimize_hyperparameters_basic(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify basic optimization flow."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)

        best_params = hp_component._optimize_hyperparameters(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
        )

        assert isinstance(best_params, dict)
        assert trainer_with_optuna_fixture._optuna_study is not None
        # Check that hyperparameters were suggested
        assert "lr" in best_params or "depth" in best_params

    def test_optimize_hyperparameters_with_validation_returns(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        sample_returns: npt.NDArray[np.float64],
    ) -> None:
        """Verify optimization with validation returns for Sharpe metric."""
        optuna_config = MockOptunaConfig(
            enabled=True,
            n_trials=2,
            metric="sharpe_ratio",
        )
        config = MockConfig(optuna_config=optuna_config)
        trainer = TestableTrainer(config)
        hp_component = HyperparameterComponent(trainer)

        best_params = hp_component._optimize_hyperparameters(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
            validation_returns=sample_returns[:15],
        )

        assert isinstance(best_params, dict)


# ============================================================================
# Happy Path Tests - _resolve_optuna_metric_name
# ============================================================================


class TestResolveOptunaMetricName:
    """Tests for _resolve_optuna_metric_name method."""

    def test_resolve_optuna_metric_name_uses_config(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        binary_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify metric from config used."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)
        optuna_cfg = MockOptunaConfig(metric="auc")

        metric = hp_component._resolve_optuna_metric_name(binary_labels, optuna_cfg)

        assert metric == "auc"

    def test_resolve_optuna_metric_name_classification_default(
        self,
        trainer_fixture: TestableTrainer,
        binary_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify classification default metric."""
        hp_component = HyperparameterComponent(trainer_fixture)

        metric = hp_component._resolve_optuna_metric_name(binary_labels, None)

        assert metric == "accuracy"

    def test_resolve_optuna_metric_name_regression_default(
        self,
        trainer_fixture: TestableTrainer,
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify regression default metric."""
        hp_component = HyperparameterComponent(trainer_fixture)

        metric = hp_component._resolve_optuna_metric_name(continuous_labels, None)

        assert metric == "rmse"


# ============================================================================
# Happy Path Tests - _optuna_direction_for_metric (Static)
# ============================================================================


class TestOptunaDirectionForMetric:
    """Tests for _optuna_direction_for_metric static method."""

    def test_optuna_direction_for_metric_maximize_accuracy(self) -> None:
        """Verify maximize direction for accuracy metric."""
        assert HyperparameterComponent._optuna_direction_for_metric("accuracy") == "maximize"

    def test_optuna_direction_for_metric_maximize_auc(self) -> None:
        """Verify maximize direction for AUC metric."""
        assert HyperparameterComponent._optuna_direction_for_metric("auc") == "maximize"

    def test_optuna_direction_for_metric_maximize_sharpe(self) -> None:
        """Verify maximize direction for Sharpe ratio metric."""
        assert HyperparameterComponent._optuna_direction_for_metric("sharpe_ratio") == "maximize"

    def test_optuna_direction_for_metric_minimize_rmse(self) -> None:
        """Verify minimize direction for RMSE metric."""
        assert HyperparameterComponent._optuna_direction_for_metric("rmse") == "minimize"

    def test_optuna_direction_for_metric_minimize_mae(self) -> None:
        """Verify minimize direction for MAE metric."""
        assert HyperparameterComponent._optuna_direction_for_metric("mae") == "minimize"

    def test_optuna_direction_for_metric_case_insensitive(self) -> None:
        """Verify case insensitivity."""
        assert HyperparameterComponent._optuna_direction_for_metric("RMSE") == "minimize"
        assert HyperparameterComponent._optuna_direction_for_metric("Accuracy") == "maximize"


# ============================================================================
# Happy Path Tests - _calculate_optuna_metric
# ============================================================================


class TestCalculateOptunaMetric:
    """Tests for _calculate_optuna_metric method."""

    def test_calculate_optuna_metric_accuracy(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify accuracy calculation."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0, 1, 1, 0], dtype=np.float64)
        y_pred = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float64)

        score = hp_component._calculate_optuna_metric("accuracy", y_true, y_pred)

        # All predictions correct after thresholding: [0, 1, 1, 0]
        assert score == 1.0

    def test_calculate_optuna_metric_rmse(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify RMSE calculation."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0.5, 0.9], dtype=np.float64)
        y_pred = np.array([0.4, 1.1], dtype=np.float64)

        score = hp_component._calculate_optuna_metric("rmse", y_true, y_pred)

        # RMSE = sqrt(((0.1)^2 + (0.2)^2)/2) = sqrt(0.025) ~ 0.158
        expected_rmse = np.sqrt((0.01 + 0.04) / 2)
        assert score == pytest.approx(expected_rmse, rel=1e-5)

    def test_calculate_optuna_metric_mae(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify MAE calculation."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0.5, 1.0], dtype=np.float64)
        y_pred = np.array([0.4, 1.2], dtype=np.float64)

        score = hp_component._calculate_optuna_metric("mae", y_true, y_pred)

        # MAE = (0.1 + 0.2) / 2 = 0.15
        assert score == pytest.approx(0.15, rel=1e-5)

    def test_calculate_optuna_metric_r2(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify R2 calculation."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y_pred = np.array([1.1, 1.9, 3.0, 4.1, 4.9], dtype=np.float64)

        score = hp_component._calculate_optuna_metric("r2", y_true, y_pred)

        # R2 should be close to 1 for good predictions
        assert 0.9 < score <= 1.0

    def test_calculate_optuna_metric_sharpe_ratio(
        self,
        trainer_fixture: TestableTrainer,
        sample_returns: npt.NDArray[np.float64],
    ) -> None:
        """Verify Sharpe ratio calculation."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0.3, 0.7, 0.4, 0.8, 0.3, 0.6, 0.2, 0.9], dtype=np.float64)

        score = hp_component._calculate_optuna_metric(
            "sharpe_ratio",
            y_true,
            y_pred,
            validation_returns=sample_returns[:8],
        )

        # Score should be finite float
        assert np.isfinite(score)


# ============================================================================
# Happy Path Tests - _probabilities_to_labels (Static)
# ============================================================================


class TestProbabilitiesToLabels:
    """Tests for _probabilities_to_labels static method."""

    def test_probabilities_to_labels_default_threshold(self) -> None:
        """Verify default threshold of 0.5."""
        predictions = np.array([0.4, 0.6, 0.5, 0.51], dtype=np.float64)

        labels = HyperparameterComponent._probabilities_to_labels(predictions)

        expected = np.array([0, 1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)

    def test_probabilities_to_labels_custom_threshold(self) -> None:
        """Verify custom threshold."""
        predictions = np.array([0.4, 0.6], dtype=np.float64)

        labels = HyperparameterComponent._probabilities_to_labels(predictions, threshold=0.3)

        expected = np.array([1, 1], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)

    def test_probabilities_to_labels_all_zeros(self) -> None:
        """Verify all predictions below threshold."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        labels = HyperparameterComponent._probabilities_to_labels(predictions)

        expected = np.array([0, 0, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)

    def test_probabilities_to_labels_all_ones(self) -> None:
        """Verify all predictions above threshold."""
        predictions = np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float64)

        labels = HyperparameterComponent._probabilities_to_labels(predictions)

        expected = np.array([1, 1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)


# ============================================================================
# Happy Path Tests - _calculate_sharpe_metric
# ============================================================================


class TestCalculateSharpeMetric:
    """Tests for _calculate_sharpe_metric method."""

    def test_calculate_sharpe_metric_basic(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify Sharpe ratio formula."""
        hp_component = HyperparameterComponent(trainer_fixture)
        predictions = np.array([0.6, 0.7, 0.4, 0.8], dtype=np.float64)
        targets = np.array([1, 1, 0, 1], dtype=np.float64)
        returns = np.array([0.01, 0.02, -0.01, 0.03], dtype=np.float64)

        sharpe = hp_component._calculate_sharpe_metric(predictions, targets, returns)

        # Should be finite and positive for mostly positive aligned returns
        assert np.isfinite(sharpe)

    def test_calculate_sharpe_metric_zero_std(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify zero std returns zero."""
        hp_component = HyperparameterComponent(trainer_fixture)
        predictions = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float64)
        targets = np.array([1, 1, 1, 1], dtype=np.float64)
        returns = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Zero returns

        sharpe = hp_component._calculate_sharpe_metric(predictions, targets, returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_metric_empty_array(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify empty array returns zero."""
        hp_component = HyperparameterComponent(trainer_fixture)
        predictions = np.array([], dtype=np.float64)
        targets = np.array([], dtype=np.float64)
        returns = np.array([], dtype=np.float64)

        sharpe = hp_component._calculate_sharpe_metric(predictions, targets, returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_metric_applies_cost_model(self) -> None:
        """Verify cost model reduces Sharpe ratio."""
        cost_semantics = _build_cost_semantics(cost_bps=10.0)
        trainer_cost = TestableTrainer(
            MockConfig(optuna_config=None, target_semantics=cost_semantics),
        )
        trainer_raw = TestableTrainer(MockConfig(optuna_config=None))

        predictions = np.array([0.7, 0.8, 0.9, 0.6], dtype=np.float64)
        targets = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        returns = np.array([0.01, 0.012, 0.011, 0.013], dtype=np.float64)

        sharpe_cost = HyperparameterComponent(trainer_cost)._calculate_sharpe_metric(
            predictions,
            targets,
            returns,
        )
        sharpe_raw = HyperparameterComponent(trainer_raw)._calculate_sharpe_metric(
            predictions,
            targets,
            returns,
        )

        assert sharpe_cost < sharpe_raw


# ============================================================================
# Happy Path Tests - _build_optuna_sampler
# ============================================================================


class TestBuildOptunaSampler:
    """Tests for _build_optuna_sampler method."""

    def test_build_optuna_sampler_tpe_uses_trainer_random_seed(self) -> None:
        config = MockConfig(random_seed=77)
        trainer = TestableTrainer(config)
        hp_component = HyperparameterComponent(trainer)
        optuna_cfg = MockOptunaConfig(sampler="tpe")

        sampler_stub = MagicMock()
        with patch("ml.training.common.hyperparameter.optuna.samplers.TPESampler") as sampler_ctor:
            sampler_ctor.return_value = sampler_stub
            sampler = hp_component._build_optuna_sampler(optuna_cfg)

        sampler_ctor.assert_called_once_with(seed=77)
        assert sampler is sampler_stub

    def test_build_optuna_sampler_tpe(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify TPE sampler built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.sampler = "tpe"

        sampler = hp_component._build_optuna_sampler(optuna_config_fixture)

        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_build_optuna_sampler_random(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify Random sampler built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.sampler = "random"

        sampler = hp_component._build_optuna_sampler(optuna_config_fixture)

        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_build_optuna_sampler_random_uses_trainer_random_seed(self) -> None:
        config = MockConfig(random_seed=19)
        trainer = TestableTrainer(config)
        hp_component = HyperparameterComponent(trainer)
        optuna_cfg = MockOptunaConfig(sampler="random")

        sampler_stub = MagicMock()
        with patch("ml.training.common.hyperparameter.optuna.samplers.RandomSampler") as sampler_ctor:
            sampler_ctor.return_value = sampler_stub
            sampler = hp_component._build_optuna_sampler(optuna_cfg)

        sampler_ctor.assert_called_once_with(seed=19)
        assert sampler is sampler_stub

    def test_build_optuna_sampler_cmaes(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify CMA-ES sampler built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.sampler = "cmaes"

        sampler = hp_component._build_optuna_sampler(optuna_config_fixture)

        assert isinstance(sampler, optuna.samplers.CmaEsSampler)

    def test_build_optuna_sampler_grid_fallback(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify Grid sampler falls back to TPE with warning."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.sampler = "grid"

        with caplog.at_level(logging.WARNING):
            sampler = hp_component._build_optuna_sampler(optuna_config_fixture)

        assert isinstance(sampler, optuna.samplers.TPESampler)
        assert any("grid sampler" in record.message.lower() for record in caplog.records)

    def test_build_optuna_sampler_when_seed_missing_uses_none(self) -> None:
        config = MockConfig(random_seed=None)
        trainer = TestableTrainer(config)
        hp_component = HyperparameterComponent(trainer)
        optuna_cfg = MockOptunaConfig(sampler="tpe")

        sampler_stub = MagicMock()
        with patch("ml.training.common.hyperparameter.optuna.samplers.TPESampler") as sampler_ctor:
            sampler_ctor.return_value = sampler_stub
            sampler = hp_component._build_optuna_sampler(optuna_cfg)

        sampler_ctor.assert_called_once_with(seed=None)
        assert sampler is sampler_stub


# ============================================================================
# Happy Path Tests - _build_optuna_pruner
# ============================================================================


class TestBuildOptunaPruner:
    """Tests for _build_optuna_pruner method."""

    def test_build_optuna_pruner_median(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify median pruner built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.pruner = "median"

        pruner = hp_component._build_optuna_pruner(optuna_config_fixture)

        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_build_optuna_pruner_hyperband(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify Hyperband pruner built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.pruner = "hyperband"

        pruner = hp_component._build_optuna_pruner(optuna_config_fixture)

        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    def test_build_optuna_pruner_percentile(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify Percentile pruner built."""
        import optuna

        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.pruner = "percentile"

        pruner = hp_component._build_optuna_pruner(optuna_config_fixture)

        assert isinstance(pruner, optuna.pruners.PercentilePruner)

    def test_build_optuna_pruner_none(
        self,
        trainer_fixture: TestableTrainer,
        optuna_config_fixture: MockOptunaConfig,
    ) -> None:
        """Verify no pruner when disabled."""
        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_config_fixture.pruner = "none"

        pruner = hp_component._build_optuna_pruner(optuna_config_fixture)

        assert pruner is None


# ============================================================================
# Happy Path Tests - _train_with_params
# ============================================================================


class TestTrainWithParams:
    """Tests for _train_with_params method."""

    def test_train_with_params_basic(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify train with specific params."""
        hp_component = HyperparameterComponent(trainer_fixture)
        params = {"learning_rate": 0.05, "max_depth": 5}

        model = hp_component._train_with_params(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
            params,
        )

        assert model is not None
        assert isinstance(model, MockModel)
        assert "_train_model" in trainer_fixture._call_log


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestErrorConditions:
    """Tests for error conditions and fallbacks."""

    def test_optimize_hyperparameters_raises_without_optuna(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify error when Optuna unavailable."""
        # We need to patch at module level AND in _imports where check_ml_dependencies looks
        import ml._imports as imports_module
        import ml.training.common.hyperparameter as hp_module

        monkeypatch.setattr(hp_module, "HAS_OPTUNA", False)
        monkeypatch.setattr(imports_module, "HAS_OPTUNA", False)
        # Ensure the module under test uses the patched dependency guard even if
        # another test reloads `ml._imports` and invalidates earlier imports.
        monkeypatch.setattr(hp_module, "check_ml_dependencies", imports_module.check_ml_dependencies)

        hp_component = HyperparameterComponent(trainer_fixture)

        # check_ml_dependencies raises ImportError when dependency missing
        with pytest.raises(ImportError):
            hp_component._optimize_hyperparameters(
                sample_feature_array[:35],
                sample_labels[:35],
                sample_feature_array[35:],
                sample_labels[35:],
            )

    def test_calculate_optuna_metric_sharpe_without_returns(
        self,
        trainer_fixture: TestableTrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify fallback when returns missing for Sharpe."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0.3, 0.7, 0.2, 0.8], dtype=np.float64)

        with caplog.at_level(logging.WARNING):
            score = hp_component._calculate_optuna_metric(
                "sharpe_ratio",
                y_true,
                y_pred,
                validation_returns=None,
            )

        # Should fall back to accuracy
        assert any("without validation returns" in record.message.lower() for record in caplog.records)
        # Accuracy: predictions threshold to [0, 1, 0, 1], matches exactly
        assert score == 1.0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required for this test")
    def test_calculate_optuna_metric_auc_with_sklearn(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify AUC calculation with sklearn."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0, 0, 1, 1], dtype=np.float64)
        y_pred = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)

        score = hp_component._calculate_optuna_metric("auc", y_true, y_pred)

        # Perfect ranking should give AUC = 1.0
        assert score == pytest.approx(1.0)

    def test_calculate_optuna_metric_auc_without_sklearn(
        self,
        trainer_fixture: TestableTrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify AUC fallback without sklearn."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0.3, 0.7, 0.2, 0.8], dtype=np.float64)

        with patch("ml.training.common.hyperparameter.HAS_SKLEARN", False):
            with caplog.at_level(logging.WARNING):
                score = hp_component._calculate_optuna_metric("auc", y_true, y_pred)

        # Should fall back to accuracy
        assert any("sklearn unavailable" in record.message.lower() for record in caplog.records)
        assert np.isfinite(score)


# ============================================================================
# Property Tests
# ============================================================================


class TestPropertyTests:
    """Property tests for invariants."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_probabilities_to_labels_bounded(self, predictions_list: list[float]) -> None:
        """Labels always 0 or 1."""
        predictions = np.array(predictions_list, dtype=np.float64)

        labels = HyperparameterComponent._probabilities_to_labels(predictions)

        assert all(label in {0, 1} for label in labels)

    @given(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        ),
    )
    @settings(max_examples=30)
    def test_optuna_metric_finite(
        self,
        y_true_list: list[float],
        y_pred_list: list[float],
    ) -> None:
        """Metrics always return finite values."""
        # Ensure same length
        min_len = min(len(y_true_list), len(y_pred_list))
        y_true = np.array(y_true_list[:min_len], dtype=np.float64)
        y_pred = np.array(y_pred_list[:min_len], dtype=np.float64)

        trainer = TestableTrainer()
        hp_component = HyperparameterComponent(trainer)

        for metric in ["accuracy", "rmse", "mae"]:
            result = hp_component._calculate_optuna_metric(metric, y_true, y_pred)
            assert np.isfinite(result)

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_threshold_boundary_behavior(self, threshold: float) -> None:
        """Labels change correctly around threshold."""
        predictions = np.array([threshold - 0.001, threshold, threshold + 0.001], dtype=np.float64)
        predictions = np.clip(predictions, 0.0, 1.0)

        labels = HyperparameterComponent._probabilities_to_labels(predictions, threshold=threshold)

        # All should be valid binary labels
        assert all(label in {0, 1} for label in labels)


# ============================================================================
# Classification Detection Tests
# ============================================================================


class TestIsClassificationProblem:
    """Tests for _is_classification_problem method."""

    def test_is_classification_problem_binary(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify binary classification detection."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y = np.array([0, 1, 0, 1], dtype=np.float64)

        assert hp_component._is_classification_problem(y) is True

    def test_is_classification_problem_multiclass(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify multiclass detection."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)

        assert hp_component._is_classification_problem(y) is True

    def test_is_classification_problem_regression(
        self,
        trainer_fixture: TestableTrainer,
        continuous_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify regression detection."""
        hp_component = HyperparameterComponent(trainer_fixture)

        assert hp_component._is_classification_problem(continuous_labels) is False

    def test_is_classification_many_unique_values(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify >10 unique values = regression."""
        hp_component = HyperparameterComponent(trainer_fixture)
        y = np.arange(15, dtype=np.float64)

        assert hp_component._is_classification_problem(y) is False


# ============================================================================
# Direction Resolution Tests
# ============================================================================


class TestResolveOptunaDirection:
    """Tests for _resolve_optuna_direction method."""

    def test_resolve_direction_matches_metric(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify direction matches metric's natural direction."""
        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_cfg = MockOptunaConfig(direction="maximize", metric="rmse")

        direction = hp_component._resolve_optuna_direction("rmse", optuna_cfg)

        # RMSE should minimize, not maximize
        assert direction == "minimize"

    def test_resolve_direction_logs_warning_on_mismatch(
        self,
        trainer_fixture: TestableTrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning logged when config direction mismatches metric."""
        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_cfg = MockOptunaConfig(direction="maximize", metric="rmse")

        with caplog.at_level(logging.WARNING):
            hp_component._resolve_optuna_direction("rmse", optuna_cfg)

        assert any("mismatches" in record.message.lower() for record in caplog.records)

    def test_resolve_direction_no_warning_when_correct(
        self,
        trainer_fixture: TestableTrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify no warning when direction matches."""
        hp_component = HyperparameterComponent(trainer_fixture)
        optuna_cfg = MockOptunaConfig(direction="maximize", metric="accuracy")

        with caplog.at_level(logging.WARNING):
            direction = hp_component._resolve_optuna_direction("accuracy", optuna_cfg)

        assert direction == "maximize"
        # No mismatch warnings
        assert not any("mismatches" in record.message.lower() for record in caplog.records)


# ============================================================================
# Component Integration Tests
# ============================================================================


class TestComponentIntegration:
    """Tests verifying component integration with trainer."""

    def test_hp_component_logs_via_trainer(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component uses trainer's logging methods."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)
        hp_component._optimize_hyperparameters(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
        )

        # Check that info logs were recorded
        info_logs = [log for log in trainer_with_optuna_fixture._call_log if log.startswith("info:")]
        assert len(info_logs) > 0

    def test_hp_component_creates_models_via_trainer(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component creates models through trainer."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)
        hp_component._optimize_hyperparameters(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
        )

        # Should have created models for each trial (3 trials)
        assert len(trainer_with_optuna_fixture._models_created) >= 1

    def test_hp_component_sets_study_on_trainer(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component sets _optuna_study on trainer."""
        hp_component = HyperparameterComponent(trainer_with_optuna_fixture)

        assert trainer_with_optuna_fixture._optuna_study is None

        hp_component._optimize_hyperparameters(
            sample_feature_array[:35],
            sample_labels[:35],
            sample_feature_array[35:],
            sample_labels[35:],
        )

        assert trainer_with_optuna_fixture._optuna_study is not None


__all__ = [
    "MockConfig",
    "MockModel",
    "MockOptunaConfig",
    "TestableTrainer",
]
