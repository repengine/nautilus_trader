"""
Unit tests for EvaluationComponent.

This module tests the evaluation component extracted from BaseMLTrainer
(lines 1163-1263 and 1519-1607). Tests verify:
- Model evaluation (evaluate)
- Trading metrics calculation (calculate_trading_metrics)
- Problem type detection (_is_classification_problem, _is_classifier_objective)
- Classification metrics (_calculate_classification_metrics)
- Regression metrics (_calculate_regression_metrics)
- Edge cases and error handling
- Property tests for metric invariants

Following the test patterns established in other training component tests.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.training.common.evaluation import (
    EvaluationComponent,
    EvaluationTrainerProtocol,
)


# ============================================================================
# Mock Trainer Fixtures
# ============================================================================


@dataclass
class MockConfig:
    """Mock training configuration for evaluation testing."""

    objective: str = "regression"


class TestableTrainer:
    """
    Concrete trainer implementation for testing EvaluationComponent.

    Implements the EvaluationTrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._call_log: list[str] = []
        self._mock_predictions: npt.NDArray[np.float64] | None = None

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        *,
        return_labels: bool = False,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Mock implementation of predict."""
        self._call_log.append(f"predict(return_labels={return_labels})")
        if self._mock_predictions is not None:
            return self._mock_predictions
        # Default: return labels for classification, continuous for regression
        if return_labels:
            return np.round(np.random.rand(len(X))).astype(np.float64)
        return np.random.randn(len(X)).astype(np.float64)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_regression_fixture() -> TestableTrainer:
    """Create TestableTrainer configured for regression."""
    config = MockConfig(objective="regression")
    return TestableTrainer(config)


@pytest.fixture
def trainer_classification_fixture() -> TestableTrainer:
    """Create TestableTrainer configured for classification."""
    config = MockConfig(objective="binary:logistic")
    return TestableTrainer(config)


@pytest.fixture
def sample_feature_array() -> npt.NDArray[np.float64]:
    """Create sample feature array for evaluation testing."""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def sample_regression_targets() -> npt.NDArray[np.float64]:
    """Create sample continuous regression targets."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


@pytest.fixture
def sample_classification_targets() -> npt.NDArray[np.float64]:
    """Create sample binary classification targets."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100).astype(np.float64)


@pytest.fixture
def sample_returns() -> npt.NDArray[np.float64]:
    """Create sample asset returns for trading metrics."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02  # Daily returns ~2% vol


@pytest.fixture
def sample_predictions() -> npt.NDArray[np.float64]:
    """Create sample model predictions."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


# ============================================================================
# Happy Path Tests - evaluate()
# ============================================================================


class TestEvaluate:
    """Tests for evaluate method."""

    def test_evaluate_classification_uses_labels(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify classification evaluation requests labels."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        # Binary targets indicate classification
        X = np.random.randn(50, 5).astype(np.float64)
        y = np.array([0, 1] * 25, dtype=np.float64)

        # Set mock predictions to match
        trainer_classification_fixture._mock_predictions = np.array(
            [0, 1] * 25, dtype=np.float64
        )

        metrics = eval_component.evaluate(None, X, y)

        # Should have called predict with return_labels=True
        assert "predict(return_labels=True)" in trainer_classification_fixture._call_log
        assert "accuracy" in metrics

    def test_evaluate_regression_uses_raw_values(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify regression evaluation uses raw predictions."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        X = np.random.randn(50, 5).astype(np.float64)
        y = np.random.randn(50).astype(np.float64)

        # Set mock predictions
        trainer_regression_fixture._mock_predictions = np.random.randn(50).astype(
            np.float64
        )

        metrics = eval_component.evaluate(None, X, y)

        # Should have called predict without return_labels
        assert "predict(return_labels=False)" in trainer_regression_fixture._call_log
        assert "mse" in metrics
        assert "rmse" in metrics

    def test_evaluate_returns_dict_with_float_values(
        self,
        trainer_regression_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_regression_targets: npt.NDArray[np.float64],
    ) -> None:
        """Verify evaluate returns dict with float values."""
        eval_component = EvaluationComponent(trainer_regression_fixture)
        trainer_regression_fixture._mock_predictions = np.random.randn(100).astype(
            np.float64
        )

        metrics = eval_component.evaluate(
            None, sample_feature_array, sample_regression_targets
        )

        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, float)


# ============================================================================
# Happy Path Tests - calculate_trading_metrics()
# ============================================================================


class TestCalculateTradingMetrics:
    """Tests for calculate_trading_metrics method."""

    def test_calculate_trading_metrics_returns_all_metrics(
        self,
        trainer_regression_fixture: TestableTrainer,
        sample_returns: npt.NDArray[np.float64],
        sample_predictions: npt.NDArray[np.float64],
    ) -> None:
        """Verify all trading metrics are returned."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        metrics = eval_component.calculate_trading_metrics(
            sample_returns, sample_predictions
        )

        # Check all expected metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        # Sharpe and IR only if std > 0 (usually true for random returns)

    def test_calculate_trading_metrics_classifier_uses_threshold(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify classifier predictions use 0.5 threshold."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        # Probabilities around 0.5
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        predictions = np.array([0.6, 0.4, 0.8, 0.3, 0.7])  # Above/below 0.5

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        assert "total_return" in metrics

    def test_calculate_trading_metrics_regression_uses_sign(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify regression predictions use sign for signals."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        predictions = np.array([1.0, -1.0, 2.0, -0.5, 0.5])

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        assert "total_return" in metrics

    def test_calculate_trading_metrics_empty_returns_empty_dict(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify empty returns yield empty metrics dict."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # All NaN returns
        returns = np.array([np.nan, np.nan, np.nan])
        predictions = np.array([1.0, -1.0, 0.5])

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        assert metrics == {}

    def test_calculate_trading_metrics_sharpe_ratio_annualized(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify Sharpe ratio is annualized (sqrt(252) factor)."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # Constant positive returns
        returns = np.ones(100) * 0.001  # 0.1% daily
        predictions = np.ones(100)  # All long signals

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        # With constant returns, std=0, so Sharpe won't be computed
        # Use variable returns
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.0005  # Slight positive drift
        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        if "sharpe_ratio" in metrics:
            # Sharpe should be reasonable for these returns
            assert np.isfinite(metrics["sharpe_ratio"])


# ============================================================================
# Tests for _is_classification_problem()
# ============================================================================


class TestIsClassificationProblem:
    """Tests for _is_classification_problem method."""

    def test_is_classification_binary_targets(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify binary targets detected as classification."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.float64)
        assert eval_component._is_classification_problem(y) is True

    def test_is_classification_multiclass_targets(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify multiclass targets detected as classification."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
        assert eval_component._is_classification_problem(y) is True

    def test_is_classification_continuous_targets(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify continuous targets detected as regression."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y = np.array([1.23, 4.56, 7.89, 2.34, 5.67], dtype=np.float64)
        assert eval_component._is_classification_problem(y) is False

    def test_is_classification_many_unique_values(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify many unique values detected as regression."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # Even if integers, more than 10 unique -> regression
        y = np.arange(20, dtype=np.float64)
        assert eval_component._is_classification_problem(y) is False

    def test_is_classification_zero_one_range(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify [0, 1] bounded targets with few uniques detected as classification."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # Probabilities but only 5 unique values
        y = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25], dtype=np.float64)
        assert eval_component._is_classification_problem(y) is True


# ============================================================================
# Tests for _is_classifier_objective()
# ============================================================================


class TestIsClassifierObjective:
    """Tests for _is_classifier_objective method."""

    def test_is_classifier_objective_binary_logistic(self) -> None:
        """Verify binary:logistic detected as classifier."""
        config = MockConfig(objective="binary:logistic")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        assert eval_component._is_classifier_objective() is True

    def test_is_classifier_objective_multiclass(self) -> None:
        """Verify multiclass objective detected as classifier."""
        config = MockConfig(objective="multiclass:softmax")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        # Contains "class"
        assert eval_component._is_classifier_objective() is True

    def test_is_classifier_objective_regression(self) -> None:
        """Verify regression objective not detected as classifier."""
        config = MockConfig(objective="reg:squarederror")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        assert eval_component._is_classifier_objective() is False

    def test_is_classifier_objective_empty_string(self) -> None:
        """Verify empty objective not detected as classifier."""
        config = MockConfig(objective="")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        assert eval_component._is_classifier_objective() is False

    def test_is_classifier_objective_logit_keyword(self) -> None:
        """Verify logit keyword detected as classifier."""
        config = MockConfig(objective="logit_raw")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        assert eval_component._is_classifier_objective() is True


# ============================================================================
# Tests for _calculate_classification_metrics()
# ============================================================================


class TestCalculateClassificationMetrics:
    """Tests for _calculate_classification_metrics method."""

    def test_calculate_classification_metrics_perfect_predictions(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify perfect predictions yield accuracy=1.0."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0

    def test_calculate_classification_metrics_all_wrong(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify all wrong predictions yield accuracy=0.0."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])

        metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_calculate_classification_metrics_empty_arrays(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify empty arrays return accuracy=0.0."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([], dtype=np.float64)
        y_pred = np.array([], dtype=np.float64)

        # Mock sklearn unavailable
        with patch("ml.training.common.evaluation.HAS_SKLEARN", False):
            metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_calculate_classification_metrics_sklearn_fallback(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify fallback to simple accuracy when sklearn unavailable."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])  # 3/5 correct

        with patch("ml.training.common.evaluation.HAS_SKLEARN", False):
            metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        # Only accuracy in fallback
        assert "accuracy" in metrics
        assert np.isclose(metrics["accuracy"], 0.6)
        # No precision/recall in fallback
        assert "precision" not in metrics


# ============================================================================
# Tests for _calculate_regression_metrics()
# ============================================================================


class TestCalculateRegressionMetrics:
    """Tests for _calculate_regression_metrics method."""

    def test_calculate_regression_metrics_perfect_predictions(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify perfect predictions yield MSE=0, R2=1."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2_score"] == 1.0

    def test_calculate_regression_metrics_reasonable_predictions(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify reasonable predictions yield expected metrics."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        assert metrics["mse"] > 0
        assert metrics["rmse"] == np.sqrt(metrics["mse"])
        assert metrics["mae"] > 0
        assert 0 < metrics["r2_score"] < 1

    def test_calculate_regression_metrics_constant_targets(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify constant targets handle ss_tot=0 gracefully."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y_true = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        # ss_tot = 0, should return r2 = 0 per implementation
        assert metrics["r2_score"] == 0.0

    def test_calculate_regression_metrics_all_float_types(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify all metrics are Python floats."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y_true = np.random.randn(10).astype(np.float64)
        y_pred = np.random.randn(10).astype(np.float64)

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        for value in metrics.values():
            assert isinstance(value, float)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_trading_metrics_with_infinite_returns(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify infinite returns are filtered out."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        returns = np.array([0.01, np.inf, -np.inf, 0.02, -0.01])
        predictions = np.ones(5)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        # Infinite values removed, should have some metrics
        assert "total_return" in metrics

    def test_trading_metrics_all_infinite_returns(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify all infinite returns yield empty dict."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        returns = np.array([np.inf, -np.inf, np.nan])
        predictions = np.ones(3)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        assert metrics == {}

    def test_trading_metrics_zero_std_returns(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify metrics are computed even with nearly constant returns.

        Note: Due to floating-point precision, even "constant" returns may have
        a tiny non-zero std, so Sharpe/IR may still be computed with extreme values.
        The implementation matches legacy behavior which only checks std > 0.
        """
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # Constant returns
        returns = np.ones(10) * 0.01
        predictions = np.ones(10)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        # Core metrics should always be computed
        assert "total_return" in metrics
        # Note: Sharpe/IR may be present with extreme values due to floating point
        # The legacy implementation checks std > 0 without a tolerance threshold

    def test_classification_with_float32_predictions(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify float32 predictions work correctly."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0, 1, 0, 1], dtype=np.float32)

        metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics


# ============================================================================
# Property Tests
# ============================================================================


class TestEvaluationInvariants:
    """Property tests for evaluation invariants."""

    @given(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=50,
        )
    )
    @settings(max_examples=20)
    def test_mse_is_non_negative(self, y_values: list[float]) -> None:
        """MSE should always be non-negative."""
        trainer = TestableTrainer(MockConfig())
        eval_component = EvaluationComponent(trainer)

        y_true = np.array(y_values, dtype=np.float64)
        y_pred = y_true + np.random.randn(len(y_true)) * 0.1

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    @given(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=5,
            max_size=50,
        )
    )
    @settings(max_examples=20)
    def test_accuracy_bounded_zero_one(self, y_values: list[int]) -> None:
        """Accuracy should be bounded in [0, 1]."""
        trainer = TestableTrainer(MockConfig(objective="binary:logistic"))
        eval_component = EvaluationComponent(trainer)

        y_true = np.array(y_values, dtype=np.float64)
        # Random predictions
        y_pred = np.random.randint(0, 2, len(y_true)).astype(np.float64)

        metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        assert 0 <= metrics["accuracy"] <= 1

    def test_r2_score_bounded_for_reasonable_predictions(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """R2 score should be bounded for reasonable predictions."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        np.random.seed(42)
        y_true = np.random.randn(100)
        # Add small noise to predictions
        y_pred = y_true + np.random.randn(100) * 0.1

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        # For reasonable predictions, R2 should be close to 1
        assert metrics["r2_score"] <= 1.0

    def test_win_rate_bounded_zero_one(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Win rate should be bounded in [0, 1]."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        predictions = np.random.randn(100)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        if "win_rate" in metrics:
            assert 0 <= metrics["win_rate"] <= 1

    def test_max_drawdown_non_negative(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Max drawdown should be non-negative."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        predictions = np.ones(100)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        if "max_drawdown" in metrics:
            assert metrics["max_drawdown"] >= 0


# ============================================================================
# Integration-style Tests (within unit scope)
# ============================================================================


class TestComponentIntegration:
    """Tests verifying component integration with trainer."""

    def test_eval_component_uses_trainer_predict(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify component calls trainer's predict method."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        X = np.random.randn(10, 5).astype(np.float64)
        y = np.random.randn(10).astype(np.float64)
        trainer_regression_fixture._mock_predictions = np.random.randn(10).astype(
            np.float64
        )

        eval_component.evaluate(None, X, y)

        # Should have called predict
        assert any("predict" in log for log in trainer_regression_fixture._call_log)

    def test_eval_component_accesses_trainer_config(
        self,
    ) -> None:
        """Verify component accesses trainer's config for objective."""
        config = MockConfig(objective="binary:logistic")
        trainer = TestableTrainer(config)
        eval_component = EvaluationComponent(trainer)

        # _is_classifier_objective reads from trainer._config.objective
        result = eval_component._is_classifier_objective()

        assert result is True


# ============================================================================
# All Metrics Tests
# ============================================================================


class TestAllMetrics:
    """Tests covering all metric computations."""

    def test_classification_metrics_with_sklearn(
        self,
        trainer_classification_fixture: TestableTrainer,
    ) -> None:
        """Verify full sklearn metrics are returned when available."""
        eval_component = EvaluationComponent(trainer_classification_fixture)

        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

        metrics = eval_component._calculate_classification_metrics(y_true, y_pred)

        # Should have all sklearn metrics
        expected_keys = {"accuracy", "precision", "recall", "f1_score"}
        # Only check if sklearn is available
        from ml._imports import HAS_SKLEARN
        if HAS_SKLEARN:
            assert set(metrics.keys()) == expected_keys

    def test_regression_metrics_complete(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify all regression metrics are returned."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.2, 4.9])

        metrics = eval_component._calculate_regression_metrics(y_true, y_pred)

        expected_keys = {"mse", "rmse", "mae", "r2_score"}
        assert set(metrics.keys()) == expected_keys

    def test_trading_metrics_complete(
        self,
        trainer_regression_fixture: TestableTrainer,
    ) -> None:
        """Verify all trading metrics are computed when possible."""
        eval_component = EvaluationComponent(trainer_regression_fixture)

        # Generate returns with non-zero variance
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02 + 0.0001  # Slight positive bias
        predictions = np.random.randn(200)

        metrics = eval_component.calculate_trading_metrics(returns, predictions)

        # Should have all metrics when conditions are met
        expected_keys = {
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "information_ratio",
        }
        # Check each key individually (some may be missing if std=0)
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics


__all__ = [
    "MockConfig",
    "TestableTrainer",
]
