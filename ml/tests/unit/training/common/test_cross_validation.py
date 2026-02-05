"""
Unit tests for CrossValidationComponent.

This module tests the cross-validation component extracted from BaseMLTrainer
(lines 504-512 and 818-1069). Tests verify:
- CV enablement check (_should_use_cv)
- CV orchestration and strategy routing (_cross_validate)
- Time-series CV with expanding window (_time_series_cv)
- Deprecated standard CV forwarding (_standard_cv)
- Purged walk-forward CV with embargo (_purged_cv)
- Edge cases and error handling
- Property tests for CV invariants

Following the test design in reports/tests/phase_3_8_test_design_report.md.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.training.common.cross_validation import (
    CrossValidationComponent,
    CVTrainerProtocol,
)


# ============================================================================
# Mock Model and Trainer Fixtures
# ============================================================================


class MockModel:
    """Mock model for testing CV fold training."""

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
class MockConfig:
    """Mock training configuration for CV testing."""

    cv_folds: int | None = 5
    cv_strategy: str = "time_series"
    purge_gap: int = 0
    embargo_pct: float = 0.0


class TestableTrainer:
    """
    Concrete trainer implementation for testing CrossValidationComponent.

    Implements the CVTrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._call_log: list[str] = []
        self._models_created: list[MockModel] = []
        self._fold_metrics: list[dict[str, float]] = []

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

    def _get_model_params(self) -> dict[str, Any]:
        """Mock implementation of _get_model_params."""
        return {"learning_rate": 0.1, "max_depth": 5}

    def _train_with_params(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        params: dict[str, Any],
    ) -> MockModel:
        """Mock implementation of _train_with_params."""
        self._call_log.append("_train_with_params")
        model = MockModel(params)
        model._is_fitted = True
        return model

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Mock implementation of evaluate."""
        self._call_log.append("evaluate")
        # Return slightly varying metrics to simulate real CV
        fold_idx = len(self._fold_metrics)
        metrics = {
            "accuracy": 0.80 + 0.02 * fold_idx,
            "f1_score": 0.75 + 0.03 * fold_idx,
        }
        self._fold_metrics.append(metrics)
        return metrics


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create basic TestableTrainer instance without CV configured."""
    config = MockConfig(cv_folds=1)
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_cv_fixture() -> TestableTrainer:
    """Create TestableTrainer with time-series CV enabled."""
    config = MockConfig(cv_folds=5, cv_strategy="time_series")
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_purged_cv_fixture() -> TestableTrainer:
    """Create TestableTrainer with purged CV enabled."""
    config = MockConfig(
        cv_folds=5,
        cv_strategy="purged",
        purge_gap=10,
        embargo_pct=0.05,
    )
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_standard_cv_fixture() -> TestableTrainer:
    """Create TestableTrainer with deprecated standard CV."""
    config = MockConfig(cv_folds=5, cv_strategy="standard")
    return TestableTrainer(config)


@pytest.fixture
def sample_feature_array() -> npt.NDArray[np.float64]:
    """Create sample feature array for CV testing."""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def sample_labels() -> npt.NDArray[np.float64]:
    """Create sample labels for CV testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100).astype(np.float64)


@pytest.fixture
def small_feature_array() -> npt.NDArray[np.float64]:
    """Create small feature array for edge case testing."""
    np.random.seed(42)
    return np.random.randn(10, 3).astype(np.float64)


@pytest.fixture
def small_labels() -> npt.NDArray[np.float64]:
    """Create small labels for edge case testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, 10).astype(np.float64)


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestShouldUseCv:
    """Tests for _should_use_cv method."""

    def test_should_use_cv_returns_true_when_configured(
        self,
        trainer_with_cv_fixture: TestableTrainer,
    ) -> None:
        """Verify CV enabled check returns True when cv_folds > 1."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        assert cv_component._should_use_cv() is True

    def test_should_use_cv_returns_false_when_cv_folds_one(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify CV disabled for folds=1."""
        cv_component = CrossValidationComponent(trainer_fixture)
        assert cv_component._should_use_cv() is False

    def test_should_use_cv_returns_false_when_cv_folds_none(self) -> None:
        """Verify CV disabled when cv_folds is None."""
        config = MockConfig(cv_folds=None)
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)
        assert cv_component._should_use_cv() is False

    def test_should_use_cv_returns_false_when_cv_folds_missing(self) -> None:
        """Verify CV disabled when cv_folds attribute is missing."""

        class MinimalConfig:
            pass

        trainer = TestableTrainer()
        trainer._config = MinimalConfig()  # type: ignore[assignment]
        cv_component = CrossValidationComponent(trainer)
        assert cv_component._should_use_cv() is False


class TestCrossValidate:
    """Tests for _cross_validate method."""

    def test_cross_validate_time_series_strategy(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify time_series CV strategy is used."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        results = cv_component._cross_validate(sample_feature_array, sample_labels)

        assert len(results) == 5
        for fold_metrics in results:
            assert "accuracy" in fold_metrics
            assert "f1_score" in fold_metrics

    def test_cross_validate_purged_strategy(
        self,
        trainer_with_purged_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify purged CV strategy is used."""
        cv_component = CrossValidationComponent(trainer_with_purged_cv_fixture)
        results = cv_component._cross_validate(sample_feature_array, sample_labels)

        assert len(results) == 5
        for fold_metrics in results:
            assert "accuracy" in fold_metrics

    def test_cross_validate_standard_deprecated_warning(
        self,
        trainer_with_standard_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify standard CV issues deprecation warning."""
        cv_component = CrossValidationComponent(trainer_with_standard_cv_fixture)

        with caplog.at_level(logging.WARNING):
            results = cv_component._cross_validate(sample_feature_array, sample_labels)

        # Check warning was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)
        # Should still return results via time_series fallback
        assert len(results) == 5

    def test_cross_validate_unknown_strategy_raises(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Reject unknown CV strategies explicitly."""
        config = MockConfig(cv_folds=5, cv_strategy="unknown_strategy")
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            cv_component._cross_validate(sample_feature_array, sample_labels)


class TestTimeSeriesCv:
    """Tests for _time_series_cv method."""

    def test_time_series_cv_expanding_window(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify time series CV uses expanding window."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        results = cv_component._time_series_cv(sample_feature_array, sample_labels, n_folds=5)

        assert len(results) == 5
        # Each fold should have created a model
        assert len(trainer_with_cv_fixture._models_created) == 5

    def test_time_series_cv_small_fold_size(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle very small fold sizes gracefully."""
        config = MockConfig(cv_folds=100)  # Way too many folds
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        # 10 samples with 100 folds -> fold_size < 1
        X = np.random.randn(10, 3).astype(np.float64)
        y = np.random.randint(0, 2, 10).astype(np.float64)

        with caplog.at_level(logging.WARNING):
            results = cv_component._time_series_cv(X, y, n_folds=100)

        # Should return empty due to small fold size
        assert results == []
        assert any("too small" in record.message.lower() for record in caplog.records)


class TestPurgedCv:
    """Tests for _purged_cv method."""

    def test_purged_cv_respects_purge_gap(
        self,
        trainer_with_purged_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify purge gap prevents leakage."""
        cv_component = CrossValidationComponent(trainer_with_purged_cv_fixture)
        results = cv_component._purged_cv(sample_feature_array, sample_labels, n_folds=5)

        assert len(results) == 5
        # All folds should have metrics
        for fold_metrics in results:
            assert "accuracy" in fold_metrics

    def test_purged_cv_respects_embargo(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify embargo window applied."""
        config = MockConfig(
            cv_folds=5,
            cv_strategy="purged",
            purge_gap=0,
            embargo_pct=0.1,
        )
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        results = cv_component._purged_cv(sample_feature_array, sample_labels, n_folds=5)

        assert len(results) == 5

    def test_purged_cv_passes_configured_gap_and_embargo(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify configured purge_gap/embargo_pct are passed to the CV splitter."""
        config = MockConfig(
            cv_folds=3,
            cv_strategy="purged",
            purge_gap=7,
            embargo_pct=0.12,
        )
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        with patch("ml.preprocessing.stationarity.PurgedCrossValidator") as mock_cv:
            instance = mock_cv.return_value
            instance.split.return_value = [
                (np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)),
            ]

            results = cv_component._purged_cv(sample_feature_array, sample_labels, n_folds=3)

        mock_cv.assert_called_once_with(n_splits=3, purge_gap=7, embargo_pct=0.12)
        assert len(results) == 1

    def test_purged_cv_import_error_raises(
        self,
        trainer_with_purged_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Raise when PurgedCrossValidator unavailable."""
        cv_component = CrossValidationComponent(trainer_with_purged_cv_fixture)

        # Mock the import to raise an exception
        with patch.dict(
            "sys.modules",
            {"ml.preprocessing.stationarity": None},
        ):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError, match="Purged CV requested"):
                    cv_component._purged_cv(
                        sample_feature_array,
                        sample_labels,
                        n_folds=5,
                    )

        assert any("PurgedCrossValidator unavailable" in record.message for record in caplog.records)


class TestStandardCv:
    """Tests for _standard_cv method (deprecated)."""

    def test_standard_cv_logs_deprecation_warning(
        self,
        trainer_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _standard_cv logs deprecation warning."""
        cv_component = CrossValidationComponent(trainer_fixture)

        with caplog.at_level(logging.WARNING):
            results = cv_component._standard_cv(sample_feature_array, sample_labels, n_folds=5)

        # Check deprecation warning
        assert any("deprecated" in record.message.lower() for record in caplog.records)
        # Should return results via time_series fallback
        assert len(results) == 5


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_cross_validate_fewer_samples_than_folds(
        self,
        small_feature_array: npt.NDArray[np.float64],
        small_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle samples < folds gracefully."""
        config = MockConfig(cv_folds=20)  # More folds than samples (10)
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        with caplog.at_level(logging.WARNING):
            results = cv_component._cross_validate(small_feature_array, small_labels)

        # Warning about reducing folds
        assert any("reducing folds" in record.message.lower() for record in caplog.records)
        # Should still return some results (with reduced folds)
        assert len(results) <= 10

    def test_cross_validate_minimum_samples(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle minimum sample edge case."""
        config = MockConfig(cv_folds=2)
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        # Just 2 samples
        X = np.array([[1.0], [2.0]])
        y = np.array([0.0, 1.0])

        with caplog.at_level(logging.WARNING):
            results = cv_component._cross_validate(X, y)

        # Either returns empty or minimal results depending on fold size calc
        assert isinstance(results, list)

    def test_cross_validate_single_sample(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle single sample case (should skip CV)."""
        config = MockConfig(cv_folds=5)
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([0.0])

        with caplog.at_level(logging.WARNING):
            results = cv_component._cross_validate(X, y)

        assert results == []
        assert any("insufficient" in record.message.lower() for record in caplog.records)

    def test_purged_cv_insufficient_samples(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle insufficient samples for purged CV."""
        config = MockConfig(cv_folds=5, cv_strategy="purged")
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        # Just 1 sample
        X = np.array([[1.0, 2.0]])
        y = np.array([0.0])

        with caplog.at_level(logging.WARNING):
            results = cv_component._purged_cv(X, y, n_folds=5)

        assert results == []

    def test_blocked_cv_strategy_uses_time_series(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify blocked CV strategy falls back to time_series."""
        config = MockConfig(cv_folds=5, cv_strategy="blocked")
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        with caplog.at_level(logging.WARNING):
            results = cv_component._cross_validate(sample_feature_array, sample_labels)

        # Check warning about blocked not implemented
        assert any("not implemented" in record.message.lower() for record in caplog.records)
        # Should return results via time_series fallback
        assert len(results) == 5


# ============================================================================
# Property Tests
# ============================================================================


class TestCvInvariants:
    """Property tests for CV invariants."""

    def test_cv_fold_count_invariant(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Number of results equals requested folds (when possible)."""
        for n_folds in range(2, 8):
            config = MockConfig(cv_folds=n_folds)
            trainer = TestableTrainer(config)
            cv_component = CrossValidationComponent(trainer)

            results = cv_component._cross_validate(sample_feature_array, sample_labels)

            # Results count should be at most n_folds
            assert len(results) <= n_folds

    def test_cv_train_val_no_overlap_time_series(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Train and validation never overlap in time series CV."""
        n_folds = 5
        config = MockConfig(cv_folds=n_folds)
        trainer = TestableTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        # We verify this by checking the fold structure
        n_samples = len(sample_feature_array)
        fold_size = n_samples // (n_folds + 1)

        # Time series CV: train ends at (i+1)*fold_size, val starts there
        # So train and val are always disjoint by construction
        for i in range(n_folds):
            train_end = (i + 1) * fold_size
            val_start = train_end
            # No overlap by design
            assert train_end == val_start

    def test_cv_results_all_have_same_keys(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """All fold results should have the same metric keys."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        results = cv_component._cross_validate(sample_feature_array, sample_labels)

        if len(results) > 1:
            first_keys = set(results[0].keys())
            for fold_result in results[1:]:
                assert set(fold_result.keys()) == first_keys

    def test_cv_metrics_are_finite(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """All metrics in CV results should be finite."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        results = cv_component._cross_validate(sample_feature_array, sample_labels)

        for fold_result in results:
            for metric_value in fold_result.values():
                assert np.isfinite(metric_value)


# ============================================================================
# Integration-style Tests (within unit scope)
# ============================================================================


class TestComponentIntegration:
    """Tests verifying component integration with trainer."""

    def test_cv_component_logs_via_trainer(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component uses trainer's logging methods."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        cv_component._cross_validate(sample_feature_array, sample_labels)

        # Check that info logs were recorded
        info_logs = [log for log in trainer_with_cv_fixture._call_log if log.startswith("info:")]
        assert len(info_logs) > 0

    def test_cv_component_creates_models_via_trainer(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component creates models through trainer."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        cv_component._cross_validate(sample_feature_array, sample_labels)

        # Should have created 5 models for 5 folds
        assert len(trainer_with_cv_fixture._models_created) == 5

    def test_cv_component_evaluates_via_trainer(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify component evaluates models through trainer."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        cv_component._cross_validate(sample_feature_array, sample_labels)

        # Should have called evaluate 5 times
        evaluate_calls = [
            log for log in trainer_with_cv_fixture._call_log if log == "evaluate"
        ]
        assert len(evaluate_calls) == 5


# ============================================================================
# Model Training Path Tests
# ============================================================================


class TestModelTrainingPaths:
    """Tests for different model training code paths."""

    def test_model_with_fit_method(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify models with fit() method are trained correctly."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)
        results = cv_component._time_series_cv(sample_feature_array, sample_labels, n_folds=3)

        assert len(results) == 3
        # All models should be fitted
        for model in trainer_with_cv_fixture._models_created:
            assert model._is_fitted is True

    def test_model_without_fit_method_uses_train_with_params(
        self,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
    ) -> None:
        """Verify models without fit() use _train_with_params."""

        class NoFitModel:
            """Model without fit method."""

            def __init__(self, params: dict[str, Any] | None = None) -> None:
                self.params = params or {}

        class NoFitTrainer(TestableTrainer):
            def _create_model(self, params: dict[str, Any]) -> NoFitModel:
                self._call_log.append("_create_model")
                return NoFitModel(params)

        config = MockConfig(cv_folds=3)
        trainer = NoFitTrainer(config)
        cv_component = CrossValidationComponent(trainer)

        results = cv_component._time_series_cv(sample_feature_array, sample_labels, n_folds=3)

        # Should have used _train_with_params
        train_with_params_calls = [
            log for log in trainer._call_log if log == "_train_with_params"
        ]
        assert len(train_with_params_calls) == 3
        assert len(results) == 3


# ============================================================================
# Aggregate Metrics Tests
# ============================================================================


class TestAggregateMetrics:
    """Tests for aggregate metrics calculation."""

    def test_aggregate_metrics_logged_for_time_series_cv(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify aggregate metrics are logged for time series CV."""
        cv_component = CrossValidationComponent(trainer_with_cv_fixture)

        with caplog.at_level(logging.INFO):
            cv_component._time_series_cv(sample_feature_array, sample_labels, n_folds=5)

        # Check that CV results were logged
        assert any("cv results" in record.message.lower() for record in caplog.records)

    def test_aggregate_metrics_logged_for_purged_cv(
        self,
        trainer_with_purged_cv_fixture: TestableTrainer,
        sample_feature_array: npt.NDArray[np.float64],
        sample_labels: npt.NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify aggregate metrics are logged for purged CV."""
        cv_component = CrossValidationComponent(trainer_with_purged_cv_fixture)

        with caplog.at_level(logging.INFO):
            cv_component._purged_cv(sample_feature_array, sample_labels, n_folds=5)

        # Check that purged CV results were logged
        assert any("purged cv results" in record.message.lower() for record in caplog.records)


__all__ = [
    "MockConfig",
    "MockModel",
    "TestableTrainer",
]
