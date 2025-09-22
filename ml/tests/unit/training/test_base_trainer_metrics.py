#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from ml.config.base import MLTrainingConfig
from ml.training.base import BaseMLTrainer


class _DummyTrainer(BaseMLTrainer):
    def prepare_data(
        self,
        data: object,
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return np.empty((0, 0)), np.empty((0,), dtype=np.float64), {}

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object]:
        return {"model": object(), "metrics": {}}

    def predict(self, model: object, X: np.ndarray, **kwargs: object) -> np.ndarray:
        return np.zeros(shape=len(X), dtype=np.float32)

    def _create_model(self, params: dict[str, object]) -> object:  # pragma: no cover - not used
        return object()

    def _get_model_params(self) -> dict[str, object]:  # pragma: no cover - not used
        return {}

    def _convert_to_onnx(self, model: object, path: Path) -> None:  # pragma: no cover
        return None

    def _suggest_hyperparameters(self, trial: object) -> dict[str, object]:  # pragma: no cover
        return {}


@pytest.fixture
def trainer() -> _DummyTrainer:
    return _DummyTrainer(MLTrainingConfig(data_source="memory"))


class _ClassifierTrainer(_DummyTrainer):
    def _is_classifier_objective(self) -> bool:
        return True


def test_calculate_optuna_metric_accuracy(trainer: _DummyTrainer) -> None:
    y_true = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    y_pred = np.array([0.2, 0.7, 0.8, 0.1], dtype=np.float32)
    score = trainer._calculate_optuna_metric("accuracy", y_true, y_pred)
    assert pytest.approx(score) == 1.0


def test_calculate_optuna_metric_rmse(trainer: _DummyTrainer) -> None:
    y_true = np.array([0.5, 0.9], dtype=np.float64)
    y_pred = np.array([0.4, 1.1], dtype=np.float32)
    score = trainer._calculate_optuna_metric("rmse", y_true, y_pred)
    assert pytest.approx(score, rel=1e-6) == pytest.approx(np.sqrt(((0.5 - 0.4) ** 2 + (0.9 - 1.1) ** 2) / 2))


def test_calculate_optuna_metric_sharpe(trainer: _DummyTrainer) -> None:
    preds = np.array([0.9, 0.2, 0.7], dtype=np.float32)
    y_labels = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    returns = np.array([0.01, -0.02, 0.015], dtype=np.float64)
    score = trainer._calculate_optuna_metric(
        "sharpe_ratio",
        y_labels,
        preds,
        validation_returns=returns,
    )
    assert score != 0.0


def test_calculate_trading_metrics_binary(tmp_path: Path) -> None:
    trainer = _ClassifierTrainer(MLTrainingConfig(data_source="memory"))
    returns = np.array([0.01, -0.02, 0.015], dtype=np.float64)
    predictions = np.array([0.8, 0.4, 0.9], dtype=np.float32)
    metrics = trainer.calculate_trading_metrics(returns, predictions)
    assert metrics["total_return"] != 0.0
