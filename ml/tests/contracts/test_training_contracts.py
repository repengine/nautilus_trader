from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ml._imports import optuna
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.training.base_facade import BaseMLTrainerFacade

pytestmark = pytest.mark.contracts


class _StubTrainer(BaseMLTrainerFacade):
    def prepare_data(
        self,
        data: Any,
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        X = np.zeros((1, 1), dtype=np.float64)
        y = np.zeros(1, dtype=np.float64)
        return X, y, {"feature_names": ["x"]}

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"model": object(), "metrics": {"loss": 0.0}}

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        return np.zeros(X.shape[0], dtype=np.float32)

    def _create_model(self, params: dict[str, Any]) -> Any:
        return object()

    def _get_model_params(self) -> dict[str, Any]:
        return {}

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {}

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        path.write_bytes(b"onnx")


def _training_config(**overrides: Any) -> MLTrainingConfig:
    return MLTrainingConfig(
        data_source="unit-test",
        **overrides,
    )


def test_training_facade_uses_imports_for_optional_dependencies() -> None:
    from ml import _imports
    from ml.training import base_facade as base_module

    assert base_module.optuna is _imports.optuna


def test_training_facade_initializes_feature_store_from_db_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeFeatureStore:
        def __init__(
            self,
            *,
            connection_string: str,
            feature_config: MLFeatureConfig,
            pipeline_spec: object | None,
        ) -> None:
            self.connection_string = connection_string
            self.feature_config = feature_config
            self.pipeline_spec = pipeline_spec

    monkeypatch.setattr("ml.stores.feature_store.FeatureStore", _FakeFeatureStore)

    feature_config = MLFeatureConfig()
    config = _training_config(db_connection="sqlite://", feature_config=feature_config)
    trainer = _StubTrainer(config)

    assert isinstance(trainer._feature_store, _FakeFeatureStore)
    assert trainer._feature_store.connection_string == "sqlite://"
    assert trainer._feature_store.feature_config is feature_config


def test_training_facade_exports_onnx_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("ml.training.common.persistence.HAS_ONNX", True)
    trainer = _StubTrainer(_training_config())
    trainer._model = object()
    trainer._is_fitted = True
    target = tmp_path / "model.onnx"

    trainer.export_to_onnx(target)

    assert target.read_bytes() == b"onnx"
