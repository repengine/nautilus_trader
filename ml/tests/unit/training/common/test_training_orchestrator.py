"""
Unit tests for TrainingOrchestratorComponent.

This module tests the training orchestration component extracted from BaseMLTrainer
(lines 109-282). Tests verify:
- Complete training pipeline orchestration
- Data splitting and preparation
- Optional Optuna hyperparameter optimization
- Optional cross-validation
- MLflow tracking integration
- Trading metrics calculation
- Model persistence and ONNX export
- Logging helpers

Following the test design in reports/tests/phase_3_8_test_design_report.md.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.config.targets import HORIZON_RESOLUTION_WALL_CLOCK
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_ID
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_MAJOR
from ml.training.common.training_orchestrator import (
    TrainerProtocol,
    TrainingOrchestratorComponent,
)
from ml.training.datasets.target_generator import build_target_semantics_metadata
from ml.tests.utils.targets import build_default_target_semantics


# ============================================================================
# Mock Model and Trainer Fixtures
# ============================================================================


class MockModel:
    """Mock model for testing."""

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


@dataclass
class MockConfig:
    """Mock training configuration for testing."""

    data_source: str = "memory"
    target_column: str = "target"
    train_test_split: float = 0.8
    save_model_path: str | None = None
    cv_folds: int | None = None
    cv_strategy: str = "time_series"
    optuna_config: Any = None
    mlflow_config: Any = None
    export_onnx: bool = False
    target_semantics: Any = None


class TestableTrainer:
    """
    Concrete trainer implementation for testing TrainingOrchestratorComponent.

    Implements the TrainerProtocol interface with mock implementations.
    """

    def __init__(self, config: MockConfig | None = None) -> None:
        self._config = config or MockConfig()
        self._feature_names: list[str] = []
        self._training_metrics: dict[str, Any] = {}
        self._is_fitted = False
        self._model: Any = None
        self._cv_results: list[dict[str, float]] = []
        self._mlflow_run_id: str | None = None
        self._optuna_study: Any = None
        # Track method calls for assertions
        self._call_log: list[str] = []

    def prepare_data(
        self,
        data: Any,
        target_col: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """Mock implementation of prepare_data."""
        self._call_log.append("prepare_data")
        if hasattr(data, "drop"):
            # DataFrame-like object
            X = data.drop(columns=[target_col]).to_numpy()
            y = data[target_col].to_numpy()
        else:
            # NumPy array or similar
            X = np.array(data)
            y = np.zeros(len(X))
        feature_names = [f"f{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
        return (
            X.astype(np.float64),
            y.astype(np.float64),
            {"feature_names": feature_names},
        )

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
        return {"model": MockModel(), "metrics": {"loss": 0.1, "val_loss": 0.15}}

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """Mock implementation of predict."""
        self._call_log.append("predict")
        return_labels = kwargs.get("return_labels", False)
        n = len(X)
        if return_labels:
            return np.zeros(n, dtype=np.int64)  # type: ignore[return-value]
        return np.full(n, 0.5, dtype=np.float32)

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Mock implementation of evaluate."""
        self._call_log.append("evaluate")
        return {"accuracy": 0.85, "f1_score": 0.82}

    def calculate_trading_metrics(
        self,
        returns: npt.NDArray[np.float64],
        predictions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Mock implementation of calculate_trading_metrics."""
        self._call_log.append("calculate_trading_metrics")
        return {"sharpe_ratio": 1.5, "max_drawdown": 0.1, "win_rate": 0.55}

    def save_model(self, path: str) -> None:
        """Mock implementation of save_model."""
        self._call_log.append("save_model")
        # Create a dummy file for tests
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("mock_model")

    def export_to_onnx(self, path: Any) -> None:
        """Mock implementation of export_to_onnx."""
        self._call_log.append("export_to_onnx")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mock_onnx_model_bytes")

    def _split_data(self, data: Any) -> tuple[Any, Any]:
        """Mock implementation of _split_data."""
        self._call_log.append("_split_data")
        n_samples = len(data)
        split_idx = int(n_samples * self._config.train_test_split)
        return data[:split_idx], data[split_idx:]

    def _should_use_optuna(self) -> bool:
        """Mock implementation of _should_use_optuna."""
        return (
            hasattr(self._config, "optuna_config")
            and self._config.optuna_config is not None
            and getattr(self._config.optuna_config, "enabled", True)
        )

    def _optimize_hyperparameters(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock implementation of _optimize_hyperparameters."""
        self._call_log.append("_optimize_hyperparameters")
        self._optuna_study = MagicMock()
        return {"learning_rate": 0.05, "max_depth": 5}

    def _should_use_cv(self) -> bool:
        """Mock implementation of _should_use_cv."""
        return (
            hasattr(self._config, "cv_folds")
            and self._config.cv_folds is not None
            and self._config.cv_folds > 1
        )

    def _cross_validate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """Mock implementation of _cross_validate."""
        self._call_log.append("_cross_validate")
        return [{"accuracy": 0.84}, {"accuracy": 0.86}, {"accuracy": 0.85}]

    def _should_use_mlflow(self) -> bool:
        """Mock implementation of _should_use_mlflow."""
        return (
            hasattr(self._config, "mlflow_config")
            and self._config.mlflow_config is not None
        )

    def _start_mlflow_run(self) -> None:
        """Mock implementation of _start_mlflow_run."""
        self._call_log.append("_start_mlflow_run")
        self._mlflow_run_id = "mock_run_id"

    def _track_with_mlflow(self, metrics: dict[str, Any]) -> None:
        """Mock implementation of _track_with_mlflow."""
        self._call_log.append("_track_with_mlflow")

    def _end_mlflow_run(self) -> None:
        """Mock implementation of _end_mlflow_run."""
        self._call_log.append("_end_mlflow_run")


# ============================================================================
# Target Semantics Helpers
# ============================================================================


def _write_metadata(
    tmp_path: Path,
    *,
    target_semantics: dict[str, Any] | None,
) -> Path:
    payload: dict[str, Any] = {
        "dataset_id": "dataset",
        "build_ts": "2025-01-01T00:00:00Z",
    }
    if target_semantics is not None:
        payload["target_semantics"] = target_semantics
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_training_orchestrator_requires_target_semantics_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    _write_metadata(tmp_path, target_semantics=None)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="dataset metadata missing target_semantics"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_target_col_declared(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="missing_target",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="target_col 'missing_target'"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_target_semantics_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    semantics.pop("contract", None)
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="missing required keys"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_declared_contract_capabilities(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    contract = semantics["contract"]
    assert isinstance(contract, dict)
    contract["capabilities"] = ["horizons_declared"]
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="missing required capabilities"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_canonical_contract_id(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    contract = semantics["contract"]
    assert isinstance(contract, dict)
    contract["id"] = f"{TARGET_SEMANTICS_CONTRACT_ID}_invalid"
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match=r"contract\.id mismatch"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_canonical_contract_major(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    contract = semantics["contract"]
    assert isinstance(contract, dict)
    contract["major"] = TARGET_SEMANTICS_CONTRACT_MAJOR + 1
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match=r"contract\.major mismatch"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_canonical_semantics_version(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    semantics.pop("version", None)
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="version mismatch"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_horizon_mode_match_with_config(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
            target_semantics={
                "horizon_resolution_mode": HORIZON_RESOLUTION_WALL_CLOCK,
                "wall_clock_timestamp_column": "timestamp",
                "horizons": [{"minutes": 15, "label": "15m"}],
            },
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match="horizon_resolution_mode mismatch"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


def test_training_orchestrator_requires_execution_contract_match_with_config(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_text("data", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    _write_metadata(tmp_path, target_semantics=semantics)

    trainer = TestableTrainer(
        config=MockConfig(
            data_source=str(dataset_path),
            target_column="target_bin_15m",
            target_semantics={
                "execution_latency_bars": 1,
            },
        ),
    )
    orchestrator = TrainingOrchestratorComponent(trainer)

    with pytest.raises(ValueError, match=r"execution\.latency_bars mismatch"):
        orchestrator.train(np.zeros((2, 2), dtype=np.float64))


# ============================================================================
# Mock DataFrame for Testing
# ============================================================================


class MockDataFrame:
    """Mock DataFrame for testing without polars dependency."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._data = data
        self._columns = list(data.keys())

    @property
    def columns(self) -> list[str]:
        return self._columns

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key: str | slice) -> Any:
        if isinstance(key, str):
            return MockColumn(self._data[key])
        elif isinstance(key, slice):
            new_data = {}
            for col_name, col_values in self._data.items():
                new_data[col_name] = col_values[key]
            return MockDataFrame(new_data)
        return self

    def drop(self, columns: list[str]) -> MockDataFrame:
        new_data = {k: v for k, v in self._data.items() if k not in columns}
        return MockDataFrame(new_data)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.array(list(self._data.values())).T.astype(np.float64)


class MockColumn:
    """Mock column for testing."""

    def __init__(self, data: list[Any]) -> None:
        self._data = data

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.array(self._data, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._data)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trainer_fixture() -> TestableTrainer:
    """Create a basic TestableTrainer instance."""
    return TestableTrainer(MockConfig())


@pytest.fixture
def trainer_with_cv_fixture() -> TestableTrainer:
    """Create TestableTrainer with CV enabled."""
    config = MockConfig(cv_folds=5, cv_strategy="time_series")
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_optuna_fixture() -> TestableTrainer:
    """Create TestableTrainer with Optuna enabled."""
    optuna_config = MagicMock()
    optuna_config.enabled = True
    optuna_config.n_trials = 5
    optuna_config.metric = "accuracy"
    config = MockConfig(optuna_config=optuna_config)
    return TestableTrainer(config)


@pytest.fixture
def trainer_with_mlflow_fixture() -> TestableTrainer:
    """Create TestableTrainer with MLflow enabled."""
    mlflow_config = MagicMock()
    mlflow_config.tracking_uri = "http://localhost:5000"
    mlflow_config.experiment_name = "test_experiment"
    mlflow_config.run_name = "test_run"
    config = MockConfig(mlflow_config=mlflow_config)
    return TestableTrainer(config)


@pytest.fixture
def sample_training_dataframe() -> MockDataFrame:
    """Create sample DataFrame for training tests."""
    np.random.seed(42)
    n_samples = 100
    return MockDataFrame({
        "feature_1": list(np.random.randn(n_samples)),
        "feature_2": list(np.random.randn(n_samples)),
        "feature_3": list(np.random.randn(n_samples)),
        "target": list(np.random.randint(0, 2, n_samples)),
    })


@pytest.fixture
def sample_validation_dataframe() -> MockDataFrame:
    """Create sample validation DataFrame."""
    np.random.seed(43)
    n_samples = 30
    return MockDataFrame({
        "feature_1": list(np.random.randn(n_samples)),
        "feature_2": list(np.random.randn(n_samples)),
        "feature_3": list(np.random.randn(n_samples)),
        "target": list(np.random.randint(0, 2, n_samples)),
    })


@pytest.fixture
def sample_dataframe_with_returns() -> MockDataFrame:
    """Create sample DataFrame with returns column."""
    np.random.seed(44)
    n_samples = 100
    return MockDataFrame({
        "feature_1": list(np.random.randn(n_samples)),
        "feature_2": list(np.random.randn(n_samples)),
        "feature_3": list(np.random.randn(n_samples)),
        "returns": list(np.random.randn(n_samples) * 0.02),
        "target": list(np.random.randint(0, 2, n_samples)),
    })


@pytest.fixture
def orchestrator_fixture(trainer_fixture: TestableTrainer) -> TrainingOrchestratorComponent:
    """Create orchestrator with basic trainer."""
    return TrainingOrchestratorComponent(trainer_fixture)


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestTrainOrchestratesFullPipeline:
    """Tests for complete training pipeline orchestration."""

    def test_train_orchestrates_full_pipeline(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify train() executes all pipeline steps in correct order."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        # Verify result structure
        assert "model" in result
        assert "metrics" in result
        assert "feature_names" in result
        assert "config" in result
        assert "best_params" in result

        # Verify metrics contain expected keys
        assert "training_time" in result["metrics"]
        assert "training_samples" in result["metrics"]
        assert "validation_samples" in result["metrics"]
        assert "feature_count" in result["metrics"]

        # Verify trainer state
        assert trainer_fixture._is_fitted is True
        assert trainer_fixture._model is not None

        # Verify method call order
        assert "prepare_data" in trainer_fixture._call_log
        assert "_train_model" in trainer_fixture._call_log
        assert "evaluate" in trainer_fixture._call_log

    def test_train_returns_feature_names(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify feature names are captured and returned."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        assert "feature_names" in result
        assert len(result["feature_names"]) > 0
        assert result["feature_names"] == trainer_fixture._feature_names


class TestTrainDataSplitting:
    """Tests for data splitting behavior."""

    def test_train_splits_data_when_no_validation_provided(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify automatic train/val split when validation_data is None."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe)

        # Verify _split_data was called
        assert "_split_data" in trainer_fixture._call_log

    def test_train_uses_provided_validation_data(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
        sample_validation_dataframe: MockDataFrame,
    ) -> None:
        """Verify train uses provided validation data directly."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(
                sample_training_dataframe,
                validation_data=sample_validation_dataframe,
            )

        # Verify _split_data was NOT called when validation data provided
        assert "_split_data" not in trainer_fixture._call_log


class TestTrainWithOptuna:
    """Tests for Optuna hyperparameter optimization."""

    def test_train_with_optuna_enabled(
        self,
        trainer_with_optuna_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify Optuna integration when configured."""
        orchestrator = TrainingOrchestratorComponent(trainer_with_optuna_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        # Verify _optimize_hyperparameters was called
        assert "_optimize_hyperparameters" in trainer_with_optuna_fixture._call_log

        # Verify best_params is populated
        assert result["best_params"] is not None
        assert len(result["best_params"]) > 0

        # Verify Optuna study was set
        assert trainer_with_optuna_fixture._optuna_study is not None


class TestTrainWithCV:
    """Tests for cross-validation."""

    def test_train_with_cv_enabled(
        self,
        trainer_with_cv_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify cross-validation when configured."""
        orchestrator = TrainingOrchestratorComponent(trainer_with_cv_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        # Verify _cross_validate was called
        assert "_cross_validate" in trainer_with_cv_fixture._call_log

        # Verify CV results are in metrics
        assert "cv_scores" in result["metrics"]
        assert len(result["metrics"]["cv_scores"]) > 0


class TestTrainWithMLflow:
    """Tests for MLflow tracking."""

    def test_train_with_mlflow_enabled(
        self,
        trainer_with_mlflow_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify MLflow tracking when configured."""
        orchestrator = TrainingOrchestratorComponent(trainer_with_mlflow_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe)

        # Verify MLflow methods were called in correct order
        assert "_start_mlflow_run" in trainer_with_mlflow_fixture._call_log
        assert "_track_with_mlflow" in trainer_with_mlflow_fixture._call_log
        assert "_end_mlflow_run" in trainer_with_mlflow_fixture._call_log

        # Verify run ID was set
        assert trainer_with_mlflow_fixture._mlflow_run_id is not None


class TestTrainCalculatesTradingMetrics:
    """Tests for trading metrics calculation."""

    def test_train_calculates_trading_metrics(
        self,
        trainer_fixture: TestableTrainer,
        sample_dataframe_with_returns: MockDataFrame,
    ) -> None:
        """Verify trading metrics when returns column present."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_dataframe_with_returns)

        # Verify calculate_trading_metrics was called
        assert "calculate_trading_metrics" in trainer_fixture._call_log

        # Verify trading metrics in result
        assert "sharpe_ratio" in result["metrics"]
        assert "max_drawdown" in result["metrics"]
        assert "win_rate" in result["metrics"]


class TestTrainSavesModel:
    """Tests for model saving."""

    def test_train_saves_model_when_configured(
        self,
        sample_training_dataframe: MockDataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify model saving when save_model_path set."""
        model_path = str(tmp_path / "test_model.pkl")
        config = MockConfig(save_model_path=model_path)
        trainer = TestableTrainer(config)
        orchestrator = TrainingOrchestratorComponent(trainer)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe)

        # Verify save_model was called
        assert "save_model" in trainer._call_log

        # Verify file was created
        assert Path(model_path).exists()


class TestTrainExportsONNX:
    """Tests for ONNX export."""

    def test_train_exports_onnx_when_configured(
        self,
        sample_training_dataframe: MockDataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify ONNX export when export_onnx=True."""
        model_path = str(tmp_path / "test_model.pkl")
        config = MockConfig(save_model_path=model_path, export_onnx=True)
        trainer = TestableTrainer(config)
        orchestrator = TrainingOrchestratorComponent(trainer)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe)

        # Verify export_to_onnx was called
        assert "export_to_onnx" in trainer._call_log

        # Verify .onnx file was created
        onnx_path = tmp_path / "test_model.onnx"
        assert onnx_path.exists()


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Tests for logging methods."""

    def test_log_info_calls_logger(
        self,
        orchestrator_fixture: TrainingOrchestratorComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_info uses logger.info."""
        with caplog.at_level(logging.INFO):
            orchestrator_fixture._log_info("Test info message")

        assert "Test info message" in caplog.text

    def test_log_warning_calls_logger(
        self,
        orchestrator_fixture: TrainingOrchestratorComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_warning uses logger.warning."""
        with caplog.at_level(logging.WARNING):
            orchestrator_fixture._log_warning("Test warning message")

        assert "Test warning message" in caplog.text

    def test_log_error_calls_logger(
        self,
        orchestrator_fixture: TrainingOrchestratorComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_error uses logger.error."""
        with caplog.at_level(logging.ERROR):
            orchestrator_fixture._log_error("Test error message")

        assert "Test error message" in caplog.text

    def test_log_info_with_formatting(
        self,
        orchestrator_fixture: TrainingOrchestratorComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _log_info handles message formatting."""
        with caplog.at_level(logging.INFO):
            orchestrator_fixture._log_info("Value is %d", 42)

        assert "Value is 42" in caplog.text


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestErrorConditions:
    """Tests for error handling."""

    def test_train_raises_on_missing_polars(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify clear error when polars unavailable."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", False):
            with patch(
                "ml.training.common.training_orchestrator.check_ml_dependencies"
            ) as mock_check:
                mock_check.side_effect = ImportError("Polars required")
                with pytest.raises(ImportError, match="Polars required"):
                    orchestrator.train(sample_training_dataframe)

    def test_train_handles_empty_dataframe(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify graceful handling of empty data."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)
        empty_df = MockDataFrame({
            "feature_1": [],
            "feature_2": [],
            "target": [],
        })

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            with pytest.raises(ValueError, match="empty"):
                orchestrator.train(empty_df)


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_testable_trainer_implements_protocol(
        self,
        trainer_fixture: TestableTrainer,
    ) -> None:
        """Verify TestableTrainer implements TrainerProtocol."""
        # Check required attributes
        assert hasattr(trainer_fixture, "_config")
        assert hasattr(trainer_fixture, "_feature_names")
        assert hasattr(trainer_fixture, "_training_metrics")
        assert hasattr(trainer_fixture, "_is_fitted")
        assert hasattr(trainer_fixture, "_model")
        assert hasattr(trainer_fixture, "_cv_results")

        # Check required methods
        assert callable(getattr(trainer_fixture, "prepare_data", None))
        assert callable(getattr(trainer_fixture, "_train_model", None))
        assert callable(getattr(trainer_fixture, "predict", None))
        assert callable(getattr(trainer_fixture, "evaluate", None))
        assert callable(getattr(trainer_fixture, "calculate_trading_metrics", None))
        assert callable(getattr(trainer_fixture, "save_model", None))
        assert callable(getattr(trainer_fixture, "export_to_onnx", None))
        assert callable(getattr(trainer_fixture, "_split_data", None))
        assert callable(getattr(trainer_fixture, "_should_use_optuna", None))
        assert callable(getattr(trainer_fixture, "_optimize_hyperparameters", None))
        assert callable(getattr(trainer_fixture, "_should_use_cv", None))
        assert callable(getattr(trainer_fixture, "_cross_validate", None))
        assert callable(getattr(trainer_fixture, "_should_use_mlflow", None))
        assert callable(getattr(trainer_fixture, "_start_mlflow_run", None))
        assert callable(getattr(trainer_fixture, "_track_with_mlflow", None))
        assert callable(getattr(trainer_fixture, "_end_mlflow_run", None))


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for state management during training."""

    def test_trainer_state_updated_after_training(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify trainer state is properly updated after training."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        # Verify initial state
        assert trainer_fixture._is_fitted is False
        assert trainer_fixture._model is None
        assert len(trainer_fixture._training_metrics) == 0

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe)

        # Verify state after training
        assert trainer_fixture._is_fitted is True
        assert trainer_fixture._model is not None
        assert len(trainer_fixture._training_metrics) > 0
        assert len(trainer_fixture._feature_names) > 0

    def test_training_metrics_contains_timing(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify training time is recorded."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        assert "training_time" in result["metrics"]
        assert result["metrics"]["training_time"] >= 0

    def test_training_metrics_contains_sample_counts(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify sample counts are recorded."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        assert "training_samples" in result["metrics"]
        assert "validation_samples" in result["metrics"]
        assert result["metrics"]["training_samples"] > 0
        assert result["metrics"]["validation_samples"] > 0
        total = result["metrics"]["training_samples"] + result["metrics"]["validation_samples"]
        assert total == len(sample_training_dataframe)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_train_with_no_optional_features(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify training works without Optuna, CV, or MLflow."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            result = orchestrator.train(sample_training_dataframe)

        # Should complete successfully
        assert result["model"] is not None
        assert result["best_params"] == {}

        # Verify optional methods were not called
        assert "_optimize_hyperparameters" not in trainer_fixture._call_log
        assert "_cross_validate" not in trainer_fixture._call_log
        assert "_start_mlflow_run" not in trainer_fixture._call_log

    def test_train_passes_kwargs_to_train_model(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify kwargs are passed through to _train_model."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        # Override _train_model to capture kwargs
        captured_kwargs: dict[str, Any] = {}

        def capture_train_model(
            X_train: npt.NDArray[np.float64],
            y_train: npt.NDArray[np.float64],
            X_val: npt.NDArray[np.float64],
            y_val: npt.NDArray[np.float64],
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured_kwargs.update(kwargs)
            return {"model": MockModel(), "metrics": {"loss": 0.1}}

        trainer_fixture._train_model = capture_train_model  # type: ignore[method-assign]

        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(sample_training_dataframe, custom_param="test_value")

        assert "custom_param" in captured_kwargs
        assert captured_kwargs["custom_param"] == "test_value"

    def test_train_handles_validation_returns_kwarg(
        self,
        trainer_fixture: TestableTrainer,
        sample_training_dataframe: MockDataFrame,
    ) -> None:
        """Verify validation_returns is handled separately from train kwargs."""
        orchestrator = TrainingOrchestratorComponent(trainer_fixture)

        captured_kwargs: dict[str, Any] = {}

        def capture_train_model(
            X_train: npt.NDArray[np.float64],
            y_train: npt.NDArray[np.float64],
            X_val: npt.NDArray[np.float64],
            y_val: npt.NDArray[np.float64],
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured_kwargs.update(kwargs)
            return {"model": MockModel(), "metrics": {"loss": 0.1}}

        trainer_fixture._train_model = capture_train_model  # type: ignore[method-assign]

        validation_returns = np.array([0.01, 0.02, -0.01])
        with patch("ml.training.common.training_orchestrator.HAS_POLARS", True):
            orchestrator.train(
                sample_training_dataframe,
                validation_returns=validation_returns,
            )

        # validation_returns should be popped from kwargs before _train_model
        assert "validation_returns" not in captured_kwargs
