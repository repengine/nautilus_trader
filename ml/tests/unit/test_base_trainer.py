# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Unit tests for ML trainer base classes.

Tests cover:
- BaseMLTrainer initialization and configuration
- Training pipeline orchestration
- Data preparation and feature engineering
- Model evaluation and metrics calculation
- Trading-specific performance metrics
- Model serialization and loading
- Error handling and edge cases

"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.training.base import BaseMLTrainer


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


class MockModel:
    """
    Mock ML model for testing.
    """

    def __init__(self) -> None:
        self.is_fitted = False
        self.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.25, 0.15])

    def fit(self, X: Any, y: Any) -> MockModel:
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        # Return dummy predictions
        return rng.choice([-1, 0, 1], size=len(X))

    def score(self, X: Any, y: Any) -> float:
        # Return dummy score
        return 0.75


class MockMLTrainer(BaseMLTrainer):
    """
    Mock implementation of BaseMLTrainer for testing.
    """

    def __init__(self, config: MLTrainingConfig):
        super().__init__(config)
        self.model = MockModel()
        self.prepare_data_called = False
        self.train_model_called = False
        self.evaluate_called = False

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Mock data preparation.
        """
        self.prepare_data_called = True
        # Create dummy features and target
        n_samples = len(data)

        # Work around pandas/numpy issue by extracting columns individually
        feature_names = []
        feature_values = []

        for col in data.columns:
            if col != target_col and col != "timestamp":
                feature_names.append(col)
                feature_values.append(data[col].to_numpy())

        if feature_values:
            X = np.column_stack(feature_values)
        else:
            # Create dummy features if none exist
            X = rng.standard_normal((n_samples, 5))
            feature_names = [f"feature_{i}" for i in range(5)]

        # Create or extract target
        if target_col in data.columns:
            y = data[target_col].to_numpy()
        else:
            y = rng.choice([-1, 0, 1], size=n_samples)

        metadata = {"feature_names": feature_names}
        return X, y, metadata

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Mock model training.
        """
        self.train_model_called = True
        self.model.fit(X_train, y_train)
        return {
            "model": self.model,
            "metrics": {
                "train_score": 0.85,
                "val_score": 0.75,
            },
        }


class TestBaseMLTrainer:
    """
    Test BaseMLTrainer base class.
    """

    @pytest.fixture
    def training_config(self) -> MLTrainingConfig:
        """
        Create test training configuration.
        """
        return MLTrainingConfig(
            data_source="test_data.parquet",
            feature_config=MLFeatureConfig(
                lookback_window=100,
                indicators={
                    "sma": {"period": 20},
                    "rsi": {"period": 14},
                },
                feature_names=["open", "high", "low", "close", "volume"],
                normalize_features=True,
            ),
            save_model_path="test_output/test_model.pkl",
            train_test_split=0.8,
            random_seed=42,
        )

    @pytest.fixture
    def training_config_no_save(self) -> MLTrainingConfig:
        """
        Create test training configuration without save path.
        """
        return MLTrainingConfig(
            data_source="test_data.parquet",
            train_test_split=0.8,
            random_seed=42,
        )

    @pytest.fixture
    def trainer(self, training_config: MLTrainingConfig) -> MockMLTrainer:
        """
        Create test trainer instance.
        """
        return MockMLTrainer(training_config)

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """
        Create sample trading data.
        """
        # Using rng = np.random.default_rng(42) instead  # For reproducibility
        n = 1000

        # Generate price data
        open_prices = 100.0 + np.cumsum(rng.standard_normal(n))
        high_prices = open_prices + np.abs(rng.standard_normal(n))
        low_prices = open_prices - np.abs(rng.standard_normal(n))
        close_prices = low_prices + (high_prices - low_prices) * rng.random(n)
        volumes = rng.integers(1000, 10000, n)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
        )

        return df

    def test_initialization(self, training_config: MLTrainingConfig) -> None:
        """
        Test trainer initialization with configuration.
        """
        # Act
        trainer = MockMLTrainer(training_config)

        # Assert
        assert trainer._config == training_config
        assert trainer.model is not None
        assert trainer._is_fitted is False

    def test_train_pipeline(self, trainer: MockMLTrainer, sample_data: pd.DataFrame) -> None:
        """
        Test full training pipeline execution.
        """
        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(sample_data)

        # Assert
        assert trainer.prepare_data_called is True
        assert trainer.train_model_called is True
        assert trainer._is_fitted is True
        assert "model" in results
        assert "metrics" in results
        assert "feature_names" in results

    def test_train_saves_model(self, sample_data: pd.DataFrame) -> None:
        """
        Test that training saves the model to disk.
        """
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MLTrainingConfig(
                data_source="test_data.parquet",
                save_model_path=str(Path(tmpdir) / "test_model.pkl"),
                train_test_split=0.8,
            )
            trainer = MockMLTrainer(config)

            # Act
            with patch("ml.training.base.HAS_POLARS", True):
                trainer.train(sample_data)

            # Assert
            model_path = Path(config.save_model_path)  # type: ignore[arg-type]
            assert model_path.exists()

    def test_train_no_save_path(self, trainer: MockMLTrainer, sample_data: pd.DataFrame) -> None:
        """
        Test training without save_model_path doesn't save.
        """
        # Arrange
        config = MLTrainingConfig(
            data_source="test_data.parquet",
            train_test_split=0.8,
        )
        trainer = MockMLTrainer(config)

        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(sample_data)

        # Assert
        assert trainer._is_fitted is True
        assert "model" in results

    def test_train_saves_metrics(self, trainer: MockMLTrainer, sample_data: pd.DataFrame) -> None:
        """
        Test that training saves evaluation metrics.
        """
        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(sample_data)

        # Assert
        assert "metrics" in results
        assert trainer._training_metrics is not None
        assert "training_time" in trainer._training_metrics
        assert "training_samples" in trainer._training_metrics
        assert "validation_samples" in trainer._training_metrics

    def test_train_with_validation_split(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test training with train/validation split.
        """
        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(sample_data)

        # Assert
        # Verify train/test split was applied (80/20 split)
        assert trainer._is_fitted is True
        metrics = results["metrics"]
        # Check that we have correct number of samples
        assert metrics["training_samples"] == 800  # 80% of 1000
        assert metrics["validation_samples"] == 200  # 20% of 1000

    def test_load_model(self, trainer: MockMLTrainer) -> None:
        """
        Test loading a trained model from disk.
        """
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_data = {
                "model": MockModel(),
                "feature_names": ["f1", "f2", "f3"],
                "training_metrics": {"accuracy": 0.8},
            }
            pickle.dump(model_data, f)
            model_path = f.name

        # Act
        trainer.load_model(model_path)

        # Assert
        assert trainer._model is not None
        assert trainer._is_fitted is True

        # Cleanup
        Path(model_path).unlink()

    def test_load_model_file_not_found(self, trainer: MockMLTrainer) -> None:
        """
        Test loading model with non-existent file.
        """
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            trainer.load_model("non_existent_model.pkl")

    def test_save_model_without_training(self, trainer: MockMLTrainer) -> None:
        """
        Test save_model raises error when model not trained.
        """
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            # Act & Assert
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                trainer.save_model(f.name)

    def test_model_prediction_after_training(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test model prediction after training.
        """
        # Arrange
        with patch("ml.training.base.HAS_POLARS", True):
            trainer.train(sample_data)

        X = rng.standard_normal((10, 5))  # 10 samples, 5 features

        # Act
        predictions = trainer._model.predict(X)

        # Assert
        assert len(predictions) == 10
        assert all(pred in [-1, 0, 1] for pred in predictions)

    def test_feature_names_stored_after_training(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test feature names are stored after training.
        """
        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(sample_data)

        # Assert
        assert trainer._feature_names is not None
        assert len(trainer._feature_names) > 0
        assert trainer._feature_names == results["feature_names"]

    def test_model_has_feature_importance_after_training(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test model has feature importance after training.
        """
        # Arrange
        with patch("ml.training.base.HAS_POLARS", True):
            trainer.train(sample_data)

        # Act
        importance = trainer._model.feature_importances_

        # Assert
        assert isinstance(importance, np.ndarray)
        assert len(importance) == 5
        assert all(0 <= v <= 1 for v in importance)
        assert abs(sum(importance) - 1.0) < 0.01  # Should sum to ~1

    def test_calculate_trading_metrics(self, trainer: MockMLTrainer) -> None:
        """
        Test calculation of trading-specific metrics.
        """
        # Arrange
        returns = np.array([0.01, -0.02, 0.005, 0.015, -0.01, 0.02, -0.005, -0.015])
        predictions = np.array([1, -1, 1, 1, -1, 0, 0, -1])

        # Act
        metrics = trainer.calculate_trading_metrics(returns, predictions)

        # Assert
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert all(isinstance(v, int | float) for v in metrics.values())

    def test_calculate_trading_metrics_with_zero_signals(self, trainer: MockMLTrainer) -> None:
        """
        Test trading metrics when all predictions are neutral (regression case).
        """
        # Arrange - returns but neutral predictions (use floats to avoid classification)
        returns = np.array([0.01, -0.02, 0.015, -0.01])
        predictions = np.array(
            [0.0001, 0.0001, 0.0001, 0.0001],
        )  # Very small to be treated as regression

        # Act
        metrics = trainer.calculate_trading_metrics(returns, predictions)

        # Assert
        # With near-zero predictions, signals would be 1 (positive), so we'd get actual returns
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    def test_train_with_validation_data(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test training with separate validation data.
        """
        # Arrange
        train_data = sample_data[:800]
        val_data = sample_data[800:]

        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            _ = trainer.train(train_data, validation_data=val_data)

        # Assert
        assert trainer._is_fitted is True
        assert trainer.train_model_called is True

    def test_prepare_data_with_empty_dataframe(self, trainer: MockMLTrainer) -> None:
        """
        Test prepare_data with empty dataframe.
        """
        # Arrange
        empty_data = pd.DataFrame()

        # Act
        X, y, metadata = trainer.prepare_data(empty_data)

        # Assert
        assert len(X) == 0
        assert len(y) == 0

    def test_train_with_small_dataset(self, trainer: MockMLTrainer) -> None:
        """
        Test training with small dataset still works.
        """
        # Arrange
        small_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10),
                "close": rng.standard_normal(10),
                "target": rng.choice([-1, 0, 1], 10),
            },
        )

        # Act
        with patch("ml.training.base.HAS_POLARS", True):
            results = trainer.train(small_data)

        # Assert
        assert trainer._is_fitted is True
        assert "model" in results

    def test_evaluate_classification_model(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test that evaluate returns classification metrics.
        """
        # Arrange
        with patch("ml.training.base.HAS_POLARS", True):
            trainer.train(sample_data)
        X = rng.standard_normal((100, 5))
        y = rng.choice([0, 1], 100)  # Binary classification

        # Act
        metrics = trainer.evaluate(trainer._model, X, y)

        # Assert
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], int | float)
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_model_creates_directory(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test that save_model creates output directory if it doesn't exist.
        """
        # Arrange
        with patch("ml.training.base.HAS_POLARS", True):
            trainer.train(sample_data)  # Need to train first

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_dir" / "model.pkl"

            # Act
            trainer.save_model(output_path)

            # Assert
            assert output_path.parent.exists()
            assert output_path.exists()

    def test_feature_engineering_pipeline(
        self,
        trainer: MockMLTrainer,
        sample_data: pd.DataFrame,
    ) -> None:
        """
        Test that feature engineering is applied during data preparation.
        """
        # Act
        X, y, metadata = trainer.prepare_data(sample_data)

        # Assert
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.shape[1] == 5  # 5 features (excluding timestamp)
        assert "feature_names" in metadata
