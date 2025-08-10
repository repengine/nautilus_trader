
"""
Unit tests for XGBoostTrainer.

Tests focus on XGBoost-specific functionality while mocking the actual XGBoost training
to ensure test isolation and speed.

"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from ml.config.shared import XGBoostGPUConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.xgboost import XGBoostTrainer


class TestXGBoostTrainerInitialization:
    """
    Test XGBoostTrainer initialization.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_init_with_basic_config(self) -> None:
        """
        Test initialization with basic XGBoost configuration.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary:logistic",
            n_estimators=100,
        )

        # Act
        trainer = XGBoostTrainer(config)

        # Assert
        assert trainer._xgb_config == config
        assert trainer._booster is None
        assert trainer._dtrain is None
        assert trainer._dval is None


class TestXGBoostTrainerModelTraining:
    """
    Test XGBoostTrainer model training.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_train_model_basic(self, mock_xgb: Any) -> None:
        """
        Test basic model training.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary:logistic",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feat1", "feat2"]

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = rng.integers(0, 2, 80).astype(np.float64)
        X_val = rng.standard_normal((20, 2))
        y_val = rng.integers(0, 2, 20).astype(np.float64)

        # Mock XGBoost components
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_xgb.DMatrix.side_effect = [mock_dtrain, mock_dval]

        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_xgb.train.return_value = mock_booster

        # Act
        result = trainer._train_model(X_train, y_train, X_val, y_val)

        # Assert
        assert result["model"] == mock_booster
        assert "metrics" in result
        assert result["metrics"]["best_iteration"] == 50
        mock_xgb.train.assert_called_once()

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_train_model_with_gpu(self, mock_xgb: Any) -> None:
        """
        Test model training with GPU configuration.
        """
        # Arrange
        gpu_config = XGBoostGPUConfig(enabled=True, device_id=0)
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            gpu_config=gpu_config,
        )
        trainer = XGBoostTrainer(config)

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = rng.integers(0, 2, 80).astype(np.float64)
        X_val = rng.standard_normal((20, 2))
        y_val = rng.integers(0, 2, 20).astype(np.float64)

        # Mock XGBoost components
        mock_xgb.DMatrix.return_value = MagicMock()
        mock_xgb.train.return_value = MagicMock()

        # Act
        result = trainer._train_model(X_train, y_train, X_val, y_val)

        # Assert
        # Check that GPU parameters were set
        call_args = mock_xgb.train.call_args[0][0]
        assert call_args["tree_method"] == "hist"
        assert call_args["device"] == "cuda:0"


class TestXGBoostTrainerPrediction:
    """
    Test XGBoostTrainer prediction functionality.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_predict_binary_classification(self, mock_xgb: Any) -> None:
        """
        Test prediction for binary classification.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary:logistic",
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feat1", "feat2"]

        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.7, 0.4, 0.9] + [0.5] * 6)

        mock_dmatrix = MagicMock()
        mock_xgb.DMatrix.return_value = mock_dmatrix

        # Act
        predictions = trainer.predict(mock_model, X)

        # Assert
        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
        mock_model.predict.assert_called_once_with(mock_dmatrix)

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_predict_with_custom_threshold(self, mock_xgb: Any) -> None:
        """
        Test prediction with custom threshold.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary:logistic",
        )
        trainer = XGBoostTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 2))
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.3, 0.6, 0.7, 0.8])

        mock_xgb.DMatrix.return_value = MagicMock()

        # Act
        predictions = trainer.predict(mock_model, X, threshold=0.3)

        # Assert
        expected = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(predictions, expected)

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_predict_regression(self, mock_xgb: Any) -> None:
        """
        Test prediction for regression.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="reg:squarederror",
        )
        trainer = XGBoostTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))
        mock_model = MagicMock()
        expected_predictions = rng.standard_normal(10)
        mock_model.predict.return_value = expected_predictions

        mock_xgb.DMatrix.return_value = MagicMock()

        # Act
        predictions = trainer.predict(mock_model, X)

        # Assert
        np.testing.assert_array_equal(predictions, expected_predictions)


class TestXGBoostTrainerHyperparameters:
    """
    Test XGBoostTrainer hyperparameter functionality.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_get_model_params(self) -> None:
        """
        Test getting default model parameters.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
        )
        trainer = XGBoostTrainer(config)

        # Act
        params = trainer._get_model_params()

        # Assert
        assert params["objective"] == "binary:logistic"
        assert params["eval_metric"] == "logloss"
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.3
        assert params["subsample"] == 0.8
        assert params["seed"] == 42

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_get_model_params_with_scale_pos_weight(self) -> None:
        """
        Test getting model parameters with scale_pos_weight.
        """
        # Arrange
        config = XGBoostTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            scale_pos_weight=2.0,
        )
        trainer = XGBoostTrainer(config)

        # Act
        params = trainer._get_model_params()

        # Assert
        assert params["scale_pos_weight"] == 2.0

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_suggest_hyperparameters_for_optuna(self) -> None:
        """
        Test hyperparameter suggestion for Optuna optimization.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)

        # Mock Optuna trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [6, 1]  # max_depth, min_child_weight
        mock_trial.suggest_float.side_effect = [
            0.1,  # learning_rate
            0.7,  # subsample
            0.8,  # colsample_bytree
            0.5,  # gamma
            1.0,  # reg_alpha
            2.0,  # reg_lambda
        ]

        # Act
        params = trainer._suggest_hyperparameters(mock_trial)

        # Assert
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.1
        assert params["subsample"] == 0.7
        assert params["colsample_bytree"] == 0.8
        assert params["gamma"] == 0.5
        assert params["reg_alpha"] == 1.0
        assert params["reg_lambda"] == 2.0
        assert params["min_child_weight"] == 1


class TestXGBoostTrainerFeatureImportance:
    """
    Test XGBoostTrainer feature importance functionality.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_get_feature_importance_when_fitted(self) -> None:
        """
        Test getting feature importance from fitted model.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["feat1", "feat2", "feat3"]

        # Mock booster with get_score method
        mock_booster = MagicMock()
        mock_booster.get_score.return_value = {
            "feat1": 100.0,
            "feat2": 60.0,
            "feat3": 40.0,
        }
        trainer._booster = mock_booster

        # Act
        importance = trainer.get_feature_importance()

        # Assert
        assert importance == {"feat1": 100.0, "feat2": 60.0, "feat3": 40.0}
        mock_booster.get_score.assert_called_once_with(importance_type="gain")

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_get_feature_importance_when_not_fitted(self) -> None:
        """
        Test getting feature importance when not fitted returns None.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)

        # Act
        importance = trainer.get_feature_importance()

        # Assert
        assert importance is None

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_get_feature_importance_with_xgb_default_names(self) -> None:
        """
        Test feature importance with XGBoost default feature names.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["feat1", "feat2", "feat3"]

        # Mock booster with get_score using f0, f1, f2 names
        mock_booster = MagicMock()
        mock_booster.get_score.return_value = {"f0": 100.0, "f1": 60.0, "f2": 40.0}
        trainer._booster = mock_booster

        # Act
        importance = trainer.get_feature_importance()

        # Assert
        assert importance == {"feat1": 100.0, "feat2": 60.0, "feat3": 40.0}


class TestXGBoostTrainerPersistence:
    """
    Test XGBoostTrainer model persistence.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_save_model_native_format(self, tmp_path: Any) -> None:
        """
        Test saving model in XGBoost native format.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["feat1", "feat2"]
        trainer._training_metrics = {"accuracy": 0.95}

        mock_booster = MagicMock()
        trainer._booster = mock_booster

        save_path = tmp_path / "model.xgb"

        # Act
        trainer.save_model(save_path)

        # Assert
        mock_booster.save_model.assert_called_once_with(str(save_path))
        # Check metadata file was created
        metadata_path = save_path.with_suffix(".meta")
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
            assert metadata["feature_names"] == ["feat1", "feat2"]
            assert metadata["training_metrics"]["accuracy"] == 0.95

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    @patch("ml.training.xgboost.xgb")
    def test_load_model_native_format(self, mock_xgb: Any, tmp_path: Any) -> None:
        """
        Test loading model from XGBoost native format.
        """
        # Arrange
        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = XGBoostTrainer(config)

        # Create mock model file
        model_path = tmp_path / "model.xgb"
        model_path.touch()

        # Create metadata file
        metadata = {
            "feature_names": ["feat1", "feat2"],
            "training_metrics": {"accuracy": 0.95},
        }
        metadata_path = model_path.with_suffix(".meta")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        mock_booster = MagicMock()
        mock_xgb.Booster.return_value = mock_booster

        # Act
        trainer.load_model(model_path)

        # Assert
        assert trainer._is_fitted is True
        assert trainer._booster == mock_booster
        assert trainer._feature_names == ["feat1", "feat2"]
        assert trainer._training_metrics["accuracy"] == 0.95
        mock_booster.load_model.assert_called_once_with(str(model_path))


class TestXGBoostTrainerBackwardCompatibility:
    """
    Test backward compatibility aliases.
    """

    @patch("ml.training.xgboost.HAS_XGBOOST", True)
    def test_unified_xgboost_trainer_alias(self) -> None:
        """
        Test UnifiedXGBoostTrainer alias works.
        """
        # Arrange
        from ml.training.xgboost import UnifiedXGBoostTrainer

        config = XGBoostTrainingConfig(data_source="test_data.csv", target_column="target")

        # Act
        trainer = UnifiedXGBoostTrainer(config)

        # Assert
        assert isinstance(trainer, XGBoostTrainer)
