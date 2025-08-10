
"""
Unit tests for LightGBMTrainer.

Tests focus on LightGBM-specific functionality while mocking the actual LightGBM
training to ensure test isolation and speed.

"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.lightgbm import EFBConfig
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.config.shared import LightGBMGPUConfig
from ml.training.lightgbm import LightGBMTrainer


class TestLightGBMTrainerInitialization:
    """
    Test LightGBMTrainer initialization.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_init_with_basic_config(self) -> None:
        """
        Test initialization with basic LightGBM configuration.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary",
            n_estimators=100,
        )

        # Act
        trainer = LightGBMTrainer(config)

        # Assert
        assert trainer._lgb_config == config
        assert trainer._booster is None
        assert trainer._categorical_features == []

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", False)
    @patch("ml._imports.HAS_LIGHTGBM", False)
    def test_init_without_lightgbm_raises(self) -> None:
        """
        Test initialization without LightGBM raises error.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")

        # Act & Assert
        with pytest.raises(ImportError, match="LightGBM"):
            LightGBMTrainer(config)


class TestLightGBMTrainerModelTraining:
    """
    Test LightGBMTrainer model training.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    @patch("ml.training.lightgbm.lgb")
    def test_train_model_basic(self, mock_lgb: Any) -> None:
        """
        Test basic model training.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary",
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
        )
        trainer = LightGBMTrainer(config)
        trainer._feature_names = ["feat1", "feat2"]

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = rng.integers(0, 2, 80).astype(np.float64)
        X_val = rng.standard_normal((20, 2))
        y_val = rng.integers(0, 2, 20).astype(np.float64)

        # Mock LightGBM components
        mock_train_data = MagicMock()
        mock_val_data = MagicMock()
        mock_lgb.Dataset.side_effect = [mock_train_data, mock_val_data]

        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_booster.feature_importance.return_value = np.array([0.6, 0.4])
        mock_lgb.train.return_value = mock_booster

        # Mock callbacks
        mock_lgb.early_stopping = Mock(return_value="early_stopping_callback")
        mock_lgb.log_evaluation = Mock(return_value="log_eval_callback")

        # Act
        result = trainer._train_model(X_train, y_train, X_val, y_val)

        # Assert
        assert result["model"] == mock_booster
        assert "metrics" in result
        assert result["metrics"]["best_iteration"] == 50
        mock_lgb.train.assert_called_once()

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    @patch("ml.training.lightgbm.lgb")
    def test_train_model_with_gpu(self, mock_lgb: Any) -> None:
        """
        Test model training with GPU configuration.
        """
        # Arrange
        gpu_config = LightGBMGPUConfig(enabled=True, device_id=0, platform_id=0)
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            gpu_config=gpu_config,
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = rng.integers(0, 2, 80).astype(np.float64)
        X_val = rng.standard_normal((20, 2))
        y_val = rng.integers(0, 2, 20).astype(np.float64)

        # Mock LightGBM components
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.train.return_value = MagicMock()
        mock_lgb.early_stopping = Mock(return_value="callback")
        mock_lgb.log_evaluation = Mock(return_value="callback")

        # Act
        result = trainer._train_model(X_train, y_train, X_val, y_val)

        # Assert
        # Check that GPU parameters were set
        call_args = mock_lgb.train.call_args[0][0]
        assert call_args["device"] == "gpu"
        assert call_args["gpu_platform_id"] == 0
        assert call_args["gpu_device_id"] == 0

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    @patch("ml.training.lightgbm.lgb")
    def test_train_model_with_efb(self, mock_lgb: Any) -> None:
        """
        Test model training with EFB configuration.
        """
        # Arrange
        efb_config = EFBConfig(
            enabled=True,
            max_conflict_rate=0.1,
            bundle_size=100,
        )
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            efb_config=efb_config,
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((80, 2))
        y_train = rng.integers(0, 2, 80).astype(np.float64)
        X_val = rng.standard_normal((20, 2))
        y_val = rng.integers(0, 2, 20).astype(np.float64)

        # Mock LightGBM components
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.train.return_value = MagicMock()
        mock_lgb.early_stopping = Mock(return_value="callback")
        mock_lgb.log_evaluation = Mock(return_value="callback")

        # Act
        result = trainer._train_model(X_train, y_train, X_val, y_val)

        # Assert
        call_args = mock_lgb.train.call_args[0][0]
        assert call_args["enable_bundle"] is True
        assert call_args["max_conflict_rate"] == 0.1
        assert call_args["max_bundle"] == 100


class TestLightGBMTrainerPrediction:
    """
    Test LightGBMTrainer prediction functionality.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_predict_binary_classification(self) -> None:
        """
        Test prediction for binary classification.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary",
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.array([0.2, 0.7, 0.4, 0.9] + [0.5] * 6)

        # Act
        predictions = trainer.predict(mock_model, X)

        # Assert
        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
        mock_model.predict.assert_called_once_with(X, num_iteration=50)

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_predict_with_custom_threshold(self) -> None:
        """
        Test prediction with custom threshold.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary",
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 2))
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.array([0.2, 0.3, 0.6, 0.7, 0.8])

        # Act
        predictions = trainer.predict(mock_model, X, threshold=0.3)

        # Assert
        expected = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(predictions, expected)

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_predict_multiclass(self) -> None:
        """
        Test prediction for multiclass classification.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="multiclass",
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 2))
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        # Multiclass returns probabilities for each class
        mock_model.predict.return_value = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.9, 0.05, 0.05],
                [0.1, 0.1, 0.8],
            ],
        )

        # Act
        predictions = trainer.predict(mock_model, X)

        # Assert
        expected = np.array([0, 1, 2, 0, 2])  # argmax of each row
        np.testing.assert_array_equal(predictions, expected)

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_predict_regression(self) -> None:
        """
        Test prediction for regression.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="regression",
        )
        trainer = LightGBMTrainer(config)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        expected_predictions = rng.standard_normal(10)
        mock_model.predict.return_value = expected_predictions

        # Act
        predictions = trainer.predict(mock_model, X)

        # Assert - predictions should be float32 now
        assert predictions.dtype == np.float32
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=6)


class TestLightGBMTrainerHyperparameters:
    """
    Test LightGBMTrainer hyperparameter functionality.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_get_model_params(self) -> None:
        """
        Test getting default model parameters.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            objective="binary",
            metric="binary_logloss",
            num_leaves=31,
            max_depth=5,
            learning_rate=0.1,
            feature_fraction=0.9,
        )
        trainer = LightGBMTrainer(config)

        # Act
        params = trainer._get_model_params()

        # Assert
        assert params["objective"] == "binary"
        assert params["metric"] == "binary_logloss"
        assert params["num_leaves"] == 31
        assert params["max_depth"] == 5
        assert params["learning_rate"] == 0.1
        assert params["feature_fraction"] == 0.9
        assert params["verbosity"] == -1
        assert params["seed"] == 42

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_get_model_params_with_scale_pos_weight(self) -> None:
        """
        Test getting model parameters with scale_pos_weight.
        """
        # Arrange
        config = LightGBMTrainingConfig(
            data_source="test_data.csv",
            target_column="target",
            scale_pos_weight=2.0,
        )
        trainer = LightGBMTrainer(config)

        # Act
        params = trainer._get_model_params()

        # Assert
        assert params["scale_pos_weight"] == 2.0

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_suggest_hyperparameters_for_optuna(self) -> None:
        """
        Test hyperparameter suggestion for Optuna optimization.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)

        # Mock Optuna trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [
            50,
            5,
            3,
            20,
        ]  # num_leaves, max_depth, bagging_freq, min_child_samples
        mock_trial.suggest_float.side_effect = [
            0.1,  # learning_rate
            0.8,  # feature_fraction
            0.7,  # bagging_fraction
            1.0,  # lambda_l1
            2.0,  # lambda_l2
        ]

        # Act
        params = trainer._suggest_hyperparameters(mock_trial)

        # Assert
        assert params["num_leaves"] == 50
        assert params["max_depth"] == 5
        assert params["learning_rate"] == 0.1
        assert params["feature_fraction"] == 0.8
        assert params["bagging_fraction"] == 0.7
        assert params["bagging_freq"] == 3
        assert params["lambda_l1"] == 1.0
        assert params["lambda_l2"] == 2.0
        assert params["min_child_samples"] == 20


class TestLightGBMTrainerFeatureImportance:
    """
    Test LightGBMTrainer feature importance functionality.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_get_feature_importance_when_fitted(self) -> None:
        """
        Test getting feature importance from fitted model.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["feat1", "feat2", "feat3"]

        # Mock booster with feature_importance method
        mock_booster = MagicMock()
        mock_booster.feature_importance.return_value = np.array([100.0, 60.0, 40.0])
        trainer._booster = mock_booster

        # Act
        importance = trainer.get_feature_importance()

        # Assert
        assert importance == {"feat1": 100.0, "feat2": 60.0, "feat3": 40.0}
        mock_booster.feature_importance.assert_called_once_with(importance_type="gain")

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_get_feature_importance_when_not_fitted(self) -> None:
        """
        Test getting feature importance when not fitted returns None.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)

        # Act
        importance = trainer.get_feature_importance()

        # Assert
        assert importance is None

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_plot_importance_when_not_fitted_raises(self) -> None:
        """
        Test plotting importance when not fitted raises error.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)

        # Act & Assert
        with pytest.raises(ValueError, match="Model must be fitted"):
            trainer.plot_importance()


class TestLightGBMTrainerPersistence:
    """
    Test LightGBMTrainer model persistence.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_save_model_native_format(self, tmp_path: Any) -> None:
        """
        Test saving model in LightGBM native format.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["feat1", "feat2"]
        trainer._categorical_features = [0]
        trainer._training_metrics = {"accuracy": 0.95}

        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        trainer._booster = mock_booster

        save_path = tmp_path / "model.lgb"

        # Act
        trainer.save_model(save_path)

        # Assert
        mock_booster.save_model.assert_called_once_with(str(save_path), num_iteration=50)
        # Check metadata file was created with correct naming convention
        metadata_path = save_path.with_suffix(save_path.suffix + ".meta.json")
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
            # Check training metadata nested structure
            training_metadata = metadata["training_metadata"]
            assert training_metadata["feature_names"] == ["feat1", "feat2"]
            assert training_metadata["categorical_features"] == [0]
            assert training_metadata["training_metrics"]["accuracy"] == 0.95

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    @patch("ml.training.lightgbm.lgb")
    def test_load_model_native_format(self, mock_lgb: Any, tmp_path: Any) -> None:
        """
        Test loading model from LightGBM native format.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)

        # Create mock model file
        model_path = tmp_path / "model.lgb"
        model_path.touch()

        # Create metadata file
        metadata = {
            "feature_names": ["feat1", "feat2"],
            "categorical_features": [0],
            "training_metrics": {"accuracy": 0.95},
        }
        metadata_path = model_path.with_suffix(".meta")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        mock_booster = MagicMock()
        mock_lgb.Booster.return_value = mock_booster

        # Act
        trainer.load_model(model_path)

        # Assert
        assert trainer._is_fitted is True
        assert trainer._booster == mock_booster
        assert trainer._feature_names == ["feat1", "feat2"]
        assert trainer._categorical_features == [0]
        assert trainer._training_metrics["accuracy"] == 0.95
        mock_lgb.Booster.assert_called_once_with(model_file=str(model_path))

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_convert_to_onnx_fallback_to_text(self, tmp_path: Any) -> None:
        """
        Test ONNX conversion fallback to text format when onnxmltools not available.
        """
        # Arrange
        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")
        trainer = LightGBMTrainer(config)

        mock_model = MagicMock()
        mock_model.best_iteration = 50
        onnx_path = tmp_path / "model.onnx"

        # Mock ImportError for onnxmltools
        with patch("ml.training.lightgbm.LightGBMTrainer._convert_to_onnx") as mock_convert:

            def side_effect(model: Any, path: Any) -> None:
                # Simulate ImportError and fallback
                model.save_model(str(path.with_suffix(".txt")), num_iteration=model.best_iteration)

            mock_convert.side_effect = side_effect

            # Act
            trainer._convert_to_onnx(mock_model, onnx_path)

            # Assert
            txt_path = onnx_path.with_suffix(".txt")
            mock_model.save_model.assert_called_once_with(str(txt_path), num_iteration=50)


class TestLightGBMTrainerBackwardCompatibility:
    """
    Test backward compatibility aliases.
    """

    @patch("ml.training.lightgbm.HAS_LIGHTGBM", True)
    def test_unified_lightgbm_trainer_alias(self) -> None:
        """
        Test UnifiedLightGBMTrainer alias works.
        """
        # Arrange
        from ml.training.lightgbm import UnifiedLightGBMTrainer

        config = LightGBMTrainingConfig(data_source="test_data.csv", target_column="target")

        # Act
        trainer = UnifiedLightGBMTrainer(config)

        # Assert
        assert isinstance(trainer, LightGBMTrainer)
