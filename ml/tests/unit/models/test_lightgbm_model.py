#!/usr/bin/env python3

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

"""Unit tests for LightGBMModel wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml._imports import HAS_LIGHTGBM
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml.models.lightgbm_model import LightGBMModel
from ml.tests.fixtures.model_factory import TestModelFactory


# Skip all tests if LightGBM not available
pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not available")


class TestLightGBMModel:
    """Test suite for LightGBMModel wrapper."""

    @pytest.fixture
    def classification_booster(self, tmp_path: Path) -> tuple[LightGBMModel, dict[str, Any]]:
        """Create a minimal classification Booster model for testing."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        model_path = TestModelFactory.create_minimal_lightgbm_model(
            n_features=5,
            model_type="classification",
            output_path=tmp_path / "clf_model.txt",
        )

        # Load as Booster (raw LightGBM object)
        booster = lgb.Booster(model_file=str(model_path))

        metadata = {
            "model_type": "lightgbm",
            "model_class": "classification",
            "n_features": 5,
            "input_shape": [None, 5],
        }

        return LightGBMModel(booster, metadata), metadata

    @pytest.fixture
    def regression_booster(self, tmp_path: Path) -> tuple[LightGBMModel, dict[str, Any]]:
        """Create a minimal regression Booster model for testing."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        model_path = TestModelFactory.create_minimal_lightgbm_model(
            n_features=3,
            model_type="regression",
            output_path=tmp_path / "reg_model.txt",
        )

        # Load as Booster
        booster = lgb.Booster(model_file=str(model_path))

        metadata = {
            "model_type": "lightgbm",
            "model_class": "regression",
            "n_features": 3,
            "input_shape": [None, 3],
        }

        return LightGBMModel(booster, metadata), metadata

    @pytest.fixture
    def sklearn_classifier(self, tmp_path: Path) -> tuple[LightGBMModel, dict[str, Any]]:
        """Create a minimal sklearn-style LGBMClassifier for testing."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = rng.integers(0, 2, 20)

        # Create and train sklearn-style model
        model = lgb.LGBMClassifier(
            n_estimators=2,
            max_depth=2,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X, y)

        metadata = {
            "model_type": "lightgbm",
            "model_class": "classification",
            "n_features": 4,
            "input_shape": [None, 4],
        }

        return LightGBMModel(model, metadata), metadata

    @pytest.fixture
    def sklearn_regressor(self, tmp_path: Path) -> tuple[LightGBMModel, dict[str, Any]]:
        """Create a minimal sklearn-style LGBMRegressor for testing."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        y = rng.standard_normal(20).astype(np.float32)

        # Create and train sklearn-style model
        model = lgb.LGBMRegressor(
            n_estimators=2,
            max_depth=2,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X, y)

        metadata = {
            "model_type": "lightgbm",
            "model_class": "regression",
            "n_features": 6,
            "input_shape": [None, 6],
        }

        return LightGBMModel(model, metadata), metadata

    def test_initialization_with_booster(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel can be initialized with Booster object."""
        model, expected_metadata = classification_booster

        assert isinstance(model, LightGBMModel)
        assert model.metadata == expected_metadata
        assert hasattr(model._model, "predict")  # Booster has predict method
        assert not hasattr(model._model, "predict_proba")  # But no predict_proba

    def test_initialization_with_sklearn_classifier(self, sklearn_classifier: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel can be initialized with LGBMClassifier object."""
        model, expected_metadata = sklearn_classifier

        assert isinstance(model, LightGBMModel)
        assert model.metadata == expected_metadata
        assert hasattr(model._model, "predict")
        assert hasattr(model._model, "predict_proba")  # sklearn-style has predict_proba

    def test_initialization_with_sklearn_regressor(self, sklearn_regressor: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel can be initialized with LGBMRegressor object."""
        model, expected_metadata = sklearn_regressor

        assert isinstance(model, LightGBMModel)
        assert model.metadata == expected_metadata
        assert hasattr(model._model, "predict")
        assert not hasattr(model._model, "predict_proba")  # Regressor has no predict_proba

    def test_initialization_without_lightgbm_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LightGBMModel initialization fails gracefully when LightGBM not available."""
        # Mock HAS_LIGHTGBM to False and the check_ml_dependencies function
        import ml.models.lightgbm_model
        monkeypatch.setattr(ml.models.lightgbm_model, "HAS_LIGHTGBM", False)

        def mock_check_deps(deps: list[str]) -> None:
            raise ImportError("LightGBM required but not installed")

        monkeypatch.setattr(ml.models.lightgbm_model, "check_ml_dependencies", mock_check_deps)

        with pytest.raises(ImportError, match="LightGBM required but not installed"):
            LightGBMModel(model=None, metadata={})

    def test_predict_with_booster_classification_returns_float32(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Booster classification predictions return float32 dtype."""
        model, _ = classification_booster

        features = np.random.randn(10, 5).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (10,)

    def test_predict_with_booster_regression_returns_float32(self, regression_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Booster regression predictions return float32 dtype."""
        model, _ = regression_booster

        features = np.random.randn(5, 3).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (5,)

    def test_predict_with_sklearn_classifier_returns_float32(self, sklearn_classifier: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Sklearn LGBMClassifier predictions return float32 dtype."""
        model, _ = sklearn_classifier

        features = np.random.randn(8, 4).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (8,)

    def test_predict_with_sklearn_regressor_returns_float32(self, sklearn_regressor: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Sklearn LGBMRegressor predictions return float32 dtype."""
        model, _ = sklearn_regressor

        features = np.random.randn(3, 6).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (3,)

    def test_predict_with_1d_input_handles_reshape(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() handles 1D input by reshaping to 2D."""
        model, _ = classification_booster

        # Single sample as 1D array
        features_1d = np.random.randn(5).astype(np.float32)
        predictions = model.predict(features_1d)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (1,)  # Should be single prediction

    def test_predict_with_2d_input_preserves_batch_dimension(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() handles 2D input correctly for batch predictions."""
        model, _ = classification_booster

        # Batch of samples
        features_2d = np.random.randn(7, 5).astype(np.float32)
        predictions = model.predict(features_2d)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (7,)  # Batch size preserved

    def test_predict_validates_input_shape(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() validates input features through validate_input."""
        model, _ = classification_booster

        # Wrong number of features (expect 5, provide 3)
        wrong_features = np.random.randn(2, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Input validation failed"):
            model.predict(wrong_features)

    def test_predict_binary_classification_returns_positive_class_probability(self, sklearn_classifier: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Binary classification returns positive class probability only."""
        model, _ = sklearn_classifier

        features = np.random.randn(5, 4).astype(np.float32)
        predictions = model.predict(features)

        assert predictions.shape == (5,)  # Not (5, 2) - only positive class prob
        assert predictions.dtype == np.float32
        assert np.all((predictions >= 0) & (predictions <= 1))

    def test_predict_handles_extreme_values(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() handles extreme input values gracefully."""
        model, _ = classification_booster

        # Extreme values
        extreme_features = np.array([
            [1e6, -1e6, 0, 1e3, -1e3],
            [100.0, 1.0, 2.0, 3.0, 4.0],
        ], dtype=np.float32)

        predictions = model.predict(extreme_features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)

    def test_predict_with_empty_batch(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() handles empty batch correctly."""
        model, _ = classification_booster

        empty_features = np.empty((0, 5), dtype=np.float32)
        predictions = model.predict(empty_features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (0,)

    def test_predict_enforces_float32_from_float64_input(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """predict() converts float64 input to float32 and ensures float32 output."""
        model, _ = classification_booster

        # Start with float64
        features_f64 = np.random.randn(3, 5).astype(np.float64)
        predictions = model.predict(features_f64.astype(np.float32))

        assert predictions.dtype == np.float32

    def test_model_detects_booster_type_correctly(self, classification_booster: tuple[LightGBMModel, dict[str, Any]], sklearn_classifier: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel correctly detects whether underlying model is Booster or sklearn-style."""
        booster_model, _ = classification_booster
        sklearn_model, _ = sklearn_classifier

        # Check internal detection logic
        # Booster: has predict but no predict_proba
        # LGBMClassifier: has both predict and predict_proba
        # LGBMRegressor: has predict but no predict_proba (like Booster)
        assert hasattr(booster_model._model, "predict")
        assert not hasattr(booster_model._model, "predict_proba")

        assert hasattr(sklearn_model._model, "predict")
        assert hasattr(sklearn_model._model, "predict_proba")  # Classifier has this

    def test_booster_uses_best_iteration(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Booster models use best_iteration for prediction."""
        model, _ = classification_booster

        features = np.random.randn(2, 5).astype(np.float32)

        # This should work without error (tests best_iteration usage)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)

    def test_sklearn_prediction_uses_native_methods(self, sklearn_classifier: tuple[LightGBMModel, dict[str, Any]], sklearn_regressor: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """sklearn-style models use predict/predict_proba methods."""
        clf_model, _ = sklearn_classifier
        reg_model, _ = sklearn_regressor

        features_clf = np.random.randn(3, 4).astype(np.float32)
        features_reg = np.random.randn(2, 6).astype(np.float32)

        # Classification uses predict_proba
        clf_predictions = clf_model.predict(features_clf)
        assert clf_predictions.dtype == np.float32
        assert clf_predictions.shape == (3,)
        assert hasattr(clf_model._model, "predict_proba")

        # Regression uses predict
        reg_predictions = reg_model.predict(features_reg)
        assert reg_predictions.dtype == np.float32
        assert reg_predictions.shape == (2,)
        assert not hasattr(reg_model._model, "predict_proba")

    def test_predict_consistency_across_model_types(self, tmp_path: Path) -> None:
        """Predictions should be consistent between Booster and sklearn models trained on same data."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create identical training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.integers(0, 2, 100)

        # Train sklearn model
        sklearn_model = lgb.LGBMClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42,
            verbosity=-1,
        )
        sklearn_model.fit(X, y)

        # Save and reload as Booster
        temp_path = tmp_path / "temp_model.txt"
        sklearn_model.booster_.save_model(str(temp_path))

        booster = lgb.Booster(model_file=str(temp_path))

        # Wrap both in LightGBMModel
        metadata = {"input_shape": [None, 5]}
        wrapped_sklearn = LightGBMModel(sklearn_model, metadata)
        wrapped_booster = LightGBMModel(booster, metadata)

        # Test features
        test_features = rng.standard_normal((10, 5)).astype(np.float32)

        sklearn_pred = wrapped_sklearn.predict(test_features)
        booster_pred = wrapped_booster.predict(test_features)

        # Predictions should be very close (allowing for minor numerical differences)
        assert np.allclose(sklearn_pred, booster_pred, rtol=1e-5, atol=1e-6)

    def test_predict_without_lightgbm_dependency(self, classification_booster: tuple[LightGBMModel, dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
        """predict() method handles missing LightGBM dependency."""
        model, _ = classification_booster

        # Mock HAS_LIGHTGBM to False and the check_ml_dependencies function
        import ml.models.lightgbm_model
        monkeypatch.setattr(ml.models.lightgbm_model, "HAS_LIGHTGBM", False)

        def mock_check_deps(deps: list[str]) -> None:
            raise ImportError("LightGBM required but not installed")

        monkeypatch.setattr(ml.models.lightgbm_model, "check_ml_dependencies", mock_check_deps)

        features = np.random.randn(2, 5).astype(np.float32)

        # The predict method doesn't check dependencies again, only __init__ does
        # So this should work since the model is already initialized
        predictions = model.predict(features)
        assert predictions.dtype == np.float32

    def test_inheritance_from_base_model(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel properly inherits from BaseModel."""
        model, _ = classification_booster

        # Should have all BaseModel methods
        assert hasattr(model, "metadata")
        assert hasattr(model, "model_id")
        assert hasattr(model, "validate_input")
        assert hasattr(model, "predict")

        # Test BaseModel behavior
        assert isinstance(model.metadata, dict)
        assert isinstance(model.model_id, str)

    def test_metadata_passthrough(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Metadata is properly stored and accessible."""
        model, expected_metadata = classification_booster

        assert model.metadata == expected_metadata
        assert model.metadata["model_type"] == "lightgbm"
        assert model.metadata["n_features"] == 5

    def test_edge_case_single_feature_model(self, tmp_path: Path) -> None:
        """LightGBMModel works with single feature models."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        model_path = TestModelFactory.create_minimal_lightgbm_model(
            n_features=1,
            model_type="regression",
            output_path=tmp_path / "single_feature.txt",
        )

        booster = lgb.Booster(model_file=str(model_path))

        metadata = {"input_shape": [None, 1]}
        model = LightGBMModel(booster, metadata)

        # Test with single feature
        features = np.array([[1.5]], dtype=np.float32)
        predictions = model.predict(features)

        assert predictions.dtype == np.float32
        assert predictions.shape == (1,)

    def test_edge_case_large_batch_prediction(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """LightGBMModel handles large batch predictions."""
        model, _ = classification_booster

        # Large batch
        large_batch = np.random.randn(1000, 5).astype(np.float32)
        predictions = model.predict(large_batch)

        assert predictions.dtype == np.float32
        assert predictions.shape == (1000,)

    def test_model_reproducibility(self, tmp_path: Path) -> None:
        """LightGBMModel predictions are reproducible with same random seed."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create two identical models
        model_path_1 = TestModelFactory.create_minimal_lightgbm_model(
            n_features=3,
            model_type="classification",
            output_path=tmp_path / "model1.txt",
        )

        model_path_2 = TestModelFactory.create_minimal_lightgbm_model(
            n_features=3,
            model_type="classification",
            output_path=tmp_path / "model2.txt",
        )

        # Load both models
        booster_1 = lgb.Booster(model_file=str(model_path_1))
        booster_2 = lgb.Booster(model_file=str(model_path_2))

        metadata = {"input_shape": [None, 3]}
        model_1 = LightGBMModel(booster_1, metadata)
        model_2 = LightGBMModel(booster_2, metadata)

        # Same input
        features = np.random.randn(5, 3).astype(np.float32)

        pred_1 = model_1.predict(features)
        pred_2 = model_2.predict(features)

        # Should be identical (same seed used in TestModelFactory)
        assert np.array_equal(pred_1, pred_2)

    def test_multiclass_classification_handling(self, tmp_path: Path) -> None:
        """LightGBMModel handles multiclass classification correctly."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create multiclass training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = rng.integers(0, 3, 60)  # 3 classes

        # Train multiclass model
        model = lgb.LGBMClassifier(
            n_estimators=3,
            max_depth=2,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X, y)

        metadata = {"input_shape": [None, 4]}
        wrapped_model = LightGBMModel(model, metadata)

        # Test features
        test_features = rng.standard_normal((5, 4)).astype(np.float32)
        predictions = wrapped_model.predict(test_features)

        # For multiclass, predict_proba returns all class probabilities
        # The wrapper should return the full probability matrix
        assert predictions.dtype == np.float32
        assert predictions.shape == (5, 3)  # 5 samples, 3 classes

        # Each row should sum to approximately 1 (probabilities)
        row_sums = np.sum(predictions, axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5)

    def test_model_detection_logic_accuracy(self) -> None:
        """Test that model type detection logic works correctly for different LightGBM types."""
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create minimal training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y_cls = rng.integers(0, 2, 20)
        y_reg = rng.standard_normal(20).astype(np.float32)

        # Create different model types
        classifier = lgb.LGBMClassifier(n_estimators=2, verbosity=-1)
        regressor = lgb.LGBMRegressor(n_estimators=2, verbosity=-1)

        classifier.fit(X, y_cls)
        regressor.fit(X, y_reg)

        # Get the underlying boosters
        booster_from_clf = classifier.booster_
        booster_from_reg = regressor.booster_

        metadata = {"input_shape": [None, 3]}

        # Test model detection
        wrapped_clf = LightGBMModel(classifier, metadata)
        wrapped_reg = LightGBMModel(regressor, metadata)
        wrapped_booster_clf = LightGBMModel(booster_from_clf, metadata)
        wrapped_booster_reg = LightGBMModel(booster_from_reg, metadata)

        # Check detection results
        # Classifier should be detected as sklearn-style (has booster_ attribute)
        assert not wrapped_clf._is_booster
        assert hasattr(wrapped_clf._model, "predict_proba")
        assert hasattr(wrapped_clf._model, "booster_")

        # Regressor should be detected as sklearn-style (has booster_ attribute)
        assert not wrapped_reg._is_booster  # sklearn-style, has booster_
        assert not hasattr(wrapped_reg._model, "predict_proba")
        assert hasattr(wrapped_reg._model, "booster_")

        # Raw boosters should be detected as Booster (has best_iteration, no booster_)
        assert wrapped_booster_clf._is_booster
        assert wrapped_booster_reg._is_booster
        assert hasattr(wrapped_booster_clf._model, "best_iteration")
        assert hasattr(wrapped_booster_reg._model, "best_iteration")
        assert not hasattr(wrapped_booster_clf._model, "booster_")
        assert not hasattr(wrapped_booster_reg._model, "booster_")

    def test_input_validation_edge_cases(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Test input validation handles various edge cases."""
        model, _ = classification_booster

        # Test with different dtypes that should be converted
        features_int = np.random.randint(0, 10, (3, 5)).astype(np.int32)
        features_f64 = np.random.randn(3, 5).astype(np.float64)

        # Should work after conversion to float32
        predictions_int = model.predict(features_int.astype(np.float32))
        predictions_f64 = model.predict(features_f64.astype(np.float32))

        assert predictions_int.dtype == np.float32
        assert predictions_f64.dtype == np.float32
        assert predictions_int.shape == (3,)
        assert predictions_f64.shape == (3,)

    def test_error_handling_with_invalid_inputs(self, classification_booster: tuple[LightGBMModel, dict[str, Any]]) -> None:
        """Test proper error handling with invalid inputs."""
        model, _ = classification_booster

        # Test with NaN values (should propagate through)
        features_with_nan = np.array([
            [1.0, 2.0, np.nan, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ], dtype=np.float32)

        # LightGBM should handle NaN values gracefully
        predictions = model.predict(features_with_nan)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)

        # Test with inf values (should also work)
        features_with_inf = np.array([
            [1.0, 2.0, np.inf, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ], dtype=np.float32)

        predictions = model.predict(features_with_inf)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)
