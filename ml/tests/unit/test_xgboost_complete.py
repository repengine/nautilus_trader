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
Comprehensive unit tests for XGBoostTrainer.

Tests all functionality with mocked dependencies to ensure tests run without external
packages installed.

"""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.tests.unit.test_fixtures import mock_sklearn
from ml.training.xgboost import XGBoostTrainer


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


class TestXGBoostTrainingConfig:
    """
    Test XGBoost training configuration.
    """

    def test_default_config_creation(self) -> None:
        """
        Test creating config with default values.
        """
        config = XGBoostTrainingConfig(data_source="test")

        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.3
        assert config.tree_method == "hist"
        assert config.objective == "binary:logistic"
        assert config.multi_asset is False

    def test_config_validation_subsample(self) -> None:
        """
        Test validation of subsample parameter.
        """
        with pytest.raises(ValueError, match="subsample must be in"):
            XGBoostTrainingConfig(data_source="test", subsample=0.0)

        with pytest.raises(ValueError, match="subsample must be in"):
            XGBoostTrainingConfig(data_source="test", subsample=1.5)

    def test_config_validation_tree_method(self) -> None:
        """
        Test validation of tree_method parameter.
        """
        with pytest.raises(ValueError, match="tree_method must be one of"):
            XGBoostTrainingConfig(data_source="test", tree_method="invalid")

    def test_config_validation_objective(self) -> None:
        """
        Test validation of objective parameter.
        """
        with pytest.raises(ValueError, match="objective must be one of"):
            XGBoostTrainingConfig(data_source="test", objective="invalid")

    def test_config_validation_multi_asset(self) -> None:
        """
        Test validation of multi-asset configuration.
        """
        with pytest.raises(ValueError, match="sector_map is required"):
            XGBoostTrainingConfig(data_source="test", multi_asset=True)

    def test_config_validation_monotonic_constraints(self) -> None:
        """
        Test validation of monotonic constraints.
        """
        with pytest.raises(ValueError, match="monotonic constraint"):
            XGBoostTrainingConfig(
                data_source="test",
                monotonic_constraints={"feature1": 2},  # Invalid constraint
            )

    def test_config_validation_colsample(self) -> None:
        """
        Test validation of colsample parameters.
        """
        with pytest.raises(ValueError, match="colsample_bytree must be in"):
            XGBoostTrainingConfig(data_source="test", colsample_bytree=0.0)

        with pytest.raises(ValueError, match="colsample_bylevel must be in"):
            XGBoostTrainingConfig(data_source="test", colsample_bylevel=1.5)

    def test_get_xgb_params(self) -> None:
        """
        Test XGBoost parameter extraction.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            tree_method="gpu_hist",
            gpu_id=1,
        )

        params = config.get_xgb_params()

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 4
        assert params["learning_rate"] == 0.1
        assert params["tree_method"] == "gpu_hist"
        assert params["gpu_id"] == 1
        assert params["predictor"] == "gpu_predictor"
        assert params["n_jobs"] == -1

    def test_valid_monotonic_constraints(self) -> None:
        """
        Test valid monotonic constraints configuration.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            monotonic_constraints={"feature1": 1, "feature2": -1, "feature3": 0},
        )

        assert config.monotonic_constraints == {"feature1": 1, "feature2": -1, "feature3": 0}

    def test_valid_multi_asset_config(self) -> None:
        """
        Test valid multi-asset configuration.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech", "MSFT": "Tech"},
        )

        assert config.multi_asset is True
        assert config.sector_map == {"AAPL": "Tech", "MSFT": "Tech"}


class TestXGBoostTrainer:
    """
    Test XGBoost trainer implementation.
    """

    def test_trainer_initialization(self) -> None:
        """
        Test trainer initialization.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        assert trainer._xgb_config == config
        assert trainer._is_multi_asset is False
        assert trainer._xgb is None  # Lazy import
        assert trainer._shap is None
        assert trainer._optuna is None
        assert trainer._feature_engineer is not None

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    def test_prepare_data_no_polars_error(self) -> None:
        """
        Test error when Polars not available.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        with (
            patch("ml.training.base.HAS_POLARS", False),
            patch("ml.training.xgboost.HAS_POLARS", False),
        ):
            with pytest.raises(ImportError, match="Polars is required"):
                trainer.prepare_data({})

    def test_train_model_no_xgboost_error(self) -> None:
        """
        Test error when XGBoost not available.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        X_train = rng.standard_normal((100, 10))
        y_train = rng.integers(0, 2, 100)
        X_val = rng.standard_normal((20, 10))
        y_val = rng.integers(0, 2, 20)

        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="XGBoost is required"):
                trainer._train_model(X_train, y_train, X_val, y_val)

    def test_create_monotonic_constraints(self) -> None:
        """
        Test monotonic constraints string creation.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        feature_names = ["feature1", "feature2", "feature3", "feature4"]
        constraints_dict = {"feature1": 1, "feature3": -1}

        result = trainer._create_monotonic_constraints(feature_names, constraints_dict)

        assert result == "(1,0,-1,0)"

    def test_create_monotonic_constraints_empty(self) -> None:
        """
        Test monotonic constraints with empty dict.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        feature_names = ["feature1", "feature2"]
        constraints_dict: dict[str, int] = {}

        result = trainer._create_monotonic_constraints(feature_names, constraints_dict)

        assert result == "(0,0)"

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    def test_single_asset_data_preparation(self, mock_pl: MagicMock) -> None:
        """
        Test single asset data preparation.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df.__len__.return_value = 200
        mock_df.drop.return_value = mock_df
        mock_df.with_columns.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.to_numpy.return_value = rng.standard_normal((150, 10))

        # Mock column access
        mock_series = MagicMock()
        mock_series.shift.return_value = mock_series
        mock_series.__gt__.return_value = mock_series
        mock_series.cast.return_value = mock_series
        mock_series.to_numpy.return_value = rng.integers(0, 2, 149)  # One less due to shift
        mock_series.__getitem__.return_value = mock_series
        # Special handling for slicing
        mock_series.__getitem__.side_effect = lambda x: (
            mock_series if x == -1 else np.array([100.0] * 149)
        )
        mock_df.__getitem__.return_value = mock_series
        mock_df.__getitem__.side_effect = lambda x: mock_series if x == "close" else mock_df

        # Mock feature engineer
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            mock_calc.return_value = (mock_df, None)
            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = [f"feature_{i}" for i in range(10)]

                X, y, metadata = trainer.prepare_data(mock_df)

                assert isinstance(X, np.ndarray)
                assert isinstance(y, np.ndarray)
                assert len(X) == len(y)
                assert X.shape[0] > 0
                assert X.shape[1] > 0
                assert "feature_names" in metadata
                assert "n_features" in metadata
                assert "target_type" in metadata

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.features.engineering.pl")
    def test_multi_asset_data_preparation(self, mock_eng_pl: MagicMock, mock_pl: MagicMock) -> None:
        """
        Test multi-asset data preparation.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"},
        )
        trainer = XGBoostTrainer(config)

        # Create mock DataFrames for each asset
        data_dict = {}
        for ticker in ["AAPL", "MSFT", "JPM"]:
            mock_df = MagicMock()
            mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            mock_df.__len__.return_value = 150
            mock_df.with_columns.return_value = mock_df
            mock_df.select.return_value = mock_df
            mock_df.to_numpy.return_value = rng.standard_normal((100, 10))

            # Mock column access with proper chaining
            mock_returns = MagicMock()
            mock_returns.__gt__.return_value = MagicMock(
                cast=MagicMock(
                    return_value=MagicMock(
                        to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
                    ),
                ),
            )

            mock_series = MagicMock()
            mock_shifted = MagicMock()
            mock_ratio = MagicMock()
            mock_series.shift.return_value = mock_shifted
            mock_shifted.__truediv__.return_value = mock_ratio
            mock_ratio.__sub__.return_value = mock_returns
            mock_series.to_numpy.return_value = rng.random(100)
            mock_df.__getitem__.return_value = mock_series

            data_dict[ticker] = mock_df

        # Mock polars functions
        mock_combined = MagicMock()
        mock_combined.columns = ["feature1", "feature2", "ticker", "sector"]
        # Set up select to return proper mock with to_numpy
        mock_selected = MagicMock()
        mock_selected.to_numpy.return_value = rng.standard_normal((300, 2))
        mock_combined.select.return_value = mock_selected
        mock_combined.with_columns.return_value = mock_combined
        # Handle with_row_count chain for cross-sectional features
        mock_combined.with_row_count.return_value = mock_combined
        mock_pl.concat.return_value = mock_combined
        mock_pl.lit.return_value = MagicMock(alias=MagicMock(side_effect=lambda x: x))

        # Mock feature engineer
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            # Create a proper mock features DataFrame
            mock_features = MagicMock()
            mock_features.__len__.return_value = 100
            mock_features.with_columns.return_value = mock_features
            mock_features.__getitem__.return_value = mock_features
            mock_calc.return_value = (mock_features, None)

            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = ["feature1", "feature2"]

                # Patch _add_cross_sectional_features to return the mocked DataFrame
                with patch.object(
                    trainer,
                    "_add_cross_sectional_features",
                    return_value=mock_combined,
                ):
                    X, y, metadata = trainer.prepare_data(data_dict)

                assert isinstance(X, np.ndarray)
                assert isinstance(y, np.ndarray)
                assert X.shape[0] > 0
                assert metadata["n_assets"] == 3

    def test_train_model_classification(self) -> None:
        """
        Test training classification model.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=10,
            objective="binary:logistic",
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = [f"feature_{i}" for i in range(5)]

        # Mock XGBoost
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.25, 0.15])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model
        trainer._xgb = mock_xgb

        X_train = rng.standard_normal((100, 5))
        y_train = rng.integers(0, 2, 100)
        X_val = rng.standard_normal((20, 5))
        y_val = rng.integers(0, 2, 20)

        result = trainer._train_model(X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "metrics" in result
        assert "feature_importance" in result
        assert result["metrics"]["best_iteration"] == 5
        assert result["metrics"]["best_score"] == 0.85
        mock_model.fit.assert_called_once()

    def test_train_model_regression(self) -> None:
        """
        Test training regression model.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=10,
            objective="reg:squarederror",
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = [f"feature_{i}" for i in range(5)]

        # Mock XGBoost
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 7
        mock_model.best_score = 0.92
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

        mock_xgb = MagicMock()
        mock_xgb.XGBRegressor.return_value = mock_model
        trainer._xgb = mock_xgb

        X_train = rng.standard_normal((100, 5))
        y_train = rng.standard_normal(100)
        X_val = rng.standard_normal((20, 5))
        y_val = rng.standard_normal(20)

        result = trainer._train_model(X_train, y_train, X_val, y_val)

        assert result["model"] == mock_model
        assert result["metrics"]["best_iteration"] == 7
        mock_xgb.XGBRegressor.assert_called_once()

    def test_train_model_with_monotonic_constraints(self) -> None:
        """
        Test training with monotonic constraints.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=10,
            monotonic_constraints={"feature_0": 1, "feature_2": -1},
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = [f"feature_{i}" for i in range(5)]

        # Mock XGBoost
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.25, 0.15])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model
        trainer._xgb = mock_xgb

        X_train = rng.standard_normal((100, 5))
        y_train = rng.integers(0, 2, 100)
        X_val = rng.standard_normal((20, 5))
        y_val = rng.integers(0, 2, 20)

        _ = trainer._train_model(X_train, y_train, X_val, y_val)

        # Check that monotonic constraints were applied
        call_args = mock_xgb.XGBClassifier.call_args
        assert "monotone_constraints" in call_args[1]
        assert call_args[1]["monotone_constraints"] == "(1,0,-1,0,0)"

    def test_calculate_feature_importance(self) -> None:
        """
        Test feature importance calculation.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feature1", "feature2", "feature3"]

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        importance = trainer._calculate_feature_importance(mock_model)

        assert importance == {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}
        assert list(importance.values()) == sorted(importance.values(), reverse=True)

    def test_calculate_shap_values(self) -> None:
        """
        Test SHAP values calculation.
        """
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feature1", "feature2", "feature3"]

        # Mock SHAP
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.15, 0.25, 0.35],
            ],
        )
        mock_explainer.expected_value = 0.5

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        trainer._shap = mock_shap

        mock_model = MagicMock()
        X_sample = rng.standard_normal((2, 3))

        result = trainer._calculate_shap_values(mock_model, X_sample)

        assert "shap_values" in result
        assert "shap_importance" in result
        assert "expected_value" in result
        assert result["expected_value"] == 0.5
        assert len(result["shap_importance"]) == 3

    def test_calculate_shap_values_no_shap(self) -> None:
        """
        Test SHAP calculation when SHAP not available.
        """
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._shap = None

        mock_model = MagicMock()
        X_sample = rng.standard_normal((10, 5))

        with patch("builtins.__import__", side_effect=ImportError):
            result = trainer._calculate_shap_values(mock_model, X_sample)

        assert result == {}

    def test_save_model(self) -> None:
        """
        Test model saving.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._model = MagicMock()
        trainer._feature_names = ["feature1", "feature2"]
        trainer._training_metrics = {"accuracy": 0.95}
        trainer._scaler = MagicMock()

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("pickle.dump") as mock_pickle:
                trainer.save_model("test_model.pkl")

                mock_pickle.assert_called_once()
                saved_data = mock_pickle.call_args[0][0]
                assert "model" in saved_data
                assert "feature_names" in saved_data
                assert "training_metrics" in saved_data
                assert "config" in saved_data

    def test_save_model_not_fitted(self) -> None:
        """
        Test error when saving unfitted model.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        with pytest.raises(ValueError, match="Model must be fitted"):
            trainer.save_model("test_model.pkl")

    def test_get_feature_importance_summary(self) -> None:
        """
        Test feature importance summary.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = [f"feature_{i}" for i in range(15)]

        # Mock model with feature importances
        trainer._model = MagicMock()
        importances = rng.random(15)
        importances = importances / importances.sum()  # Normalize
        trainer._model.feature_importances_ = importances

        # Mock SHAP results
        shap_importance = {f"feature_{i}": float(importances[i] * 0.9) for i in range(15)}
        trainer._training_metrics = {
            "shap_results": {
                "shap_importance": shap_importance,
            },
        }

        summary = trainer.get_feature_importance_summary()

        assert "xgb_importance" in summary
        assert "shap_importance" in summary
        assert "top_10_features" in summary
        assert len(summary["top_10_features"]) == 10

    def test_get_feature_importance_summary_not_fitted(self) -> None:
        """
        Test error when getting importance from unfitted model.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        with pytest.raises(ValueError, match="Model must be fitted"):
            trainer.get_feature_importance_summary()

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    def test_add_cross_sectional_features(self, mock_pl: MagicMock) -> None:
        """
        Test cross-sectional feature addition.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"A": "Tech", "B": "Finance"},
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["ticker", "sector", "timestamp", "return_5", "return_20", "rsi"]
        mock_df.with_columns.return_value = mock_df

        # Mock column operations
        mock_col = MagicMock()
        mock_col.rank.return_value = mock_col
        mock_col.over.return_value = mock_col
        mock_col.alias.return_value = mock_col
        mock_col.mean.return_value = mock_col
        mock_col.__sub__.return_value = mock_col
        mock_pl.col.return_value = mock_col

        result = trainer._add_cross_sectional_features(mock_df)

        assert result == mock_df
        # Verify that ranking was attempted
        mock_pl.col.assert_called()

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.features.engineering.pl")
    def test_multi_asset_with_scaler(
        self,
        mock_eng_pl: MagicMock,
        mock_pl: MagicMock,
    ) -> None:
        """
        Test multi-asset preparation with feature scaling.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # The scaler is already mocked via the decorator @patch

        # Mock DataFrame with proper close column chaining
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df.__len__.return_value = 150

        # Mock the close column with proper chaining for target creation
        mock_close_series = MagicMock()
        mock_close_series.shift.return_value = mock_close_series
        mock_division_result = MagicMock()
        mock_division_result.__sub__.return_value = mock_division_result
        mock_comparison_result = MagicMock()
        mock_comparison_result.cast.return_value = MagicMock(
            to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
        )
        mock_division_result.__gt__ = MagicMock(return_value=mock_comparison_result)
        mock_close_series.__truediv__.return_value = mock_division_result

        def mock_getitem(key: str) -> Any:
            if key == "close":
                return mock_close_series
            else:
                return MagicMock(to_numpy=MagicMock(return_value=rng.random(100)))

        mock_df.__getitem__.side_effect = mock_getitem

        data_dict = {"AAPL": mock_df}

        # Mock polars operations
        mock_combined = MagicMock()
        mock_combined.columns = ["feature1", "feature2", "ticker", "sector"]
        mock_combined.select.return_value = MagicMock(
            to_numpy=MagicMock(return_value=rng.standard_normal((100, 2))),
        )
        mock_combined.with_columns.return_value = mock_combined
        mock_pl.concat.return_value = mock_combined
        mock_pl.lit.return_value = MagicMock(alias=MagicMock(side_effect=lambda x: x))

        # Mock feature engineer
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            # Create a proper mock features DataFrame
            mock_features = MagicMock()
            mock_features.__len__.return_value = 100
            mock_features.with_columns.return_value = mock_features
            mock_features.__getitem__.return_value = mock_features
            mock_calc.return_value = (mock_features, None)

            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = ["feature1", "feature2"]

                # Patch _add_cross_sectional_features to return the mocked DataFrame
                with patch.object(
                    trainer,
                    "_add_cross_sectional_features",
                    return_value=mock_combined,
                ):
                    X, y, metadata = trainer.prepare_data(data_dict)

                # Verify we got results
                assert isinstance(X, np.ndarray)
                assert isinstance(y, np.ndarray)
