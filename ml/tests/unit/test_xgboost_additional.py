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
Additional tests for XGBoostTrainer to reach 90% coverage.

Tests specific edge cases and error handling paths.

"""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.tests.unit.test_fixtures import mock_sklearn
from ml.training.xgboost import XGBoostTrainer


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


class TestXGBoostTrainerAdditional:
    """
    Additional tests for XGBoost trainer coverage.
    """

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    def test_single_asset_with_existing_target(self, mock_pl: MagicMock) -> None:
        """
        Test single asset data preparation with existing target column.
        """
        config = XGBoostTrainingConfig(data_source="test", target_column="my_target")
        trainer = XGBoostTrainer(config)

        # Mock DataFrame with target column
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "my_target"]
        mock_df.__len__.return_value = 200
        mock_df.drop.return_value = mock_df
        mock_df.with_columns.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.to_numpy.return_value = rng.standard_normal((150, 10))

        # Mock target column access
        mock_target = MagicMock()
        mock_target.to_numpy.return_value = rng.random(150)
        mock_df.__getitem__.side_effect = lambda x: mock_target if x == "my_target" else mock_df

        # Mock features DataFrame returned by calculate_features_batch
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal((200, 10))
        mock_features_df.__len__.return_value = 200

        # Mock feature engineer
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            mock_calc.return_value = (mock_features_df, None)
            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = [f"feature_{i}" for i in range(10)]

                X, y, metadata = trainer.prepare_data(mock_df, "my_target")

                assert isinstance(X, np.ndarray)
                assert isinstance(y, np.ndarray)
                assert metadata["target_type"] == "classification"

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    def test_single_asset_regression_target(self, mock_pl: MagicMock) -> None:
        """
        Test single asset with regression objective.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            objective="reg:squarederror",
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "target"]
        mock_df.__len__.return_value = 200
        mock_df.drop.return_value = mock_df
        mock_df.to_numpy.return_value = rng.standard_normal((150, 10))

        # Mock target column with continuous values
        mock_target = MagicMock()
        mock_target.to_numpy.return_value = rng.standard_normal(200)  # Continuous values
        mock_df.__getitem__.side_effect = lambda x: mock_target if x == "target" else mock_df

        # Mock features DataFrame
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal((200, 10))
        mock_features_df.__len__.return_value = 200

        # Mock feature engineer
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            mock_calc.return_value = (mock_features_df, None)
            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = [f"feature_{i}" for i in range(10)]

                X, y, metadata = trainer.prepare_data(mock_df, "target")

                assert metadata["target_type"] == "regression"
                # Verify target is not binarized for regression
                assert not np.all((y == 0) | (y == 1))

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    def test_single_asset_with_scaler(self, mock_pl: MagicMock) -> None:
        """
        Test single asset with feature scaling.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df.__len__.return_value = 200
        mock_df.drop.return_value = mock_df
        mock_df.to_numpy.return_value = rng.standard_normal((150, 10))

        # Create proper mock for column access
        def create_column_mock(data: Any) -> MagicMock:
            mock_col = MagicMock()
            mock_col.to_numpy.return_value = data
            shifted_data = np.concatenate([data[1:], [np.nan]])
            mock_shifted = MagicMock()
            mock_shifted.to_numpy.return_value = shifted_data
            mock_col.shift.return_value = mock_shifted
            comparison = shifted_data > data
            mock_comparison = MagicMock()
            mock_comparison.to_numpy.return_value = comparison
            mock_comparison.cast.return_value = MagicMock(
                __getitem__=MagicMock(
                    side_effect=lambda idx: MagicMock(
                        to_numpy=MagicMock(return_value=comparison[idx].astype(np.int32)),
                    ),
                ),
            )
            mock_shifted.__gt__.return_value = mock_comparison
            mock_col.__getitem__ = MagicMock(
                side_effect=lambda idx: MagicMock(
                    to_numpy=MagicMock(return_value=data[idx]),
                ),
            )
            return mock_col

        closes = rng.standard_normal(200) * 2 + 100
        mock_df.__getitem__.side_effect = lambda col: create_column_mock(closes)

        # Mock scaler
        mock_scaler = MagicMock()

        # Mock features DataFrame
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal(
            (199, 10),
        )  # 199 after removing last
        mock_features_df.__len__.return_value = 200
        mock_features_df.__getitem__.return_value = mock_features_df

        # Mock feature engineer with scaler
        with patch.object(trainer._feature_engineer, "calculate_features_batch") as mock_calc:
            mock_calc.return_value = (mock_features_df, mock_scaler)
            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = [f"feature_{i}" for i in range(10)]

                # Mock column access for target creation
                mock_series = MagicMock()
                mock_series.shift.return_value = MagicMock(
                    __gt__=MagicMock(
                        return_value=MagicMock(
                            cast=MagicMock(
                                return_value=MagicMock(
                                    __getitem__=MagicMock(
                                        return_value=MagicMock(
                                            to_numpy=MagicMock(
                                                return_value=rng.integers(0, 2, 149),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
                mock_df.__getitem__.return_value = mock_series

                X, y, metadata = trainer.prepare_data(mock_df, "target")

                assert trainer._scaler == mock_scaler
                assert metadata["scaler"] == mock_scaler

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.features.engineering.pl")
    def test_multi_asset_insufficient_data(
        self,
        mock_eng_pl: MagicMock,
        mock_pl: MagicMock,
    ) -> None:
        """
        Test multi-asset with some assets having insufficient data.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"},
            feature_config=MLFeatureConfig(lookback_window=100),
        )
        trainer = XGBoostTrainer(config)

        # Create mock DataFrames with varying lengths
        data_dict = {}

        # AAPL - insufficient data
        mock_df_aapl = MagicMock()
        mock_df_aapl.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df_aapl.__len__.return_value = 50  # Less than lookback_window
        data_dict["AAPL"] = mock_df_aapl

        # MSFT - sufficient data
        mock_df_msft = MagicMock()
        mock_df_msft.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df_msft.__len__.return_value = 150
        mock_df_msft.with_columns.return_value = mock_df_msft
        mock_df_msft.to_numpy.return_value = rng.standard_normal((100, 10))

        # Mock column operations for MSFT
        mock_returns = MagicMock()
        mock_returns.__gt__.return_value = MagicMock(
            cast=MagicMock(
                return_value=MagicMock(
                    to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
                ),
            ),
        )
        mock_series = MagicMock()
        mock_series.shift.return_value = mock_series
        mock_series.__truediv__.return_value = MagicMock(
            __sub__=MagicMock(return_value=mock_returns),
        )
        mock_df_msft.__getitem__.return_value = mock_series
        data_dict["MSFT"] = mock_df_msft

        # JPM - sufficient data
        mock_df_jpm = MagicMock()
        mock_df_jpm.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df_jpm.__len__.return_value = 150
        mock_df_jpm.with_columns.return_value = mock_df_jpm
        mock_df_jpm.to_numpy.return_value = rng.standard_normal((100, 10))
        mock_df_jpm.__getitem__.return_value = mock_series
        data_dict["JPM"] = mock_df_jpm

        # Mock polars concat
        mock_combined = MagicMock()
        mock_combined.columns = ["feature1", "feature2", "ticker", "sector"]
        # Set up select to return proper mock with to_numpy
        mock_selected = MagicMock()
        mock_selected.to_numpy.return_value = rng.standard_normal((200, 2))
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

                # Should skip AAPL but process MSFT and JPM
                assert metadata["n_assets"] == 3
                assert len(metadata["asset_metadata"]) == 2  # Only MSFT and JPM

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    def test_multi_asset_all_insufficient_data(self, mock_pl: MagicMock) -> None:
        """
        Test error when all assets have insufficient data.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(lookback_window=100),
        )
        trainer = XGBoostTrainer(config)

        # Create mock DataFrame with insufficient data
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df.__len__.return_value = 50  # Less than lookback_window

        data_dict = {"AAPL": mock_df}

        with pytest.raises(ValueError, match="No assets had sufficient data"):
            trainer.prepare_data(data_dict)

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.features.engineering.pl")
    def test_multi_asset_with_existing_target(
        self,
        mock_eng_pl: MagicMock,
        mock_pl: MagicMock,
    ) -> None:
        """
        Test multi-asset with existing target columns.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            target_column="my_target",
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame with target
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "my_target"]
        mock_df.__len__.return_value = 150
        mock_df.with_columns.return_value = mock_df
        mock_df.to_numpy.return_value = rng.standard_normal((100, 10))

        # Mock the target column access
        mock_target_series = MagicMock()
        mock_target_series.to_numpy.return_value = rng.integers(0, 2, 100)
        mock_df.__getitem__.return_value = mock_target_series

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

        # Mock target column
        mock_target = MagicMock()
        mock_target.to_numpy.return_value = rng.random(100)

        def mock_getitem(key: str) -> Any:
            if key == "my_target":
                return mock_target
            elif key == "close":
                return mock_close_series
            else:
                return mock_df

        mock_df.__getitem__.side_effect = mock_getitem

        data_dict = {"AAPL": mock_df}

        # Mock polars concat
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

                assert isinstance(X, np.ndarray)
                assert isinstance(y, np.ndarray)

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.training.xgboost.HAS_SKLEARN", False)
    def test_multi_asset_no_sklearn_error(self, mock_pl: MagicMock) -> None:
        """
        Test error when sklearn not available for multi-asset scaling.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_df.__len__.return_value = 150
        mock_df.with_columns.return_value = mock_df

        # Mock column operations
        mock_returns = MagicMock()
        mock_returns.__gt__.return_value = MagicMock(
            cast=MagicMock(
                return_value=MagicMock(
                    to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
                ),
            ),
        )
        mock_series = MagicMock()
        mock_series.shift.return_value = mock_series
        mock_series.__truediv__.return_value = MagicMock(
            __sub__=MagicMock(return_value=mock_returns),
        )
        mock_df.__getitem__.return_value = mock_series

        data_dict = {"AAPL": mock_df}

        # Mock polars concat
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
            mock_calc.return_value = (MagicMock(), None)
            with patch.object(trainer._feature_engineer, "get_feature_names") as mock_names:
                mock_names.return_value = ["feature1", "feature2"]

                with pytest.raises(ImportError, match="sklearn is required"):
                    trainer.prepare_data(data_dict)

    def test_calculate_shap_values_binary_classification(self) -> None:
        """
        Test SHAP calculation for binary classification (list output).
        """
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feature1", "feature2", "feature3"]

        # Mock SHAP with list output (binary classification)
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.array([[0.05, 0.1, 0.15], [0.1, 0.15, 0.2]]),  # Negative class
            np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),  # Positive class
        ]
        mock_explainer.expected_value = [0.3, 0.7]

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        trainer._shap = mock_shap

        mock_model = MagicMock()
        X_sample = rng.standard_normal((2, 3))

        result = trainer._calculate_shap_values(mock_model, X_sample)

        # Should use positive class (index 1)
        assert result["shap_values"].shape == (2, 3)
        assert np.array_equal(
            result["shap_values"],
            np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
        )

    def test_train_model_with_shap_enabled(self) -> None:
        """
        Test training with SHAP analysis enabled.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=10,
            enable_shap=True,
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

        # Mock SHAP
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = rng.standard_normal((20, 5))
        mock_explainer.expected_value = 0.5

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        trainer._shap = mock_shap

        X_train = rng.standard_normal((100, 5))
        y_train = rng.integers(0, 2, 100)
        X_val = rng.standard_normal((20, 5))
        y_val = rng.integers(0, 2, 20)

        result = trainer._train_model(X_train, y_train, X_val, y_val)

        assert "shap_results" in result
        assert "shap_importance" in result["shap_results"]

    def test_calculate_shap_values_max_samples(self) -> None:
        """
        Test SHAP calculation with sample limit.
        """
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["feature1", "feature2"]

        # Mock SHAP
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = rng.standard_normal((100, 2))  # Limited to 100
        mock_explainer.expected_value = 0.5

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        trainer._shap = mock_shap

        mock_model = MagicMock()
        X_sample = rng.standard_normal((2000, 2))  # Large sample

        _ = trainer._calculate_shap_values(mock_model, X_sample, max_samples=100)

        # Should only use first 100 samples
        call_args = mock_explainer.shap_values.call_args
        assert call_args[0][0].shape[0] == 100  # Only 100 samples used

    def test_get_feature_importance_without_shap(self) -> None:
        """
        Test feature importance summary without SHAP results.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = [f"feature_{i}" for i in range(5)]

        # Mock model with feature importances
        trainer._model = MagicMock()
        trainer._model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        # No SHAP results
        trainer._training_metrics = {}

        summary = trainer.get_feature_importance_summary()

        assert "xgb_importance" in summary
        assert "shap_importance" not in summary
        assert "top_10_features" in summary
        assert len(summary["top_10_features"]) == 5  # Only 5 features
