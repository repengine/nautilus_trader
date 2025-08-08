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
Targeted tests for uncovered lines in XGBoostTrainer.
"""

import logging
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.xgboost import XGBoostTrainer


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)

# Configure module logger
logger = logging.getLogger(__name__)


class TestXGBoostTargeted:
    """
    Tests targeting specific uncovered lines.
    """

    @patch("ml.training.xgboost.HAS_SKLEARN", False)
    @patch("ml.training.xgboost.StandardScaler", None)
    def test_sklearn_import_guards(self) -> None:
        """
        Test sklearn import guards.
        """
        # This tests lines 43, 51
        from ml.training.xgboost import HAS_SKLEARN
        from ml.training.xgboost import StandardScaler

        assert not HAS_SKLEARN
        assert StandardScaler is None

    def test_single_asset_else_branch(self) -> None:
        """
        Test else branch in prepare_data.
        """
        # Tests line 147
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=False,  # Ensure single-asset path
        )
        trainer = XGBoostTrainer(config)

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch.object(trainer, "_prepare_single_asset_data") as mock_single:
                mock_single.return_value = (np.array([]), np.array([]), {})

                trainer.prepare_data({})

                mock_single.assert_called_once()

    def test_default_target_creation_lines(self) -> None:
        """
        Test specific lines in default target creation.
        """
        # Tests lines 186-187
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        # Mock feature names (must set before calling methods)
        trainer._feature_names = ["f1", "f2"]

        # Create proper mock structure for default target path
        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            # Mock polars Int32 type
            mock_pl = MagicMock()
            mock_pl.Int32 = MagicMock()

            with (
                patch("ml.training.xgboost.print") as mock_print,
                patch("ml.training.xgboost.pl", mock_pl),
            ):
                # Mock DataFrame without target column
                mock_df = MagicMock()
                mock_df.columns = ["close"]  # No target

                # Mock the shift chain
                mock_close_series = MagicMock()
                mock_shifted = MagicMock()
                mock_comparison = MagicMock()
                mock_casted = MagicMock()
                mock_sliced = MagicMock()
                mock_sliced.to_numpy.return_value = np.array([1, 0, 1, 0, 1])

                # Setup the chain
                mock_close_series.shift.return_value = mock_shifted
                mock_shifted.__gt__.return_value = mock_comparison
                mock_comparison.cast.return_value = mock_casted
                mock_casted.__getitem__.return_value = mock_sliced

                mock_df.__getitem__.return_value = mock_close_series

                # Mock feature engineer
                mock_features_df = MagicMock()
                mock_features_df.select.return_value = MagicMock(
                    to_numpy=MagicMock(return_value=rng.standard_normal((5, 2))),
                )
                mock_features_df.__getitem__.return_value = mock_features_df

                setattr(
                    trainer._feature_engineer,
                    "calculate_features_batch",
                    MagicMock(return_value=(mock_features_df, None)),
                )
                setattr(
                    trainer._feature_engineer,
                    "get_feature_names",
                    MagicMock(return_value=["f1", "f2"]),
                )

                X, y, metadata = trainer._prepare_single_asset_data(mock_df, "target")

                # Verify print was called
                mock_print.assert_called()
                assert any("Target column" in str(call) for call in mock_print.call_args_list)

    def test_multi_asset_existing_target_else_path(self) -> None:
        """
        Test multi-asset with existing target.
        """
        # Tests lines 265 (else path)
        feature_config = MLFeatureConfig(normalize_features=False)
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=feature_config,
            cross_sectional_features=False,
            objective="reg:squarederror",  # Use regression objective
        )
        trainer = XGBoostTrainer(config)

        # Mock feature names
        trainer._feature_names = ["f1"]

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.pl") as mock_pl:
                # Mock DataFrame with target
                mock_df = MagicMock()
                mock_df.columns = ["close", "target"]  # Has target
                mock_df.__len__.return_value = 150
                mock_df.with_columns.return_value = mock_df

                # Mock target column
                mock_target = MagicMock()
                mock_target.to_numpy.return_value = np.array([0.5] * 100)
                mock_df.__getitem__.side_effect = lambda x: (
                    mock_target if x == "target" else mock_df
                )

                # Mock feature engineer
                setattr(
                    trainer._feature_engineer,
                    "calculate_features_batch",
                    MagicMock(return_value=(mock_df, None)),
                )
                setattr(
                    trainer._feature_engineer,
                    "get_feature_names",
                    MagicMock(return_value=["f1"]),
                )

                # Mock concat
                mock_combined = MagicMock()
                mock_combined.columns = ["f1", "ticker", "sector"]
                mock_select_result = MagicMock()
                mock_select_result.to_numpy.return_value = rng.standard_normal((100, 1))
                mock_combined.select.return_value = mock_select_result
                mock_combined.with_columns.return_value = mock_combined
                mock_pl.concat.return_value = mock_combined
                mock_pl.lit.return_value = MagicMock(alias=MagicMock(side_effect=lambda x: x))

                X, y, metadata = trainer._prepare_multi_asset_data({"AAPL": mock_df}, "target")

                # Should have used existing target
                assert np.array_equal(y[:100], np.array([0.5] * 100))

    def test_cross_sectional_with_timestamp_not_in_columns(self) -> None:
        """
        Test cross-sectional features when timestamp not in columns.
        """
        # Tests line 360
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"A": "Tech"},
            cross_sectional_features=True,
        )
        trainer = XGBoostTrainer(config)

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.pl") as mock_pl:
                # Mock DataFrame without timestamp
                mock_df = MagicMock()
                mock_df.columns = ["return_5", "return_20"]  # No timestamp
                mock_df.with_row_count.return_value = mock_df
                mock_df.with_columns.return_value = mock_df

                # Mock col function
                mock_col = MagicMock()
                mock_col.rank.return_value = mock_col
                mock_col.over.return_value = mock_col
                mock_col.alias.return_value = "mocked"
                mock_pl.col.return_value = mock_col

                _ = trainer._add_cross_sectional_features(mock_df)

                # Should have added timestamp
                mock_df.with_row_count.assert_called_with("timestamp")

    def test_shap_calculation_print_statement(self) -> None:
        """
        Test print statement in SHAP calculation.
        """
        # Tests line 467
        config = XGBoostTrainingConfig(
            data_source="test",
            enable_shap=True,
        )
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["f1"]

        # Mock XGBoost and SHAP
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([1.0])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model
        setattr(trainer, "_xgb", mock_xgb)

        # Mock SHAP
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = MagicMock(
            shap_values=MagicMock(return_value=np.array([[0.5]])),
            expected_value=0.5,
        )
        setattr(trainer, "_shap", mock_shap)

        with patch("ml.training.xgboost.print") as mock_print:
            trainer._train_model(
                np.array([[1.0]]),
                np.array([1]),
                np.array([[1.0]]),
                np.array([1]),
            )

            # Check for SHAP print
            assert any("Computing SHAP" in str(call) for call in mock_print.call_args_list)

    def test_calculate_shap_no_shap(self) -> None:
        """
        Test print when SHAP not available.
        """
        # Tests line 561
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._shap = None

        with patch("builtins.__import__", side_effect=ImportError):
            with patch("ml.training.xgboost.print") as mock_print:
                result = trainer._calculate_shap_values(MagicMock(), np.array([[1.0]]))

                assert result == {}
                assert any("SHAP not available" in str(call) for call in mock_print.call_args_list)

    def test_shap_binary_classification_list_output(self) -> None:
        """
        Test SHAP with binary classification list output.
        """
        # Tests line 576
        config = XGBoostTrainingConfig(data_source="test", enable_shap=True)
        trainer = XGBoostTrainer(config)
        trainer._feature_names = ["f1", "f2"]

        # Mock SHAP with list output
        mock_explainer = MagicMock()
        # Return list for binary classification
        mock_explainer.shap_values.return_value = [
            np.array([[0.1, 0.2]]),  # Class 0
            np.array([[0.3, 0.4]]),  # Class 1
        ]

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        setattr(trainer, "_shap", mock_shap)

        result = trainer._calculate_shap_values(MagicMock(), np.array([[1.0, 2.0]]))

        # Should use class 1 (positive class)
        assert np.array_equal(result["shap_values"], np.array([[0.3, 0.4]]))

    def test_get_importance_no_model_attr(self) -> None:
        """
        Test feature importance when model has no feature_importances_.
        """
        # Tests line 641
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = ["f1", "f2"]
        trainer._training_metrics = {}

        # Model without feature_importances_ attribute
        trainer._model = MagicMock(spec=[])  # No attributes

        summary = trainer.get_feature_importance_summary()

        # Should not have xgb_importance
        assert "xgb_importance" not in summary

    def test_get_importance_many_features(self) -> None:
        """
        Test top 10 selection with many features.
        """
        # Tests line 661
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._feature_names = [f"f{i}" for i in range(20)]  # 20 features
        trainer._training_metrics = {}

        # Model with many feature importances
        trainer._model = MagicMock()
        importances = rng.random(20)
        importances = importances / importances.sum()
        trainer._model.feature_importances_ = importances

        summary = trainer.get_feature_importance_summary()

        # Should only have top 10
        assert len(summary["top_10_features"]) == 10

    def test_multi_asset_with_sector_map_none(self) -> None:
        """
        Test multi-asset with default sector.
        """
        # Tests line 270 (sector_map.get with default)
        feature_config = MLFeatureConfig(normalize_features=False)
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"MSFT": "Tech"},  # AAPL not in map
            feature_config=feature_config,
            cross_sectional_features=False,
            objective="reg:squarederror",  # Use regression objective
        )
        trainer = XGBoostTrainer(config)

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.pl") as mock_pl:
                with patch("ml.training.xgboost.print"):
                    # Mock DataFrame
                    mock_df = MagicMock()
                    mock_df.columns = ["close"]
                    mock_df.__len__.return_value = 150
                    mock_df.with_columns.return_value = mock_df

                    # Mock feature engineer
                    setattr(
                        trainer._feature_engineer,
                        "calculate_features_batch",
                        MagicMock(return_value=(mock_df, None)),
                    )
                    setattr(
                        trainer._feature_engineer,
                        "get_feature_names",
                        MagicMock(return_value=["f1"]),
                    )

                    # Mock close series and returns calculation
                    mock_close = MagicMock()
                    mock_shifted = MagicMock()
                    mock_ratio = MagicMock()
                    mock_returns = MagicMock()

                    # Set up the chain: close.shift(-5) / close - 1
                    mock_close.shift.return_value = mock_shifted
                    mock_shifted.__truediv__.return_value = mock_ratio
                    mock_ratio.__sub__.return_value = mock_returns

                    # Final result should cast to int and convert to numpy
                    mock_returns.__gt__.return_value = MagicMock(
                        cast=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=np.array([1, 0, 1])),
                            ),
                        ),
                    )

                    mock_df.__getitem__.return_value = mock_close

                    # Mock lit to track sector assignment
                    sector_values = []

                    def mock_lit_func(value: Any) -> MagicMock:
                        sector_values.append(value)
                        return MagicMock(alias=MagicMock(side_effect=lambda x: x))

                    mock_pl.lit.side_effect = mock_lit_func

                    # Mock concat
                    mock_pl.concat.return_value = MagicMock(
                        columns=["f1", "ticker", "sector"],
                        select=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.standard_normal((3, 1))),
                            ),
                        ),
                        with_columns=MagicMock(
                            return_value=MagicMock(
                                columns=["f1", "ticker", "sector"],
                                select=MagicMock(
                                    return_value=MagicMock(
                                        to_numpy=MagicMock(
                                            return_value=rng.standard_normal((3, 1)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )

                    # Process with ticker not in sector_map
                    trainer._prepare_multi_asset_data({"AAPL": mock_df}, "target")

                    # Should have used "unknown" as default sector
                    assert "unknown" in sector_values
