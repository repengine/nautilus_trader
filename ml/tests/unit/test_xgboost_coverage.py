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
Simple tests to reach 90% coverage for XGBoost trainer.

Uses direct method testing with minimal mocking.

"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.xgboost import XGBoostTrainer


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


class TestXGBoostCoverage:
    """
    Tests for XGBoost coverage.
    """

    def test_trainer_base_functionality(self) -> None:
        """
        Test base trainer functionality.
        """
        # Test with feature_config = None
        config1 = XGBoostTrainingConfig(data_source="test", feature_config=None)
        trainer1 = XGBoostTrainer(config1)
        assert trainer1._feature_engineer is not None

        # Test with feature_config provided
        config2 = XGBoostTrainingConfig(
            data_source="test",
            feature_config=MLFeatureConfig(lookback_window=50),
        )
        trainer2 = XGBoostTrainer(config2)
        assert trainer2._feature_engineer is not None

    def test_prepare_methods_directly(self) -> None:
        """
        Test prepare data methods with proper mocking.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        # Mock feature engineer methods
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(
                return_value=(
                    MagicMock(
                        select=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.standard_normal((100, 10))),
                            ),
                        ),
                        __len__=MagicMock(return_value=100),
                        __getitem__=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.random(100)),
                            ),
                        ),
                    ),
                    None,
                ),
            ),
        )
        setattr(
            trainer._feature_engineer,
            "get_feature_names",
            MagicMock(return_value=["f1", "f2", "f3", "f4", "f5"]),
        )

        # Test single asset path
        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            # Mock DataFrame
            mock_df = MagicMock()
            mock_df.columns = ["open", "high", "low", "close", "volume", "target"]
            mock_df.__getitem__.return_value = MagicMock(
                to_numpy=MagicMock(return_value=rng.random(100)),
            )

            X, y, metadata = trainer._prepare_single_asset_data(mock_df, "target")

            assert X.shape == (100, 10)
            assert y.shape == (100,)
            assert metadata["n_features"] == 10
            assert metadata["target_type"] == "classification"

    def test_feature_engineer_with_scaler(self) -> None:
        """
        Test feature engineering with scaler.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # Mock scaler
        mock_scaler = MagicMock()

        # Mock feature engineer to return scaler
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(
                return_value=(
                    MagicMock(
                        select=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.standard_normal((100, 5))),
                            ),
                        ),
                        __len__=MagicMock(return_value=100),
                        __getitem__=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.random(100)),
                            ),
                        ),
                    ),
                    mock_scaler,
                ),
            ),
        )
        setattr(
            trainer._feature_engineer,
            "get_feature_names",
            MagicMock(return_value=["f1", "f2"]),
        )

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            mock_df = MagicMock()
            mock_df.columns = ["close", "target"]
            mock_df.__getitem__.return_value = MagicMock(
                to_numpy=MagicMock(return_value=rng.random(100)),
            )

            X, y, metadata = trainer._prepare_single_asset_data(mock_df, "target")

            assert trainer._scaler == mock_scaler
            assert metadata["scaler"] == mock_scaler

    def test_target_creation_default_path(self) -> None:
        """
        Test default target creation when column missing.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        # Setup mocks
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(
                return_value=(
                    MagicMock(
                        select=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.standard_normal((99, 5))),
                            ),
                        ),
                        __len__=MagicMock(return_value=99),
                        __getitem__=MagicMock(
                            side_effect=lambda idx: (
                                MagicMock(
                                    select=MagicMock(
                                        return_value=MagicMock(
                                            to_numpy=MagicMock(
                                                return_value=rng.standard_normal((98, 5)),
                                            ),
                                        ),
                                    ),
                                    __len__=MagicMock(return_value=98),
                                )
                                if isinstance(idx, slice)
                                else MagicMock()
                            ),
                        ),
                    ),
                    None,
                ),
            ),
        )
        setattr(
            trainer._feature_engineer,
            "get_feature_names",
            MagicMock(return_value=["f1", "f2"]),
        )

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.pl") as mock_pl:
                mock_pl.Int32 = "Int32"  # Mock the Int32 type

                # Mock DataFrame without target column
                mock_df = MagicMock()
                mock_df.columns = ["close", "open"]  # No target column
                mock_df.__len__.return_value = 100

                # Mock close column behavior for target creation
                mock_close = MagicMock()
                mock_shifted = MagicMock()
                mock_comparison = MagicMock()
                mock_casted = MagicMock()

                # Setup chained calls
                mock_close.shift.return_value = mock_shifted
                mock_shifted.__gt__.return_value = mock_comparison
                mock_comparison.cast.return_value = mock_casted
                mock_casted.__getitem__.return_value = MagicMock(
                    to_numpy=MagicMock(
                        return_value=rng.integers(
                            0,
                            2,
                            98,
                        ),
                    ),  # Match features shape after slicing
                )

                mock_df.__getitem__.side_effect = lambda x: mock_close if x == "close" else mock_df

                X, y, metadata = trainer._prepare_single_asset_data(mock_df, "target")

                assert X.shape[0] == y.shape[0]
                assert np.all((y == 0) | (y == 1))  # Binary target

    def test_regression_objective_path(self) -> None:
        """
        Test regression objective doesn't binarize target.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            objective="reg:squarederror",
        )
        trainer = XGBoostTrainer(config)

        # Setup mocks
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(
                return_value=(
                    MagicMock(
                        select=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.standard_normal((100, 5))),
                            ),
                        ),
                        __len__=MagicMock(return_value=100),
                    ),
                    None,
                ),
            ),
        )
        setattr(
            trainer._feature_engineer,
            "get_feature_names",
            MagicMock(return_value=["f1", "f2"]),
        )

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            mock_df = MagicMock()
            mock_df.columns = ["close", "target"]
            mock_df.__getitem__.return_value = MagicMock(
                to_numpy=MagicMock(return_value=rng.standard_normal(100)),  # Continuous values
            )

            X, y, metadata = trainer._prepare_single_asset_data(mock_df, "target")

            assert metadata["target_type"] == "regression"
            # For regression, target should not be binary
            assert not np.all((y == 0) | (y == 1))

    def test_multi_asset_cross_sectional_disabled(self) -> None:
        """
        Test multi-asset without cross-sectional features.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            cross_sectional_features=False,  # Disabled
        )
        trainer = XGBoostTrainer(config)

        # Mock DataFrame - should return unchanged
        mock_df = MagicMock()
        mock_df.with_row_count.return_value = mock_df  # For timestamp adding

        _ = trainer._add_cross_sectional_features(mock_df)

        # When disabled, should just add timestamp and return
        mock_df.with_row_count.assert_called_once_with("timestamp")

    def test_multi_asset_insufficient_data_error(self) -> None:
        """
        Test error when all multi-asset data is insufficient.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(lookback_window=100),
        )
        trainer = XGBoostTrainer(config)

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            # Mock insufficient data
            mock_df = MagicMock()
            mock_df.__len__.return_value = 50  # Less than lookback

            with pytest.raises(ValueError, match="No assets had sufficient data"):
                trainer._prepare_multi_asset_data({"AAPL": mock_df}, "target")

    def test_save_model_creates_parent_dirs(self) -> None:
        """
        Test save_model creates parent directories.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)
        trainer._is_fitted = True
        trainer._model = MagicMock()
        trainer._feature_names = ["f1"]
        trainer._training_metrics = {}

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("builtins.open", create=True):
                with patch("pickle.dump"):
                    trainer.save_model("nested/path/model.pkl")

                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_print_statements_in_methods(self) -> None:
        """
        Test print statements are covered.
        """
        config = XGBoostTrainingConfig(data_source="test")
        trainer = XGBoostTrainer(config)

        # Capture prints from training
        trainer._feature_names = ["f1"]
        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = MagicMock(
            fit=MagicMock(),
            best_iteration=5,
            best_score=0.85,
            feature_importances_=np.array([1.0]),
        )
        setattr(trainer, "_xgb", mock_xgb)

        with patch("builtins.print") as mock_print:
            trainer._train_model(
                rng.standard_normal((10, 1)),
                rng.integers(0, 2, 10),
                rng.standard_normal((5, 1)),
                rng.integers(0, 2, 5),
            )

            # Check various print statements
            print_messages = [str(call) for call in mock_print.call_args_list]
            assert any("Training XGBoost" in str(msg) for msg in print_messages)
            assert any("completed" in str(msg) for msg in print_messages)

    def test_multi_asset_sklearn_scaling(self) -> None:
        """
        Test multi-asset with sklearn scaling.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # Setup feature engineer
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(return_value=(MagicMock(), None)),
        )
        setattr(
            trainer._feature_engineer,
            "get_feature_names",
            MagicMock(return_value=["f1", "f2"]),
        )

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.HAS_SKLEARN", True):
                with patch("ml.training.xgboost.StandardScaler") as mock_scaler_class:
                    mock_scaler = MagicMock()
                    mock_scaler.fit_transform.return_value = rng.standard_normal((100, 2))
                    mock_scaler_class.return_value = mock_scaler

                    # Setup mocks
                    mock_df = MagicMock()
                    mock_df.__len__.return_value = 150
                    mock_df.columns = ["close"]
                    mock_df.with_columns.return_value = mock_df

                    # Mock target creation
                    mock_returns = MagicMock()
                    mock_returns.__gt__.return_value = MagicMock(
                        cast=MagicMock(
                            return_value=MagicMock(
                                to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
                            ),
                        ),
                    )
                    mock_close = MagicMock()
                    mock_close.shift.return_value = MagicMock(
                        __truediv__=MagicMock(
                            return_value=MagicMock(__sub__=MagicMock(return_value=mock_returns)),
                        ),
                    )
                    mock_df.__getitem__.return_value = mock_close

                    # Mock concat
                    import sys

                    mock_pl = MagicMock()
                    mock_combined = MagicMock()
                    mock_combined.columns = ["f1", "f2", "ticker", "sector"]
                    mock_combined.select.return_value = MagicMock(
                        to_numpy=MagicMock(return_value=rng.standard_normal((100, 2))),
                    )
                    mock_combined.with_columns.return_value = mock_combined
                    mock_pl.concat.return_value = mock_combined
                    mock_pl.lit.return_value = MagicMock(alias=MagicMock(side_effect=lambda x: x))

                    with patch.dict(sys.modules, {"polars": mock_pl}):
                        with patch("ml.training.xgboost.pl", mock_pl):
                            X, y, metadata = trainer._prepare_multi_asset_data(
                                {"AAPL": mock_df},
                                "target",
                            )

                            assert trainer._scaler == mock_scaler
                            mock_scaler.fit_transform.assert_called_once()

    def test_sklearn_not_available_error(self) -> None:
        """
        Test error when sklearn not available for scaling.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            multi_asset=True,
            sector_map={"AAPL": "Tech"},
            feature_config=MLFeatureConfig(normalize_features=True),
        )
        trainer = XGBoostTrainer(config)

        # Setup mocks
        setattr(
            trainer._feature_engineer,
            "calculate_features_batch",
            MagicMock(return_value=(MagicMock(), None)),
        )

        with (
            patch("ml.training.base.HAS_POLARS", True),
            patch("ml.training.xgboost.HAS_POLARS", True),
        ):
            with patch("ml.training.xgboost.HAS_SKLEARN", False):
                # Setup minimal mocks to reach sklearn check
                mock_df = MagicMock()
                mock_df.__len__.return_value = 150
                mock_df.columns = ["close"]
                mock_df.with_columns.return_value = mock_df

                # Mock returns
                mock_returns = MagicMock()
                mock_returns.__gt__.return_value = MagicMock(
                    cast=MagicMock(
                        return_value=MagicMock(
                            to_numpy=MagicMock(return_value=rng.integers(0, 2, 100)),
                        ),
                    ),
                )
                mock_close = MagicMock()
                mock_close.shift.return_value = MagicMock(
                    __truediv__=MagicMock(
                        return_value=MagicMock(__sub__=MagicMock(return_value=mock_returns)),
                    ),
                )
                mock_df.__getitem__.return_value = mock_close

                # Mock concat
                import sys

                mock_pl = MagicMock()
                mock_combined = MagicMock()
                mock_combined.columns = ["f1", "ticker", "sector"]
                mock_combined.select.return_value = MagicMock(
                    to_numpy=MagicMock(return_value=rng.standard_normal((100, 1))),
                )
                mock_pl.concat.return_value = mock_combined
                mock_pl.lit.return_value = MagicMock(alias=MagicMock(side_effect=lambda x: x))

                with patch.dict(sys.modules, {"polars": mock_pl}):
                    with patch("ml.training.xgboost.pl", mock_pl):
                        with pytest.raises(ImportError, match="sklearn is required"):
                            trainer._prepare_multi_asset_data({"AAPL": mock_df}, "target")
