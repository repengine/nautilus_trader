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
Unit tests for XGBoostTrainer.

Tests the XGBoost trainer implementation including configuration validation, feature
preparation, model training, and various advanced features.

"""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.tests.unit.test_fixtures import mock_sklearn
from ml.training.xgboost import HAS_POLARS


if TYPE_CHECKING:
    pass


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


@pytest.fixture
def mock_polars_dataframe() -> MagicMock:
    """
    Mock Polars DataFrame with valid OHLC data.
    """
    # Using rng = np.random.default_rng(42) instead  # For reproducible tests
    n_samples = 150  # Match what's returned in arrays

    # Generate valid OHLC data
    base_price = 100.0
    opens = base_price + rng.standard_normal(n_samples) * 2
    closes = opens + rng.standard_normal(n_samples) * 1.5

    # Ensure high >= max(open, close) and low <= min(open, close)
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n_samples)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n_samples)) * 0.5

    # Round to 4 decimal places to avoid precision issues
    opens = np.round(opens, 4)
    highs = np.round(highs, 4)
    lows = np.round(lows, 4)
    closes = np.round(closes, 4)
    volumes = np.round(1000 + rng.standard_normal(n_samples) * 100, 0)

    mock_df = MagicMock()
    columns_list = ["timestamp", "open", "high", "low", "close", "volume", "target"]
    mock_df.columns = columns_list

    # Create a proper mock for columns that supports 'in' operator
    mock_columns = MagicMock()
    mock_columns.__contains__ = MagicMock(side_effect=lambda x: x in columns_list)
    mock_columns.__iter__ = MagicMock(return_value=iter(columns_list))
    mock_df.columns = mock_columns
    mock_df.__len__.return_value = n_samples
    mock_df.drop.return_value = mock_df
    mock_df.with_columns.return_value = mock_df
    # Mock select to return a DataFrame that has proper to_numpy
    mock_select_result = MagicMock()
    mock_select_result.to_numpy.return_value = rng.standard_normal(
        (
            150,
            15,
        ),
    )  # Match expected feature shape
    mock_df.select.return_value = mock_select_result

    # Stack OHLCV data in expected order: timestamp, open, high, low, close, volume
    timestamps = np.arange(n_samples, dtype=float)
    stacked_data = np.column_stack([timestamps, opens, highs, lows, closes, volumes])
    mock_df.to_numpy.return_value = stacked_data

    # Create a more sophisticated mock for column access
    def create_column_mock(data: Any) -> MagicMock:
        mock_col = MagicMock()
        mock_col.to_numpy.return_value = data

        # Mock shift operation
        shifted_data = np.concatenate([data[1:], [np.nan]])
        mock_shifted = MagicMock()
        mock_shifted.to_numpy.return_value = shifted_data
        mock_col.shift.return_value = mock_shifted

        # Mock comparison operation
        # Create comparison result based on actual data
        comparison = shifted_data > data

        # Create the comparison result mock
        mock_comparison = MagicMock()
        mock_comparison.to_numpy.return_value = comparison

        # Mock the cast operation to return a mock that handles slicing
        mock_casted = MagicMock()
        mock_casted.to_numpy.return_value = comparison.astype(np.int32)

        # Handle slicing of the casted result
        def handle_cast_slice(idx: Any) -> Any:
            if isinstance(idx, slice):
                sliced = comparison[idx].astype(np.int32)
                return MagicMock(to_numpy=MagicMock(return_value=sliced))
            return mock_casted

        mock_casted.__getitem__ = MagicMock(side_effect=handle_cast_slice)
        mock_comparison.cast.return_value = mock_casted

        # Set up comparison that works with MagicMock limitations
        # We'll mock the entire comparison expression result instead
        mock_shifted.__gt__ = MagicMock(return_value=mock_comparison)

        # Handle slicing - when indexed with slice, return mock with sliced data
        def handle_getitem(idx: Any) -> Any:
            if isinstance(idx, slice):
                sliced_data = data[idx]
                return MagicMock(to_numpy=MagicMock(return_value=sliced_data))
            else:
                return MagicMock(to_numpy=MagicMock(return_value=data))

        mock_col.__getitem__ = MagicMock(side_effect=handle_getitem)

        return mock_col

    # Mock column access with proper method chaining
    # Create column mocks that will be reused
    column_mocks = {}

    def get_column(col: Any) -> Any:
        if col not in column_mocks:
            if col == "close":
                column_mocks[col] = create_column_mock(closes)
            elif col == "target":
                # Create target column with binary values
                target_values = rng.integers(0, 2, n_samples)
                column_mocks[col] = MagicMock(to_numpy=MagicMock(return_value=target_values))
            else:
                # Return a basic mock for other columns
                column_mocks[col] = MagicMock(
                    to_numpy=MagicMock(return_value=rng.random(n_samples)),
                )
        return column_mocks[col]

    mock_df.__getitem__.side_effect = get_column
    return mock_df


@pytest.fixture
def sample_bar_data(mock_polars_dataframe: MagicMock) -> MagicMock:
    """
    Create sample OHLCV data for testing.
    """
    return mock_polars_dataframe


@pytest.fixture
def basic_config() -> XGBoostTrainingConfig:
    """
    Create basic XGBoost training configuration.
    """
    return XGBoostTrainingConfig(
        data_source="test_data",
        target_column="target",
        feature_config=MLFeatureConfig(
            lookback_window=50,
            normalize_features=True,
        ),
        n_estimators=10,  # Small for testing
        max_depth=3,
        learning_rate=0.1,
        early_stopping_rounds=5,
    )


@pytest.fixture
def multi_asset_config() -> XGBoostTrainingConfig:
    """
    Create multi-asset XGBoost configuration.
    """
    return XGBoostTrainingConfig(
        data_source="test_data",
        target_column="target",
        feature_config=MLFeatureConfig(
            lookback_window=50,
            normalize_features=True,
        ),
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        multi_asset=True,
        sector_map={"AAPL": "Technology", "MSFT": "Technology", "JPM": "Finance"},
        cross_sectional_features=True,
    )


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


class TestXGBoostTrainer:
    """
    Test XGBoost trainer implementation.
    """

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    def test_trainer_initialization(
        self,
        mock_pl: MagicMock,
        basic_config: XGBoostTrainingConfig,
    ) -> None:
        """
        Test trainer initialization.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(basic_config)

        assert trainer._xgb_config == basic_config
        assert trainer._is_multi_asset is False
        assert trainer._xgb is None  # Lazy import
        assert trainer._feature_engineer is not None

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler")
    @patch("ml.features.engineering.pl")
    def test_single_asset_data_preparation(
        self,
        mock_eng_pl: MagicMock,
        mock_scaler: MagicMock,
        mock_pl: MagicMock,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: MagicMock,
    ) -> None:
        """
        Test single asset data preparation.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock scaler instance
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit.return_value = mock_scaler_instance
        mock_scaler_instance.transform.return_value = rng.standard_normal(
            (
                150,
                15,
            ),
        )  # Match expected shape
        mock_scaler.return_value = mock_scaler_instance

        # Mock the calculate_features_batch method
        trainer = XGBoostTrainer(basic_config)

        # Create mock return values for calculate_features_batch
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal(
            (
                149,
                15,
            ),
        )  # 149 rows after removing last
        mock_features_df.__getitem__.return_value = mock_features_df
        mock_features_df.__len__.return_value = 150

        # Mock the feature engineer methods
        trainer._feature_engineer.calculate_features_batch = MagicMock(  # type: ignore[method-assign]
            return_value=(mock_features_df, mock_scaler_instance),
        )
        trainer._feature_engineer.get_feature_names = MagicMock(  # type: ignore[method-assign]
            return_value=[f"feature_{i}" for i in range(15)],
        )

        X, y, metadata = trainer.prepare_data(sample_bar_data)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        assert "feature_names" in metadata
        assert "n_features" in metadata
        assert "target_type" in metadata
        assert metadata["target_type"] == "classification"

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.pl")
    def test_single_asset_training(
        self,
        mock_eng_pl: MagicMock,
        mock_pl: MagicMock,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: MagicMock,
    ) -> None:
        """
        Test single asset model training.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock pl.Int32 for target series casting
        mock_pl.Int32 = MagicMock()

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        # Mock predict methods to return proper arrays
        mock_model.predict.return_value = rng.integers(0, 2, 149)
        mock_model.predict_proba.return_value = rng.random((149, 2))

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        # Mock the feature engineer methods
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal((149, 15))
        mock_features_df.__getitem__.return_value = mock_features_df
        mock_features_df.__len__.return_value = 150

        trainer._feature_engineer.calculate_features_batch = MagicMock(  # type: ignore[method-assign]
            return_value=(mock_features_df, None),
        )
        trainer._feature_engineer.get_feature_names = MagicMock(  # type: ignore[method-assign]
            return_value=[f"feature_{i}" for i in range(15)],
        )

        # Mock prepare_data to return successful results and avoid the comparison issue
        X_mock = rng.standard_normal((149, 15))
        y_mock = rng.integers(0, 2, 149)
        metadata_mock = {
            "feature_names": [f"feature_{i}" for i in range(15)],
            "target_type": "classification",
            "n_features": 15,
        }
        trainer.prepare_data = MagicMock(return_value=(X_mock, y_mock, metadata_mock))  # type: ignore[method-assign]

        # Train the model
        results = trainer.train(sample_bar_data)

        assert "model" in results
        assert "metrics" in results
        assert "feature_names" in results
        assert "config" in results

        # Check training metrics
        metrics = results["metrics"]
        assert "training_time" in metrics
        assert "best_iteration" in metrics
        assert "accuracy" in metrics  # From base trainer evaluation

        # Check model is fitted
        assert trainer._is_fitted is True
        assert trainer._model is not None

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl")
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.features.engineering.pl")
    def test_multi_asset_data_preparation(
        self,
        mock_eng_pl: MagicMock,
        mock_pl: MagicMock,
        multi_asset_config: XGBoostTrainingConfig,
    ) -> None:
        """
        Test multi-asset data preparation.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(multi_asset_config)

        # Mock the feature engineer methods for multi-asset
        mock_features_df = MagicMock()
        mock_features_df.select.return_value = mock_features_df
        mock_features_df.to_numpy.return_value = rng.standard_normal(
            (
                300,
                25,
            ),
        )  # More features for multi-asset
        mock_features_df.__getitem__.return_value = mock_features_df
        mock_features_df.__len__.return_value = 300

        trainer._feature_engineer.calculate_features_batch = MagicMock(  # type: ignore[method-assign]
            return_value=(mock_features_df, None),
        )
        trainer._feature_engineer.get_feature_names = MagicMock(  # type: ignore[method-assign]
            return_value=[f"feature_{i}" for i in range(25)],
        )

        # Create multi-asset data
        data_dict = {}
        for ticker in ["AAPL", "MSFT", "JPM"]:
            mock_df = MagicMock()
            mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "target"]
            mock_df.__len__.return_value = 150
            mock_df.with_columns.return_value = mock_df
            mock_df.select.return_value = mock_df
            mock_df.to_numpy.return_value = rng.standard_normal((100, 10))
            mock_df.__getitem__.return_value = MagicMock(
                to_numpy=lambda: rng.integers(0, 2, 100),
            )
            data_dict[ticker] = mock_df

        # Mock polars concat
        mock_pl.concat.return_value = MagicMock(
            columns=["timestamp", "open", "high", "low", "close", "volume", "ticker", "sector"],
            select=lambda cols: MagicMock(to_numpy=lambda: rng.standard_normal((300, len(cols)))),
            with_columns=lambda x: MagicMock(
                columns=["timestamp", "open", "high", "low", "close", "volume", "ticker", "sector"],
                select=lambda cols: MagicMock(
                    to_numpy=lambda: rng.standard_normal((300, len(cols))),
                ),
            ),
        )

        X, y, metadata = trainer.prepare_data(data_dict)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] > 0
        assert X.shape[1] > 0  # Should have cross-sectional features
        assert metadata["n_assets"] == 3
        assert "asset_metadata" in metadata

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_feature_importance_calculation(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: MagicMock,
    ) -> None:
        """
        Test feature importance calculation.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4, 0.15, 0.25, 0.05, 0.35])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        # Train model
        _ = trainer.train(sample_bar_data)

        # Get feature importance
        importance_summary = trainer.get_feature_importance_summary()

        assert "xgb_importance" in importance_summary
        assert "top_10_features" in importance_summary
        assert len(importance_summary["top_10_features"]) <= 10

        # Check that importance values are sorted descending
        xgb_importance = importance_summary["xgb_importance"]
        importance_values = list(xgb_importance.values())
        assert importance_values == sorted(importance_values, reverse=True)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_monotonic_constraints(self, sample_bar_data: MagicMock) -> None:
        """
        Test monotonic constraints functionality.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        config = XGBoostTrainingConfig(
            data_source="test",
            n_estimators=10,
            monotonic_constraints={"return_1": 1, "return_5": -1},
        )
        trainer = XGBoostTrainer(config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        # This should not raise an error
        results = trainer.train(sample_bar_data)
        assert results["model"] is not None

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_model_saving_and_loading(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: MagicMock,
        tmp_path: Any,
    ) -> None:
        """
        Test model saving and loading.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        # Train model
        _ = trainer.train(sample_bar_data)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        trainer.save_model(model_path)

        assert model_path.exists()

        # Load model
        new_trainer = XGBoostTrainer(basic_config)
        new_trainer.load_model(model_path)

        assert new_trainer._is_fitted is True
        assert new_trainer._model is not None
        assert new_trainer._feature_names == trainer._feature_names

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_target_creation_without_target_column(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: MagicMock,
    ) -> None:
        """
        Test automatic target creation when target column is missing.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(basic_config)

        # Remove any existing target column
        data_without_target = sample_bar_data.drop("target", strict=False)

        X, y, metadata = trainer.prepare_data(data_without_target)

        # Target should be created automatically (binary: up/down)
        assert len(np.unique(y)) <= 2
        assert np.all((y == 0) | (y == 1))

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_regression_objective(self, sample_bar_data: MagicMock) -> None:
        """
        Test regression objective.
        """
        import polars as pl

        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost regression model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])

        mock_xgb = MagicMock()
        mock_xgb.XGBRegressor.return_value = mock_model

        config = XGBoostTrainingConfig(
            data_source="test",
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=10,
        )
        trainer = XGBoostTrainer(config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        # Add continuous target
        rng = np.random.default_rng(42)
        data_with_target = sample_bar_data.with_columns(
            [
                pl.Series("target", rng.standard_normal(len(sample_bar_data))),
            ],
        )

        results = trainer.train(data_with_target)

        assert results["model"] is not None
        assert "rmse" in str(results["metrics"]["best_score"]) or isinstance(
            results["metrics"]["best_score"],
            float,
        )

    def test_gpu_configuration(self) -> None:
        """
        Test GPU configuration.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            tree_method="gpu_hist",
            gpu_id=1,
        )

        params = config.get_xgb_params()
        assert params["tree_method"] == "gpu_hist"
        assert params["gpu_id"] == 1
        assert params["predictor"] == "gpu_predictor"

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_cross_sectional_features(self, multi_asset_config: XGBoostTrainingConfig) -> None:
        """
        Test cross-sectional feature addition.
        """
        import polars as pl

        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(multi_asset_config)

        # Create test DataFrame with multiple assets
        df = pl.DataFrame(
            {
                "ticker": ["A", "B", "A", "B"],
                "sector": ["Tech", "Finance", "Tech", "Finance"],
                "timestamp": [1, 1, 2, 2],
                "return_5": [0.1, -0.05, 0.02, 0.08],
                "return_20": [0.15, -0.1, 0.05, 0.12],
                "rsi": [0.6, 0.4, 0.55, 0.65],
                "volume_ratio_5": [1.2, 0.8, 1.1, 1.3],
            },
        )

        df_with_features = trainer._add_cross_sectional_features(df)

        # Check that ranking features were added
        assert "return_5_rank" in df_with_features.columns
        assert "return_20_rank" in df_with_features.columns
        assert "rsi_rank" in df_with_features.columns

        # Check sector-relative features
        assert "return_5_sector_mean" in df_with_features.columns
        assert "return_5_sector_rel" in df_with_features.columns

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_nan_handling(self, basic_config: XGBoostTrainingConfig) -> None:
        """
        Test handling of NaN values in data.
        """
        import polars as pl

        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(basic_config)

        # Create data with NaN values
        data_with_nans = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 3, 1),
                    interval="1d",
                )[:60],
                "open": [100.0] * 60,
                "high": [102.0] * 58 + [None, None],  # NaN in last 2 values
                "low": [98.0] * 60,
                "close": [101.0] * 60,
                "volume": [1000.0] * 60,
            },
        )

        X, y, metadata = trainer.prepare_data(data_with_nans)

        # Should handle NaN values without crashing
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_insufficient_data_handling(self, basic_config: XGBoostTrainingConfig) -> None:
        """
        Test handling of insufficient data.
        """
        import polars as pl

        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(basic_config)

        # Create very small dataset
        small_data = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 10),
                    interval="1d",
                ),
                "open": [100.0] * 9,
                "high": [102.0] * 9,
                "low": [98.0] * 9,
                "close": [101.0] * 9,
                "volume": [1000.0] * 9,
            },
        )

        # Should handle small datasets gracefully
        X, y, metadata = trainer.prepare_data(small_data)
        assert X.shape[0] > 0
        assert y.shape[0] > 0

    def test_configuration_inheritance(self) -> None:
        """
        Test that XGBoost config properly inherits from base config.
        """
        config = XGBoostTrainingConfig(
            data_source="test",
            train_test_split=0.7,
            random_seed=123,
            n_estimators=50,
        )

        # Base config attributes
        assert config.train_test_split == 0.7
        assert config.random_seed == 123

        # XGBoost-specific attributes
        assert config.n_estimators == 50

        # XGBoost params should include inherited random seed
        params = config.get_xgb_params()
        assert params["random_state"] == 123
