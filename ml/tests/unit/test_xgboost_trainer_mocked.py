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
Unit tests for XGBoostTrainer with full mocking.

Tests the XGBoost trainer implementation including configuration validation, feature
preparation, model training, and various advanced features. All external dependencies
are mocked to ensure tests can run in any environment.

"""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.config.xgboost import XGBoostTrainingConfig
from ml.features.engineering import FeatureConfig
from ml.tests.unit.test_fixtures import mock_polars
from ml.tests.unit.test_fixtures import mock_sklearn


# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)

# Use MockPolarsModule from test_fixtures
MockPolarsModule = mock_polars


class TestXGBoostTrainer:
    """
    Test XGBoostTrainer functionality with mocks.
    """

    @pytest.fixture
    def basic_config(self) -> XGBoostTrainingConfig:
        """
        Create basic configuration for testing.
        """
        return XGBoostTrainingConfig(
            data_source="test_data",
            target_column="target",
            feature_config=FeatureConfig(
                lookback_window=20,
                return_periods=[1, 5],
                normalize_features=False,
            ),
            n_estimators=10,
            max_depth=3,
            early_stopping_rounds=5,
        )

    @pytest.fixture
    def multi_asset_config(self) -> XGBoostTrainingConfig:
        """
        Create multi-asset configuration for testing.
        """
        return XGBoostTrainingConfig(
            data_source="test_data",
            target_column="target",
            feature_config=FeatureConfig(
                lookback_window=20,
                return_periods=[1, 5],
                normalize_features=True,  # This will require sklearn mock
            ),
            n_estimators=10,
            max_depth=3,
            multi_asset=True,
            sector_map={
                "AAPL": "Technology",
                "MSFT": "Technology",
                "JPM": "Finance",
            },
            cross_sectional_features=True,
        )

    @pytest.fixture
    def sample_bar_data(self) -> Any:
        """
        Create sample bar data for testing.
        """
        n_samples = 200
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(n_samples) * 0.02)
        # Ensure prices are positive
        prices = np.abs(prices) + 50

        return MockPolarsModule.DataFrame(
            {
                "timestamp": list(range(n_samples)),
                "open": list(prices),
                "high": list(prices + rng.uniform(0, 2, n_samples)),
                "low": list(prices - rng.uniform(0, 2, n_samples)),
                "close": list(prices),
                "volume": list(rng.uniform(1000, 10000, n_samples)),  # Already positive
            },
        )

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_single_asset_data_preparation(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: Any,
    ) -> None:
        """
        Test single asset data preparation.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(basic_config)
        X, y, metadata = trainer.prepare_data(sample_bar_data)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0
        assert metadata["n_samples"] == X.shape[0]
        assert metadata["n_features"] == X.shape[1]

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_single_asset_training(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: Any,
    ) -> None:
        """
        Test single asset model training.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = rng.random(10)
        mock_model.predict.return_value = rng.integers(0, 2, 199)
        mock_model.predict_proba.return_value = rng.random((199, 2))

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        # Directly set the mocked xgb module instead of patching the import
        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]

        results = trainer.train(sample_bar_data)

        assert "metrics" in results
        assert "model" in results
        assert "feature_names" in results
        # Check that metrics exist (structure may vary based on whether train/val metrics are present)
        assert isinstance(results["metrics"], dict)

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_multi_asset_data_preparation(self, multi_asset_config: XGBoostTrainingConfig) -> None:
        """
        Test multi-asset data preparation.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(multi_asset_config)

        # Create multi-asset data
        rng = np.random.default_rng(42)
        data_dict = {}
        for ticker in ["AAPL", "MSFT", "JPM"]:
            n_samples = 150
            prices = 100 + np.cumsum(rng.standard_normal(n_samples) * 0.02)
            data_dict[ticker] = MockPolarsModule.DataFrame(
                {
                    "timestamp": list(range(n_samples)),
                    "open": list(prices),
                    "high": list(prices + rng.uniform(0, 2, n_samples)),
                    "low": list(prices - rng.uniform(0, 2, n_samples)),
                    "close": list(prices),
                    "volume": list(rng.uniform(1000, 10000, n_samples)),
                },
            )

        X, y, metadata = trainer.prepare_data(data_dict)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] > 0
        assert X.shape[1] > 0  # Should have cross-sectional features
        assert metadata["n_assets"] == 3
        assert "asset_metadata" in metadata

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_feature_importance_calculation(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: Any,
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
        # Create more feature importances to match expected number
        n_features = 10
        mock_model.feature_importances_ = rng.random(n_features)
        # Mock predict methods to return proper arrays
        mock_model.predict.return_value = rng.integers(0, 2, 199)
        mock_model.predict_proba.return_value = rng.random((199, 2))

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        # Directly set the mocked xgb module instead of patching the import
        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]
        trainer._feature_names = [f"feature_{i}" for i in range(n_features)]

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

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_model_saving_and_loading(
        self,
        basic_config: XGBoostTrainingConfig,
        sample_bar_data: Any,
        tmp_path: Any,
    ) -> None:
        """
        Test model saving and loading functionality.
        """
        from ml.training.xgboost import XGBoostTrainer

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.best_iteration = 5
        mock_model.best_score = 0.85
        mock_model.feature_importances_ = rng.random(10)
        mock_model.predict.return_value = rng.integers(0, 2, 199)
        mock_model.predict_proba.return_value = rng.random((199, 2))
        mock_model.save_model = MagicMock()

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model

        # Directly set the mocked xgb module instead of patching the import
        trainer = XGBoostTrainer(basic_config)
        trainer._xgb = mock_xgb  # type: ignore[assignment]
        _ = trainer.train(sample_bar_data)

        # Save model - mock pickle to avoid MagicMock serialization issues
        model_path = tmp_path / "test_model.json"
        with patch("pickle.dump") as mock_pickle:
            trainer.save_model(str(model_path))
            # Verify pickle.dump was called
            mock_pickle.assert_called_once()

        # Test loading - mock pickle.load too
        new_trainer = XGBoostTrainer(basic_config)
        new_trainer._xgb = mock_xgb  # type: ignore[assignment]
        with patch("pickle.load") as mock_load:
            mock_load.return_value = {
                "model": mock_model,
                "feature_names": ["test_feature"],
                "training_metrics": {"accuracy": 0.85},
                "scaler": None,
                "config": {},
            }
            new_trainer.load_model(str(model_path))
            mock_load.assert_called_once()

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_cross_sectional_features(self, multi_asset_config: XGBoostTrainingConfig) -> None:
        """
        Test cross-sectional feature generation.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(multi_asset_config)

        # Create a mock DataFrame with required columns
        n_samples = 100
        mock_df = MockPolarsModule.DataFrame(
            {
                "return_1": list(rng.standard_normal(n_samples)),
                "return_5": list(rng.standard_normal(n_samples)),
                "momentum_5": list(rng.standard_normal(n_samples)),
                "ticker": ["AAPL"] * 40 + ["MSFT"] * 30 + ["JPM"] * 30,
                "sector": ["Technology"] * 70 + ["Finance"] * 30,
                "timestamp": list(range(n_samples)),
            },
        )

        # Test _add_cross_sectional_features method
        result_df = trainer._add_cross_sectional_features(mock_df)

        # Should have added cross-sectional features
        assert result_df is not None

    @patch("ml.training.base.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_POLARS", True)
    @patch("ml.training.xgboost.HAS_SKLEARN", True)
    @patch("ml.training.xgboost.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    @patch("ml.training.xgboost.pl", mock_polars)
    @patch("ml.features.engineering.pl", mock_polars)
    @patch("ml.features.engineering.POLARS_AVAILABLE", True)
    @patch("ml.features.engineering.SKLEARN_AVAILABLE", True)
    @patch("ml.features.engineering.StandardScaler", mock_sklearn.preprocessing.StandardScaler)
    def test_multi_asset_with_sklearn(self, multi_asset_config: XGBoostTrainingConfig) -> None:
        """
        Test multi-asset training with sklearn scaling.
        """
        from ml.training.xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(multi_asset_config)

        # Create multi-asset data
        rng = np.random.default_rng(42)
        data_dict = {}
        for ticker in ["AAPL", "MSFT"]:
            n_samples = 100
            prices = 100 + np.cumsum(rng.standard_normal(n_samples) * 0.02)
            data_dict[ticker] = MockPolarsModule.DataFrame(
                {
                    "timestamp": list(range(n_samples)),
                    "open": list(prices),
                    "high": list(prices + rng.uniform(0, 2, n_samples)),
                    "low": list(prices - rng.uniform(0, 2, n_samples)),
                    "close": list(prices),
                    "volume": list(rng.uniform(1000, 10000, n_samples)),
                },
            )

        # Should work with sklearn mocked
        X, y, metadata = trainer.prepare_data(data_dict)
        assert X.shape[0] > 0
