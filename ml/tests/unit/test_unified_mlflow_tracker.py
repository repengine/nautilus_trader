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
Unit tests for unified MLflow tracker.

This test suite provides coverage for the unified MLflowTracker that supports multiple
ML frameworks through a single interface.

"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.config.shared import MLflowConfig
from ml.training.mlflow_tracker import MLflowTracker


class TestUnifiedMLflowTracker:
    """
    Test unified MLflow tracker functionality.
    """

    @pytest.fixture
    def basic_config(self):
        """
        Create basic MLflow configuration.
        """
        return MLflowConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            model_name="test_model",
        )

    def test_tracker_initialization_with_auto(self, basic_config) -> None:
        """
        Test tracker initialization with auto framework detection.
        """
        tracker = MLflowTracker(basic_config, framework="auto")

        assert tracker.config == basic_config
        assert tracker.framework == "auto"
        assert tracker._mlflow is None
        assert tracker._client is None

    def test_tracker_initialization_with_xgboost(self, basic_config) -> None:
        """
        Test tracker initialization with explicit XGBoost framework.
        """
        tracker = MLflowTracker(basic_config, framework="xgboost")

        assert tracker.config == basic_config
        assert tracker.framework == "xgboost"

    def test_tracker_initialization_with_lightgbm(self, basic_config) -> None:
        """
        Test tracker initialization with explicit LightGBM framework.
        """
        tracker = MLflowTracker(basic_config, framework="lightgbm")

        assert tracker.config == basic_config
        assert tracker.framework == "lightgbm"

    def test_framework_detection_xgboost(self, basic_config) -> None:
        """
        Test automatic framework detection for XGBoost models.
        """
        tracker = MLflowTracker(basic_config, framework="auto")

        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        mock_model.__class__.__module__ = "xgboost.sklearn"

        detected = tracker._detect_framework(mock_model)
        assert detected == "xgboost"

    def test_framework_detection_lightgbm(self, basic_config) -> None:
        """
        Test automatic framework detection for LightGBM models.
        """
        tracker = MLflowTracker(basic_config, framework="auto")

        # Mock LightGBM model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LGBMClassifier"
        mock_model.__class__.__module__ = "lightgbm.sklearn"

        detected = tracker._detect_framework(mock_model)
        assert detected == "lightgbm"

    def test_framework_detection_sklearn(self, basic_config) -> None:
        """
        Test automatic framework detection for scikit-learn models.
        """
        tracker = MLflowTracker(basic_config, framework="auto")

        # Mock sklearn model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "RandomForestClassifier"
        mock_model.__class__.__module__ = "sklearn.ensemble"

        detected = tracker._detect_framework(mock_model)
        assert detected == "sklearn"

    @patch("ml.training.mlflow_tracker.HAS_XGBOOST", True)
    @patch("ml.training.mlflow_tracker.HAS_MLFLOW", True)
    @patch("ml.training.mlflow_tracker.mlflow")
    def test_get_mlflow_module_xgboost(self, mock_mlflow, basic_config) -> None:
        """
        Test getting XGBoost MLflow module.
        """
        tracker = MLflowTracker(basic_config, framework="xgboost")
        tracker._mlflow = mock_mlflow

        module = tracker._get_mlflow_module("xgboost")
        assert module == mock_mlflow.xgboost

    @patch("ml.training.mlflow_tracker.HAS_LIGHTGBM", True)
    @patch("ml.training.mlflow_tracker.HAS_MLFLOW", True)
    @patch("ml.training.mlflow_tracker.mlflow")
    def test_get_mlflow_module_lightgbm(self, mock_mlflow, basic_config) -> None:
        """
        Test getting LightGBM MLflow module.
        """
        tracker = MLflowTracker(basic_config, framework="lightgbm")
        tracker._mlflow = mock_mlflow

        module = tracker._get_mlflow_module("lightgbm")
        assert module == mock_mlflow.lightgbm

    @patch("ml.training.mlflow_tracker.HAS_MLFLOW", True)
    @patch("ml.training.mlflow_tracker.mlflow")
    def test_get_mlflow_module_sklearn(self, mock_mlflow, basic_config) -> None:
        """
        Test getting scikit-learn MLflow module.
        """
        tracker = MLflowTracker(basic_config, framework="sklearn")
        tracker._mlflow = mock_mlflow

        module = tracker._get_mlflow_module("sklearn")
        assert module == mock_mlflow.sklearn

    def test_backward_compatibility_xgboost(self, basic_config) -> None:
        """
        Test backward compatibility with MLflowXGBoostTracker.
        """
        from ml.training.mlflow_tracker import MLflowXGBoostTracker

        # Should create tracker with xgboost framework
        with patch("builtins.print") as mock_print:
            tracker = MLflowXGBoostTracker(basic_config)
            mock_print.assert_called_with(
                "DeprecationWarning: MLflowXGBoostTracker is deprecated. "
                "Use MLflowTracker(config, framework='xgboost') instead.",
            )

        assert isinstance(tracker, MLflowTracker)
        assert tracker.framework == "xgboost"

    def test_backward_compatibility_lightgbm(self, basic_config) -> None:
        """
        Test backward compatibility with MLflowLightGBMTracker.
        """
        from ml.training.mlflow_tracker import MLflowLightGBMTracker

        # Should create tracker with lightgbm framework
        with patch("builtins.print") as mock_print:
            tracker = MLflowLightGBMTracker(basic_config)
            mock_print.assert_called_with(
                "DeprecationWarning: MLflowLightGBMTracker is deprecated. "
                "Use MLflowTracker(config, framework='lightgbm') instead.",
            )

        assert isinstance(tracker, MLflowTracker)
        assert tracker.framework == "lightgbm"
