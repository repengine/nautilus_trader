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
Test common configuration classes.
"""

import pytest

from ml.config.shared import MLflowConfig
from ml.config.shared import OptunaConfig


class TestOptunaConfig:
    """
    Test Optuna configuration validation and defaults.
    """

    def test_default_config(self) -> None:
        """
        Test default Optuna configuration values.
        """
        config = OptunaConfig()

        assert config.enabled is False
        assert config.n_trials == 100
        assert config.direction == "maximize"
        assert config.metric == "sharpe_ratio"
        assert config.pruner == "median"
        assert config.sampler == "tpe"
        assert config.timeout is None
        assert config.study_name is None
        assert config.storage_url is None

    def test_custom_config(self) -> None:
        """
        Test custom Optuna configuration.
        """
        config = OptunaConfig(
            enabled=True,
            n_trials=200,
            direction="minimize",
            metric="rmse",
            pruner="hyperband",
            sampler="random",
            timeout=3600,
            study_name="test_study",
            storage_url="sqlite:///optuna.db",
        )

        assert config.enabled is True
        assert config.n_trials == 200
        assert config.direction == "minimize"
        assert config.metric == "rmse"
        assert config.pruner == "hyperband"
        assert config.sampler == "random"
        assert config.timeout == 3600
        assert config.study_name == "test_study"
        assert config.storage_url == "sqlite:///optuna.db"

    def test_invalid_n_trials(self) -> None:
        """
        Test invalid n_trials raises ValueError.
        """
        with pytest.raises(ValueError, match="n_trials must be positive"):
            OptunaConfig(n_trials=0)

    def test_invalid_direction(self) -> None:
        """
        Test invalid direction raises ValueError.
        """
        with pytest.raises(ValueError, match="direction must be one of"):
            OptunaConfig(direction="invalid")

    def test_invalid_metric(self) -> None:
        """
        Test invalid metric raises ValueError.
        """
        with pytest.raises(ValueError, match="metric must be one of"):
            OptunaConfig(metric="invalid")

    def test_invalid_pruner(self) -> None:
        """
        Test invalid pruner raises ValueError.
        """
        with pytest.raises(ValueError, match="pruner must be one of"):
            OptunaConfig(pruner="invalid")

    def test_invalid_sampler(self) -> None:
        """
        Test invalid sampler raises ValueError.
        """
        with pytest.raises(ValueError, match="sampler must be one of"):
            OptunaConfig(sampler="invalid")

    def test_invalid_timeout(self) -> None:
        """
        Test invalid timeout raises ValueError.
        """
        with pytest.raises(ValueError, match="timeout must be positive or None"):
            OptunaConfig(timeout=-1)


class TestMLflowConfig:
    """
    Test MLflow configuration validation and defaults.
    """

    def test_default_config(self) -> None:
        """
        Test default MLflow configuration values.
        """
        config = MLflowConfig()

        assert config.enabled is False
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "ml_experiment"
        assert config.register_model is True
        assert config.model_name == "ml_model"
        assert config.log_artifacts is True
        assert config.log_model is True
        assert config.auto_log is False

    def test_custom_config(self) -> None:
        """
        Test custom MLflow configuration.
        """
        config = MLflowConfig(
            enabled=True,
            tracking_uri="https://mlflow.example.com",
            experiment_name="custom_experiment",
            register_model=False,
            model_name="custom_model",
            log_artifacts=False,
            log_model=False,
            auto_log=True,
        )

        assert config.enabled is True
        assert config.tracking_uri == "https://mlflow.example.com"
        assert config.experiment_name == "custom_experiment"
        assert config.register_model is False
        assert config.model_name == "custom_model"
        assert config.log_artifacts is False
        assert config.log_model is False
        assert config.auto_log is True

    def test_empty_experiment_name(self) -> None:
        """
        Test empty experiment name raises ValueError.
        """
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            MLflowConfig(experiment_name="")

    def test_empty_model_name_with_register(self) -> None:
        """
        Test empty model name with register_model=True raises ValueError.
        """
        with pytest.raises(
            ValueError,
            match="model_name cannot be empty when register_model is True",
        ):
            MLflowConfig(register_model=True, model_name="")

    def test_empty_model_name_without_register(self) -> None:
        """
        Test empty model name is allowed when register_model=False.
        """
        config = MLflowConfig(register_model=False, model_name="")
        assert config.model_name == ""
        assert config.register_model is False

    def test_valid_tracking_uris(self) -> None:
        """
        Test various valid tracking URI formats.
        """
        valid_uris = [
            "http://localhost:5000",
            "https://mlflow.example.com",
            "file:///tmp/mlflow",
            "sqlite:///mlflow.db",
        ]

        for uri in valid_uris:
            config = MLflowConfig(tracking_uri=uri)
            assert config.tracking_uri == uri

    def test_invalid_tracking_uri(self) -> None:
        """
        Test invalid tracking URI raises ValueError.
        """
        with pytest.raises(ValueError, match="tracking_uri must start with"):
            MLflowConfig(tracking_uri="invalid://uri")


class TestIntegration:
    """
    Test integration with unified configurations.
    """

    def test_common_configs_in_lightgbm(self) -> None:
        """
        Test common configs work with LightGBM unified config.
        """
        from ml.config.lightgbm import UnifiedLightGBMConfig

        optuna_config = OptunaConfig(enabled=True, n_trials=50)
        mlflow_config = MLflowConfig(
            enabled=True,
            experiment_name="lightgbm_test",
            model_name="lgb_model",
        )

        config = UnifiedLightGBMConfig(
            data_source="test_data.parquet",
            optuna_config=optuna_config,
            mlflow_config=mlflow_config,
        )

        assert config.optuna_config.enabled is True
        assert config.optuna_config.n_trials == 50
        assert config.mlflow_config.experiment_name == "lightgbm_test"
        assert config.mlflow_config.model_name == "lgb_model"

    def test_common_configs_in_xgboost(self) -> None:
        """
        Test common configs work with XGBoost unified config.
        """
        from ml.config.xgboost import UnifiedXGBoostConfig

        optuna_config = OptunaConfig(enabled=True, n_trials=100)
        mlflow_config = MLflowConfig(
            enabled=True,
            experiment_name="xgboost_test",
            model_name="xgb_model",
        )

        config = UnifiedXGBoostConfig(
            data_source="test_data.parquet",
            optuna_config=optuna_config,
            mlflow_config=mlflow_config,
        )

        assert config.optuna_config.enabled is True
        assert config.optuna_config.n_trials == 100
        assert config.mlflow_config.experiment_name == "xgboost_test"
        assert config.mlflow_config.model_name == "xgb_model"

    def test_default_overrides_in_lightgbm(self) -> None:
        """
        Test that LightGBM config overrides default MLflow values.
        """
        from ml.config.lightgbm import UnifiedLightGBMConfig

        config = UnifiedLightGBMConfig(data_source="test.parquet")

        # Should have LightGBM-specific defaults
        assert config.mlflow_config.experiment_name == "lightgbm_unified"
        assert config.mlflow_config.model_name == "lightgbm_unified"

    def test_default_overrides_in_xgboost(self) -> None:
        """
        Test that XGBoost config overrides default MLflow values.
        """
        from ml.config.xgboost import UnifiedXGBoostConfig

        config = UnifiedXGBoostConfig(data_source="test.parquet")

        # Should have XGBoost-specific defaults
        assert config.mlflow_config.experiment_name == "xgboost_unified"
        assert config.mlflow_config.model_name == "xgboost_unified"
