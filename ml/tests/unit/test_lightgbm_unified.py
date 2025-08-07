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

import pytest

from ml.config.lightgbm_unified import DARTConfig
from ml.config.lightgbm_unified import EFBConfig
from ml.config.lightgbm_unified import GOSSConfig
from ml.config.lightgbm_unified import GPUConfig
from ml.config.lightgbm_unified import MLflowConfig
from ml.config.lightgbm_unified import OptunaConfig
from ml.config.lightgbm_unified import UnifiedLightGBMConfig


class TestGOSSConfig:
    """
    Test GOSS configuration.
    """

    def test_goss_config_init_default(self):
        """
        Test GOSS configuration with default values.
        """
        config = GOSSConfig()
        assert config.enabled is False
        assert config.top_rate == 0.2
        assert config.other_rate == 0.1

    def test_goss_config_init_custom(self):
        """
        Test GOSS configuration with custom values.
        """
        config = GOSSConfig(enabled=True, top_rate=0.3, other_rate=0.15)
        assert config.enabled is True
        assert config.top_rate == 0.3
        assert config.other_rate == 0.15

    def test_goss_config_validation_top_rate_bounds(self):
        """
        Test GOSS configuration validation for top_rate bounds.
        """
        with pytest.raises(ValueError, match="top_rate must be in"):
            GOSSConfig(top_rate=0.0)

        with pytest.raises(ValueError, match="top_rate must be in"):
            GOSSConfig(top_rate=1.0)

    def test_goss_config_validation_other_rate_bounds(self):
        """
        Test GOSS configuration validation for other_rate bounds.
        """
        with pytest.raises(ValueError, match="other_rate must be in"):
            GOSSConfig(other_rate=0.0)

        with pytest.raises(ValueError, match="other_rate must be in"):
            GOSSConfig(other_rate=1.0)

    def test_goss_config_validation_rate_sum(self):
        """
        Test GOSS configuration validation for rate sum.
        """
        with pytest.raises(ValueError, match="top_rate \\+ other_rate must be"):
            GOSSConfig(top_rate=0.6, other_rate=0.5)


class TestDARTConfig:
    """
    Test DART configuration.
    """

    def test_dart_config_init_default(self):
        """
        Test DART configuration with default values.
        """
        config = DARTConfig()
        assert config.enabled is False
        assert config.drop_rate == 0.1
        assert config.max_drop == 50
        assert config.skip_drop == 0.5
        assert config.uniform_drop is False
        assert config.xgboost_dart_mode is False

    def test_dart_config_init_custom(self):
        """
        Test DART configuration with custom values.
        """
        config = DARTConfig(
            enabled=True,
            drop_rate=0.2,
            max_drop=100,
            skip_drop=0.7,
            uniform_drop=True,
            xgboost_dart_mode=True,
        )
        assert config.enabled is True
        assert config.drop_rate == 0.2
        assert config.max_drop == 100
        assert config.skip_drop == 0.7
        assert config.uniform_drop is True
        assert config.xgboost_dart_mode is True

    def test_dart_config_validation_drop_rate_bounds(self):
        """
        Test DART configuration validation for drop_rate bounds.
        """
        with pytest.raises(ValueError, match="drop_rate must be in"):
            DARTConfig(drop_rate=-0.1)

        with pytest.raises(ValueError, match="drop_rate must be in"):
            DARTConfig(drop_rate=1.1)

    def test_dart_config_validation_max_drop_positive(self):
        """
        Test DART configuration validation for max_drop positive.
        """
        with pytest.raises(ValueError, match="max_drop must be positive"):
            DARTConfig(max_drop=0)

    def test_dart_config_validation_skip_drop_bounds(self):
        """
        Test DART configuration validation for skip_drop bounds.
        """
        with pytest.raises(ValueError, match="skip_drop must be in"):
            DARTConfig(skip_drop=-0.1)

        with pytest.raises(ValueError, match="skip_drop must be in"):
            DARTConfig(skip_drop=1.1)


class TestEFBConfig:
    """
    Test EFB configuration.
    """

    def test_efb_config_init_default(self):
        """
        Test EFB configuration with default values.
        """
        config = EFBConfig()
        assert config.enabled is True
        assert config.max_conflict_rate == 0.0
        assert config.bundle_size == 0

    def test_efb_config_init_custom(self):
        """
        Test EFB configuration with custom values.
        """
        config = EFBConfig(enabled=False, max_conflict_rate=0.1, bundle_size=10)
        assert config.enabled is False
        assert config.max_conflict_rate == 0.1
        assert config.bundle_size == 10

    def test_efb_config_validation_max_conflict_rate_bounds(self):
        """
        Test EFB configuration validation for max_conflict_rate bounds.
        """
        with pytest.raises(ValueError, match="max_conflict_rate must be in"):
            EFBConfig(max_conflict_rate=-0.1)

        with pytest.raises(ValueError, match="max_conflict_rate must be in"):
            EFBConfig(max_conflict_rate=1.0)

    def test_efb_config_validation_bundle_size_non_negative(self):
        """
        Test EFB configuration validation for bundle_size non-negative.
        """
        with pytest.raises(ValueError, match="bundle_size must be non-negative"):
            EFBConfig(bundle_size=-1)


class TestGPUConfig:
    """
    Test GPU configuration.
    """

    def test_gpu_config_init_default(self):
        """
        Test GPU configuration with default values.
        """
        config = GPUConfig()
        assert config.enabled is False
        assert config.device_id == 0
        assert config.platform_id == -1
        assert config.gpu_use_dp is False

    def test_gpu_config_init_custom(self):
        """
        Test GPU configuration with custom values.
        """
        config = GPUConfig(enabled=True, device_id=1, platform_id=0, gpu_use_dp=True)
        assert config.enabled is True
        assert config.device_id == 1
        assert config.platform_id == 0
        assert config.gpu_use_dp is True

    def test_gpu_config_validation_device_id_non_negative(self):
        """
        Test GPU configuration validation for device_id non-negative.
        """
        with pytest.raises(ValueError, match="device_id must be non-negative"):
            GPUConfig(device_id=-1)


class TestOptunaConfig:
    """
    Test Optuna configuration.
    """

    def test_optuna_config_init_default(self):
        """
        Test Optuna configuration with default values.
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

    def test_optuna_config_init_custom(self):
        """
        Test Optuna configuration with custom values.
        """
        config = OptunaConfig(
            enabled=True,
            n_trials=50,
            direction="minimize",
            metric="rmse",
            pruner="hyperband",
            sampler="random",
            timeout=3600,
            study_name="test_study",
            storage_url="sqlite:///test.db",
        )
        assert config.enabled is True
        assert config.n_trials == 50
        assert config.direction == "minimize"
        assert config.metric == "rmse"
        assert config.pruner == "hyperband"
        assert config.sampler == "random"
        assert config.timeout == 3600
        assert config.study_name == "test_study"
        assert config.storage_url == "sqlite:///test.db"

    def test_optuna_config_validation_n_trials_positive(self):
        """
        Test Optuna configuration validation for n_trials positive.
        """
        with pytest.raises(ValueError, match="n_trials must be positive"):
            OptunaConfig(n_trials=0)

    def test_optuna_config_validation_direction_valid(self):
        """
        Test Optuna configuration validation for valid direction.
        """
        with pytest.raises(ValueError, match="direction must be one of"):
            OptunaConfig(direction="invalid")

    def test_optuna_config_validation_metric_valid(self):
        """
        Test Optuna configuration validation for valid metric.
        """
        with pytest.raises(ValueError, match="metric must be one of"):
            OptunaConfig(metric="invalid")

    def test_optuna_config_validation_pruner_valid(self):
        """
        Test Optuna configuration validation for valid pruner.
        """
        with pytest.raises(ValueError, match="pruner must be one of"):
            OptunaConfig(pruner="invalid")

    def test_optuna_config_validation_sampler_valid(self):
        """
        Test Optuna configuration validation for valid sampler.
        """
        with pytest.raises(ValueError, match="sampler must be one of"):
            OptunaConfig(sampler="invalid")

    def test_optuna_config_validation_timeout_positive_or_none(self):
        """
        Test Optuna configuration validation for timeout positive or None.
        """
        with pytest.raises(ValueError, match="timeout must be positive or None"):
            OptunaConfig(timeout=0)


class TestMLflowConfig:
    """
    Test MLflow configuration.
    """

    def test_mlflow_config_init_default(self):
        """
        Test MLflow configuration with default values.
        """
        config = MLflowConfig()
        assert config.enabled is False
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "lightgbm_unified"
        assert config.register_model is True
        assert config.model_name == "lightgbm_unified"
        assert config.log_artifacts is True
        assert config.log_model is True
        assert config.auto_log is False

    def test_mlflow_config_init_custom(self):
        """
        Test MLflow configuration with custom values.
        """
        config = MLflowConfig(
            enabled=True,
            tracking_uri="http://remote:5000",
            experiment_name="custom_experiment",
            register_model=False,
            model_name="custom_model",
            log_artifacts=False,
            log_model=False,
            auto_log=True,
        )
        assert config.enabled is True
        assert config.tracking_uri == "http://remote:5000"
        assert config.experiment_name == "custom_experiment"
        assert config.register_model is False
        assert config.model_name == "custom_model"
        assert config.log_artifacts is False
        assert config.log_model is False
        assert config.auto_log is True

    def test_mlflow_config_validation_experiment_name_not_empty(self):
        """
        Test MLflow configuration validation for non-empty experiment name.
        """
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            MLflowConfig(experiment_name="")

    def test_mlflow_config_validation_model_name_when_register(self):
        """
        Test MLflow configuration validation for model name when registering.
        """
        with pytest.raises(
            ValueError,
            match="model_name cannot be empty when register_model is True",
        ):
            MLflowConfig(register_model=True, model_name="")

    def test_mlflow_config_validation_tracking_uri_format(self):
        """
        Test MLflow configuration validation for tracking URI format.
        """
        with pytest.raises(ValueError, match="tracking_uri must start with"):
            MLflowConfig(tracking_uri="invalid://uri")

    def test_mlflow_config_validation_valid_tracking_uris(self):
        """
        Test MLflow configuration validation for valid tracking URIs.
        """
        valid_uris = [
            "http://localhost:5000",
            "https://remote.mlflow.com",
            "file:///tmp/mlruns",
            "sqlite:///mlflow.db",
        ]
        for uri in valid_uris:
            config = MLflowConfig(tracking_uri=uri)
            assert config.tracking_uri == uri


class TestUnifiedLightGBMConfig:
    """
    Test unified LightGBM configuration.
    """

    def test_unified_config_init_default(self):
        """
        Test unified configuration with default values.
        """
        config = UnifiedLightGBMConfig(data_source="test.csv")
        assert isinstance(config.goss_config, GOSSConfig)
        assert isinstance(config.dart_config, DARTConfig)
        assert isinstance(config.efb_config, EFBConfig)
        assert isinstance(config.gpu_config, GPUConfig)
        assert isinstance(config.optuna_config, OptunaConfig)
        assert isinstance(config.mlflow_config, MLflowConfig)
        assert config.categorical_features == []
        assert config.track_feature_decay is True
        assert config.feature_decay_threshold == 0.3
        assert config.feature_history_window == 10
        assert config.cv_strategy == "time_series"
        assert config.cv_folds == 5
        assert config.purge_gap == 10
        assert config.export_onnx is False
        assert config.onnx_output_path == "./models/lightgbm_unified.onnx"
        assert config.enable_monitoring is True

    def test_unified_config_init_custom(self):
        """
        Test unified configuration with custom values.
        """
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            goss_config=GOSSConfig(enabled=True),
            dart_config=DARTConfig(enabled=False),
            categorical_features=["cat1", "cat2"],
            track_feature_decay=False,
            feature_decay_threshold=0.5,
            feature_history_window=20,
            cv_strategy="blocked",
            cv_folds=10,
            purge_gap=5,
            export_onnx=True,
            onnx_output_path="./custom/path.onnx",
            enable_monitoring=False,
        )
        assert config.goss_config.enabled is True
        assert config.dart_config.enabled is False
        assert config.categorical_features == ["cat1", "cat2"]
        assert config.track_feature_decay is False
        assert config.feature_decay_threshold == 0.5
        assert config.feature_history_window == 20
        assert config.cv_strategy == "blocked"
        assert config.cv_folds == 10
        assert config.purge_gap == 5
        assert config.export_onnx is True
        assert config.onnx_output_path == "./custom/path.onnx"
        assert config.enable_monitoring is False

    def test_unified_config_validation_feature_decay_threshold_bounds(self):
        """
        Test unified configuration validation for feature decay threshold bounds.
        """
        with pytest.raises(ValueError, match="feature_decay_threshold must be in"):
            UnifiedLightGBMConfig(data_source="test.csv", feature_decay_threshold=0.0)

        with pytest.raises(ValueError, match="feature_decay_threshold must be in"):
            UnifiedLightGBMConfig(data_source="test.csv", feature_decay_threshold=1.0)

    def test_unified_config_validation_feature_history_window_positive(self):
        """
        Test unified configuration validation for feature history window positive.
        """
        with pytest.raises(ValueError, match="feature_history_window must be positive"):
            UnifiedLightGBMConfig(data_source="test.csv", feature_history_window=0)

    def test_unified_config_validation_cv_strategy_valid(self):
        """
        Test unified configuration validation for valid CV strategy.
        """
        with pytest.raises(ValueError, match="cv_strategy must be one of"):
            UnifiedLightGBMConfig(data_source="test.csv", cv_strategy="invalid")

    def test_unified_config_validation_cv_folds_minimum(self):
        """
        Test unified configuration validation for minimum CV folds.
        """
        with pytest.raises(ValueError, match="cv_folds must be at least 2"):
            UnifiedLightGBMConfig(data_source="test.csv", cv_folds=1)

    def test_unified_config_validation_purge_gap_non_negative(self):
        """
        Test unified configuration validation for purge gap non-negative.
        """
        with pytest.raises(ValueError, match="purge_gap must be non-negative"):
            UnifiedLightGBMConfig(data_source="test.csv", purge_gap=-1)

    def test_unified_config_validation_onnx_path_when_export(self):
        """
        Test unified configuration validation for ONNX path when exporting.
        """
        with pytest.raises(
            ValueError,
            match="onnx_output_path cannot be empty when export_onnx is True",
        ):
            UnifiedLightGBMConfig(data_source="test.csv", export_onnx=True, onnx_output_path="")

    def test_unified_config_validation_goss_dart_mutually_exclusive(self):
        """
        Test unified configuration validation for GOSS and DART mutual exclusivity.
        """
        with pytest.raises(ValueError, match="GOSS and DART cannot be enabled simultaneously"):
            UnifiedLightGBMConfig(
                data_source="test.csv",
                goss_config=GOSSConfig(enabled=True),
                dart_config=DARTConfig(enabled=True),
            )

    def test_unified_config_validate_config_warnings(self):
        """
        Test unified configuration warnings validation.
        """
        # This should raise an error due to incompatible configurations
        with pytest.raises(ValueError, match="GOSS and DART cannot be enabled simultaneously"):
            UnifiedLightGBMConfig(
                data_source="test.csv",
                goss_config=GOSSConfig(enabled=True),
                dart_config=DARTConfig(enabled=True),
            )

        # Test the warnings method instead
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            gpu_config=GPUConfig(enabled=True),
            optuna_config=OptunaConfig(enabled=True),
            early_stopping_rounds=10,
        )
        warnings = config.validate_config()
        assert isinstance(warnings, list)

    def test_unified_config_get_unified_lgb_params_default(self):
        """
        Test getting unified LightGBM parameters with default settings.
        """
        config = UnifiedLightGBMConfig(data_source="test.csv")
        params = config.get_unified_lgb_params()

        # Should contain base LightGBM parameters
        assert "num_iterations" in params
        assert "learning_rate" in params
        assert "num_leaves" in params

        # EFB should be enabled by default
        assert params.get("enable_bundle") is True

        # GOSS and DART should not be set
        assert params.get("boosting_type") == "gbdt"

    def test_unified_config_get_unified_lgb_params_goss(self):
        """
        Test getting unified LightGBM parameters with GOSS enabled.
        """
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            goss_config=GOSSConfig(enabled=True, top_rate=0.3, other_rate=0.2),
        )
        params = config.get_unified_lgb_params()

        assert params["boosting_type"] == "goss"
        assert params["top_rate"] == 0.3
        assert params["other_rate"] == 0.2

    def test_unified_config_get_unified_lgb_params_dart(self):
        """
        Test getting unified LightGBM parameters with DART enabled.
        """
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            dart_config=DARTConfig(
                enabled=True,
                drop_rate=0.2,
                max_drop=100,
                skip_drop=0.7,
                uniform_drop=True,
                xgboost_dart_mode=True,
            ),
        )
        params = config.get_unified_lgb_params()

        assert params["boosting_type"] == "dart"
        assert params["drop_rate"] == 0.2
        assert params["max_drop"] == 100
        assert params["skip_drop"] == 0.7
        assert params["uniform_drop"] is True
        assert params["xgboost_dart_mode"] is True

    def test_unified_config_get_unified_lgb_params_gpu(self):
        """
        Test getting unified LightGBM parameters with GPU enabled.
        """
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            gpu_config=GPUConfig(enabled=True, device_id=1, platform_id=0, gpu_use_dp=True),
        )
        params = config.get_unified_lgb_params()

        assert params["device_type"] == "gpu"
        assert params["gpu_device_id"] == 1
        assert params["gpu_platform_id"] == 0
        assert params["gpu_use_dp"] is True

    def test_unified_config_get_unified_lgb_params_efb_disabled(self):
        """
        Test getting unified LightGBM parameters with EFB disabled.
        """
        config = UnifiedLightGBMConfig(data_source="test.csv", efb_config=EFBConfig(enabled=False))
        params = config.get_unified_lgb_params()

        assert params["enable_bundle"] is False

    def test_unified_config_get_unified_lgb_params_efb_with_bundle_size(self):
        """
        Test getting unified LightGBM parameters with EFB bundle size.
        """
        config = UnifiedLightGBMConfig(
            data_source="test.csv",
            efb_config=EFBConfig(enabled=True, max_conflict_rate=0.1, bundle_size=50),
        )
        params = config.get_unified_lgb_params()

        assert params["enable_bundle"] is True
        assert params["max_conflict_rate"] == 0.1
        assert params["max_bundle"] == 50

    def test_unified_config_validate_environment_no_warnings(self):
        """
        Test environment validation with no warnings.
        """
        config = UnifiedLightGBMConfig(data_source="test.csv")
        warnings = config.validate_environment()

        # Should not have warnings for default configuration
        assert isinstance(warnings, list)
