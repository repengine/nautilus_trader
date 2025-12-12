from __future__ import annotations

from collections.abc import Mapping

import pytest
from msgspec import ValidationError

from nautilus_trader.model.identifiers import InstrumentId

from ml.config.lightgbm import LightGBMTrainingConfig
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import LightGBMGPUConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from ml.config.base import HealthMonitorConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLInferenceConfig
from ml.config.base import MLStrategyConfig
from ml.config.runtime import OnnxRuntimeConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.tests.utils.db import build_postgres_url


def _mapping(env: dict[str, str]) -> Mapping[str, str]:
    """
    Helper to create an immutable mapping for env-style lookups.
    """
    return env


def test_ml_inference_config_from_env_with_model_path_only() -> None:
    env = _mapping(
        {
            "MODEL_PATH": "/tmp/models/test_model.onnx",
            "MODEL_REGISTRY_DIR": "/tmp/registry",
            "ML_PREDICTION_THRESHOLD": "0.6",
            "ML_INFERENCE_BATCH_SIZE": "4",
            "ML_USE_MANIFEST_FEATURES": "false",
        },
    )

    cfg = MLInferenceConfig.from_env(env=env)

    assert cfg.model_path == "/tmp/models/test_model.onnx"
    assert cfg.model_id is None
    assert pytest.approx(cfg.prediction_threshold, rel=1e-9) == 0.6
    assert cfg.batch_size == 4
    assert cfg.use_manifest_features is False


def test_ml_inference_config_from_env_requires_identifier() -> None:
    with pytest.raises(ValueError):
        MLInferenceConfig.from_env(env={})


def test_ml_actor_config_from_env_resolves_core_fields() -> None:
    db_connection = build_postgres_url(
        user="user",
        password="pass",
        host="db",
        database="ml",
    )
    env = _mapping(
        {
            "MODEL_PATH": "/tmp/models/test_model.onnx",
            "MODEL_ID": "signal-model-v1",
            "MODEL_REGISTRY_DIR": "/tmp/registry",
            "INSTRUMENT_ID": "SPY.EQUS",
            "BAR_TYPE": "SPY.EQUS-1-MINUTE-LAST-EXTERNAL",
            "ML_DB_CONNECTION": db_connection,
            "ML_PUBLISH_SIGNALS": "false",
            "ML_ENABLE_ASYNC_PERSISTENCE": "false",
            "ML_PERSISTENCE_BATCH_SIZE": "64",
            "ML_SIGNAL_DATA_TYPE": "TestSignal",
            "COMPONENT_ID": "Component-001",
        },
    )

    cfg = MLActorConfig.from_env(env=env)

    assert cfg.model_id == "signal-model-v1"
    assert str(cfg.instrument_id) == "SPY.EQUS"
    assert str(cfg.bar_type) == "SPY.EQUS-1-MINUTE-LAST-EXTERNAL"
    assert cfg.db_connection == db_connection
    assert cfg.publish_signals is False
    assert cfg.enable_async_persistence is False
    assert cfg.persistence_batch_size == 64
    assert cfg.signal_data_type == "TestSignal"
    assert cfg.component_id is not None
    assert str(cfg.component_id) == "Component-001"


def test_ml_actor_config_from_env_requires_instrument() -> None:
    env = _mapping(
        {
            "MODEL_PATH": "/tmp/models/test_model.onnx",
            "BAR_TYPE": "SPY.EQUS-1-MINUTE-LAST-EXTERNAL",
            "INSTRUMENT_ID": "",
        },
    )
    with pytest.raises(ValueError):
        MLActorConfig.from_env(env=env)


def test_ml_strategy_config_from_env_parses_values() -> None:
    env = _mapping(
        {
            "STRATEGY_INSTRUMENT_ID": "SPY.EQUS",
            "ML_SIGNAL_SOURCE": "MLSignalActor-001",
            "ML_POSITION_SIZE_PCT": "0.2",
            "ML_MIN_CONFIDENCE": "0.8",
            "ML_MAX_POSITIONS": "3",
            "ML_STOP_LOSS_PCT": "0.01",
            "ML_TAKE_PROFIT_PCT": "0.03",
            "ML_USE_STRATEGY_STORE": "false",
            "ML_PERSIST_ALL_SIGNALS": "true",
            "ML_EXECUTE_TRADES": "true",
        },
    )

    cfg = MLStrategyConfig.from_env(env=env)

    assert cfg.instrument_id.value == "SPY.EQUS"
    assert cfg.ml_signal_source == "MLSignalActor-001"
    assert pytest.approx(cfg.position_size_pct, rel=1e-9) == 0.2
    assert pytest.approx(cfg.min_confidence, rel=1e-9) == 0.8
    assert cfg.max_positions == 3
    assert pytest.approx(cfg.stop_loss_pct, rel=1e-9) == 0.01
    assert pytest.approx(cfg.take_profit_pct, rel=1e-9) == 0.03
    assert cfg.use_strategy_store is False
    assert cfg.persist_all_signals is True
    assert cfg.execute_trades is True


def test_ml_strategy_config_validation_enforces_bounds() -> None:
    with pytest.raises(ValidationError):
        MLStrategyConfig(
            instrument_id=InstrumentId.from_str("SPY.EQUS"),
            ml_signal_source="actor",
            position_size_pct=1.5,
        )


def test_health_monitor_config_threshold_bounds() -> None:
    with pytest.raises(ValidationError):
        HealthMonitorConfig(degraded_success_rate_threshold=1.5)  # type: ignore[arg-type]


def test_onnx_runtime_config_requires_provider() -> None:
    with pytest.raises(ValidationError):
        OnnxRuntimeConfig(providers=[])


def test_onnx_runtime_config_rejects_non_positive_threads() -> None:
    with pytest.raises(ValidationError):
        OnnxRuntimeConfig(intra_threads=0)


def test_optuna_config_from_env() -> None:
    env = _mapping(
        {
            "ML_OPTUNA_ENABLED": "true",
            "ML_OPTUNA_TRIALS": "25",
            "ML_OPTUNA_DIRECTION": "minimize",
            "ML_OPTUNA_METRIC": "rmse",
            "ML_OPTUNA_PRUNER": "hyperband",
            "ML_OPTUNA_SAMPLER": "random",
            "ML_OPTUNA_TIMEOUT": "300",
            "ML_OPTUNA_STUDY_NAME": "lgbm-study",
            "ML_OPTUNA_STORAGE_URL": "sqlite:///optuna.db",
        },
    )

    cfg = OptunaConfig.from_env(env=env)

    assert cfg.enabled is True
    assert cfg.n_trials == 25
    assert cfg.direction == "minimize"
    assert cfg.metric == "rmse"
    assert cfg.pruner == "hyperband"
    assert cfg.sampler == "random"
    assert cfg.timeout == 300
    assert cfg.study_name == "lgbm-study"
    assert cfg.storage_url == "sqlite:///optuna.db"


def test_xgb_gpu_config_from_env() -> None:
    env = _mapping(
        {
            "ML_XGB_GPU_ENABLED": "true",
            "ML_XGB_GPU_DEVICE_ID": "2",
            "ML_XGB_GPU_PREDICTOR": "cpu_predictor",
            "ML_XGB_GPU_MAX_BIN": "512",
        },
    )

    cfg = XGBoostGPUConfig.from_env(env=env)

    assert cfg.enabled is True
    assert cfg.device_id == 2
    assert cfg.predictor == "cpu_predictor"
    assert cfg.max_bin == 512


def test_lightgbm_training_config_from_env() -> None:
    env = _mapping(
        {
            "ML_LGBM_DATA_SOURCE": "/data/shards/lgbm.parquet",
            "ML_LGBM_TARGET_COLUMN": "label",
            "ML_LGBM_N_ESTIMATORS": "200",
            "ML_LGBM_LEARNING_RATE": "0.05",
            "ML_LGBM_GPU_ENABLED": "true",
            "ML_LGBM_GPU_DEVICE_ID": "1",
            "ML_LGBM_GOSS_ENABLED": "true",
            "ML_LGBM_GOSS_TOP_RATE": "0.3",
            "ML_OPTUNA_ENABLED": "true",
            "ML_OPTUNA_TRIALS": "5",
            "ML_TRAIN_TRACK_FEATURE_DECAY": "false",
        },
    )

    cfg = LightGBMTrainingConfig.from_env(env=env)

    assert cfg.data_source == "/data/shards/lgbm.parquet"
    assert cfg.target_column == "label"
    assert cfg.n_estimators == 200
    assert pytest.approx(cfg.learning_rate, rel=1e-9) == 0.05
    assert cfg.gpu_config is not None and isinstance(cfg.gpu_config, LightGBMGPUConfig)
    assert cfg.gpu_config.enabled is True
    assert cfg.goss_config is not None and cfg.goss_config.enabled is True
    assert cfg.optuna_config is not None and cfg.optuna_config.enabled is True
    assert cfg.advanced_config is not None and isinstance(cfg.advanced_config, AdvancedTrainingConfig)
    assert cfg.advanced_config.track_feature_decay is False


def test_xgboost_training_config_from_env() -> None:
    env = _mapping(
        {
            "ML_XGB_DATA_SOURCE": "/data/xgb.csv",
            "ML_XGB_TARGET_COLUMN": "y",
            "ML_XGB_N_ESTIMATORS": "150",
            "ML_XGB_TREE_METHOD": "gpu_hist",
            "ML_XGB_GPU_ENABLED": "true",
            "ML_XGB_GPU_DEVICE_ID": "0",
            "ML_TRAIN_ONNX_OUTPUT_PATH": "/tmp/model.onnx",
        },
    )

    cfg = XGBoostTrainingConfig.from_env(env=env)

    assert cfg.data_source == "/data/xgb.csv"
    assert cfg.n_estimators == 150
    assert cfg.tree_method == "gpu_hist"
    assert cfg.gpu_config is not None and cfg.gpu_config.enabled is True
    assert cfg.advanced_config is not None
    assert cfg.advanced_config.onnx_output_path == "/tmp/model.onnx"
