from __future__ import annotations

import pytest
from msgspec import ValidationError

from ml.config.streaming_pipeline import CurriculumGuardContext
from ml.config.streaming_pipeline import CurriculumGuardRule
from ml.config.streaming_pipeline import CurriculumScheduleConfig
from ml.config.streaming_pipeline import CurriculumStageConfig
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import EnsembleMemberConfig
from ml.config.streaming_pipeline import StreamingEnsembleConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.config.streaming_pipeline import parse_curriculum_guard_spec
from ml.config.streaming_pipeline import parse_curriculum_stage_spec
from ml.config.streaming_pipeline import parse_ensemble_member_spec


def test_dataset_service_config_validates_rows() -> None:
    cfg = DatasetServiceConfig(parquet_root="/data/tft")
    assert cfg.shard_row_budget == 200_000
    assert cfg.retry_backoff_seconds == 5.0


def test_dataset_service_config_raises_when_rows_invalid() -> None:
    with pytest.raises(ValidationError):
        DatasetServiceConfig(
            parquet_root="/data/tft",
            shard_row_budget=500_000,
            max_total_rows=100_000,
        )


def test_dataset_service_config_from_env_parses_toggles(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "ML_STREAMING_SHARD_ROW_BUDGET": "150000",
        "ML_STREAMING_MAX_TOTAL_ROWS": "200000",
        "ML_STREAMING_MAX_TOTAL_SEQUENCES": "50000",
        "ML_STREAMING_MAX_SHARDS": "16",
        "ML_STREAMING_INCLUDE_MACRO": "1",
        "ML_STREAMING_INCLUDE_CALENDAR": "true",
        "ML_STREAMING_INCLUDE_EVENTS": "yes",
        "ML_STREAMING_INCLUDE_EARNINGS": "on",
        "ML_STREAMING_INCLUDE_MICRO": "1",
        "ML_STREAMING_INCLUDE_L2": "true",
        "ML_STREAMING_INCLUDE_MACRO_REVISIONS": "1",
        "ML_STREAMING_INCLUDE_MACRO_DELTAS": "y",
        "ML_STREAMING_INCLUDE_CALENDAR_LAGS": "yes",
        "ML_STREAMING_INCLUDE_CLUSTERING_TAGS": "true",
        "ML_STREAMING_INCLUDE_CONTEXT_FEATURES": "on",
    }
    cfg = DatasetServiceConfig.from_env("/datasets/tft", env=env)
    assert cfg.parquet_root == "/datasets/tft"
    assert cfg.shard_row_budget == 150_000
    assert cfg.max_total_rows == 200_000
    assert cfg.max_total_sequences == 50_000
    assert cfg.max_shards == 16
    assert cfg.include_macro is True
    assert cfg.include_calendar is True
    assert cfg.include_events is True
    assert cfg.include_earnings is True
    assert cfg.include_micro is True
    assert cfg.include_l2 is True
    assert cfg.include_macro_revisions is True
    assert cfg.include_macro_deltas is True
    assert cfg.include_calendar_lags is True
    assert cfg.include_clustering_tags is True
    assert cfg.include_context_features is True


def test_dataset_service_config_from_env_handles_defaults() -> None:
    cfg = DatasetServiceConfig.from_env("/datasets/tft", env={})
    assert cfg.parquet_root == "/datasets/tft"
    assert cfg.shard_row_budget == 200_000
    assert cfg.max_total_rows is None
    assert cfg.max_total_sequences is None
    assert cfg.max_shards is None
    assert cfg.include_macro is False
    assert cfg.include_calendar is False
    assert cfg.include_events is False
    assert cfg.include_earnings is False
    assert cfg.include_micro is False
    assert cfg.include_l2 is False
    assert cfg.include_macro_revisions is False
    assert cfg.include_macro_deltas is False
    assert cfg.include_calendar_lags is False
    assert cfg.include_clustering_tags is False
    assert cfg.include_context_features is False


def test_streaming_worker_config_validates_shard_capacity() -> None:
    cfg = StreamingWorkerConfig(max_concurrent_jobs=1, max_shards=2)
    assert cfg.max_shards == 2
    assert cfg.max_retry_attempts == 3
    assert cfg.retry_backoff_seconds == 5.0

    with pytest.raises(ValidationError):
        StreamingWorkerConfig(max_concurrent_jobs=3, max_shards=2)

    with pytest.raises(ValidationError):
        StreamingWorkerConfig(train_fraction=1.0)
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(train_fraction=0.0)
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(model_id=" ")
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(validation_metric="accuracy")
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(logits_artifact_key="  ")
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(max_retry_attempts=0)
    with pytest.raises(ValidationError):
        StreamingWorkerConfig(retry_backoff_seconds=-1.0)


def test_streaming_worker_config_from_env_parses_overrides() -> None:
    env = {
        "ML_STREAMING_MAX_TOTAL_ROWS": "150000",
        "ML_STREAMING_MAX_TOTAL_SEQUENCES": "110000",
        "ML_STREAMING_MAX_SHARDS": "40",
        "ML_STREAMING_MAX_EPOCHS": "2",
        "ML_STREAMING_MAX_RUNTIME_SECONDS": "2400",
        "ML_STREAMING_HEARTBEAT_INTERVAL_SECONDS": "45",
        "ML_STREAMING_MAX_RETRY_ATTEMPTS": "5",
        "ML_STREAMING_RETRY_BACKOFF_SECONDS": "7.5",
        "ML_STREAMING_ACCELERATOR": "cuda",
        "ML_STREAMING_DEVICES": "2",
        "ML_STREAMING_TRAIN_FRACTION": "0.75",
        "ML_STREAMING_LOGITS_KEY": "logits_v2",
        "ML_STREAMING_VALIDATION_METRIC": "roc_auc",
        "ML_STREAMING_GPU_MONITOR_INTERVAL_SECONDS": "0",
        "ML_STREAMING_TFT_HIDDEN_SIZE": "128",
        "ML_STREAMING_TFT_LSTM_LAYERS": "3",
        "ML_STREAMING_TFT_ATTENTION_HEAD_SIZE": "4",
        "ML_STREAMING_TFT_DROPOUT": "0.2",
        "ML_STREAMING_TFT_LEARNING_RATE": "0.0005",
        "ML_STREAMING_TFT_OPTIMIZER": "adamw",
        "ML_STREAMING_TFT_LR_SCHEDULER": "cosine",
        "ML_STREAMING_TFT_LOSS": "bce",
        "ML_STREAMING_TFT_LOSS_POS_WEIGHT": "3.0",
        "ML_STREAMING_VALIDATION_RETURN_COLUMN": "custom_forward",
        "ML_STREAMING_TFT_ENABLE_TEMPERATURE_CALIBRATION": "true",
        "ML_STREAMING_TFT_TEMPERATURE_MIN": "0.5",
        "ML_STREAMING_TFT_TEMPERATURE_MAX": "4.0",
        "ML_STREAMING_TFT_TEMPERATURE_STEPS": "15",
        "ML_STREAMING_TFT_ENABLE_PLATT_CALIBRATION": "1",
        "ML_STREAMING_TFT_ENABLE_ISOTONIC_CALIBRATION": "true",
        "ML_STREAMING_PRECISION": "bf16",
        "ML_STREAMING_ENABLE_AMP": "true",
        "ML_STREAMING_AMP_PRECISION": "16-mixed",
        "ML_STREAMING_AMP_GUARD_THRESHOLD_MB": "2500",
        "ML_STREAMING_CURRICULUM_ENABLED": "1",
        "ML_STREAMING_CURRICULUM_STAGES": "50000:0.55;*:0.8",
        "ML_STREAMING_CURRICULUM_DEFAULT_TRAIN_FRACTION": "0.65",
        "ML_STREAMING_CURRICULUM_GUARDS": "phase1:min_rows=20000,max_gpu_mb=2500,fallback_fraction=0.6,reason=guarded",
        "ML_STREAMING_ENSEMBLE_ENABLED": "1",
        "ML_STREAMING_ENSEMBLE_BLEND_MODE": "mean",
        "ML_STREAMING_ENSEMBLE_NORMALIZE_WEIGHTS": "0",
        "ML_STREAMING_ENSEMBLE_MEMBERS": "peer1.npz:0.5:required;peer2.npz:0.5",
    }
    cfg = StreamingWorkerConfig.from_env(env=env)
    assert cfg.max_total_rows == 150_000
    assert cfg.max_total_sequences == 110_000
    assert cfg.max_shards == 40
    assert cfg.max_epochs == 2
    assert cfg.max_runtime_seconds == 2_400
    assert cfg.heartbeat_interval_seconds == 45
    assert cfg.max_retry_attempts == 5
    assert cfg.retry_backoff_seconds == pytest.approx(7.5)
    assert cfg.accelerator == "cuda"
    assert cfg.devices == 2
    assert cfg.train_fraction == pytest.approx(0.75)
    assert cfg.logits_artifact_key == "logits_v2"
    assert cfg.validation_metric == "roc_auc"
    assert cfg.gpu_memory_monitor_interval_seconds is None
    assert cfg.hidden_size == 128
    assert cfg.lstm_layers == 3
    assert cfg.attention_head_size == 4
    assert cfg.dropout == pytest.approx(0.2)
    assert cfg.learning_rate == pytest.approx(0.0005)
    assert cfg.optimizer == "adamw"
    assert cfg.lr_scheduler == "cosine"
    assert cfg.loss_name == "bce"
    assert cfg.loss_pos_weight == pytest.approx(3.0)
    assert cfg.enable_temperature_calibration is True
    assert cfg.temperature_calibration_min == pytest.approx(0.5)
    assert cfg.temperature_calibration_max == pytest.approx(4.0)
    assert cfg.temperature_calibration_steps == 15
    assert cfg.enable_platt_calibration is True
    assert cfg.enable_isotonic_calibration is True
    assert cfg.precision == "bf16"
    assert cfg.enable_amp is True
    assert cfg.amp_precision == "16-mixed"
    assert cfg.amp_guard_threshold_mb == pytest.approx(2500.0)
    assert cfg.curriculum.enabled is True
    assert len(cfg.curriculum.stages) == 2
    assert cfg.curriculum.default_train_fraction == pytest.approx(0.65)
    assert cfg.curriculum.stages[0].train_fraction == pytest.approx(0.55)
    assert len(cfg.curriculum.guards) == 1
    guard = cfg.curriculum.guards[0]
    assert guard.stage_label == "phase1"
    assert guard.fallback_train_fraction == pytest.approx(0.6)
    assert cfg.ensemble.enabled is True
    assert cfg.ensemble.blend_mode == "mean"
    assert cfg.ensemble.normalize_weights is False
    assert len(cfg.ensemble.members) == 2
    assert cfg.validation_return_column == "custom_forward"


def test_streaming_worker_config_from_env_defaults() -> None:
    cfg = StreamingWorkerConfig.from_env(env={})
    assert cfg.max_total_rows is None
    assert cfg.max_total_sequences is None
    assert cfg.max_shards is None
    assert cfg.max_epochs == 1
    assert cfg.max_runtime_seconds == 1_800
    assert cfg.heartbeat_interval_seconds == 30
    assert cfg.max_retry_attempts == 3
    assert cfg.retry_backoff_seconds == pytest.approx(5.0)
    assert cfg.accelerator == "auto"
    assert cfg.devices == 1
    assert cfg.train_fraction == pytest.approx(0.8)
    assert cfg.logits_artifact_key == "logits"
    assert cfg.validation_metric == "roc_auc"
    assert cfg.gpu_memory_monitor_interval_seconds == pytest.approx(30.0)
    assert cfg.hidden_size == 16
    assert cfg.lstm_layers == 1
    assert cfg.attention_head_size == 2
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.optimizer == "adam"
    assert cfg.lr_scheduler == "reduce_on_plateau"
    assert cfg.loss_name == "bce"
    assert cfg.loss_pos_weight is None
    assert cfg.enable_temperature_calibration is True
    assert cfg.temperature_calibration_min == pytest.approx(0.25)
    assert cfg.temperature_calibration_max == pytest.approx(5.0)
    assert cfg.temperature_calibration_steps == 25
    assert cfg.enable_platt_calibration is False
    assert cfg.enable_isotonic_calibration is False
    assert cfg.amp_guard_threshold_mb is None
    assert cfg.validation_return_column == "forward_return"


def test_curriculum_parsing_and_resolution() -> None:
    stage = parse_curriculum_stage_spec("*:0.6:phaseA")
    assert stage.max_total_rows is None
    assert stage.train_fraction == pytest.approx(0.6)
    assert stage.label == "phaseA"

    schedule = CurriculumScheduleConfig(
        enabled=True,
        stages=(
            CurriculumStageConfig(max_total_rows=50_000, train_fraction=0.55, label="phase-a"),
            CurriculumStageConfig(max_total_rows=100_000, train_fraction=0.7),
        ),
        default_train_fraction=0.65,
    )
    assert schedule.resolve_fraction(total_rows=40_000) == pytest.approx(0.55)

    guard = CurriculumGuardRule(
        stage_label="phase-a",
        min_total_rows=30_000,
        fallback_train_fraction=0.6,
        reason="rows below guard",
    )
    guarded = CurriculumScheduleConfig(
        enabled=True,
        stages=(CurriculumStageConfig(max_total_rows=60_000, train_fraction=0.5, label="phase-a"),),
        default_train_fraction=0.7,
        guards=(guard,),
    )
    context = CurriculumGuardContext(total_rows=25_000)
    resolution = guarded.resolve_with_context(total_rows=25_000, context=context)
    assert resolution.train_fraction == pytest.approx(0.6)
    assert resolution.guard_reason == "rows below guard"


def test_parse_curriculum_guard_spec() -> None:
    guard = parse_curriculum_guard_spec(
        "phase1:min_rows=10000,min_roc_auc=0.55,max_gpu_mb=2500,train_fraction=0.6,reason=guarded",
    )
    assert guard.stage_label == "phase1"
    assert guard.min_total_rows == 10_000
    assert guard.min_roc_auc == pytest.approx(0.55)
    assert guard.max_gpu_mb == pytest.approx(2_500.0)
    assert guard.fallback_train_fraction == pytest.approx(0.6)
    assert guard.reason == "guarded"


def test_ensemble_parsing_and_validation() -> None:
    member = parse_ensemble_member_spec("peer.npz:0.4:required")
    assert member.artifact_path == "peer.npz"
    assert member.weight == pytest.approx(0.4)
    assert member.required is True

    optional_member = parse_ensemble_member_spec("relative_peer.npz")
    assert optional_member.weight == pytest.approx(1.0)
    assert optional_member.required is False

    with pytest.raises(ValidationError):
        StreamingEnsembleConfig(enabled=True, members=())


def test_training_orchestrator_config_enforces_topics() -> None:
    cfg = TrainingOrchestratorConfig(
        command_topic="ml.dataset.cmd",
        result_topic="ml.training.result",
        heartbeat_topic="ml.training.heartbeat",
    )
    assert cfg.command_topic != cfg.result_topic
    assert cfg.heartbeat_interval_hint > 0.0
    assert cfg.adaptive_backlog_threshold is None
    assert cfg.adaptive_gpu_threshold_mb is None
    assert cfg.adaptive_cooldown_seconds == pytest.approx(120.0)
    assert cfg.adaptive_interval_multiplier == pytest.approx(2.0)

    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="same",
            result_topic="same",
            heartbeat_topic="hb",
        )
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="cmd",
        )


def test_training_orchestrator_config_requires_timeout_margin() -> None:
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            worker_timeout_seconds=10,
            backlog_warning_threshold=0,
        )


def test_training_orchestrator_config_allows_topic_builder_defaults() -> None:
    cfg = TrainingOrchestratorConfig(
        command_topic="",
        result_topic="",
        heartbeat_topic="",
    )
    assert cfg.command_topic == ""
    assert cfg.result_topic == ""
    assert cfg.heartbeat_topic == ""


def test_training_orchestrator_config_validates_retry_window() -> None:
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            worker_timeout_seconds=100,
            max_plan_age_seconds=80,
        )
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            worker_timeout_seconds=100,
            max_plan_age_seconds=200,
            retry_window_seconds=400,
        )


def test_training_orchestrator_config_validates_adaptive_guards() -> None:
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            adaptive_backlog_threshold=0,
        )
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            adaptive_gpu_threshold_mb=0.0,
        )
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            adaptive_cooldown_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        TrainingOrchestratorConfig(
            command_topic="cmd",
            result_topic="result",
            heartbeat_topic="hb",
            adaptive_interval_multiplier=0.5,
        )


def test_streaming_persistence_config_defaults() -> None:
    cfg = StreamingPersistenceConfig()
    assert cfg.enabled
    assert cfg.batch_size == 128
    assert cfg.block_ms == 1_000


def test_streaming_persistence_config_validates_path_and_interval() -> None:
    with pytest.raises(ValidationError):
        StreamingPersistenceConfig(state_path=" ")
    with pytest.raises(ValidationError):
        StreamingPersistenceConfig(poll_interval_seconds=-0.1)


def test_streaming_persistence_config_from_env_overrides() -> None:
    env = {
        "ML_STREAM_PERSIST_ENABLE": "0",
        "ML_STREAM_PERSIST_STATE_PATH": "/tmp/custom.json",
        "ML_STREAM_PERSIST_BATCH_SIZE": "64",
        "ML_STREAM_PERSIST_BLOCK_MS": "250",
        "ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS": "1.5",
    }
    cfg = StreamingPersistenceConfig.from_env(env)
    assert cfg.enabled is False
    assert cfg.state_path == "/tmp/custom.json"
    assert cfg.batch_size == 64
    assert cfg.block_ms == 250
    assert cfg.poll_interval_seconds == pytest.approx(1.5)
