from __future__ import annotations

import pytest
from msgspec import ValidationError

from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig


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


def test_training_orchestrator_config_enforces_topics() -> None:
    cfg = TrainingOrchestratorConfig(
        command_topic="ml.dataset.cmd",
        result_topic="ml.training.result",
        heartbeat_topic="ml.training.heartbeat",
    )
    assert cfg.command_topic != cfg.result_topic
    assert cfg.heartbeat_interval_hint > 0.0

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
