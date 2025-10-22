"""Configuration classes for the streaming training pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping

from msgspec import ValidationError

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class DatasetServiceConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for the dataset planning microservice."""

    parquet_root: str
    shard_row_budget: PositiveInt = 200_000
    max_total_rows: PositiveInt | None = None
    max_total_sequences: PositiveInt | None = None
    max_shards: PositiveInt | None = None
    plan_cache_ttl_seconds: PositiveInt = 600
    retry_backoff_seconds: PositiveFloat = 5.0
    max_retry_attempts: PositiveInt = 3
    include_macro: bool = False
    include_calendar: bool = False
    include_events: bool = False
    include_earnings: bool = False
    include_micro: bool = False
    include_l2: bool = False
    include_macro_revisions: bool = False

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.max_total_rows is not None and self.max_total_rows < self.shard_row_budget:
            raise ValidationError("max_total_rows must be >= shard_row_budget when set")

    @classmethod
    def from_env(
        cls,
        parquet_root: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> DatasetServiceConfig:
        """
        Build a planning configuration from environment variables.

        Environment overrides:
            ML_STREAMING_SHARD_ROW_BUDGET: integer shard budget
            ML_STREAMING_MAX_TOTAL_ROWS: integer or <=0 for unlimited
            ML_STREAMING_MAX_TOTAL_SEQUENCES: integer or <=0 for unlimited
            ML_STREAMING_MAX_SHARDS: integer or <=0 for unlimited
            ML_STREAMING_INCLUDE_MACRO: truthy/falsey toggle
            ML_STREAMING_INCLUDE_CALENDAR: truthy/falsey toggle
            ML_STREAMING_INCLUDE_EVENTS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_EARNINGS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_MICRO: truthy/falsey toggle
            ML_STREAMING_INCLUDE_L2: truthy/falsey toggle
            ML_STREAMING_INCLUDE_MACRO_REVISIONS: truthy/falsey toggle

        Args:
            parquet_root: Root directory containing parquet datasets.
            env: Optional environment mapping. Defaults to ``os.environ``.

        Returns:
            DatasetServiceConfig: Parsed configuration.
        """
        source = env or os.environ

        def _int(name: str, default: int | None) -> int | None:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = int(str(raw).strip())
            except ValueError:
                return default
            return None if value <= 0 else value

        def _truthy(name: str, default: bool) -> bool:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized in {"1", "true", "yes", "y", "on"}

        return cls(
            parquet_root=parquet_root,
            shard_row_budget=_int("ML_STREAMING_SHARD_ROW_BUDGET", 200_000) or 200_000,
            max_total_rows=_int("ML_STREAMING_MAX_TOTAL_ROWS", None),
            max_total_sequences=_int("ML_STREAMING_MAX_TOTAL_SEQUENCES", None),
            max_shards=_int("ML_STREAMING_MAX_SHARDS", None),
            include_macro=_truthy("ML_STREAMING_INCLUDE_MACRO", False),
            include_calendar=_truthy("ML_STREAMING_INCLUDE_CALENDAR", False),
            include_events=_truthy("ML_STREAMING_INCLUDE_EVENTS", False),
            include_earnings=_truthy("ML_STREAMING_INCLUDE_EARNINGS", False),
            include_micro=_truthy("ML_STREAMING_INCLUDE_MICRO", False),
            include_l2=_truthy("ML_STREAMING_INCLUDE_L2", False),
            include_macro_revisions=_truthy("ML_STREAMING_INCLUDE_MACRO_REVISIONS", False),
        )


class StreamingWorkerConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for bounded streaming training workers."""

    max_total_rows: PositiveInt | None = 500_000
    max_total_sequences: PositiveInt | None = 300_000
    max_shards: PositiveInt | None = 4
    max_epochs: PositiveInt = 1
    max_concurrent_jobs: PositiveInt = 1
    max_runtime_seconds: PositiveInt = 1_800
    heartbeat_interval_seconds: PositiveInt = 30
    max_retry_attempts: PositiveInt = 3
    retry_backoff_seconds: NonNegativeFloat = 5.0
    accelerator: str = "auto"
    devices: PositiveInt = 1
    model_id: str = "tft-streaming-teacher"
    train_fraction: PositiveFloat = 0.8
    logits_artifact_key: str = "logits"
    validation_metric: str = "roc_auc"
    gpu_memory_monitor_interval_seconds: NonNegativeFloat | None = 30.0

    def __post_init__(self) -> None:
        """Ensure time and resource constraints are consistent."""
        if self.max_concurrent_jobs > 1 and self.max_shards is not None:
            if self.max_shards < self.max_concurrent_jobs:
                raise ValidationError("max_shards must be >= max_concurrent_jobs when both set")
        if not (0.0 < float(self.train_fraction) < 1.0):
            raise ValidationError("train_fraction must be in the open interval (0, 1)")
        if not self.model_id.strip():
            raise ValidationError("model_id must be non-empty")
        if not self.logits_artifact_key.strip():
            raise ValidationError("logits_artifact_key must be non-empty")
        metric = self.validation_metric.strip().lower()
        if metric not in {"roc_auc"}:
            raise ValidationError("validation_metric must be one of {'roc_auc'}")
        if self.max_retry_attempts < 1:
            raise ValidationError("max_retry_attempts must be >= 1")
        if float(self.retry_backoff_seconds) < 0.0:
            raise ValidationError("retry_backoff_seconds must be >= 0.0")
        if self.gpu_memory_monitor_interval_seconds is not None and float(
            self.gpu_memory_monitor_interval_seconds,
        ) <= 0.0:
            raise ValidationError("gpu_memory_monitor_interval_seconds must be > 0 when set")
        if int(self.max_epochs) < 1:
            raise ValidationError("max_epochs must be >= 1")


class TrainingOrchestratorConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for event-driven streaming orchestrator."""

    command_topic: str
    result_topic: str
    heartbeat_topic: str
    max_in_flight_plans: PositiveInt = 8
    dataset_retry_limit: PositiveInt = 2
    worker_timeout_seconds: PositiveInt = 600
    retry_window_seconds: PositiveInt = 300
    max_plan_age_seconds: PositiveInt = 7_200
    saturation_heartbeat_limit: PositiveInt = 5
    backlog_warning_threshold: NonNegativeInt = 10
    enable_state_persistence: bool = True
    publish_retry_attempts: PositiveInt = 3
    publish_retry_delay_seconds: NonNegativeFloat = 0.5

    def __post_init__(self) -> None:
        """Validate orchestrator settings."""
        if self.command_topic and self.command_topic == self.result_topic:
            raise ValidationError("command_topic and result_topic must differ when provided")
        if self.heartbeat_topic and self.heartbeat_topic == self.command_topic:
            raise ValidationError("heartbeat_topic must differ from command_topic when provided")
        if self.heartbeat_topic and self.heartbeat_topic == self.result_topic:
            raise ValidationError("heartbeat_topic must differ from result_topic when provided")
        if self.worker_timeout_seconds <= self.heartbeat_interval_hint:
            raise ValidationError("worker_timeout_seconds must exceed heartbeat interval hint")
        if self.max_plan_age_seconds <= self.worker_timeout_seconds:
            raise ValidationError("max_plan_age_seconds must exceed worker_timeout_seconds")
        if self.retry_window_seconds >= self.max_plan_age_seconds:
            raise ValidationError("retry_window_seconds must be less than max_plan_age_seconds")
        if int(self.publish_retry_attempts) < 1:
            raise ValidationError("publish_retry_attempts must be >= 1")
        if float(self.publish_retry_delay_seconds) < 0.0:
            raise ValidationError("publish_retry_delay_seconds must be >= 0")

    @property
    def heartbeat_interval_hint(self) -> NonNegativeFloat:
        """Preferred heartbeat cadence derived from backlog threshold."""
        # Ensure non-zero divisor; fallback to 1.0s when backlog threshold is zero
        divisor = float(max(1, int(self.backlog_warning_threshold or 1)))
        return float(self.worker_timeout_seconds) / divisor


class StreamingPersistenceConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the streaming training persistence worker.

    Args:
        enabled: Whether the worker should actively poll the message bus.
        state_path: Filesystem path used to persist streaming state snapshots.
        batch_size: Maximum number of Redis stream entries to process per poll.
        block_ms: Milliseconds to block on each Redis `XREAD` invocation.
        poll_interval_seconds: Delay applied after idle polls to reduce load.

    Example:
        >>> cfg = StreamingPersistenceConfig()
        >>> cfg.batch_size
        128
    """

    enabled: bool = True
    state_path: str = "./ml_out/streaming_training_state.json"
    batch_size: PositiveInt = 128
    block_ms: NonNegativeInt = 1_000
    poll_interval_seconds: NonNegativeFloat = 0.5

    def __post_init__(self) -> None:
        """Validate configuration fields."""
        if not self.state_path.strip():
            raise ValidationError("state_path must be non-empty")
        if float(self.poll_interval_seconds) < 0.0:
            raise ValidationError("poll_interval_seconds must be >= 0.0")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> StreamingPersistenceConfig:
        """
        Build persistence configuration from environment variables.

        Environment overrides:
            ML_STREAM_PERSIST_ENABLE: truthy/falsey toggle (default: current value)
            ML_STREAM_PERSIST_STATE_PATH: filesystem path for persistence snapshot
            ML_STREAM_PERSIST_BATCH_SIZE: integer batch size per poll
            ML_STREAM_PERSIST_BLOCK_MS: integer Redis block duration in ms
            ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS: float idle interval between polls

        Args:
            env: Optional mapping of environment variables to read. Defaults to ``os.environ``.

        Returns:
            StreamingPersistenceConfig: Parsed configuration with overrides applied.
        """
        source = env or os.environ

        def _truthy(name: str, default: bool) -> bool:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized in {"1", "true", "yes", "y", "on"}

        def _int(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return int(str(raw).strip())
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return float(str(raw).strip())
            except ValueError:
                return default

        base = cls()
        state_path_raw = source.get("ML_STREAM_PERSIST_STATE_PATH", base.state_path)
        return cls(
            enabled=_truthy("ML_STREAM_PERSIST_ENABLE", bool(base.enabled)),
            state_path=str(state_path_raw),
            batch_size=_int("ML_STREAM_PERSIST_BATCH_SIZE", int(base.batch_size)),
            block_ms=_int("ML_STREAM_PERSIST_BLOCK_MS", int(base.block_ms)),
            poll_interval_seconds=_float(
                "ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS",
                float(base.poll_interval_seconds),
            ),
        )


__all__ = [
    "DatasetServiceConfig",
    "StreamingPersistenceConfig",
    "StreamingWorkerConfig",
    "TrainingOrchestratorConfig",
]
