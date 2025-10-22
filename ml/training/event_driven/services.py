"""Service skeletons for the event-driven streaming pipeline."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


@dataclass(slots=True, frozen=True)
class DatasetPlanEvent:
    """Event emitted when a dataset plan is created."""

    plan_id: str
    dataset_id: str
    parquet_path: Path
    metadata: TFTStreamingMetadata
    metadata_summary: TFTStreamingSummary
    limits: StreamingLimitSummary
    streaming_config: TFTStreamingConfig
    caps: dict[str, float | int | None]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: EventStatus = EventStatus.SUCCESS


@dataclass(slots=True, frozen=True)
class TrainingResultEvent:
    """Event emitted after a bounded streaming training job completes."""

    plan_id: str
    dataset_id: str
    model_id: str
    telemetry: StreamingRunTelemetry
    artifact_paths: dict[str, str]
    metrics: dict[str, float]
    status: EventStatus
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True, frozen=True)
class TrainingHeartbeatEvent:
    """Heartbeat emitted by streaming workers to report liveness."""

    worker_id: str
    plan_id: str | None
    dataset_id: str | None
    progress_pct: float
    rss_mb: float
    shards_processed: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True, frozen=True)
class DatasetPlanRequest:
    """Request describing how to plan a dataset for streaming training."""

    dataset_id: str
    streaming_config: TFTStreamingConfig
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    parquet_path: Path | None = None


class DatasetPlanner(ABC):
    """Abstract interface for dataset planning service."""

    def __init__(self, config: DatasetServiceConfig) -> None:
        self._config = config

    @property
    def config(self) -> DatasetServiceConfig:
        """Return planner configuration."""
        return self._config

    @abstractmethod
    def plan(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        """Produce a dataset plan for downstream streaming workers."""


class TrainingWorker(ABC):
    """Abstract base for streaming training workers."""

    def __init__(self, config: StreamingWorkerConfig) -> None:
        self._config = config

    @property
    def config(self) -> StreamingWorkerConfig:
        """Return worker configuration."""
        return self._config

    @abstractmethod
    def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
        """Execute a bounded training job for the provided dataset plan."""


class OrchestratorBus(Protocol):
    """Protocol describing orchestrator message bus interactions."""

    def publish_plan(self, topic: str, event: DatasetPlanEvent) -> None:
        """Publish a dataset plan event."""

    def publish_result(self, topic: str, event: TrainingResultEvent) -> None:
        """Publish a training result event."""

    def publish_heartbeat(self, topic: str, event: TrainingHeartbeatEvent) -> None:
        """Publish a worker heartbeat event."""


class StreamingTrainingOrchestrator(ABC):
    """Coordinating service for dataset planner and streaming workers."""

    def __init__(
        self,
        config: TrainingOrchestratorConfig,
        planner: DatasetPlanner,
        bus: OrchestratorBus,
    ) -> None:
        self._config = config
        self._planner = planner
        self._bus = bus

    @property
    def config(self) -> TrainingOrchestratorConfig:
        """Return orchestrator configuration."""
        return self._config

    @abstractmethod
    def enqueue_training(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        """Plan and schedule a streaming training job for the provided dataset."""

    @abstractmethod
    def handle_heartbeat(self, heartbeat: TrainingHeartbeatEvent) -> None:
        """React to worker heartbeat to maintain orchestration state."""


__all__ = [
    "DatasetPlanEvent",
    "DatasetPlanRequest",
    "DatasetPlanner",
    "OrchestratorBus",
    "StreamingTrainingOrchestrator",
    "TrainingHeartbeatEvent",
    "TrainingResultEvent",
    "TrainingWorker",
]
