"""Telemetry DTOs for TFT streaming training."""

from __future__ import annotations

from dataclasses import dataclass

from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_loader import count_sequences


@dataclass(slots=True, frozen=True)
class StreamingLoaderTelemetry:
    """Aggregate selection statistics for a streaming DataLoader."""

    loader: str
    total_shards: int
    selected_shards: int
    skipped_shards: int
    total_rows: int
    selected_rows: int
    skipped_rows: int
    total_sequences: int
    selected_sequences: int
    skipped_sequences: int

    @classmethod
    def from_metadata(
        cls,
        loader: str,
        metadata: TFTStreamingMetadata,
        summary: StreamingLimitSummary,
        config: TFTStreamingConfig,
    ) -> StreamingLoaderTelemetry:
        """Return telemetry derived from limited metadata and skip summary."""
        selected_shards = len(metadata.shard_indices)
        selected_rows = sum(metadata.instrument_row_counts.values())
        selected_sequences = max(0, count_sequences(metadata, config))
        total_shards = selected_shards + summary.skipped_shards
        total_rows = selected_rows + summary.skipped_rows
        total_sequences = selected_sequences + summary.skipped_sequences
        return cls(
            loader=loader,
            total_shards=total_shards,
            selected_shards=selected_shards,
            skipped_shards=summary.skipped_shards,
            total_rows=total_rows,
            selected_rows=selected_rows,
            skipped_rows=summary.skipped_rows,
            total_sequences=total_sequences,
            selected_sequences=selected_sequences,
            skipped_sequences=summary.skipped_sequences,
        )

    def as_dict(self) -> dict[str, int | str]:
        """Return telemetry as a serializable dictionary."""
        return {
            "loader": self.loader,
            "total_shards": self.total_shards,
            "selected_shards": self.selected_shards,
            "skipped_shards": self.skipped_shards,
            "total_rows": self.total_rows,
            "selected_rows": self.selected_rows,
            "skipped_rows": self.skipped_rows,
            "total_sequences": self.total_sequences,
            "selected_sequences": self.selected_sequences,
            "skipped_sequences": self.skipped_sequences,
        }

    def as_logging_extra(self) -> dict[str, int | str]:
        """Return telemetry formatted for structured logging."""
        return self.as_dict()


@dataclass(slots=True, frozen=True)
class StreamingRunTelemetry:
    """Telemetry describing a full streaming training run."""

    metadata_summary: TFTStreamingSummary
    caps: dict[str, float | int | None]
    train: StreamingLoaderTelemetry
    validation: StreamingLoaderTelemetry
    max_gpu_memory_mb: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return run telemetry as a serializable mapping."""
        return {
            "caps": self.caps,
            "metadata": {
                "total_shards": self.metadata_summary.total_shards,
                "total_rows": self.metadata_summary.total_rows,
                "max_shard_rows": self.metadata_summary.max_shard_rows,
            },
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "resources": (
                {"max_gpu_memory_mb": self.max_gpu_memory_mb}
                if self.max_gpu_memory_mb is not None
                else {}
            ),
        }


__all__ = [
    "StreamingLoaderTelemetry",
    "StreamingRunTelemetry",
]
