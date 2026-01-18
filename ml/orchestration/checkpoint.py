#!/usr/bin/env python3

"""
Pipeline checkpoint support for resumability.

This module provides checkpoint persistence to enable pipeline execution to resume
from interruption points without re-executing completed stages.

"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


logger = logging.getLogger(__name__)


# Metrics
_CHECKPOINT_SAVE_COUNTER = get_counter(
    "ml_checkpoint_save_total",
    "Total number of checkpoint save operations",
)
_CHECKPOINT_LOAD_COUNTER = get_counter(
    "ml_checkpoint_load_total",
    "Total number of checkpoint load operations",
)
_CHECKPOINT_SAVE_DURATION = get_histogram(
    "ml_checkpoint_save_duration_seconds",
    "Duration of checkpoint save operations in seconds",
)
_CHECKPOINT_LOAD_DURATION = get_histogram(
    "ml_checkpoint_load_duration_seconds",
    "Duration of checkpoint load operations in seconds",
)


class PipelineCheckpointProtocol(Protocol):
    """Protocol for pipeline checkpoint persistence."""

    def save(self, path: Path) -> None:
        """
        Save checkpoint to disk.

        Args:
            path: File path to write checkpoint to

        Raises:
            OSError: If checkpoint cannot be written to disk
            ValueError: If checkpoint data is invalid

        """
        ...

    @classmethod
    def load(cls, path: Path) -> PipelineCheckpoint:
        """
        Load checkpoint from disk.

        Args:
            path: File path to read checkpoint from

        Returns:
            PipelineCheckpoint instance loaded from disk

        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint file is corrupt or invalid
            OSError: If checkpoint cannot be read from disk

        """
        ...


@dataclass(frozen=True)
class PipelineCheckpoint:
    """
    Pipeline execution checkpoint for resumability.

    This checkpoint captures the state of a pipeline execution at a specific point,
    enabling resumption from interruption without re-executing completed stages.

    Attributes:
        pipeline_id: Unique identifier for the pipeline run
        stage: Current stage being executed when checkpoint was created
        timestamp: Unix timestamp (nanoseconds) when checkpoint was created
        state: Arbitrary state dictionary for stage-specific data
        completed_stages: List of stage names that have been successfully completed
        progress: Fractional progress within current stage (0.0 to 1.0)

    Example:
        >>> checkpoint = PipelineCheckpoint(
        ...     pipeline_id="pipeline_20250115_123456",
        ...     stage="DATASET",
        ...     timestamp=1705324800000000000,
        ...     state={"rows_processed": 1500, "total_rows": 5000},
        ...     completed_stages=["INGEST"],
        ...     progress=0.3,
        ... )
        >>> checkpoint.save(Path("/tmp/checkpoint.json"))
        >>> loaded = PipelineCheckpoint.load(Path("/tmp/checkpoint.json"))
        >>> assert loaded.completed_stages == ["INGEST"]
        >>> assert loaded.progress == 0.3

    """

    pipeline_id: str
    stage: str
    timestamp: int
    state: dict[str, Any] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)
    progress: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate checkpoint fields.

        Raises:
            ValueError: If progress is not in [0.0, 1.0] range or timestamp is negative

        """
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"progress must be in [0.0, 1.0], got {self.progress}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if not self.pipeline_id:
            raise ValueError("pipeline_id cannot be empty")
        if not self.stage:
            raise ValueError("stage cannot be empty")

    def save(self, path: Path) -> None:
        """
        Save checkpoint to disk as JSON.

        Args:
            path: File path to write checkpoint to

        Raises:
            OSError: If checkpoint cannot be written to disk
            ValueError: If checkpoint data cannot be serialized

        Example:
            >>> checkpoint = PipelineCheckpoint(
            ...     pipeline_id="test_pipeline",
            ...     stage="TRAIN",
            ...     timestamp=1705324800000000000,
            ...     completed_stages=["INGEST", "DATASET"],
            ...     progress=0.5,
            ... )
            >>> checkpoint.save(Path("/tmp/checkpoint.json"))

        """
        import time

        start_time = time.perf_counter()

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize checkpoint to JSON
            checkpoint_data = asdict(self)

            # Write atomically via temporary file
            temp_path = path.with_suffix(".tmp")
            with temp_path.open("w") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Atomic rename (overwrites existing checkpoint)
            temp_path.replace(path)

            duration = time.perf_counter() - start_time
            _CHECKPOINT_SAVE_DURATION.observe(duration)
            _CHECKPOINT_SAVE_COUNTER.inc()

            logger.info(
                "Checkpoint saved",
                extra={
                    "pipeline_id": self.pipeline_id,
                    "stage": self.stage,
                    "path": str(path),
                    "completed_stages": self.completed_stages,
                    "progress": self.progress,
                    "duration_seconds": duration,
                },
            )

        except OSError as e:
            logger.error(
                "Failed to save checkpoint",
                extra={
                    "pipeline_id": self.pipeline_id,
                    "stage": self.stage,
                    "path": str(path),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
        except (TypeError, ValueError) as e:
            logger.error(
                "Failed to serialize checkpoint",
                extra={
                    "pipeline_id": self.pipeline_id,
                    "stage": self.stage,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ValueError(f"Cannot serialize checkpoint: {e}") from e

    @classmethod
    def load(cls, path: Path) -> PipelineCheckpoint:
        """
        Load checkpoint from disk.

        Args:
            path: File path to read checkpoint from

        Returns:
            PipelineCheckpoint instance loaded from disk

        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint file is corrupt or invalid
            OSError: If checkpoint cannot be read from disk

        Example:
            >>> checkpoint = PipelineCheckpoint.load(Path("/tmp/checkpoint.json"))
            >>> assert checkpoint.pipeline_id
            >>> assert checkpoint.stage
            >>> assert len(checkpoint.completed_stages) >= 0

        """
        import time

        start_time = time.perf_counter()

        if not path.exists():
            logger.warning(
                "Checkpoint file not found",
                extra={"path": str(path)},
            )
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            # Read checkpoint JSON
            with path.open("r") as f:
                checkpoint_data = json.load(f)

            # Validate required fields
            required_fields = {"pipeline_id", "stage", "timestamp"}
            missing_fields = required_fields - checkpoint_data.keys()
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Reconstruct checkpoint (validation happens in __post_init__)
            checkpoint = cls(**checkpoint_data)

            duration = time.perf_counter() - start_time
            _CHECKPOINT_LOAD_DURATION.observe(duration)
            _CHECKPOINT_LOAD_COUNTER.inc()

            logger.info(
                "Checkpoint loaded",
                extra={
                    "pipeline_id": checkpoint.pipeline_id,
                    "stage": checkpoint.stage,
                    "path": str(path),
                    "completed_stages": checkpoint.completed_stages,
                    "progress": checkpoint.progress,
                    "duration_seconds": duration,
                },
            )

            return checkpoint

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse checkpoint JSON",
                extra={
                    "path": str(path),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ValueError(f"Corrupt checkpoint file: {e}") from e
        except (TypeError, KeyError) as e:
            logger.error(
                "Invalid checkpoint data",
                extra={
                    "path": str(path),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ValueError(f"Invalid checkpoint data: {e}") from e
        except OSError as e:
            logger.error(
                "Failed to read checkpoint",
                extra={
                    "path": str(path),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
