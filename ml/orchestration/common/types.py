"""
Shared type definitions for orchestration components.

This module provides shared dataclasses used across orchestration components
to avoid duplication and ensure consistent data structures.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineCheckpoint:
    """
    Checkpoint state for pipeline resume capability.

    Tracks completed stages, progress, and allows resuming interrupted pipelines.

    Attributes
    ----------
    pipeline_id : str
        Unique identifier for the pipeline run
    stage : str
        Current/last completed stage name
    timestamp : int
        Timestamp in nanoseconds when checkpoint was created
    state : dict[str, object]
        Stage-specific state data
    completed_stages : list[str]
        List of completed stage names
    progress : float
        Progress fraction (0.0 to 1.0)

    Examples
    --------
    >>> checkpoint = PipelineCheckpoint(
    ...     pipeline_id="pipeline_20241201_120000",
    ...     stage="DATASET",
    ...     timestamp=1701432000000000000,
    ...     completed_stages=["PRE_INGEST", "AUTO_FILL", "DATASET"],
    ...     progress=0.4,
    ... )
    >>> checkpoint.save(Path("/tmp/checkpoint.json"))

    """

    pipeline_id: str
    stage: str
    timestamp: int
    state: dict[str, object] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)
    progress: float = 0.0

    def save(self, path: Path) -> None:
        """
        Save checkpoint to file.

        Parameters
        ----------
        path : Path
            File path to save checkpoint

        Raises
        ------
        OSError
            If file cannot be written

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline_id": self.pipeline_id,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "state": self.state,
            "completed_stages": self.completed_stages,
            "progress": self.progress,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug(
            "Checkpoint saved",
            extra={
                "path": str(path),
                "stage": self.stage,
                "completed_stages": self.completed_stages,
            },
        )

    @classmethod
    def load(cls, path: Path) -> PipelineCheckpoint:
        """
        Load checkpoint from file.

        Parameters
        ----------
        path : Path
            File path to load checkpoint from

        Returns
        -------
        PipelineCheckpoint
            Loaded checkpoint

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist
        ValueError
            If checkpoint file is invalid

        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid checkpoint JSON: {exc}") from exc

        return cls(
            pipeline_id=str(data.get("pipeline_id", "")),
            stage=str(data.get("stage", "")),
            timestamp=int(data.get("timestamp", 0)),
            state=dict(data.get("state", {})),
            completed_stages=list(data.get("completed_stages", [])),
            progress=float(data.get("progress", 0.0)),
        )


@dataclass(slots=True, frozen=True)
class EmptyDatasetError(Exception):
    """
    Dataset build produced zero rows.

    This error is raised when a dataset build completes but produces
    no data rows, indicating a data availability issue.

    Attributes
    ----------
    message : str
        Error message
    row_count : int | None
        Actual row count (usually 0)

    """

    message: str
    row_count: int | None = None

    def __str__(self) -> str:
        """Return string representation."""
        return self.message
