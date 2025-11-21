"""
Canonical event constants for ML pipeline stages and sources.

Use these to avoid ad-hoc string literals across the codebase. Values are persisted to
the database and must match schema constraints.

"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """
    Processing stages for ML pipeline events.

    Values must match database check constraints.

    """

    DATASET_PLANNED = "DATASET_PLANNED"
    DATA_INGESTED = "INGESTED"
    CATALOG_WRITTEN = "CATALOG_WRITTEN"
    FEATURE_COMPUTED = "FEATURE_COMPUTED"
    # Back-compat alias used by some tests (equivalent to PREDICTION_EMITTED)
    MODEL_INFERRED = "MODEL_INFERRED"
    MODEL_TRAINING_STARTED = "MODEL_TRAINING_STARTED"
    MODEL_TRAINING_COMPLETED = "MODEL_TRAINING_COMPLETED"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    PREDICTION_EMITTED = "PREDICTION_EMITTED"
    SIGNAL_EMITTED = "SIGNAL_EMITTED"


class Source(str, Enum):
    """
    Allowed event sources persisted by the registry.
    """

    LIVE = "live"
    # Back-compat alias used in some tests
    BATCH = "batch"
    HISTORICAL = "historical"
    BACKFILL = "backfill"


class EventStatus(str, Enum):
    """
    Standardized status values for emitted events.

    Values are persisted and validated by Pandera/DB contracts. Use ``.value`` when
    storing to the database or serializing payloads.

    """

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    DEFERRED = "deferred"


LEGACY_STAGE_ALIAS_MAP: dict[str, Stage] = {
    # Legacy aliases used by older pipelines and integration tests.
    "MODEL_INFERRED": Stage.PREDICTION_EMITTED,
}
"""Mapping of legacy stage identifiers to canonical ``Stage`` members."""

CANONICAL_STAGE_EQUIVALENTS: dict[Stage, Stage] = {
    Stage.MODEL_INFERRED: Stage.PREDICTION_EMITTED,
}
"""Canonical representation for ``Stage`` members with legacy aliases."""


__all__ = [
    "CANONICAL_STAGE_EQUIVALENTS",
    "LEGACY_STAGE_ALIAS_MAP",
    "EventStatus",
    "Source",
    "Stage",
]
