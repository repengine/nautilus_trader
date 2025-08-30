"""
Canonical event constants for ML pipeline stages and sources.

Use these to avoid ad-hoc string literals across the codebase.
Values are persisted to the database and must match schema constraints.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """
    Processing stages for ML pipeline events.

    Values must match database check constraints.
    """

    DATA_INGESTED = "INGESTED"
    CATALOG_WRITTEN = "CATALOG_WRITTEN"
    FEATURE_COMPUTED = "FEATURE_COMPUTED"
    PREDICTION_EMITTED = "PREDICTION_EMITTED"
    SIGNAL_EMITTED = "SIGNAL_EMITTED"


class Source(str, Enum):
    """
    Allowed event sources persisted by the registry.
    """

    LIVE = "live"
    HISTORICAL = "historical"
    BACKFILL = "backfill"


__all__ = ["Stage", "Source"]
