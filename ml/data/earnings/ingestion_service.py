"""Compatibility shim for earnings ingestion service."""

from __future__ import annotations

import warnings

from ml.features.earnings.ingestion.service import EarningsIngestionService


warnings.warn(
    "ml.data.earnings.ingestion_service is deprecated; "
    "import from ml.features.earnings.ingestion.service instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["EarningsIngestionService"]
