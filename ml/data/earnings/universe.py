"""Compatibility shim for earnings ingestion universe helpers."""

from __future__ import annotations

import warnings

from ml.features.earnings.ingestion.universe import ResolvedUniverse
from ml.features.earnings.ingestion.universe import resolve_ingestion_universe


warnings.warn(
    "ml.data.earnings.universe is deprecated; "
    "import from ml.features.earnings.ingestion.universe instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ResolvedUniverse", "resolve_ingestion_universe"]
