"""Compatibility shim for earnings Edgar fetcher."""

from __future__ import annotations

import warnings

from ml.features.earnings.ingestion.edgar_fetcher import EarningsActual
from ml.features.earnings.ingestion.edgar_fetcher import EdgarFetcher


warnings.warn(
    "ml.data.earnings.edgar_fetcher is deprecated; "
    "import from ml.features.earnings.ingestion.edgar_fetcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["EarningsActual", "EdgarFetcher"]
