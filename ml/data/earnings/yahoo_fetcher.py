"""Compatibility shim for Yahoo earnings consensus fetcher."""

from __future__ import annotations

import warnings

from ml.features.earnings.ingestion import yahoo_fetcher as _yahoo
from ml.features.earnings.ingestion.yahoo_fetcher import EarningsConsensus
from ml.features.earnings.ingestion.yahoo_fetcher import YahooFetcher
from ml.features.earnings.ingestion.yahoo_fetcher import set_yfinance_override


warnings.warn(
    "ml.data.earnings.yahoo_fetcher is deprecated; "
    "import from ml.features.earnings.ingestion.yahoo_fetcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Expose yfinance symbol so existing patches continue to work
yfinance = _yahoo.yfinance

__all__ = ["EarningsConsensus", "YahooFetcher", "set_yfinance_override", "yfinance"]
