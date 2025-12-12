"""
Backwards-compatibility shim for earnings data module.

DEPRECATED: Import from ml.features.earnings instead:
    from ml.features.earnings import EarningsCache, EdgarFetcher, YahooFetcher, etc.
"""

import warnings

from ml.features.earnings.cache import EarningsCache
from ml.features.earnings.ingestion.edgar_fetcher import EarningsActual
from ml.features.earnings.ingestion.edgar_fetcher import EdgarFetcher
from ml.features.earnings.ingestion.service import EarningsIngestionService
from ml.features.earnings.ingestion.universe import ResolvedUniverse
from ml.features.earnings.ingestion.universe import resolve_ingestion_universe
from ml.features.earnings.ingestion.xbrl_parser import XBRLParser
from ml.features.earnings.ingestion.yahoo_fetcher import EarningsConsensus
from ml.features.earnings.ingestion.yahoo_fetcher import YahooFetcher


warnings.warn(
    "ml.data.earnings is deprecated. "
    "Import from ml.features.earnings instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    "EarningsActual",
    "EarningsCache",
    "EarningsConsensus",
    "EarningsIngestionService",
    "EdgarFetcher",
    "ResolvedUniverse",
    "XBRLParser",
    "YahooFetcher",
    "resolve_ingestion_universe",
]
