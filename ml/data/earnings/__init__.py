"""
Earnings data fetchers for Nautilus Trader ML.

This module provides integration with SEC EDGAR (primary) and Yahoo Finance (secondary)
to fetch corporate earnings data including actuals, consensus estimates, and calendars.

Performance targets: API fetch <2s (EDGAR), <500ms (Yahoo); processing <100ms
Hot/Cold path separation: All data fetching is cold-path only; hot path uses pre-cached data

Public API exports (alphabetically sorted):
- EarningsActual: Dataclass for actual earnings data
- EarningsCache: Point-in-time cache for backtesting correctness
- EarningsConsensus: Dataclass for consensus estimate data
- EdgarFetcher: Fetch actual earnings from SEC EDGAR filings (10-Q, 10-K, 8-K)
- XBRLParser: Utilities for parsing XBRL financial data
- YahooFetcher: Fetch consensus estimates from Yahoo Finance

"""

from ml.data.earnings.earnings_cache import EarningsCache
from ml.data.earnings.edgar_fetcher import EarningsActual
from ml.data.earnings.edgar_fetcher import EdgarFetcher
from ml.data.earnings.xbrl_parser import XBRLParser
from ml.data.earnings.yahoo_fetcher import EarningsConsensus
from ml.data.earnings.yahoo_fetcher import YahooFetcher


__all__ = [
    "EarningsActual",
    "EarningsCache",
    "EarningsConsensus",
    "EdgarFetcher",
    "XBRLParser",
    "YahooFetcher",
]
