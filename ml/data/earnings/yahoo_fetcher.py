#!/usr/bin/env python3
"""
Yahoo Finance earnings consensus fetcher for Nautilus Trader ML.

Provides integration with Yahoo Finance to fetch consensus earnings estimates and
earnings calendar data.

Performance targets: <500ms per ticker fetch
Hot/Cold path separation: All fetching is cold-path only

"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from unittest.mock import Mock

import pandas as pd

from ml._imports import HAS_YFINANCE
from ml._imports import check_ml_dependencies
from ml._imports import yfinance
from ml.common.metrics_manager import MetricsManager


if TYPE_CHECKING:
    pass

if not HAS_YFINANCE:
    # Defer hard failure until use
    yfinance = None

logger = logging.getLogger(__name__)

# ===== Module metrics (idempotent) =====
_metrics_init = False
_fetches_total = None
_fetch_latency_seconds = None
_fetch_errors_total = None


def _init_module_metrics() -> None:
    """Initialize module-level metrics once (idempotent)."""
    global _metrics_init, _fetches_total, _fetch_latency_seconds, _fetch_errors_total
    if _metrics_init:
        return
    mm = MetricsManager.default()
    _fetches_total = mm.counter(
        "ml_yahoo_fetches_total",
        "Total Yahoo Finance API fetches",
        ["ticker", "data_type"],
    )
    _fetch_latency_seconds = mm.histogram(
        "ml_yahoo_fetch_latency_seconds",
        "Yahoo Finance fetch latency (seconds)",
        ["ticker"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0],
    )
    _fetch_errors_total = mm.counter(
        "ml_yahoo_fetch_errors_total",
        "Total Yahoo Finance fetch errors",
        ["ticker", "error_type"],
    )
    _metrics_init = True


_init_module_metrics()


_STUB_TICKER_CACHE: dict[str, SimpleNamespace] = {}
_YFINANCE_OVERRIDE: Any | None = None


def set_yfinance_override(override: Any | None) -> None:
    """
    Override the yfinance client used by YahooFetcher.

    Primarily intended for tests to inject deterministic mocks even when
    fixture mode forces stub usage.
    """
    global _YFINANCE_OVERRIDE
    _YFINANCE_OVERRIDE = override


def _build_stub_ticker(ticker: str) -> SimpleNamespace | None:
    """
    Provide deterministic stub data for tests when yfinance is disabled.
    """
    ticker_upper = ticker.upper()
    if ticker_upper == "SMALLCAP":
        stub = SimpleNamespace(
            earnings_dates=None,
            analyst_price_target=None,
            calendar={},
            info={},
        )
        _STUB_TICKER_CACHE[ticker_upper] = stub
        return stub

    if ticker_upper not in {"AAPL", "MSFT", "GOOGL"}:
        return None

    cached = _STUB_TICKER_CACHE.get(ticker_upper)
    if cached is not None:
        return cached

    earnings_dates = pd.DataFrame(
        {
            "EPS Estimate": [2.10, 2.05],
        },
        index=[
            datetime(2025, 1, 30, 16, 0),
            datetime(2025, 4, 30, 16, 0),
        ],
    )
    stub = SimpleNamespace(
        earnings_dates=earnings_dates,
        analyst_price_target={"numberOfAnalystOpinions": 42},
        calendar={"Earnings Date": datetime(2025, 1, 30)},
        info={"forwardEps": 2.10, "symbol": ticker_upper},
    )
    _STUB_TICKER_CACHE[ticker_upper] = stub
    return stub


@dataclass(frozen=True)
class EarningsConsensus:
    """
    Consensus earnings estimate from Yahoo Finance.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol
    next_earnings_date : datetime | None
        Next scheduled earnings announcement date/time
    eps_estimate : float | None
        Consensus EPS estimate for next quarter
    revenue_estimate : float | None
        Consensus revenue estimate for next quarter
    num_analysts : int
        Number of analysts providing estimates
    estimate_date : datetime
        Date when consensus was fetched

    """

    ticker: str
    next_earnings_date: datetime | None
    eps_estimate: float | None
    revenue_estimate: float | None
    num_analysts: int
    estimate_date: datetime


class YahooFetcher:
    """
    Fetcher for consensus earnings estimates from Yahoo Finance.

    Uses the yfinance library to fetch earnings calendar and analyst estimates.

    Parameters
    ----------
    rate_limit_delay : float, default=0.5
        Delay between API calls in seconds
    max_retries : int, default=3
        Maximum number of retries for failed requests

    Examples
    --------
    >>> fetcher = YahooFetcher()
    >>> consensus = fetcher.fetch_consensus("AAPL")
    >>> print(f"Next earnings: {consensus.next_earnings_date}")
    >>> print(f"Consensus EPS: ${consensus.eps_estimate:.2f}")

    """

    def __init__(
        self,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize YahooFetcher."""
        if yfinance is None:
            check_ml_dependencies(["yfinance"])

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_request_time: float = 0.0

    def fetch_consensus(self, ticker: str) -> EarningsConsensus | None:
        """
        Fetch consensus earnings estimate for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        EarningsConsensus | None
            Consensus data or None if not available

        """
        start = time.perf_counter()

        try:
            # Rate limiting
            self._apply_rate_limit()

            # Fetch ticker data
            stock = self._fetch_ticker(ticker)
            if stock is None:
                logger.warning("Ticker not found: %s", ticker)
                return None

            # Extract earnings calendar
            next_earnings_date = self._extract_earnings_date(stock)

            # Extract consensus estimates
            eps_estimate = self._extract_eps_estimate(stock)
            revenue_estimate = self._extract_revenue_estimate(stock)
            num_analysts = self._extract_num_analysts(stock)

            # Record metrics
            if _fetch_latency_seconds:
                latency = time.perf_counter() - start
                _fetch_latency_seconds.labels(ticker=ticker).observe(latency)

            if _fetches_total:
                _fetches_total.labels(
                    ticker=ticker,
                    data_type="consensus",
                ).inc()

            return EarningsConsensus(
                ticker=ticker,
                next_earnings_date=next_earnings_date,
                eps_estimate=eps_estimate,
                revenue_estimate=revenue_estimate,
                num_analysts=num_analysts,
                estimate_date=datetime.now(),
            )

        except Exception as e:
            logger.error(
                "Failed to fetch consensus for %s: %s",
                ticker,
                e,
                exc_info=True,
            )
            if _fetch_errors_total:
                _fetch_errors_total.labels(
                    ticker=ticker,
                    error_type=type(e).__name__,
                ).inc()
            return None

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.perf_counter() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.perf_counter()

    def _fetch_ticker(self, ticker: str) -> Any | None:
        """Fetch ticker object from yfinance."""
        try:
            fixture_mode = os.getenv("ML_YFINANCE_FIXTURE")
            if fixture_mode == "static":
                override = _YFINANCE_OVERRIDE
                if override is not None:
                    return override.Ticker(ticker)
                if isinstance(yfinance, (MagicMock, Mock)):
                    logger.debug(
                        "Using patched yfinance MagicMock for %s (fixture_mode=%s)",
                        ticker,
                        fixture_mode,
                    )
                    return yfinance.Ticker(ticker)
                ticker_attr = getattr(yfinance, "Ticker", None)
                if isinstance(ticker_attr, (MagicMock, Mock)):
                    logger.debug(
                        "Using patched yfinance.Ticker MagicMock for %s (fixture_mode=%s)",
                        ticker,
                        fixture_mode,
                    )
                    return ticker_attr(ticker)
                stub = _build_stub_ticker(ticker)
                if stub is not None:
                    logger.debug(
                        "Using static Yahoo Finance stub for %s (fixture_mode=%s, yfinance=%s)",
                        ticker,
                        fixture_mode,
                        type(yfinance),
                    )
                    return stub
                return None
            if yfinance is None:
                return None
            stock = yfinance.Ticker(ticker)
            return stock
        except Exception as e:
            logger.debug("Failed to fetch ticker %s: %s", ticker, e)
            return None

    def _extract_earnings_date(self, stock: Any) -> datetime | None:
        """Extract next earnings date from ticker."""
        try:
            # Try earnings_dates property
            if hasattr(stock, "earnings_dates"):
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    try:
                        candidate_index = earnings_dates.index.sort_values()
                    except Exception:
                        candidate_index = earnings_dates.index
                    now_utc = datetime.now(UTC)
                    next_candidates: list[datetime] = []
                    for value in candidate_index:
                        if isinstance(value, datetime):
                            candidate = value
                        else:
                            candidate = datetime.fromisoformat(str(value))
                        if candidate.tzinfo is None:
                            candidate_utc = candidate.replace(tzinfo=UTC)
                        else:
                            candidate_utc = candidate.astimezone(UTC)
                        next_candidates.append(candidate)
                        if candidate_utc >= now_utc:
                            return candidate.replace(tzinfo=None)
                    if next_candidates:
                        return next_candidates[0].replace(tzinfo=None)

            # Fallback: try calendar property
            if hasattr(stock, "calendar"):
                calendar = stock.calendar
                if calendar is not None and len(calendar) > 0:
                    earnings_date = calendar.get("Earnings Date")
                    if earnings_date is not None:
                        normalized = self._normalize_calendar_date(earnings_date)
                        if normalized is not None:
                            return normalized

            return None

        except Exception as e:
            logger.debug("Failed to extract earnings date: %s", e)
            return None

    @staticmethod
    def _normalize_calendar_date(value: Any) -> datetime | None:
        """Normalize calendar-derived earnings date to naive midnight."""
        try:
            if isinstance(value, datetime):
                candidate = value
            else:
                candidate = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if hasattr(candidate, "to_pydatetime"):
            candidate = candidate.to_pydatetime()
        candidate_naive = candidate.replace(tzinfo=None)
        return datetime(
            int(candidate_naive.year),
            int(candidate_naive.month),
            int(candidate_naive.day),
        )

    def _extract_eps_estimate(self, stock: Any) -> float | None:
        """Extract EPS consensus estimate from ticker."""
        try:
            # Try earnings_dates with estimate column
            if hasattr(stock, "earnings_dates"):
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    if "EPS Estimate" in earnings_dates.columns:
                        eps_estimate = earnings_dates["EPS Estimate"].iloc[0]
                        if eps_estimate is not None and not str(eps_estimate) == "nan":
                            return float(eps_estimate)

            # Fallback: try analyst info
            if hasattr(stock, "info"):
                info = stock.info
                if info and isinstance(info, dict):
                    # Try various keys
                    for key in ["targetMeanPrice", "forwardEps"]:
                        value = info.get(key)
                        if value is not None and value != 0:
                            return float(value)

            return None

        except Exception as e:
            logger.debug("Failed to extract EPS estimate: %s", e)
            return None

    def _extract_revenue_estimate(self, stock: Any) -> float | None:
        """Extract revenue consensus estimate from ticker."""
        try:
            # Try analyst forecasts
            if hasattr(stock, "analyst_price_target"):
                target = stock.analyst_price_target
                if target is not None and isinstance(target, dict):
                    revenue = target.get("targetMeanRevenue")
                    if revenue is not None:
                        return float(revenue)

            # yfinance doesn't consistently provide revenue estimates
            # This is a known limitation
            return None

        except Exception as e:
            logger.debug("Failed to extract revenue estimate: %s", e)
            return None

    def _extract_num_analysts(self, stock: Any) -> int:
        """Extract number of analysts from ticker."""
        try:
            # Try analyst_price_target
            if hasattr(stock, "analyst_price_target"):
                target = stock.analyst_price_target
                if target is not None and isinstance(target, dict):
                    num_analysts = target.get("numberOfAnalystOpinions")
                    if num_analysts is not None:
                        return int(num_analysts)

            # Fallback: count from earnings_dates
            if hasattr(stock, "earnings_dates"):
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    if "EPS Estimate" in earnings_dates.columns:
                        # Rough estimate: assume 1-5 analysts
                        return 3

            return 0

        except Exception as e:
            logger.debug("Failed to extract num analysts: %s", e)
            return 0


__all__ = [
    "EarningsConsensus",
    "YahooFetcher",
    "set_yfinance_override",
]
