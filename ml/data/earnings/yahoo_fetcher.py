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
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

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
                    # Get first upcoming date
                    next_date = earnings_dates.index[0]
                    if isinstance(next_date, datetime):
                        return next_date
                    # Convert to datetime if timestamp
                    return datetime.fromisoformat(str(next_date))

            # Fallback: try calendar property
            if hasattr(stock, "calendar"):
                calendar = stock.calendar
                if calendar is not None and len(calendar) > 0:
                    earnings_date = calendar.get("Earnings Date")
                    if earnings_date is not None:
                        if isinstance(earnings_date, datetime):
                            return earnings_date
                        return datetime.fromisoformat(str(earnings_date))

            return None

        except Exception as e:
            logger.debug("Failed to extract earnings date: %s", e)
            return None

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
]
