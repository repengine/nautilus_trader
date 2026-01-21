#!/usr/bin/env python3
"""
SEC EDGAR earnings data fetcher for Nautilus Trader ML.

Provides integration with SEC EDGAR to fetch actual earnings data from 10-Q/10-K filings.
Uses the edgartools library for API access and XBRL parsing.

Performance targets: <2s per filing fetch, <100ms per parse
Hot/Cold path separation: All fetching is cold-path only

"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_EDGARTOOLS
from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import load_edgartools
from ml._imports import pd as pd_runtime
from ml.common.metrics_manager import MetricsManager
from ml.common.retry_utils import retry_with_backoff
from ml.features.earnings.ingestion.xbrl_parser import XBRLParser


if TYPE_CHECKING:
    pass

edgartools: ModuleType | None = None

logger = logging.getLogger(__name__)

# ===== Module metrics (idempotent) =====
_metrics_init = False
_fetches_total = None
_fetch_latency_seconds = None
_parse_errors_total = None


def _coerce_date(value: object) -> date | None:
    """Coerce a date-like value into a ``date``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                return None
    return None


def _init_module_metrics() -> None:
    """Initialize module-level metrics once (idempotent)."""
    global _metrics_init, _fetches_total, _fetch_latency_seconds, _parse_errors_total
    if _metrics_init:
        return
    mm = MetricsManager.default()
    _fetches_total = mm.counter(
        "ml_edgar_fetches_total",
        "Total EDGAR API fetches",
        ["ticker", "form_type"],
    )
    _fetch_latency_seconds = mm.histogram(
        "ml_edgar_fetch_latency_seconds",
        "EDGAR fetch latency (seconds)",
        ["ticker"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
    )
    _parse_errors_total = mm.counter(
        "ml_edgar_parse_errors_total",
        "Total EDGAR parsing errors",
        ["ticker", "error_type"],
    )
    _metrics_init = True


_init_module_metrics()


def _resolve_edgartools() -> ModuleType | None:
    """Return the edgartools module, loading it on demand."""
    global edgartools
    if edgartools is not None:
        return edgartools
    if not HAS_EDGARTOOLS:
        return None
    edgartools = load_edgartools()
    return edgartools


@dataclass(frozen=True)
class EarningsActual:
    """
    Actual earnings data from SEC EDGAR filing.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol
    period_end : date
        Quarter end date
    filing_date : date
        Date 10-Q/10-K was filed with SEC
    eps_basic : float | None
        Basic earnings per share
    eps_diluted : float | None
        Diluted earnings per share
    revenue : float | None
        Total revenue in dollars
    net_income : float | None
        Net income in dollars
    operating_income : float | None
        Operating income in dollars
    shares_outstanding : int | None
        Shares outstanding
    filing_type : str
        Form type ('10-Q' or '10-K')
    fiscal_year : int
        Fiscal year
    fiscal_quarter : int
        Fiscal quarter (1-4)

    """

    ticker: str
    period_end: date
    filing_date: date
    eps_basic: float | None
    eps_diluted: float | None
    revenue: float | None
    net_income: float | None
    operating_income: float | None
    shares_outstanding: int | None
    filing_type: str
    fiscal_year: int
    fiscal_quarter: int

    @property
    def fiscal_period(self) -> str:
        """Return the fiscal period token (e.g., ``Q4``)."""
        return f"Q{self.fiscal_quarter}"


class EdgarFetcher:
    """
    Fetcher for actual earnings data from SEC EDGAR.

    Uses the edgartools library to fetch and parse 10-Q/10-K filings with XBRL data.

    Parameters
    ----------
    rate_limit_delay : float, default=1.0
        Delay between API calls in seconds (SEC recommends <10 req/sec)
    max_retries : int, default=3
        Maximum number of retries for failed requests

    Examples
    --------
    >>> fetcher = EdgarFetcher()
    >>> actuals = fetcher.fetch_earnings("AAPL", quarters=4)
    >>> print(f"Latest EPS: ${actuals[0].eps_diluted:.2f}")

    """

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize EdgarFetcher."""
        if not HAS_EDGARTOOLS:
            check_ml_dependencies(["edgartools"])

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_request_time: float = 0.0

    def fetch_earnings(
        self,
        ticker: str,
        quarters: int = 4,
        form: str = "10-Q",
    ) -> list[EarningsActual]:
        """
        Fetch actual earnings for a ticker from SEC EDGAR.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        quarters : int, default=4
            Number of quarters to fetch
        form : str, default="10-Q"
            Form type to fetch ('10-Q' or '10-K')

        Returns
        -------
        list[EarningsActual]
            List of earnings actuals, sorted by period_end descending (newest first)

        """
        start = time.perf_counter()

        try:
            # Fetch company and filings
            company = self._fetch_company(ticker)
            if company is None:
                logger.warning("Company not found for ticker %s", ticker)
                return []

            filings = self._fetch_filings(company, form, quarters)
            if not filings:
                logger.warning("No filings found for ticker %s", ticker)
                return []

            # Parse earnings from filings
            actuals = []
            for filing in filings:
                actual = self._parse_filing(ticker, filing, form)
                if actual is not None:
                    actuals.append(actual)

            # Record metrics
            if _fetch_latency_seconds:
                latency = time.perf_counter() - start
                _fetch_latency_seconds.labels(ticker=ticker).observe(latency)

            if _fetches_total:
                _fetches_total.labels(
                    ticker=ticker,
                    form_type=form,
                ).inc()

            return actuals

        except Exception as e:
            logger.error(
                "Failed to fetch earnings for %s: %s",
                ticker,
                e,
                exc_info=True,
            )
            if _parse_errors_total:
                _parse_errors_total.labels(
                    ticker=ticker,
                    error_type=type(e).__name__,
                ).inc()
            return []

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.perf_counter() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.perf_counter()

    def _fetch_company(self, ticker: str) -> Any | None:
        """Fetch company object from EDGAR."""
        def _fetch_once() -> Any | None:
            self._apply_rate_limit()
            tools = _resolve_edgartools()
            if tools is None:
                return None
            return tools.Company(ticker)

        def _on_exc(attempt: int, exc: BaseException) -> None:
            logger.warning(
                "EDGAR company fetch retry %d/%d for %s: %s",
                attempt + 1,
                self.max_retries,
                ticker,
                exc,
                exc_info=True,
            )

        try:
            return retry_with_backoff(
                _fetch_once,
                max_attempts=int(self.max_retries),
                initial_delay=float(self.rate_limit_delay),
                multiplier=2.0,
                max_delay=60.0,
                jitter=0.0,
                sleep_fn=time.sleep,
                on_exception=_on_exc,
            )
        except Exception as exc:
            logger.debug("Failed to fetch company %s: %s", ticker, exc, exc_info=True)
            return None

    def _fetch_filings(
        self,
        company: Any,
        form: str,
        quarters: int,
    ) -> list[Any]:
        """Fetch filings from company object."""
        def _fetch_once() -> list[Any]:
            self._apply_rate_limit()
            filings = company.get_filings(form=form).latest(quarters)
            if not filings:
                return []
            if isinstance(filings, Iterable) and not isinstance(filings, (str, bytes)):
                return list(filings)
            return [filings]

        def _on_exc(attempt: int, exc: BaseException) -> None:
            logger.warning(
                "EDGAR filings fetch retry %d/%d: %s",
                attempt + 1,
                self.max_retries,
                exc,
                exc_info=True,
            )

        try:
            return retry_with_backoff(
                _fetch_once,
                max_attempts=int(self.max_retries),
                initial_delay=float(self.rate_limit_delay),
                multiplier=2.0,
                max_delay=60.0,
                jitter=0.0,
                sleep_fn=time.sleep,
                on_exception=_on_exc,
            )
        except Exception as exc:
            logger.debug("Failed to fetch filings: %s", exc, exc_info=True)
            return []

    def _parse_filing(
        self,
        ticker: str,
        filing: Any,
        form: str,
    ) -> EarningsActual | None:
        """Parse earnings data from a filing."""
        try:
            # Extract XBRL facts
            facts = self._extract_xbrl_facts(filing)
            if not facts:
                logger.debug(
                    "No XBRL facts found in filing",
                    extra={"ticker": ticker, "form": form},
                )
                return None

            # Parse using XBRLParser
            eps_diluted = XBRLParser.extract_eps(facts, prefer_diluted=True)
            eps_basic = XBRLParser.extract_eps(facts, prefer_diluted=False)
            revenue = XBRLParser.extract_revenue(facts)
            net_income = XBRLParser.extract_net_income(facts)
            shares = XBRLParser.extract_shares_outstanding(facts)

            # Extract dates and metadata
            period_end = self._extract_period_end(filing)
            filing_date = self._extract_filing_date(filing)
            fiscal_year, fiscal_quarter = self._extract_fiscal_period(filing)

            if period_end is None or filing_date is None:
                logger.debug(
                    "Missing required dates in filing",
                    extra={
                        "ticker": ticker,
                        "form": form,
                        "period_end": period_end,
                        "filing_date": filing_date,
                    },
                )
                return None
            if all(
                metric is None
                for metric in (eps_diluted, eps_basic, revenue, net_income, shares)
            ):
                logger.debug(
                    "No usable metrics in filing",
                    extra={"ticker": ticker, "form": form},
                )
                return None

            return EarningsActual(
                ticker=ticker,
                period_end=period_end,
                filing_date=filing_date,
                eps_basic=eps_basic,
                eps_diluted=eps_diluted,
                revenue=revenue,
                net_income=net_income,
                operating_income=None,  # Not commonly available in XBRL
                shares_outstanding=shares,
                filing_type=form,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
            )

        except Exception as e:
            logger.debug("Failed to parse filing: %s", e, exc_info=True)
            return None

    def _extract_xbrl_facts(self, filing: Any) -> dict[str, Any]:
        """Extract XBRL facts dictionary from filing."""
        try:
            # edgartools provides xbrl() method
            xbrl = filing.xbrl()
            if xbrl is None:
                return {}

            # Convert to dict
            facts = xbrl.facts if hasattr(xbrl, "facts") else {}
            if isinstance(facts, dict):
                return facts
            if hasattr(facts, "get_facts_by_concept"):
                return self._extract_facts_view(facts)
            return {}

        except Exception as e:
            logger.debug("Failed to extract XBRL: %s", e)
            return {}

    def _extract_facts_view(self, facts_view: Any) -> dict[str, Any]:
        """Extract a tag -> value mapping from a FactsView-like object."""
        if not HAS_PANDAS or pd_runtime is None:
            check_ml_dependencies(["pandas"])
        assert pd_runtime is not None
        pd_local = pd_runtime
        tags = [
            "us-gaap:EarningsPerShareDiluted",
            "us-gaap:EarningsPerShareBasic",
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "us-gaap:SalesRevenueGoodsNet",
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss",
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        ]
        values: dict[str, Any] = {}
        for tag in tags:
            try:
                frame = facts_view.get_facts_by_concept(tag)
            except Exception:
                continue
            if frame is None or getattr(frame, "empty", True):
                continue
            series = None
            if "numeric_value" in frame.columns:
                series = frame["numeric_value"]
            elif "value" in frame.columns:
                series = frame["value"]
            if series is None:
                continue
            if "period_end" in frame.columns:
                frame = frame.copy()
                frame["period_end"] = pd_local.to_datetime(
                    frame["period_end"],
                    errors="coerce",
                )
                frame = frame.sort_values("period_end")
                series = frame["numeric_value"] if "numeric_value" in frame.columns else frame["value"]
            series = series.dropna()
            if series.empty:
                continue
            values[tag] = series.iloc[-1]
        return values

    def _extract_period_end(self, filing: Any) -> date | None:
        """Extract period end date from filing."""
        try:
            # Try to get period of report
            if hasattr(filing, "period_of_report"):
                parsed = _coerce_date(filing.period_of_report)
                if parsed is not None:
                    return parsed

            if hasattr(filing, "report_date"):
                parsed = _coerce_date(filing.report_date)
                if parsed is not None:
                    return parsed

            # Fallback: try filing date
            if hasattr(filing, "filing_date"):
                parsed = _coerce_date(filing.filing_date)
                if parsed is not None:
                    return parsed

            return None

        except Exception as e:
            logger.debug("Failed to extract period end: %s", e)
            return None

    def _extract_filing_date(self, filing: Any) -> date | None:
        """Extract filing date from filing."""
        try:
            if hasattr(filing, "filing_date"):
                parsed = _coerce_date(filing.filing_date)
                if parsed is not None:
                    return parsed
            if hasattr(filing, "acceptance_datetime"):
                parsed = _coerce_date(filing.acceptance_datetime)
                if parsed is not None:
                    return parsed
            return None

        except Exception as e:
            logger.debug("Failed to extract filing date: %s", e)
            return None

    def _extract_fiscal_period(self, filing: Any) -> tuple[int, int]:
        """Extract fiscal year and quarter from filing."""
        try:
            # Try to get fiscal year and period
            fiscal_year = 0
            fiscal_quarter = 0

            if hasattr(filing, "fiscal_year_end"):
                fiscal_year_raw = filing.fiscal_year_end
                if fiscal_year_raw:
                    try:
                        candidate = int(fiscal_year_raw)
                        if 1900 <= candidate <= 2100:
                            fiscal_year = candidate
                    except (TypeError, ValueError):
                        fiscal_year = 0

            if hasattr(filing, "fiscal_period"):
                period = filing.fiscal_period
                if period and period.startswith("Q"):
                    fiscal_quarter = int(period[1])  # "Q1" -> 1
                elif period == "FY":
                    fiscal_quarter = 4  # Full year = Q4

            if fiscal_year > 0 and fiscal_quarter > 0:
                return fiscal_year, fiscal_quarter

            fallback_date = self._extract_period_end(filing) or self._extract_filing_date(filing)
            if fallback_date is None:
                return 0, 0
            inferred_year = fallback_date.year
            inferred_quarter = ((fallback_date.month - 1) // 3) + 1
            return inferred_year, inferred_quarter

        except Exception as e:
            logger.debug("Failed to extract fiscal period: %s", e)
            return 0, 0


__all__ = [
    "EarningsActual",
    "EdgarFetcher",
]
