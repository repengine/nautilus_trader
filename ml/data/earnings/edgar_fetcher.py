"""Compatibility shim for earnings Edgar fetcher."""

from __future__ import annotations

import calendar
import logging
import warnings
from datetime import date
from datetime import datetime
from types import ModuleType
from typing import cast

from ml._imports import HAS_EDGARTOOLS
from ml._imports import check_ml_dependencies
from ml._imports import load_edgartools
from ml.features.earnings.ingestion.edgar_fetcher import EarningsActual
from ml.features.earnings.ingestion.edgar_fetcher import EdgarFetcher as _BaseEdgarFetcher


warnings.warn(
    "ml.data.earnings.edgar_fetcher is deprecated; "
    "import from ml.features.earnings.ingestion.edgar_fetcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)
edgartools: ModuleType | None = None


def _resolve_edgartools() -> ModuleType | None:
    """Return the edgartools module, loading it on demand."""
    global edgartools
    if edgartools is not None:
        return edgartools
    if not HAS_EDGARTOOLS:
        return None
    edgartools = load_edgartools()
    return edgartools


def _coerce_date(value: object | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    token = str(value).strip()
    if not token:
        return None
    try:
        return datetime.strptime(token, "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(token).date()
        except ValueError:
            parts = token.split("-")
            if len(parts) != 3:
                return None
            try:
                year = int(parts[0])
                month = max(1, min(12, int(parts[1])))
                day = int(parts[2])
                last_day = calendar.monthrange(year, month)[1]
                day = max(1, min(last_day, day))
                return date(year, month, day)
            except Exception:
                return None


class EdgarFetcher(_BaseEdgarFetcher):
    """
    Deprecated shim for the earnings Edgar fetcher.

    Exists to preserve legacy patch points (tests patch `ml.data.earnings.edgar_fetcher.edgartools`).
    """

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        if not HAS_EDGARTOOLS:
            check_ml_dependencies(["edgartools"])
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_request_time = 0.0

    def _fetch_company(self, ticker: str) -> object | None:
        try:
            tools = _resolve_edgartools()
            if tools is None:
                return None
            return cast(object, tools.Company(ticker))
        except Exception as exc:
            logger.debug("Failed to fetch company %s: %s", ticker, exc, exc_info=True)
            return None

    def _extract_period_end(self, filing: object) -> date | None:
        period_raw = getattr(filing, "period_of_report", None)
        parsed = _coerce_date(period_raw)
        if parsed is not None:
            return parsed
        filing_raw = getattr(filing, "filing_date", None)
        return _coerce_date(filing_raw)

    def _extract_filing_date(self, filing: object) -> date | None:
        filing_raw = getattr(filing, "filing_date", None)
        return _coerce_date(filing_raw)

    def _extract_fiscal_period(self, filing: object) -> tuple[int, int]:
        fiscal_year = 0
        fiscal_quarter = 0

        fiscal_year_raw = getattr(filing, "fiscal_year_end", None)
        if fiscal_year_raw is not None and str(fiscal_year_raw).strip():
            try:
                fiscal_year = int(fiscal_year_raw)
            except (TypeError, ValueError):
                fiscal_year = 0

        fiscal_period_raw = getattr(filing, "fiscal_period", None)
        if fiscal_period_raw is not None and str(fiscal_period_raw).strip():
            period = str(fiscal_period_raw).strip()
            if period.startswith("Q") and len(period) >= 2 and period[1].isdigit():
                fiscal_quarter = int(period[1])
            elif period.upper() == "FY":
                fiscal_quarter = 4

        if fiscal_year > 0 and fiscal_quarter > 0:
            return fiscal_year, fiscal_quarter

        fallback_date = self._extract_period_end(filing) or self._extract_filing_date(filing)
        if fallback_date is None:
            return 0, 0
        inferred_year = fallback_date.year
        inferred_quarter = ((fallback_date.month - 1) // 3) + 1
        return inferred_year, inferred_quarter


__all__ = ["EarningsActual", "EdgarFetcher", "edgartools"]
