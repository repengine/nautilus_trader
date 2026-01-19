#!/usr/bin/env python3
"""
Service orchestrating earnings ingestion from EDGAR and Yahoo Finance.
"""

from __future__ import annotations

import calendar
import logging
import time
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from typing import Protocol

from ml._imports import HAS_EDGARTOOLS
from ml._imports import load_edgartools
from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.features.earnings.ingestion.edgar_fetcher import EarningsActual
from ml.features.earnings.ingestion.edgar_fetcher import EdgarFetcher
from ml.features.earnings.ingestion.universe import ResolvedUniverse
from ml.features.earnings.ingestion.universe import resolve_ingestion_universe
from ml.features.earnings.ingestion.yahoo_fetcher import EarningsConsensus
from ml.features.earnings.ingestion.yahoo_fetcher import YahooFetcher


logger = logging.getLogger(__name__)


class _EarningsWriterProtocol(Protocol):
    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = ...,
        net_income: float | None = ...,
        operating_income: float | None = ...,
        shares_outstanding: int | None = ...,
        filing_type: str | None = ...,
        fiscal_year: int | None = ...,
        fiscal_quarter: int | None = ...,
        source: str = ...,
        run_id: str | None = ...,
    ) -> object:
        ...

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = ...,
        num_analysts: int | None = ...,
        source: str = ...,
        run_id: str | None = ...,
    ) -> object:
        ...


@dataclass(frozen=True)
class EarningsIngestionResult:
    """Summary of an ingestion run."""

    universe: ResolvedUniverse
    tickers_attempted: int
    actuals_written: int
    estimates_written: int
    skipped_actuals: dict[str, str]
    failures: dict[str, str]
    duration_seconds: float


class EarningsIngestionService:
    """
    Coordinate earnings ingestion for a resolved universe.
    """

    def __init__(
        self,
        *,
        config: EarningsIngestionConfig,
        writer: _EarningsWriterProtocol,
        edgar_fetcher: EdgarFetcher | None = None,
        yahoo_fetcher: YahooFetcher | None = None,
    ) -> None:
        self._config = config
        self._writer = writer
        self._edgar = edgar_fetcher or EdgarFetcher(
            rate_limit_delay=config.edgar_rate_limit,
            max_retries=config.edgar_max_retries,
        )
        self._yahoo = yahoo_fetcher or YahooFetcher(
            rate_limit_delay=config.yahoo_rate_limit,
            max_retries=config.yahoo_max_retries,
        )
        if config.sec_identity and HAS_EDGARTOOLS:
            try:
                tools = load_edgartools()
                tools.set_identity(config.sec_identity)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to set SEC identity: %s", exc, exc_info=True)

    def run(self) -> EarningsIngestionResult:
        start = time.perf_counter()
        universe = resolve_ingestion_universe(self._config)
        skip_set = set(self._config.normalized_skip_set())
        actuals_written = 0
        estimates_written = 0
        skipped_actuals: dict[str, str] = {}
        failures: dict[str, str] = {}

        for ticker in universe.tickers:
            try:
                actuals_written += self._ingest_actuals(ticker, skip_set, skipped_actuals)
            except Exception as exc:  # pragma: no cover - defensive safety
                failures[ticker] = f"actuals:{exc}"
                logger.error("Failed to ingest EDGAR data for %s: %s", ticker, exc, exc_info=True)

            if not self._config.enable_yahoo:
                continue
            try:
                estimates_written += self._ingest_estimate(ticker)
            except Exception as exc:  # pragma: no cover - defensive safety
                failures[ticker] = f"estimates:{exc}"
                logger.error("Failed to ingest Yahoo consensus for %s: %s", ticker, exc, exc_info=True)

        duration = time.perf_counter() - start
        return EarningsIngestionResult(
            universe=universe,
            tickers_attempted=len(universe.tickers),
            actuals_written=actuals_written,
            estimates_written=estimates_written,
            skipped_actuals=skipped_actuals,
            failures=failures,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#

    def _ingest_actuals(
        self,
        ticker: str,
        skip_set: set[str],
        skipped_actuals: dict[str, str],
    ) -> int:
        if ticker in skip_set:
            skipped_actuals[ticker] = "skip_list"
            return 0

        raw_actuals = self._edgar.fetch_earnings(ticker, quarters=self._config.edgar_quarters, form="10-Q")
        raw_actuals.extend(self._edgar.fetch_earnings(ticker, quarters=max(1, self._config.edgar_quarters // 4), form="10-K"))
        deduped = self._deduplicate_actuals(raw_actuals)

        written = 0
        for actual in deduped:
            ts_event = _date_to_ns(actual.filing_date)
            ts_init = time.time_ns()
            self._writer.write_earnings_actual(
                ticker=actual.ticker,
                period_end=str(actual.period_end),
                filing_date=str(actual.filing_date),
                eps_diluted=actual.eps_diluted,
                revenue=actual.revenue,
                ts_event=ts_event,
                ts_init=ts_init,
                eps_basic=actual.eps_basic,
                net_income=actual.net_income,
                operating_income=actual.operating_income,
                shares_outstanding=actual.shares_outstanding,
                filing_type=actual.filing_type,
                fiscal_year=actual.fiscal_year,
                fiscal_quarter=actual.fiscal_quarter,
                source="historical",
            )
            written += 1
        if written == 0:
            skipped_actuals[ticker] = "no_filings"
        return written

    def _ingest_estimate(self, ticker: str) -> int:
        consensus = self._yahoo.fetch_consensus(ticker)
        if consensus is None:
            return 0

        period_end = self._infer_period_end(consensus)
        if period_end is None:
            logger.debug("Skipping Yahoo consensus for %s due to missing period_end", ticker)
            return 0

        ts_event = _datetime_to_ns(consensus.estimate_date)
        ts_init = time.time_ns()
        self._writer.write_earnings_estimate(
            ticker=consensus.ticker,
            estimate_date=consensus.estimate_date.date().isoformat(),
            period_end=period_end.isoformat(),
            eps_consensus=consensus.eps_estimate,
            ts_event=ts_event,
            ts_init=ts_init,
            revenue_consensus=consensus.revenue_estimate,
            num_analysts=consensus.num_analysts,
            source="historical",
        )
        return 1

    @staticmethod
    def _deduplicate_actuals(actuals: list[EarningsActual]) -> list[EarningsActual]:
        deduped: dict[tuple[date, str], EarningsActual] = {}
        for actual in actuals:
            key = (actual.period_end, actual.filing_type)
            existing = deduped.get(key)
            if existing is None or actual.filing_date >= existing.filing_date:
                deduped[key] = actual
        deduped_list = list(deduped.values())
        deduped_list.sort(key=lambda item: item.period_end, reverse=True)
        return deduped_list

    @staticmethod
    def _infer_period_end(consensus: EarningsConsensus) -> date | None:
        earnings_date = consensus.next_earnings_date
        if earnings_date is None:
            return None
        base = earnings_date.date()
        quarter_index = (base.month - 1) // 3
        end_month = quarter_index * 3
        year = base.year
        if end_month == 0:
            end_month = 12
            year -= 1
        day = calendar.monthrange(year, end_month)[1]
        return date(year, end_month, day)


def _datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000_000)


def _date_to_ns(value: date) -> int:
    dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


__all__ = ["EarningsIngestionResult", "EarningsIngestionService"]
