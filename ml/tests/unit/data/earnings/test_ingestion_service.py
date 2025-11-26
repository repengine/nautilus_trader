#!/usr/bin/env python3

"""
Unit tests for the earnings ingestion service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.data.earnings.edgar_fetcher import EarningsActual
from ml.data.earnings.ingestion_service import EarningsIngestionService
from ml.data.earnings.yahoo_fetcher import EarningsConsensus


@dataclass
class _Recorder:
    actuals: list[dict[str, Any]]
    estimates: list[dict[str, Any]]

    def write_earnings_actual(self, **kwargs: Any) -> None:
        self.actuals.append(kwargs)

    def write_earnings_estimate(self, **kwargs: Any) -> None:
        self.estimates.append(kwargs)


class _StubEdgar:
    def __init__(self, actuals: list[EarningsActual]) -> None:
        self._actuals = actuals

    def fetch_earnings(self, ticker: str, quarters: int = 4, form: str = "10-Q") -> list[EarningsActual]:
        return [actual for actual in self._actuals if actual.ticker == ticker]


class _StubYahoo:
    def __init__(self, consensus: dict[str, EarningsConsensus]) -> None:
        self._consensus = consensus

    def fetch_consensus(self, ticker: str) -> EarningsConsensus | None:
        return self._consensus.get(ticker)


def test_ingestion_service_writes_actuals_and_estimates() -> None:
    recorder = _Recorder(actuals=[], estimates=[])

    actual = EarningsActual(
        ticker="AAPL",
        period_end=date(2024, 3, 31),
        filing_date=date(2024, 5, 1),
        eps_basic=1.50,
        eps_diluted=1.48,
        revenue=90000000000.0,
        net_income=24000000000.0,
        operating_income=None,
        shares_outstanding=16000000000,
        filing_type="10-Q",
        fiscal_year=2024,
        fiscal_quarter=1,
    )
    consensus = EarningsConsensus(
        ticker="AAPL",
        next_earnings_date=datetime(2024, 7, 25, 21, 30, tzinfo=UTC),
        eps_estimate=1.52,
        revenue_estimate=91000000000.0,
        num_analysts=32,
        estimate_date=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
    )

    service = EarningsIngestionService(
        config=EarningsIngestionConfig(
            postgres_dsn="postgresql://unused",
            override_symbols=("AAPL",),
            enable_yahoo=True,
        ),
        writer=recorder,
        edgar_fetcher=_StubEdgar([actual]),
        yahoo_fetcher=_StubYahoo({"AAPL": consensus}),
    )

    result = service.run()
    assert result.actuals_written == 1
    assert result.estimates_written == 1
    assert not result.failures
    assert recorder.actuals and recorder.estimates


def test_ingestion_service_respects_skip_list() -> None:
    recorder = _Recorder(actuals=[], estimates=[])
    service = EarningsIngestionService(
        config=EarningsIngestionConfig(
            postgres_dsn="postgresql://unused",
            override_symbols=("SPY",),
            skip_actuals=("SPY",),
            enable_yahoo=False,
        ),
        writer=recorder,
        edgar_fetcher=_StubEdgar([]),
        yahoo_fetcher=_StubYahoo({}),
    )
    result = service.run()
    assert result.actuals_written == 0
    assert result.skipped_actuals.get("SPY") == "skip_list"
