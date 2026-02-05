#!/usr/bin/env python3

"""
Unit tests for the earnings ingestion service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import patch

from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.config.events import Source
from ml.config import WatermarkWindowConfig
from ml.data.earnings.edgar_fetcher import EarningsActual
from ml.data.earnings.ingestion_service import EarningsIngestionService
from ml.data.earnings.yahoo_fetcher import EarningsConsensus
from ml.registry.watermark import Watermark


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
        self.calls: list[dict[str, Any]] = []

    def fetch_earnings(self, ticker: str, quarters: int = 4, form: str = "10-Q") -> list[EarningsActual]:
        self.calls.append(
            {
                "ticker": ticker,
                "quarters": quarters,
                "form": form,
            },
        )
        return [actual for actual in self._actuals if actual.ticker == ticker]


class _StubYahoo:
    def __init__(self, consensus: dict[str, EarningsConsensus]) -> None:
        self._consensus = consensus

    def fetch_consensus(self, ticker: str) -> EarningsConsensus | None:
        return self._consensus.get(ticker)

    def fetch_estimates(self, ticker: str) -> list[EarningsConsensus]:
        consensus = self._consensus.get(ticker)
        return [consensus] if consensus is not None else []


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


def test_ingestion_service_aligns_estimate_period_end() -> None:
    recorder = _Recorder(actuals=[], estimates=[])

    actual = EarningsActual(
        ticker="AAPL",
        period_end=date(2024, 6, 28),
        filing_date=date(2024, 8, 1),
        eps_basic=1.50,
        eps_diluted=1.48,
        revenue=90000000000.0,
        net_income=24000000000.0,
        operating_income=None,
        shares_outstanding=16000000000,
        filing_type="10-Q",
        fiscal_year=2024,
        fiscal_quarter=3,
    )
    consensus = EarningsConsensus(
        ticker="AAPL",
        next_earnings_date=datetime(2024, 7, 31, 16, 0, tzinfo=UTC),
        eps_estimate=1.52,
        revenue_estimate=91000000000.0,
        num_analysts=32,
        estimate_date=datetime(2024, 7, 31, 16, 0, tzinfo=UTC),
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
    assert result.estimates_written == 1
    assert recorder.estimates
    assert recorder.estimates[0]["period_end"] == "2024-06-28"


def test_ingestion_service_uses_watermark_for_actuals_window() -> None:
    watermark_date = date(2024, 5, 1)
    watermark_ns = int(datetime(2024, 5, 1, tzinfo=UTC).timestamp() * 1_000_000_000)

    class _Registry:
        def get_watermark(
            self,
            dataset_id: str,
            instrument_id: str,
            source: Source | str,
        ) -> Watermark | None:
            return Watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=str(source),
                last_success_ns=watermark_ns,
                last_attempt_ns=watermark_ns,
                last_count=1,
                completeness_pct=100.0,
                updated_at=0.0,
            )

    class _WriterWithRegistry(_Recorder):
        registry = _Registry()

    recorder = _WriterWithRegistry(actuals=[], estimates=[])
    actual_before = EarningsActual(
        ticker="AAPL",
        period_end=date(2024, 3, 31),
        filing_date=date(2024, 4, 15),
        eps_basic=1.0,
        eps_diluted=1.0,
        revenue=1.0,
        net_income=1.0,
        operating_income=None,
        shares_outstanding=1,
        filing_type="10-Q",
        fiscal_year=2024,
        fiscal_quarter=1,
    )
    actual_after = EarningsActual(
        ticker="AAPL",
        period_end=date(2024, 6, 30),
        filing_date=date(2024, 5, 15),
        eps_basic=2.0,
        eps_diluted=2.0,
        revenue=2.0,
        net_income=2.0,
        operating_income=None,
        shares_outstanding=2,
        filing_type="10-Q",
        fiscal_year=2024,
        fiscal_quarter=2,
    )
    stub_edgar = _StubEdgar([actual_before, actual_after])

    service = EarningsIngestionService(
        config=EarningsIngestionConfig(
            postgres_dsn="postgresql://unused",
            override_symbols=("AAPL",),
            enable_yahoo=False,
            watermark_config=WatermarkWindowConfig(
                use_watermark=True,
                lookback_days=0,
                max_window_days=None,
                fallback_start_days=None,
            ),
            edgar_quarter_days=10_000,
            edgar_min_quarters=1,
            edgar_max_quarters=8,
        ),
        writer=recorder,
        edgar_fetcher=stub_edgar,
        yahoo_fetcher=_StubYahoo({}),
    )

    result = service.run()

    assert result.actuals_written == 1
    assert recorder.actuals[0]["filing_date"] == str(actual_after.filing_date)
    assert stub_edgar.calls
    assert all(call["quarters"] == 1 for call in stub_edgar.calls)
    assert recorder.actuals
    assert actual_after.filing_date >= watermark_date


def test_ingestion_service_clamps_future_estimate_ts_event_to_init_time() -> None:
    recorder = _Recorder(actuals=[], estimates=[])

    actual = EarningsActual(
        ticker="AAPL",
        period_end=date(2024, 6, 28),
        filing_date=date(2024, 8, 1),
        eps_basic=1.50,
        eps_diluted=1.48,
        revenue=90000000000.0,
        net_income=24000000000.0,
        operating_income=None,
        shares_outstanding=16000000000,
        filing_type="10-Q",
        fiscal_year=2024,
        fiscal_quarter=3,
    )
    consensus = EarningsConsensus(
        ticker="AAPL",
        next_earnings_date=datetime(2024, 7, 31, 16, 0, tzinfo=UTC),
        eps_estimate=1.52,
        revenue_estimate=91000000000.0,
        num_analysts=32,
        estimate_date=datetime(2030, 1, 1, 12, 0, tzinfo=UTC),
    )
    fixed_now = int(datetime(2024, 9, 1, 12, 0, tzinfo=UTC).timestamp() * 1_000_000_000)

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

    with patch("ml.features.earnings.ingestion.service.time.time_ns", return_value=fixed_now):
        result = service.run()

    assert result.estimates_written == 1
    assert recorder.estimates
    assert recorder.estimates[0]["ts_event"] == fixed_now
    assert recorder.estimates[0]["ts_init"] == fixed_now
