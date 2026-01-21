#!/usr/bin/env python3
"""
Unit tests for EDGAR fetcher parsing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime

import pytest

pytest.importorskip("pandas")

import pandas as pd

from ml.features.earnings.ingestion.edgar_fetcher import EdgarFetcher


@dataclass
class _StubXbrl:
    facts: dict[str, object]


@dataclass
class _StubFiling:
    period_of_report: object | None = None
    filing_date: object | None = None
    report_date: object | None = None
    fiscal_year_end: object | None = None
    fiscal_period: object | None = None
    acceptance_datetime: object | None = None
    _facts: dict[str, object] | None = None

    def xbrl(self) -> _StubXbrl | None:
        if self._facts is None:
            return None
        return _StubXbrl(self._facts)


def test_parse_filing_accepts_date_objects() -> None:
    fetcher = EdgarFetcher(rate_limit_delay=0.0)
    filing = _StubFiling(
        period_of_report=date(2025, 6, 28),
        filing_date=datetime(2025, 8, 1, tzinfo=UTC),
        fiscal_period=None,
        fiscal_year_end=None,
        _facts={
            "us-gaap:EarningsPerShareDiluted": 2.5,
            "us-gaap:NetIncomeLoss": 1_000.0,
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 1000,
        },
    )

    actual = fetcher._parse_filing("AAPL", filing, "10-Q")

    assert actual is not None
    assert actual.period_end == date(2025, 6, 28)
    assert actual.filing_date == date(2025, 8, 1)
    assert actual.fiscal_quarter == 2
    assert actual.fiscal_year == 2025


def test_extract_facts_view_latest_period() -> None:
    fetcher = EdgarFetcher(rate_limit_delay=0.0)
    frame = pd.DataFrame(
        [
            {"numeric_value": 1.5, "period_end": "2024-03-30"},
            {"numeric_value": 2.0, "period_end": "2024-06-29"},
        ],
    )

    class _FactsView:
        def get_facts_by_concept(self, _: str) -> pd.DataFrame:
            return frame

    values = fetcher._extract_facts_view(_FactsView())

    assert values["us-gaap:EarningsPerShareDiluted"] == 2.0
