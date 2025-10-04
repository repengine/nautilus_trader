#!/usr/bin/env python3
"""
Unit tests for EdgarFetcher.

Tests cover:
- Fetching earnings data from EDGAR
- Parsing XBRL from filings
- Rate limiting
- Error handling for invalid tickers
- Edge cases (missing data, non-standard XBRL)

"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml._imports import HAS_EDGARTOOLS
from ml.data.earnings.edgar_fetcher import EarningsActual
from ml.data.earnings.edgar_fetcher import EdgarFetcher


@pytest.mark.skipif(not HAS_EDGARTOOLS, reason="edgartools not installed")
class TestEdgarFetcher:
    """Test suite for EdgarFetcher."""

    def test_initialization(self) -> None:
        """Test EdgarFetcher initialization."""
        fetcher = EdgarFetcher(rate_limit_delay=0.5, max_retries=2)

        assert fetcher.rate_limit_delay == 0.5
        assert fetcher.max_retries == 2
        assert fetcher.last_request_time == 0.0

    @patch("ml.data.earnings.edgar_fetcher.edgartools")
    def test_fetch_earnings_success(self, mock_edgartools: Mock) -> None:
        """Test successful earnings fetch."""
        # Mock company
        mock_company = MagicMock()
        mock_edgartools.Company.return_value = mock_company

        # Mock filing
        mock_filing = MagicMock()
        mock_filing.period_of_report = "2024-09-30"
        mock_filing.filing_date = "2024-10-31"
        mock_filing.fiscal_year_end = "2024"
        mock_filing.fiscal_period = "Q4"

        # Mock XBRL data
        mock_xbrl = MagicMock()
        mock_xbrl.facts = {
            "us-gaap:EarningsPerShareDiluted": 2.52,
            "us-gaap:EarningsPerShareBasic": 2.55,
            "us-gaap:Revenues": 94.9e9,
            "us-gaap:NetIncomeLoss": 22.9e9,
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 15000000000,
        }
        mock_filing.xbrl.return_value = mock_xbrl

        # Mock filings list
        mock_filings = MagicMock()
        mock_filings.latest.return_value = [mock_filing]
        mock_company.get_filings.return_value = mock_filings

        # Test
        fetcher = EdgarFetcher(rate_limit_delay=0.0)
        actuals = fetcher.fetch_earnings("AAPL", quarters=1)

        assert len(actuals) == 1
        actual = actuals[0]
        assert actual.ticker == "AAPL"
        assert actual.eps_diluted == 2.52
        assert actual.eps_basic == 2.55
        assert actual.revenue == 94.9e9
        assert actual.net_income == 22.9e9
        assert actual.shares_outstanding == 15000000000
        assert actual.filing_type == "10-Q"

    @patch("ml.data.earnings.edgar_fetcher.edgartools")
    def test_fetch_earnings_invalid_ticker(self, mock_edgartools: Mock) -> None:
        """Test graceful handling of invalid ticker."""
        # Mock company not found
        mock_edgartools.Company.side_effect = Exception("Company not found")

        # Test
        fetcher = EdgarFetcher(rate_limit_delay=0.0)
        actuals = fetcher.fetch_earnings("INVALID_TICKER_XYZ", quarters=1)

        assert actuals == []  # Should return empty list, not raise

    @patch("ml.data.earnings.edgar_fetcher.edgartools")
    def test_fetch_earnings_no_filings(self, mock_edgartools: Mock) -> None:
        """Test handling of ticker with no filings."""
        # Mock company with no filings
        mock_company = MagicMock()
        mock_edgartools.Company.return_value = mock_company

        mock_filings = MagicMock()
        mock_filings.latest.return_value = []
        mock_company.get_filings.return_value = mock_filings

        # Test
        fetcher = EdgarFetcher(rate_limit_delay=0.0)
        actuals = fetcher.fetch_earnings("NEWCO", quarters=1)

        assert actuals == []

    @patch("ml.data.earnings.edgar_fetcher.edgartools")
    def test_fetch_earnings_missing_xbrl(self, mock_edgartools: Mock) -> None:
        """Test handling of filing with missing XBRL data."""
        # Mock company
        mock_company = MagicMock()
        mock_edgartools.Company.return_value = mock_company

        # Mock filing without XBRL
        mock_filing = MagicMock()
        mock_filing.period_of_report = "2024-09-30"
        mock_filing.filing_date = "2024-10-31"
        mock_filing.fiscal_year_end = "2024"
        mock_filing.fiscal_period = "Q4"
        mock_filing.xbrl.return_value = None  # No XBRL

        mock_filings = MagicMock()
        mock_filings.latest.return_value = [mock_filing]
        mock_company.get_filings.return_value = mock_filings

        # Test
        fetcher = EdgarFetcher(rate_limit_delay=0.0)
        actuals = fetcher.fetch_earnings("OLDCO", quarters=1)

        assert actuals == []  # Should skip filing without XBRL

    @patch("ml.data.earnings.edgar_fetcher.edgartools")
    def test_fetch_earnings_multiple_quarters(self, mock_edgartools: Mock) -> None:
        """Test fetching multiple quarters."""
        # Mock company
        mock_company = MagicMock()
        mock_edgartools.Company.return_value = mock_company

        # Mock multiple filings
        filings = []
        for i, quarter in enumerate(["Q4", "Q3", "Q2", "Q1"]):
            mock_filing = MagicMock()
            mock_filing.period_of_report = f"2024-{9-i*3:02d}-30"
            mock_filing.filing_date = f"2024-{10-i*3:02d}-31"
            mock_filing.fiscal_year_end = "2024"
            mock_filing.fiscal_period = quarter

            mock_xbrl = MagicMock()
            mock_xbrl.facts = {
                "us-gaap:EarningsPerShareDiluted": 2.5 - i * 0.1,
                "us-gaap:Revenues": 90.0e9 + i * 1.0e9,
            }
            mock_filing.xbrl.return_value = mock_xbrl
            filings.append(mock_filing)

        mock_filings = MagicMock()
        mock_filings.latest.return_value = filings
        mock_company.get_filings.return_value = mock_filings

        # Test
        fetcher = EdgarFetcher(rate_limit_delay=0.0)
        actuals = fetcher.fetch_earnings("AAPL", quarters=4)

        assert len(actuals) == 4
        # Verify ordering (should be newest first based on input)
        assert actuals[0].fiscal_period == "Q4"
        assert actuals[1].fiscal_period == "Q3"
        assert actuals[2].fiscal_period == "Q2"
        assert actuals[3].fiscal_period == "Q1"

    def test_earnings_actual_dataclass(self) -> None:
        """Test EarningsActual dataclass creation."""
        actual = EarningsActual(
            ticker="AAPL",
            period_end=date(2024, 9, 30),
            filing_date=date(2024, 10, 31),
            eps_basic=2.55,
            eps_diluted=2.52,
            revenue=94.9e9,
            net_income=22.9e9,
            operating_income=None,
            shares_outstanding=15000000000,
            filing_type="10-Q",
            fiscal_year=2024,
            fiscal_quarter=4,
        )

        assert actual.ticker == "AAPL"
        assert actual.eps_diluted == 2.52
        assert actual.revenue == 94.9e9
        assert actual.filing_type == "10-Q"
        assert actual.fiscal_quarter == 4

    def test_rate_limiting(self) -> None:
        """Test rate limiting between requests."""
        import time

        fetcher = EdgarFetcher(rate_limit_delay=0.1)

        # Simulate first request
        fetcher.last_request_time = time.perf_counter() - 0.05

        # Apply rate limit (should sleep ~0.05s)
        start = time.perf_counter()
        fetcher._apply_rate_limit()
        elapsed = time.perf_counter() - start

        # Should have slept approximately 0.05s (with tolerance)
        assert 0.03 < elapsed < 0.15
