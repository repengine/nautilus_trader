#!/usr/bin/env python3
"""
Unit tests for YahooFetcher.

Tests cover:
- Fetching consensus estimates from Yahoo Finance
- Extracting earnings calendar
- Rate limiting
- Error handling for invalid tickers
- Edge cases (missing estimates, no analyst coverage)

"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml._imports import HAS_YFINANCE
from ml.data.earnings.yahoo_fetcher import EarningsConsensus
from ml.data.earnings.yahoo_fetcher import YahooFetcher
from ml.data.earnings.yahoo_fetcher import set_yfinance_override


@contextmanager
def _yfinance_override(mock: Mock) -> Iterator[None]:
    set_yfinance_override(mock)
    try:
        yield
    finally:
        set_yfinance_override(None)


@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
class TestYahooFetcher:
    """
    Test suite for YahooFetcher.
    """

    def test_initialization(self) -> None:
        """
        Test YahooFetcher initialization.
        """
        fetcher = YahooFetcher(rate_limit_delay=0.3, max_retries=2)

        assert fetcher.rate_limit_delay == 0.3
        assert fetcher.max_retries == 2
        assert fetcher.last_request_time == 0.0

    def test_fetch_consensus_success(self) -> None:
        """
        Test successful consensus fetch.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            # Mock ticker
            mock_ticker = MagicMock()

            # Mock earnings dates DataFrame
            import pandas as pd

            earnings_dates_df = pd.DataFrame(
                {
                    "EPS Estimate": [2.10, 2.05],
                },
                index=[
                    datetime(2025, 1, 30, 16, 0),
                    datetime(2025, 4, 30, 16, 0),
                ],
            )
            mock_ticker.earnings_dates = earnings_dates_df

            # Mock analyst price target
            mock_ticker.analyst_price_target = {
                "numberOfAnalystOpinions": 42,
            }

            mock_yfinance.Ticker.return_value = mock_ticker

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("AAPL")

                assert consensus is not None
                assert consensus.ticker == "AAPL"
                assert consensus.next_earnings_date == datetime(2025, 1, 30, 16, 0)
                assert consensus.eps_estimate == 2.10
                assert consensus.num_analysts == 42

    def test_fetch_consensus_invalid_ticker(self) -> None:
        """
        Test graceful handling of invalid ticker.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            # Mock ticker not found
            mock_yfinance.Ticker.side_effect = Exception("Ticker not found")

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("INVALID_TICKER_XYZ")

                assert consensus is None  # Should return None, not raise

    def test_fetch_consensus_no_estimates(self) -> None:
        """
        Test handling of ticker with no analyst estimates.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            # Mock ticker without estimates
            mock_ticker = MagicMock()
            mock_ticker.earnings_dates = None
            mock_ticker.analyst_price_target = None
            mock_ticker.info = {}

            mock_yfinance.Ticker.return_value = mock_ticker

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("SMALLCAP")

                # Should still return consensus object with None values
                assert consensus is not None
                assert consensus.ticker == "SMALLCAP"
                assert consensus.next_earnings_date is None
                assert consensus.eps_estimate is None
                assert consensus.num_analysts == 0

    def test_fetch_consensus_calendar_fallback(self) -> None:
        """
        Test fallback to calendar property for earnings date.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            # Mock ticker with calendar property
            mock_ticker = MagicMock()
            mock_ticker.earnings_dates = None  # No earnings_dates

            # Mock calendar
            mock_ticker.calendar = {
                "Earnings Date": datetime(2025, 1, 30),
            }

            mock_yfinance.Ticker.return_value = mock_ticker

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("AAPL")

                assert consensus is not None
                assert consensus.next_earnings_date == datetime(2025, 1, 30)

    def test_fetch_consensus_info_fallback(self) -> None:
        """
        Test fallback to info dict for EPS estimate.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            # Mock ticker with info
            mock_ticker = MagicMock()
            mock_ticker.earnings_dates = None

            # Mock info dict
            mock_ticker.info = {
                "forwardEps": 2.15,
                "symbol": "AAPL",
            }

            mock_yfinance.Ticker.return_value = mock_ticker

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("AAPL")

                assert consensus is not None
                assert consensus.eps_estimate == 2.15

    def test_earnings_consensus_dataclass(self) -> None:
        """
        Test EarningsConsensus dataclass creation.
        """
        consensus = EarningsConsensus(
            ticker="AAPL",
            next_earnings_date=datetime(2025, 1, 30, 16, 0),
            eps_estimate=2.10,
            revenue_estimate=125.0e9,
            num_analysts=42,
            estimate_date=datetime(2025, 1, 15),
        )

        assert consensus.ticker == "AAPL"
        assert consensus.eps_estimate == 2.10
        assert consensus.num_analysts == 42
        assert consensus.next_earnings_date.year == 2025

    def test_rate_limiting(self) -> None:
        """
        Test rate limiting between requests.
        """
        import time

        fetcher = YahooFetcher(rate_limit_delay=0.1)

        # Simulate first request
        fetcher.last_request_time = time.perf_counter() - 0.05

        # Apply rate limit (should sleep ~0.05s)
        start = time.perf_counter()
        fetcher._apply_rate_limit()
        elapsed = time.perf_counter() - start

        # Should have slept approximately 0.05s (with tolerance)
        assert 0.03 < elapsed < 0.15

    def test_fetch_consensus_nan_handling(self) -> None:
        """
        Test handling of NaN values in estimates.
        """
        with patch("ml.data.earnings.yahoo_fetcher.yfinance") as mock_yfinance:
            import numpy as np
            import pandas as pd

            # Mock ticker with NaN estimate
            mock_ticker = MagicMock()

            earnings_dates_df = pd.DataFrame(
                {
                    "EPS Estimate": [np.nan, 2.05],
                },
                index=[
                    datetime(2025, 1, 30, 16, 0),
                    datetime(2025, 4, 30, 16, 0),
                ],
            )
            mock_ticker.earnings_dates = earnings_dates_df

            mock_yfinance.Ticker.return_value = mock_ticker

            with _yfinance_override(mock_yfinance):
                # Test
                fetcher = YahooFetcher(rate_limit_delay=0.0)
                consensus = fetcher.fetch_consensus("AAPL")

                # Should handle NaN gracefully
                assert consensus is not None
                # eps_estimate should be None when NaN
                assert consensus.eps_estimate is None or consensus.eps_estimate == 2.05
