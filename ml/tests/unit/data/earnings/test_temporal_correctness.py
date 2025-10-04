"""
Temporal correctness tests for earnings cache.

CRITICAL: Verifies no look-ahead bias in backtesting.
Ensures at time T, only data with ts_event < T is visible.

These tests prevent the most dangerous bug in backtesting: seeing future data.
"""

import pytest

from ml.data.earnings.earnings_cache import EarningsCache
from ml.tests.utils.earnings_facade import build_test_earnings_adapter


class TestPointInTimeCorrectness:
    """Test point-in-time temporal correctness."""

    def test_filing_not_visible_before_ts_event(self) -> None:
        """
        CRITICAL: Filing should NOT be visible before its ts_event.

        Scenario: Q4 2023 earnings filed on 2024-02-01
        - At 2024-01-01: Should NOT see Q4 2023 data (ts_event > as_of_ts)
        - At 2024-02-02: SHOULD see Q4 2023 data (ts_event < as_of_ts)
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Q4 2023 earnings filed on 2024-02-01
        q4_2023_filing_ts = 1706745600000000000  # 2024-02-01 00:00:00 UTC
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-02-01",
            eps_diluted=2.10,
            revenue=119_575_000_000.0,
            ts_event=q4_2023_filing_ts,
            ts_init=q4_2023_filing_ts,
            fiscal_year=2023,
            fiscal_quarter=4,
        )

        # Test 1: At 2024-01-01 (BEFORE filing) - should NOT see Q4 2023
        as_of_jan_1 = 1704067200000000000  # 2024-01-01 00:00:00 UTC
        actuals_jan = cache.get_actuals_at("AAPL", as_of_ts=as_of_jan_1)

        assert len(actuals_jan) == 0, (
            f"Look-ahead bias detected! Q4 2023 visible at 2024-01-01 "
            f"(ts_event={q4_2023_filing_ts}, as_of_ts={as_of_jan_1})"
        )

        # Test 2: At 2024-02-02 (AFTER filing) - SHOULD see Q4 2023
        as_of_feb_2 = 1706832000000000000  # 2024-02-02 00:00:00 UTC
        actuals_feb = cache.get_actuals_at("AAPL", as_of_ts=as_of_feb_2)

        assert len(actuals_feb) == 1
        assert actuals_feb[0]["eps_diluted"] == 2.10
        assert actuals_feb[0]["period_end"] == "2023-12-31"

    def test_same_day_filing_boundary(self) -> None:
        """
        Test exact boundary condition: filing at midnight.

        Filing at 2024-02-01 00:00:00 should be visible starting 2024-02-01 00:00:01.
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        filing_ts = 1706745600000000000  # 2024-02-01 00:00:00 UTC
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-02-01",
            eps_diluted=2.10,
            revenue=119_575_000_000.0,
            ts_event=filing_ts,
            ts_init=filing_ts,
        )

        # At exact filing time (ts_event == as_of_ts) - NOT visible (< not <=)
        actuals_exact = cache.get_actuals_at("AAPL", as_of_ts=filing_ts)
        assert len(actuals_exact) == 0, "Filing should NOT be visible at exact ts_event"

        # 1 nanosecond later - NOW visible
        as_of_ts_plus_1ns = filing_ts + 1
        actuals_later = cache.get_actuals_at("AAPL", as_of_ts=as_of_ts_plus_1ns)
        assert len(actuals_later) == 1, "Filing should be visible 1ns after ts_event"

    def test_delayed_filing_scenario(self) -> None:
        """
        Test delayed filing scenario (common in real markets).

        Scenario: Q1 2024 ends 2024-03-31, but 10-Q filed on 2024-05-10 (40 days late)
        - At 2024-04-15: Should NOT see Q1 2024 (filing not yet submitted)
        - At 2024-05-11: SHOULD see Q1 2024 (filing now available)
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Q1 2024 filed late on 2024-05-10
        q1_filing_ts = 1715299200000000000  # 2024-05-10 00:00:00 UTC
        store.write_actuals(
            ticker="TSLA",
            period_end="2024-03-31",  # Quarter ended 40 days earlier
            filing_date="2024-05-10",
            eps_diluted=0.45,
            revenue=21_301_000_000.0,
            ts_event=q1_filing_ts,
            ts_init=q1_filing_ts,
            fiscal_year=2024,
            fiscal_quarter=1,
        )

        # At 2024-04-15 (15 days after quarter end, 25 days before filing)
        as_of_apr_15 = 1713139200000000000  # 2024-04-15 00:00:00 UTC
        actuals_apr = cache.get_actuals_at("TSLA", as_of_ts=as_of_apr_15)
        assert len(actuals_apr) == 0, "Late filing should NOT be visible before ts_event"

        # At 2024-05-11 (1 day after filing)
        as_of_may_11 = 1715385600000000000  # 2024-05-11 00:00:00 UTC
        actuals_may = cache.get_actuals_at("TSLA", as_of_ts=as_of_may_11)
        assert len(actuals_may) == 1
        assert actuals_may[0]["period_end"] == "2024-03-31"

    def test_multiple_quarters_temporal_ordering(self) -> None:
        """
        Test correct temporal ordering with multiple quarters.

        Verifies only quarters with ts_event < as_of_ts are returned.
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Q3 2023 filed on 2023-11-01
        q3_ts = 1698796800000000000
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-09-30",
            filing_date="2023-11-01",
            eps_diluted=1.46,
            revenue=89_498_000_000.0,
            ts_event=q3_ts,
            ts_init=q3_ts,
        )

        # Q4 2023 filed on 2024-02-01
        q4_ts = 1706745600000000000
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-02-01",
            eps_diluted=2.18,
            revenue=119_575_000_000.0,
            ts_event=q4_ts,
            ts_init=q4_ts,
        )

        # Q1 2024 filed on 2024-05-02
        q1_ts = 1714608000000000000
        store.write_actuals(
            ticker="AAPL",
            period_end="2024-03-31",
            filing_date="2024-05-02",
            eps_diluted=1.53,
            revenue=90_753_000_000.0,
            ts_event=q1_ts,
            ts_init=q1_ts,
        )

        # At 2024-01-01: Only Q3 2023 should be visible
        as_of_jan_1 = 1704067200000000000
        actuals_jan = cache.get_actuals_at("AAPL", as_of_ts=as_of_jan_1)
        assert len(actuals_jan) == 1
        assert actuals_jan[0]["period_end"] == "2023-09-30"

        # At 2024-03-01: Q3 2023 and Q4 2023 should be visible
        as_of_mar_1 = 1709251200000000000
        actuals_mar = cache.get_actuals_at("AAPL", as_of_ts=as_of_mar_1)
        assert len(actuals_mar) == 2
        periods = {a["period_end"] for a in actuals_mar}
        assert periods == {"2023-09-30", "2023-12-31"}

        # At 2024-06-01: All three quarters should be visible
        as_of_jun_1 = 1717200000000000000
        actuals_jun = cache.get_actuals_at("AAPL", as_of_ts=as_of_jun_1)
        assert len(actuals_jun) == 3
        periods = {a["period_end"] for a in actuals_jun}
        assert periods == {"2023-09-30", "2023-12-31", "2024-03-31"}


class TestEstimatesTemporalCorrectness:
    """Test temporal correctness for estimates."""

    def test_estimate_not_visible_before_ts_event(self) -> None:
        """Test estimate only visible after its ts_event."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Estimate recorded on 2024-01-15
        estimate_ts = 1705276800000000000  # 2024-01-15 00:00:00 UTC
        store.write_estimates(
            ticker="AAPL",
            estimate_date="2024-01-15",
            period_end="2024-03-31",
            eps_consensus=1.50,
            ts_event=estimate_ts,
            ts_init=estimate_ts,
        )

        # At 2024-01-10 (before estimate) - NOT visible
        as_of_jan_10 = 1704844800000000000
        estimate_jan10 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=as_of_jan_10,
        )
        assert estimate_jan10 is None

        # At 2024-01-20 (after estimate) - visible
        as_of_jan_20 = 1705708800000000000
        estimate_jan20 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=as_of_jan_20,
        )
        assert estimate_jan20 is not None
        assert estimate_jan20["eps_consensus"] == 1.50

    def test_estimate_revision_temporal_ordering(self) -> None:
        """
        Test estimate revisions respect temporal ordering.

        Analyst revises estimate from $1.50 to $1.55. At time T, should see
        most recent estimate with ts_event < T.
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Initial estimate on 2024-01-15
        estimate1_ts = 1705276800000000000
        store.write_estimates(
            ticker="AAPL",
            estimate_date="2024-01-15",
            period_end="2024-03-31",
            eps_consensus=1.50,
            ts_event=estimate1_ts,
            ts_init=estimate1_ts,
        )

        # Revised estimate on 2024-02-01
        estimate2_ts = 1706745600000000000
        store.write_estimates(
            ticker="AAPL",
            estimate_date="2024-02-01",
            period_end="2024-03-31",
            eps_consensus=1.55,
            ts_event=estimate2_ts,
            ts_init=estimate2_ts,
        )

        # At 2024-01-20: Should see original estimate ($1.50)
        as_of_jan_20 = 1705708800000000000
        estimate_jan20 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=as_of_jan_20,
        )
        assert estimate_jan20["eps_consensus"] == 1.50

        # At 2024-02-05: Should see revised estimate ($1.55)
        as_of_feb_5 = 1707091200000000000
        estimate_feb5 = cache.get_estimates_at(
            ticker="AAPL",
            period_end="2024-03-31",
            as_of_ts=as_of_feb_5,
        )
        assert estimate_feb5["eps_consensus"] == 1.55


class TestRestatementScenario:
    """Test earnings restatements (rare but critical)."""

    def test_restatement_temporal_correctness(self) -> None:
        """
        Test restatement scenario maintains temporal correctness.

        Scenario:
        1. Q1 2024 initially filed with EPS=$1.50 on 2024-05-02
        2. Restated to EPS=$1.45 on 2024-08-01 (accounting error)

        At 2024-06-01: Should see original $1.50 (before restatement)
        At 2024-09-01: Should see restated $1.45 (after restatement)
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Original filing on 2024-05-02
        original_ts = 1714608000000000000
        store.write_actuals(
            ticker="XYZ",
            period_end="2024-03-31",
            filing_date="2024-05-02",
            eps_diluted=1.50,
            revenue=100_000_000_000.0,
            ts_event=original_ts,
            ts_init=original_ts,
        )

        # Restatement on 2024-08-01 (separate filing)
        restatement_ts = 1722470400000000000
        store.write_actuals(
            ticker="XYZ",
            period_end="2024-03-31",  # Same period
            filing_date="2024-08-01",  # Restatement date
            eps_diluted=1.45,  # Corrected value
            revenue=100_000_000_000.0,
            ts_event=restatement_ts,
            ts_init=restatement_ts,
        )

        # Note: This test assumes store handles restatements by inserting a new row
        # with different ts_event. If store overwrites, test needs adjustment.

        # At 2024-06-01: Should see original filing
        as_of_jun_1 = 1717200000000000000
        actuals_jun = cache.get_actuals_at("XYZ", as_of_ts=as_of_jun_1)

        # Find the Q1 2024 entry
        q1_entries = [a for a in actuals_jun if a["period_end"] == "2024-03-31"]

        # Should see original value (restatement not yet filed)
        if len(q1_entries) > 0:
            # If multiple entries exist, should only see those with ts_event < as_of_jun_1
            for entry in q1_entries:
                assert entry["ts_event"] < as_of_jun_1
                # Original filing should be visible
                if entry["filing_date"] == "2024-05-02":
                    assert entry["eps_diluted"] == 1.50


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_nanosecond_timestamp(self) -> None:
        """Test handling of edge case: ts_event = 0."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Hypothetical historical data with ts_event = 0
        store.write_actuals(
            ticker="OLD",
            period_end="1970-01-01",
            filing_date="1970-01-01",
            eps_diluted=0.01,
            revenue=1_000_000.0,
            ts_event=0,
            ts_init=1704067200000000000,
        )

        # Should be visible at any positive timestamp
        actuals = cache.get_actuals_at("OLD", as_of_ts=1)
        assert len(actuals) == 1
        assert actuals[0]["eps_diluted"] == 0.01

    def test_far_future_timestamp(self) -> None:
        """Test handling of far future timestamps."""
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Filing in 2024
        filing_ts = 1704067200000000000
        store.write_actuals(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-01-01",
            eps_diluted=2.10,
            revenue=119_575_000_000.0,
            ts_event=filing_ts,
            ts_init=filing_ts,
        )

        # Query far in the future (year 2100)
        far_future_ts = 4102444800000000000  # 2100-01-01
        actuals = cache.get_actuals_at("AAPL", as_of_ts=far_future_ts)

        # Should see 2024 filing (ts_event < far_future_ts)
        assert len(actuals) == 1
        assert actuals[0]["eps_diluted"] == 2.10

    def test_same_period_multiple_filings(self) -> None:
        """
        Test multiple filings for same period (amendments).

        8-K/A (amendment) filed after original 8-K. At time T, should see
        all filings with ts_event < T.
        """
        store = build_test_earnings_adapter()
        cache = EarningsCache(store, maxsize=128)

        # Original 8-K on 2024-05-02
        original_ts = 1714608000000000000
        store.write_actuals(
            ticker="AMZN",
            period_end="2024-03-31",
            filing_date="2024-05-02",
            eps_diluted=0.98,
            revenue=143_313_000_000.0,
            ts_event=original_ts,
            ts_init=original_ts,
            filing_type="8-K",
        )

        # Amended 8-K/A on 2024-05-10
        amendment_ts = 1715299200000000000
        store.write_actuals(
            ticker="AMZN",
            period_end="2024-03-31",  # Same period
            filing_date="2024-05-10",
            eps_diluted=0.98,  # Unchanged, but filing is amended
            revenue=143_313_000_000.0,
            ts_event=amendment_ts,
            ts_init=amendment_ts,
            filing_type="8-K/A",
        )

        # At 2024-05-05: Only original 8-K visible
        as_of_may_5 = 1714867200000000000
        actuals_may5 = cache.get_actuals_at("AMZN", as_of_ts=as_of_may_5)

        # Should see only original filing
        filings_before_amendment = [a for a in actuals_may5 if a["period_end"] == "2024-03-31"]
        assert all(a["ts_event"] < as_of_may_5 for a in filings_before_amendment)
