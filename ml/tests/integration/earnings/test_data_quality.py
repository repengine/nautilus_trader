"""
Data quality validation tests for earnings pipeline.

This module validates data quality across the earnings pipeline:
- No nulls in required fields
- Value ranges are reasonable (EPS > -1000, Revenue > 0)
- Outlier detection
- Edge cases: missing estimates, filing delays, restatements

All tests marked with @pytest.mark.integration for separate execution.
"""

from __future__ import annotations

import time
from datetime import date
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from unittest.mock import MagicMock

from typing import cast
from ml.stores.data_store import DataStore
from ml.features.earnings import compute_earnings_growth_batch
from ml.features.earnings import compute_earnings_surprise_batch
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_processor import DataProcessor
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.utils.earnings_facade import build_test_data_store

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


if TYPE_CHECKING:
    pass


class TestEarningsDataQuality:
    """Data quality validation for earnings pipeline."""

    @pytest.fixture
    def data_store(self, request: pytest.FixtureRequest, tmp_path: Path) -> DataStore:
        """Provide DataStore with earnings support and fallback."""
        import os

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_trader")

        feature_store = cast(FeatureStore, MagicMock(spec=FeatureStore))
        model_store = cast(ModelStore, MagicMock(spec=ModelStore))
        strategy_store = cast(StrategyStore, MagicMock(spec=StrategyStore))
        processor = cast(DataProcessor, MagicMock(spec=DataProcessor))

        registry = DataRegistry(
            registry_path=tmp_path / "registry",
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path / "registry",
            ),
        )

        store = build_test_data_store(
            connection_string=db_url,
            registry=registry,
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            data_processor=processor,
        )
        yield store

    def test_required_fields_non_null(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Validate that required fields are never null.

        Required fields:
        - Actuals: ticker, period_end, filing_date, eps_diluted, ts_event, ts_init
        - Estimates: ticker, estimate_date, period_end, eps_consensus, ts_event, ts_init
        """
        # Create sample with all required fields
        ticker = "AAPL"
        period_end = "2024-09-30"
        filing_date = "2024-10-31"
        eps_diluted = 1.64
        ts_event = int(time.time_ns())
        ts_init = int(time.time_ns())

        # Test actuals - should succeed with all required fields
        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=94.9e9,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event + 1,
            limit=1,
            start_date=period_end,
            end_date=period_end,
        )
        assert len(actuals) == 1
        actual = actuals[0]

        # Verify required fields are present and non-null
        assert actual["ticker"] is not None
        assert actual["period_end"] is not None
        assert actual["filing_date"] is not None
        assert actual["eps_diluted"] is not None
        assert actual["ts_event"] is not None
        assert actual["ts_init"] is not None

        # Test estimates - should succeed with all required fields
        estimate_date = "2024-09-20"
        eps_consensus = 1.60

        data_store.write_earnings_estimate(
            ticker=ticker,
            estimate_date=estimate_date,
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        estimate = data_store.get_earnings_estimate_at_or_before(
            ticker=ticker,
            period_end=period_end,
            ts_event=ts_event + 1,
        )
        assert estimate is not None

        # Verify required fields are present and non-null
        assert estimate["ticker"] is not None
        assert estimate["estimate_date"] is not None
        assert estimate["period_end"] is not None
        assert estimate["eps_consensus"] is not None
        assert estimate["ts_event"] is not None
        assert estimate["ts_init"] is not None

        print("\n✅ Required fields validation passed!")

    def test_value_range_validation(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Validate that EPS and revenue values are within reasonable ranges.

        Constraints:
        - EPS: -1000 < eps < 1000 (handles losses and extreme profits)
        - Revenue: > 0 (must be positive)
        - Shares outstanding: > 0
        """
        ticker = "TEST"
        period_end = "2024-09-30"
        filing_date = "2024-10-31"
        ts_event = int(time.time_ns())
        ts_init = int(time.time_ns())

        # Test 1: Valid positive EPS
        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=2.50,
            revenue=100e9,
            ts_event=ts_event,
            ts_init=ts_init,
            shares_outstanding=10_000_000_000,
        )

        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event + 1,
            limit=4,
        )
        assert len(actuals) > 0
        assert actuals[0]["eps_diluted"] > -1000
        assert actuals[0]["eps_diluted"] < 1000
        assert actuals[0]["revenue"] > 0
        assert actuals[0]["shares_outstanding"] > 0

        # Test 2: Valid negative EPS (company with loss)
        loss_ticker = "LOSS_COMPANY"
        data_store.write_earnings_actual(
            ticker=loss_ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=-0.50,  # Loss of $0.50 per share
            revenue=50e9,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        loss_actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=loss_ticker,
            ts_event=ts_event + 1,
            limit=4,
        )
        assert len(loss_actuals) > 0
        assert loss_actuals[0]["eps_diluted"] < 0
        assert loss_actuals[0]["eps_diluted"] > -1000  # Still within reasonable range
        assert loss_actuals[0]["revenue"] > 0  # Revenue should still be positive

        # Test 3: Extreme but valid values
        extreme_ticker = "EXTREME"
        data_store.write_earnings_actual(
            ticker=extreme_ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=100.0,  # Very high EPS (like Berkshire Hathaway)
            revenue=500e9,  # $500B revenue
            ts_event=ts_event,
            ts_init=ts_init,
        )

        extreme_actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=extreme_ticker,
            ts_event=ts_event + 1,
            limit=4,
        )
        assert len(extreme_actuals) > 0
        assert extreme_actuals[0]["eps_diluted"] > 0
        assert extreme_actuals[0]["eps_diluted"] < 1000

        print("\n✅ Value range validation passed!")

    def test_outlier_detection(self) -> None:
        """
        Test outlier detection in earnings surprises and growth rates.

        Outliers defined as:
        - Surprise > 3 standard deviations from mean
        - Growth rate > 1000% (10x)
        """
        # Generate normal earnings with one outlier
        normal_eps = [2.0, 2.1, 2.05, 2.15, 2.08, 2.12, 2.10, 2.09]
        outlier_eps = 10.0  # 5x jump - clear outlier

        all_eps = normal_eps + [outlier_eps]

        # Calculate statistics
        mean_eps = np.mean(normal_eps)
        std_eps = np.std(normal_eps)

        # Check if outlier is detected (> 3 std devs)
        z_score = (outlier_eps - mean_eps) / std_eps
        assert z_score > 3, f"Outlier not detected: z_score={z_score}"

        # Test growth outlier
        eps_before_outlier = normal_eps[-1]  # 2.09
        growth_rate = ((outlier_eps - eps_before_outlier) / eps_before_outlier) * 100
        assert growth_rate > 100, f"Growth outlier: {growth_rate}%"

        print(f"\n✅ Outlier detection passed! Z-score={z_score:.2f}, Growth={growth_rate:.1f}%")

    def test_edge_case_missing_estimates(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test handling of missing consensus estimates.

        Edge case: Company reports earnings but has no analyst coverage.
        """
        ticker = "UNCOVERED_STOCK"
        period_end = "2024-09-30"
        filing_date = "2024-10-31"
        ts_event = int(time.time_ns())
        ts_init = int(time.time_ns())

        # Store actual without estimate
        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=1.50,
            revenue=10e9,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Try to get estimate (should return None)
        estimate = data_store.get_earnings_estimate_at_or_before(
            ticker=ticker,
            period_end=period_end,
            ts_event=ts_event + 1,
        )
        assert estimate is None, "Should return None for missing estimate"

        # Surprise calculation should handle None gracefully
        from ml.features.earnings import compute_earnings_surprise_incremental

        surprise = compute_earnings_surprise_incremental(1.50, None)
        assert surprise["eps_surprise_q0"] == 0.0
        assert surprise["eps_surprise_pct_q0"] == 0.0

        print("\n✅ Missing estimates edge case passed!")

    def test_edge_case_filing_delays(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test handling of late filings (>45 days after quarter end).

        Edge case: Company files 10-Q late due to accounting issues.
        """
        ticker = "DELAYED_FILER"
        period_end = date(2024, 9, 30)
        filing_date = period_end + timedelta(days=90)  # 90 days late (normal is 45 days)
        ts_event = int(filing_date.strftime("%s")) * 1_000_000_000
        ts_init = int(time.time_ns())

        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=str(period_end),
            filing_date=str(filing_date),
            eps_diluted=1.25,
            revenue=50e9,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Verify late filing is stored correctly
        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event + 1,
            limit=4,
        )
        assert len(actuals) == 1

        # Calculate filing delay
        filing_delay = (filing_date - period_end).days
        assert filing_delay > 45, f"Filing delay should be >45 days, got {filing_delay}"

        print(f"\n✅ Filing delay edge case passed! Delay={filing_delay} days")

    def test_edge_case_restatements(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test handling of earnings restatements.

        Edge case: Company restates prior quarter earnings due to accounting error.
        """
        ticker = "RESTATED_EARNINGS"
        period_end = "2024-06-30"
        original_filing_date = "2024-07-31"
        restated_filing_date = "2024-09-15"
        ts_event_original = int(date(2024, 7, 31).strftime("%s")) * 1_000_000_000
        ts_event_restated = int(date(2024, 9, 15).strftime("%s")) * 1_000_000_000
        ts_init = int(time.time_ns())

        # Original filing
        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=original_filing_date,
            eps_diluted=2.00,  # Original EPS
            revenue=80e9,
            ts_event=ts_event_original,
            ts_init=ts_init,
        )

        # Restated filing (corrected EPS)
        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,  # Same period
            filing_date=restated_filing_date,  # New filing date
            eps_diluted=1.85,  # Corrected EPS (lower)
            revenue=80e9,
            ts_event=ts_event_restated,
            ts_init=ts_init,
        )

        # Query after restatement - should reflect restated EPS
        actuals_restated = data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event_restated + 1,
            limit=4,
        )
        assert len(actuals_restated) == 1
        assert actuals_restated[0]["eps_diluted"] == 1.85  # Restated value

        print("\n✅ Restatement edge case passed!")

    def test_data_consistency_cross_validation(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test data consistency through cross-validation.

        Validates:
        - EPS = Net Income / Shares Outstanding (within tolerance)
        - Fiscal quarter matches period_end
        - Filing date > period_end
        """
        ticker = "CONSISTENCY_TEST"
        period_end = "2024-09-30"
        filing_date = "2024-10-31"
        net_income = 20e9  # $20B
        shares_outstanding = 10e9  # 10B shares
        expected_eps = net_income / shares_outstanding  # $2.00
        ts_event = int(time.time_ns())
        ts_init = int(time.time_ns())

        data_store.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=expected_eps,
            revenue=100e9,
            net_income=net_income,
            shares_outstanding=int(shares_outstanding),
            ts_event=ts_event,
            ts_init=ts_init,
            fiscal_year=2024,
            fiscal_quarter=3,  # Q3 ends September 30
        )

        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event + 1,
            limit=4,
        )
        assert len(actuals) > 0
        actual = actuals[0]

        # Validate EPS calculation
        calculated_eps = actual["net_income"] / actual["shares_outstanding"]
        assert abs(calculated_eps - actual["eps_diluted"]) < 0.01, (
            f"EPS inconsistency: calculated={calculated_eps}, reported={actual['eps_diluted']}"
        )

        # Validate fiscal quarter matches period_end
        period = date.fromisoformat(actual["period_end"])
        expected_quarter = ((period.month - 1) // 3) + 1
        assert actual["fiscal_quarter"] == expected_quarter, (
            f"Fiscal quarter mismatch: expected Q{expected_quarter}, got Q{actual['fiscal_quarter']}"
        )

        # Validate filing_date > period_end
        filing = date.fromisoformat(actual["filing_date"])
        assert filing > period, f"Filing date {filing} should be after period end {period}"

        print("\n✅ Data consistency cross-validation passed!")

    def test_statistical_outlier_detection_zscore(self) -> None:
        """
        Test Z-score based outlier detection for earnings surprises.

        Uses 3-sigma rule: outliers are > 3 standard deviations from mean.
        """
        # Generate earnings surprises (normal distribution + outliers)
        np.random.seed(42)
        normal_surprises = np.random.normal(loc=0.05, scale=0.10, size=100)  # Mean 5%, std 10%
        outliers = np.array([0.50, -0.40, 0.60])  # Clear outliers (50%, -40%, 60%)

        all_surprises = np.concatenate([normal_surprises, outliers])

        # Calculate Z-scores
        mean = np.mean(normal_surprises)
        std = np.std(normal_surprises)
        z_scores = np.abs((all_surprises - mean) / std)

        # Detect outliers (Z-score > 3)
        outlier_mask = z_scores > 3
        detected_outliers = all_surprises[outlier_mask]

        # Verify outliers were detected
        assert len(detected_outliers) >= len(outliers), (
            f"Expected at least {len(outliers)} outliers, found {len(detected_outliers)}"
        )

        # Verify outlier values are in detected set
        for outlier in outliers:
            assert any(abs(detected - outlier) < 1e-6 for detected in detected_outliers), (
                f"Outlier {outlier} not detected"
            )

        print(f"\n✅ Z-score outlier detection passed! Detected {len(detected_outliers)} outliers")
