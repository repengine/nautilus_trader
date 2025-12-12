"""
End-to-end integration tests for earnings pipeline.

This module tests the complete earnings data flow:
1. Fetch actuals from EDGAR (or mock)
2. Store in PostgreSQL via EarningsStore
3. Fetch estimates from Yahoo (or mock)
4. Store estimates
5. Compute all 8 earnings features
6. Verify values are reasonable

Tests are marked with @pytest.mark.integration and designed to run against
real or mock data sources with PostgreSQL backend.

Performance targets:
- Full pipeline for 1 instrument × 8 quarters: < 10s
- Feature computation: < 50ms batch
- Data persistence: < 100ms for 8 records
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml.stores.data_store import DataStore
from ml.tests.utils.db import build_postgres_url

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


@dataclass(frozen=True)
class _SampleActual:
    ticker: str
    period_end: date
    filing_date: date
    eps_diluted: float
    revenue: float
    eps_basic: float
    net_income: float
    operating_income: float
    shares_outstanding: int
    filing_type: str
    fiscal_year: int
    fiscal_quarter: int
    ts_event: int
    ts_init: int


@dataclass(frozen=True)
class _SampleEstimate:
    ticker: str
    estimate_date: date
    period_end: date
    eps_consensus: float
    revenue_consensus: float
    num_analysts: int
    ts_event: int
    ts_init: int
from ml.features.earnings import compute_calendar_features_batch
from ml.features.earnings import compute_calendar_features_incremental
from ml.features.earnings import compute_earnings_growth_batch
from ml.features.earnings import compute_earnings_growth_incremental
from ml.features.earnings import compute_earnings_momentum_batch
from ml.features.earnings import compute_earnings_momentum_incremental
from ml.features.earnings import compute_earnings_surprise_batch
from ml.features.earnings import compute_earnings_surprise_incremental
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_processor import DataProcessor
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.utils.earnings_facade import build_test_data_store


if TYPE_CHECKING:
    pass


@pytest.mark.integration
class TestEarningsEndToEnd:
    """End-to-end integration tests for earnings pipeline."""

    @pytest.fixture
    def data_store(self, request: pytest.FixtureRequest, tmp_path: Path) -> DataStore:
        """
        Provide DataStore with earnings support and progressive fallback.
        """
        import os

        db_url = os.getenv(
            "DATABASE_URL",
            build_postgres_url(database="nautilus_trader"),
        )

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

    @pytest.fixture
    def sample_actuals(self) -> list[_SampleActual]:
        """Generate sample AAPL actual earnings for 8 quarters."""
        actuals = []
        base_date = date(2024, 9, 30)  # Start with Q3 2024

        # Q3 2024 → Q4 2022 (8 quarters)
        eps_values = [1.64, 1.40, 1.53, 1.52, 1.29, 1.20, 1.26, 1.24]
        revenue_values = [94.9e9, 85.8e9, 90.8e9, 89.5e9, 81.8e9, 82.0e9, 90.1e9, 83.0e9]

        for i, (eps, revenue) in enumerate(zip(eps_values, revenue_values)):
            quarter_end = base_date - timedelta(days=90 * i)
            filing_date = quarter_end + timedelta(days=45)  # 45 days after quarter end

            actual = _SampleActual(
                ticker="AAPL",
                period_end=quarter_end,
                filing_date=filing_date,
                eps_diluted=eps,
                revenue=revenue,
                eps_basic=eps + 0.01,  # Slightly higher basic EPS
                net_income=revenue * 0.25,  # 25% margin
                operating_income=revenue * 0.30,
                shares_outstanding=int(16e9),  # 16B shares
                filing_type="10-Q" if i < 7 else "10-K",
                fiscal_year=2024 - (i // 4),
                fiscal_quarter=(4 - (i % 4)) if i < 4 else (4 - ((i - 4) % 4)),
                ts_event=int(filing_date.strftime("%s")) * 1_000_000_000,
                ts_init=int(time.time_ns()),
            )
            actuals.append(actual)

        return actuals

    @pytest.fixture
    def sample_estimates(self) -> list[_SampleEstimate]:
        """Generate sample AAPL consensus estimates."""
        estimates = []
        base_date = date(2024, 9, 30)

        # Estimates are slightly lower than actuals (to generate positive surprises)
        eps_estimates = [1.60, 1.35, 1.50, 1.48, 1.25, 1.18, 1.23, 1.20]

        for i, eps_est in enumerate(eps_estimates):
            quarter_end = base_date - timedelta(days=90 * i)
            estimate_date = quarter_end - timedelta(days=10)  # 10 days before quarter end

            consensus = _SampleEstimate(
                ticker="AAPL",
                estimate_date=estimate_date,
                period_end=quarter_end,
                eps_consensus=eps_est,
                revenue_consensus=eps_est * 16e9 * 4,  # Rough revenue estimate
                num_analysts=35,
                ts_event=int(estimate_date.strftime("%s")) * 1_000_000_000,
                ts_init=int(time.time_ns()),
            )
            estimates.append(consensus)

        return estimates

    def test_full_pipeline_aapl(
        self,
        data_store: DataStore,
        sample_actuals: list[_SampleActual],
        sample_estimates: list[_SampleEstimate],
    ) -> None:
        """
        Test full pipeline: EDGAR → PostgreSQL → Features → Validation.

        Steps:
        1. Store actual earnings (8 quarters)
        2. Store consensus estimates
        3. Compute all features (surprise, growth, momentum, calendar)
        4. Verify feature values are reasonable
        """
        # Step 1: Store actuals
        start_time = time.perf_counter()
        for actual in sample_actuals:
            data_store.write_earnings_actual(
                ticker=actual.ticker,
                period_end=str(actual.period_end),
                filing_date=str(actual.filing_date),
                eps_diluted=actual.eps_diluted,
                revenue=actual.revenue,
                ts_event=actual.ts_event,
                ts_init=actual.ts_init,
                eps_basic=actual.eps_basic,
                net_income=actual.net_income,
                operating_income=actual.operating_income,
                shares_outstanding=actual.shares_outstanding,
                filing_type=actual.filing_type,
                fiscal_year=actual.fiscal_year,
                fiscal_quarter=actual.fiscal_quarter,
            )
        actuals_write_time = time.perf_counter() - start_time

        # Step 2: Store estimates
        start_time = time.perf_counter()
        for estimate in sample_estimates:
            data_store.write_earnings_estimate(
                ticker=estimate.ticker,
                estimate_date=str(estimate.estimate_date),
                period_end=str(estimate.period_end),
                eps_consensus=estimate.eps_consensus,
                ts_event=estimate.ts_event,
                ts_init=estimate.ts_init,
                revenue_consensus=estimate.revenue_consensus,
                num_analysts=estimate.num_analysts,
            )
        estimates_write_time = time.perf_counter() - start_time

        # Step 3: Retrieve stored data
        ts_cutoff = int(time.time_ns())
        actuals_from_db = data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=ts_cutoff,
            limit=16,
            start_date="2022-01-01",
            end_date="2024-12-31",
        )
        assert len(actuals_from_db) == 8, f"Expected 8 actuals, got {len(actuals_from_db)}"

        # Step 4: Compute earnings surprise features (incremental)
        actual_q0 = sample_actuals[0].eps_diluted
        estimate_q0 = sample_estimates[0].eps_consensus
        surprise_incremental = compute_earnings_surprise_incremental(actual_q0, estimate_q0)

        assert "eps_surprise_q0" in surprise_incremental
        assert "eps_surprise_pct_q0" in surprise_incremental

        # Verify surprise calculation
        expected_surprise = actual_q0 - estimate_q0  # 1.64 - 1.60 = 0.04
        expected_surprise_pct = (expected_surprise / estimate_q0) * 100  # (0.04 / 1.60) * 100 = 2.5%

        assert abs(surprise_incremental["eps_surprise_q0"] - expected_surprise) < 1e-10
        assert abs(surprise_incremental["eps_surprise_pct_q0"] - expected_surprise_pct) < 1e-6

        # Step 5: Compute earnings growth features (batch)
        eps_history_desc = sample_actuals[:5]
        eps_series_batch = np.array([a.eps_diluted for a in reversed(eps_history_desc)], dtype=float)
        growth_batch = compute_earnings_growth_batch(eps_series_batch)

        assert "eps_growth_yoy" in growth_batch
        assert "eps_growth_qoq" in growth_batch

        # Verify YoY growth: (EPS_Q0 - EPS_Q4) / EPS_Q4 * 100
        expected_yoy = (
            (
                eps_history_desc[0].eps_diluted
                - eps_history_desc[4].eps_diluted
            )
            / eps_history_desc[4].eps_diluted
        ) * 100
        assert abs(growth_batch["eps_growth_yoy"][-1] - expected_yoy) < 1e-6

        # Step 6: Compute earnings momentum features
        surprises_desc = [
            a.eps_diluted - e.eps_consensus
            for a, e in zip(sample_actuals[:4], sample_estimates[:4])
        ]
        surprises_batch = np.array(list(reversed(surprises_desc)), dtype=float)
        momentum_batch = compute_earnings_momentum_batch(
            surprises_batch,
            eps_series_batch[-4:],
        )

        assert "earnings_beat_streak" in momentum_batch
        assert "eps_volatility_4q" in momentum_batch

        # All surprises are positive, so beat streak should be 4
        assert int(momentum_batch["earnings_beat_streak"][-1]) == 4
        assert float(momentum_batch["eps_volatility_4q"][-1]) >= 0.0

        # Step 7: Compute calendar features
        current_date = datetime.now()
        next_earnings_date = current_date + timedelta(days=45)
        calendar_incremental = compute_calendar_features_incremental(next_earnings_date, current_date)

        assert "days_to_next_earnings" in calendar_incremental
        assert calendar_incremental["days_to_next_earnings"] == 45

        # Step 8: Performance validation
        print("\nPerformance Metrics:")
        print(f"  Actuals write time: {actuals_write_time*1000:.2f}ms for 8 records")
        print(f"  Estimates write time: {estimates_write_time*1000:.2f}ms for 8 records")
        print(f"  Total write time: {(actuals_write_time + estimates_write_time)*1000:.2f}ms")

        # Performance assertions
        assert actuals_write_time < 1.0, f"Actuals write too slow: {actuals_write_time}s"
        assert estimates_write_time < 1.0, f"Estimates write too slow: {estimates_write_time}s"

        print("\n✅ Full pipeline test passed!")

    def test_point_in_time_correctness(
        self,
        data_store: DataStore,
        sample_actuals: list[EarningsActual],
    ) -> None:
        """
        Test point-in-time correctness - no look-ahead bias.

        Verifies that querying with as_of_ts only returns data filed before that timestamp.
        """
        # Store all actuals
        for actual in sample_actuals:
            data_store.write_earnings_actual(
                ticker=actual.ticker,
                period_end=str(actual.period_end),
                filing_date=str(actual.filing_date),
                eps_diluted=actual.eps_diluted,
                revenue=actual.revenue,
                ts_event=actual.ts_event,
                ts_init=actual.ts_init,
                filing_type=actual.filing_type,
            )

        # Query as of Q1 2024 filing date (should only see Q4 2023 and earlier)
        q1_2024_filing_ts = sample_actuals[2].ts_event  # 3rd quarter back

        actuals_as_of_q1 = data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=q1_2024_filing_ts,
            limit=16,
        )

        # Should only get actuals filed before Q1 2024 filing
        assert len(actuals_as_of_q1) == 5, f"Expected 5 actuals as of Q1 2024, got {len(actuals_as_of_q1)}"

        # Verify all returned actuals have ts_event < q1_2024_filing_ts
        for actual_dict in actuals_as_of_q1:
            assert actual_dict["ts_event"] < q1_2024_filing_ts, (
                f"Look-ahead bias detected: ts_event {actual_dict['ts_event']} >= {q1_2024_filing_ts}"
            )

        print("\n✅ Point-in-time correctness test passed!")

    def test_batch_vs_incremental_parity(
        self,
        sample_actuals: list[_SampleActual],
        sample_estimates: list[_SampleEstimate],
    ) -> None:
        """
        Test parity between batch and incremental feature computation.

        Verifies that batch and incremental paths produce identical results to rtol=1e-10.
        """
        # Test earnings surprise parity
        actual = sample_actuals[0].eps_diluted
        estimate = sample_estimates[0].eps_consensus

        surprise_incremental = compute_earnings_surprise_incremental(actual, estimate)
        surprise_batch = compute_earnings_surprise_batch(
            np.array([actual], dtype=float),
            np.array([estimate], dtype=float),
        )

        assert (
            abs(
                surprise_incremental["eps_surprise_q0"]
                - float(surprise_batch["eps_surprise_q0"][0])
            )
            < 1e-10
        )
        assert (
            abs(
                surprise_incremental["eps_surprise_pct_q0"]
                - float(surprise_batch["eps_surprise_pct_q0"][0])
            )
            < 1e-10
        )

        # Test earnings growth parity
        eps_history_desc = sample_actuals[:5]
        growth_incremental = compute_earnings_growth_incremental(
            [a.eps_diluted for a in eps_history_desc],
        )
        eps_series_batch = np.array([a.eps_diluted for a in reversed(eps_history_desc)], dtype=float)
        growth_batch = compute_earnings_growth_batch(eps_series_batch)

        assert (
            abs(
                growth_incremental["eps_growth_yoy"]
                - float(growth_batch["eps_growth_yoy"][-1])
            )
            < 1e-10
        )
        assert (
            abs(
                growth_incremental["eps_growth_qoq"]
                - float(growth_batch["eps_growth_qoq"][-1])
            )
            < 1e-10
        )

        # Test earnings momentum parity
        surprises_desc = [
            a.eps_diluted - e.eps_consensus
            for a, e in zip(sample_actuals[:4], sample_estimates[:4])
        ]
        surprises = np.array(list(reversed(surprises_desc)), dtype=float)

        momentum_incremental = compute_earnings_momentum_incremental(
            surprises=surprises_desc,
            eps_history=[a.eps_diluted for a in sample_actuals[:4]],
        )
        momentum_batch = compute_earnings_momentum_batch(
            surprises,
            eps_series_batch[-4:],
        )

        assert (
            momentum_incremental["earnings_beat_streak"]
            == float(momentum_batch["earnings_beat_streak"][-1])
        )
        assert (
            abs(
                momentum_incremental["eps_volatility_4q"]
                - float(momentum_batch["eps_volatility_4q"][-1])
            )
            < 1e-10
        )

        print("\n✅ Batch vs incremental parity test passed!")

    def test_missing_data_handling(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test graceful handling of missing estimates and actuals.

        Verifies that the pipeline handles edge cases:
        - Missing consensus estimates
        - Missing actuals
        - Partial data availability
        """
        # Test 1: Get actuals for non-existent ticker
        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker="NONEXISTENT_TICKER",
            ts_event=int(time.time_ns()),
        )
        assert actuals == [], "Should return empty list for non-existent ticker"

        # Test 2: Get estimates for non-existent period
        estimate = data_store.get_earnings_estimate_at_or_before(
            ticker="AAPL",
            period_end="2050-12-31",
            ts_event=int(time.time_ns()),
        )
        assert estimate is None, "Should return None for non-existent estimate"

        # Test 3: Surprise calculation with missing estimate
        surprise_with_none = compute_earnings_surprise_incremental(2.50, None)
        assert surprise_with_none["eps_surprise_q0"] == 0.0
        assert surprise_with_none["eps_surprise_pct_q0"] == 0.0

        # Test 4: Growth calculation with insufficient history
        insufficient_eps = [1.50, 1.40]  # Only 2 quarters, need 5
        growth = compute_earnings_growth_batch(insufficient_eps)
        # Should handle gracefully (returns 0.0 or raises ValueError depending on implementation)
        assert isinstance(growth, dict)

        print("\n✅ Missing data handling test passed!")
