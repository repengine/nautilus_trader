# Earnings Integration Plan: SEC EDGAR + Yahoo Finance

## Executive Summary

This document outlines the implementation plan for integrating corporate earnings data into the Nautilus Trader ML pipeline. The approach uses **SEC EDGAR (free, 100% coverage)** as the primary source for actual earnings data, supplemented by **Yahoo Finance (free, ~70% coverage)** for consensus estimates.

**Goal**: Add 8-15 fundamental features per equity instrument to complement the existing 680+ technical/macro/cross-asset features.

**Timeline**: 5-7 days of implementation + 2-3 days of testing

**Cost**: $0 (all free data sources)

---

> **Integration Status (2024.06)** — The DataStore facade now fronts earnings storage with
> PostgreSQL → `FileEarningsStore` → `DummyEarningsStore` fallback and exposes typed
> read/write helpers. Update any references in this plan to instantiate `EarningsStore`
> directly: tests, CLIs, and docs should use the facade or `DataStoreEarningsAdapter`.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Architecture Design](#architecture-design)
3. [Data Sources & Coverage](#data-sources--coverage)
4. [Implementation Phases](#implementation-phases)
5. [Feature Specifications](#feature-specifications)
6. [Database Schema](#database-schema)
7. [Pipeline Integration](#pipeline-integration)
8. [Testing Strategy](#testing-strategy)
9. [Performance Requirements](#performance-requirements)
10. [Future Enhancements](#future-enhancements)

---

## Current State Analysis

### Existing Feature Coverage (as of 2025-10-02)

#### ✅ What We Have (680 features per instrument)
- **Market/Technical** (45): OHLCV + 40 technical indicators
- **Macro Composites** (36): Credit, duration, liquidity, growth/inflation, FX
- **Cross-Asset** (~600): EWMA betas, z-scored spreads, rolling correlations
- **Static Metadata** (5): Duration bucket, issuer type, liquidity tier, security type, credit rating

#### ❌ Critical Gap: Corporate Fundamentals
- **No earnings data**: EPS actuals, estimates, surprises
- **No earnings calendar**: Upcoming announcements
- **No analyst revisions**: Consensus changes over time
- **No company guidance**: Forward-looking statements

### Instruments Requiring Earnings Data

From the 95-instrument universe, earnings data is relevant for:

**Equities** (~30-40 instruments):
- Individual stocks: AAPL.NASDAQ, MSFT.NASDAQ, JPM.NYSE, etc.
- Equity index futures: ES (S&P 500), NQ (Nasdaq 100), RTY (Russell 2000)

**Corporate Bonds** (~10-15 instruments):
- Investment Grade: AAPL-5Y.BOND, MSFT-10Y.BOND, JPM-5Y.BOND
- High Yield: XYZ-7Y.BOND, EMCORP-5Y.BOND

**Not Applicable** (~40-50 instruments):
- Treasuries: UST-2Y, UST-10Y, UST-30Y (no corporate earnings)
- Sovereign bonds: BUND-10Y, GILT-10Y, JGB-10Y
- Pure derivatives: Interest rate swaps

**Estimated Coverage Need**: 40-50 instruments × 8-15 features = **320-750 new earnings features**

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
├─────────────────────────────────────────────────────────────────┤
│  SEC EDGAR (Free)              │  Yahoo Finance (Free)           │
│  - 10-Q/10-K filings           │  - Consensus estimates          │
│  - 8-K earnings releases       │  - Earnings calendar            │
│  - XBRL financial data         │  - Analyst targets              │
│  - 100% US coverage            │  - ~70% coverage                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ml/data/earnings/                             │
├─────────────────────────────────────────────────────────────────┤
│  edgar_fetcher.py     │ Fetch & parse SEC filings (edgartools)  │
│  yahoo_fetcher.py     │ Fetch consensus estimates (yfinance)    │
│  earnings_processor.py│ Calculate surprises, growth rates       │
│  earnings_cache.py    │ Point-in-time caching for backtesting   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL Storage                             │
├─────────────────────────────────────────────────────────────────┤
│  ml.earnings_actuals  │ Historical EPS/Revenue (EDGAR)          │
│  ml.earnings_estimates│ Consensus forecasts (Yahoo)             │
│  ml.earnings_calendar │ Upcoming announcements                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ml/features/earnings/                           │
├─────────────────────────────────────────────────────────────────┤
│  earnings_features.py │ Calculate surprise, growth, momentum    │
│  earnings_transforms.py│ PipelineSpec integration               │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TFT Training Dataset                         │
│  Market (45) + Macro (36) + Cross-Asset (600) + Metadata (5)    │
│                  + Earnings (320-750) = 1,000-1,430 features     │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ml/
├── data/
│   └── earnings/
│       ├── __init__.py
│       ├── edgar_fetcher.py           # SEC EDGAR API client
│       ├── yahoo_fetcher.py           # Yahoo Finance API client
│       ├── earnings_processor.py      # Surprise calculation, growth rates
│       ├── earnings_cache.py          # Point-in-time caching
│       └── xbrl_parser.py             # XBRL parsing utilities
│
├── features/
│   └── earnings/
│       ├── __init__.py
│       ├── earnings_features.py       # Feature engineering
│       └── earnings_transforms.py     # TransformSpec for pipeline
│
├── schema/
│   └── earnings.sql                   # Database schema
│
├── stores/
│   ├── earnings_store.py              # EarningsStore (Protocol-based)
│   └── protocols.py                   # Add EarningsStoreProtocol
│
└── tests/
    ├── unit/
    │   ├── data/
    │   │   └── earnings/
    │   │       ├── test_edgar_fetcher.py
    │   │       ├── test_yahoo_fetcher.py
    │   │       └── test_earnings_processor.py
    │   └── features/
    │       └── earnings/
    │           └── test_earnings_features.py
    └── integration/
        └── earnings/
            └── test_earnings_pipeline.py
```

---

## Data Sources & Coverage

### Primary Source: SEC EDGAR (Free)

#### Python Library: `edgartools`
```bash
pip install edgartools
```

#### Capabilities
- **Actual Earnings**: 100% coverage of US public companies
- **Financial Statements**: 10-Q (quarterly), 10-K (annual)
- **Material Events**: 8-K filings (earnings releases, guidance)
- **XBRL Data**: Structured financial data (EPS, revenue, income)
- **Historical**: Data back to 2000s
- **Update Frequency**: T+1 day after filing

#### Data Extracted from EDGAR
```python
# From 10-Q/10-K filings (XBRL tags)
earnings_actuals = {
    "eps_basic": "us-gaap:EarningsPerShareBasic",
    "eps_diluted": "us-gaap:EarningsPerShareDiluted",
    "revenue": "us-gaap:Revenues",
    "net_income": "us-gaap:NetIncomeLoss",
    "operating_income": "us-gaap:OperatingIncomeLoss",
    "shares_outstanding": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
}

# From 8-K filings (Item 2.02 - Results of Operations)
guidance_data = {
    "forward_eps_guidance": "Parsed from text",
    "forward_revenue_guidance": "Parsed from text",
    "management_commentary": "Extracted from Item 2.02",
}
```

#### Limitations
- ❌ No analyst consensus estimates (actuals only)
- ❌ ~24 hour delay (filings processed overnight)
- ❌ Parsing complexity for non-standard XBRL filers

### Secondary Source: Yahoo Finance (Free)

#### Python Library: `yfinance`
```bash
pip install yfinance
```

#### Capabilities
- **Consensus Estimates**: Analyst EPS/revenue forecasts
- **Earnings Calendar**: Next earnings date
- **Historical Estimates**: Limited (last 2-4 quarters)
- **Coverage**: ~70% of US equities (major stocks)
- **Update Frequency**: Daily

#### Data Extracted from Yahoo Finance
```python
stock = yf.Ticker("AAPL")

# Earnings calendar
earnings_dates = stock.earnings_dates  # Next 4 quarters
next_earnings_date = earnings_dates.index[0]
consensus_estimate = earnings_dates['EPS Estimate'].iloc[0]

# Analyst targets
analyst_info = stock.analyst_price_target
num_analysts = analyst_info.get('numberOfAnalystOpinions', 0)
```

#### Limitations
- ❌ Limited historical consensus (only recent quarters)
- ❌ ~70% coverage (missing small-cap stocks)
- ❌ No detailed revision history
- ❌ Rate limiting (use with caution)

### Coverage Matrix

| Feature | EDGAR | Yahoo | Combined Coverage |
|---------|-------|-------|-------------------|
| Actual EPS | ✅ 100% | ❌ | ✅ 100% |
| Actual Revenue | ✅ 100% | ❌ | ✅ 100% |
| Consensus Estimate | ❌ | ✅ 70% | ✅ 70% |
| Earnings Surprise | ❌ | ❌ | ✅ 70% (calculated) |
| Earnings Calendar | ✅ 100% | ✅ 100% | ✅ 100% |
| Company Guidance | ✅ 80% | ❌ | ✅ 80% |
| Analyst Revisions | ❌ | ⚠️ Limited | ⚠️ 30% |

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)

#### Day 1: Data Fetchers
**Tasks**:
1. Implement `edgar_fetcher.py` with `edgartools`
2. Implement `yahoo_fetcher.py` with `yfinance`
3. Create unit tests for both fetchers
4. Add error handling and rate limiting

**Deliverables**:
- ✅ `EarningsDataFetcher` class (EDGAR + Yahoo)
- ✅ Unit tests: `test_edgar_fetcher.py`, `test_yahoo_fetcher.py`
- ✅ Type-checked with mypy --strict
- ✅ Linted with ruff

**Acceptance Criteria**:
- Fetch 10-Q for AAPL and extract EPS/revenue
- Fetch Yahoo consensus for AAPL
- Handle missing data gracefully (no exceptions)
- All tests pass

#### Day 2: Database Schema & Storage
**Tasks**:
1. Design PostgreSQL schema (`ml/schema/earnings.sql`)
2. Implement `EarningsStore` with Protocol-first design
3. Create `DummyEarningsStore` fallback
4. Write integration tests with PostgreSQL

**Deliverables**:
- ✅ `ml/schema/earnings.sql` (3 tables: actuals, estimates, calendar)
- ✅ `EarningsStore` + `DummyEarningsStore`
- ✅ `EarningsStoreProtocol` in `ml/stores/protocols.py`
- ✅ Integration tests

**Acceptance Criteria**:
- Schema creates successfully in PostgreSQL
- Write/read earnings data
- Temporal queries work (point-in-time)
- All indexes created and used

### Phase 2: Feature Engineering (Days 3-4)

#### Day 3: Earnings Features
**Tasks**:
1. Implement `earnings_features.py` with surprise, growth, momentum calculations
2. Create batch and real-time computation paths
3. Add parity validation tests
4. Implement earnings calendar features

**Deliverables**:
- ✅ `compute_earnings_surprise()` - batch and incremental
- ✅ `compute_earnings_growth()` - YoY, QoQ growth
- ✅ `compute_earnings_momentum()` - Beat/miss streaks
- ✅ `compute_calendar_features()` - Days to earnings, announcement flags
- ✅ Parity tests (rtol=1e-10)

**Acceptance Criteria**:
- Surprise = (Actual - Estimate) / Estimate
- Growth = (EPS_Q0 - EPS_Q-4) / EPS_Q-4
- Batch vs incremental parity verified
- All tests pass

#### Day 4: Pipeline Integration
**Tasks**:
1. Create `earnings_transforms.py` with `TransformSpec` classes
2. Integrate with existing `PipelineRunner`
3. Add to `ml/features/__init__.py` exports
4. Update TFT dataset builder

**Deliverables**:
- ✅ `EarningsSurpriseTransformSpec`
- ✅ `EarningsGrowthTransformSpec`
- ✅ `EarningsCalendarTransformSpec`
- ✅ Pipeline integration tests

**Acceptance Criteria**:
- TransformSpec can be added to PipelineSpec
- Features computed in TFT dataset
- Exports alphabetically sorted
- Type-checked and linted

### Phase 3: Point-in-Time Caching (Day 5)

#### Day 5: Temporal Correctness
**Tasks**:
1. Implement `earnings_cache.py` for point-in-time lookups
2. Ensure no look-ahead bias in backtesting
3. Add cache invalidation logic
4. Write temporal correctness tests

**Deliverables**:
- ✅ `EarningsCache` class with temporal queries
- ✅ Point-in-time correctness validation
- ✅ Cache warming for backtesting
- ✅ Temporal unit tests

**Acceptance Criteria**:
- At time T, only data with filing_date < T is visible
- No look-ahead bias (verified in tests)
- Cache hit rate >90% for sequential access
- Performance: <1ms per lookup

### Phase 4: Testing & Documentation (Days 6-7)

#### Day 6: Integration Testing
**Tasks**:
1. End-to-end pipeline test (EDGAR → PostgreSQL → Features → TFT)
2. Performance benchmarks
3. Data quality validation
4. Edge case testing

**Deliverables**:
- ✅ `test_earnings_pipeline.py` (integration test)
- ✅ Performance benchmarks (<50ms batch, <5ms incremental)
- ✅ Data quality checks (no nulls, valid ranges)
- ✅ Edge case tests (missing estimates, filing delays)

**Acceptance Criteria**:
- Full pipeline runs for 10 instruments × 8 quarters
- All features computed correctly
- Performance targets met
- Zero data quality issues

#### Day 7: Documentation & Code Review
**Tasks**:
1. Write user documentation (`ml/features/README_EARNINGS.md`)
2. Add inline docstrings (100% coverage)
3. Update TFT dataset structure docs
4. Final code review

**Deliverables**:
- ✅ `README_EARNINGS.md` with usage examples
- ✅ 100% docstring coverage
- ✅ Updated `TFT_DATASET_STRUCTURE.md`
- ✅ Code review passed

**Acceptance Criteria**:
- All public APIs documented
- Examples run successfully
- No linting/type errors
- Ready for merge

---

## Feature Specifications

### Per-Instrument Earnings Features (8 core features)

#### 1. Earnings Surprise Features (3 features)
```python
# From EDGAR actual vs Yahoo estimate
"eps_surprise_q0": float           # Actual - Estimate (dollars)
"eps_surprise_pct_q0": float       # (Actual - Estimate) / Estimate × 100
"revenue_surprise_pct_q0": float   # Same for revenue

# Calculation (earnings_features.py)
def compute_earnings_surprise(actual: float, estimate: float) -> dict:
    """
    Calculate earnings surprise metrics.

    Parameters
    ----------
    actual : float
        Actual reported EPS from EDGAR 10-Q
    estimate : float
        Consensus estimate from Yahoo Finance

    Returns
    -------
    dict
        Surprise in dollars and percentage
    """
    surprise = actual - estimate
    surprise_pct = (surprise / estimate) * 100 if estimate != 0 else 0.0

    return {
        "eps_surprise_q0": surprise,
        "eps_surprise_pct_q0": surprise_pct,
    }
```

#### 2. Earnings Growth Features (2 features)
```python
# Year-over-year and quarter-over-quarter growth
"eps_growth_yoy": float            # (EPS_Q0 - EPS_Q-4) / EPS_Q-4 × 100
"eps_growth_qoq": float            # (EPS_Q0 - EPS_Q-1) / EPS_Q-1 × 100

# Calculation (earnings_features.py)
def compute_earnings_growth(eps_history: list[float]) -> dict:
    """
    Calculate YoY and QoQ EPS growth.

    Parameters
    ----------
    eps_history : list[float]
        EPS for last 5 quarters [Q0, Q-1, Q-2, Q-3, Q-4]

    Returns
    -------
    dict
        YoY and QoQ growth percentages
    """
    eps_q0 = eps_history[0]
    eps_q1 = eps_history[1]
    eps_q4 = eps_history[4]

    yoy_growth = ((eps_q0 - eps_q4) / eps_q4 * 100) if eps_q4 != 0 else 0.0
    qoq_growth = ((eps_q0 - eps_q1) / eps_q1 * 100) if eps_q1 != 0 else 0.0

    return {
        "eps_growth_yoy": yoy_growth,
        "eps_growth_qoq": qoq_growth,
    }
```

#### 3. Earnings Momentum Features (2 features)
```python
# Beat/miss streak and volatility
"earnings_beat_streak": int        # Consecutive quarters beating estimates
"eps_volatility_4q": float         # Std dev of last 4 quarters

# Calculation (earnings_features.py)
def compute_earnings_momentum(surprises: list[float], eps_history: list[float]) -> dict:
    """
    Calculate earnings momentum indicators.

    Parameters
    ----------
    surprises : list[float]
        Earnings surprises for last N quarters
    eps_history : list[float]
        EPS for last 4 quarters

    Returns
    -------
    dict
        Beat streak and EPS volatility
    """
    import numpy as np

    # Count consecutive beats (positive surprises)
    beat_streak = 0
    for surprise in surprises:
        if surprise > 0:
            beat_streak += 1
        else:
            break

    # EPS volatility (coefficient of variation)
    eps_std = np.std(eps_history)
    eps_mean = np.mean(eps_history)
    eps_volatility = (eps_std / eps_mean) if eps_mean != 0 else 0.0

    return {
        "earnings_beat_streak": beat_streak,
        "eps_volatility_4q": eps_volatility,
    }
```

#### 4. Earnings Calendar Features (1 feature)
```python
# Days until next earnings announcement
"days_to_next_earnings": int       # Calendar days until next 10-Q due

# Calculation (earnings_features.py)
def compute_calendar_features(ticker: str, current_date: datetime) -> dict:
    """
    Calculate earnings calendar features.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    current_date : datetime
        Current date for calculation

    Returns
    -------
    dict
        Days to next earnings
    """
    # Get next earnings date from Yahoo or estimate from last filing
    next_earnings_date = get_next_earnings_date(ticker)

    if next_earnings_date:
        days_to_earnings = (next_earnings_date - current_date).days
    else:
        # Estimate: 90 days after last quarter end
        days_to_earnings = estimate_next_earnings_days(ticker, current_date)

    return {
        "days_to_next_earnings": days_to_earnings,
    }
```

### Optional Advanced Features (Future Enhancement)

#### 5. Analyst Revision Features (if Refinitiv available)
```python
"analyst_upgrades_30d": int        # Number of estimate upgrades
"analyst_downgrades_30d": int      # Number of estimate downgrades
"consensus_revision_30d": float    # Change in consensus estimate
```

#### 6. Company Guidance Features (from 8-K filings)
```python
"guidance_eps_q1": float           # Company's EPS guidance
"guidance_vs_consensus": float     # Guidance - Consensus
```

#### 7. Aggregate Market Features (cross-sectional)
```python
"sp500_avg_surprise": float        # Average surprise across S&P 500
"sector_earnings_growth": float    # Sector average YoY growth
"market_beat_rate": float          # % of companies beating estimates
```

---

## Database Schema

### SQL Schema Definition

```sql
-- ml/schema/earnings.sql

-- Table 1: Actual Earnings (from EDGAR)
CREATE TABLE IF NOT EXISTS ml.earnings_actuals (
    ticker VARCHAR(20) NOT NULL,
    period_end DATE NOT NULL,           -- Quarter end date (e.g., 2024-09-30)
    filing_date DATE NOT NULL,          -- Date 10-Q was filed
    ts_event BIGINT NOT NULL,           -- Filing date in nanoseconds (for point-in-time)
    ts_init BIGINT NOT NULL,            -- Record creation timestamp

    -- Actual results (from EDGAR XBRL)
    eps_basic DOUBLE PRECISION,
    eps_diluted DOUBLE PRECISION,
    revenue DOUBLE PRECISION,           -- In dollars
    net_income DOUBLE PRECISION,
    operating_income DOUBLE PRECISION,
    shares_outstanding BIGINT,

    -- Metadata
    filing_type VARCHAR(10),            -- '10-Q' or '10-K'
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,             -- 1, 2, 3, 4
    data_source VARCHAR(20) DEFAULT 'EDGAR',

    PRIMARY KEY (ticker, period_end),

    -- Indexes for performance
    INDEX idx_earnings_actuals_ts_event (ts_event),
    INDEX idx_earnings_actuals_ticker (ticker),
    INDEX idx_earnings_actuals_filing_date (filing_date)
);

COMMENT ON TABLE ml.earnings_actuals IS 'Historical earnings actuals from SEC EDGAR filings';
COMMENT ON COLUMN ml.earnings_actuals.ts_event IS 'Filing date in nanoseconds for point-in-time queries';

-- Table 2: Earnings Estimates (from Yahoo Finance)
CREATE TABLE IF NOT EXISTS ml.earnings_estimates (
    ticker VARCHAR(20) NOT NULL,
    estimate_date DATE NOT NULL,        -- Date estimate was recorded
    period_end DATE NOT NULL,           -- Quarter being estimated
    ts_event BIGINT NOT NULL,           -- Estimate date in nanoseconds
    ts_init BIGINT NOT NULL,

    -- Consensus estimates
    eps_consensus DOUBLE PRECISION,
    revenue_consensus DOUBLE PRECISION,
    num_analysts INTEGER,

    -- Metadata
    data_source VARCHAR(20) DEFAULT 'YAHOO',

    PRIMARY KEY (ticker, estimate_date, period_end),

    -- Indexes
    INDEX idx_earnings_estimates_ts_event (ts_event),
    INDEX idx_earnings_estimates_ticker (ticker),
    INDEX idx_earnings_estimates_period (period_end)
);

COMMENT ON TABLE ml.earnings_estimates IS 'Consensus earnings estimates from Yahoo Finance';

-- Table 3: Earnings Calendar (upcoming announcements)
CREATE TABLE IF NOT EXISTS ml.earnings_calendar (
    ticker VARCHAR(20) NOT NULL,
    earnings_date TIMESTAMP NOT NULL,   -- Scheduled announcement date/time
    period_end DATE NOT NULL,           -- Quarter being reported
    ts_event BIGINT NOT NULL,           -- Calendar update time in nanoseconds
    ts_init BIGINT NOT NULL,

    -- Estimates for upcoming earnings
    eps_consensus DOUBLE PRECISION,
    revenue_consensus DOUBLE PRECISION,
    num_analysts INTEGER,

    -- Status
    is_confirmed BOOLEAN DEFAULT FALSE, -- Whether date is confirmed
    time_of_day VARCHAR(20),            -- 'BMO' (before market), 'AMC' (after market)

    PRIMARY KEY (ticker, earnings_date),

    -- Indexes
    INDEX idx_earnings_calendar_date (earnings_date),
    INDEX idx_earnings_calendar_ticker (ticker)
);

COMMENT ON TABLE ml.earnings_calendar IS 'Upcoming earnings announcements calendar';

-- View: Combined Earnings (actuals + estimates)
CREATE OR REPLACE VIEW ml.earnings_combined AS
SELECT
    a.ticker,
    a.period_end,
    a.filing_date,
    a.eps_diluted AS eps_actual,
    a.revenue AS revenue_actual,
    e.eps_consensus AS eps_estimate,
    e.revenue_consensus AS revenue_estimate,
    -- Calculate surprises
    (a.eps_diluted - e.eps_consensus) AS eps_surprise,
    ((a.eps_diluted - e.eps_consensus) / NULLIF(e.eps_consensus, 0) * 100) AS eps_surprise_pct,
    a.fiscal_year,
    a.fiscal_quarter
FROM ml.earnings_actuals a
LEFT JOIN ml.earnings_estimates e
    ON a.ticker = e.ticker
    AND a.period_end = e.period_end
    AND e.estimate_date <= a.filing_date  -- Point-in-time estimate
ORDER BY a.ticker, a.period_end DESC;

COMMENT ON VIEW ml.earnings_combined IS 'Actuals joined with estimates for surprise calculation';
```

### Migration Script

```bash
#!/bin/bash
# scripts/migrate_earnings_schema.sh

# Run earnings schema migration
psql -h localhost -U postgres -d nautilus_trader -f ml/schema/earnings.sql

# Verify tables created
psql -h localhost -U postgres -d nautilus_trader -c "
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'ml'
    AND table_name LIKE 'earnings_%'
"

echo "✅ Earnings schema migration complete"
```

---

## Pipeline Integration

### TransformSpec Classes

```python
# ml/features/earnings/earnings_transforms.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ml.features.pipeline import TransformSpec

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class EarningsSurpriseTransformSpec(TransformSpec):
    """
    Transform for earnings surprise calculation.

    Computes surprise (actual - estimate) and percentage surprise
    for EPS and revenue.
    """

    name: str = "earnings_surprise"
    ticker: str = ""
    lookback_quarters: int = 1

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [
            f"eps_surprise_q0_{self.ticker}",
            f"eps_surprise_pct_q0_{self.ticker}",
            f"revenue_surprise_pct_q0_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsGrowthTransformSpec(TransformSpec):
    """
    Transform for earnings growth calculation.

    Computes YoY and QoQ growth rates for EPS.
    """

    name: str = "earnings_growth"
    ticker: str = ""
    lookback_quarters: int = 5

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [
            f"eps_growth_yoy_{self.ticker}",
            f"eps_growth_qoq_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsMomentumTransformSpec(TransformSpec):
    """
    Transform for earnings momentum indicators.

    Computes beat streak and EPS volatility.
    """

    name: str = "earnings_momentum"
    ticker: str = ""
    lookback_quarters: int = 4

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [
            f"earnings_beat_streak_{self.ticker}",
            f"eps_volatility_4q_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsCalendarTransformSpec(TransformSpec):
    """
    Transform for earnings calendar features.

    Computes days to next earnings announcement.
    """

    name: str = "earnings_calendar"
    ticker: str = ""

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [
            f"days_to_next_earnings_{self.ticker}",
        ]
```

### Usage in Pipeline

```python
# Example: Configure earnings features for ES (S&P 500 futures)
from ml.features.pipeline import PipelineSpec
from ml.features.earnings.earnings_transforms import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsCalendarTransformSpec,
)

# For equity index futures, use aggregate S&P 500 earnings
pipeline = PipelineSpec(
    name="es_futures_with_earnings",
    version="1.0.0",
    transforms=[
        # Existing transforms
        # ... (market, macro, cross-asset transforms)

        # Add earnings transforms
        EarningsSurpriseTransformSpec(ticker="SPY"),  # S&P 500 ETF as proxy
        EarningsGrowthTransformSpec(ticker="SPY"),
        EarningsMomentumTransformSpec(ticker="SPY"),
        EarningsCalendarTransformSpec(ticker="SPY"),
    ]
)

# For individual stocks
stock_pipeline = PipelineSpec(
    name="aapl_with_earnings",
    version="1.0.0",
    transforms=[
        EarningsSurpriseTransformSpec(ticker="AAPL"),
        EarningsGrowthTransformSpec(ticker="AAPL"),
        EarningsMomentumTransformSpec(ticker="AAPL"),
        EarningsCalendarTransformSpec(ticker="AAPL"),
    ]
)
```

---

## Testing Strategy

### Unit Tests

#### Test Coverage Requirements
- **Code coverage**: ≥90% for all earnings modules
- **Type checking**: mypy --strict passes with zero errors
- **Linting**: ruff check passes with zero violations
- **Parity**: Batch vs incremental rtol ≤ 1e-10 (where applicable)

#### Test Files

**1. `test_edgar_fetcher.py`**
```python
import pytest
from ml.data.earnings.edgar_fetcher import EdgarFetcher

class TestEdgarFetcher:
    def test_fetch_10q_aapl(self):
        """Test fetching AAPL 10-Q and parsing XBRL."""
        fetcher = EdgarFetcher()
        earnings = fetcher.fetch_earnings("AAPL", quarters=1, form="10-Q")

        assert len(earnings) == 1
        assert earnings[0].eps_diluted is not None
        assert earnings[0].revenue > 0
        assert earnings[0].ticker == "AAPL"

    def test_handle_missing_ticker(self):
        """Test graceful handling of invalid ticker."""
        fetcher = EdgarFetcher()
        earnings = fetcher.fetch_earnings("INVALID_TICKER_XYZ", quarters=1)

        assert earnings == []  # Empty list, no exception

    def test_xbrl_parsing_edge_cases(self):
        """Test XBRL parsing with non-standard tags."""
        fetcher = EdgarFetcher()
        # Test with company that uses custom XBRL tags
        earnings = fetcher.fetch_earnings("TSLA", quarters=1)

        assert earnings[0].eps_diluted is not None or earnings[0].eps_basic is not None
```

**2. `test_yahoo_fetcher.py`**
```python
import pytest
from ml.data.earnings.yahoo_fetcher import YahooFetcher

class TestYahooFetcher:
    def test_fetch_consensus_aapl(self):
        """Test fetching consensus estimate from Yahoo."""
        fetcher = YahooFetcher()
        consensus = fetcher.fetch_consensus("AAPL")

        assert consensus.eps_estimate is not None
        assert consensus.next_earnings_date is not None

    def test_handle_rate_limiting(self):
        """Test rate limiting doesn't cause failures."""
        fetcher = YahooFetcher(rate_limit_delay=1.0)

        # Fetch multiple tickers rapidly
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            consensus = fetcher.fetch_consensus(ticker)
            assert consensus is not None
```

**3. `test_earnings_features.py`**
```python
import pytest
import numpy as np
from ml.features.earnings.earnings_features import (
    compute_earnings_surprise,
    compute_earnings_growth,
    compute_earnings_momentum,
)

class TestEarningsFeatures:
    def test_earnings_surprise_calculation(self):
        """Test surprise calculation accuracy."""
        actual = 2.52
        estimate = 2.45

        surprise = compute_earnings_surprise(actual, estimate)

        assert surprise["eps_surprise_q0"] == pytest.approx(0.07, abs=1e-10)
        assert surprise["eps_surprise_pct_q0"] == pytest.approx(2.857, rel=1e-3)

    def test_earnings_growth_yoy(self):
        """Test YoY growth calculation."""
        eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]  # Q0, Q-1, Q-2, Q-3, Q-4

        growth = compute_earnings_growth(eps_history)

        expected_yoy = ((2.52 - 2.20) / 2.20) * 100  # 14.54%
        assert growth["eps_growth_yoy"] == pytest.approx(expected_yoy, rel=1e-3)

    def test_earnings_momentum_beat_streak(self):
        """Test consecutive beat streak counting."""
        surprises = [0.07, 0.05, 0.03, -0.02, 0.01]  # Last 5 quarters
        eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]

        momentum = compute_earnings_momentum(surprises, eps_history)

        assert momentum["earnings_beat_streak"] == 3  # First 3 are positive
```

### Integration Tests

**`test_earnings_pipeline.py`**
```python
import pytest
from datetime import UTC, datetime

from ml.features.earnings import compute_earnings_features
from ml.tests.utils.earnings_facade import build_test_data_store


@pytest.mark.integration
class TestEarningsPipeline:
    def test_full_pipeline_aapl(self, postgres_connection):
        """Exercise the DataStore facade with progressive fallbacks enabled."""

        data_store = build_test_data_store(connection_string=postgres_connection)

        data_store.write_earnings_actual(
            ticker="AAPL",
            period_end="2024-03-31",
            filing_date="2024-05-02",
            eps_diluted=1.52,
            revenue=94_900_000_000,
            ts_event=int(datetime(2024, 5, 2, tzinfo=UTC).timestamp() * 1_000_000_000),
            ts_init=int(datetime.now(tz=UTC).timestamp() * 1_000_000_000),
        )
        data_store.write_earnings_estimate(
            ticker="AAPL",
            period_end="2024-03-31",
            estimate_date="2024-04-20",
            eps_consensus=1.48,
            ts_event=int(datetime(2024, 4, 20, tzinfo=UTC).timestamp() * 1_000_000_000),
            ts_init=int(datetime.now(tz=UTC).timestamp() * 1_000_000_000),
        )

        features = compute_earnings_features(
            "AAPL",
            as_of_date="2024-06-01",
            data_store=data_store,
        )

        assert "eps_surprise_q0" in features
        assert "eps_growth_yoy" in features
        assert features["eps_surprise_pct_q0"] is not None

    def test_point_in_time_correctness(self, postgres_connection):
        """Point-in-time reads go through the same facade helpers."""
        data_store = build_test_data_store(connection_string=postgres_connection)

        data_store.write_earnings_actual(
            ticker="AAPL",
            period_end="2023-12-31",
            filing_date="2024-02-01",
            eps_diluted=1.40,
            revenue=81_800_000_000,
            ts_event=int(datetime(2024, 2, 1, tzinfo=UTC).timestamp() * 1_000_000_000),
            ts_init=int(datetime.now(tz=UTC).timestamp() * 1_000_000_000),
        )

        jan_records = data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000),
        )
        may_records = data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=int(datetime(2024, 5, 1, tzinfo=UTC).timestamp() * 1_000_000_000),
        )

        assert jan_records[0]["period_end"] == "2023-09-30"
        assert may_records[0]["period_end"] == "2023-12-31"
```

### Performance Tests

**`test_earnings_performance.py`**
```python
import pytest
import time
from ml.features.earnings import compute_earnings_surprise_incremental

class TestEarningsPerformance:
    def test_incremental_surprise_latency(self):
        """Test incremental surprise calculation meets <5ms requirement."""
        latencies = []

        for _ in range(1000):
            start = time.perf_counter_ns()
            compute_earnings_surprise_incremental(2.52, 2.45)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)  # Convert to ms

        p99 = np.percentile(latencies, 99)
        assert p99 < 5.0, f"P99 latency {p99}ms exceeds 5ms target"
```

---

## Performance Requirements

### Latency Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| EDGAR fetch (10-Q) | <2s | Per filing download |
| Yahoo fetch (consensus) | <500ms | Per ticker |
| Database write (100 records) | <100ms | Batch insert |
| Database read (point-in-time) | <1ms | Single query |
| Feature computation (incremental) | <5ms | Per instrument |
| Feature computation (batch) | <50ms | Per 100 instruments |

### Scalability Targets

- **Instruments supported**: 95 (initial), 500+ (future)
- **Historical depth**: 10 years of quarterly data
- **Backtest speed**: Process 1 year of data for 95 instruments in <5 minutes
- **Storage**: <100MB per year per instrument (compressed)

### Resource Usage

- **Memory**: <500MB for 95 instruments × 10 years
- **PostgreSQL**: <1GB storage for earnings tables
- **API calls**:
  - EDGAR: No rate limit (but be respectful, ~1 req/sec)
  - Yahoo Finance: <100 requests/day per IP (free tier)

---

## Future Enhancements

### Phase 2 Enhancements (Optional, Post-MVP)

#### 1. Commercial Data Integration
**Objective**: Upgrade from Yahoo Finance to institutional-grade consensus data

**Options**:
- **Refinitiv I/B/E/S**: Full consensus history, analyst revisions ($15k/year)
- **FactSet Estimates**: Consensus + guidance tracking ($20k/year)
- **Bloomberg Terminal**: Real-time estimates, alerts ($24k/year)

**Benefits**:
- 100% coverage (vs 70% with Yahoo)
- 20+ years of consensus history
- Real-time analyst revisions
- Intraday earnings alerts

#### 2. Alternative Data Sources
**Objective**: Add non-traditional earnings signals

**Sources**:
- **Earnings call transcripts**: NLP sentiment analysis (Alpha Vantage, Seeking Alpha)
- **Social media sentiment**: Twitter/Reddit earnings buzz (Sentiment API)
- **Options market**: Implied earnings move from straddles (CBOE data)
- **Insider trading**: Form 4 filings around earnings (SEC EDGAR)

#### 3. Advanced Features
**Objective**: More sophisticated earnings-based signals

**Features**:
- **Earnings quality score**: Accruals, cash flow quality metrics
- **Estimate dispersion**: Std dev of analyst estimates (uncertainty proxy)
- **Estimate revision momentum**: Acceleration of upgrades/downgrades
- **Earnings leverage**: Sensitivity of stock price to earnings surprises
- **Cross-sectional ranking**: Percentile rank of earnings metrics vs sector

#### 4. Real-Time Earnings Alerts
**Objective**: Trigger strategies on earnings releases

**Implementation**:
- Monitor SEC EDGAR for 8-K filings (Item 2.02)
- Parse earnings data within minutes of filing
- Emit trading signals based on surprise magnitude
- Integrate with Nautilus event-driven architecture

#### 5. Sector/Market Aggregates
**Objective**: Macro earnings signals for index futures

**Features**:
- S&P 500 aggregate EPS growth
- Sector earnings momentum (Tech, Financials, Energy, etc.)
- Market-wide beat/miss rate
- Earnings recession indicators (declining EPS for 2+ quarters)

---

## Migration Checklist

### Pre-Implementation
- [ ] Review and approve this plan
- [ ] Confirm 40-50 instruments need earnings data
- [ ] Verify EDGAR access works (test `edgartools` library)
- [ ] Verify Yahoo Finance access (test `yfinance` library)
- [ ] Allocate 7 days for implementation + testing

### Phase 1: Infrastructure (Days 1-2)
- [ ] Install dependencies: `pip install edgartools yfinance`
- [ ] Implement `edgar_fetcher.py` with XBRL parsing
- [ ] Implement `yahoo_fetcher.py` with consensus fetching
- [ ] Create unit tests for fetchers
- [ ] Design PostgreSQL schema (`earnings.sql`)
- [ ] Implement `EarningsStore` + `DummyEarningsStore`
- [ ] Add `EarningsStoreProtocol` to `ml/stores/protocols.py`
- [ ] Run integration tests with PostgreSQL

### Phase 2: Features (Days 3-4)
- [ ] Implement `earnings_features.py` (surprise, growth, momentum)
- [ ] Create batch and incremental computation paths
- [ ] Write parity validation tests (rtol=1e-10)
- [ ] Implement `earnings_transforms.py` with TransformSpec classes
- [ ] Integrate with `PipelineRunner`
- [ ] Update `ml/features/__init__.py` exports (alphabetically sorted)
- [ ] Update `TFTDatasetBuilder` to include earnings features

### Phase 3: Caching (Day 5)
- [ ] Implement `earnings_cache.py` for point-in-time lookups
- [ ] Add temporal correctness validation
- [ ] Write cache performance tests
- [ ] Verify no look-ahead bias in backtests

### Phase 4: Testing (Days 6-7)
- [ ] Write end-to-end integration test
- [ ] Run performance benchmarks (latency targets)
- [ ] Validate data quality (no nulls, valid ranges)
- [ ] Test edge cases (missing estimates, filing delays)
- [ ] Write user documentation (`README_EARNINGS.md`)
- [ ] Add inline docstrings (100% coverage)
- [ ] Update `TFT_DATASET_STRUCTURE.md`
- [ ] Final code review

### Post-Implementation
- [ ] Merge to main branch
- [ ] Deploy to staging environment
- [ ] Backfill historical earnings data (10 years)
- [ ] Validate TFT training with earnings features
- [ ] Monitor performance in production
- [ ] Gather user feedback
- [ ] Plan Phase 2 enhancements (commercial data, real-time alerts)

---

## Appendix A: Python Dependencies

### Required Libraries
```bash
# Core earnings data
pip install edgartools          # SEC EDGAR API client
pip install yfinance            # Yahoo Finance API client

# Already installed (existing dependencies)
pip install polars              # DataFrame operations
pip install numpy               # Numerical computations
pip install sqlalchemy          # Database ORM
pip install psycopg2-binary     # PostgreSQL driver
pip install pytest              # Testing framework
pip install mypy                # Type checking
pip install ruff                # Linting
```

### Library Versions (Recommended)
```txt
# requirements-earnings.txt
edgartools>=2.0.0
yfinance>=0.2.40
polars>=0.20.0
numpy>=1.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.9
pytest>=7.4.0
mypy>=1.8.0
ruff>=0.1.0
```

---

## Appendix B: Example Code

### Complete Example: Facade-First Dataset Build

```bash
uv run --active --no-sync python -m ml.cli.build_tft_dataset \
  --data_dir data/tier1 \
  --symbols AAPL \
  --out_dir builds/tft_aapl \
  --include_macro --macro_lag_days 1 \
  --include_earnings --earnings_lag_days 2 \
  --lookback_periods 30 --horizon_minutes 15 --threshold 0.001
```

1. The CLI resolves `DataStore` configuration (respecting `MessageBusConfig` + environment) and initializes the PostgreSQL → `FileEarningsStore` → `DummyEarningsStore` fallback cascade.
2. Earnings I/O go through the facade helpers (`write_earnings_actual`, `get_earnings_actuals_at_or_before`, `get_earnings_estimate_at_or_before`) to preserve point-in-time correctness.
3. Outputs land under `builds/tft_aapl/` with the earnings columns joined (e.g., `eps_surprise_q0_AAPL`, `eps_growth_yoy_AAPL`, `is_earnings_available`).
4. `ml_fallback_activations_total{component="data_store",level}` increments whenever a fallback stage is activated; capture this from logs or Prometheus when running drills.

### Operations Checklist

- **Environment**: export `ML_FILE_STORE_PATH=/srv/nautilus/ml/file_store` (or similar) so the
  file fallback has a persistent root. Set `NAUTILUS_REGISTRY_DB_URL` when bootstrapping the
  PostgreSQL-backed registry.
- **Registry**: run `uv run --active --no-sync python -m ml.registry.bootstrap_datasets` to seed
  earnings manifests/contracts (`--backend postgres` for shared registries).
- **Metrics**: verify `ml_fallback_activations_total{component="data_store"}` via Prometheus or
  CLI logs whenever failovers are drilled.
- **Validation**: execute `make validate-metrics` and `make validate-events` after documentation or
  configuration changes touching earnings flows.

---

## Appendix C: Useful Resources

### SEC EDGAR Resources
- **EDGAR Homepage**: https://www.sec.gov/edgar/searchedgar/companysearch.html
- **EDGAR API Documentation**: https://www.sec.gov/edgar/sec-api-documentation
- **XBRL Taxonomy**: https://xbrl.sec.gov/
- **edgartools Documentation**: https://github.com/dgunning/edgartools

### Yahoo Finance Resources
- **yfinance Documentation**: https://github.com/ranaroussi/yfinance
- **Yahoo Finance API**: https://finance.yahoo.com/

### XBRL Parsing
- **Common XBRL Tags**: https://www.sec.gov/info/edgar/edgartaxonomies.shtml
- **XBRL US GAAP Taxonomy**: https://xbrl.us/home/filers/sec-reporting/

### Earnings Calendar Sources
- **Earnings Whispers**: https://www.earningswhispers.com/calendar
- **Nasdaq Earnings Calendar**: https://www.nasdaq.com/market-activity/earnings

---

## Document Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-02 | Claude Code | Initial plan created |

---

**End of Earnings Integration Plan**

For questions or clarifications, refer to:
- Architecture: `ml/docs/architecture/universal_patterns_guide.md`
- Coding standards: `ml/docs/development/CODING_STANDARDS.md`
- TFT dataset: `ml/docs/TFT_DATASET_STRUCTURE.md`
