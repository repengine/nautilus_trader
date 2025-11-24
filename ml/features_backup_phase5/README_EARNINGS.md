# Earnings Features - Corporate Fundamentals Integration

## Overview

The earnings features module provides corporate fundamentals integration for Nautilus Trader's ML pipeline. It combines **SEC EDGAR actuals** (100% coverage) with **Yahoo Finance estimates** (~70% coverage) to compute 8 core earnings-based features per equity instrument.

### Key Features

- **Data Sources**: SEC EDGAR (actuals) + Yahoo Finance (estimates)
- **Coverage**: 100% US public companies for actuals, ~70% for consensus
- **Features**: 8 core features across 4 categories
- **Performance**: Hot path <5ms P99, cold path <50ms for 100 instruments
- **Point-in-Time Correctness**: Prevents look-ahead bias in backtesting

---

## Feature Categories

### 1. Earnings Surprise Features (3 features)

Measures the gap between actual reported earnings and analyst consensus estimates.

**Features:**
- `eps_surprise_q0_{ticker}`: Dollar surprise (Actual - Estimate)
- `eps_surprise_pct_q0_{ticker}`: Percentage surprise
- `revenue_surprise_pct_q0_{ticker}`: Revenue percentage surprise

**Formula:**
```python
eps_surprise = actual_eps - consensus_eps
eps_surprise_pct = (eps_surprise / consensus_eps) * 100
```

**Example:**
- Actual EPS: $2.52
- Consensus: $2.45
- Surprise: $0.07 (2.86%)

### 2. Earnings Growth Features (2 features)

Calculates year-over-year and quarter-over-quarter EPS growth rates.

**Features:**
- `eps_growth_yoy_{ticker}`: Year-over-year growth (%)
- `eps_growth_qoq_{ticker}`: Quarter-over-quarter growth (%)

**Formula:**
```python
yoy_growth = ((eps_q0 - eps_q4) / eps_q4) * 100  # Compare to same quarter last year
qoq_growth = ((eps_q0 - eps_q1) / eps_q1) * 100  # Compare to previous quarter
```

**Example:**
- Q3 2024 EPS: $2.52
- Q3 2023 EPS: $2.20
- YoY Growth: 14.5%

### 3. Earnings Momentum Features (2 features)

Tracks consecutive earnings beats and EPS volatility.

**Features:**
- `earnings_beat_streak_{ticker}`: Consecutive quarters beating estimates (integer)
- `eps_volatility_4q_{ticker}`: 4-quarter EPS volatility (coefficient of variation)

**Formula:**
```python
beat_streak = count_consecutive_positive_surprises()
eps_volatility = np.std(eps_last_4q) / np.mean(eps_last_4q)
```

**Example:**
- Last 4 surprises: [+2.5%, +3.1%, +1.8%, +2.2%]
- Beat streak: 4
- EPS volatility: 0.12

### 4. Earnings Calendar Features (1 feature)

Days until next earnings announcement.

**Features:**
- `days_to_next_earnings_{ticker}`: Calendar days until next 10-Q filing

**Formula:**
```python
days_to_earnings = (next_earnings_date - current_date).days
```

**Example:**
- Current date: 2025-01-15
- Next earnings: 2025-02-28
- Days to earnings: 44

---

## Usage Examples

### Example 1: Single Instrument

```python
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsCalendarTransformSpec,
)

# Define transforms for AAPL
ticker = "AAPL"

surprise_spec = EarningsSurpriseTransformSpec(ticker=ticker)
growth_spec = EarningsGrowthTransformSpec(ticker=ticker, lookback_quarters=5)
momentum_spec = EarningsMomentumTransformSpec(ticker=ticker, lookback_quarters=4)
calendar_spec = EarningsCalendarTransformSpec(ticker=ticker)

# Get feature names
print(surprise_spec.compute_feature_names())
# Output: ['eps_surprise_q0_AAPL', 'eps_surprise_pct_q0_AAPL', 'revenue_surprise_pct_q0_AAPL']

print(growth_spec.compute_feature_names())
# Output: ['eps_growth_yoy_AAPL', 'eps_growth_qoq_AAPL']

print(momentum_spec.compute_feature_names())
# Output: ['earnings_beat_streak_AAPL', 'eps_volatility_4q_AAPL']

print(calendar_spec.compute_feature_names())
# Output: ['days_to_next_earnings_AAPL']
```

### Example 2: Multi-Instrument Portfolio

```python
from ml.features.pipeline import PipelineSpec
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
)

# Define pipeline for multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

transforms = []
for ticker in tickers:
    transforms.append(EarningsSurpriseTransformSpec(ticker=ticker))
    transforms.append(EarningsGrowthTransformSpec(ticker=ticker))

pipeline = PipelineSpec(
    name="multi_stock_earnings",
    version="1.0.0",
    transforms=transforms,
)

# Total features: 5 tickers × (3 surprise + 2 growth) = 25 earnings features
```

### Example 3: Incremental (Hot Path) Computation

```python
from ml.features.earnings import (
    compute_earnings_surprise_incremental,
    compute_earnings_growth_incremental,
    compute_earnings_momentum_incremental,
    compute_calendar_features_incremental,
)
from datetime import datetime, timedelta

# Surprise calculation (O(1), <5ms P99)
actual_eps = 2.52
consensus_eps = 2.45
surprise = compute_earnings_surprise_incremental(actual_eps, consensus_eps)
print(f"EPS Surprise: ${surprise['eps_surprise_q0']:.2f} ({surprise['eps_surprise_pct_q0']:.2f}%)")

# Growth calculation (O(1), <5ms P99)
eps_q0, eps_q1, eps_q4 = 2.52, 2.45, 2.20
growth = compute_earnings_growth_incremental(eps_q0, eps_q1, eps_q4)
print(f"YoY Growth: {growth['eps_growth_yoy']:.2f}%")

# Momentum calculation (O(1), <5ms P99)
surprises = [0.07, 0.05, 0.03, -0.02]
eps_history = [2.52, 2.45, 2.38, 2.30]
momentum = compute_earnings_momentum_incremental(surprises, eps_history)
print(f"Beat Streak: {momentum['earnings_beat_streak']} quarters")

# Calendar calculation (O(1), <5ms P99)
current_date = datetime.now()
next_earnings = current_date + timedelta(days=45)
calendar = compute_calendar_features_incremental(current_date, next_earnings)
print(f"Days to Earnings: {calendar['days_to_next_earnings']}")
```

### Example 4: Batch (Cold Path) Computation

```python
from ml.features.earnings import (
    compute_earnings_surprise_batch,
    compute_earnings_growth_batch,
    compute_earnings_momentum_batch,
)
import numpy as np

# Batch surprise for 100 instruments (<50ms)
actuals = np.random.uniform(1.0, 3.0, size=100)
estimates = actuals - np.random.uniform(-0.1, 0.1, size=100)
surprises = compute_earnings_surprise_batch(actuals, estimates)

# Batch growth for single instrument with 5-quarter history
eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]
growth = compute_earnings_growth_batch(eps_history)

# Batch momentum
surprises_list = [0.07, 0.05, 0.03, -0.02]
momentum = compute_earnings_momentum_batch(surprises_list, eps_history[:4])
```

---

## Point-in-Time Correctness

### Why Point-in-Time Matters

In backtesting, using future information (look-ahead bias) leads to unrealistic performance. The earnings module ensures **strict point-in-time correctness** by:

1. **Timestamping all data**: Every record has `ts_event` (filing date) and `ts_init` (ingestion time)
2. **Temporal queries**: Only data with `ts_event < as_of_ts` is visible
3. **Estimate tracking**: Uses the most recent estimate before actual filing

### Example: Preventing Look-Ahead Bias

```python
from ml.stores.data_store import DataStore

store = DataStore(connection_string="postgresql://...")

# Backtest as of January 1, 2024 (before Q1 2024 filing)
actuals_jan = store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=1704067200000000000,  # 2024-01-01 in nanoseconds
)
# Result: Only Q4 2023 and earlier (Q1 2024 filed in May)

# Backtest as of May 1, 2024 (after Q1 2024 filing)
actuals_may = store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=1714521600000000000,  # 2024-05-01 in nanoseconds
)
# Result: Q1 2024 and earlier (now includes Q1 2024)
```

---

## Performance

### Hot Path (Incremental) SLAs

All incremental functions meet strict performance requirements:

| Function | P99 Latency | Complexity | Allocations |
|----------|-------------|------------|-------------|
| `compute_earnings_surprise_incremental` | <5ms | O(1) | Zero after warmup |
| `compute_earnings_growth_incremental` | <5ms | O(1) | Zero after warmup |
| `compute_earnings_momentum_incremental` | <5ms | O(1) | Zero after warmup |
| `compute_calendar_features_incremental` | <5ms | O(1) | Zero after warmup |

### Cold Path (Batch) SLAs

Batch operations use vectorized numpy for efficiency:

| Function | Target | Measured |
|----------|--------|----------|
| `compute_earnings_surprise_batch` (100 instruments) | <50ms | ~15ms |
| `compute_earnings_growth_batch` (single, 5 quarters) | <10ms | ~2ms |
| Cache lookup | <1ms P99 | ~0.3ms |

### Validation

Run performance benchmarks:

```bash
# Run performance tests
poetry run pytest ml/tests/performance/test_earnings_performance.py -v

# Expected output:
# ✅ PASS: Incremental Surprise P99=2.45ms < 5ms SLA
# ✅ PASS: Incremental Growth P99=1.82ms < 5ms SLA
# ✅ PASS: Batch Surprise 100x=18.32ms < 50ms SLA
# ✅ PASS: Cache Lookup P99=0.34ms < 1ms SLA
```

### Metrics

Incremental earnings metrics are disabled by default to preserve zero-allocation hot paths. Set the
environment variable `ML_EARNINGS_ENABLE_METRICS=1` when you need histogram and counter updates
via `MetricsManager` during debugging or benchmarking.

---

## Troubleshooting

### Common Issues

#### 1. Missing Consensus Estimates

**Problem**: `get_estimates()` returns `None` for some tickers.

**Cause**: Yahoo Finance only covers ~70% of US equities (major stocks).

**Solution**:
- Use default estimate (e.g., prior EPS) when consensus unavailable
- Set surprise to 0.0 when estimate is missing
- Upgrade to commercial data (Refinitiv, FactSet) for 100% coverage

```python
# Graceful handling of missing estimates
estimate = store.get_estimates(ticker, period_end) or {"eps_consensus": None}
eps_estimate = estimate["eps_consensus"] if estimate else eps_actual  # Use actual as fallback
```

#### 2. Point-in-Time Query Returns No Data

**Problem**: `get_actuals(as_of_ts=...)` returns empty list.

**Cause**: `as_of_ts` is before any filings.

**Solution**: Check filing dates and ensure `as_of_ts` is after at least one filing.

```python
# Debug: Check earliest filing
actuals = store.get_actuals(ticker)  # Get all
if actuals:
    earliest_filing = min(a["ts_event"] for a in actuals)
    print(f"Earliest filing: {earliest_filing / 1e9} (Unix timestamp)")
```

#### 3. Slow Performance in Hot Path

**Problem**: Incremental functions exceed 5ms P99 latency.

**Cause**: Not enough warmup iterations or allocations in loop.

**Solution**:
- Run 100+ warmup iterations before timing
- Verify zero allocations with `tracemalloc`
- Check for hidden I/O or logging in hot path

```python
# Proper warmup
for _ in range(100):
    compute_earnings_surprise_incremental(2.52, 2.45)

# Now measure performance
import time
start = time.perf_counter_ns()
result = compute_earnings_surprise_incremental(2.52, 2.45)
latency_ms = (time.perf_counter_ns() - start) / 1_000_000
assert latency_ms < 5.0
```

#### 4. Filing Delays / Restatements

**Problem**: Company files earnings 90+ days after quarter end.

**Cause**: Accounting issues, auditor changes, or restatements.

**Solution**: The store handles this via upsert. Latest filing for a `(ticker, period_end)` always wins.

```python
# Original filing
store.write_actuals(ticker="XYZ", period_end="2024-06-30", eps_diluted=2.00, ...)

# Restated filing (90 days later) - overwrites original
store.write_actuals(ticker="XYZ", period_end="2024-06-30", eps_diluted=1.85, ...)

# Latest value is used
actuals = store.get_actuals("XYZ", period_end="2024-06-30")
assert actuals[0]["eps_diluted"] == 1.85  # Restated value
```

---

## Data Sources

### SEC EDGAR (Primary - Actuals)

- **Library**: `edgartools`
- **Coverage**: 100% US public companies
- **Data**: 10-Q (quarterly), 10-K (annual), 8-K (material events)
- **Update frequency**: T+1 day after filing
- **Cost**: Free

**Installation**:
```bash
pip install edgartools
```

### Yahoo Finance (Secondary - Estimates)

- **Library**: `yfinance`
- **Coverage**: ~70% US equities (major stocks)
- **Data**: Consensus EPS/revenue estimates, earnings calendar
- **Update frequency**: Daily
- **Cost**: Free

**Installation**:
```bash
pip install yfinance
```

---

## API Reference

### Transform Specifications

#### `EarningsSurpriseTransformSpec`

```python
@dataclass(frozen=True)
class EarningsSurpriseTransformSpec:
    name: str = "earnings_surprise"
    ticker: str = ""
    lookback_quarters: int = 1

    def compute_feature_names(self) -> list[str]:
        """Returns: ['eps_surprise_q0_{ticker}', 'eps_surprise_pct_q0_{ticker}', 'revenue_surprise_pct_q0_{ticker}']"""
```

#### `EarningsGrowthTransformSpec`

```python
@dataclass(frozen=True)
class EarningsGrowthTransformSpec:
    name: str = "earnings_growth"
    ticker: str = ""
    lookback_quarters: int = 5

    def compute_feature_names(self) -> list[str]:
        """Returns: ['eps_growth_yoy_{ticker}', 'eps_growth_qoq_{ticker}']"""
```

#### `EarningsMomentumTransformSpec`

```python
@dataclass(frozen=True)
class EarningsMomentumTransformSpec:
    name: str = "earnings_momentum"
    ticker: str = ""
    lookback_quarters: int = 4

    def compute_feature_names(self) -> list[str]:
        """Returns: ['earnings_beat_streak_{ticker}', 'eps_volatility_4q_{ticker}']"""
```

#### `EarningsCalendarTransformSpec`

```python
@dataclass(frozen=True)
class EarningsCalendarTransformSpec:
    name: str = "earnings_calendar"
    ticker: str = ""

    def compute_feature_names(self) -> list[str]:
        """Returns: ['days_to_next_earnings_{ticker}']"""
```

### Incremental Functions (Hot Path)

All incremental functions are O(1) and meet <5ms P99 latency SLA.

```python
def compute_earnings_surprise_incremental(
    actual: float | None,
    estimate: float | None,
) -> dict[str, float]:
    """
    Calculate earnings surprise incrementally.

    Parameters
    ----------
    actual : float | None
        Actual reported EPS
    estimate : float | None
        Consensus EPS estimate

    Returns
    -------
    dict[str, float]
        {"eps_surprise_q0": ..., "eps_surprise_pct_q0": ...}
    """

def compute_earnings_growth_incremental(
    eps_q0: float,
    eps_q1: float,
    eps_q4: float,
) -> dict[str, float]:
    """
    Calculate earnings growth incrementally.

    Parameters
    ----------
    eps_q0 : float
        Current quarter EPS
    eps_q1 : float
        Previous quarter EPS (for QoQ)
    eps_q4 : float
        Same quarter last year EPS (for YoY)

    Returns
    -------
    dict[str, float]
        {"eps_growth_yoy": ..., "eps_growth_qoq": ...}
    """

def compute_earnings_momentum_incremental(
    surprises: list[float],
    eps_history: list[float],
) -> dict[str, float]:
    """
    Calculate earnings momentum incrementally.

    Parameters
    ----------
    surprises : list[float]
        Last N earnings surprises
    eps_history : list[float]
        Last 4 quarters of EPS

    Returns
    -------
    dict[str, float | int]
        {"earnings_beat_streak": int, "eps_volatility_4q": float}
    """

def compute_calendar_features_incremental(
    current_date: datetime,
    next_earnings_date: datetime,
) -> dict[str, int]:
    """
    Calculate earnings calendar features incrementally.

    Parameters
    ----------
    current_date : datetime
        Current date
    next_earnings_date : datetime
        Next scheduled earnings announcement

    Returns
    -------
    dict[str, int]
        {"days_to_next_earnings": ...}
    """
```

### Batch Functions (Cold Path)

All batch functions use vectorized numpy operations.

```python
def compute_earnings_surprise_batch(
    actuals: np.ndarray,
    estimates: np.ndarray,
) -> dict[str, float]:
    """Batch earnings surprise for multiple instruments."""

def compute_earnings_growth_batch(
    eps_history: list[float],
) -> dict[str, float]:
    """Batch earnings growth for single instrument with history."""

def compute_earnings_momentum_batch(
    surprises: list[float],
    eps_history: list[float],
) -> dict[str, float | int]:
    """Batch earnings momentum for single instrument."""

def compute_calendar_features_batch(
    current_dates: np.ndarray,
    next_earnings_dates: np.ndarray,
) -> np.ndarray:
    """Batch calendar features for multiple instruments."""
```

---

## Integration with TFT Dataset

The earnings features integrate seamlessly with the TFT (Temporal Fusion Transformer) dataset builder:

```python
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.features.pipeline import PipelineSpec
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
)

# Define pipeline with earnings features
pipeline = PipelineSpec(
    name="tft_with_earnings",
    version="1.0.0",
    transforms=[
        # ... existing market/macro/cross-asset transforms ...

        # Add earnings features
        EarningsSurpriseTransformSpec(ticker="AAPL"),
        EarningsGrowthTransformSpec(ticker="AAPL"),
    ],
)

# Build TFT dataset
builder = TFTDatasetBuilder(pipeline=pipeline)
dataset = builder.build(
    symbols=["AAPL"],
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# Total features: Market (45) + Macro (36) + Cross-Asset (600) + Earnings (8) = 689 features
```

---

## Testing

### Unit Tests

```bash
# Run earnings feature unit tests
poetry run pytest ml/tests/unit/features/earnings -v

# Expected: All tests pass with ≥90% coverage
```

### Integration Tests

```bash
# Run integration tests (requires PostgreSQL or uses DummyEarningsStore)
poetry run pytest ml/tests/integration/earnings -v -m integration

# Tests:
# - Full pipeline (EDGAR → PostgreSQL → Features)
# - Point-in-time correctness
# - Batch vs incremental parity
```

### Performance Tests

```bash
# Run performance benchmarks
poetry run pytest ml/tests/performance/test_earnings_performance.py -v

# Validates all SLA requirements
```

---

## Future Enhancements

### Phase 2 (Commercial Data Integration)

- **Refinitiv I/B/E/S**: Full consensus history, analyst revisions ($15k/year)
- **FactSet Estimates**: 100% coverage, real-time updates ($20k/year)
- **Bloomberg Terminal**: Intraday alerts, detailed guidance tracking ($24k/year)

### Phase 3 (Advanced Features)

- **Earnings quality score**: Accruals, cash flow quality metrics
- **Estimate dispersion**: Std dev of analyst estimates (uncertainty proxy)
- **Estimate revision momentum**: Acceleration of upgrades/downgrades
- **Cross-sectional ranking**: Percentile rank vs sector

### Phase 4 (Real-Time Alerts)

- Monitor SEC EDGAR for 8-K filings (Item 2.02)
- Parse earnings within minutes of filing
- Emit trading signals based on surprise magnitude

---

## Support

For questions or issues:

- Architecture: `ml/docs/architecture/universal_patterns_guide.md`
- Coding standards: `ml/docs/development/CODING_STANDARDS.md`
- TFT dataset: `ml/docs/TFT_DATASET_STRUCTURE.md`
- Data sources: `ml/data/earnings/README.md`
