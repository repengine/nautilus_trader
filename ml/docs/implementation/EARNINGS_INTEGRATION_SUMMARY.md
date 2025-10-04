# Earnings Integration - Executive Summary

## Problem Statement

The current `EarningsStore` implementation **violates the Universal ML Architecture** by operating outside the mandatory 4-store + 4-registry pattern. This creates architectural debt and prevents proper integration with the ML pipeline.

> **Status Update (2024.06)** — The production stack now routes all earnings access through
> the `DataStore` facade with PostgreSQL → `FileEarningsStore` → `DummyEarningsStore`
> fallback. Use this summary as a reminder of the anti-patterns to avoid and replicate
> the facade-first examples below in every guide.

### Current Issues

- ❌ **Isolated data silo** - Earnings data not accessible via standard protocols
- ❌ **No contract validation** - No schema enforcement or lineage tracking
- ❌ **No progressive fallback** - Breaks when PostgreSQL unavailable
- ❌ **Direct access pattern** - TFT builder bypasses DataStore facade
- ❌ **Actor incompatibility** - Actors cannot access earnings via `self.data_store`

---

## Solution Architecture

### Two-Layer Integration

**Layer 1: Raw Earnings Data → DataStore + DataRegistry**

```python
# Raw earnings actuals/estimates flow through DataStore
actor.data_store.write_earnings_actual(
    ticker="AAPL",
    period_end="2024-03-31",
    eps_diluted=2.52,
    ts_event=filing_ts,
)

actuals = actor.data_store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=backtest_ts,  # Point-in-time correctness
    limit=5,
)
```

**Layer 2: Computed Features → FeatureStore + FeatureRegistry**

```python
# Computed earnings features flow through FeatureStore
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    compute_earnings_surprise_batch,
)

# Declarative pipeline
pipeline = PipelineSpec(transforms=[
    EarningsSurpriseTransformSpec(ticker="AAPL"),
    EarningsGrowthTransformSpec(ticker="AAPL"),
])

# Features automatically validated and stored
feature_store.write_features(
    feature_set_id="earnings_features_v1",
    features={
        'eps_surprise_q0_AAPL': 0.07,
        'eps_growth_yoy_AAPL': 12.5,
    },
    ts_event=bar.ts_event,
)
```

---

## Key Design Decisions

### 1. DataStore = Raw Data, FeatureStore = Computed Features

**Rationale:**
- Earnings actuals/estimates are **raw inputs** (like bars, ticks)
- Earnings surprise/growth/momentum are **computed features** (like RSI, MACD)
- Maintain clear separation of concerns

### 2. EarningsStore Becomes Internal Implementation

**Before (Wrong):**
```python
# ❌ Direct access bypasses facade
earnings_store = EarningsStore(connection_string)
actuals = earnings_store.get_actuals(ticker, as_of_ts)
```

**After (Correct):**
```python
# ✓ Access via DataStore facade
actuals = data_store.get_earnings_actuals_at_or_before(ticker, ts_event)
# EarningsStore is internal implementation detail
```

### 3. Progressive Fallback Strategy

```
Production:    DataStore → EarningsStore (PostgreSQL)
Testing/CI:    DataStore → DummyEarningsStore (in-memory)
File Backend:  DataStore → FileEarningsStore (Parquet)
```

### 4. Point-in-Time Correctness at Data Layer

```python
# Filtering happens in DataStore, not in feature computation
actuals = data_store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=backtest_ts,  # Only returns filings before this time
    limit=5,
)
# Feature computation uses already-filtered data
surprise = compute_earnings_surprise_batch(actuals, estimates)
```

---

## Implementation Plan

### Phase 1: DataStore Integration (Week 1-2)

**Files to Modify:**
- `ml/stores/protocols.py` - Add earnings methods to `DataStoreFacadeProtocol`
- `ml/stores/data_store.py` - Implement earnings methods, add `_init_earnings_store()`
- `ml/core/integration.py` - Wire earnings stores in progressive fallback
- `ml/registry/data_registry.py` - Register earnings data contracts

**Estimated LOC:** ~270 lines

### Phase 2: TFT Builder Update (Week 3)

**Files to Modify:**
- `ml/data/tft_dataset_builder.py` - Replace `earnings_store` param with `data_store`
- Update `_fetch_earnings_features()` to use `data_store.get_earnings_*()`

**Estimated LOC:** ~200 lines

### Phase 3: Testing & Validation (Week 4)

**Test Coverage:**
- Unit tests: DataStore earnings methods
- Integration tests: End-to-end with real stores
- Performance benchmarks: Validate <10% overhead
- Parity tests: Hot/cold path validation

**Estimated LOC:** ~100 lines (tests)

### Total Implementation

- **6 files modified**
- **~570 lines of code**
- **4 weeks timeline**
- **Zero breaking changes** (phased migration)

---

## Benefits

### ✓ Architectural Compliance

1. **Follows Universal Pattern #1** - All data through 4-store + 4-registry
2. **Protocol-based contracts** - Type-safe, mypy-validated interfaces
3. **Progressive fallback** - Works in test environments without PostgreSQL
4. **Event-driven observability** - Automatic metrics, events, watermarks

### ✓ Developer Experience

1. **Actors get earnings automatically** - Via `self.data_store` property
2. **Consistent APIs** - Same pattern as market data, features, predictions
3. **Contract validation** - Prevents invalid data at write time
4. **Lineage tracking** - Full data provenance via DataRegistry

### ✓ Operational Excellence

1. **No data silos** - Earnings integrated with all other data types
2. **Centralized monitoring** - All reads/writes emit metrics
3. **Audit trail** - DataRegistry tracks all dataset versions
4. **Graceful degradation** - DummyStore fallback for CI/testing

---

## Migration Path

### Week 1-2: Add DataStore Integration (No Breaking Changes)

```python
# Old code continues to work
earnings_store = EarningsStore(connection_string)
actuals = earnings_store.get_actuals(ticker, as_of_ts)

# New code uses DataStore
actuals = data_store.get_earnings_actuals_at_or_before(ticker, ts_event)
```

### Week 3-4: Update TFT Builder

```python
# Old constructor (deprecated but works)
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=symbols,
    earnings_store=earnings_store,  # Deprecated
)

# New constructor (recommended)
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=symbols,
    data_store=data_store,  # Includes earnings access
)
```

### Month 6+: Deprecation Warnings

```python
# Add deprecation warnings to direct EarningsStore usage
warnings.warn(
    "Direct EarningsStore usage is deprecated. "
    "Use DataStore.get_earnings_*() methods instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

### Month 12+: Remove Direct Access (Breaking Change)

```python
# Remove public EarningsStore from API
# Keep as internal implementation detail in DataStore
```

---

## Code Examples

### Example 1: Actor Using Earnings Data

```python
from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig

class EarningsAwareActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)
        # DataStore automatically initialized with earnings access

    def on_bar(self, bar: Bar) -> None:
        """Process bar with earnings context."""
        # Get earnings actuals with point-in-time correctness
        actuals = self.data_store.get_earnings_actuals_at_or_before(
            ticker=bar.instrument_id.symbol.value,
            ts_event=bar.ts_event,
            limit=5,
        )

        if actuals:
            # Hot path: Incremental feature computation
            from ml.features.earnings import compute_earnings_surprise_incremental

            surprise = compute_earnings_surprise_incremental(
                actual=actuals[0]['eps_diluted'],
                estimate=actuals[0].get('eps_consensus', actuals[0]['eps_diluted']),
            )

            # Store features for training/inference parity
            self.feature_store.write_features(
                feature_set_id="earnings_surprise_realtime",
                instrument_id=str(bar.instrument_id),
                features=surprise,
                ts_event=bar.ts_event,
                ts_init=bar.ts_init,
            )

            # Generate signal based on earnings surprise
            if surprise['eps_surprise_pct_q0'] > 5.0:  # >5% beat
                self.log.info(f"Large earnings beat: {surprise['eps_surprise_pct_q0']:.2f}%")
                # ... signal generation logic
```

### Example 2: TFT Builder Integration

```python
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.stores.data_store import DataStore
from ml.registry import DataRegistry

# Initialize DataStore (includes earnings access)
data_registry = DataRegistry(registry_path="./ml_registry/datasets")
data_store = DataStore(
    connection_string="postgresql://localhost/nautilus",
    registry=data_registry,
)

# Create TFT builder with earnings features
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["AAPL", "MSFT", "GOOGL"],
    data_store=data_store,  # Unified data access
    include_earnings=True,
    earnings_lag_days=1,  # Publication lag for point-in-time correctness
)

# Build training dataset (earnings features automatically included)
df = builder.build_training_dataset(
    start=datetime(2023, 1, 1),
    end=datetime(2024, 1, 1),
)

# Verify earnings features present
earnings_cols = [col for col in df.columns if 'eps_' in col or 'earnings_' in col]
print(f"Earnings features: {earnings_cols}")
# Output: ['eps_surprise_q0_AAPL', 'eps_surprise_pct_q0_AAPL',
#          'eps_growth_yoy_AAPL', 'earnings_beat_streak_AAPL', ...]
```

### Example 3: Earnings Data Ingestion

```python
from ml.stores.data_store import DataStore
from datetime import datetime, timezone

# Initialize DataStore
data_store = DataStore(
    connection_string="postgresql://localhost/nautilus",
    schema="ml",
)

# Ingest earnings actual from SEC EDGAR
filing_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
ts_event = int(filing_date.timestamp() * 1e9)

data_store.write_earnings_actual(
    ticker="AAPL",
    period_end="2024-03-31",
    filing_date="2024-04-01",
    eps_diluted=2.52,
    revenue=90_753_000_000,
    ts_event=ts_event,
    ts_init=int(datetime.now(timezone.utc).timestamp() * 1e9),
)
# ✓ Contract validation (via DataRegistry)
# ✓ Event emission (observability)
# ✓ Stored in PostgreSQL (via internal EarningsStore)

# Ingest earnings estimate from Yahoo Finance
estimate_date = datetime(2024, 3, 15, tzinfo=timezone.utc)
ts_event_estimate = int(estimate_date.timestamp() * 1e9)

data_store.write_earnings_estimate(
    ticker="AAPL",
    estimate_date="2024-03-15",
    period_end="2024-03-31",
    eps_consensus=2.45,
    revenue_consensus=89_500_000_000,
    num_analysts=42,
    ts_event=ts_event_estimate,
    ts_init=int(datetime.now(timezone.utc).timestamp() * 1e9),
)

# Query with point-in-time correctness
backtest_ts = int(datetime(2024, 4, 2, tzinfo=timezone.utc).timestamp() * 1e9)
actuals = data_store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=backtest_ts,
    limit=5,
)

print(f"Actuals available as of 2024-04-02: {len(actuals)} quarters")
# Output: "Actuals available as of 2024-04-02: 1 quarters"
# (Only Q1 2024 filing is visible, Q2 hasn't filed yet)
```

---

## Testing Strategy

### Unit Tests

```python
def test_data_store_earnings_write_and_read():
    """Test earnings write/read via DataStore facade."""
    data_store = DataStore(connection_string=test_db_url)

    # Write earnings actual
    data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        revenue=90753000000,
        ts_event=1711929600000000000,
        ts_init=int(time.time() * 1e9),
    )

    # Read with point-in-time correctness
    actuals = data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1711929600000000001,  # Just after filing
        limit=5,
    )

    assert len(actuals) == 1
    assert actuals[0]['eps_diluted'] == 2.52


def test_earnings_point_in_time_correctness():
    """Test that future filings are not visible in backtest."""
    data_store = DataStore(connection_string=test_db_url)

    # Write Q1 filing (2024-04-01)
    data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        ts_event=1711929600000000000,  # 2024-04-01
        ts_init=int(time.time() * 1e9),
    )

    # Query as of 2024-03-31 (before filing)
    actuals_before = data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1711843200000000000,  # 2024-03-31
        limit=5,
    )

    # Query as of 2024-04-02 (after filing)
    actuals_after = data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1712016000000000000,  # 2024-04-02
        limit=5,
    )

    assert len(actuals_before) == 0  # Filing not visible before ts_event
    assert len(actuals_after) == 1   # Filing visible after ts_event


def test_progressive_fallback_to_dummy_store():
    """Test fallback to DummyEarningsStore when PostgreSQL unavailable."""
    # Intentionally bad connection string
    data_store = DataStore(connection_string="postgresql://invalid:5432/db")

    # Should fall back to DummyEarningsStore (in-memory)
    data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        ts_event=1711929600000000000,
        ts_init=int(time.time() * 1e9),
    )

    actuals = data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=1712016000000000000,
        limit=5,
    )

    assert len(actuals) == 1
    assert actuals[0]['eps_diluted'] == 2.52
    # Data stored in memory (not persisted)
```

### Integration Tests

```python
def test_tft_builder_earnings_integration():
    """Test TFT builder with earnings features end-to-end."""
    # Setup
    data_store = DataStore(connection_string=test_db_url)
    catalog = ParquetDataCatalog(catalog_path)

    # Ingest earnings data
    data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        ts_event=1711929600000000000,
        ts_init=int(time.time() * 1e9),
    )

    # Build dataset with earnings
    builder = TFTDatasetBuilder(
        catalog=catalog,
        symbols=["AAPL"],
        data_store=data_store,
        include_earnings=True,
    )

    df = builder.build_training_dataset(
        start=datetime(2024, 4, 1),
        end=datetime(2024, 4, 2),
    )

    # Verify earnings features present
    assert "eps_surprise_q0_AAPL" in df.columns
    assert "eps_growth_yoy_AAPL" in df.columns
    assert "earnings_beat_streak_AAPL" in df.columns
    assert "is_earnings_available" in df.columns

    # Verify feature values are correct
    assert df["eps_surprise_q0_AAPL"].notna().sum() > 0
```

### Performance Benchmarks

```python
def test_earnings_query_performance():
    """Ensure DataStore earnings queries meet SLA (<10ms P99)."""
    data_store = DataStore(connection_string=test_db_url)

    # Populate with 100 quarters of data
    for i in range(100):
        data_store.write_earnings_actual(
            ticker="AAPL",
            period_end=f"2024-{i%4+1:02d}-{(i%28)+1:02d}",
            filing_date=f"2024-{i%4+1:02d}-{(i%28)+1:02d}",
            eps_diluted=2.0 + (i * 0.01),
            ts_event=1700000000000000000 + (i * 86400000000000),
            ts_init=int(time.time() * 1e9),
        )

    # Benchmark queries
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        actuals = data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=1700000000000000000 + (50 * 86400000000000),
            limit=5,
        )
        latencies.append(time.perf_counter() - start)

    p99_latency_ms = np.percentile(latencies, 99) * 1000

    assert p99_latency_ms < 10.0  # P99 < 10ms
    assert len(actuals) == 5
```

---

## Documentation Updates

### Files to Update

1. **[ml/docs/README.md](ml/docs/README.md)**
   - Add "Earnings Data Integration" section
   - Document DataStore earnings methods

2. **[ml/docs/development/CODING_STANDARDS.md](ml/docs/development/CODING_STANDARDS.md)**
   - Update "Data Storage" section with earnings examples

3. **[ml/docs/architecture/universal_patterns_guide.md](ml/docs/architecture/universal_patterns_guide.md)**
   - Add earnings as example of 4-store + 4-registry pattern

4. **Create new:** `ml/docs/guides/earnings_integration_guide.md`
   - Step-by-step guide for using earnings data
   - Actor examples
   - TFT builder examples
   - Feature computation examples

---

## Success Criteria

### ✓ Phase 1 Complete When:

- [ ] DataStore has earnings methods in protocol
- [ ] EarningsStore integrated with progressive fallback
- [ ] All unit tests pass (100% coverage on new code)
- [ ] Performance benchmarks meet SLA (P99 < 10ms)

### ✓ Phase 2 Complete When:

- [ ] TFT builder uses DataStore for earnings
- [ ] All integration tests pass
- [ ] Feature parity validated (hot/cold paths match)
- [ ] Documentation updated

### ✓ Phase 3 Complete When:

- [ ] Deprecation warnings added to old APIs
- [ ] Migration guide published
- [ ] All existing earnings usage migrated
- [ ] Performance regressions identified and fixed

---

## Risks & Mitigation

### Risk 1: Performance Degradation

**Concern:** DataStore facade adds overhead to earnings queries

**Mitigation:**
- Benchmark early and often
- Target <10% overhead vs. direct EarningsStore
- Optimize hot paths (caching, indexing)
- Use async queries where possible

### Risk 2: Breaking Changes

**Concern:** Existing code depends on direct EarningsStore access

**Mitigation:**
- Phased migration (3 months)
- Keep old APIs working with deprecation warnings
- Provide migration guide
- Extensive testing before removal

### Risk 3: Contract Validation Overhead

**Concern:** DataRegistry validation slows down writes

**Mitigation:**
- Make validation optional (flag in config)
- Cache contracts to avoid repeated lookups
- Use async validation where possible
- Skip validation in hot paths (validate in cold paths only)

### Risk 4: Test Environment Setup

**Concern:** Tests require PostgreSQL setup

**Mitigation:**
- Progressive fallback to DummyEarningsStore
- Unit tests use in-memory stores
- Integration tests use Docker PostgreSQL
- CI/CD auto-starts test database

---

## Next Steps

1. **Week 1:** Implement DataStore earnings methods
2. **Week 2:** Add progressive fallback and tests
3. **Week 3:** Update TFT builder
4. **Week 4:** Performance benchmarks and documentation
5. **Month 2:** Migrate existing code
6. **Month 6:** Add deprecation warnings
7. **Month 12:** Remove old APIs

**Start Date:** TBD
**Estimated Completion:** 3 months
**Breaking Changes:** Month 12+ (with 6-month deprecation period)
