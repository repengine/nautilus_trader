# ML Testing Strategy

## Overview

This document describes the comprehensive testing strategy for the Nautilus Trader ML system, focusing on achieving high confidence with fewer, more effective tests using property-based, metamorphic, contract, and pairwise testing approaches.

## Testing Philosophy

**"Write less tests, get more coverage"**

Instead of writing hundreds of brittle example-based tests, we focus on:

- **Invariants**: Properties that must always hold
- **Contracts**: Behavioral guarantees at boundaries
- **Relationships**: How outputs change under input transformations
- **Combinations**: Efficient coverage of parameter interactions

## Test Categories

### 1. Property-Based Tests (`ml/tests/property/`)

Property-based tests verify mathematical properties and invariants that must hold regardless of specific input values.

**Location**: `ml/tests/property/test_store_invariants.py`

**Key Invariants Tested**:

- **FeatureStore**: Timestamp monotonicity, feature immutability, partition consistency
- **ModelStore**: Prediction bounds [-1, 1], version ordering consistency
- **StrategyStore**: Signal temporal ordering, position state consistency
- **DataStore**: Watermark progression (non-decreasing), event ordering

**Example**:

```python
@given(
    ts_events=st.lists(nanosecond_timestamps(), min_size=1, max_size=100, unique=True)
)
def test_timestamp_monotonicity_invariant(self, ts_events):
    """Features stored with increasing timestamps must maintain order."""
    # Property: Retrieved timestamps should be monotonically increasing
```

**Benefits**:

- Catches edge cases automatically
- Tests hold for all valid inputs
- No need to maintain golden files

### 2. Contract/Schema Tests (`ml/tests/contracts/`)

Contract tests define and validate data contracts at component boundaries using Pandera schemas.

**Location**: `ml/tests/contracts/test_store_schemas.py`

**Schemas Defined**:

- `FeatureInputSchema`: Validates feature store inputs
- `PredictionSchema`: Validates model predictions
- `SignalSchema`: Validates strategy signals
- `WatermarkSchema`: Validates watermark progression
- `EventLogSchema`: Validates event logging

**Example**:

```python
class PredictionSchema(pa.DataFrameModel):
    prediction: Series[float] = pa.Field(ge=-1.0, le=1.0)
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0)
    ts_event: Series[np.int64] = pa.Field(ge=0)

    @pa.check("ts_init", "ts_event")
    def check_timestamp_ordering(cls, df):
        return df["ts_init"] >= df["ts_event"]
```

**Benefits**:

- Early detection of data quality issues
- Clear documentation of expected formats
- Automated validation at boundaries

### 3. Metamorphic Tests (`ml/tests/metamorphic/`)

Metamorphic tests verify relationships between outputs under controlled input transformations.

**Locations**:

- `test_feature_transforms.py`: Feature engineering transformations
- `test_signal_predictions.py`: ML predictions and signals

**Key Relations Tested**:

- **Price Scaling**: Normalized features unchanged, raw features scale proportionally
- **Time Reversal**: Magnitude features unchanged, directional features reversed
- **Noise Addition**: Bounded output changes for small input perturbations
- **Data Duplication**: Stability under repeated observations

**Example**:

```python
def test_price_scaling_invariance(self, base_price, scale_factor):
    """Scaling prices should scale price features but not normalized ones."""
    # Returns should be unchanged (normalized)
    assert features_original["returns"] ≈ features_scaled["returns"]
    # Moving averages should scale proportionally
    assert features_scaled["ma_5"] ≈ features_original["ma_5"] * scale_factor
```

**Benefits**:

- No need for exact expected values
- Tests algorithmic properties, not implementations
- Robust to code changes that preserve behavior

### 4. Pairwise/Combinatorial Tests (`ml/tests/combinatorial/`)

Pairwise tests efficiently cover parameter interactions without full cartesian products.

**Location**: `ml/tests/combinatorial/test_config_combinations.py`

**Coverage Areas**:

- Feature configuration parameters
- Model × Instrument × Timeframe combinations
- Store configuration options
- Critical three-way interactions

**Example**:

```python
# Instead of 8,748 full combinations, test 15 pairwise
parameters = {
    "return_periods": [[1], [1, 5], [1, 5, 10, 20]],
    "rsi_period": [7, 14, 21],
    "bb_period": [10, 20, 30],
    # ...
}
pairs = AllPairs(parameters.values())
# Results: 99.8% reduction in test count while covering all 2-way interactions
```

**Benefits**:

- Dramatically reduces test count (often 90%+ reduction)
- Still catches most bugs (which typically involve 2-way interactions)
- Systematic coverage of parameter space

### 5. Stateful Property Tests

Stateful tests verify complex workflows using state machines.

**Location**: `ml/tests/property/test_store_invariants.py` (StoreStateMachine class)

**Example**:

```python
class StoreStateMachine(RuleBasedStateMachine):
    @rule(feature_set=feature_sets, timestamp=timestamps)
    def store_features(self, feature_set, timestamp):
        # Store features and verify invariants hold
        assert timestamps == sorted(timestamps)
```

**Benefits**:

- Tests complex sequences of operations
- Finds bugs in state management
- Generates minimal reproduction cases

## Test Execution Guidelines

### Quick Validation (Green Tests)

```bash
# Run property tests (high value, fast)
pytest ml/tests/property -x

# Run contract tests (catch data issues)
pytest ml/tests/contracts -x

# Run metamorphic tests (algorithmic correctness)
pytest ml/tests/metamorphic -x

# Run pairwise tests (config coverage)
pytest ml/tests/combinatorial -x
```

### Full Test Suite

```bash
# All new test categories
pytest ml/tests/property ml/tests/contracts ml/tests/metamorphic ml/tests/combinatorial -x
```

### Profiles & Runners (recommended)

Use focused profiles to keep local feedback loops under a few minutes:

```bash
# dev-fast: unit + property + contracts (parallel, no integration/serial)
TEST_DB_SKIP_TRUNCATE=1 \
ML_DISABLE_METRICS_SERVER=1 \
HYPOTHESIS_PROFILE=ci \
pytest -q -m "not integration and not serial" -n auto --dist=loadscope \
  ml/tests/unit ml/tests/property ml/tests/contracts

# dev-medium: add integration (real Postgres), still parallel where safe
pytest -q -m integration ml/tests/integration  # prefer module/class scoped cleanup

# area-focused: stores only, exit on first failure
pytest -q -k "stores and not performance" --maxfail=1 ml/tests
```

Tip: prefer `uv run --active --no-sync` as a faster runner when available.

### Environment Controls & Isolation

- `TEST_DB_SKIP_TRUNCATE=1` — skip per-test TRUNCATE; rely on class/module-scoped cleanup.
- `ML_DISABLE_METRICS_SERVER=1` — do not start the Prometheus HTTP server in unit tests.
- `HYPOTHESIS_PROFILE=ci` — deterministic, bounded property tests.
- `PYTHONHASHSEED` — pin for reproducible dict ordering when needed.

### Database Strategy

- Unit/Contract: use JSON-backed registry/fakes; no DB.
- Integration: real Postgres via EngineManager or Testcontainers; mark `@pytest.mark.integration`.
- Cleanup: prefer transaction rollback; otherwise TRUNCATE only ML tables (allowlist) with a short `statement_timeout` (e.g., 2s). Keep per-test cleanup disabled by default (see `TEST_DB_SKIP_TRUNCATE`).

### Observability & Metrics

- Unit/Contract: disable server (`ML_DISABLE_METRICS_SERVER=1`); use DTO builders + no-op collectors.
- Integration: collectors allowed; keep server timeouts small to ensure fast teardown.

### CI Pipeline (suggested stages)

1. Lint + Types: ruff, `mypy ml --strict`, import-linter.
2. Unit/Contract/Property: parallel, deterministic profiles; produce coverage + JUnit.
3. Integration: spin up Postgres, run `@integration` (serial only where marked).
4. Extended: metamorphic, validators (`make validate-metrics`, `make validate-events`), duplication/security scans.

### pytest.ini (example)

```ini
[pytest]
addopts = -ra --strict-markers --tb=short --durations=10
markers =
    integration: tests requiring real services (e.g., Postgres)
    serial: tests that must not run in parallel
env =
    HYPOTHESIS_PROFILE = ci
    ML_DISABLE_METRICS_SERVER = 1
    TEST_DB_SKIP_TRUNCATE = 1
```

### Legacy Imports & Shims

Where older tests expect legacy module paths (e.g., `ml.scripts.build_tft_dataset`), provide thin shims that delegate to modern CLI modules. This reduces churn when refactoring package layout.

## New Paper-Trading Critical Tests

- Data routing (DB‑free) ensures the `DataStore` facade dispatches correctly:
  - `predictions_<model_id>` → `ModelStore.read_predictions(...)`
  - `signals_<strategy_id>` → `StrategyStore.read_signals(...)`
  - See `ml/tests/unit/stores/test_data_store_routing.py` (read routing cases).

- Upsert dedup/update (DB) validates conflict handling in batch hot paths:
  - Within‑batch duplicates are deduplicated (last row wins).
  - Cross‑batch upserts update existing rows.
  - See `ml/tests/unit/stores/test_upsert_dedup_update.py`.

- Registry events for predictions (DB‑free) verify non‑blocking emission on flush:
  - See `ml/tests/unit/stores/test_model_store_registry_emit.py`.

- Strategy performance aggregation (DB) verifies summary rollups:
  - See `ml/tests/unit/stores/test_strategy_performance_agg.py`.

- Conversion helpers validate robust ingestion inputs (dict/pandas/polars):
  - See `ml/tests/unit/stores/test_data_store_conversions.py`.

All new tests follow strict typing (mypy --strict), lint (ruff), and run in the green lane (make pytest-green). Validators are advisory and can be run with `make validate-nautilus-patterns`.

## Writing New Tests

### When to Use Each Type

**Property-Based Tests**:

- Mathematical invariants (monotonicity, bounds, conservation)
- Data structure properties (ordering, uniqueness)
- Algorithmic guarantees (convergence, stability)

**Contract Tests**:

- API boundaries
- Data pipeline interfaces
- External service integrations

**Metamorphic Tests**:

- ML models (no ground truth available)
- Feature engineering
- Signal generation

**Pairwise Tests**:

- Configuration parameters
- Multi-dimensional parameter spaces
- Feature flags and options

### Best Practices

1. **Focus on Properties, Not Examples**
   - ❌ `assert compute_return(100, 101) == 0.01`
   - ✅ `assert all returns are bounded by [-1, 1]`

2. **Test Relationships, Not Values**
   - ❌ `assert feature_value == 42.5`
   - ✅ `assert scaled_feature == original_feature * scale_factor`

3. **Use Schemas at Boundaries**
   - Define Pandera schemas for all public interfaces
   - Validate inputs and outputs systematically

4. **Minimize Test Count**
   - Use pairwise testing for configurations
   - Generate test data with Hypothesis
   - One property test can replace dozens of examples

## Mutation Testing

To measure test effectiveness, use mutation testing:

```bash
# Install mutmut
poetry add mutmut --group dev

# Run mutation testing on critical modules
mutmut run --paths-to-mutate ml/stores --tests-dir ml/tests/property

# View results
mutmut results
```

Mutation testing reveals whether tests actually catch bugs by making small code changes and verifying tests fail.

## Migration from Example-Based Tests

When refactoring existing tests:

1. **Identify the Property**: What invariant is the test really checking?
2. **Generalize the Input**: Use Hypothesis strategies instead of fixed values
3. **Assert Relationships**: Check properties, not specific outputs
4. **Remove Redundancy**: One property test can replace many examples

## Success Metrics

- **Test Count**: Reduced by 80%+ while maintaining coverage
- **Failure Detection**: Catches real bugs, not implementation details
- **Maintenance**: Tests rarely need updates when code changes
- **Execution Time**: Faster overall despite property generation
- **Confidence**: Higher confidence from testing invariants

## Summary

This testing strategy provides:

- **Higher confidence** with fewer tests
- **Better bug detection** through property exploration
- **Lower maintenance** by testing behavior not implementation
- **Clearer documentation** through contracts and schemas
- **Efficient coverage** via pairwise testing

The result: A robust, maintainable test suite that gives you the "green tests" you need while actually improving quality.
