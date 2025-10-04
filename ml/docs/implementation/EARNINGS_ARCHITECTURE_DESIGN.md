# Earnings Data Architecture - Proper Integration Design

## Executive Summary

This document defines the **correct architectural integration** for earnings data within the Universal ML Architecture's 4-store + 4-registry pattern. The current implementation has earnings data isolated in a standalone `EarningsStore`, which violates architectural principles. This design shows how to properly integrate earnings through the mandated data flow paths.

> **2024.06 Update**: The `DataStore` facade now fronts the entire earnings stack with a
> progressive fallback chain (`EarningsStore` → `FileEarningsStore` → `DummyEarningsStore`).
> All new integration guides, playgrounds, and CLIs must reference the facade methods
> (`write_earnings_actual`, `get_earnings_actuals_at_or_before`, etc.) rather than importing
> the concrete stores directly. Monitor `ml_fallback_activations_total` to confirm fallback
> behavior during failover drills.

---

## Current State Analysis

### What Exists Today

**✓ Good: Feature Computation Layer** ([ml/features/earnings/](ml/features/earnings/))
- Hot/cold path separation (incremental + batch)
- TransformSpec classes for declarative pipelines
- Performance validated (<5ms P99)
- Protocol-compliant metrics

**✗ Problem: Data Storage Layer** ([ml/stores/earnings_store.py](ml/stores/earnings_store.py))
- **Standalone implementation** outside 4-store pattern
- **Not integrated with DataRegistry** (no lineage, versioning, contracts)
- **Not accessible to actors** via standard protocols
- **No progressive fallback** when PostgreSQL unavailable
- **TFT builder uses direct access** bypassing DataStore facade

### Architectural Violations

```
❌ Current (WRONG):
TFT Builder → EarningsStore (direct) → PostgreSQL
             ↓
          EarningsCache → Feature Computation

✓ Correct (THIS DESIGN):
Actor/TFT → DataStore → EarningsStore (internal) → PostgreSQL
         ↓         ↓
  FeatureStore  DataRegistry (manifest, contract, lineage)
         ↓
  FeatureRegistry (schema validation)
```

---

## Architectural Principles

### Universal ML Architecture Pattern #1: 4 Stores + 4 Registries

**ALL data must flow through:**

1. **DataStore** (Raw data ingestion, validation, lineage)
   - **Earnings actuals/estimates** ← NEW RESPONSIBILITY
   - Market data (bars, ticks, order book)
   - Alternative data sources

2. **FeatureStore** (Computed features, training/inference parity)
   - **Earnings features** (surprise, growth, momentum, calendar)
   - Technical indicators
   - Microstructure features
   - Macro features

3. **ModelStore** (Predictions, performance tracking)
4. **StrategyStore** (Signals, execution state)

**ALL schemas/contracts enforced by:**

1. **DataRegistry** (Dataset manifests, lineage, contracts)
   - **Earnings data contracts** ← NEW RESPONSIBILITY
2. **FeatureRegistry** (Feature schemas, validation)
   - **Earnings feature schemas** ← ALREADY SUPPORTED
3. **ModelRegistry** (Model versions, A/B testing)
4. **StrategyRegistry** (Strategy compatibility)

---

## Correct Design: Two-Layer Integration

### Layer 1: Raw Earnings Data → DataStore + DataRegistry

**Purpose:** Store and validate raw earnings actuals/estimates

**Implementation:**

#### 1.1 Add Earnings Methods to DataStoreFacadeProtocol

**File:** [ml/stores/protocols.py:221-262](ml/stores/protocols.py:221-262)

```python
@runtime_checkable
class DataStoreFacadeProtocol(Protocol):
    """Minimal facade for actor-attached data store."""

    # ... existing methods ...

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs: Any,
    ) -> None:
        """Write earnings actual with contract validation."""
        ...

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs: Any,
    ) -> None:
        """Write earnings estimate with contract validation."""
        ...

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get earnings actuals with point-in-time correctness.

        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., 'AAPL')
        ts_event : int
            Query timestamp in nanoseconds (only returns filings with ts_event < this)
        limit : int
            Maximum number of quarters to return

        Returns
        -------
        list[dict[str, Any]]
            Earnings records sorted by period_end descending
        """
        ...

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """Get earnings estimate for specific period with point-in-time correctness."""
        ...
```

#### 1.2 Implement in DataStore Class

**File:** [ml/stores/data_store.py](ml/stores/data_store.py)

```python
class DataStore(DataRegistryMixin, BusPublisherMixin, MLComponentMixin):
    """Unified data facade with contract validation and event emission."""

    def __init__(
        self,
        connection_string: str,
        registry: RegistryProtocol,
        schema: str = "ml",
        **kwargs: Any,
    ) -> None:
        # ... existing init ...

        # Initialize earnings store with progressive fallback
        self._earnings_store = self._init_earnings_store()

    def _init_earnings_store(self) -> EarningsStoreProtocol:
        """Initialize earnings store with progressive fallback."""
        try:
            from ml.stores.earnings_store import EarningsStore

            store = EarningsStore(
                connection_string=self._connection_string,
                schema=self._schema,
            )
            logger.info("EarningsStore initialized (PostgreSQL)")
            return store
        except Exception as exc:
            logger.warning(
                f"EarningsStore init failed, falling back to DummyEarningsStore: {exc}",
            )
            from ml.stores.earnings_store import DummyEarningsStore

            return DummyEarningsStore()

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs: Any,
    ) -> None:
        """Write earnings actual with contract validation and event emission."""
        # 1. Validate against DataRegistry contract (if exists)
        if self._registry:
            try:
                contract = self._registry.get_contract("earnings_actuals")
                if contract:
                    # Validate required fields, types, ranges
                    self._validate_earnings_actual(
                        ticker=ticker,
                        period_end=period_end,
                        eps_diluted=eps_diluted,
                        revenue=revenue,
                        contract=contract,
                    )
            except Exception as exc:
                logger.warning(f"Earnings contract validation failed: {exc}")

        # 2. Write to underlying EarningsStore
        self._earnings_store.write_actuals(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=revenue,
            ts_event=ts_event,
            ts_init=ts_init,
            **kwargs,
        )

        # 3. Emit dataset event for observability
        if self._enable_publishing:
            emit_dataset_event_and_watermark(
                dataset_id="earnings_actuals",
                instrument_id=ticker,
                stage=Stage.INGESTION,
                source=Source.EDGAR,
                status=EventStatus.SUCCESS,
                ts_min=ts_event,
                ts_max=ts_event,
                record_count=1,
                registry=self._registry,
                publisher=self._publisher,
            )

        logger.debug(
            "Wrote earnings actual",
            ticker=ticker,
            period_end=period_end,
            eps_diluted=eps_diluted,
        )

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get earnings actuals with point-in-time correctness."""
        # Convert ts_event (nanoseconds) to ISO date for store query
        from datetime import datetime, timezone

        as_of_date = datetime.fromtimestamp(ts_event / 1e9, tz=timezone.utc)
        end_date = as_of_date.strftime("%Y-%m-%d")

        actuals = self._earnings_store.get_actuals(
            ticker=ticker,
            as_of_ts=ts_event,
            end_date=end_date,
        )

        # Emit read event for observability
        if self._enable_publishing:
            self._increment_read_metric("earnings_actuals", count=len(actuals))

        return actuals[:limit] if actuals else []

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """Get earnings estimate for specific period."""
        estimate = self._earnings_store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=ts_event,
        )

        if self._enable_publishing:
            self._increment_read_metric("earnings_estimates", count=1 if estimate else 0)

        return estimate
```

#### 1.3 Register Earnings Contracts in DataRegistry

**File:** [ml/registry/data_registry.py](ml/registry/data_registry.py) (extend existing)

```python
# Define earnings data contract
EARNINGS_ACTUALS_CONTRACT = DataContract(
    dataset_id="earnings_actuals",
    schema_version="1.0.0",
    required_fields=["ticker", "period_end", "filing_date", "ts_event", "ts_init"],
    optional_fields=["eps_basic", "eps_diluted", "revenue", "net_income"],
    field_types={
        "ticker": "string",
        "period_end": "date",
        "filing_date": "date",
        "ts_event": "int64",
        "ts_init": "int64",
        "eps_diluted": "float64",
        "revenue": "float64",
    },
    validation_rules=[
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field="ts_event",
            params={"min": 0, "max": 9999999999999999999},
        ),
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="ticker",
        ),
    ],
)

# Register on DataRegistry initialization
data_registry.register_contract(EARNINGS_ACTUALS_CONTRACT)
```

---

### Layer 2: Computed Earnings Features → FeatureStore + FeatureRegistry

**Purpose:** Store computed earnings features (surprise, growth, momentum, calendar)

**Implementation:**

#### 2.1 Earnings Features Already Flow Through FeatureStore ✓

The existing `FeatureStore` already handles earnings features via the pipeline system:

```python
# ALREADY WORKS (no changes needed):
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
)
from ml.features.pipeline import PipelineSpec, PipelineRunner

# Define pipeline
pipeline_spec = PipelineSpec(transforms=[
    EarningsSurpriseTransformSpec(ticker="AAPL"),
    EarningsGrowthTransformSpec(ticker="AAPL"),
    EarningsMomentumTransformSpec(ticker="AAPL"),
])

# Run pipeline (cold path)
runner = PipelineRunner(pipeline_spec, data_store=data_store, feature_store=feature_store)
features_df = runner.run_batch(bars_df)

# Features automatically stored in FeatureStore
feature_store.write_features(
    feature_set_id="earnings_features_v1",
    instrument_id="AAPL.NASDAQ",
    features=features_df.to_dict(orient="records")[0],
    ts_event=ts_event,
    ts_init=ts_init,
)
```

#### 2.2 Register Earnings Feature Schemas in FeatureRegistry

**File:** [ml/registry/feature_registry.py](ml/registry/feature_registry.py) (extend existing)

```python
# Register earnings feature manifest
from ml.registry.dataclasses import FeatureManifest, FeatureSchema

earnings_feature_manifest = FeatureManifest(
    feature_set_id="earnings_features_v1",
    version="1.0.0",
    features=[
        FeatureSchema(
            name="eps_surprise_q0_AAPL",
            dtype="float64",
            description="EPS surprise (actual - estimate) for most recent quarter",
            source_transform="EarningsSurpriseTransformSpec",
        ),
        FeatureSchema(
            name="eps_surprise_pct_q0_AAPL",
            dtype="float64",
            description="EPS surprise as percentage of estimate",
            source_transform="EarningsSurpriseTransformSpec",
        ),
        FeatureSchema(
            name="eps_growth_yoy_AAPL",
            dtype="float64",
            description="Year-over-year EPS growth percentage",
            source_transform="EarningsGrowthTransformSpec",
        ),
        FeatureSchema(
            name="earnings_beat_streak_AAPL",
            dtype="int32",
            description="Consecutive quarters beating consensus",
            source_transform="EarningsMomentumTransformSpec",
        ),
        # ... additional features
    ],
    data_requirements="earnings_actuals,earnings_estimates",
    schema_hash=compute_schema_hash(feature_names),
)

feature_registry.register_manifest(earnings_feature_manifest)
```

---

## Integration with TFT Dataset Builder

### Before (WRONG - Direct EarningsStore):

```python
# ❌ BAD: Bypasses DataStore facade
from ml.stores.earnings_store import EarningsStore
from ml.data.earnings.earnings_cache import EarningsCache

builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["AAPL"],
    earnings_store=EarningsStore(...),  # Direct access
    earnings_cache=EarningsCache(...),   # Separate caching layer
    include_earnings=True,
)
```

### After (CORRECT - Via DataStore):

```python
# ✓ GOOD: Uses DataStore facade
from ml.stores.data_store import DataStore
from ml.registry import DataRegistry

# Initialize unified data access (includes earnings)
data_registry = DataRegistry(...)
data_store = DataStore(
    connection_string="postgresql://...",
    registry=data_registry,
)

builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["AAPL"],
    data_store=data_store,  # Unified facade (includes earnings)
    include_earnings=True,
)
```

### TFT Builder Implementation Changes

**File:** [ml/data/tft_dataset_builder.py](ml/data/tft_dataset_builder.py)

```python
from ml.stores.protocols import DataStoreFacadeProtocol

class TFTDatasetBuilder:
    def __init__(
        self,
        catalog: ParquetDataCatalog,
        symbols: list[str],
        # ... existing params ...
        data_store: DataStoreFacadeProtocol | None = None,  # ← Unified access
        include_earnings: bool = False,
        earnings_lag_days: int = 1,
    ) -> None:
        # ... existing init ...
        self.data_store = data_store
        self.include_earnings = include_earnings and data_store is not None
        self.earnings_lag_days = earnings_lag_days

    def _fetch_earnings_features(
        self,
        ticker: str,
        timestamps: pl.Series,
        as_of_date: datetime | None = None,
    ) -> pl.DataFrame | None:
        """Fetch earnings features via DataStore facade."""
        if not self.include_earnings or not self.data_store:
            return None

        # Query via DataStore (not direct EarningsStore)
        max_ts_ns = int(timestamps.max().timestamp() * 1e9)

        # Fetch actuals with point-in-time correctness
        actuals = self.data_store.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=max_ts_ns,
            limit=5,  # Last 5 quarters for YoY growth
        )

        if not actuals:
            logger.debug(f"No earnings actuals for {ticker}")
            return None

        # Build EPS series for feature computation
        eps_series = np.array([a['eps_diluted'] or 0.0 for a in actuals], dtype=np.float64)

        # Fetch estimates for surprise calculation
        estimates = []
        for actual in actuals:
            estimate = self.data_store.get_earnings_estimate_at_or_before(
                ticker=ticker,
                period_end=actual['period_end'],
                ts_event=actual['ts_event'],
            )
            estimates.append(estimate['eps_consensus'] if estimate else actual['eps_diluted'])

        estimates_array = np.array(estimates, dtype=np.float64)

        # Compute features using earnings feature functions
        from ml.features.earnings import (
            compute_earnings_surprise_batch,
            compute_earnings_growth_batch,
            compute_earnings_momentum_batch,
        )

        surprise_features = compute_earnings_surprise_batch(eps_series, estimates_array)
        growth_features = compute_earnings_growth_batch(eps_series)
        momentum_features = compute_earnings_momentum_batch(
            surprise_features['eps_surprise_q0'],
            eps_series,
        )

        # Build quarterly feature DataFrame
        quarterly_data = {
            'period_end': [a['period_end'] for a in actuals],
            'filing_date': [a['filing_date'] for a in actuals],
            f'eps_surprise_q0_{ticker}': surprise_features['eps_surprise_q0'],
            f'eps_surprise_pct_q0_{ticker}': surprise_features['eps_surprise_pct_q0'],
            f'eps_growth_yoy_{ticker}': growth_features['eps_growth_yoy'],
            f'eps_growth_qoq_{ticker}': growth_features['eps_growth_qoq'],
            f'earnings_beat_streak_{ticker}': momentum_features['earnings_beat_streak'],
            f'eps_volatility_4q_{ticker}': momentum_features['eps_volatility_4q'],
        }

        quarterly_df = pl.DataFrame(quarterly_data)
        quarterly_df = quarterly_df.with_columns([
            pl.col('filing_date').str.strptime(pl.Date, "%Y-%m-%d"),
        ])

        # Add publication lag for point-in-time correctness
        lag_duration = pl.duration(days=self.earnings_lag_days)
        quarterly_df = quarterly_df.with_columns([
            (pl.col('filing_date').cast(pl.Datetime('ns')) + lag_duration).alias('effective_date')
        ])

        # Expand to bar-level via ASOF join
        bar_df = pl.DataFrame({'timestamp': timestamps})

        result = bar_df.join_asof(
            quarterly_df,
            left_on='timestamp',
            right_on='effective_date',
            strategy='backward',  # Use last available earnings
        )

        # Select feature columns
        feature_cols = [col for col in result.columns if col.startswith((
            'eps_surprise', 'eps_growth', 'earnings_beat', 'eps_volatility'
        ))]

        result = result.select(['timestamp'] + feature_cols)

        logger.debug(
            f"Fetched earnings features for {ticker}: {len(result)} rows, "
            f"{len(feature_cols)} features from {len(actuals)} quarters"
        )

        return result
```

---

## Actor Integration

### Actors Automatically Get Earnings Access

All actors extending `BaseMLInferenceActor` automatically have earnings data access via `self.data_store`:

```python
from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig

class MyEarningsAwareActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)

        # DataStore automatically initialized by base class
        # Includes earnings access methods

    def on_bar(self, bar: Bar) -> None:
        """Process bar with earnings context."""
        # Get earnings actuals with point-in-time correctness
        actuals = self.data_store.get_earnings_actuals_at_or_before(
            ticker=bar.instrument_id.symbol.value,
            ts_event=bar.ts_event,
            limit=5,
        )

        if actuals:
            latest_eps = actuals[0]['eps_diluted']
            logger.info(f"Latest EPS for {bar.instrument_id}: {latest_eps}")

            # Compute features incrementally (hot path)
            from ml.features.earnings import compute_earnings_surprise_incremental

            surprise = compute_earnings_surprise_incremental(
                actual=actuals[0]['eps_diluted'],
                estimate=actuals[0].get('eps_consensus', actuals[0]['eps_diluted']),
            )

            # Store features via FeatureStore
            self.feature_store.write_features(
                feature_set_id="earnings_surprise_realtime",
                instrument_id=str(bar.instrument_id),
                features=surprise,
                ts_event=bar.ts_event,
                ts_init=bar.ts_init,
            )
```

---

## Progressive Fallback Strategy

### Fallback Chain

```
PostgreSQL Available:
  DataStore → EarningsStore (PostgreSQL) → earnings_actuals/earnings_estimates tables

PostgreSQL Unavailable (Test/CI):
  DataStore → DummyEarningsStore (in-memory dict) → Warning logged

File Fallback Mode (ML_FILE_STORE_PATH set):
  DataStore → FileEarningsStore (Parquet files) → {base_path}/earnings/*.parquet
```

### Implementation

**File:** [ml/core/integration.py](ml/core/integration.py) (extend `_init_stores()`)

```python
def _init_stores(self) -> None:
    """Initialize all store components with progressive fallback."""
    if self._file_fallback:
        file_root = self._file_store_path
        self.feature_store = FileFeatureStore(base_path=file_root / "features")
        self.model_store = FileModelStore(base_path=file_root / "models")
        self.strategy_store = FileStrategyStore(base_path=file_root / "strategies")

        # NEW: Add earnings to file fallback
        from ml.stores.file_backed import FileEarningsStore
        earnings_store = FileEarningsStore(base_path=file_root / "earnings")

        self.data_store = FileDataStore(
            base_path=file_root / "datastore",
            earnings_store=earnings_store,  # Inject earnings
        )

    elif self._json_fallback:
        from ml.stores.base import DummyStore
        from ml.stores.earnings_store import DummyEarningsStore

        self.feature_store = DummyStore()
        self.model_store = DummyStore()
        self.strategy_store = DummyStore()

        # NEW: Add dummy earnings store
        self.data_store = DummyStore(earnings_store=DummyEarningsStore())

    else:
        # PostgreSQL path (production)
        # ... existing store initialization ...

        # DataStore initialization includes earnings (ALREADY SHOWN ABOVE)
        self.data_store = create_data_store(
            registry=self.data_registry,
            connection_string=self.db_connection,
            # EarningsStore automatically initialized inside DataStore._init_earnings_store()
        )
```

---

## Migration Path

### Phase 1: Add DataStore Integration (No Breaking Changes)

**Week 1:**
1. ✓ Add earnings methods to `DataStoreFacadeProtocol`
2. ✓ Implement in `DataStore` class (delegates to internal `_earnings_store`)
3. ✓ Keep standalone `EarningsStore` for backward compatibility
4. ✓ Add unit tests for DataStore earnings methods

**Week 2:**
5. ✓ Register earnings data contracts in DataRegistry
6. ✓ Add progressive fallback (`DummyEarningsStore`, `FileEarningsStore`)
7. ✓ Update integration manager to wire earnings stores

### Phase 2: Update TFT Builder

**Week 3:**
1. ✓ Change TFT builder constructor to accept `data_store: DataStoreFacadeProtocol`
2. ✓ Update `_fetch_earnings_features()` to use `self.data_store.get_earnings_*()`
3. ✓ Remove direct `EarningsStore` imports
4. ✓ Update tests to use DataStore facade

**Week 4:**
5. ✓ Add integration tests with real DataStore + EarningsStore
6. ✓ Performance benchmarks to ensure <10% overhead
7. ✓ Update documentation and examples

### Phase 3: Deprecation (6+ months later)

**Month 6:**
1. Mark standalone `EarningsStore` usage as deprecated
2. Add migration guide for users
3. Log warnings when EarningsStore used directly

**Month 12:**
4. Remove direct access patterns from public API
5. Keep EarningsStore as internal implementation detail

---

## Benefits Summary

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

## Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| [ml/stores/protocols.py](ml/stores/protocols.py) | Add earnings methods to `DataStoreFacadeProtocol` | +40 |
| [ml/stores/data_store.py](ml/stores/data_store.py) | Implement earnings methods, add `_init_earnings_store()` | +150 |
| [ml/core/integration.py](ml/core/integration.py) | Wire earnings stores in progressive fallback | +30 |
| [ml/registry/data_registry.py](ml/registry/data_registry.py) | Register earnings data contracts | +50 |
| [ml/data/tft_dataset_builder.py](ml/data/tft_dataset_builder.py) | Replace `earnings_store` param with `data_store` | +200 |
| [ml/stores/file_backed.py](ml/stores/file_backed.py) | Add `FileEarningsStore` for file fallback | +100 |

**Total Implementation:** ~570 lines of code, 6 files modified

---

## Testing Strategy

### Unit Tests

1. **DataStore earnings methods**
   - Write/read actuals and estimates
   - Point-in-time correctness validation
   - Contract validation (valid/invalid data)
   - Fallback behavior (PostgreSQL → Dummy)

2. **TFT builder integration**
   - Earnings features via DataStore
   - ASOF join correctness
   - Null handling and availability masks

### Integration Tests

1. **End-to-end with real stores**
   - Write earnings via DataStore
   - Read in TFT builder
   - Compute features
   - Store in FeatureStore
   - Validate in FeatureRegistry

2. **Actor integration**
   - Actor gets earnings via `self.data_store`
   - Hot path incremental features
   - Cold path batch features
   - Training/inference parity

### Performance Benchmarks

1. **DataStore overhead**
   - Baseline: Direct EarningsStore access
   - With DataStore: <5% overhead target
   - With contract validation: <10% overhead target

2. **TFT builder overhead**
   - Baseline: No earnings features
   - With earnings: <15% overhead target

### Operations Checklist

- Export `ML_FILE_STORE_PATH` for the file fallback and ensure the directory is writable on each
  host running dataset tasks or actors.
- Seed the registry with earnings manifests/contracts via
  `uv run --active --no-sync python -m ml.registry.bootstrap_datasets` (supply
  `--backend postgres` plus `NAUTILUS_REGISTRY_DB_URL` in shared environments).
- Monitor `ml_fallback_activations_total{component="data_store",level}` alongside
  `ml_earnings_writes_total`/`ml_earnings_reads_total` during drills to confirm fallback and load
  patterns.
- Run `make validate-metrics` and `make validate-events` before sign-off on earnings changes to
  confirm observability contracts.

---

## Conclusion

This design **properly integrates earnings data** into the Universal ML Architecture by:

1. **Routing raw data through DataStore + DataRegistry** (Layer 1)
   - Contract validation, event emission, lineage tracking
   - Progressive fallback for testing/CI

2. **Routing computed features through FeatureStore + FeatureRegistry** (Layer 2)
   - Schema validation, training/inference parity
   - Hot/cold path separation

3. **Providing actor access via standard protocols** (Integration)
   - All actors get `self.data_store` with earnings methods
   - No special initialization required

4. **Maintaining backward compatibility** (Migration)
   - Phased rollout over 3 months
   - Deprecation period before breaking changes

The result is a **unified, observable, type-safe data flow** that treats earnings as a first-class citizen in the ML architecture.
