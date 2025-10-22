# Context: ML Schema, Evaluation & Registry

**Last Updated:** 2025-10-19
**Directories:** `ml/schema/`, `ml/evaluation/`, `ml/registry/`, `ml/ml_registry/`

## Executive Summary

- **ml/schema/**: 3 SQL DDL files (494 LOC total). **PARTIALLY IMPLEMENTED** - earnings and instruments have stores, cross-asset features do not.
- **ml/evaluation/**: 2 Python modules (147 LOC total). NumPy-only binary classification metrics. **FUNCTIONAL** and used in training.
- **ml/registry/**: 29 Python files (16,582 LOC). **PRODUCTION** - mandatory 4-registry architecture with decomposed components.
- **ml/ml_registry/**: Empty directory with only `datasets/` subdirectory (also empty). **ABANDONED/UNUSED**.

**Status**: Schema is partially materialized via SQLAlchemy stores (not SQL files). Evaluation is minimal but sufficient. Registry is mature and stable.

---

## ml/schema/ - SQL Blueprints with Selective Implementation

### File Inventory

| File | LOC | Purpose | Implementation Status |
|------|-----|---------|----------------------|
| `earnings.sql` | 110 | Earnings actuals/estimates/calendar tables | ✅ **IMPLEMENTED** via `EarningsStore` |
| `instruments.sql` | 173 | Temporal instrument metadata with factor mappings | ✅ **IMPLEMENTED** via `InstrumentMetadataStore` |
| `cross_asset_features.sql` | 211 | Beta/spread/correlation feature storage | ❌ **NOT IMPLEMENTED** (features exist, storage does not) |

### Key Design Patterns

All schema files follow Nautilus conventions:
- **Timestamps**: `ts_event` (nanoseconds), `ts_init` (nanoseconds)
- **Joinability**: `instrument_id` field for joining with market data
- **Partitioning**: Monthly range partitions for time-series data (managed by `PartitionManager`)
- **Indexes**: BRIN indexes on time columns for efficient scans

### Implementation Reality: SQLAlchemy Tables, Not SQL Files

**CRITICAL FINDING**: The SQL files in `ml/schema/` are **design blueprints only**. Actual table creation happens via SQLAlchemy in store constructors:

#### 1. Earnings Tables (✅ Implemented)

**Store**: `ml/stores/earnings_store.py` (EarningsStore class)

**Tables Created**:
- `ml.earnings_actuals` - SEC EDGAR filings (10-Q/10-K)
- `ml.earnings_estimates` - Yahoo Finance consensus estimates

**Creation Method**: SQLAlchemy `Table()` + `metadata.create_all()` in `__init__`

**Usage**:
```python
from ml.stores.earnings_store import EarningsStore
store = EarningsStore("postgresql://...")
store.write_actuals(ticker="AAPL", period_end="2024-09-30", ...)
actuals = store.get_actuals(ticker="AAPL", start_date="2024-01-01")
```

**Key Features**:
- Point-in-time queries via `as_of_ts` parameter (prevents look-ahead bias)
- Upsert logic (INSERT ... ON CONFLICT DO UPDATE)
- Prometheus metrics: `ml_earnings_writes_total`, `ml_earnings_reads_total`

**Tests**:
- `ml/tests/unit/stores/test_earnings_store.py`
- `ml/tests/integration/earnings/test_earnings_store_db.py`
- `ml/tests/integration/earnings/test_earnings_end_to_end.py`

#### 2. Instrument Metadata Table (✅ Implemented)

**Store**: `ml/stores/instrument_metadata_store.py` (InstrumentMetadataStore class)

**Table Created**: `ml.instrument_metadata`

**Creation Method**: SQLAlchemy `Table()` with dynamic table name in `_define_table()`

**Columns**:
- `instrument_id`, `ts_event`, `ts_init` (Nautilus standard)
- `duration_bucket` (0=Short, 1=Medium, 2=Long)
- `issuer_type` (0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY)
- `liquidity_tier` (1=High, 2=Medium, 3=Low)
- `region`, `sector`, `rating` (optional metadata)
- `valid_from_ns`, `valid_until_ns` (temporal versioning)

**Usage**:
```python
from ml.stores.instrument_metadata_store import InstrumentMetadataStore
store = InstrumentMetadataStore("postgresql://...")
store.write_metadata(
    instrument_id="US10Y.BOND",
    duration_bucket=2,  # Long duration
    issuer_type=0,      # Sovereign
    liquidity_tier=1,   # High liquidity
    ts_event=time.time_ns(),
    ts_init=time.time_ns(),
)
metadata = store.get_metadata("US10Y.BOND")
```

**Tests**:
- `ml/tests/unit/stores/test_instrument_metadata_store.py`
- `ml/tests/unit/stores/test_instrument_metadata_basic.py`
- `ml/tests/validation_reports/instrument_metadata_db_validation.py`

#### 3. Cross-Asset Feature Tables (❌ Not Implemented)

**Schema Designed**: `ml/schema/cross_asset_features.sql`

**Tables Specified**:
- `ml_cross_asset_betas` (EWMA beta, covariance, market variance)
- `ml_cross_asset_spreads` (z-scored spreads for pairs trading)
- `ml_cross_asset_correlations` (rolling correlation coefficients)

**Feature Computation EXISTS**: `ml/features/cross_asset/` (1,288 LOC)
- `beta.py` (241 LOC) - EWMA beta computation (hot + cold path)
- `spreads.py` (263 LOC) - Z-scored spread computation
- `correlation.py` (274 LOC) - Rolling correlation
- `state.py` (385 LOC) - State management for incremental updates

**Storage Layer MISSING**: No `CrossAssetFeatureStore` exists

**Gap**: Features are computed in-memory but **not persisted to database**. Cross-asset features likely stored via generic `FeatureStore` as JSONB or not stored at all.

**Recommendation**: Create `CrossAssetFeatureStore` if persistence is needed, otherwise document that cross-asset features are ephemeral (computed on-demand).

### Why SQL Files Exist If Not Used

The SQL files serve as **reference implementations** showing:
1. Desired schema structure
2. Index strategy (BRIN for time-series)
3. Partitioning approach
4. Comments documenting design intent

Stores implement the same schema programmatically for:
- Type safety (SQLAlchemy validation)
- Dynamic table creation (no migration scripts needed)
- Easier testing (tables created in-memory for tests)

### Migration Strategy

No traditional migration scripts exist. Tables are created **lazily on first use** via:
```python
self._metadata.create_all(self._engine, tables=[...])
```

This approach:
- ✅ Simplifies deployment (no migration runner)
- ✅ Enables in-memory testing
- ❌ Harder to track schema changes over time
- ❌ No rollback mechanism

---

## ml/evaluation/ - Lightweight Metrics Layer

### File Inventory

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 75 | Public API surface (3 functions exported) |
| `metrics.py` | 72 | Binary classification metrics (NumPy-only) |

**Total**: 147 LOC (minimal, focused implementation)

### Public API

```python
from ml.evaluation import roc_auc, pr_auc, binary_logloss

# All functions: y_true: NDArray, scores: NDArray -> float
roc_auc(y_true, scores)           # Mann-Whitney U statistic
pr_auc(y_true, scores, n_thresh)  # Trapezoidal rule, default 200 thresholds
binary_logloss(y_true, p, eps)    # Log loss with eps=1e-12 clipping
```

### Implementation Details

**ROC AUC** (`roc_auc`):
- Algorithm: Mann-Whitney U statistic (equivalent to AUC)
- Complexity: O(n log n) for sorting
- Handles: Degenerate cases (all positives/negatives → 0.0)

**PR AUC** (`pr_auc`):
- Algorithm: Trapezoidal rule over precision-recall curve
- Thresholds: Linearly spaced from 0.0 to 1.0
- Compatibility: Handles NumPy < 2.0 (`np.trapz`) and >= 2.0 (`np.trapezoid`)

**Binary Log Loss** (`binary_logloss`):
- Clipping: `eps` (default 1e-12) to `1.0 - eps` to prevent log(0)
- Stable: Uses `np.log` for numerical stability

### Design Philosophy

**Cold Path Only**:
- No integration with hot-path inference
- No real-time evaluation metrics
- Used in training/validation workflows only

**NumPy-Only Dependencies**:
- Zero external ML libraries (scikit-learn, PyTorch, etc.)
- Compatible with restricted environments
- Lightweight for CI/CD pipelines

**No Prometheus Integration**:
- Evaluation metrics are **computed values**, not monitoring metrics
- Not exposed via `ml.common.metrics_bootstrap`
- Results logged/returned, not scraped

### Usage Found (Grep Evidence)

1. **Training Pipelines**:
   - `ml/training/teacher/tft_cli.py`: Conditional imports for model evaluation
   - `ml/training/event_driven/worker.py`: ROC-AUC for validation

2. **Tests**:
   - `ml/tests/unit/evaluation/test_metrics.py`: Basic unit tests
   - `ml/tests/metamorphic/test_metrics_metamorphic.py`: Metamorphic tests (precision/recall monotonicity)

### Known Gaps

- **No regression metrics**: MSE, MAE, R² not provided
- **No calibration metrics**: ECE (Expected Calibration Error), MCE not provided
- **No feature importance**: Permutation importance, SHAP not in scope
- **No drift detection**: Kolmogorov-Smirnov, PSI not here (handled in `ml/monitoring/`)

### Why It's Minimal

Evaluation is **not a priority** for this ML platform because:
1. Focus is on **trading performance** (Sharpe, drawdown), not model metrics
2. Model selection uses **backtesting**, not cross-validation
3. External tools (MLflow, Weights & Biases) handle rich evaluation

This module provides **just enough** for internal validation without heavy dependencies.

---

## ml/registry/ - Core Production Architecture

### File Inventory (29 files, 16,582 LOC)

#### Tier 1: 4 Mandatory Registries (Pattern 1)

| File | LOC | Purpose |
|------|-----|---------|
| `feature_registry.py` | 472 | Feature set lifecycle + schema hashing |
| `model_registry.py` | 1,186 | Model deployment facade (strangler fig pattern) |
| `strategy_registry.py` | 767 | Strategy compatibility validation |
| `data_registry.py` | 872 | Dataset manifest + lineage (new) |
| `data_registry_legacy.py` | 2,539 | Dataset manifest (old, under flag) |

**Feature Flag**: `ML_USE_LEGACY_DATA_REGISTRY` (0=new, 1=legacy)

#### Tier 2: Model Registry Decomposition (Phase 2.3)

**Strangler Fig Pattern**: `ModelRegistry` is a facade delegating to specialized components.

| Component | LOC | Responsibility |
|-----------|-----|----------------|
| `model_persistence.py` | 1,177 | Save/load, artifact I/O, SHA-256 validation |
| `model_quality_validator.py` | 157 | Quality gates, validation rules |
| `model_deployment_mgr.py` | 635 | Deployment tracking, version management, hot reload |
| `ab_testing_manager.py` | 417 | A/B test config, statistical analysis |
| `canary_deployment_mgr.py` | 474 | Gradual rollout, promotion |

**Total Model Subsystem**: 3,860 LOC (facade + components)

**Backward Compatibility**:
```python
# Old code still works (facade maintains 100% API compatibility)
registry.save_model(model, path)
registry.get_quality_gate(model_id)
registry.deploy_canary(model_id)

# Actually delegates to:
ModelPersistence.save(...)
ModelQualityValidator.evaluate(...)
CanaryDeploymentManager.deploy(...)
```

**Feature Flag**: `ML_USE_LEGACY_MODEL_REGISTRY` (0=new facade, 1=monolithic legacy)

#### Tier 3: Data Registry Support Components

| Component | LOC | Responsibility |
|-----------|-----|----------------|
| `manifest_manager.py` | 790 | Manifest lifecycle (create, update, validate) |
| `lineage_manager.py` | 473 | Dataset lineage tracking (DAG relationships) |
| `event_manager.py` | 386 | Event-driven registry updates |
| `watermark_manager.py` | 683 | Watermark progression + lag tracking |
| `contract_manager.py` | 354 | Data contracts + schema validation |

**Total Data Subsystem**: 2,686 LOC (data registry + support)

#### Tier 4: Infrastructure & Utilities

| File | LOC | Purpose |
|------|-----|---------|
| `base.py` | 460 | Enums: DeploymentStatus, ModelRole, DataRequirements |
| `abstract_registry.py` | 132 | ABC for all registry implementations |
| `protocols.py` | 81 | Protocol interfaces for structural typing |
| `dataclasses.py` | 980 | Shared DTOs: QualityGate, ValidationResult, CanaryConfig, DatasetManifest |
| `persistence.py` | 425 | PersistenceManager, BackendType (JSON/Postgres) |
| `bootstrap_datasets.py` | 823 | Bootstrap utilities for test datasets |
| `statistics.py` | 237 | Welch t-test, sample size calculation |
| `artifacts.py` | 134 | Artifact metadata handling |
| `utils.py` | 179 | Feature schema builders, compatibility checks |
| `summaries.py` | 104 | Model summary generation |
| `_typing_utils.py` | 164 | Type casting helpers |
| `mixins.py` | 84 | Mixin base classes |
| `__init__.py` | 188 | Public API (47 exports) |

### Architecture: Universal ML Pattern 1 (Mandatory 4 Registries)

**Every ML actor** must integrate with 4 registries via `BaseMLInferenceActor`:

```python
# Automatic initialization in BaseMLInferenceActor.__init__()
self.feature_registry: FeatureRegistry
self.model_registry: ModelRegistry
self.strategy_registry: StrategyRegistry
self.data_registry: DataRegistry
```

**Progressive Fallback**:
- PostgreSQL unavailable → Dummy registries (in-memory, warnings logged)
- Registry loading fails → Direct file loading (with model path fallback)

### Critical Integration Points

#### 1. Feature Registry → Stores

```python
# Schema validation
compute_schema_hash(feature_names) -> str  # Hash for versioning
FeatureManifest.feature_names -> list[str]  # Ordered list for alignment

# Usage in FeatureStore
manifest = feature_registry.get_manifest(feature_set_id)
assert compute_schema_hash(df.columns) == manifest.schema_hash
```

#### 2. Model Registry → Training

```python
# Model lifecycle
ModelPersistence.save(model, path) -> str         # Save + SHA-256 hash
ModelPersistence.load(model_id) -> Any            # Load with validation
ModelQualityValidator.evaluate(metrics) -> bool   # Quality gates
ModelDeploymentManager.deploy(model_id) -> None   # Version tracking
```

**ONNX Export**: `ModelPersistence` exports to ONNX for hot-path inference.

#### 3. Data Registry → Orchestration

```python
# Watermark tracking (used by PipelineOrchestrator)
watermark_mgr.get_watermark(dataset_id) -> Watermark
watermark_mgr.update_watermark(dataset_id, ts_event, lag_ns)

# Data contracts (schema validation at ingestion boundaries)
contract_mgr.validate_contract(dataset_id, df) -> ValidationResult

# Lineage (dataset → feature → model relationships)
lineage_mgr.track_lineage(parent_id, child_id, relationship_type)
```

#### 4. Strategy Registry → Orchestration

```python
# Strategy manifests declare requirements
strategy_registry.register_strategy(
    strategy_id="ml_signal_v1",
    model_requirements=["model_1", "model_2"],
    input_schema=["feature_1", "feature_2"],
    market_regime="high_volatility",
)

# Enforced at actor initialization
strategy_registry.validate_compatibility(strategy_id, market_regime)
```

### Dual-Path Pattern: Legacy vs. Decomposed

**Runtime Selection** (`ml/registry/__init__.py` line ~117):

```python
USE_LEGACY_DATA_REGISTRY = os.getenv("ML_USE_LEGACY_DATA_REGISTRY", "0") == "1"

if USE_LEGACY_DATA_REGISTRY:
    DataRegistry = DataRegistryLegacy  # 2,539 LOC, monolithic
else:
    DataRegistry = DataRegistry        # 872 LOC, decomposed
```

**Why Dual-Path?**
- **Legacy (monolithic)**: Simpler for offline analysis, backward-compatible
- **New (decomposed)**: Better separation of concerns, easier testing, supports event-driven workflows

**Type Safety**: Both paths maintain identical public APIs (structural typing via protocols).

### Test Coverage

**Unit Tests**: `ml/tests/unit/registry/` (each component tested independently)

**Integration Tests**: `ml/tests/integration/registry/` (end-to-end workflows)

**Coverage**: Not measured separately, but registry is heavily tested due to criticality.

**No Metamorphic Tests**: Opportunity for improvement (test registry invariants).

### Known Gaps

1. **Lineage Materialization**: Lineage tracked but not materialized into queryable index
2. **A/B Test Power Analysis**: `welch_t_test` exists, but no prior power calculation
3. **Schema Evolution History**: No version history for schema changes (only current schema stored)
4. **Cross-Registry Constraints**: No enforcement of e.g., "feature must exist in FeatureRegistry before ModelRegistry references it"

---

## ml/ml_registry/ - Abandoned Directory

### Structure

```
ml/ml_registry/
└── datasets/
    (empty)
```

**Total Files**: 0 Python files
**Total Code**: 0 LOC

### History (Inferred)

**Hypothesis**: This directory was created during an earlier registry refactoring but was **abandoned** in favor of:
- `ml/registry/` (current production registry)
- `ml/registry/bootstrap_datasets.py` (dataset bootstrapping)

**Evidence**:
- No imports reference `ml.ml_registry`
- No tests reference this directory
- Empty subdirectory suggests incomplete implementation

**Recommendation**: **Delete this directory** to avoid confusion. All registry functionality lives in `ml/registry/`.

---

## Integration Reality Check

### What Actually Uses What

| Component | Used By | Status |
|-----------|---------|--------|
| **ml/schema/earnings.sql** | `EarningsStore` (as blueprint) | ✅ Implemented via SQLAlchemy |
| **ml/schema/instruments.sql** | `InstrumentMetadataStore` (as blueprint) | ✅ Implemented via SQLAlchemy |
| **ml/schema/cross_asset_features.sql** | **Nothing** | ❌ Features exist, storage missing |
| **ml/evaluation/metrics.py** | Training pipelines, validation CLIs | ✅ Functional |
| **ml/registry/** | All actors via `BaseMLInferenceActor` | ✅ Production |
| **ml/ml_registry/** | **Nothing** | ❌ Abandoned |

### Critical Dependencies

1. **Stores → Schema**: Stores implement schema programmatically, SQL files are reference only
2. **Actors → Registries**: All actors MUST use 4 registries (Pattern 1 enforcement)
3. **Training → Evaluation**: Training uses `roc_auc`, `pr_auc`, `binary_logloss`

### Missing Connections

1. **Schema → Registry**: No automatic table creation from registry manifests
2. **Registry → Stores**: Stores use protocol interfaces, not concrete registry classes (decoupled by design)
3. **Evaluation → Monitoring**: Metrics computed but not exposed to Prometheus
4. **Cross-Asset Features → Stores**: Features computed but not persisted

---

## Recommendations

### 1. ml/schema/ - Clarify Intent

**Options**:
1. **Document as blueprints**: Add README explaining SQL files are reference implementations, actual tables via SQLAlchemy
2. **Remove SQL files**: Delete and rely solely on SQLAlchemy Table definitions in stores
3. **Create migration runner**: Build tool to execute SQL files (adds complexity)

**Recommended**: **Option 1** (document as blueprints). SQL files are useful documentation.

**Action**: Create `ml/schema/README.md` explaining:
- SQL files are design references
- Actual implementation via SQLAlchemy in `ml/stores/`
- Cross-asset features have no store (ephemeral)

### 2. ml/evaluation/ - Sufficient for Current Needs

**No Action Needed** unless:
- Regression models added → Add MSE, MAE, R²
- Calibration needed → Add ECE, MCE
- Drift monitoring needed → Add KS test, PSI (likely in `ml/monitoring/`)

**Current Status**: ✅ **SUFFICIENT**

### 3. ml/registry/ - Stable, Consider Enhancements

**Optional Enhancements**:
1. **Lineage Query API**: `get_feature_lineage(feature_id) -> DAG`
2. **Cross-Registry Constraints**: Validate feature exists before model references it
3. **Schema Version History**: Track schema evolution over time
4. **Metamorphic Tests**: Add invariant tests for registry operations

**Current Status**: ✅ **PRODUCTION READY**

### 4. ml/ml_registry/ - Delete

**Action**: Remove `ml/ml_registry/` directory entirely.

```bash
rm -rf ml/ml_registry
```

### 5. Cross-Asset Features - Create Store or Document Ephemeral

**Options**:
1. **Create `CrossAssetFeatureStore`**: Persist beta/spreads/correlations to database
2. **Document as ephemeral**: Features computed on-demand, not persisted

**Recommended**: **Option 2** (ephemeral) unless backtesting requires historical cross-asset features.

**If persistent storage needed**:
- Implement `ml/stores/cross_asset_feature_store.py`
- Follow pattern from `EarningsStore` (SQLAlchemy Table + upsert)
- Add integration tests

---

## Quick Commands

```bash
# Verify schema files not loaded as migrations
grep -r "earnings.sql\|instruments.sql\|cross_asset_features.sql" ml/stores/migrations
# (Should return nothing - confirms SQL files not used)

# Check registry health
poetry run pytest ml/tests/unit/registry -q

# Verify evaluation metrics work
poetry run pytest ml/tests/unit/evaluation -q

# Count registry components
find ml/registry -name "*.py" | wc -l  # Should be 29

# Verify ml_registry is empty
ls -la ml/ml_registry/datasets/  # Should be empty

# Check stores implement schema
grep -l "Table(" ml/stores/earnings_store.py ml/stores/instrument_metadata_store.py
# (Should return both files - confirms SQLAlchemy implementation)
```

---

## Summary Table

| Directory | Files | LOC | Status | Key Components |
|-----------|-------|-----|--------|----------------|
| **ml/schema/** | 3 SQL | 494 | Blueprint | earnings (✅), instruments (✅), cross_asset (❌) |
| **ml/evaluation/** | 2 Python | 147 | Functional | roc_auc, pr_auc, binary_logloss |
| **ml/registry/** | 29 Python | 16,582 | Production | 4 mandatory registries + decomposed components |
| **ml/ml_registry/** | 0 Python | 0 | Abandoned | Delete recommended |

**Overall Assessment**: Registry is mature and stable. Evaluation is minimal but sufficient. Schema is partially implemented via stores. Cross-asset features need storage decision.
