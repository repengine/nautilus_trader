# Context: Examples & Scripts Modules

## Overview

The `ml/examples/` and `ml/scripts/` directories provide complementary operational tooling for the Nautilus Trader ML system. Examples demonstrate correct usage patterns and integration techniques, while scripts provide production-ready utilities for database operations, data transformation, and system maintenance. Together, they form the practical interface layer between the ML system's architecture and its real-world deployment.

Both directories emphasize **executable, self-contained tools** that adhere to the mandatory architectural patterns defined in CLAUDE.md. All code is production-ready, type-safe, and follows the 5 universal ML architecture patterns.

**Last Updated**: 2025-10-19
**Primary Maintainers**: ML System Team
**Test Coverage**: Examples serve as integration test fixtures; scripts are covered by CLI unit tests

---

## Architecture

### Directory Structure

```
ml/
├── examples/                          # Educational demonstrations & usage patterns
│   ├── Actor Patterns
│   │   ├── simple_ml_actor.py        # 229 lines - Basic ML actor with technical indicators
│   │   ├── mandatory_stores_example.py # 184 lines - 4-store + 4-registry pattern demo
│   │   └── async_persistence_demo.py  # 324 lines - MLPersistenceWorker async patterns
│   │
│   ├── Registry & Backend Integration
│   │   ├── test_registry_backends.py  # 211 lines - Backend functionality testing
│   │   ├── postgres_registry_demo.py  # 263 lines - PostgreSQL vs JSON comparison
│   │   ├── strategy_registry_demo.py  # 370 lines - Strategy lifecycle management
│   │   └── feature_store_example.py   # 220 lines - Training/inference parity
│   │
│   ├── Production Deployment
│   │   ├── dry_run_example.py        # 238 lines - Complete dry-run testing
│   │   ├── scheduler_with_features.py # 144 lines - DataScheduler + FeatureStore
│   │   ├── scheduler_with_metrics.py  # 215 lines - Prometheus metrics demo
│   │   └── tft_with_feature_store.py  # 167 lines - TFT dataset + feature parity
│   │
│   ├── Utilities
│   │   ├── create_dummy_model.py     # 234 lines - ONNX model creation for testing
│   │   └── calendar_provider_demo.py  # 164 lines - Market calendar integration
│   │
└── scripts/                           # Operational utilities & maintenance tools
    ├── Database Migrations
    │   ├── apply_migrations.py        # 19 lines - Compatibility shim → ml.cli
    │   └── convert_stores_to_partitioned.py # 328 lines - Convert tables to partitioned
    │
    ├── Data Transformation
    │   ├── convert_vintage_age.py     # 12 lines - Shim → ml.cli.convert_vintage_age
    │   └── build_tft_dataset.py       # 17 lines - Shim → ml.cli.build_tft_dataset
    │
    ├── Refactoring & Maintenance
    │   ├── refactor_error_handlers.py # 221 lines - Automated error handling refactoring
    │   └── run_streaming_cohort.py    # 387 lines - Event-driven streaming training
    │
    └── __init__.py                    # 12 lines - Module marker
```

### Design Philosophy

**Examples (`ml/examples/`)**: Educational, self-contained demonstrations
- Each example is **independently executable**
- Focuses on **one primary concept** or pattern
- Includes **inline documentation** and usage instructions
- Serves as **integration test fixtures** for architectural patterns
- Demonstrates **correct usage** rather than optimal performance

**Scripts (`ml/scripts/`)**: Production-ready operational utilities
- **Compatibility shims** for CLI consolidation (most moved to `ml/cli/`)
- **Database operations** (migrations, partitioning)
- **Automated refactoring** tools (error handling standardization)
- **Event-driven workflows** (streaming training cohorts)
- Designed for **cron jobs, CI/CD pipelines, and system maintenance**

---

## ml/examples/ - Educational Demonstrations

### Actor Pattern Examples

#### `simple_ml_actor.py` - Basic ML Actor Implementation (229 lines)

**Purpose**: Foundation example for creating custom ML actors with technical indicators.

**Key Demonstrations**:
- `BaseMLInferenceActor` inheritance with proper configuration handling (lines 24-43)
- Technical indicator initialization (`SimpleMovingAverage`, `RelativeStrengthIndex`, `ExponentialMovingAverage`) in `_initialize_features()` (lines 54-63)
- Feature computation with **pre-allocated buffers** for hot-path optimization (lines 52, 91-130)
- Secure ONNX model loading with fallback to `DummyModel` (lines 65-89)
- Zero-allocation patterns: `self._feature_buffer.copy()` reused across bar updates (line 127)

**Type Safety**:
```python
def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
    """Compute feature vector from bar data."""
    # Pre-allocated buffer for zero allocations
    self._feature_buffer[0] = close_price / sma_fast_val
    return self._feature_buffer.copy()
```

**Usage Pattern**:
```python
from ml.examples.simple_ml_actor import SimpleMLActor
from ml.config.base import MLActorConfig

config = MLActorConfig(
    model_path="path/to/model.onnx",
    bar_type=bar_type,
    instrument_id=instrument_id,
)
actor = SimpleMLActor(config)
```

**Security Note**: Explicitly forbids pickle formats (line 71), only supports ONNX via `ml.common.security.secure_onnx_load()` (line 81).

---

#### `mandatory_stores_example.py` - 4-Store + 4-Registry Pattern (184 lines)

**Purpose**: Demonstrates automatic data persistence without manual configuration.

**Key Demonstrations**:
- Automatic store initialization via `BaseMLInferenceActor.__init__()` (lines 30-55)
- All features, predictions, and signals **automatically persisted** by base class (lines 70-120)
- Property accessors: `self._feature_store`, `self._model_store`, `self._strategy_store` (lines 35-37)
- Progressive fallback: PostgreSQL → DummyStore with warning logs (verified in base class)
- No explicit `write_*()` calls needed in actor logic (lines 85-86, 101-102)

**Architecture Compliance**:
```python
class ProductionMLActor(BaseMLInferenceActor):
    """All stores initialized automatically - no configuration needed."""

    def __init__(self, config: MLActorConfig):
        super().__init__(config)  # Stores initialized here

        print(f"Feature store: {type(self._feature_store).__name__}")
        print(f"Model store: {type(self._model_store).__name__}")
        print(f"Strategy store: {type(self._strategy_store).__name__}")
```

**Production Benefits**:
- Complete audit trail for compliance (line 147)
- Enables A/B testing and model drift detection (lines 145-149)
- Replay capabilities for debugging (line 148)
- No additional persistence code required (lines 132-138)

**Pattern Compliance**: ✅ Pattern 1 (Mandatory 4-Store + 4-Registry Integration)

---

#### `async_persistence_demo.py` - Asynchronous Persistence Patterns (324 lines)

**Purpose**: Demonstrates `MLPersistenceWorker` for non-blocking hot-path persistence.

**Key Demonstrations**:
- Synchronous write latency baseline: 10ms/bar with 5ms DB writes (lines 63-102)
- Async write latency: <0.5ms/bar with background queue processing (lines 104-165)
- Backpressure handling with bounded queue: graceful degradation when queue full (lines 167-211)
- Database failure resilience: inference continues despite DB outages (lines 213-295)

**Performance Comparison** (lines 303-320):
```python
# Synchronous (BLOCKING hot path)
store.write_features(...)  # 5ms block
# Result: 10ms/bar average latency

# Async (NON-BLOCKING hot path)
worker.enqueue_features(...)  # <0.1ms
# Result: <0.5ms/bar average latency (20x faster)
```

**Backpressure Behavior** (lines 193-206):
- Queue fills when DB writes slow
- `enqueue_features()` returns `False` when queue full
- Dropped writes tracked via metrics counter
- Prevents memory exhaustion in production

**Database Failure Resilience** (lines 272-294):
- Inference continues even when DB down (line 283)
- Writes queued until DB recovers (line 286)
- Auto-recovery: flush queue when DB returns (lines 290-294)
- **Critical for production uptime**: hot path never blocks on DB

**Pattern Compliance**: ✅ Pattern 3 (Hot/Cold Path Separation) - Persistence is cold path

---

### Registry & Backend Integration Examples

#### `test_registry_backends.py` - Backend Functionality Testing (211 lines)

**Purpose**: Standalone test suite verifying registry PostgreSQL backend support.

**Test Coverage**:
1. **Persistence Layer Tests** (lines 20-85):
   - `BackendType` enum validation (JSON vs POSTGRES)
   - `PersistenceConfig` initialization for both backends
   - `PersistenceManager` JSON operations (save/load)
   - Audit logging to JSONL files (lines 69-82)

2. **SQLAlchemy Models Tests** (lines 88-126):
   - `ModelTable`, `FeatureTable`, `StrategyTable`, `AuditLogTable` schema validation
   - Attribute checking: `model_id`, `extra_metadata`, `entity_type`
   - Schema compatibility verification

3. **Registry Backend Support** (lines 129-151):
   - Configuration capability tests (no full Nautilus import chain)
   - Backend capabilities documentation (JSON vs PostgreSQL)

**Usage**:
```bash
python ml/examples/test_registry_backends.py
# Output: 3 test suites with detailed pass/fail results
```

**Exit Codes** (lines 195-207):
- `0`: All tests passed
- `1`: One or more tests failed

**Integration Note**: Lightweight testing without Nautilus dependencies for CI/CD pipelines.

---

#### `postgres_registry_demo.py` - Backend Comparison Demo (263 lines)

**Purpose**: Compare JSON file-based vs PostgreSQL database persistence patterns.

**Key Demonstrations**:
- **Backend Types**: JSON (development) vs PostgreSQL (production) configuration
- **Teacher-Student Models**: Knowledge distillation workflow tracking
- **Model Deployment**: Deployment status and target environment tracking
- **ACID Compliance**: Transactional operations and concurrent access support
- **Migration Utilities**: Tools for migrating from JSON to PostgreSQL

**Configuration Patterns**:
```python
# JSON Backend (Development)
json_config = PersistenceConfig(
    backend=BackendType.JSON,
    json_path=Path("ml_registry")
)

# PostgreSQL Backend (Production)
postgres_config = PersistenceConfig(
    backend=BackendType.POSTGRES,
    connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
)
```

**Production Features**:
- Automatic versioning and timestamps
- Full audit trail of all changes
- ACID compliance for multi-user environments
- Concurrent access without file locking issues

**Migration Path**: Provides example code for converting JSON manifests to PostgreSQL.

---

#### `strategy_registry_demo.py` - Strategy Lifecycle Management (370 lines)

**Purpose**: Comprehensive demonstration of `StrategyRegistry` features.

**Key Demonstrations**:
- **Strategy Manifests**: Complete metadata including performance metrics, requirements
- **Market Regime Matching**: Query strategies by market conditions (`MarketRegime.TRENDING_UP`)
- **Performance Ranking**: Sort strategies by Sharpe ratio, return, or custom metrics
- **Compatibility Checking**: Validate strategy requirements against available resources
- **Strategy Lineage**: Track strategy evolution and parent-child relationships
- **Live Performance Updates**: Update metrics from live trading results

**Usage Pattern**:
```python
# Query strategies by market conditions
trending_strategies = registry.get_strategies_for_regime(MarketRegime.TRENDING_UP)

# Validate requirements before deployment
can_run = registry.validate_requirements(
    strategy_id="trend_follow_ma_cross",
    available_models=["lgb_directional_v1"],
    available_features=["sma_20", "rsi_14"]
)

# Update live performance metrics
registry.update_live_performance(
    strategy_id="mean_reversion_v1",
    sharpe_ratio=1.85,
    total_return=0.23,
    max_drawdown=-0.08
)
```

**Manifest Structure**:
- Strategy metadata: name, version, description
- Performance metrics: Sharpe, return, drawdown, win rate
- Requirements: required models, features, data sources
- Regime suitability: trending, ranging, volatile, calm
- Deployment status: development, staging, production

---

#### `feature_store_example.py` - Training/Inference Parity (220 lines)

**Purpose**: Demonstrate guaranteed feature parity between training and inference.

**Key Demonstrations**:
- **Training Mode**: Features stored during model training
- **Inference Mode**: Features retrieved for consistency verification
- **Parity Validation**: `np.testing.assert_allclose(..., rtol=1e-10)` ensures identical features
- **Batch vs Online**: Demonstrates parity between batch preprocessing and online computation
- **Feature Versioning**: Track feature computation changes over time

**Parity Validation Pattern**:
```python
# Training: Store features
feature_store.write_features(
    feature_set_id="sma_rsi_v1",
    instrument_id="EUR/USD",
    features={"sma_20": 1.0850, "rsi_14": 65.3},
    ts_event=bar.ts_event,
    ts_init=bar.ts_init
)

# Inference: Verify features match
stored_features = feature_store.read_features(...)
np.testing.assert_allclose(
    computed_features,
    stored_features,
    rtol=1e-10  # Strict tolerance for parity
)
```

**Production Benefits**:
- Prevents training/serving skew
- Enables reproducible model evaluation
- Supports feature rollback and A/B testing
- Audit trail for regulatory compliance

---

### Production Deployment Examples

#### `dry_run_example.py` - Complete Dry-Run Testing (238 lines)

**Purpose**: Risk-free testing framework for ML trading systems.

**Key Demonstrations**:
- **Risk-Free Testing**: Execute trades disabled, signal generation only
- **Backtest Engine Integration**: Historical data simulation without live connections
- **Configuration Validation**: Verify all components work together correctly
- **Error Simulation**: Test failure scenarios and recovery mechanisms
- **Deployment Preparation**: Step-by-step production readiness checklist

**Execution Modes**:
```bash
# Backtest mode (default)
python ml/examples/dry_run_example.py

# Live dry run mode (paper trading)
python ml/examples/dry_run_example.py --live
```

**Safety Features**:
- Orders never submitted to real exchanges
- Account balances are simulated
- Network failures don't affect testing
- Reproducible results with fixed random seeds

**Production Readiness Checklist**:
1. Dry run passes without errors
2. Performance metrics meet SLAs
3. Error handling tested and verified
4. Logging and monitoring operational
5. Configuration validated

---

#### `scheduler_with_features.py` - DataScheduler Integration (144 lines)

**Purpose**: Automated data collection with feature computation.

**Key Demonstrations**:
- **Databento Integration**: Real-time market data ingestion configuration
- **FeatureStore Integration**: Automatic feature computation and storage
- **Symbol Universe Management**: Multi-asset data collection configuration
- **Retention Policies**: Automatic data cleanup and archival
- **Error Handling**: Retry logic and failure recovery mechanisms

**Configuration Pattern**:
```python
config = SchedulerConfig(
    symbols=["AAPL.XNAS", "MSFT.XNAS", "GOOGL.XNAS"],
    retention_days=30,
    databento=DatabentoConfig(
        dataset="EQUS.MINI",
        schema="ohlcv-1m"
    ),
    max_retries=2,
    retry_delay_seconds=5
)

scheduler = DataScheduler(config, catalog)
scheduler.start()
```

**Production Features**:
- Scheduled daily updates
- Historical backfill support
- Data quality validation
- Gap detection and filling

---

#### `scheduler_with_metrics.py` - Prometheus Metrics Demo (215 lines)

**Purpose**: Comprehensive metrics collection and monitoring demonstration.

**Key Demonstrations**:
- **Metrics Endpoint**: HTTP server on port 8000 serving `/metrics` and `/health` (lines 30-68)
- **Key Metrics Tracking**: Data collection, feature computation, freshness, errors (lines 47-54)
- **Health Monitoring**: Component status verification (lines 56-62)
- **Request Patterns**: Safe timeout handling with `_REQUEST_TIMEOUT` env var (line 22)

**Metrics Endpoint Usage**:
```python
# Fetch metrics programmatically
response = requests.get("http://localhost:8000/metrics", timeout=5.0)
metrics_text = response.text

# Parse Prometheus format
for line in metrics_text.split("\n"):
    if line.startswith("nautilus_ml_"):
        print(line)  # nautilus_ml_predictions_total{model_id="..."} 1234
```

**Health Check Pattern**:
```python
health_response = requests.get("http://localhost:8000/health", timeout=5.0)
# Response: {"status": "healthy", "components": {"scheduler": "ok", "catalog": "ok"}}
```

**Monitored Metrics** (examples):
- `nautilus_ml_predictions_total` - Total predictions made
- `nautilus_ml_prediction_latency_seconds` - Inference latency histogram
- `nautilus_ml_feature_computation_errors_total` - Feature computation failures
- `nautilus_ml_data_freshness_seconds` - Data staleness tracking

**Pattern Compliance**: ✅ Pattern 5 (Centralized Metrics Bootstrap) - Uses `ml.common.metrics_bootstrap`

---

#### `tft_with_feature_store.py` - TFT Dataset with Feature Parity (167 lines)

**Purpose**: Advanced model training with guaranteed training/inference feature parity.

**Key Demonstrations**:
- **TFTDatasetBuilder Integration**: Polars-based efficient data preparation
- **PostgreSQL Integration**: Direct access to Nautilus data warehouse
- **Feature Source Validation**: Verify features come from same computation engine
- **Batch vs Online Parity**: Validate identical feature computation between modes
- **Training Dataset Construction**: Large-scale dataset building with feature consistency

**Parity Validation Workflow**:
```python
# 1. Build training dataset from FeatureStore
dataset_builder = TFTDatasetBuilder(feature_store, model_registry)
training_data = dataset_builder.build(
    instruments=["EUR/USD", "GBP/USD"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# 2. Train model on stored features
model.fit(training_data)

# 3. During inference, verify feature parity
online_features = actor.compute_features(bar)
stored_features = feature_store.read_features(...)
assert_feature_parity(online_features, stored_features)
```

**Production Benefits**:
- Eliminates training/serving skew
- Reproducible model evaluation
- Feature versioning for A/B tests
- Compliance audit trail

---

### Utility Examples

#### `create_dummy_model.py` - ONNX Model Creation for Testing (234 lines)

**Purpose**: Create secure ONNX models for dry-run testing and development.

**Security-First Design**:
- **ONNX-only exports**: Explicitly forbids pickle formats (line 179)
- **Reproducible models**: Deterministic RNG with fixed seeds (lines 47, 67, 128)
- **Production safety**: Models created using `skl2onnx` for secure inference (lines 144-171)

**Model Types Created**:
1. **Bullish Model** (`dummy_bullish_model.onnx`): Bias toward BUY signals (lines 188-194)
   - Class weights: `{0: 0.8, 1: 1.2}` favoring positive class
2. **Bearish Model** (`dummy_bearish_model.onnx`): Bias toward SELL signals (lines 197-203)
   - Class weights: `{0: 1.2, 1: 0.8}` favoring negative class
3. **Neutral Model** (`dummy_neutral_model.onnx`): Balanced signals (lines 206-212)
   - No class weights, equal distribution

**Implementation Details**:
```python
# Create sklearn pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        class_weight={0: 0.8, 1: 1.2}  # Bullish bias
    ))
])

# Export to ONNX for secure inference
export_to_onnx(model, output_path, feature_names)
```

**Usage**:
```bash
python ml/examples/create_dummy_model.py
# Creates 3 ONNX models in ml/models/
# - dummy_bullish_model.onnx
# - dummy_bearish_model.onnx
# - dummy_neutral_model.onnx
```

**DummyModel Class** (lines 35-89):
- Simple in-memory model for testing without ONNX dependencies
- Linear combination with sigmoid activation (lines 62-64)
- Deterministic noise for reproducibility (lines 67-69)
- Compatible with sklearn API (`predict()`, `predict_proba()`)

**Security Note** (line 228): "These models use ONNX format for production safety. Legacy pickle models are no longer supported."

---

#### `calendar_provider_demo.py` - Market Calendar Integration (164 lines)

**Purpose**: Demonstrate market schedule and trading hours integration.

**Key Demonstrations**:
- **Multi-Exchange Support**: NYSE, NASDAQ, LSE, JPX, crypto markets
- **Holiday Detection**: Automatic market closure identification
- **Trading Hours**: Pre-market, regular hours, after-hours detection
- **Time Zone Handling**: Proper UTC and local time conversions
- **Schedule Queries**: Market open/close times and remaining trading time

**Usage Pattern**:
```python
from ml.examples.calendar_provider_demo import get_market_calendar

# Get NYSE calendar
nyse = get_market_calendar("NYSE")

# Check if market open
is_open = nyse.is_open_now()

# Get trading hours for date
schedule = nyse.schedule(start_date, end_date)

# Check for holidays
is_holiday = nyse.is_holiday(date)
```

**Supported Exchanges**:
- US: NYSE, NASDAQ, CME
- Europe: LSE, EUREX
- Asia: JPX, HKEX
- Crypto: 24/7 markets

---

## ml/scripts/ - Operational Utilities

### Overview

Most operational scripts have been **consolidated into `ml/cli/`** (as of Sept 2024). The remaining scripts in `ml/scripts/` are:
1. **Compatibility shims** - Delegate to `ml/cli/` for backward compatibility
2. **Database utilities** - Advanced partitioning and conversion tools
3. **Refactoring tools** - Automated code transformation utilities
4. **Event-driven workflows** - Streaming training orchestration

**Total Lines**: ~960 lines across 7 files (down from original ~2000+ lines pre-consolidation)

---

### Database Migration Scripts

#### `apply_migrations.py` - CLI Compatibility Shim (19 lines)

**Purpose**: Preserve documented entry points while delegating to `ml.cli.apply_migrations`.

**Implementation**:
```python
from ml.cli.apply_migrations import main as _main

def main(argv: list[str] | None = None) -> int:
    return _main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
```

**Usage** (documented in Makefile and README):
```bash
# Via python -m
python -m ml.scripts.apply_migrations --db-url postgresql://...

# Delegates to ml.cli.apply_migrations internally
```

**Rationale**: Maintains backward compatibility for existing CI/CD pipelines and documentation while consolidating implementation in `ml/cli/`.

---

#### `convert_stores_to_partitioned.py` - Table Partitioning Utility (328 lines)

**Purpose**: Convert ML store tables to monthly-partitioned tables for improved performance and manageability.

**Tables Covered** (lines 46-129):
1. **ml_feature_values** - Feature storage with `ts_event` partitioning key
2. **ml_model_predictions** - Prediction history with `ts_event` partitioning key
3. **ml_strategy_signals** - Strategy signals with `ts_event` partitioning key

**Conversion Process** (lines 208-269):
1. **Check if already partitioned** - Skip if table already uses `PARTITION BY RANGE` (lines 132-146)
2. **Create new partitioned parent** - `{table}_p` with monthly range partitioning (lines 217-223)
3. **Pre-create partitions** - Cover existing data range + `--ahead N` months (lines 225-252)
4. **Copy rows** - Stream data from old table to partitioned parent (lines 254-256)
5. **Swap names** - Rename old to `{table}_legacy`, new to `{table}` (lines 258-263)
6. **Recreate indexes** - Standard indexes on partitioned table (lines 265-267)
7. **Rename child partitions** - Ensure canonical naming `{table}_YYYY_MM` (lines 271-295)

**Partition Naming** (lines 165-183):
```python
# Canonical format: ml_feature_values_2025_10
name = f"{base_table}_{year:04d}_{month:02d}"

# Example partitions created:
# ml_feature_values_2024_01  FOR VALUES FROM (ts_jan_2024) TO (ts_feb_2024)
# ml_feature_values_2024_02  FOR VALUES FROM (ts_feb_2024) TO (ts_mar_2024)
# ...
```

**Usage**:
```bash
# Convert all three tables, pre-create 3 months ahead
uv run --active --no-sync python -m ml.scripts.convert_stores_to_partitioned \
    --db-url postgresql://postgres:postgres@localhost:5432/nautilus \
    --ahead 3

# Subset of tables
python -m ml.scripts.convert_stores_to_partitioned \
    --tables ml_feature_values,ml_model_predictions \
    --ahead 3

# Preview without executing
python -m ml.scripts.convert_stores_to_partitioned --dry-run
```

**Performance Benefits**:
- **Query optimization**: Partition pruning for time-range queries
- **Maintenance efficiency**: Drop old partitions instead of DELETE operations
- **Index locality**: Smaller indexes per partition
- **Parallel operations**: Concurrent partition maintenance

**Safety Features**:
- Dry-run mode to preview changes (line 303)
- Existing partitioned tables skipped (lines 209-210)
- Old table preserved as `{table}_legacy` for rollback (line 262)
- Safe for dev/test DBs; production requires maintenance window (line 24)

**Implementation Notes**:
- Uses `ml.core.db_engine.EngineManager.get_engine()` for connection management (line 313)
- Timestamp conversion: nanoseconds to months using `_ts_month_floor()` (lines 149-151)
- Partition iteration via `_iter_months()` generator (lines 154-162)
- Schema-aware column mapping for `created_at` handling (lines 195-196)

---

### Data Transformation Scripts

#### `convert_vintage_age.py` - Vintage Age CLI Shim (12 lines)

**Purpose**: Compatibility wrapper delegating to `ml.cli.convert_vintage_age`.

**Implementation**:
```python
from ml.cli.convert_vintage_age import main

if __name__ == "__main__":
    raise SystemExit(main())
```

**Actual Implementation**: See `ml/cli/convert_vintage_age.py` (144 lines) for full logic.

**What It Does** (from CLI implementation):
- Converts `*_value_vintage_ts` columns to `*_vintage_age_minutes` numeric features
- Streams parquet dataset for memory efficiency (batch_size=32,768)
- Updates accompanying `dataset_metadata.json` with new column info
- Preserves original vintage timestamp columns as backup

**Usage**:
```bash
python -m ml.scripts.convert_vintage_age \
    --source ml_out/full_tft_95/dataset.parquet \
    --metadata ml_out/full_tft_95/dataset_metadata.json
```

---

#### `build_tft_dataset.py` - TFT Dataset CLI Shim (17 lines)

**Purpose**: Compatibility wrapper for legacy import path used by tests.

**Implementation**:
```python
from ml.cli.build_tft_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
```

**Actual Implementation**: See `ml/cli/build_tft_dataset.py` for full dataset building logic.

**What It Does** (from CLI implementation):
- Orchestrates complete TFT dataset construction pipeline
- Handles L0/L1 historical data → microstructure features → regime indicators
- Constructs training-ready datasets with 20M+ samples
- Outputs TFT-compatible parquet files with metadata

---

### Refactoring & Maintenance Scripts

#### `refactor_error_handlers.py` - Automated Error Handling Refactoring (221 lines)

**Purpose**: Automatically refactor try/except patterns to use standardized error handling utilities from `ml.common.error_handlers`.

**Refactoring Patterns Detected** (lines 33-47):
```python
# Pattern: Registry fallback with warning + return
try:
    # registry operation
except Exception as e:
    logger.warning("...", e)
    return None  # or [] or {}

# Refactored to:
@with_fallback(fallback_value=None, log_level='warning')
def registry_operation():
    # registry operation
```

**Workflow** (lines 169-219):
1. **Find candidates**: Use grep to locate files with `except Exception as e:` patterns (lines 57-84)
2. **Analyze imports**: Determine which error handler imports needed (lines 87-97)
3. **Add imports**: Insert `from ml.common.error_handlers import ...` after last `ml.*` import (lines 99-144)
4. **Apply patterns**: Regex-based transformation to decorator patterns (lines 147-166)
5. **Report results**: Summary of files modified, imports added, patterns refactored (lines 207-217)

**Usage**:
```bash
# Preview changes without applying
python ml/scripts/refactor_error_handlers.py --dry-run

# Refactor top 50 files
python ml/scripts/refactor_error_handlers.py --top-n 50

# Apply changes to specific directory
python ml/scripts/refactor_error_handlers.py --ml-dir ml/registry
```

**Error Handlers Imported** (lines 125-133):
- `db_operation_handler` - Database operation error handling
- `registry_operation_handler` - Registry operation error handling
- `with_db_error_handling` - Decorator for DB error handling
- `with_fallback` - Decorator for fallback value patterns

**Safety Features**:
- Dry-run mode for preview (lines 138-142, 158-160)
- Top-N file limiting to prevent mass changes (line 173)
- Per-file change reporting (lines 195-205)
- Exit code 0 (advisory mode) - doesn't fail builds

**Implementation Notes**:
- Uses `ml.common.subprocess_utils.run_command()` for grep execution (line 61)
- Regex patterns compiled for performance (lines 35-46)
- Named tuple `RefactoringPattern` for maintainability (lines 24-30)

---

#### `run_streaming_cohort.py` - Event-Driven Streaming Training (387 lines)

**Purpose**: Run a single streaming TFT cohort end-to-end using event-driven helpers.

**Workflow Overview** (lines 6-19):
1. **Load dataset metadata** - Including vintage-age features from `dataset_metadata.json`
2. **Plan streaming cohort** - Via `StreamingDatasetPlanner` with row/sequence/shard constraints
3. **Execute training worker** - `LightningStreamingWorker` produces logits/metrics
4. **Persist events** - Mirror emitted events into snapshot for dashboard consumption

**Key Components**:

**1. Dataset Planning** (lines 127-153):
```python
planner = StreamingDatasetPlanner(
    DatasetServiceConfig(
        parquet_root=str(inputs.dataset_dir),
        shard_row_budget=shard_row_budget,
        max_total_rows=max_total_rows,
        max_total_sequences=max_total_sequences,
        max_shards=max_shards,
    )
)
plan = planner.plan(request)  # Returns DatasetPlanEvent
```

**2. Worker Execution** (lines 179-193):
```python
worker = LightningStreamingWorker(worker_config, output_dir=output_dir)
result = worker.run(plan)  # Returns StreamingRunTelemetry + metrics + artifacts

print(f"training status: {result.status.value}")
print(f"metrics: {json.dumps(result.metrics)}")
print(f"gpu_peak_mb: {result.telemetry.max_gpu_memory_mb}")
print(f"logits_path: {result.artifact_paths['logits']}")
```

**3. Event Persistence** (lines 195-221):
```python
service = StreamingTrainingPersistenceService.create(state_path=state_path)

# Emit dataset plan event
plan_message = build_plan_message(plan).as_dict()
service.handle(f"events.ml.DATASET_PLANNED.{plan.dataset_id}", plan_message)

# Emit training result event
result_message = build_result_message(result_payload).as_dict()
service.handle(f"events.ml.MODEL_TRAINING_COMPLETED.{plan.dataset_id}", result_message)

# Write snapshot to JSON
snapshot = service.snapshot()
state_path.write_text(json.dumps(snapshot))
```

**Configuration Parameters** (lines 223-343):
- `--dataset-dir`: Parquet dataset + metadata location (required)
- `--output-dir`: Worker artifacts output (logits, telemetry) (required)
- `--state-path`: Persistence snapshot JSON path (default: `ml_out/streaming_training_state_snapshot.json`)
- `--max-total-rows`: Max rows per cohort (default: 120,000)
- `--max-total-sequences`: Max sequences per cohort (default: 90,000)
- `--max-shards`: Max shards per cohort (default: 32)
- `--batch-size`: DataLoader batch size (default: 48)
- `--accelerator`: Lightning accelerator (`cpu`, `gpu`, `auto`)
- `--devices`: Number of Lightning devices (default: 1)
- `--gpu-monitor-interval`: GPU memory sampling interval (default: 30.0s)

**Example Usage**:
```bash
poetry run python -m ml.scripts.run_streaming_cohort \
    --dataset-dir ml_out/full_tft_95 \
    --state-path ml_out/streaming_training_state_snapshot.json \
    --max-total-rows 120000 \
    --max-total-sequences 90000 \
    --max-shards 32 \
    --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \
    --accelerator cpu
```

**Telemetry Tracking** (lines 189-192):
- Max GPU memory usage (MB)
- Training duration
- Artifact paths (logits, checkpoints)
- Metrics (ROC-AUC, validation loss, etc.)

**Integration Points**:
- `ml.config.streaming_pipeline`: `StreamingWorkerConfig`, `DatasetServiceConfig`
- `ml.training.event_driven`: Event-driven training services and payloads
- `ml.consumers.streaming_training_service`: Persistence service for dashboard
- `ml.training.teacher.streaming_loader`: `TFTStreamingConfig` for dataset configuration

**Production Readiness**:
- Cohort-based training for large datasets
- GPU memory monitoring
- Event-driven observability
- State persistence for resumption
- Configurable resource constraints

---

## Integration Points

### Examples Integration

**Actor Framework** (Nautilus Trader):
- Seamless integration with actor lifecycle (`on_start()`, `on_stop()`)
- Proper use of domain types: `InstrumentId`, `BarType`, `ComponentId`
- Message bus integration for signal publishing
- Event system participation

**ML Pipeline**:
- Automatic integration with 4 mandatory stores (`FeatureStore`, `ModelStore`, `StrategyStore`, `DataStore`)
- Registry system integration for component lifecycle
- Feature engineering with `FeatureEngineer` and parity validation
- Multi-format model support with security validation

**Data Sources**:
- Real-time and historical data ingestion
- External API integration (Databento for professional market data)
- Calendar systems for trading schedule and holiday detection
- Database systems (PostgreSQL and SQLite)

**Monitoring & Observability**:
- Prometheus metrics collection and export
- Health monitoring with circuit breakers
- Performance tracking with latency monitoring
- Audit logging for compliance

---

### Scripts Integration

**Database Operations**:
- Uses `ml.core.db_engine.EngineManager.get_engine()` for connection management
- SQLAlchemy for schema operations and migrations
- Partitioning support for time-series tables
- Safe schema evolution with legacy table preservation

**CLI Consolidation**:
- Most operational scripts moved to `ml/cli/` for centralization
- Compatibility shims in `ml/scripts/` preserve backward compatibility
- Consistent argument parsing via `argparse`
- Makefile integration for common operations

**Event-Driven Workflows**:
- Redis Streams integration via `MessageBusConfig`
- Event emission for dashboard consumption
- State persistence to JSON snapshots
- Telemetry tracking for GPU memory, training metrics

**CI/CD Integration**:
- `--dry-run` modes for safe preview
- Exit code conventions (0 = success, 1 = failure)
- JSON output for programmatic consumption
- Logging to stdout/stderr for capture

---

## Usage Patterns

### Learning Path for Examples

**1. Foundation** → `simple_ml_actor.py`:
- Learn basic actor patterns and feature computation
- Understand hot-path optimization with pre-allocated buffers
- See secure ONNX model loading patterns

**2. Data Management** → `mandatory_stores_example.py`:
- Understand automatic store integration
- Learn progressive fallback patterns (PostgreSQL → DummyStore)
- See audit trail and compliance benefits

**3. Performance** → `async_persistence_demo.py`:
- Understand hot/cold path separation
- Learn backpressure handling patterns
- See database failure resilience in action

**4. Registry Systems** → `test_registry_backends.py`, `strategy_registry_demo.py`:
- Learn backend configuration (JSON vs PostgreSQL)
- Understand strategy lifecycle management
- See performance tracking and compatibility checking

**5. Production Deployment** → `dry_run_example.py`:
- Learn risk-free testing workflows
- Understand configuration validation patterns
- See error simulation and recovery testing

**6. Monitoring** → `scheduler_with_metrics.py`:
- Learn Prometheus metrics patterns
- Understand health check implementations
- See observability best practices

**7. Advanced Integration** → `tft_with_feature_store.py`:
- Understand training/inference parity validation
- Learn complex dataset building workflows
- See production feature engineering patterns

---

### Common Script Operations

**Database Migrations**:
```bash
# Apply baseline migrations
python -m ml.scripts.apply_migrations \
    --db-url postgresql://postgres:postgres@localhost:5432/nautilus

# Apply full migration set (hardening, views, fixes)
python -m ml.scripts.apply_migrations \
    --db-url postgresql://... \
    --full
```

**Table Partitioning**:
```bash
# Convert all three tables to partitioned
python -m ml.scripts.convert_stores_to_partitioned \
    --db-url postgresql://postgres:postgres@localhost:5432/nautilus \
    --ahead 3

# Preview changes without executing
python -m ml.scripts.convert_stores_to_partitioned --dry-run
```

**Data Transformation**:
```bash
# Convert vintage timestamps to age features
python -m ml.scripts.convert_vintage_age \
    --source ml_out/full_tft_95/dataset.parquet \
    --metadata ml_out/full_tft_95/dataset_metadata.json
```

**Streaming Training**:
```bash
# Run streaming cohort with GPU
python -m ml.scripts.run_streaming_cohort \
    --dataset-dir ml_out/full_tft_95 \
    --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \
    --accelerator gpu \
    --devices 1
```

**Error Handler Refactoring**:
```bash
# Preview refactoring changes
python ml/scripts/refactor_error_handlers.py --dry-run --top-n 50

# Apply refactoring to top 50 files
python ml/scripts/refactor_error_handlers.py --top-n 50
```

---

## Implementation Notes

### Examples Design Principles

**Self-Contained Execution**:
- Each example can run independently without complex setup
- Minimal external dependencies beyond core ML system
- Clear inline documentation and usage instructions

**Educational Focus**:
- Focus on **correct usage** over optimal performance
- Demonstrate **one primary concept** per example
- Include **error cases** and **recovery patterns**

**Production Patterns**:
- Follow 5 universal ML architecture patterns
- Use secure model formats (ONNX only)
- Implement proper error handling and logging
- Include metrics and monitoring patterns

**Type Safety**:
- Complete type annotations on all functions/methods
- Strict mypy compliance
- Explicit return types and parameter types

---

### Scripts Design Principles

**Backward Compatibility**:
- Shims preserve documented entry points
- Delegate to `ml/cli/` for implementation
- Maintain Makefile integration

**Safety First**:
- Dry-run modes for preview
- Legacy table preservation on conversions
- Explicit confirmation for destructive operations

**Automation-Friendly**:
- JSON output modes for programmatic consumption
- Standard exit codes (0 = success, 1 = failure)
- Logging to stdout/stderr for capture
- Environment variable configuration

**Resource Efficiency**:
- Streaming operations for large datasets
- Batch processing with configurable sizes
- Memory-efficient algorithms (no full dataset loads)

---

## Known Gaps & Future Work

### Examples

**Missing Coverage**:
- ❌ **Ensemble models example**: Demonstrate model blending and voting strategies
- ❌ **Real-time streaming example**: Live data ingestion and inference demo
- ❌ **Multi-instrument portfolio example**: Cross-asset strategy coordination
- ❌ **Model A/B testing example**: Show registry-based A/B test patterns

**Planned Enhancements**:
- 🔄 **Graph neural network example**: Demonstrate order book GNN integration
- 🔄 **Reinforcement learning example**: Show RL agent integration patterns
- 🔄 **Knowledge distillation example**: Teacher-student model workflow

**Documentation Gaps**:
- Integration with Docker Compose deployment
- Complete end-to-end production deployment guide
- Performance tuning guidelines per example

---

### Scripts

**CLI Consolidation Status** (Sept 2024):
- ✅ Most operational scripts moved to `ml/cli/`
- ✅ Compatibility shims in place for backward compatibility
- ✅ Makefile updated with new entry points

**Remaining Consolidation**:
- `refactor_error_handlers.py` - Could move to `ml/cli/refactor.py`
- `run_streaming_cohort.py` - Could move to `ml/cli/training_cohort.py`

**Planned Enhancements**:
- 🔄 **Partition management CLI**: Automated partition creation/deletion for time-series tables
- 🔄 **Schema evolution tools**: Automated schema migration generation
- 🔄 **Data quality checks**: Automated validation of ingested data

**Missing Utilities**:
- ❌ **Backup/restore scripts**: Database backup and restore automation
- ❌ **Performance profiling**: SQL query performance analysis tools
- ❌ **Index management**: Automated index creation/deletion based on usage

---

## Testing Strategy

### Example Testing

**Examples as Test Fixtures**:
- Examples serve as **integration tests** for architectural patterns
- `test_registry_backends.py` provides standalone test suite (3 test suites, lines 154-207)
- CI/CD runs examples to verify system health

**Test Coverage Requirements**:
- Examples themselves are not unit-tested (they ARE the tests)
- Components demonstrated by examples have ≥90% ML module coverage
- Examples must pass `mypy --strict` and `ruff check`

**Validation**:
```bash
# Run example as integration test
python ml/examples/test_registry_backends.py
# Expected exit code: 0 (all tests passed)

# Verify type safety
mypy ml/examples --strict

# Verify linting
ruff check ml/examples
```

---

### Script Testing

**Unit Test Coverage**:
- CLI scripts have dedicated unit tests in `ml/tests/unit/cli/`
- Database scripts tested with mock engines (no real DB required)
- Refactoring scripts tested with fixture files

**Integration Tests**:
- `convert_stores_to_partitioned.py` tested with local PostgreSQL
- `run_streaming_cohort.py` tested in `ml/tests/integration/cli/`
- Event emission verified via persistence service tests

**Test Locations**:
- `ml/tests/unit/cli/test_streaming_persistence_worker_cli_unit.py` - Worker CLI tests
- `ml/tests/integration/cli/` - Integration tests for CLI consolidation
- `ml/tests/contracts/test_streaming_payloads.py` - Event payload validation

**Validation Commands**:
```bash
# Run CLI unit tests
pytest ml/tests/unit/cli/ -q

# Run CLI integration tests
pytest ml/tests/integration/cli/ -q

# Validate streaming payloads
pytest ml/tests/contracts/test_streaming_payloads.py -q
```

---

## Cross-Module References

**Related Documentation**:
- **Actors**: `context_actors.md` - ML actor architecture and base classes
- **Stores**: `context_stores.md` - Persistence layer implementation details
- **Registry**: `context_registry.md` - Component lifecycle and schema management
- **Config**: `context_config.md` - Configuration system patterns
- **CLI**: `ml/cli/README_PIPELINE.md` - Production CLI documentation
- **Testing**: `ml/tests/docs/TESTING_STRATEGY.md` - Testing methodology

**Architecture Guides**:
- **CLAUDE.md**: Core development standards and 5 universal patterns
- **ROADMAP.md**: System evolution and planned features
- **CODING_STANDARDS.md**: Type safety, documentation, and quality gates

**Operational Documentation**:
- **deployment/README.md**: Docker Compose deployment guide
- **stores/migrations/README.md**: Database migration strategy
- **BUS_PUBLISHING_STANDARDIZATION.md**: Event publishing patterns

---

## Universal Pattern Compliance

All examples and scripts demonstrate compliance with the 5 universal ML architecture patterns defined in CLAUDE.md:

### ✅ Pattern 1: Mandatory 4-Store + 4-Registry Integration

**Examples Demonstrating**:
- `mandatory_stores_example.py` - Automatic store initialization (lines 30-55)
- `simple_ml_actor.py` - Base class store integration
- `feature_store_example.py` - FeatureStore usage patterns
- `async_persistence_demo.py` - Asynchronous store writes

**Key Points**:
- Stores initialized automatically via `BaseMLInferenceActor.__init__()`
- Progressive fallback: PostgreSQL → DummyStore with warnings
- Property accessors provide clean interface: `.feature_store`, `.model_store`
- No manual persistence code required in actors

---

### ✅ Pattern 2: Protocol-First Interface Design

**Examples Demonstrating**:
- `test_registry_backends.py` - Protocol compatibility testing (SQLAlchemy models)
- `async_persistence_demo.py` - `SlowStore` adheres to store protocols (lines 21-60)

**Key Points**:
- Structural typing without implementation coupling
- Duck typing support for testing (DummyStore conforms)
- Type safety without circular dependencies

---

### ✅ Pattern 3: Hot/Cold Path Separation

**Examples Demonstrating**:
- `async_persistence_demo.py` - Hot path (inference) vs cold path (persistence) (lines 63-165)
- `simple_ml_actor.py` - Pre-allocated feature buffers for hot path (line 52)

**Scripts Demonstrating**:
- `convert_stores_to_partitioned.py` - Partition management is cold path operation

**Performance Budget**:
- Hot path P99 < 5ms enforced
- Zero allocations in tight loops
- Model loading relegated to initialization
- Persistence happens asynchronously

---

### ✅ Pattern 4: Progressive Fallback Chains

**Examples Demonstrating**:
- `mandatory_stores_example.py` - PostgreSQL → DummyStore fallback (base class)
- `test_registry_backends.py` - Backend fallback testing (JSON vs PostgreSQL)
- `async_persistence_demo.py` - Database failure resilience (lines 213-295)

**Fallback Order**:
- PRIMARY → CACHED → FILE → DUMMY
- PostgreSQL → DummyStore (no persistence, warnings logged)
- Registry loading → Direct file loading
- Network failures → Local caches

---

### ✅ Pattern 5: Centralized Metrics Bootstrap

**Examples Demonstrating**:
- `scheduler_with_metrics.py` - Prometheus metrics via bootstrap (lines 30-68)
- All actors use `ml.common.metrics_bootstrap.get_counter()`, `get_histogram()`

**Key Points**:
- NEVER import `prometheus_client` directly
- Safe for module reloads and testing
- Consistent naming and labeling
- DTO builders + service pattern

---

## File Statistics

**Examples** (13 files, ~2,600 lines):
- Actor patterns: 3 files, 737 lines
- Registry integration: 4 files, 1,064 lines
- Production deployment: 4 files, 764 lines
- Utilities: 2 files, 398 lines

**Scripts** (7 files, ~960 lines):
- Database migrations: 2 files, 347 lines
- Data transformation: 2 files, 29 lines (shims)
- Refactoring tools: 2 files, 608 lines
- Module marker: 1 file, 12 lines

**Combined Total**: 20 files, ~3,560 lines of production-ready code

---

## Summary

The **ml/examples/** and **ml/scripts/** directories provide the operational interface to the Nautilus Trader ML system:

**Examples** offer **educational, self-contained demonstrations** of correct usage patterns:
- Actor implementation with automatic store integration
- Registry backend configuration and lifecycle management
- Production deployment patterns with dry-run testing
- Performance optimization via async persistence
- Feature parity validation for training/inference consistency
- Secure ONNX model creation for testing

**Scripts** provide **production-ready utilities** for system operations:
- Database migration and partitioning tools
- Compatibility shims for CLI consolidation
- Automated refactoring utilities
- Event-driven streaming training orchestration

**Together**, they ensure developers can:
1. **Learn** the ML system through executable examples
2. **Deploy** with confidence using proven patterns
3. **Operate** efficiently with automated utilities
4. **Maintain** system health with diagnostic tools

All code follows the **5 universal ML architecture patterns**, maintains **strict type safety**, and provides **comprehensive observability** for production environments.
