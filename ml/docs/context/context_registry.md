# Registry System Context Document

## Executive Summary

The `ml/registry/` directory implements a comprehensive, production-ready ML lifecycle management system with self-describing manifests, configurable persistence backends, and automated compatibility validation. This system serves as the central orchestrator for all ML components in Nautilus Trader, ensuring type-safe model deployment, feature schema compatibility, strategy requirement validation, and data lineage tracking through a unified manifest-based architecture.

All registries now implement the Universal ML Component Protocol (`ml/common/protocols.py`) to standardize health reporting, performance metrics, and configuration validation across domains. Protocol compliance is verified by the Integration Manager at startup (warn by default; strict via `ML_STRICT_PROTOCOL_VALIDATION`).

### Migrations & DB Functions

The registry relies on PostgreSQL-side functions for event emission and watermarks. Ensure migrations in `ml/stores/migrations/004_data_registry.sql` are applied in every environment so that:

- `emit_data_event(...)` exists and records dataset events
- `update_watermark(...)` exists and maintains freshness state

You can validate presence via the DB preflight utility:

```python
from ml.stores.db_preflight import check_db_prereqs
print(check_db_prereqs("$DB_CONNECTION"))  # ok=True indicates functions/partitions are present

### Protocol-based Consumers

- Callers of the registry in ML (stores, DataStore, scheduler) are typed against a `RegistryProtocol` to enforce interface drift at compile time.
- This keeps emit/update usage consistent across JSON and Postgres backends while allowing the concrete `DataRegistry` to evolve behind the Protocol.
```

### Key Architectural Principles

1. **Self-Describing Manifests**: All components carry complete metadata for autonomous validation
2. **Schema-Based Compatibility**: Hash-based feature schema validation prevents deployment errors
3. **Multi-Backend Persistence**: Configurable JSON/PostgreSQL backends for dev/prod environments
4. **Hot/Cold Path Separation**: Models marked as serveable/non-serveable for different use cases
5. **Statistical Validation**: Built-in A/B testing and canary deployment capabilities
6. **Lineage Tracking**: Parent-child relationships for teacher-student model hierarchies
7. **Data Registry Integration**: Complete dataset lifecycle management with watermarks and events

## Module Structure

```
ml/registry/
├── __init__.py              # Public API exports (66 lines)
├── base.py                  # Abstract interfaces and core types (405 lines)
├── dataclasses.py           # Quality gates, deployment, and data structures (879 lines)
├── data_registry.py         # Data registry with lineage and watermarks (1,181 lines)
├── feature_registry.py      # Feature set management (586 lines)
├── model_registry.py        # Model lifecycle management (1,967 lines)
├── strategy_registry.py     # Trading strategy management (749 lines)
├── persistence.py           # Multi-backend persistence layer (365 lines)
├── statistics.py            # Statistical validation utilities (220 lines)
├── utils.py                 # Helper functions (120 lines)
├── bootstrap_datasets.py    # Dataset manifest bootstrapping (30+ lines)
├── migrations/              # SQL migration scripts
└── LOADING_GUIDE.md        # Comprehensive usage documentation (531 lines)
```

## Core Registry Types

### 1. ModelRegistry (`model_registry.py`)

The central model lifecycle management system with comprehensive deployment tracking.

#### Manifest Structure (`ModelManifest`)

```python
@dataclass
class ModelManifest:
    model_id: str                           # Unique identifier
    role: ModelRole                         # TEACHER/STUDENT/INFERENCE/ENSEMBLE
    data_requirements: DataRequirements     # L1_ONLY/L1_L2/L1_L2_L3/HISTORICAL
    architecture: str                       # "XGBoost"/"LightGBM"/"TFT"
    feature_schema: dict[str, str]         # {"close": "float32", "volume": "float32"}
    feature_schema_hash: str               # SHA256 hash for compatibility
    parent_id: str | None                  # Teacher model for students
    children_ids: list[str]                # Student models for teachers
    training_config: dict[str, Any]        # Hyperparameters and training setup
    performance_metrics: dict[str, float]  # {"accuracy": 0.85, "latency_ms": 1.2}
    deployment_constraints: dict[str, Any] # {"max_latency_ms": 5}
    version: str                           # Semantic version
    # Serving configuration
    serveable: bool                        # True for hot-path, False for cold-path
    artifact_format: str                   # "onnx"/"torchscript"/"none"
    # Feature registry linkage
    feature_set_id: str | None            # Linked feature set ID
    pipeline_signature: str | None        # Pipeline hash for validation
    pipeline_version: str | None          # Pipeline version
```

#### Key Features

**Security & Validation**:

- Path traversal protection with `_validate_model_path()`
- ONNX-only loading for serveable models to prevent code execution
- Feature registry linkage for production models (optional with direct model loading fallback)
- Schema hash validation between models and features

**Performance Optimization**:

- LRU cache for loaded models (`cache_size` parameter)
- Batch save operations with configurable intervals
- Thread-safe operations with RLock

**Deployment Management**:

- Five deployment states: INACTIVE/ACTIVE/TESTING/RETIRED/FAILED
- A/B testing with traffic splitting and statistical validation
- Canary deployments with automated promotion/rollback
- Hot reload capabilities for zero-downtime updates

### 2. FeatureRegistry (`feature_registry.py`)

Feature set management with schema validation and parity tracking.

#### Manifest Structure (`FeatureManifest`)

```python
@dataclass
class FeatureManifest:
    feature_set_id: str                   # Unique identifier
    name: str                             # Human-readable name
    version: str                          # Semantic version
    role: FeatureRole                     # TEACHER/STUDENT/INFERENCE_SUPPORT
    data_requirements: DataRequirements   # L1_ONLY/L1_L2/L1_L2_L3
    feature_names: list[str]              # ["close_ratio", "volume_ma", "rsi"]
    feature_dtypes: list[str]             # ["float32", "float32", "float32"]
    schema_hash: str                      # SHA256 hash of names+dtypes+pipeline
    pipeline_signature: str               # Transform graph hash
    pipeline_version: str                 # Pipeline engine version
    capability_flags: dict[str, bool]     # {"handles_nans": True, "stateful": True}
    constraints: dict[str, Any]           # {"max_latency_ms": 0.5, "min_bars_warmup": 20}
    parity_tolerance: float               # Validation tolerance (1e-10)
    parity_digest: dict[str, Any]         # Validation results summary
    perf_digest: dict[str, Any]           # Performance metrics
    parent_feature_set_id: str | None     # Parent for lineage
    stage: FeatureStage                   # CANDIDATE/STAGING/PROD/DEPRECATED/SCRAPPED
```

#### Schema Hashing Algorithm

```python
def compute_schema_hash(
    feature_names: list[str],
    feature_dtypes: list[str],
    pipeline_signature: str,
) -> str:
    """Stable hash including names, types, and pipeline transform graph."""
    h = hashlib.sha256()
    for n, t in zip(feature_names, feature_dtypes):
        h.update(n.encode("utf-8"))
        h.update(b"::")
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    h.update(b"|sig|")
    h.update(pipeline_signature.encode("utf-8"))
    return h.hexdigest()
```

#### Quality Gates System
Feature promotion through lifecycle stages via `validate_and_promote()`:

- **Gates**: Metric thresholds with comparison operators (gte/lte/gt/lt/eq)
- **Sources**: Checks `perf_digest`, `parity_digest`, `constraints` in order
- **Validation**: Required gates must pass for promotion to PROD stage

### 3. DataRegistry (`data_registry.py`)

Complete dataset lifecycle management with lineage tracking, watermarks, and event recording.

#### Manifest Structure (`DatasetManifest`)

```python
@dataclass(frozen=True)
class DatasetManifest:
    # Identity
    dataset_id: str
    dataset_type: DatasetType                 # BARS/TRADES/QUOTES/MBP1/TBBO/FEATURES/PREDICTIONS/SIGNALS

    # Storage
    storage_kind: StorageKind                 # PARQUET/POSTGRES
    location: str                             # File path or table name
    partitioning: dict[str, Any]             # {"by": ["date", "instrument_id"]}
    retention_days: int                       # Data retention period

    # Schema
    schema: dict[str, str]                   # Column names and data types
    ts_field: str                             # Timestamp field name (in nanoseconds)
    seq_field: str | None                     # Optional sequence number field
    primary_keys: list[str]                   # Primary key columns
    schema_hash: str                          # SHA256 hash for validation

    # Validation
    constraints: dict[str, Any]               # Ranges, nullability, etc.

    # Lineage
    lineage: list[str]                        # Parent dataset IDs
    pipeline_signature: str                   # Pipeline that created this dataset

    # Versioning
    version: str                              # Semantic version
    created_at: int                           # Creation timestamp (nanoseconds)
    last_modified: int                        # Last modification timestamp (nanoseconds)

    # Metadata
    metadata: dict[str, Any]                  # Additional metadata
```

#### Data Contract System (`DataContract`)

```python
@dataclass(frozen=True)
class DataContract:
    contract_id: str
    dataset_id: str
    version: str
    validation_rules: list[ValidationRule]     # Type checks, range validation, etc.
    quality_thresholds: dict[str, float]      # {"null_rate": 0.01, "duplicate_rate": 0.0}
    enforcement_mode: str                      # "strict", "lenient", or "monitor_only"
    created_at: int
    last_modified: int
    metadata: dict[str, Any]
```

#### Watermark Tracking (`Watermark`)

```python
@dataclass(frozen=True)
class Watermark:
    dataset_id: str
    instrument_id: str
    source: str                               # "live", "historical", "backfill"
    last_success_ns: int                      # Last successful processing timestamp
    last_attempt_ns: int                      # Last attempted processing timestamp
    last_count: int                           # Count from last successful processing
    completeness_pct: float                   # Percentage of expected data received (0-100)
    updated_at: float                         # Unix timestamp of last update
```

#### Event Recording

The DataRegistry tracks processing events through the data pipeline:

```python
def emit_event(
    dataset_id: str,
    instrument_id: str,
    stage: str,                               # INGESTED/CATALOG_WRITTEN/FEATURE_COMPUTED
    source: str,                              # live/historical/backfill
    run_id: str,
    ts_min: int,
    ts_max: int,
    count: int,
    status: str,                              # success/failed/partial
    error: str | None = None
) -> None
```

#### Lineage Tracking

```python
def link_lineage(
    child_dataset_id: str,
    parent_ids: list[str],
    transform_id: str,
    ts_range: dict[str, int],
    params: dict[str, Any]
) -> None
```

### 4. StrategyRegistry (`strategy_registry.py`)

Trading strategy management with market regime compatibility and dependency validation.

#### Manifest Structure (`StrategyManifest`)

```python
@dataclass
class StrategyManifest:
    # Identity
    strategy_id: str
    strategy_type: StrategyType           # TREND_FOLLOWING/MEAN_REVERSION/MOMENTUM
    version: str

    # Requirements
    required_models: list[str] | None     # Model IDs needed
    required_features: list[str]          # Feature set IDs needed

    # Market conditions
    suitable_regimes: list[MarketRegime]  # TRENDING_UP/TRENDING_DOWN/RANGING/VOLATILE
    instrument_types: list[str]           # ["FX", "CRYPTO", "EQUITY"]
    timeframe_range: tuple[str, str]      # ("1m", "1h")

    # Risk parameters
    max_position_size: float
    max_leverage: float
    max_drawdown: float
    stop_loss_type: str

    # Performance constraints
    min_sharpe_ratio: float
    min_win_rate: float
    max_correlation_with_portfolio: float

    # Dependencies
    parent_strategy_id: str | None
    incompatible_strategies: list[str]    # Mutual exclusion list

    # Configuration
    config_schema: dict[str, str]         # Type hints for parameters
    default_config: dict[str, Any]        # Default parameter values

    # Performance tracking
    backtest_metrics: dict[str, float]
    live_metrics: dict[str, float] | None
```

#### Compatibility Validation

```python
def validate_requirements(
    strategy_id: str,
    available_models: list[str],
    available_features: list[str],
) -> bool:
    """Validate all model and feature dependencies are satisfied."""

def check_compatibility(
    strategy_id: str,
    active_strategies: list[str],
) -> bool:
    """Check mutual exclusion constraints with active strategies."""
```

## Persistence Architecture

### Multi-Backend Support (`persistence.py`)

#### JSON Backend (Development)

- **Storage**: Local filesystem with JSON serialization
- **Schema**: Manual JSON structure maintenance
- **Transactions**: File-level atomicity
- **Audit**: JSONL append log
- **Performance**: Fast for small datasets, limited scalability

#### PostgreSQL Backend (Production)

- **Storage**: Relational database with SQLAlchemy ORM
- **Schema**: Automatic table creation and migrations
- **Transactions**: ACID compliance with session management
- **Audit**: Structured audit log table
- **Performance**: Scalable with indexing and connection pooling

#### Database Schema Design

```sql
-- Core tables with JSON columns for flexible metadata
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    feature_schema JSON NOT NULL,
    feature_schema_hash VARCHAR(64) NOT NULL,
    performance_metrics JSON,
    deployment_status VARCHAR(50) NOT NULL,
    -- Indexing on frequently queried fields
    INDEX idx_models_role (role),
    INDEX idx_models_status (deployment_status)
);

CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    feature_names TEXT[],
    feature_dtypes TEXT[],
    -- Performance indexes
    INDEX idx_features_role (role),
    INDEX idx_features_stage (stage)
);

CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    required_models TEXT[],
    required_features TEXT[],
    suitable_regimes TEXT[],
    instrument_types TEXT[],
    -- Query optimization indexes
    INDEX idx_strategies_type (strategy_type)
);
```

### Configuration System (`PersistenceConfig`)

```python
class PersistenceConfig:
    backend: BackendType = BackendType.JSON
    connection_string: str | None = None           # PostgreSQL URL
    json_path: Path | None = None                  # JSON storage directory
    pool_size: int = 5                             # DB connection pool
    max_overflow: int = 10                         # Pool overflow limit
    echo: bool = False                             # SQL debugging
```

## Compatibility & Validation Framework

### Feature-Model Compatibility Matrix

The registry enforces strict compatibility through schema hashing:

```python
# Registration validation
if getattr(manifest, "serveable", True):
    if not manifest.feature_set_id:
        raise ValueError("feature_set_id required for serveable models")

    finfo = feature_registry.get_feature_set(manifest.feature_set_id)
    if finfo.manifest.schema_hash != manifest.feature_schema_hash:
        raise ValueError("Feature schema hash mismatch")
```

### Quality Gate Validation (`dataclasses.py`)

The dataclasses module contains structures for both model quality validation and data registry management:

#### Data Registry Types

```python
class DatasetType(Enum):
    """Types of datasets tracked in the data registry."""
    BARS = "bars"           # OHLCV bar data
    TRADES = "trades"       # Individual trade ticks
    QUOTES = "quotes"       # Bid/ask quote ticks
    MBP1 = "mbp1"          # Market by price depth 1
    TBBO = "tbbo"          # Top of book best bid/offer
    FEATURES = "features"   # Computed feature values
    PREDICTIONS = "predictions"  # Model predictions
    SIGNALS = "signals"     # Strategy signals

class StorageKind(Enum):
    """Storage backend types for datasets."""
    PARQUET = "parquet"     # Apache Parquet file storage
    POSTGRES = "postgres"   # PostgreSQL database storage

class ValidationRuleType(Enum):
    """Types of validation rules for data contracts."""
    TYPE_CHECK = "type_check"
    RANGE = "range"
    UNIQUENESS = "uniqueness"
    MONOTONICITY = "monotonicity"
    NULLABILITY = "nullability"
    LATENESS = "lateness"

class QualityFlag(Enum):
    """Quality flags for data validation results."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
```

#### Data Validation Rules

```python
@dataclass(frozen=True)
class ValidationRule:
    rule_type: ValidationRuleType
    field_name: str              # Field to validate (or "*" for all)
    parameters: dict[str, Any]   # Rule-specific parameters
    severity: QualityFlag        # WARN or FAIL
    description: str
```

#### Model Quality Gates

```python
@dataclass
class QualityGate:
    metric_name: str                    # Metric to check
    threshold: float                    # Required value
    comparison: str = "gte"             # Comparison operator
    required: bool = True               # Must pass for validation
```

#### ValidationResult Tracking

```python
@dataclass
class ValidationResult:
    model_id: str
    overall_pass: bool = True           # All required gates passed
    gates_passed: int = 0
    gates_failed: int = 0
    gate_results: dict[str, dict[str, Any]]  # Detailed results per gate
```

### Canary Deployment System

#### Automated Decision Making

```python
def should_promote(self) -> tuple[bool, str]:
    """Evaluate canary promotion based on metrics and duration."""
    if self.metrics["sample_count"] < self.config.min_samples:
        return False, "insufficient_samples"

    error_rate = self.metrics["error_count"] / self.metrics["sample_count"]
    if error_rate > self.config.error_rate_threshold:
        return False, "high_error_rate"

    current_performance = self.metrics["metric_sum"] / self.metrics["success_count"]
    if self.baseline_performance:
        relative_performance = current_performance / self.baseline_performance
        if relative_performance < self.config.baseline_threshold:
            return False, "performance_below_baseline"
```

#### Statistical Validation (`statistics.py`)

```python
def welch_t_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Welch's t-test for unequal variances with p-value approximation."""
```

## Integration with Core Systems

### Store Integration (Three-Store Pattern)

The registry integrates with the store triad (mandatory in production, optional in testing with fallbacks):

1. **FeatureStore**: Links via `feature_set_id` in model manifests
2. **ModelStore**: Tracks predictions and performance metrics
3. **StrategyStore**: References models/features via strategy manifest requirements

### Actor Integration

#### BaseMLInferenceActor Requirements

```python
# Registry validation during actor initialization
def _validate_model_manifest(self, manifest: ModelManifest) -> None:
    if manifest.role not in [ModelRole.STUDENT, ModelRole.INFERENCE]:
        raise ValueError("Only STUDENT/INFERENCE models allowed in hot path")

    if manifest.data_requirements not in [DataRequirements.L1_ONLY]:
        raise ValueError("Only L1_ONLY models allowed for real-time inference")
```

#### MLSignalActor Integration

- **Model Loading**: Registry provides ONNX sessions for inference
- **Feature Validation**: Schema hash matching with FeatureRegistry
- **Performance Tracking**: Automatic latency and accuracy metrics

### Deployment Patterns

#### Auto-Deployment Logic

```python
if auto_deploy and is_valid:
    target = {
        ModelRole.STUDENT: "ml_signal_actor",
        ModelRole.INFERENCE: "ml_signal_actor",
        ModelRole.TEACHER: None,  # Cold-path only
    }.get(manifest.role)

    if target:
        self.deploy_model(manifest.model_id, target)
```

#### Hot Reload Capabilities

```python
def hot_reload_model(self, target: str, new_model_id: str) -> bool:
    """Zero-downtime model replacement with validation."""
    current_model = self._get_active_model_for_target(target)
    new_model = self._models[new_model_id]

    # Validate feature compatibility
    if current_model.manifest.feature_schema_hash != new_model.manifest.feature_schema_hash:
        logger.warning("Feature schema mismatch during hot reload")

    # Atomic deployment
    self.deploy_model(new_model_id, target)
    self.retire_model(current_model.manifest.model_id)
```

## Dataset Bootstrap System

The `bootstrap_datasets.py` module provides pre-registration of standard dataset manifests to ensure consistent naming and avoid orphaned events:

```python
# Bootstrap standard datasets
python -m ml.registry.bootstrap_datasets --backend json --registry-path /tmp/registry

# Creates manifests for:
# - BARS: OHLCV bar data
# - TRADES: Individual trade ticks
# - QUOTES: Bid/ask quotes
# - MBP1: Market by price depth 1
# - TBBO: Top of book data
# - FEATURES: Computed features
# - PREDICTIONS: Model predictions
# - SIGNALS: Strategy signals
```

Each bootstrapped dataset includes:

- Proper Nautilus schema (instrument_id, ts_event, ts_init)
- Validation contracts with range and nullability rules
- Partitioning strategies
- Retention policies

## Current Implementation Status

### Completed Features ✅

1. **Core Registry Framework**
   - Self-describing manifests for all component types
   - Multi-backend persistence (JSON/PostgreSQL)
   - Thread-safe operations with proper locking
   - Comprehensive audit logging

2. **Model Lifecycle Management**
   - Registration with validation
   - Deployment state tracking
   - Hot reload and rollback capabilities
   - Parent-child relationship tracking

3. **Feature Schema System**
   - Hash-based compatibility validation
   - Quality gate promotion system
   - Parity tolerance tracking
   - Pipeline signature validation

4. **Data Registry System**
   - Dataset manifest management
   - Data contract validation with rules
   - Watermark tracking for processing progress
   - Event recording for pipeline monitoring
   - Lineage tracking for data provenance
   - Bootstrap script for standard datasets

5. **Statistical Validation**
   - A/B testing framework
   - Canary deployment automation
   - Welch's t-test implementation
   - Sample size calculation

6. **Security & Safety**
   - Path traversal protection
   - ONNX-only serving for safety
   - Feature registry linkage (recommended for production)
   - Schema hash enforcement

### Integration Points 🔄

1. **Store System**: Manifests reference store-persisted data
2. **Actor System**: Registry provides models for hot-path inference
3. **Feature Engineering**: Schema hash validation with feature pipeline
4. **Training System**: Model registration post-training

### Production Readiness Assessment

#### Strengths 💪

- **Type Safety**: Comprehensive manifest system prevents runtime errors
- **Performance**: LRU caching, batch operations, optimized queries
- **Flexibility**: Multi-backend support for different environments
- **Validation**: Multiple layers of compatibility checking
- **Observability**: Detailed audit logging and performance tracking

#### Areas for Enhancement 🚧

- **Migration System**: Automated schema migration between versions
- **Distributed Deployment**: Multi-node registry synchronization
- **Advanced Analytics**: Model drift detection and automated retraining triggers
- **UI Integration**: Management dashboard for non-technical users

## Critical Design Decisions

### 1. Manifest-Centric Architecture
**Decision**: All components carry self-describing manifests
**Rationale**: Enables autonomous validation and reduces coupling
**Trade-offs**: Larger storage footprint, but eliminates runtime configuration errors

### 2. Schema Hash Validation
**Decision**: SHA256 hash of feature names + types + pipeline signature
**Rationale**: Prevents feature-model mismatches in production
**Trade-offs**: Strict coupling, but guarantees compatibility

### 3. Hot/Cold Path Separation
**Decision**: `serveable` flag controls deployment eligibility
**Rationale**: Teacher models stay in cold path, students serve hot path
**Trade-offs**: Complexity in model hierarchy, but enables different optimization strategies

### 4. Multi-Backend Persistence
**Decision**: Configurable JSON/PostgreSQL backends
**Rationale**: Development convenience vs production scalability
**Trade-offs**: Added complexity, but supports different deployment contexts

### 5. Statistical Validation Integration
**Decision**: Built-in A/B testing and canary deployment
**Rationale**: Data-driven model promotion reduces manual intervention
**Trade-offs**: Added complexity, but enables automated ML operations

## Usage Patterns & Best Practices

### Data Registry Usage

```python
# 1. Register a dataset
manifest = DatasetManifest(
    dataset_id="bars_eurusd_1m",
    dataset_type=DatasetType.BARS,
    storage_kind=StorageKind.PARQUET,
    location="/data/bars/eurusd/1m/",
    partitioning={"by": ["date", "instrument_id"]},
    retention_days=365,
    schema={
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64"
    },
    ts_field="ts_event",
    primary_keys=["instrument_id", "ts_event"],
    schema_hash="abc123...",
    constraints={
        "ranges": {
            "open": {"min": 0.0},
            "high": {"min": 0.0},
            "low": {"min": 0.0},
            "close": {"min": 0.0},
            "volume": {"min": 0.0}
        }
    },
    lineage=[],
    pipeline_signature="data_scheduler_v1",
    version="1.0.0"
)

data_registry = DataRegistry(
    registry_path=Path("/tmp/registry"),
    persistence_config=PersistenceConfig(backend=BackendType.JSON)
)
dataset_id = data_registry.register_dataset(manifest)

# 2. Emit processing events
data_registry.emit_event(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    stage="CATALOG_WRITTEN",
    source="historical",
    run_id="run_123",
    ts_min=1234567890000000000,
    ts_max=1234567900000000000,
    count=1000,
    status="success"
)

# 3. Update watermark
data_registry.update_watermark(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    source="live",
    last_success_ns=1234567900000000000,
    count=1000,
    completeness_pct=98.5
)

# 4. Link lineage for derived datasets
data_registry.link_lineage(
    child_dataset_id="features_microstructure",
    parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
    transform_id="feature_pipeline_v1",
    ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
    params={"lookback_bars": 20, "include_imbalance": True}
)
```

### Model Registration Flow

```python
# 1. Create manifest
manifest = ModelManifest(
    model_id="lgb_student_v1",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    feature_schema_hash="abc123...",
    parent_id="tft_teacher_v1",
    feature_set_id="student_features_v1"
)

# 2. Register with validation
registry.register_model(
    model_path=Path("models/lgb_student_v1.onnx"),
    manifest=manifest,
    auto_deploy=True,  # Deploy if validation passes
    quality_gates=[
        QualityGate("accuracy", 0.80, "gte"),
        QualityGate("inference_latency_ms", 5.0, "lte")
    ]
)
```

### Feature Set Registration

```python
# 1. Compute schema hash
schema_hash = compute_schema_hash(
    feature_names=["close_ratio", "volume_ma"],
    feature_dtypes=["float32", "float32"],
    pipeline_signature="pipeline_v1_hash"
)

# 2. Create manifest
manifest = FeatureManifest(
    feature_set_id="student_features_v1",
    role=FeatureRole.STUDENT,
    feature_names=["close_ratio", "volume_ma"],
    feature_dtypes=["float32", "float32"],
    schema_hash=schema_hash,
    stage=FeatureStage.CANDIDATE
)

# 3. Register and promote
feature_registry.register_feature_set(manifest)
feature_registry.validate_and_promote(
    "student_features_v1",
    [QualityGate("parity_max_diff", 1e-10, "lte")]
)
```

### Deployment Query Patterns

```python
# Find compatible models
compatible = model_registry.list_compatible(
    schema_hash="abc123...",
    role=ModelRole.STUDENT,
    architecture="LightGBM"
)

# Get latest version
latest = model_registry.resolve_latest(
    role=ModelRole.STUDENT,
    architecture="LightGBM",
    schema_hash="abc123..."
)

# Load for inference
session = model_registry.load_model("lgb_student_v1")
```

## Conclusion

The ML Registry system provides a robust, production-ready foundation for ML lifecycle management in Nautilus Trader. Its comprehensive architecture now includes:

1. **ModelRegistry**: Complete model lifecycle management with A/B testing and canary deployments
2. **FeatureRegistry**: Feature schema validation and parity tracking
3. **DataRegistry**: Dataset lifecycle with watermarks, events, and lineage tracking
4. **StrategyRegistry**: Trading strategy management with compatibility checking

The manifest-centric architecture, multi-backend persistence, and comprehensive validation framework enable safe, automated deployment of ML components while maintaining strict compatibility guarantees. The system successfully bridges the gap between research experimentation and production deployment through its hot/cold path separation, statistical validation capabilities, and complete data lineage tracking.
## Cross-Module References

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Feature Engineering**: See `context_features.md` for feature computation
- **Stores**: See `context_stores.md` for persistence layer
- **Training**: See `context_training.md` for model training pipelines
- **Registry**: See `context_registry.md` for lifecycle management
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Deployment**: See `context_deployment.md` for containerization
- **Monitoring**: See `context_monitoring.md` for observability
- **Actors**: See `context_actors.md` for inference actors
- **Models**: See `context_models.md` for model implementations
