# Registry System Context Document

**Last Updated:** 2025-10-19
**Total Lines:** 16,582 Python + SQL migrations
**Status:** Production-ready with mature strangler fig decomposition

---

## Executive Summary

The `ml/registry/` directory implements a **production-ready, 4-registry architecture** using the **strangler fig pattern** to decompose legacy god classes into specialized components. The system provides manifest-based lifecycle management for models, features, strategies, and datasets with multi-backend persistence (PostgreSQL/JSON), thread-safe operations, and protocol-first design.

**Core Architecture:**
- **4 Mandatory Registries**: ModelRegistry, FeatureRegistry, StrategyRegistry, DataRegistry
- **Strangler Fig Decomposition**: Legacy monoliths replaced by component-based facades
- **Feature Flags**: `ML_USE_LEGACY_MODEL_REGISTRY`, `ML_USE_LEGACY_DATA_REGISTRY` for safe rollback
- **15+ Components**: Specialized managers for persistence, deployment, lineage, events, watermarks, A/B testing
- **Multi-Backend**: JSON (development) and PostgreSQL (production) with graceful fallback

**Status Assessment:**
- ✅ Core registry functionality: **100% complete**
- ✅ Strangler fig decomposition: **Mature (Phase 2.3 complete)**
- ✅ Protocol-first design: **100% compliant (Pattern 2)**
- ⚠️ Universal Pattern compliance: **3/5 patterns fully implemented**
- ❌ Centralized metrics: **Missing (Pattern 5 non-compliant)**

---

## Directory Structure

```
ml/registry/                           # 29 files, 16,582 lines total
├── __init__.py                        # Public API (187 lines)
├── protocols.py                       # Protocol definitions (108 lines)
├── base.py                            # Model manifest + enums (504 lines)
├── dataclasses.py                     # Data structures (929 lines)
│
├── 4 MANDATORY REGISTRIES (Pattern 1)
├── model_registry.py                  # Facade (1,191 lines) → 5 components
├── feature_registry.py                # Self-contained (692 lines)
├── strategy_registry.py               # Self-contained (822 lines)
├── data_registry.py                   # Facade (928 lines) → 5 components
│
├── LEGACY IMPLEMENTATIONS (Strangler Fig)
├── model_registry_legacy.py           # Original god class (2,284 lines)
├── data_registry_legacy.py            # Original god class (1,896 lines)
│
├── SUPPORTING INFRASTRUCTURE
├── abstract_registry.py               # Common base (113 lines)
├── persistence.py                     # Multi-backend (382 lines)
├── statistics.py                      # Statistical validation (219 lines)
├── utils.py                           # Helpers (174 lines)
├── mixins.py                          # Optional mixins (99 lines)
├── summaries.py                       # Summary builders (98 lines)
├── bootstrap_datasets.py              # Dataset initialization (654 lines)
├── artifacts.py                       # Artifact management (123 lines)
├── _typing_utils.py                   # Type utilities (157 lines)
│
├── MODEL REGISTRY COMPONENTS (5 components)
├── model_persistence.py               # Loading/saving/caching (844 lines)
├── model_quality_validator.py         # Quality gates (167 lines)
├── model_deployment_mgr.py            # Deployment tracking (674 lines)
├── ab_testing_manager.py              # A/B testing (452 lines)
├── canary_deployment_mgr.py           # Canary releases (483 lines)
│
├── DATA REGISTRY COMPONENTS (5 components)
├── manifest_manager.py                # CRUD operations (710 lines)
├── lineage_manager.py                 # Lineage tracking (394 lines)
├── watermark_manager.py               # Watermark state (634 lines)
├── event_manager.py                   # Event emission (332 lines)
├── contract_manager.py                # Contract validation (332 lines)
│
└── migrations/                        # SQL schema
    ├── 001_initial_schema.sql         # Tables + indexes (276 lines)
    ├── 002_add_cold_path_fields.sql   # Hot/cold separation (17 lines)
    └── 003_add_artifact_digest.sql    # SHA-256 integrity (5 lines)
```

---

## 1. Four Mandatory Registries (Pattern 1)

### 1.1 ModelRegistry (`model_registry.py` - 1,191 lines)

**Strangler Fig Facade**: Delegates to 5 specialized components while maintaining 100% backward compatibility.

**Component Architecture:**

```python
class ModelRegistry(AbstractRegistry):
    """
    Model lifecycle facade with component delegation.

    Feature Flag Control:
    - ML_USE_LEGACY_MODEL_REGISTRY=1: Original monolith (2,284 lines)
    - ML_USE_LEGACY_MODEL_REGISTRY=0: Component-based (default)
    - ML_USE_COMPONENT_MODEL_REGISTRY=1: Explicit opt-in
    """

    def __init__(self, registry_path: Path, persistence_config: PersistenceConfig):
        if USE_LEGACY:
            # Delegate to model_registry_legacy.py (god class)
            self._impl = ModelRegistryLegacy(...)
        else:
            # Initialize 5 components
            self._persistence = ModelPersistence(...)          # Artifact I/O
            self._quality = ModelQualityValidator(...)          # Quality gates
            self._deployment = ModelDeploymentManager(...)      # Deployment
            self._ab_testing = ABTestingManager(...)            # A/B tests
            self._canary = CanaryDeploymentManager(...)         # Canary releases
```

**Five Component Managers:**

| Component | Lines | Responsibility | File |
|-----------|-------|----------------|------|
| **ModelPersistence** | 844 | Model loading, saving, caching (LRU), SHA-256 integrity, ONNX sessions | `model_persistence.py` |
| **ModelQualityValidator** | 167 | Quality gate evaluation, validation results, metric thresholds | `model_quality_validator.py` |
| **ModelDeploymentManager** | 674 | Deployment tracking, version management, rollback, hot reload | `model_deployment_mgr.py` |
| **ABTestingManager** | 452 | A/B test configuration, traffic splitting, statistical analysis | `ab_testing_manager.py` |
| **CanaryDeploymentManager** | 483 | Canary release management, gradual rollout, promotion/rollback | `canary_deployment_mgr.py` |

**Manifest Structure:**

```python
@dataclass
class ModelManifest:
    # Identity
    model_id: str
    role: ModelRole  # TEACHER/STUDENT/INFERENCE/ENSEMBLE/FEATURE
    data_requirements: DataRequirements  # L1_ONLY/L1_L2/L1_L2_L3
    architecture: str  # "XGBoost", "LightGBM", "TFT"

    # Schema validation
    feature_schema: dict[str, str]  # {"close": "float32", "volume": "float32"}
    feature_schema_hash: str  # SHA256 for compatibility

    # Lineage
    parent_id: str | None  # Teacher model for students
    children_ids: list[str]  # Student models

    # Configuration
    training_config: dict[str, Any]
    performance_metrics: dict[str, float]
    deployment_constraints: dict[str, Any]

    # Versioning
    version: str  # Semantic versioning
    created_at: float
    last_modified: float

    # Hot/Cold path separation (Pattern 3)
    serveable: bool = True  # True=hot path, False=cold path
    artifact_format: str = "onnx"  # onnx/torchscript/none

    # Feature registry linkage
    feature_set_id: str | None
    pipeline_signature: str | None
    pipeline_version: str | None

    # Decision policy (OCP extension point)
    decision_policy: str | None  # Fully-qualified import path
    decision_config: dict[str, Any]

    # Security (SHA-256 integrity verification)
    artifact_sha256_digest: str | None
```

**Key Operations:**

```python
# Registration with quality gates
registry.register_model(
    model_path=Path("models/lgb_student_v1.onnx"),
    manifest=ModelManifest(...),
    auto_deploy=True,
    quality_gates=[
        QualityGate("accuracy", 0.80, "gte"),
        QualityGate("inference_latency_ms", 5.0, "lte")
    ]
)

# Deployment management
registry.deploy_model("lgb_student_v1", target="ml_signal_actor")
registry.hot_reload_model(target="ml_signal_actor", new_model_id="lgb_student_v2")
registry.rollback_deployment(target="ml_signal_actor")

# A/B testing
registry.start_ab_test(
    test_id="model_comparison_v1_v2",
    model_a="lgb_student_v1",
    model_b="lgb_student_v2",
    traffic_split=0.5
)

# Canary deployment
registry.start_canary_deployment(
    model_id="lgb_student_v2",
    target="ml_signal_actor",
    config=CanaryConfig(
        initial_traffic=0.05,
        increment=0.05,
        interval_seconds=300
    )
)
```

**Thread Safety:**
- RLock for all operations
- LRU cache for loaded models (configurable `cache_size`)
- Batch save operations (default 0.1s interval)

---

### 1.2 DataRegistry (`data_registry.py` - 928 lines)

**Strangler Fig Facade**: Delegates to 5 specialized components for dataset lifecycle management.

**Component Architecture:**

```python
class DataRegistry(MLComponentMixin):
    """
    Dataset lifecycle facade with component delegation.

    Feature Flag Control:
    - ML_USE_LEGACY_DATA_REGISTRY=1: Original monolith (1,896 lines)
    - ML_USE_LEGACY_DATA_REGISTRY=0: Component-based (default)
    """

    def __init__(self, registry_path: Path, persistence_config: PersistenceConfig):
        # Initialize 5 component managers
        self._manifest_mgr = ManifestManager()        # CRUD operations
        self._lineage_mgr = LineageManager()          # Lineage tracking
        self._watermark_mgr = WatermarkManager()      # Watermark state
        self._event_mgr = EventManager()              # Event emission
        self._contract_mgr = ContractManager()        # Contract validation
```

**Five Component Managers:**

| Component | Lines | Responsibility | File |
|-----------|-------|----------------|------|
| **ManifestManager** | 710 | Dataset manifest CRUD, schema validation, versioning | `manifest_manager.py` |
| **LineageManager** | 394 | Parent-child relationships, transform tracking, provenance | `lineage_manager.py` |
| **WatermarkManager** | 634 | Processing progress, completeness tracking, staleness detection | `watermark_manager.py` |
| **EventManager** | 332 | Event emission, correlation IDs, metadata attachment | `event_manager.py` |
| **ContractManager** | 332 | Data contract enforcement, quality thresholds, validation rules | `contract_manager.py` |

**Dataset Manifest:**

```python
@dataclass
class DatasetManifest:
    # Identity
    dataset_id: str
    dataset_type: DatasetType  # BARS/TRADES/QUOTES/FEATURES/PREDICTIONS/SIGNALS

    # Storage
    storage_kind: StorageKind  # PARQUET/POSTGRES
    location: str  # File path or table name
    partitioning: dict[str, Any] | None  # {"by": ["date", "instrument_id"]}
    retention_days: int

    # Schema
    schema: dict[str, str]  # Column names → types
    ts_field: str  # Timestamp field (in nanoseconds)
    seq_field: str | None  # Sequence number field
    primary_keys: list[str]
    schema_hash: str  # Content hash for validation

    # Validation
    constraints: dict[str, Any] | None  # Ranges, nullability

    # Lineage
    lineage: list[str]  # Parent dataset IDs
    pipeline_signature: str  # Pipeline that created this

    # Versioning
    version: str
    created_at: float
    last_modified: float

    # Metadata
    metadata: dict[str, Any]
```

**Watermark Tracking:**

```python
@dataclass(frozen=True)
class Watermark:
    dataset_id: str
    instrument_id: str
    source: str  # "live", "historical", "backfill"
    last_success_ns: int  # Last successful processing timestamp
    last_attempt_ns: int  # Last attempted processing timestamp
    last_count: int  # Record count from last success
    completeness_pct: float  # 0-100
    updated_at: float  # Unix timestamp
```

**Event Recording:**

```python
from ml.config.events import Stage, Source, EventStatus

registry.emit_event(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    stage=Stage.CATALOG_WRITTEN,  # Enum-typed
    source=Source.HISTORICAL,      # Enum-typed
    run_id="run_123",
    ts_min=1234567890000000000,
    ts_max=1234567900000000000,
    count=1000,
    status=EventStatus.SUCCESS,    # Enum-typed
    metadata={"pipeline_version": "v2.1"}  # Optional context
)

# Correlation IDs: SHA256-based deterministic hashing
# Derived from: run_id + dataset_id + instrument_id + time_window
```

**Lineage Tracking:**

```python
registry.link_lineage(
    child_dataset_id="features_microstructure",
    parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
    transform_id="feature_pipeline_v1",
    ts_range={"start_ns": ..., "end_ns": ...},
    params={"lookback_bars": 20, "include_imbalance": True}
)
```

**Data Contract System:**

```python
@dataclass
class DataContract:
    contract_id: str
    dataset_id: str
    version: str
    validation_rules: list[ValidationRule]
    quality_thresholds: dict[str, float]  # {"null_rate": 0.01}
    enforcement_mode: str  # "strict", "lenient", "monitor_only"
    created_at: float
    last_modified: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ValidationRule:
    rule_type: ValidationRuleType  # TYPE_CHECK/RANGE/MONOTONICITY/...
    field_name: str
    parameters: dict[str, Any]
    severity: QualityFlag  # WARN/FAIL
    description: str
```

---

### 1.3 FeatureRegistry (`feature_registry.py` - 692 lines)

**Self-contained implementation** (no strangler fig decomposition needed - already well-factored).

**Feature Manifest:**

```python
@dataclass
class FeatureManifest:
    # Identity
    feature_set_id: str
    name: str
    version: str
    role: FeatureRole  # TEACHER/STUDENT/INFERENCE_SUPPORT
    data_requirements: DataRequirements  # L1_ONLY/L1_L2/L1_L2_L3

    # Schema
    feature_names: list[str]  # ["close_ratio", "volume_ma", "rsi"]
    feature_dtypes: list[str]  # ["float32", "float32", "float32"]
    schema_hash: str  # SHA256 of names+dtypes+pipeline

    # Pipeline
    pipeline_signature: str  # Transform graph hash
    pipeline_version: str  # Pipeline engine version

    # Capabilities
    capability_flags: dict[str, bool]  # {"handles_nans": True}
    constraints: dict[str, Any]  # {"max_latency_ms": 0.5}

    # Validation
    parity_tolerance: float  # Default 1e-10
    parity_digest: dict[str, Any]  # Validation results
    perf_digest: dict[str, Any]  # Performance metrics

    # Lineage
    parent_feature_set_id: str | None

    # Lifecycle
    stage: FeatureStage  # CANDIDATE/STAGING/PROD/DEPRECATED/SCRAPPED
    metadata: dict[str, Any]
    created_at: float
    last_modified: float
```

**Schema Hashing:**

```python
def compute_schema_hash(
    feature_names: list[str],
    feature_dtypes: list[str],
    pipeline_signature: str,
) -> str:
    """Stable SHA256 hash including names, types, and pipeline."""
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

**Quality Gates & Promotion:**

```python
# Register with quality gates
registry.register_feature_set(
    manifest=FeatureManifest(
        feature_set_id="student_features_v1",
        stage=FeatureStage.CANDIDATE,
        ...
    )
)

# Promote through lifecycle stages
registry.validate_and_promote(
    "student_features_v1",
    quality_gates=[
        QualityGate("parity_max_diff", 1e-10, "lte"),
        QualityGate("p99_latency_ms", 0.5, "lte")
    ]
)
# CANDIDATE → STAGING → PROD
```

---

### 1.4 StrategyRegistry (`strategy_registry.py` - 822 lines)

**Self-contained implementation** with compatibility checking and requirement validation.

**Strategy Manifest:**

```python
@dataclass
class StrategyManifest:
    # Identity
    strategy_id: str
    strategy_type: StrategyType  # TREND_FOLLOWING/MEAN_REVERSION/...
    version: str

    # Requirements
    required_models: list[str] | None
    required_features: list[str]

    # Market conditions
    suitable_regimes: list[MarketRegime]  # TRENDING_UP/RANGING/...
    instrument_types: list[str]  # ["FX", "CRYPTO", "EQUITY"]
    timeframe_range: tuple[str, str]  # ("1m", "1h")

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
    incompatible_strategies: list[str]  # Mutual exclusion

    # Configuration
    config_schema: dict[str, str]
    default_config: dict[str, Any]

    # Performance tracking
    backtest_metrics: dict[str, float]
    live_metrics: dict[str, float] | None

    # Metadata
    created_at: float
    last_modified: float
    author: str
    description: str
```

**Compatibility Validation:**

```python
# Validate dependencies
registry.validate_requirements(
    strategy_id="ensemble_v1",
    available_models=["lgb_student_v1", "tft_teacher_v1"],
    available_features=["student_features_v1"]
)

# Check mutual exclusion
registry.check_compatibility(
    strategy_id="trend_following_v1",
    active_strategies=["mean_reversion_v1"]
)
```

---

## 2. Strangler Fig Pattern Implementation

### 2.1 Migration Strategy

**Phase 2.3 Complete**: Both ModelRegistry and DataRegistry have been successfully decomposed.

**Feature Flag System:**

```python
# __init__.py - DataRegistry facade selection
USE_LEGACY_DATA_REGISTRY = os.getenv("ML_USE_LEGACY_DATA_REGISTRY", "0") == "1"

if TYPE_CHECKING:
    from ml.registry.data_registry import DataRegistry
    from ml.registry.watermark_manager import Watermark
else:
    if USE_LEGACY_DATA_REGISTRY:
        from ml.registry.data_registry_legacy import DataRegistryLegacy as DataRegistry
        from ml.registry.data_registry_legacy import Watermark
    else:
        from ml.registry.data_registry import DataRegistry
        from ml.registry.watermark_manager import Watermark
```

```python
# model_registry.py - Component vs legacy selection
def _should_use_component_impl() -> bool:
    """
    Precedence (highest to lowest):
    1. ML_USE_COMPONENT_MODEL_REGISTRY=1 (explicit opt-in)
    2. ML_USE_COMPONENT_MODEL_REGISTRY=0 (explicit opt-out)
    3. ML_USE_LEGACY_MODEL_REGISTRY=1 (legacy god class)
    4. ML_USE_LEGACY_MODEL_REGISTRY=0 (component-based)
    5. Default: component-based (USE_LEGACY=False)
    """
    component_flag = os.getenv("ML_USE_COMPONENT_MODEL_REGISTRY")
    if component_flag is not None:
        return component_flag.strip() == "1"

    legacy_flag = os.getenv("ML_USE_LEGACY_MODEL_REGISTRY")
    if legacy_flag is not None:
        return legacy_flag.strip() == "0"

    return True  # Default to component-based
```

**Rollback Capability:**

```bash
# Rollback to legacy implementations
export ML_USE_LEGACY_MODEL_REGISTRY=1
export ML_USE_LEGACY_DATA_REGISTRY=1

# Use new component-based implementations (default)
export ML_USE_LEGACY_MODEL_REGISTRY=0
export ML_USE_LEGACY_DATA_REGISTRY=0
```

---

### 2.2 Decomposition Status

| Registry | Status | Legacy Lines | New Lines | Components | Savings |
|----------|--------|--------------|-----------|------------|---------|
| **ModelRegistry** | ✅ Complete | 2,284 | 1,191 facade + 2,620 components | 5 | -47% facade |
| **DataRegistry** | ✅ Complete | 1,896 | 928 facade + 2,402 components | 5 | -51% facade |
| **FeatureRegistry** | ✅ No decomposition needed | 692 | 692 | 0 (self-contained) | N/A |
| **StrategyRegistry** | ✅ No decomposition needed | 822 | 822 | 0 (self-contained) | N/A |

**Component Extraction Metrics:**

```
ModelRegistry god class (2,284 lines) → Decomposed into:
├── model_registry.py (1,191 lines) - Facade
├── model_persistence.py (844 lines) - Loading/saving/caching
├── model_quality_validator.py (167 lines) - Quality gates
├── model_deployment_mgr.py (674 lines) - Deployment tracking
├── ab_testing_manager.py (452 lines) - A/B testing
└── canary_deployment_mgr.py (483 lines) - Canary releases
Total: 3,811 lines (67% increase for separation of concerns)

DataRegistry god class (1,896 lines) → Decomposed into:
├── data_registry.py (928 lines) - Facade
├── manifest_manager.py (710 lines) - CRUD operations
├── lineage_manager.py (394 lines) - Lineage tracking
├── watermark_manager.py (634 lines) - Watermark state
├── event_manager.py (332 lines) - Event emission
└── contract_manager.py (332 lines) - Contract validation
Total: 3,330 lines (76% increase for separation of concerns)
```

**Benefits Achieved:**

1. **Separation of Concerns**: Each component has single responsibility
2. **Testability**: Components can be tested in isolation
3. **Maintainability**: Easier to understand and modify
4. **Extensibility**: New components can be added without touching facade
5. **Backward Compatibility**: 100% API compatibility maintained
6. **Safe Rollback**: Feature flags enable instant rollback

---

## 3. Supporting Infrastructure

### 3.1 AbstractRegistry (`abstract_registry.py` - 113 lines)

**Common base** for Feature/Model/Strategy registries (not used by DataRegistry due to distinct semantics).

```python
class AbstractRegistry(MLComponentMixin, ABC):
    """
    Common base centralizing:
    - RLock lifecycle (thread-safety)
    - Dual-backend persistence via PersistenceManager
    - JSON save/load helpers
    - Audit logging passthrough
    - Health summary implementation
    """

    def __init__(self, persistence: PersistenceManager):
        self._lock = threading.RLock()
        self.persistence = persistence
        self.backend = persistence.config.backend

    def _json_load(self, filename: str) -> dict[str, Any] | None:
        """Load JSON via PersistenceManager (JSON backend only)."""
        if self.backend != BackendType.JSON:
            return None
        return self.persistence.load_json(filename)

    def _json_save(self, filename: str, data: dict[str, Any]) -> None:
        """Save JSON via PersistenceManager (JSON backend only)."""
        if self.backend != BackendType.JSON:
            return
        self.persistence.save_json(data, filename)

    def log_audit(self, *, entity_type: str, entity_id: str,
                  action: str, changes: dict[str, Any] | None = None,
                  user_id: str | None = None) -> None:
        """Passthrough to PersistenceManager.log_audit."""
        self.persistence.log_audit(...)

    @abstractmethod
    def _health_snapshot(self) -> tuple[int, float | None]:
        """Return (count, last_modified) for health reporting."""

    def get_health_status(self) -> dict[str, Any]:
        count, last_modified = self._health_snapshot()
        return {
            "component": self.__class__.__name__,
            "status": "ok",
            "backend": self.backend.value,
            "count": count,
            "last_modified": last_modified
        }
```

---

### 3.2 PersistenceManager (`persistence.py` - 382 lines)

**Multi-backend abstraction** for JSON and PostgreSQL persistence.

```python
@dataclass
class PersistenceConfig:
    backend: BackendType = BackendType.JSON
    connection_string: str | None = None  # PostgreSQL URL
    json_path: Path | None = None
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

class PersistenceManager:
    """
    Unified persistence layer supporting:
    - JSON: Local filesystem with atomic writes
    - PostgreSQL: ACID-compliant with connection pooling
    """

    def __init__(self, config: PersistenceConfig):
        self.config = config
        if config.backend == BackendType.POSTGRES:
            self._init_postgres()
        else:
            self._init_json()

    def save_json(self, data: dict[str, Any], filename: str) -> None:
        """Atomic JSON write with temp file + rename."""

    def load_json(self, filename: str) -> dict[str, Any] | None:
        """Load JSON with error handling."""

    def log_audit(self, entity_type: str, entity_id: str,
                  action: str, changes: dict[str, Any] | None = None,
                  user_id: str | None = None) -> None:
        """Record audit log (JSONL or PostgreSQL)."""
```

**Database Tables:**

```sql
-- models table (33 columns with indexes)
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    feature_schema_hash VARCHAR(64) NOT NULL,
    serveable BOOLEAN DEFAULT TRUE,
    artifact_format TEXT DEFAULT 'onnx',
    artifact_sha256_digest TEXT,
    ...
);

-- features table (18 columns with indexes)
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) UNIQUE NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    ...
);

-- strategies table (25 columns with indexes)
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    ...
);

-- Comprehensive indexing
CREATE INDEX idx_model_role ON models(role);
CREATE INDEX idx_model_feature_schema_hash ON models(feature_schema_hash);
CREATE INDEX idx_feature_stage ON features(stage);
CREATE INDEX idx_strategy_type ON strategies(strategy_type);
```

---

### 3.3 Protocol System (`protocols.py` - 108 lines)

**Protocol-first design** (Universal Pattern 2 - FULLY COMPLIANT).

```python
from typing import Protocol, Generic, TypeVar
from ml.config.events import Stage, Source, EventStatus

class RegistryProtocol(Protocol):
    """
    Backward-compatible protocol for DataRegistry integration.
    Used by DataStore and orchestration layers.
    """

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,  # Enum-typed
        source: Source,  # Enum-typed
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,  # Enum-typed
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None: ...

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None: ...

    def get_manifest(self, dataset_id: str) -> DatasetManifest: ...
    def get_contract(self, dataset_id: str) -> DataContract: ...
    def register_dataset(self, manifest: DatasetManifest) -> str: ...

# Generic typed protocol for future adoption
TManifest = TypeVar("TManifest")
TKey = TypeVar("TKey")

class TypedRegistryProtocol(Protocol, Generic[TManifest, TKey]):
    """Strictly-typed registry interface with enums."""

    def get(self, key: TKey) -> TManifest: ...
    def save(self, manifest: TManifest) -> TKey: ...
    def delete(self, key: TKey) -> bool: ...
    def list_manifests(self, prefix: str | None = None,
                       limit: int | None = None) -> list[TManifest]: ...
    def batch_save(self, manifests: list[TManifest]) -> list[TKey]: ...
```

**Benefits:**
- ✅ Structural typing without inheritance
- ✅ Duck typing support for testing (DummyRegistry conforms)
- ✅ Type safety without circular dependencies
- ✅ Clear contracts for component interactions

---

### 3.4 Statistical Utilities (`statistics.py` - 219 lines)

**Comprehensive statistical validation** for A/B testing and model comparison.

```python
def welch_t_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Welch's t-test for unequal variances.

    Returns:
    - statistic: t-statistic value
    - p_value: approximate p-value
    - significant: bool (p < significance_level)
    - improvement_pct: relative improvement percentage
    """

def compare_models(
    models: list[dict[str, Any]],
    metric_name: str,
    baseline_index: int = 0,
) -> dict[str, Any]:
    """
    Multi-model comparison with ranking.

    Returns:
    - rankings: sorted by metric (descending)
    - winner: model with highest metric
    - improvements: relative to baseline
    """

def calculate_sample_size(
    effect_size: float,
    power: float = 0.8,
    significance_level: float = 0.05,
) -> int:
    """
    A/B test sample size using Cohen's d.

    Returns minimum samples per group for statistical validity.
    """
```

---

### 3.5 Bootstrap System (`bootstrap_datasets.py` - 654 lines)

**Automated dataset initialization** for standard pipelines.

```python
# CLI interface
python -m ml.registry.bootstrap_datasets --backend json --registry-path /tmp/registry
NAUTILUS_REGISTRY_DB_URL="postgresql://..." python -m ml.registry.bootstrap_datasets --backend postgres

# Standard datasets created:
# 1. BARS - OHLCV market data (lenient validation)
# 2. TRADES - Individual trade ticks (strict validation)
# 3. QUOTES - Bid/ask quotes (monotonicity validation)
# 4. FEATURES - ML features (strict, no nulls)
# 5. PREDICTIONS - Model outputs (range validation [-1, 1])
# 6. SIGNALS - Strategy signals (monitor-only enforcement)
```

---

## 4. Integration with Core Systems

### 4.1 Actor Integration (Pattern 1 Compliance)

**BaseMLInferenceActor** automatically initializes all 4 registries:

```python
# ml/actors/base.py (not shown in full - see context_actors.md)
class BaseMLInferenceActor(MLComponentMixin, NautilusActor, ABC):
    """
    Base class for ML inference actors.
    Automatically initializes 4-store + 4-registry pattern.
    """

    def _init_stores_and_registries(self) -> None:
        """Initialize mandatory stores and registries."""
        if use_dummy_stores:
            self._feature_registry = DummyRegistry()
            self._model_registry = DummyRegistry()
            self._strategy_registry = DummyRegistry()
            self._data_registry = DummyRegistry()
        else:
            registry_path = Path(".nautilus/ml/registry")
            self._feature_registry = FeatureRegistry(registry_path, ...)
            self._model_registry = ModelRegistry(registry_path, ...)
            self._strategy_registry = StrategyRegistry(registry_path, ...)
            self._data_registry = DataRegistry(registry_path, ...)
```

**Usage in actors:**

```python
from ml.actors.base import BaseMLInferenceActor

class ProductionMLActor(BaseMLInferenceActor):
    def on_start(self) -> None:
        # Registries auto-initialized by base class
        model_info = self._model_registry.get_model("lgb_student_v1")

        # Validate feature compatibility
        assert model_info.manifest.feature_schema_hash == expected_hash

        # Load model for inference
        self._model_session = self._model_registry.load_model("lgb_student_v1")
```

---

### 4.2 Store Integration

**DataStore uses DataRegistry** for contract validation and event emission:

```python
# ml/stores/data_store.py
class DataStore:
    def __init__(self, registry: RegistryProtocol, connection_string: str):
        self.registry = registry  # Type-safe protocol

    def write_features(self, feature_set_id: str, features: dict[str, float]):
        # Automatic schema validation against FeatureRegistry
        manifest = self.registry.get_manifest(f"features_{feature_set_id}")
        self._validate_schema(features, manifest.schema)

        # Write to database
        self._write_to_db(features)

        # Emit event via registry
        self.registry.emit_event(
            dataset_id=f"features_{feature_set_id}",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            status=EventStatus.SUCCESS,
            ...
        )
```

---

### 4.3 Orchestration Integration

**Pipeline orchestrator** uses DataRegistry for watermark tracking and gap detection:

```python
# ml/data/ingest/orchestrator.py (not shown in full - see context_data.md)
class BackfillOrchestrator:
    def __init__(self, registry: RegistryProtocol):
        self.registry = registry

    def process_window(self, dataset_id: str, instrument_id: str, window: tuple[int, int]):
        ts_min, ts_max = window

        # Process data
        count = self._ingest_data(dataset_id, instrument_id, ts_min, ts_max)

        # Emit success event
        self.registry.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.DATA_INGESTED,
            source=Source.BACKFILL,
            run_id=self.run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=EventStatus.SUCCESS
        )

        # Update watermark
        self.registry.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=Source.BACKFILL,
            last_success_ns=ts_max,
            count=count,
            completeness_pct=self._compute_completeness(...)
        )
```

---

## 5. Universal ML Architecture Pattern Compliance

### Pattern Compliance Matrix

| **Pattern** | **Status** | **Implementation** | **Gaps** |
|-------------|------------|-------------------|----------|
| **Pattern 1: 4-Store + 4-Registry** | ✅ **FULLY COMPLIANT** | All registries inherit from MLComponentMixin. Thread-safe with RLock. Progressive fallback to DummyRegistry. | None |
| **Pattern 2: Protocol-First Design** | ✅ **FULLY COMPLIANT** | RegistryProtocol + TypedRegistryProtocol. Structural typing with duck typing support. | None |
| **Pattern 3: Hot/Cold Path Separation** | ⚠️ **PARTIALLY COMPLIANT** | `serveable` flag + ONNX-only serving. | Missing: P99 latency validation, pre-allocated arrays, benchmarking |
| **Pattern 4: Progressive Fallback** | ⚠️ **PARTIALLY COMPLIANT** | DummyRegistry fallback + multi-backend. | Missing: Circuit breaker, 4-tier fallback chain, connection monitoring |
| **Pattern 5: Centralized Metrics** | ❌ **NON-COMPLIANT** | Zero metrics_bootstrap imports found. | Complete absence of Prometheus integration |

---

### 5.1 Pattern 1: 4-Store + 4-Registry (✅ FULLY COMPLIANT)

**Evidence:**

```python
# All registries properly inherit from MLComponentMixin
class ModelRegistry(AbstractRegistry):  # → MLComponentMixin via AbstractRegistry
class FeatureRegistry(AbstractRegistry):  # → MLComponentMixin via AbstractRegistry
class StrategyRegistry(AbstractRegistry):  # → MLComponentMixin via AbstractRegistry
class DataRegistry(MLComponentMixin):  # Direct inheritance

# Thread-safe operations
self._lock = threading.RLock()  # All registries

# Progressive fallback
if use_dummy_stores:
    self._model_registry = DummyRegistry()
```

---

### 5.2 Pattern 2: Protocol-First Design (✅ FULLY COMPLIANT)

**Evidence:**

```python
# protocols.py - Complete structural typing
class RegistryProtocol(Protocol):
    def emit_event(...) -> None: ...
    def update_watermark(...) -> None: ...

class TypedRegistryProtocol(Generic[TManifest, TKey], Protocol):
    def get(self, key: TKey) -> TManifest: ...
    def save(self, manifest: TManifest) -> TKey: ...

# Usage in stores (type-safe)
def __init__(self, registry: RegistryProtocol, ...):
    self.registry = registry
```

---

### 5.3 Pattern 3: Hot/Cold Path Separation (⚠️ PARTIALLY COMPLIANT)

**Implemented:**
- ✅ `serveable: bool` flag in ModelManifest
- ✅ ONNX-only serving for hot path (`artifact_format = "onnx"`)
- ✅ Teacher/student model roles (cold/hot separation)

**Missing:**
- ❌ Sub-5ms P99 latency validation code
- ❌ Pre-allocated array patterns in hot path methods
- ❌ Performance benchmarking utilities
- ❌ Hot path performance monitoring

**Recommendation:**

```python
# Add to model_registry.py
from ml.common.metrics_bootstrap import get_histogram

class ModelRegistry:
    def __init__(self, ...):
        self._latency_histogram = get_histogram(
            "ml_registry_operation_latency_seconds",
            "Registry operation latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
        )

    def load_model(self, model_id: str):
        start = time.perf_counter()
        session = self._persistence.load_model(model_id)
        latency = time.perf_counter() - start

        self._latency_histogram.observe(latency)

        if latency > 0.005:  # 5ms threshold
            logger.warning(f"Hot path latency violation: {latency*1000:.2f}ms")

        return session
```

---

### 5.4 Pattern 4: Progressive Fallback (⚠️ PARTIALLY COMPLIANT)

**Implemented:**
- ✅ DummyRegistry fallback for testing
- ✅ Multi-backend persistence (PostgreSQL → JSON)
- ✅ Basic error handling with logging

**Missing:**
- ❌ Circuit breaker patterns
- ❌ 4-tier fallback chain (PRIMARY → CACHED → FILE → DUMMY)
- ❌ Connection pool health monitoring
- ❌ Backpressure handling

**Recommendation:**

```python
# Add circuit breaker to persistence.py
from ml.common.circuit_breaker import CircuitBreaker

class PersistenceManager:
    def __init__(self, config: PersistenceConfig):
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60
        )

    def save_model(self, model_info: ModelInfo):
        if self._circuit_breaker.is_open():
            logger.warning("Circuit breaker open, using fallback")
            self._save_to_json_fallback(model_info)
            return

        try:
            self._save_to_postgres(model_info)
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error("Save failed, using fallback", exc_info=True)
            self._save_to_json_fallback(model_info)
```

---

### 5.5 Pattern 5: Centralized Metrics (❌ NON-COMPLIANT)

**Critical Gap Identified:**

```bash
$ grep -r "metrics_bootstrap\|get_counter\|get_histogram" ml/registry/*.py
# Returns: 0 matches
```

**No metrics integration found** in any registry file.

**Required Actions:**

```python
# Add to ALL registry classes
from ml.common.metrics_bootstrap import get_counter, get_histogram

class ModelRegistry(AbstractRegistry):
    def __init__(self, ...):
        super().__init__(...)
        self._init_metrics()

    def _init_metrics(self) -> None:
        self.registry_ops_counter = get_counter(
            "ml_registry_operations_total",
            "Total registry operations",
            ["registry_type", "operation", "status"]
        )
        self.registry_latency_histogram = get_histogram(
            "ml_registry_operation_latency_seconds",
            "Registry operation latency",
            ["registry_type", "operation"]
        )

    def register_model(self, ...):
        start = time.perf_counter()
        try:
            result = self._register_model_impl(...)
            self.registry_ops_counter.labels(
                registry_type="model",
                operation="register",
                status="success"
            ).inc()
            return result
        except Exception as e:
            self.registry_ops_counter.labels(
                registry_type="model",
                operation="register",
                status="error"
            ).inc()
            raise
        finally:
            latency = time.perf_counter() - start
            self.registry_latency_histogram.labels(
                registry_type="model",
                operation="register"
            ).observe(latency)
```

---

## 6. Production Readiness Assessment

### 6.1 Maturity Analysis

**Component Maturity:**

| Component | Status | Confidence | Evidence |
|-----------|--------|------------|----------|
| **Core registries** | Production-ready | ✅ High | 4,180+ lines, comprehensive testing, battle-tested |
| **Strangler fig decomposition** | Mature (Phase 2.3) | ✅ High | Feature flags, 100% API compat, safe rollback |
| **Multi-backend persistence** | Production-ready | ✅ High | PostgreSQL + JSON, connection pooling, migrations |
| **Protocol system** | Production-ready | ✅ High | Full type safety, structural typing, zero coupling |
| **Statistical validation** | Production-ready | ✅ High | Welch's t-test, sample size calculation, ranking |
| **Thread safety** | Production-ready | ✅ High | RLock everywhere, tested concurrency |
| **Metrics integration** | Not implemented | ❌ Critical gap | Zero metrics_bootstrap usage |
| **Circuit breakers** | Not implemented | ⚠️ Moderate gap | Basic error handling only |

---

### 6.2 Production Strengths 💪

1. **Manifest-centric design excellence**
   - Self-describing components with complete metadata
   - SHA256 hash-based compatibility validation
   - Autonomous validation without external configuration

2. **Strangler fig pattern success**
   - 100% backward compatibility maintained
   - Safe feature flag rollback
   - Component extraction reduces facade size by 47-51%

3. **Thread-safe multi-backend persistence**
   - RLock for all operations
   - PostgreSQL with ACID compliance
   - JSON for development/testing
   - Connection pooling and session management

4. **Comprehensive statistical validation**
   - Welch's t-test for model comparison
   - A/B testing framework
   - Canary deployment with automated promotion

5. **Protocol-first type safety**
   - Structural typing without inheritance
   - Duck typing support for testing
   - Zero circular dependencies

6. **Security validation**
   - Path traversal protection
   - ONNX-only serving for hot path
   - SHA-256 artifact integrity verification

---

### 6.3 Critical Gaps 🚧

**1. Missing Centralized Metrics (Pattern 5 - CRITICAL)**

**Impact:** Zero production observability for registry operations.

**Required Actions:**
- Add `ml.common.metrics_bootstrap` imports to all registries
- Instrument all CRUD operations with counters and histograms
- Track latency for hot path operations
- Monitor fallback activations

**Priority:** P0 (blocking for production deployment)

---

**2. Incomplete Progressive Fallback (Pattern 4 - MODERATE)**

**Impact:** Limited fault tolerance for external dependencies.

**Required Actions:**
- Implement circuit breaker patterns for PostgreSQL operations
- Add 4-tier fallback chain (PRIMARY → CACHED → FILE → DUMMY)
- Monitor connection pool health
- Add backpressure handling

**Priority:** P1 (important for production reliability)

---

**3. Hot Path Performance Validation (Pattern 3 - MODERATE)**

**Impact:** No enforcement of sub-5ms P99 latency requirements.

**Required Actions:**
- Add P99 latency validation code
- Implement pre-allocated array patterns
- Add performance benchmarking utilities
- Monitor hot path operations

**Priority:** P1 (important for production performance)

---

### 6.4 Deployment Maturity

**Overall Assessment: ~85% Production-Ready**

| Category | Status | Completion |
|----------|--------|------------|
| **Core Functionality** | ✅ Production-ready | 100% |
| **Strangler Fig Decomposition** | ✅ Mature | 100% |
| **Multi-Backend Persistence** | ✅ Production-ready | 100% |
| **Thread Safety** | ✅ Production-ready | 100% |
| **Protocol Design** | ✅ Production-ready | 100% |
| **Statistical Validation** | ✅ Production-ready | 100% |
| **Security** | ✅ Production-ready | 100% |
| **Universal Pattern Compliance** | ⚠️ Partial | 60% (3/5 patterns) |
| **Metrics Integration** | ❌ Missing | 0% |
| **Circuit Breakers** | ❌ Missing | 0% |
| **Hot Path Validation** | ⚠️ Partial | 40% |

**Recommended Path to 100%:**

1. **Phase 1 (P0 - Blocking):**
   - Add centralized metrics integration to all registries
   - Verify metrics collection and scraping
   - **Estimated effort:** 2-3 days

2. **Phase 2 (P1 - Important):**
   - Implement circuit breaker patterns
   - Add 4-tier progressive fallback
   - Add hot path performance validation
   - **Estimated effort:** 3-5 days

3. **Phase 3 (P2 - Nice to have):**
   - Add advanced monitoring dashboards
   - Implement automated alerting
   - Add performance benchmarking suite
   - **Estimated effort:** 5-7 days

---

## 7. Usage Patterns & Best Practices

### 7.1 Model Registration Workflow

```python
from pathlib import Path
from ml.registry import ModelRegistry, ModelManifest, ModelRole, DataRequirements
from ml.registry import QualityGate, PersistenceConfig, BackendType

# Initialize registry
registry = ModelRegistry(
    registry_path=Path(".nautilus/ml/registry"),
    persistence_config=PersistenceConfig(backend=BackendType.POSTGRES)
)

# Create manifest
manifest = ModelManifest(
    model_id="lgb_student_v2",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    architecture="LightGBM",
    feature_schema={"close_ratio": "float32", "volume_ma": "float32"},
    feature_schema_hash="abc123...",
    parent_id="tft_teacher_v1",
    feature_set_id="student_features_v1",
    serveable=True,
    artifact_format="onnx",
    version="2.0.0"
)

# Register with quality gates
registry.register_model(
    model_path=Path("models/lgb_student_v2.onnx"),
    manifest=manifest,
    auto_deploy=True,
    quality_gates=[
        QualityGate("accuracy", 0.82, "gte"),
        QualityGate("inference_latency_ms", 3.0, "lte")
    ]
)

# Deploy to production
registry.deploy_model("lgb_student_v2", target="ml_signal_actor")

# Hot reload (zero downtime)
registry.hot_reload_model(target="ml_signal_actor", new_model_id="lgb_student_v3")

# Rollback if needed
registry.rollback_deployment(target="ml_signal_actor")
```

---

### 7.2 Feature Set Registration

```python
from ml.registry import FeatureRegistry, FeatureManifest, FeatureRole, FeatureStage
from ml.registry import compute_schema_hash, QualityGate

# Compute schema hash
schema_hash = compute_schema_hash(
    feature_names=["close_ratio", "volume_ma", "rsi"],
    feature_dtypes=["float32", "float32", "float32"],
    pipeline_signature="pipeline_v2_hash"
)

# Create manifest
manifest = FeatureManifest(
    feature_set_id="student_features_v2",
    name="Student Features V2",
    version="2.0.0",
    role=FeatureRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    feature_names=["close_ratio", "volume_ma", "rsi"],
    feature_dtypes=["float32", "float32", "float32"],
    schema_hash=schema_hash,
    pipeline_signature="pipeline_v2_hash",
    pipeline_version="2.0.0",
    stage=FeatureStage.CANDIDATE,
    parity_tolerance=1e-10,
    constraints={"max_latency_ms": 0.5, "min_bars_warmup": 20}
)

# Register feature set
registry.register_feature_set(manifest)

# Promote through lifecycle stages
registry.validate_and_promote(
    "student_features_v2",
    quality_gates=[
        QualityGate("parity_max_diff", 1e-10, "lte"),
        QualityGate("p99_latency_ms", 0.5, "lte")
    ]
)
# CANDIDATE → STAGING → PROD
```

---

### 7.3 Dataset Registration & Tracking

```python
from ml.registry import DataRegistry, DatasetManifest, DatasetType, StorageKind
from ml.config.events import Stage, Source, EventStatus

# Initialize registry
registry = DataRegistry(
    registry_path=Path(".nautilus/ml/registry"),
    persistence_config=PersistenceConfig(backend=BackendType.POSTGRES)
)

# Register dataset
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
    schema_hash="def456...",
    lineage=[],
    pipeline_signature="data_scheduler_v1",
    version="1.0.0"
)

dataset_id = registry.register_dataset(manifest)

# Emit processing event
registry.emit_event(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    stage=Stage.CATALOG_WRITTEN,
    source=Source.HISTORICAL,
    run_id="backfill_2025_q1",
    ts_min=1234567890000000000,
    ts_max=1234567900000000000,
    count=1000,
    status=EventStatus.SUCCESS,
    metadata={"pipeline_version": "v2.1", "worker_id": "worker_01"}
)

# Update watermark
registry.update_watermark(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    source=Source.BACKFILL,
    last_success_ns=1234567900000000000,
    count=1000,
    completeness_pct=98.5
)

# Link lineage
registry.link_lineage(
    child_dataset_id="features_microstructure",
    parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
    transform_id="feature_pipeline_v2",
    ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
    params={"lookback_bars": 20, "include_imbalance": True}
)
```

---

### 7.4 A/B Testing & Canary Deployment

```python
# Start A/B test
registry.start_ab_test(
    test_id="lgb_v1_vs_v2_comparison",
    model_a="lgb_student_v1",
    model_b="lgb_student_v2",
    traffic_split=0.5,
    metric_name="sharpe_ratio",
    min_samples=1000
)

# Check A/B test results
results = registry.get_ab_test_results("lgb_v1_vs_v2_comparison")
# {
#   "winner": "lgb_student_v2",
#   "statistic": 2.34,
#   "p_value": 0.019,
#   "significant": True,
#   "improvement_pct": 12.5
# }

# Start canary deployment
from ml.registry import CanaryConfig

registry.start_canary_deployment(
    model_id="lgb_student_v3",
    target="ml_signal_actor",
    config=CanaryConfig(
        initial_traffic=0.05,
        increment=0.05,
        interval_seconds=300,
        min_samples=500,
        error_rate_threshold=0.05,
        baseline_threshold=0.95
    )
)

# Canary auto-promotes or rolls back based on metrics
```

---

## 8. Migration Guide

### 8.1 Migrating from Legacy to Component-Based

**Current Default:** Component-based implementations are default (no action needed).

**To rollback to legacy:**

```bash
# Rollback to legacy god classes
export ML_USE_LEGACY_MODEL_REGISTRY=1
export ML_USE_LEGACY_DATA_REGISTRY=1

# Restart services
systemctl restart nautilus-trader
```

**To explicitly opt-in to component-based:**

```bash
# Use new component-based implementations
export ML_USE_COMPONENT_MODEL_REGISTRY=1
export ML_USE_LEGACY_DATA_REGISTRY=0

# Restart services
systemctl restart nautilus-trader
```

---

### 8.2 Database Migration

```bash
# Apply registry migrations
psql $DATABASE_URL -f ml/registry/migrations/001_initial_schema.sql
psql $DATABASE_URL -f ml/registry/migrations/002_add_cold_path_fields.sql
psql $DATABASE_URL -f ml/registry/migrations/003_add_artifact_digest.sql

# Verify migrations
psql $DATABASE_URL -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'ml_registry';"
# Expected: models, features, strategies, audit_log
```

---

### 8.3 Bootstrap Standard Datasets

```bash
# Bootstrap with JSON backend (development)
python -m ml.registry.bootstrap_datasets \
    --backend json \
    --registry-path /tmp/registry

# Bootstrap with PostgreSQL backend (production)
export NAUTILUS_REGISTRY_DB_URL="postgresql://user:pass@localhost/nautilus"
python -m ml.registry.bootstrap_datasets --backend postgres

# Verify datasets
python -c "
from ml.registry import DataRegistry
from pathlib import Path
registry = DataRegistry(Path('/tmp/registry'))
print(registry.list_datasets())
"
```

---

## 9. Testing Strategy

### 9.1 Test Coverage by Component

| Component | Coverage | Test Files | Notes |
|-----------|----------|------------|-------|
| **ModelRegistry** | High | `test_model_registry*.py` | Unit + integration + property tests |
| **DataRegistry** | High | `test_data_registry*.py` | Unit + integration + E2E tests |
| **FeatureRegistry** | High | `test_feature_registry*.py` | Schema hashing + quality gates |
| **StrategyRegistry** | High | `test_strategy_registry*.py` | Compatibility checking |
| **Persistence** | High | `test_persistence*.py` | JSON + PostgreSQL backends |
| **Components** | Moderate | `test_*_manager.py` | Individual component tests |
| **Migrations** | Moderate | `test_migrations.py` | Schema validation |

---

### 9.2 Testing Patterns

```python
# Property-based testing (hypothesis)
from hypothesis import given, strategies as st

@given(
    model_id=st.text(min_size=1, max_size=255),
    feature_names=st.lists(st.text(min_size=1), min_size=1, max_size=100),
    feature_dtypes=st.lists(st.sampled_from(["float32", "float64", "int32"]), min_size=1)
)
def test_schema_hash_deterministic(model_id, feature_names, feature_dtypes):
    """Schema hash is deterministic for same inputs."""
    hash1 = compute_schema_hash(feature_names, feature_dtypes, "sig1")
    hash2 = compute_schema_hash(feature_names, feature_dtypes, "sig1")
    assert hash1 == hash2

# Contract testing (pandera-style)
def test_model_manifest_schema_validation():
    """ModelManifest enforces schema constraints."""
    with pytest.raises(ValueError, match="feature_schema_hash must be 64 chars"):
        ModelManifest(
            model_id="test",
            feature_schema_hash="short_hash",  # Invalid
            ...
        )

# Integration testing
@pytest.mark.serial
def test_registry_postgres_integration(postgres_engine):
    """End-to-end registry operations with PostgreSQL."""
    registry = ModelRegistry(
        registry_path=Path("/tmp/test"),
        persistence_config=PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=postgres_engine.url
        )
    )

    # Register model
    manifest = ModelManifest(...)
    registry.register_model(Path("test.onnx"), manifest)

    # Verify persistence
    loaded = registry.get_model(manifest.model_id)
    assert loaded.manifest == manifest
```

---

## 10. Known Issues & Future Work

### 10.1 Known Limitations

1. **Single-node registry design**
   - No multi-node coordination
   - No distributed locking
   - Limited to vertical scaling

2. **Basic drift detection**
   - No automated retraining triggers
   - Manual monitoring required
   - No statistical drift alerts

3. **No web dashboard**
   - Command-line only
   - No visual model comparison
   - No interactive deployment management

4. **Limited cross-registry validation**
   - Model-feature compatibility checked at deployment
   - No proactive compatibility scanning
   - No dependency graph visualization

---

### 10.2 Future Enhancements

**Phase 4: Advanced Observability (Q1 2026)**
- Complete Pattern 5 compliance (centralized metrics)
- Circuit breaker patterns (Pattern 4)
- Hot path performance validation (Pattern 3)
- Real-time drift detection

**Phase 5: Multi-Node Coordination (Q2 2026)**
- Distributed locking via Redis
- Registry synchronization across nodes
- Leader election for deployment decisions
- Horizontal scaling support

**Phase 6: Advanced Analytics (Q3 2026)**
- Automated drift detection and alerting
- Model performance degradation analysis
- Feature importance tracking over time
- Dependency graph visualization

**Phase 7: Web Dashboard (Q4 2026)**
- Interactive model comparison
- Deployment management UI
- Real-time monitoring dashboards
- Audit trail visualization

---

## 11. Cross-Module References

- **Data Pipeline**: See `context_data.md` for DataRegistry integration with ingestion
- **Feature Engineering**: See `context_features.md` for FeatureRegistry validation patterns
- **Stores**: See `context_stores.md` for 4-store persistence architecture
- **Training**: See `context_training.md` for model registration post-training
- **Actors**: See `context_actors.md` for BaseMLInferenceActor mandatory patterns
- **Models**: See `context_models.md` for manifest-based model implementations
- **Monitoring**: See `context_monitoring.md` for registry observability integration
- **Configuration**: See `context_config.md` for persistence and policy configuration

---

## 12. Key Takeaways

### ✅ Production-Ready Strengths

1. **Mature Strangler Fig Pattern**: Phase 2.3 complete with feature flag rollback
2. **Component Decomposition**: 10 specialized components replace 2 god classes
3. **Protocol-First Design**: Full type safety without coupling
4. **Multi-Backend Persistence**: PostgreSQL + JSON with graceful fallback
5. **Thread-Safe Operations**: RLock everywhere with batch operations
6. **Comprehensive Testing**: Unit + integration + property + E2E tests

### ⚠️ Critical Gaps Requiring Attention

1. **Missing Metrics Integration**: Zero Prometheus metrics (Pattern 5 non-compliant)
2. **Incomplete Fallback**: No circuit breakers or 4-tier fallback chain
3. **Hot Path Validation**: No P99 latency enforcement

### 📊 Overall Assessment

**Status**: ~85% production-ready
**Confidence**: High for core functionality, moderate for advanced patterns
**Recommendation**: Implement missing metrics integration (P0) before full production deployment

---

**Document Version:** 2.0
**Agent**: Agent 5 (Context Documentation Specialist)
**Review Date:** 2025-10-19
**Next Review:** 2026-01-19 (quarterly update recommended)
