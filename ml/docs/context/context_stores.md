# ML Stores Context Documentation

## Executive Summary

The Nautilus Trader ML stores infrastructure implements a sophisticated four-tier storage architecture consisting of FeatureStore, ModelStore, StrategyStore, and DataStore. This system provides mandatory data persistence for all ML actors with advanced PostgreSQL partitioning, comprehensive data processing pipelines, contract validation, and event tracking. The architecture enforces strict adherence to Nautilus conventions while delivering enterprise-grade performance and reliability.

**📝 ADDITION:** The stores architecture has been enhanced with centralized database engine management through EngineManager, improved message bus integration, and advanced protocol-based interfaces for better type safety and testing.

## Core Architecture

### The Store Quartet

The ML infrastructure is built around four stores that form the backbone of the data persistence layer:

```python
# Every ML actor MUST use these four mandatory stores
class BaseMLInferenceActor:
    def __init__(self):
        self.feature_store = FeatureStore(connection_string)
        self.model_store = ModelStore(persistence_config)
        self.strategy_store = StrategyStore(persistence_config)
        self.data_store = DataStore(connection_string, registry)  # **✨ ENHANCEMENT:** Added missing data_store

# DataStore provides unified facade with validation
class DataStore:
    def __init__(self, registry, connection_string):
        self.feature_store = FeatureStore(connection_string)
        self.model_store = ModelStore(connection_string)
        self.strategy_store = StrategyStore(connection_string)
        self.data_processor = DataProcessor(connection_string)
```

#### FeatureStore (`ml/stores/feature_store.py`)

- **Purpose**: Unified feature computation and storage ensuring training/inference parity
- **Key Capability**: Uses same FeatureEngineer for both batch (historical) and online (live) computation
- **Storage**: PostgreSQL table `ml_feature_values` with time-based partitioning
- **Schema**: Feature values stored as JSONB with instrument_id, ts_event, ts_init (nanoseconds)
- **Pipeline Integration**: Supports PipelineSpec for declarative feature transformations
- **Data Registry**: Lazy initialization of DataRegistry for event emission
- **✨ ENHANCEMENT:** Uses centralized EngineManager for database connections
- **📝 ADDITION:** Supports both PersistenceConfig and connection_string initialization patterns
- **📝 ADDITION:** Integrated with message bus publishing for real-time feature updates

#### ModelStore (`ml/stores/model_store.py`)

- **Purpose**: Model prediction storage with performance tracking
- **Key Capability**: Batch writing with auto-flushing, latency tracking, confidence scoring
- **Storage**: PostgreSQL table `ml_model_predictions` with time-based partitioning
- **Schema**: Predictions with features_used, inference_time_ms, confidence metrics
- **Persistence**: Supports both connection_string and PersistenceConfig initialization
- **📝 ADDITION:** Configurable batch_size and flush_interval_ms for performance tuning
- **📝 ADDITION:** Optional Clock integration for Nautilus timestamp consistency
- **✨ ENHANCEMENT:** Message bus publishing support with configurable publisher

#### StrategyStore (`ml/stores/strategy_store.py`)

- **Purpose**: Strategy signal storage with risk metrics and execution parameters
- **Key Capability**: Signal attribution, risk tracking, execution parameter calculation
- **Storage**: PostgreSQL table `ml_strategy_signals` with time-based partitioning
- **Schema**: Signals with model_predictions mapping, risk_metrics, execution_params
- **📝 ADDITION:** Configurable batch_size and flush_interval_ms for performance optimization
- **📝 ADDITION:** Optional Clock integration for timestamp synchronization
- **✨ ENHANCEMENT:** Message bus publishing support for real-time signal distribution

#### DataStore (`ml/stores/data_store.py`)

- **Purpose**: Typed read/write facade with contract validation and event emission
- **Key Capability**: Schema validation, quality reporting, watermark tracking
- **Features**:
  - Preflight schema checks before data processing
  - Contract-based validation with enforcement modes (strict, lenient, monitor_only)
  - Automatic event emission to DataRegistry
  - Schema migration support with dual-write windows
  - Quality scoring and violation tracking
- **📝 ADDITION:** Comprehensive event emission with correlation_id for lineage tracking
- **📝 ADDITION:** Message bus integration for event publishing to external systems
- **✨ ENHANCEMENT:** Advanced preflight validation with type compatibility checking

### Base Classes and Data Models

Located in `ml/stores/base.py`:

```python
@dataclass
class FeatureData(Data):
    """Nautilus-compatible feature data class."""
    feature_set_id: str
    instrument_id: str
    values: dict[str, float]
    _ts_event: int  # nanoseconds
    _ts_init: int   # nanoseconds
    quality_flags: int = 0  # **📝 ADDITION:** Quality tracking support

    @property
    def feature_values(self) -> dict[str, float]:
        """**📝 ADDITION:** Safe accessor for feature values dict avoiding inheritance conflicts."""
        # Defensive copy as a plain dict[str, float]
        raw: dict[str, Any] = object.__getattribute__(self, "__dict__")
        data = raw.get("values")
        if data is None:
            candidate = getattr(self, "values", {})
            data = candidate() if callable(candidate) else candidate
        try:
            return {str(k): float(v) for k, v in dict(data or {}).items()}
        except Exception:
            return {}

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

@dataclass
class ModelPrediction(Data):
    """Store model predictions and inference metadata."""
    model_id: str
    instrument_id: str
    prediction: float
    confidence: float
    features_used: dict[str, float]
    inference_time_ms: float
    _ts_event: int
    _ts_init: int

@dataclass
class StrategySignal(Data):
    """Store strategy decisions and execution signals."""
    strategy_id: str
    instrument_id: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float
    model_predictions: dict[str, float]
    risk_metrics: dict[str, float]
    execution_params: dict[str, Any]
    _ts_event: int
    _ts_init: int

class BaseStore(MLComponentMixin, ABC):
    """Abstract base class for all store implementations."""
    @abstractmethod
    def write_batch(self, data: list[Any]) -> None: ...
    @abstractmethod
    def read_range(self, start_ns: int, end_ns: int, instrument_id: str | None = None) -> pd.DataFrame: ...
    @abstractmethod
    def flush(self) -> None: ...
    @abstractmethod
    def get_latest(self, instrument_id: str, limit: int = 1) -> pd.DataFrame: ...
    @abstractmethod
    def get_statistics(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]: ...

class DummyStore:
    """Dummy store for testing when database is not available."""
    # **✨ ENHANCEMENT:** Comprehensive protocol compliance for all store methods
    # Accepts all method calls but doesn't persist anything
```

**📝 ADDITION:** Advanced Protocol Support (`ml/stores/protocols.py`)

```python
# Protocol interfaces for structural type checking and test compatibility
class BaseStoreProtocol(Protocol):
    def write_batch(self, data: list[Any]) -> None: ...
    def read_range(self, start_ns: int, end_ns: int, instrument_id: str | None = None) -> Any: ...
    def flush(self) -> None: ...
    def get_latest(self, instrument_id: str, limit: int = 1) -> Any: ...
    def get_statistics(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]: ...

class FeatureStoreProtocol(Protocol):
    def write_features(self, ...) -> None: ...
    def compute_realtime(self, bar: Any, store: bool = ..., indicator_manager: Any | None = ...) -> Any: ...

class ModelStoreProtocol(Protocol):
    def write_prediction(self, ...) -> None: ...
    def read_predictions(self, ...) -> Any: ...
    def get_model_performance(self, ...) -> dict[str, Any]: ...

class StrategyStoreProtocol(Protocol):
    def write_signal(self, ...) -> None: ...
    def read_signals(self, ...) -> Any: ...
    def get_strategy_performance(self, ...) -> dict[str, Any]: ...
    def get_signal_distribution(self, ...) -> dict[str, int]: ...
```

## Database Schema Architecture

**✨ ENHANCEMENT:** Migrations under `ml/stores/migrations/` are the canonical source of schema.
Legacy SQL files under `ml/schema/` are retained for reference only.

**📝 ADDITION:** Recent schema enhancements include:

- **006_disable_partition_triggers.sql**: Disables race-prone automatic partition triggers
- **007_add_event_metadata.sql**: Adds JSONB metadata column to events table
- **007_brin_indexes.sql**: BRIN indexes for efficient time-range queries

### Partitioning Strategy

All ML tables use sophisticated time-based partitioning managed by PostgreSQL native features:

#### Core Tables

1. **ml_feature_values** - Partitioned by ts_event (monthly partitions)
2. **ml_model_predictions** - Partitioned by ts_event (monthly partitions)
3. **ml_strategy_signals** - Partitioned by ts_event (monthly partitions)

#### Partition Management

- **Automatic Creation**: PartitionManager creates 3 months of partitions ahead
- **Future Planning**: Configurable months_ahead parameter (default: 3)
- **Retention**: Automatic cleanup of partitions older than 24 months (configurable)
- **Naming Convention**: `{table_name}_{YYYY}_{MM}` (e.g., `ml_feature_values_2024_03`)
- **⚠️ CORRECTION:** Automatic partition triggers have been disabled due to race conditions

### Migration System

Located in `ml/stores/migrations/`:

#### Recent Migration Files **📝 ADDITION:**

- **006_disable_partition_triggers.sql**: Removes problematic automatic partition triggers
- **007_add_event_metadata.sql**: Extends events table with JSONB metadata support
- **007_brin_indexes.sql**: Adds BRIN indexes for efficient time-range queries

#### 001_stores_schema.sql

- Creates core partitioned tables with helper functions
- Establishes 36 months of initial partitions (2024-2026)
- Defines optimized indexes for time-series queries

### Timestamp Policy & Normalization

All ML writes must use UNIX nanoseconds for `ts_event`/`ts_init`. To reduce operational risk, stores defensively normalize smaller-scale timestamps on write:

- If values look like seconds/milliseconds/microseconds, they are upscaled to nanoseconds and a warning is logged.
- This applies to FeatureStore, ModelStore, and StrategyStore write paths (including batch `_execute_write`).

Producers should still emit nanoseconds directly; normalization is a safety net, not a contract change.

### Query Safety

Dynamic reads in FeatureStore (historical bars) and StrategyStore are parameterized using SQLAlchemy, avoiding string interpolation and aligning with security best practices.

### Registry Interaction

- Stores emit data events and update watermarks via a Protocol-typed registry interface (`RegistryProtocol`).
- This enforces correct usage of `emit_event(...)`/`update_watermark(...)` across JSON and Postgres backends and prevents API drift from breaking callers.
- Events now include optional `metadata` persisted by the registry (JSONB in Postgres). `DataStore` attaches a deterministic `correlation_id` for lineage tracing; use it in logs/queries, not as a general metric label (avoid high cardinality).

## Universal Component Protocol

All stores implement the universal component interface defined in `ml/common/protocols.py` via `MLComponentMixin`:

- `get_health_status() -> dict[str, Any]`
- `get_performance_metrics() -> dict[str, float]`
- `validate_configuration() -> list[str]`

These methods are not used in hot write paths; use them for health endpoints and startup checks. Integration Manager validates protocol compliance (warn by default; strict via `ML_STRICT_PROTOCOL_VALIDATION`).

### DB Preflight **📝 ADDITION:** (`ml/stores/db_preflight.py`)

Use `ml/stores/db_preflight.check_db_prereqs()` at startup or during ops to verify:

- Required functions (`emit_data_event`, `update_watermark`) exist
- Current-month partitions exist for ML tables
- Creates metadata tracking tables:
  - `ml_feature_computation_stats` - Feature computation performance
  - `ml_feature_lineage` - Feature transformation tracking
  - `ml_strategy_performance` - Strategy signal statistics

```python
REQUIRED_FUNCTIONS = [
    "emit_data_event",
    "update_watermark",
]

PARTITIONED_TABLES = [
    "ml_feature_values",
    "ml_model_predictions",
    "ml_strategy_signals",
]
```

#### 002_auto_partitioning.sql

- Implements automatic partition creation functions
- Sets up triggers for on-demand partition creation
- Provides cleanup functions for old partitions
- Optional pg_cron integration for scheduled maintenance
- **⚠️ CORRECTION:** Triggers have been disabled in favor of pre-created partitions

#### 003_market_data.sql

- Market data tables with OHLCV support
- Instrument metadata with trading specifications
- Position and risk tracking infrastructure
- Statistics tables for outlier detection

#### 004_data_registry.sql

- Data registry tables for manifest and contract storage
- Event tracking and watermark management
- Dataset versioning and lineage

#### 005_schema_hardening.sql & 005_views.sql

- Schema hardening and optimization
- Materialized views for performance
- Additional indexes and constraints

#### 005a_feature_values_dedupe.sql

- Deduplication logic for feature values
- Ensures data integrity and prevents duplicates

### Indexing Strategy

Optimized for time-series query patterns:

```sql
-- Time-based queries (BRIN for compression)
CREATE INDEX idx_ml_feature_values_lookup
    ON ml_feature_values (feature_set_id, instrument_id, ts_event);

-- Live data filtering
CREATE INDEX idx_ml_feature_values_live
    ON ml_feature_values (is_live) WHERE is_live = TRUE;

-- Signal type analysis
CREATE INDEX idx_ml_strategy_signals_type
    ON ml_strategy_signals (signal_type);

-- **📝 ADDITION:** BRIN indexes for efficient range scans
CREATE INDEX IF NOT EXISTS brin_ml_feature_values_ts
    ON ml_feature_values USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS brin_ml_model_predictions_ts
    ON ml_model_predictions USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS brin_ml_strategy_signals_ts
    ON ml_strategy_signals USING BRIN (ts_event);
```

## Data Processing Pipeline

### DataProcessor Architecture

The DataProcessor (`ml/stores/data_processor.py`) implements a comprehensive data processing pipeline with quality tracking:

#### Quality Flags System

```python
class QualityFlags(IntFlag):
    CLEAN = 0
    MISSING_DATA = 1 << 0
    OUTLIER_DETECTED = 1 << 1
    DUPLICATE = 1 << 2
    STALE_DATA = 1 << 3
    INVALID_RANGE = 1 << 4
    NAN_VALUES = 1 << 5
    INF_VALUES = 1 << 6
    TIMESTAMP_ERROR = 1 << 7
```

#### Processing Stages

1. **Market Data Processing**:
   - Timestamp validation and correction
   - Outlier detection using configurable thresholds
   - Crossed market detection and correction
   - Staleness checking (default 300 seconds)

2. **Feature Processing**:
   - NaN/Inf handling with imputation
   - Range validation and normalization
   - Drift detection and monitoring
   - Lineage tracking for transformations

3. **Prediction Processing**:
   - Confidence calibration
   - Prediction validation
   - Feature attribution tracking

4. **Signal Processing**:
   - Risk metric calculation
   - Execution parameter computation
   - Position sizing logic

### Processing Metrics

```python
@dataclass
class ProcessingMetrics:
    records_processed: int = 0
    records_failed: int = 0
    outliers_removed: int = 0
    duplicates_removed: int = 0
    missing_imputed: int = 0
    processing_time_ms: float = 0.0
    quality_score: float = 1.0
```

### Metadata Caching

The DataProcessor implements intelligent caching for:

- Instrument metadata
- Statistical distributions
- Quality metrics
- With configurable TTL and LRU eviction

**📝 ADDITION:** Centralized Database Engine Management

```python
def create_engine(connection_string: str, **kwargs: Any) -> Engine:
    """Module-level factory for tests to monkeypatch."""
    return EngineManager.get_engine(connection_string, **kwargs)
```

All stores now use centralized EngineManager for connection pooling and lifecycle management.

## 🆕 Message Bus Integration

### Publisher Protocol and Configuration

All stores now support optional message bus publishing for real-time event distribution:

```python
from ml.common.message_bus import MessagePublisherProtocol, NoopPublisher
from ml.common.message_topics import build_topic, map_stage_to_topic_segments
from ml.config.events import Stage

class CustomPublisher(MessagePublisherProtocol):
    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        # Custom implementation (Kafka, Redis, etc.)
        return True

# Store configuration with publishing
store = FeatureStore(
    connection_string=db_url,
    enable_publishing=True,
    publisher=CustomPublisher(),
    publish_mode="both"  # "batch", "row", or "both"
)
```

### Canonical Topic Structure

Events are published to standardized topics based on pipeline stages:

**Topic Format:** `ml.{domain}.{operation}.{instrument_id}`

**Stage Mappings:**

- `Stage.FEATURE_COMPUTED` → `ml.features.updated.{instrument}`
- `Stage.PREDICTION_EMITTED` → `ml.models.created.{instrument}`
- `Stage.SIGNAL_EMITTED` → `ml.strategies.created.{instrument}`
- `Stage.CATALOG_WRITTEN` → `ml.data.updated.{instrument}`

### Publishing Modes

**Batch Publishing (Default):**

```python
# Publishes summary after batch operations
payload = {
    "dataset_id": "features",
    "instrument_id": instrument_id,
    "ts_min": min(timestamps),
    "ts_max": max(timestamps),
    "count": len(batch_data),
    "status": "success"
}
```

**Per-Row Publishing:**

```python
# Publishes individual events for each row
for row in batch_data:
    payload = {
        "dataset_id": "features",
        "instrument_id": row["instrument_id"],
        "ts_event": row["ts_event"],
        "correlation_id": generate_correlation_id(...),
        "status": "success"
    }
```

## 🆕 Event Correlation and Metadata

### Correlation ID System

All stores now support correlation ID tracking for event lineage:

```python
from ml.common.correlation import make_correlation_id

# Deterministic correlation ID generation
correlation_id = make_correlation_id(
    run_id="scheduler_001",
    dataset_id="features_v1",
    instrument_id="EURUSD",
    ts_min=start_time,
    ts_max=end_time,
    count=batch_size
)
```

### Event Metadata Storage

**PostgreSQL Backend:**

- Uses `emit_data_event_ext()` function with JSONB metadata column
- Fallback to legacy `emit_data_event()` when extended function unavailable
- Migration `007_add_event_metadata.sql` adds metadata support

**JSON Backend:**

- Metadata stored directly in event dictionaries
- Full metadata preservation with schema validation

### Cross-Domain Event Cascades

Stores can emit cascaded events across domains:

```python
from ml.common.cascade import emit_cascade, EventDict

# Source event from data ingestion
source_event = EventDict(
    domain="data",
    event_type="catalog_written",
    correlation_id="abc123",
    instrument_id="EURUSD",
    ts_event=1693747200000000000,
)

# Cascade to features domain
feature_event = emit_cascade(
    source_event,
    target_domain="features",
    delay_ns=1000000  # 1ms delay
)
```

## Live Data Recording

### LiveDataRecorder **📝 ADDITION:** (`ml/stores/live_data_recorder.py`)

The LiveDataRecorder automatically captures all live market data flowing through the system:

```python
class LiveDataRecorder:
    """Automatically records all live data with validation and event tracking."""

    def __init__(
        self,
        data_store: DataStore,
        data_registry: DataRegistry,
        buffer_size: int = 1000,
        flush_interval_ms: int = 1000,
        storage_path: Path | None = None,
    ):
        # Intercepts and records quotes, trades, bars
        # Automatic buffering and periodic flushing
        # Event emission and watermark tracking
```

#### Key Features

- **Automatic Interception**: Captures all market data types (quotes, trades, bars)
- **Buffered Writing**: Configurable buffer size with periodic flushing
- **Event Tracking**: Emits events to DataRegistry for observability
- **Async Support**: Non-blocking recording with async flush tasks
- **Metadata Tracking**: Records instrument IDs, timestamps, and counts

## Advanced Partitioning Management

### PartitionManager **📝 ADDITION:** (`ml/stores/partition_manager.py`)

Located in `ml/stores/partition_manager.py`:

#### Key Features

- **Automatic Creation**: Creates future partitions (3 months ahead configurable)
- **Cleanup**: Removes old partitions (24 months retention configurable)
- **Statistics**: Provides partition size and count statistics
- **Health Checks**: Ensures current month partition exists

#### Usage Patterns

```python
manager = PartitionManager(
    connection_string="postgresql://user:pass@localhost:5432/nautilus",
    months_ahead=3,
    retention_months=24
)

# Run full maintenance
results = manager.run_maintenance()
# Returns: {"created": 2, "removed": 1, "timestamp": "2024-03-15T10:30:00"}

# Get statistics
stats = manager.get_partition_stats()
# Returns partition sizes and counts by table
```

### Automated Scheduling

The scheduler (`ml/stores/schedule_partitions.py`) provides:

#### Command-Line Interface

```bash
# Single run
python schedule_partitions.py

# Daemon mode with hourly checks
python schedule_partitions.py --daemon --interval 3600

# Dry run to preview actions
python schedule_partitions.py --dry-run

# Statistics only
python schedule_partitions.py --stats-only
```

#### Cron Integration

```bash
# Daily maintenance at 2 AM
0 2 * * * /usr/bin/python3 /path/to/schedule_partitions.py
```

## Query Optimization Patterns

### Time-Range Queries

Optimized for partition pruning:

```python
# FeatureStore training data retrieval
features, timestamps, names = feature_store.get_training_data(
    instrument_id="EURUSD.IDEALPRO",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 1),
    include_bars=True
)

# ModelStore prediction analysis
predictions_df = model_store.read_predictions(
    model_id="xgboost_v1",
    instrument_id="AAPL.NASDAQ",
    start_ns=start_timestamp_ns,
    end_ns=end_timestamp_ns
)

# StrategyStore signal analysis
signals_df = strategy_store.read_signals(
    strategy_id="momentum_v1",
    instrument_id="BTCUSD.COINBASE",
    start_ns=start_timestamp_ns,
    end_ns=end_timestamp_ns
)
```

### Performance Methods

Each store provides performance analytics:

```python
# Model performance metrics
perf = model_store.get_model_performance(
    model_id="xgboost_v1",
    start_ns=start_ns,
    end_ns=end_ns
)
# Returns: prediction_count, avg_confidence, latency percentiles

# Strategy performance metrics
perf = strategy_store.get_strategy_performance(
    strategy_id="momentum_v1",
    start_ns=start_ns,
    end_ns=end_ns
)
# Returns: signal counts by type, strength statistics
```

## DataStore Contract Validation

### Validation Framework

The DataStore provides comprehensive contract-based validation:

```python
# Preflight schema validation
success, error, details = store.preflight_check(
    dataset_id="bars_eurusd_1m",
    data=df,
    strict=True
)

# Contract validation with quality reporting
report = store.validate_batch(
    dataset_id="bars_eurusd_1m",
    data=df,
    strict_mode=False
)
```

### Validation Rules

DataStore supports multiple validation rule types:

```python
class ValidationRuleType(Enum):
    TYPE_CHECK = "type_check"      # Schema type validation
    RANGE = "range"                # Min/max value constraints
    UNIQUENESS = "uniqueness"       # Primary key constraints
    MONOTONICITY = "monotonicity"   # Timestamp ordering
    NULLABILITY = "nullability"     # Required field checks
    LATENESS = "lateness"          # Data freshness validation
```

### Quality Reporting **✨ ENHANCEMENT:**

```python
@dataclass
class QualityReport:
    dataset_id: str
    total_records: int
    passed_records: int
    failed_records: int
    quality_score: float  # 0-1 score
    violations: list[ValidationViolation]
    validation_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)  # **📝 ADDITION:** Metadata support

@dataclass
class ValidationViolation:
    rule_type: ValidationRuleType
    field_name: str
    severity: QualityFlag
    violation_count: int
    sample_values: list[Any]
    description: str
```

### Enforcement Modes

- **strict**: Reject any data with violations
- **lenient**: Log warnings but allow data through
- **monitor_only**: Track violations without blocking

**📝 ADDITION:** DataStore includes fail-closed behavior for critical violations regardless of enforcement mode.

## Cache Implementation

### Metadata Caching

The DataProcessor implements sophisticated caching:

```python
class DataProcessor:
    def __init__(self, enable_caching: bool = True):
        self._metadata_cache: dict[str, Any] = {}
        self._statistics_cache: dict[str, Any] = {}
        self._cache_timestamp: int = 0

    def _get_instrument_metadata(self, instrument_id: str) -> dict[str, Any]:
        cache_key = f"instrument:{instrument_id}"
        if self.enable_caching and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        # ... database query and cache update
```

### Cache Invalidation Strategy

- **Time-based**: 5-minute TTL for statistics
- **Event-based**: Metadata cached until explicit invalidation
- **Memory management**: LRU-style cleanup for large caches

## Integration Points

### Actor Integration

All ML actors must inherit from BaseMLInferenceActor:

```python
from ml.actors.base import BaseMLInferenceActor

class CustomMLActor(BaseMLInferenceActor):
    def __init__(self, config):
        super().__init__(config)
        # self.feature_store, self.model_store, self.strategy_store
        # are automatically initialized

    def on_bar(self, bar):
        # Compute features
        features = self.feature_store.compute_realtime(bar, store=True)

        # Make prediction
        prediction = self.model.predict(features)
        self.model_store.write_prediction(...)

        # Generate signal
        signal = self.generate_signal(prediction)
        self.strategy_store.write_signal(...)
```

### Registry Integration

Stores integrate with the registry system:

```python
# Feature computation uses registry metadata
feature_manifest = self.feature_registry.get_manifest(feature_set_id)
features = self.feature_store.compute_and_store_historical(
    instrument_id, start, end,
    feature_config=feature_manifest.config
)

# Model predictions use registry versions
model_manifest = self.model_registry.get_manifest(model_id)
self.model_store.write_prediction(
    model_id=f"{model_id}_{model_manifest.version}",
    # ...
)
```

### Persistence Manager Integration

Stores use the unified persistence layer:

```python
from ml.registry.persistence import PersistenceManager, PersistenceConfig

persistence_config = PersistenceConfig(
    backend=BackendType.POSTGRES,
    connection_string="postgresql://..."
)

model_store = ModelStore(persistence_config=persistence_config)
strategy_store = StrategyStore(persistence_config=persistence_config)
```

**📝 ADDITION:** Message Bus Integration

```python
# Stores support configurable message bus publishing
store = ModelStore(
    connection_string=conn_str,
    enable_publishing=True,
    publisher=message_publisher
)

# Events published to topics derived from processing stage
# Topic format: {domain}.{operation}.{instrument_id}
# Example: "ml.prediction.emitted.EURUSD"
```

## Performance Characteristics

### Write Performance

Optimized batch writing with configurable parameters:

```python
# Batch size tuning
model_store = ModelStore(
    batch_size=1000,           # Auto-flush at 1000 records
    flush_interval_ms=100      # Auto-flush every 100ms
)

# Manual flushing for low-latency scenarios
model_store.write_prediction(...)
model_store.flush()  # Force immediate write
```

### Query Performance

#### Time-Series Optimizations

- **Partition Pruning**: Queries automatically target specific monthly partitions
- **BRIN Indexes**: Block-range indexes for timestamp columns provide compression **📝 ADDITION:**
- **Connection Pooling**: Reuse connections for reduced latency via EngineManager **✨ ENHANCEMENT:**

#### Typical Performance

- **Single prediction write**: < 1ms
- **Batch write (1000 records)**: < 50ms
- **Time-range query (1 day)**: < 100ms
- **Cross-partition query (1 month)**: < 500ms

### Memory Management

#### Buffer Management

```python
# Write buffers auto-managed
self._write_buffer: list[ModelPrediction] = []
self._last_flush_ns = 0

# Automatic flushing logic
if len(self._write_buffer) >= self.batch_size:
    self.flush()
elif self.clock and self._should_flush_by_time():
    self.flush()
```

#### Connection Management

- **Engine per store**: Separate connection pools via EngineManager **✨ ENHANCEMENT:**
- **Auto-reconnection**: Handles database disconnections gracefully
- **Transaction boundaries**: Each batch write is atomic

## Current Implementation Status

### Completed Components ✅

1. **Core Store Classes**: FeatureStore, ModelStore, StrategyStore, DataStore fully implemented
2. **Database Schema**: Complete migration system with partitioning (9 migration files) **🔄 UPDATE:**
3. **Data Processing**: Comprehensive DataProcessor with quality tracking and validation
4. **Contract Validation**: DataStore with preflight checks, quality reporting, enforcement modes
5. **Live Recording**: LiveDataRecorder for automatic market data capture
6. **Partition Management**: Automatic creation, cleanup, and monitoring
7. **Integration Layer**: DataRegistry integration, PersistenceConfig support
8. **Performance Optimizations**: Batch writing, intelligent caching, optimized indexing
9. **Event Tracking**: Full event emission and watermark management
10. **📝 ADDITION:** Protocol-based interfaces for type safety and testing
11. **📝 ADDITION:** Centralized database engine management via EngineManager
12. **📝 ADDITION:** Message bus integration for real-time event publishing
13. **📝 ADDITION:** Advanced preflight validation with comprehensive type checking

### Production Readiness ✅

- **Type Safety**: Complete type annotations with strict mypy compliance
- **Error Handling**: Comprehensive exception handling and logging
- **Testing**: Unit tests covering core functionality with protocol-based mocking **✨ ENHANCEMENT:**
- **Configuration**: Environment-based configuration support
- **Monitoring**: Health checks, statistics endpoints, Prometheus metrics
- **Schema Migration**: Support for dual-write windows during migrations

### Advanced Features ✅

1. **Contract-Based Validation**: Full implementation in DataStore
2. **Quality Scoring**: Comprehensive quality metrics and reporting
3. **Event-Driven Architecture**: DataRegistry integration for observability
4. **Async Support**: LiveDataRecorder with async flush tasks
5. **Schema Evolution**: Migration windows and version tracking
6. **📝 ADDITION:** BRIN indexes for efficient time-range queries
7. **📝 ADDITION:** Disabled partition triggers for race condition prevention
8. **📝 ADDITION:** Enhanced event metadata support with JSONB storage

### Planned Enhancements 🔄

1. **Streaming Ingestion**: Real-time market data integration (partially complete via LiveDataRecorder)
2. **A/B Testing Framework**: Model comparison infrastructure
3. **Advanced Analytics**: Complex cross-store queries and aggregations
4. **Distributed Processing**: Horizontal scaling support

### Implemented in Other Modules ✅

1. **Circuit Breaker**: Fully implemented in `ml/actors/base.py:199-289` with complete state machine, fault tolerance, and recovery mechanisms. Used by all ML actors for production resilience.

## Critical Design Decisions

### 1. Nautilus Timestamp Convention

All timestamps use nanoseconds since epoch (int64) following Nautilus conventions:

```python
# Always nanoseconds for consistency with Nautilus core
ts_event: int  # Event timestamp in nanoseconds
ts_init: int   # Initialization timestamp in nanoseconds

# Conversion utilities
ts_event = int(datetime.utcnow().timestamp() * 1e9)

# **✨ ENHANCEMENT:** Defensive normalization with warning logging
# The stores defensively normalize incoming timestamps to nanoseconds when they appear
# to be in seconds, milliseconds, or microseconds. This is a safety net for integration
# edges; a warning is logged when normalization occurs. Production writers should still
# emit nanoseconds directly.
```

### 2. JSONB for Flexible Storage

Feature values, model parameters, and execution parameters use PostgreSQL JSONB:

```sql
-- Flexible schema evolution
values JSONB NOT NULL,          -- Feature name -> value mapping
features_used JSONB,            -- Features at prediction time
execution_params JSONB,         -- Stop loss, take profit, etc.
```

**Rationale**: Enables schema evolution without migrations while maintaining query performance.

### 3. Time-Based Partitioning

Monthly partitions balance query performance with maintenance overhead:

```sql
-- Partition boundaries aligned to calendar months
FOR VALUES FROM (start_ns) TO (end_ns)
-- Where start_ns/end_ns are first-of-month timestamps
```

**Rationale**: Monthly partitions provide optimal balance of query performance, partition count, and maintenance complexity.

### 4. Batch Writing with Auto-Flush

Batch writes optimize database performance while ensuring data durability:

```python
# Configurable batching
self._write_buffer: list[ModelPrediction] = []
if len(self._write_buffer) >= self.batch_size:
    self.flush()
```

**Rationale**: Reduces database round-trips while providing predictable flush timing for low-latency requirements.

### 5. Quality-First Data Processing

Comprehensive validation ensures data integrity:

```python
# 8-bit quality flag system
quality_flags = QualityFlags.CLEAN
if np.isnan(value):
    quality_flags |= QualityFlags.NAN_VALUES
    cleaned_value = 0.0  # Imputation strategy
```

**Rationale**: Explicit quality tracking enables debugging, monitoring, and data lineage analysis.

**📝 ADDITION:** 6. Centralized Engine Management

```python
# All stores use EngineManager for connection lifecycle
def create_engine(connection_string: str, **kwargs: Any) -> Engine:
    return EngineManager.get_engine(connection_string, **kwargs)
```

**Rationale**: Centralizes connection pooling, reduces resource usage, and simplifies testing through unified mock points.

**📝 ADDITION:** 7. Protocol-First Design

```python
# Structural typing prevents implementation drift
class ModelStoreProtocol(Protocol):
    def write_prediction(...) -> None: ...
    def read_predictions(...) -> Any: ...
```

**Rationale**: Enables duck typing, simplifies testing with DummyStore, and prevents API drift between implementations.

## Configuration and Deployment

### Environment Configuration

```bash
# Database connection
ML_DB_CONNECTION=postgresql://user:pass@localhost:5432/nautilus

# Store performance tuning
ML_STORE_BATCH_SIZE=1000
ML_STORE_FLUSH_INTERVAL_MS=100

# Partition management
ML_PARTITION_MONTHS_AHEAD=3
ML_PARTITION_RETENTION_MONTHS=24

# Data processing
ML_OUTLIER_THRESHOLD=5.0
ML_STALENESS_THRESHOLD_SECONDS=300

# **📝 ADDITION:** Message bus configuration
ML_ENABLE_PUBLISHING=true
ML_MESSAGE_BUS_URL=redis://localhost:6379
```

### PostgreSQL Tuning

Recommended settings for time-series workloads:

```postgresql.conf
# Memory settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 10MB
maintenance_work_mem = 512MB

# Write performance
wal_buffers = 64MB
checkpoint_segments = 64
checkpoint_completion_target = 0.9

# Partitioning optimization
enable_partition_pruning = on
enable_partitionwise_aggregate = on
enable_partitionwise_join = on

# Time-series specific
random_page_cost = 1.1  # Assumes SSD storage
seq_page_cost = 1.0
```

### Deployment Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        MA[ML Actors] --> FS[FeatureStore]
        MA --> MS[ModelStore]
        MA --> SS[StrategyStore]
        MA --> DS[DataStore]
    end

    subgraph "Processing Layer"
        FS --> DP[DataProcessor]
        MS --> DP
        SS --> DP
        DS --> DP
    end

    subgraph "Storage Layer"
        DP --> PG[(PostgreSQL)]
        PM[PartitionManager] --> PG
        EM[EngineManager] --> PG
    end

    subgraph "Management Layer"
        SC[Scheduler] --> PM
        MON[Monitoring] --> PG
        LDR[LiveDataRecorder] --> DS
    end

    subgraph "Integration Layer"
        MB[MessageBus] --> MA
        DR[DataRegistry] --> DS
    end
```

**✨ ENHANCEMENT:** Added EngineManager, MessageBus, and LiveDataRecorder to architecture diagram.

## Future Enhancements

### Near-Term (Next 3 months)

1. **Store-Level Circuit Breakers**: While actors have circuit breakers, add dedicated database-level circuit breakers
2. **Streaming Ingestion**: Real-time market data processing (partially complete via LiveDataRecorder)
3. **Advanced Analytics**: Cross-store query optimization
4. **Monitoring Dashboard**: Grafana integration for operations

### Medium-Term (6 months)

1. **Multi-Region Support**: Database replication and failover
2. **A/B Testing Framework**: Model comparison infrastructure
3. **Data Lake Integration**: Cold storage for historical data
4. **Machine Learning Pipelines**: Automated feature engineering

### Long-Term (12 months)

1. **Real-Time Feature Serving**: Sub-millisecond feature retrieval
2. **Advanced ML Lifecycle**: Automated model deployment and monitoring
3. **Distributed Processing**: Horizontal scaling for large datasets
4. **Advanced Risk Management**: Portfolio-level risk analytics

## Conclusion

The ML stores infrastructure provides a production-ready, scalable foundation for machine learning data management in Nautilus Trader. The four-store architecture (FeatureStore, ModelStore, StrategyStore, DataStore), sophisticated partitioning system, contract-based validation, and comprehensive data processing ensure enterprise-grade reliability and performance.

Key strengths:

- **Mandatory Integration**: All ML actors must use the four-store architecture **🔄 UPDATE:**
- **Contract Validation**: DataStore provides schema validation, quality scoring, and enforcement modes
- **Sophisticated Partitioning**: Automatic partition management with PostgreSQL native features
- **Comprehensive Processing**: Quality tracking, validation, enrichment, and event emission
- **Performance Optimized**: Batch writing, intelligent caching, optimized indexing, async support
- **Production Ready**: Complete error handling, monitoring, Prometheus metrics, and configuration
- **Event-Driven**: Full integration with DataRegistry for observability and watermark tracking
- **Live Recording**: Automatic capture of all market data with LiveDataRecorder
- **✨ ENHANCEMENT:** Protocol-based interfaces for type safety and testing compatibility
- **✨ ENHANCEMENT:** Centralized engine management for resource optimization
- **✨ ENHANCEMENT:** Message bus integration for real-time event distribution
- **📝 ADDITION:** Advanced preflight validation with comprehensive type checking
- **📝 ADDITION:** BRIN indexes for efficient time-range query performance
- **📝 ADDITION:** Enhanced event metadata support for better observability

The implementation successfully provides a robust, validated, and observable foundation for both research and production ML trading systems, with comprehensive data quality controls and full lifecycle management.

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
