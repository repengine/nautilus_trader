# Context: Observability Infrastructure

**Last Updated**: 2025-01-19
**Module Size**: 3,032 lines across 13 files
**Primary Location**: `ml/observability/`
**Database Tables**: 4 tables with PostgreSQL partitioning support

## Purpose and Scope

The observability infrastructure provides **cold-path-only** telemetry collection, persistence, and analysis capabilities for the ML pipeline. It operates strictly off the hot path with zero performance impact on real-time inference, collecting comprehensive system metrics, latency watermarks, event correlation data, and health scores for offline analysis and debugging.

**Key Principle**: All observability operations are deferred, batched, and executed in background threads/tasks. Hot-path actors only enqueue lightweight events; heavy DataFrame construction and I/O occur off-path.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hot Path (ML Actors)                         │
│           ┌──────────────────────────────────────┐              │
│           │ Non-blocking enqueue only           │              │
│           │ (correlation IDs, timestamps)        │              │
│           │ Drops on backpressure with metrics  │              │
│           └──────────────────────────────────────┘              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              ObservabilityService (Cold Path)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Row Collection│  │DTO Builders │  │ DataFrame    │          │
│  │ (Lightweight)│  │ (Transform)  │  │ Generation   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Async Worker Layer (Optional)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Bounded Queue│  │ Batch Drain  │  │ Backpressure │          │
│  │ (4096 items) │  │ (256/cycle)  │  │ Drop Metrics │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Persistence Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ File Sink    │  │ DB Sink      │  │ Async DB     │          │
│  │ (JSONL/CSV)  │  │ (SQLAlchemy) │  │ (asyncpg)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Daily Rotate │  │ Partitioning │  │ Retention    │          │
│  │ Compaction   │  │ BRIN Indices │  │ Management   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### ObservabilityService (service.py - 153 lines)

**Location**: `ml/observability/service.py`

Central façade for collecting observability rows and materializing contract-compliant DataFrames. The service is deliberately minimal and type-safe, maintaining four in-memory row collections:

**Data Collection Methods** (Lines 44-124):
- `add_latency_stage()`: Record pipeline stage timing with correlation_id, instrument_id, pipeline_stage, ts_stage_start, ts_stage_end
- `add_metric()`: Record metric observation with name, type, value, timestamp, labels (dict or JSON string)
- `add_correlation()`: Record event lineage with correlation_id, event_id, parent_event_id, domain, lineage_depth, propagation_path
- `add_health()`: Record component health with component_id, health_score, subsystem_scores, timestamp, measurement_window_ms

**DataFrame Materialization Methods** (Lines 128-150):
- `latency_watermarks_df()`: Builds DataFrame with stage_latency_ns and cumulative_latency_ns columns
- `metrics_collection_df()`: Builds DataFrame with JSON-encoded labels and normalized types
- `event_correlation_df()`: Builds DataFrame with JSON-encoded propagation_path
- `health_scores_df()`: Builds DataFrame with clamped scores [0,1] and default alert_threshold

**Design Notes**:
- Service does NOT persist; callers own storage
- Not thread-safe; use separate instances per thread
- Unbounded memory growth until flushed (production must configure flush intervals)

### Pipeline DTO Builders (pipeline.py - 205 lines)

**Location**: `ml/observability/pipeline.py`

Typed builders that transform raw observability rows into pandas DataFrames with schema compliance and data normalization. These builders are the single source of truth for observability DataFrame schemas.

#### build_latency_watermarks() (Lines 20-48)

Transforms latency stage rows into watermark DataFrame:

**Input Fields**:
- correlation_id (str)
- instrument_id (str)
- pipeline_stage (str)
- ts_stage_start (int - nanoseconds)
- ts_stage_end (int - nanoseconds)

**Computed Fields**:
- `stage_latency_ns`: (ts_stage_end - ts_stage_start).clip(lower=0)
- `cumulative_latency_ns`: cumulative sum of stage_latency_ns in input order

**Edge Cases**:
- Empty input returns empty DataFrame with proper dtypes (int64 for latency columns)
- Negative latencies clipped to 0 for robustness

#### build_metrics_collection() (Lines 51-75)

Normalizes metric rows with type safety:

**Input Fields**:
- metric_name (str)
- metric_type (str): counter, histogram, gauge, summary
- value (float)
- timestamp (int - nanoseconds)
- labels (dict[str, Any] | str)

**Transformations**:
- Labels normalized to JSON string via `json.dumps()` if not already string
- Value cast to float64
- Timestamp cast to int64

#### build_event_correlation() (Lines 78-101)

Builds event lineage DataFrame:

**Input Fields**:
- correlation_id, event_id, parent_event_id (nullable), instrument_id
- domain (str): data, features, models, strategies
- lineage_depth (int >= 0)
- ts_event (int - nanoseconds)
- propagation_path (list[str] | str)

**Transformations**:
- propagation_path normalized to JSON string
- lineage_depth clipped to >= 0
- Preserves event ordering for lineage reconstruction

#### build_health_scores() (Lines 104-133)

Builds health aggregation DataFrame:

**Input Fields**:
- component_id (str)
- health_score (float)
- subsystem_scores (dict[str, float] | str)
- timestamp (int - nanoseconds)
- measurement_window_ms (int)

**Transformations**:
- health_score clamped to [0.0, 1.0]
- subsystem_scores JSON-encoded if dict
- Adds default alert_threshold=0.8 if missing (for schema compliance)

#### Aggregation Helpers (Lines 144-205)

**aggregate_metrics_by_window()**: Groups metric rows by fixed windows (window_ns parameter), aggregating by (metric_name, domain, instrument_id, window_start) with total_value (sum) and sample_count (count). Drops labels to avoid cardinality explosion.

**scale_health_scores()**: Uniformly scales health_score by factor and clips to [0,1], useful for normalization and threshold adjustment.

### Persistence Layer

#### ObservabilityPersistor (persistence.py - 136 lines)

**Location**: `ml/observability/persistence.py`

File-based persistence for observability DataFrames with rotation and compaction support.

**Configuration** (Lines 21-49):
- `base_path` (Path): Output directory
- `file_format` (str): "jsonl" (default) or "csv"
- `rotate_daily` (bool): Enable daily rotation by timestamp
- `max_file_bytes` (int | None): Size-based rotation threshold

**Methods**:

**persist()** (Lines 51-100):
- Writes non-empty DataFrames to disk
- Creates directories with `parents=True, exist_ok=True`
- Returns dict[str, Path] mapping table name to written file
- Supports daily rotation: extracts timestamp from DataFrame, creates subdirectory YYYY-MM-DD
- Supports size-based rotation: appends timestamp tag if file exceeds max_file_bytes
- JSONL uses `df.to_json(orient="records", lines=True)`
- CSV uses `df.to_csv(index=False)`

**compact_daily()** (Lines 102-133):
- Compacts per-table JSONL shards for a given day into single files
- Writes to `<base_path>/<day>/compacted/<name>.jsonl`
- Only applicable to JSONL format; CSV users rely on external compaction

**Time Column Mapping** (Lines 42-48):
```python
_time_cols = {
    "latency": "ts_stage_end",
    "metrics": "timestamp",
    "correlation": "ts_event",
    "health": "timestamp",
}
```

#### ObservabilityDBPersistor (db_persistence.py - 213 lines)

**Location**: `ml/observability/db_persistence.py`

SQL database persistence using SQLAlchemy engines provisioned by EngineManager.

**Database Schema** (Lines 56-102):

**obs_latency_watermarks**:
- correlation_id: String(64), NOT NULL
- instrument_id: String(100), NOT NULL
- pipeline_stage: String(64), NOT NULL
- ts_stage_start: BIGINT, NOT NULL
- ts_stage_end: BIGINT, NOT NULL
- stage_latency_ns: BIGINT, NOT NULL
- cumulative_latency_ns: BIGINT, NOT NULL

**obs_metrics**:
- metric_name: String(128), NOT NULL
- metric_type: String(32), NOT NULL
- value: FLOAT, NOT NULL
- timestamp: BIGINT, NOT NULL
- labels: NVARCHAR(4096), nullable

**obs_event_correlation**:
- correlation_id: String(64), NOT NULL
- event_id: String(64), NOT NULL
- parent_event_id: String(64), nullable
- instrument_id: String(100), NOT NULL
- domain: String(32), NOT NULL (data/features/models/strategies)
- lineage_depth: INTEGER, NOT NULL
- ts_event: BIGINT, NOT NULL
- propagation_path: NVARCHAR(4096), nullable

**obs_health_scores**:
- component_id: String(64), NOT NULL
- health_score: FLOAT, NOT NULL
- subsystem_scores: NVARCHAR(4096), nullable
- timestamp: BIGINT, NOT NULL
- measurement_window_ms: INTEGER, NOT NULL
- alert_threshold: FLOAT, NOT NULL

**Methods**:

**persist()** (Lines 107-147):
- Writes non-empty DataFrames to corresponding tables
- Uses `df.to_sql()` with `if_exists="append"`, `method="multi"` for batch inserts
- Returns dict[str, int] mapping table name to row count inserted
- Transactional: all tables written within `engine.begin()` context

**apply_retention()** (Lines 149-213):
- Deletes rows older than retention_days based on table-specific timestamp columns
- Uses sanitize_timestamp_ns for cutoff calculation
- Returns dict[str, int] mapping table name to rows deleted
- Gracefully handles missing tables (records 0 deletions)

**Time Column Retention Mapping** (Lines 192-197):
```python
retention_tables = {
    "obs_latency_watermarks": "ts_stage_end",
    "obs_metrics": "timestamp",
    "obs_event_correlation": "ts_event",
    "obs_health_scores": "timestamp",
}
```

#### ObservabilityAsyncDBPersistor (async_db_persistence.py - 116 lines)

**Location**: `ml/observability/async_db_persistence.py`

Optional asyncio-based persistence using SQLAlchemy async engines for high-throughput scenarios.

**Requirements**:
- Async SQLAlchemy driver (sqlite+aiosqlite:// or postgresql+asyncpg://)
- Falls back gracefully if async engine unavailable

**persist_async()** (Lines 45-113):
- Async version of persist() using `AsyncConnection.run_sync()`
- Bridges pandas `to_sql()` (synchronous) with SQLAlchemy async drivers
- Returns dict[str, int] mapping table name to row count
- Disposes engine after write: `await engine.dispose()`

### Async Worker Layer

#### ObservabilityAsyncWorker (async_worker.py - 444 lines)

**Location**: `ml/observability/async_worker.py`

Asyncio-based background worker with bounded queue for high-throughput observability collection.

**Configuration** (Lines 80-111):
- `service` (ObservabilityService): In-memory row collector
- `sink` (Literal["file", "db"]): Persistence destination
- `base_path` (Path | None): Required for file sink
- `db_connection_string` (str | None): Required for DB sink
- `flush_interval_seconds` (float): Default 5.0
- `queue_maxsize` (int): Default 4096
- `component_label` (str): Default "obs_async_worker"
- `use_async_db` (bool): Enable async DB persistence

**Metrics** (Lines 119-140):
- `nautilus_ml_observability_enqueued_total{kind}`: Items enqueued by type
- `nautilus_ml_observability_async_flush_duration_seconds{sink}`: Flush latency histogram
- `nautilus_ml_observability_queue_depth{component}`: Current queue size gauge
- `nautilus_ml_observability_errors_total{component,kind}`: Error counter
- `nautilus_ml_backpressure_drops_total{component,reason}`: Queue full drops

**Queue Item Types** (Lines 38-77):
- `_LatencyItem`: correlation_id, instrument_id, pipeline_stage, ts_stage_start, ts_stage_end
- `_MetricItem`: metric_name, metric_type, value, timestamp, labels
- `_CorrelationItem`: correlation_id, event_id, parent_event_id, instrument_id, domain, lineage_depth, ts_event, propagation_path
- `_HealthItem`: component_id, health_score, subsystem_scores, timestamp, measurement_window_ms

**Enqueue API** (Lines 190-274):
- `enqueue_latency()`: Non-blocking, returns bool (True if enqueued, False if dropped)
- `enqueue_metric()`: Non-blocking with labels normalization
- `enqueue_correlation()`: Non-blocking with lineage tracking
- `enqueue_health()`: Non-blocking with subsystem scores

**Worker Loop** (Lines 300-444):
- Drains queue in batches (max 256 items per cycle) with 0.05s timeout
- Periodic flush when `now - last_flush >= flush_interval_seconds`
- Supports sync DB/file flush via `asyncio.to_thread()`
- Supports async DB flush via `ObservabilityAsyncDBPersistor`
- Graceful error handling: logs errors, increments metrics, continues processing
- Updates queue depth gauge after each drain cycle

**Backpressure Handling** (Lines 278-298):
- `_try_put()`: Uses `queue.put_nowait()` to avoid blocking
- On `asyncio.QueueFull`: emits `nautilus_ml_backpressure_drops_total` metric, returns False
- Never raises from hot path; always returns success/failure status

#### ObservabilityFlusher (scheduler.py - 125 lines)

**Location**: `ml/observability/scheduler.py`

Thread-based background scheduler for periodic persistence (simpler alternative to async worker).

**Configuration** (Lines 25-42):
- `service` (ObservabilityService): Row collector
- `base_path` (Path): Output directory
- `file_format` (str): "jsonl" or "csv", default "jsonl"
- `interval_seconds` (float): Flush interval, default 60.0
- `now` (Callable[[], float]): Time function (mockable for testing)
- `sink` (str): "file" or "db"
- `db_connection_string` (str | None): For DB sink

**Methods**:

**flush_once()** (Lines 44-65):
- Materializes all four DataFrames via service
- Writes to file or DB sink based on configuration
- Updates `_last_flush` timestamp
- Returns dict[str, Path] for file sink or dict[str, int] for DB sink

**tick()** (Lines 67-76):
- Flushes if `interval_seconds <= 0` (immediate mode)
- Flushes if `now() - _last_flush >= interval_seconds`
- Returns consistent dict type based on sink

**start_background()** (Lines 78-122):
- Starts daemon thread that ticks until `stop_event.is_set()`
- Graceful error handling: catches exceptions, logs via debug, increments error metric
- Sleeps `max(0.01, min(0.5, interval_seconds))` between ticks to avoid busy spinning
- Returns threading.Thread for caller to join

### Database Migrations (migrations.py - 249 lines)

**Location**: `ml/observability/migrations.py`

PostgreSQL-specific performance optimizations and partitioning support.

**Table Configuration** (Lines 12-17):
```python
OBS_TABLES: Final[dict[str, str]] = {
    "obs_latency_watermarks": "ts_stage_end",
    "obs_metrics": "timestamp",
    "obs_event_correlation": "ts_event",
    "obs_health_scores": "timestamp",
}
```

#### apply_observability_indices() (Lines 125-152)

Creates BRIN and composite indices for PostgreSQL (no-op on other backends):

**BRIN Indices** (Lines 138):
- `{table}_{ts_col}_brin`: BRIN index on timestamp column for each table
- BRIN (Block Range INdexes) are extremely compact for append-heavy workloads with timestamp ordering
- Ideal for observability data which is typically time-ordered

**Composite Indices** (Lines 140-151):
- `obs_event_correlation_instrument_ts_idx`: (instrument_id, ts_event) for instrument-specific queries
- `obs_metrics_name_ts_idx`: (metric_name, timestamp) for metric time-series queries

**Safe Idempotency**: Uses `CREATE INDEX IF NOT EXISTS` within DO blocks

#### apply_observability_monthly_partitions() (Lines 242-249)

Applies monthly range partitioning to all observability tables (PostgreSQL only).

**Strategy**:
- Calls `ensure_monthly_partitions()` for each table/timestamp column pair
- Enables efficient data lifecycle management and query performance

#### ensure_monthly_partitions() (Lines 168-239)

**Partitioning Logic**:

1. **Table Existence Check** (Lines 182-186): Uses `to_regclass()` to check if table exists

2. **Partitioning Status Check** (Lines 188-203): Queries `pg_partitioned_table` to determine if table is already partitioned

3. **Migration Path for Existing Tables** (Lines 204-212):
   - If table exists but is NOT partitioned and is EMPTY: drops table CASCADE and recreates as partitioned parent
   - If table has data: no-op (avoids complex migration)
   - Uses `_drop_table_cascade()` and `_create_partitioned_parent()`

4. **Partition Creation** (Lines 220-239):
   - Creates partitions for current month and next month
   - Uses `_month_bounds()` to compute month start/end timestamps
   - Partition naming: `{table}_{YYYY}_{MM}` (e.g., obs_metrics_2025_01)
   - Each partition gets its own BRIN index on the timestamp column

**Helper Functions**:
- `_create_index()` (Lines 20-62): Safe index creation with identifier quoting and USING clause validation
- `_create_partitioned_parent()` (Lines 65-71): Creates PARTITION BY RANGE table
- `_drop_table_cascade()` (Lines 74-88): Safe table drop with CASCADE
- `_create_partition()` (Lines 91-122): Creates range partition with bounds
- `_month_bounds()` (Lines 157-165): Computes month start/end datetimes in UTC

**Timestamp Sanitization** (Lines 228-237):
- Uses `ml.common.timestamps.sanitize_timestamp_ns()` to validate partition bounds
- Ensures partition bounds are valid nanosecond timestamps
- Provides context strings for debugging

### Distributed Tracing (tracing.py - 564 lines)

**Location**: `ml/observability/tracing.py`

Optional OpenTelemetry integration for distributed tracing (cold-path only).

**Design Principles** (Lines 1-98):
1. **OFF by default** with zero overhead when disabled
2. **Cold-path only** - never used in hot-path/real-time code
3. **Optional dependency** - graceful fallback when OpenTelemetry unavailable
4. **W3C compliant** - propagates trace context via standard headers
5. **Correlation integrated** - works alongside existing correlation_id system

**Environment Configuration** (Lines 19-23):
- `ML_TRACING_ENABLED`: "true" to enable (default: "false")
- `ML_TRACING_SERVICE_NAME`: Service name (default: "nautilus-ml")
- `ML_TRACING_ENDPOINT`: OTLP endpoint URL (default: None = no export)
- `ML_TRACING_SAMPLE_RATE`: Sampling rate 0.0-1.0 (default: 0.1)

**Lazy Initialization** (Lines 139-208):
- `_ensure_tracing_backend()`: Lazy imports OpenTelemetry only when enabled
- Configures TracerProvider with sampling, resource attributes, OTLP exporter
- Caches _tracer, _context, _propagate modules for reuse
- Graceful fallback on any initialization error

**API Functions**:

**trace_cold_path()** (Lines 243-298):
- Context manager for tracing cold-path operations
- Creates span with operation_name, correlation_id, custom attributes
- No-op when tracing disabled (zero overhead)
- Example:
  ```python
  with trace_cold_path("feature_computation", correlation_id="abc123") as span:
      if span:
          span.set_attribute("instrument_id", "EUR/USD")
      features = compute_features(data)
  ```

**trace_cold_path_decorator()** (Lines 301-359):
- Decorator for automatic span creation around function execution
- Extracts correlation_id from function kwargs if `correlation_id_param` specified
- Zero overhead when disabled
- Example:
  ```python
  @trace_cold_path_decorator("model_training", correlation_id_param="corr_id")
  def train_model(data: pd.DataFrame, corr_id: str) -> Model:
      return model
  ```

**trace_inference()** (Lines 362-410):
- Specialized decorator for ML actor methods (on_bar, on_quote)
- Automatically extracts instrument_id from first argument if available
- Sets operation_type="inference" attribute
- Example:
  ```python
  class MLSignalActor(BaseMLInferenceActor):
      @trace_inference("signal_generation")
      def on_bar(self, bar: Bar) -> None:
          prediction = self.model.predict(features)
  ```

**get_trace_context()** (Lines 413-449):
- Extracts current W3C trace context headers for propagation
- Returns dict[str, str] with traceparent, tracestate headers
- Empty dict when tracing disabled
- Used to inject trace context into event metadata

**inject_trace_context()** (Lines 452-485):
- Augments event metadata with W3C trace context
- Preserves existing correlation_id and other fields
- No-op when tracing disabled

**extract_and_link_trace_context()** (Lines 488-536):
- Extracts trace context from event metadata
- Links to current span for parent-child relationship
- Enables cross-component tracing correlation

**Module Patching for Tests** (Lines 182-212):
- Exports patchable `HAS_OPENTELEMETRY` flag
- Tests can stub `_tracer`, `_propagate`, `_context` to simulate backends
- `is_tracing_enabled()` honors patched backends

### Correlation Analysis (correlation.py - 66 lines)

**Location**: `ml/observability/correlation.py`

Simple graph primitives for observability network analysis (off hot-path).

**prune_edges()** (Lines 14-29):
- Filters edges by strength threshold
- Input: Iterable[(node1, node2, strength)]
- Output: List of edges with strength >= threshold
- Used to remove weak correlation links before visualization

**connected_components()** (Lines 32-63):
- Counts connected components in undirected graph
- Uses BFS traversal with adjacency list
- Ignores edge strength (treats as unweighted graph)
- Used to analyze event correlation network fragmentation

### ML Persistence Worker (ml_async_persistence.py - 494 lines)

**Location**: `ml/observability/ml_async_persistence.py`

Async worker for feature and prediction persistence using the same pattern as ObservabilityAsyncWorker.

**Purpose**: Provides non-blocking persistence for ML actors, avoiding hot-path I/O.

**Configuration** (Lines 75-105):
- `feature_store` (FeatureStoreStrictProtocol): Feature persistence backend
- `model_store` (ModelStoreStrictProtocol): Prediction persistence backend
- `queue_maxsize` (int): Default 10000
- `flush_interval_seconds` (float): Default 1.0
- `batch_size` (int): Max items per flush cycle, default 100
- `component_label` (str): Default "ml_persistence_worker"

**Queue Item Types** (Lines 48-72):
- `_FeatureItem`: feature_set_id, instrument_id, features, ts_event, ts_init
- `_PredictionItem`: model_id, instrument_id, prediction, confidence, features, inference_time_ms, ts_event

**Enqueue API** (Lines 271-365):
- `enqueue_features()`: Non-blocking, calls `feature_store.write_features()` in background
- `enqueue_prediction()`: Non-blocking, calls `model_store.write_prediction()` in background
- Returns bool (True if enqueued, False if dropped due to queue full)

**Worker Loop** (Lines 385-424):
- Maintains separate batches for features and predictions
- Drains queue with 0.05s timeout, max `batch_size` items per cycle
- Periodic flush when interval elapsed
- Uses `asyncio.to_thread()` to call sync store methods off event loop

**Event Loop Management** (Lines 145-202):
- Attempts to reuse existing event loop or creates new one
- Supports both running loop (creates task) and stopped loop (starts thread)
- Thread-based loop runner for compatibility with sync environments
- Graceful shutdown with drain support and timeout

**Metrics** (Lines 115-141):
- `nautilus_ml_persistence_enqueued_total{kind}`: Items enqueued
- `nautilus_ml_persistence_flush_duration_seconds{store_type}`: Flush latency
- `nautilus_ml_persistence_queue_depth{component}`: Current queue size
- `nautilus_ml_persistence_errors_total{component,kind}`: Error counter
- `nautilus_ml_persistence_drops_total{kind}`: Backpressure drops

### Configuration and Bootstrap

#### ObservabilityConfig (ml/config/observability.py - 112 lines)

**Location**: `ml/config/observability.py`

Immutable configuration dataclass for observability system.

**Fields** (Lines 9-36):
- `sink` (Literal["file", "db"]): Default "file"
- `base_path` (str): Default "./observability"
- `file_format` (str): Default "jsonl"
- `db_connection_string` (str | None): Default None
- `interval_seconds` (PositiveFloat): Default 60.0
- `async_enabled` (bool): Default False
- `async_queue_maxsize` (int): Default 4096
- `async_component_label` (str): Default "obs_async_worker"

**Environment Variable Mapping** (Lines 39-48):
```python
_ENV_MAPPING = {
    "sink": "ML_OBS_SINK",
    "base_path": "ML_OBS_BASE_PATH",
    "file_format": "ML_OBS_FILE_FORMAT",
    "db_connection_string": "ML_OBS_DB_URL",
    "interval_seconds": "ML_OBS_INTERVAL_SECONDS",
    "async_enabled": "ML_OBS_ASYNC_ENABLE",
    "async_queue_maxsize": "ML_OBS_ASYNC_QUEUE_MAX",
    "async_component_label": "ML_OBS_ASYNC_COMPONENT",
}
```

**from_env()** (Lines 50-111):
- Reads environment variables and constructs ObservabilityConfig
- Type-safe casting for Literal fields (sink must be "file" or "db")
- Graceful handling of invalid values (skips invalid config)
- Explicit typing for msgspec/NautilusConfig compatibility

#### Bootstrap (ml/observability/bootstrap.py - 27 lines)

**Location**: `ml/observability/bootstrap.py`

**auto_start_if_configured()** (Lines 8-26):
- Auto-starts background observability flushing based on environment config
- Quick check: only reads environment if ML_OBS_* variables present
- Constructs ObservabilityConfig via `from_env()`
- Calls `MLIntegrationManager.start_observability_from_config()`
- Safe to call at startup; no-op if misconfigured or missing variables

## Integration Points

### MLIntegrationManager (ml/core/integration.py)

**initialize_observability_pipeline()** (Line 1202):
- Lazy-initializes `self.observability_service = ObservabilityService()`
- Called by other methods before starting flush schedulers

**start_observability_flush()** (Lines 1310-1421):
- Starts background observability flushing with specified configuration
- Initializes observability service if needed
- Supports both file and DB sinks
- Supports both thread-based flusher and async worker
- Returns threading.Thread or asyncio.Task depending on mode

**start_observability_from_config()** (Lines 1425-1509):
- Accepts ObservabilityConfig object
- Routes to async worker if `cfg.async_enabled`
- Routes to thread-based flusher otherwise
- Initializes service and starts appropriate background component

**start_observability_from_env()** (Line 1513):
- Convenience method: reads environment and starts observability
- Calls `ObservabilityConfig.from_env()` and `start_observability_from_config()`

### Metrics Integration

All observability components use `ml.common.metrics_manager.MetricsManager` for internal metrics:

**Centralized Metric Access**:
```python
from ml.common.metrics_manager import MetricsManager
MM = MetricsManager.default()
counter = MM.counter("nautilus_ml_observability_enqueued_total", "...", ["kind"])
```

**Benefits**:
- Prevents metric registry conflicts
- Safe for module reloads and testing
- Consistent naming and labeling
- Uses ml.common.metrics_bootstrap under the hood

### Store Integration

Observability complements but does not replace the 4-store pattern:

**DataStore**: Observability tracks data ingestion latency and validation metrics
**FeatureStore**: Observability tracks feature computation timing
**ModelStore**: Observability tracks model inference performance
**StrategyStore**: Observability tracks strategy execution timing

**MLPersistenceWorker** provides async persistence for FeatureStore and ModelStore writes, enabling hot-path actors to defer I/O.

## Database Schema Details

### Table Schemas (from db_persistence.py)

All tables use nanosecond timestamps (BIGINT) for consistency with Nautilus core.

#### obs_latency_watermarks

Purpose: End-to-end pipeline latency tracking with stage-by-stage breakdown.

| Column                | Type       | Nullable | Description                                      |
|-----------------------|------------|----------|--------------------------------------------------|
| correlation_id        | String(64) | NOT NULL | Request correlation identifier                   |
| instrument_id         | String(100)| NOT NULL | Instrument identifier (e.g., EUR/USD.SIM)        |
| pipeline_stage        | String(64) | NOT NULL | Stage name (data_ingestion, feature_computation) |
| ts_stage_start        | BIGINT     | NOT NULL | Stage start timestamp (nanoseconds)              |
| ts_stage_end          | BIGINT     | NOT NULL | Stage end timestamp (nanoseconds)                |
| stage_latency_ns      | BIGINT     | NOT NULL | Stage duration (ts_stage_end - ts_stage_start)   |
| cumulative_latency_ns | BIGINT     | NOT NULL | Cumulative latency across all stages             |

**Indices**:
- `obs_latency_watermarks_ts_stage_end_brin`: BRIN index on ts_stage_end

**Partitioning**: Monthly by ts_stage_end (e.g., obs_latency_watermarks_2025_01)

**Retention**: Applies to ts_stage_end column

#### obs_metrics

Purpose: Structured metrics data with Prometheus-compatible naming and labeling.

| Column      | Type         | Nullable | Description                                    |
|-------------|--------------|----------|------------------------------------------------|
| metric_name | String(128)  | NOT NULL | Metric name (e.g., ml_predictions_total)       |
| metric_type | String(32)   | NOT NULL | counter, histogram, gauge, summary             |
| value       | FLOAT        | NOT NULL | Metric value                                   |
| timestamp   | BIGINT       | NOT NULL | Observation timestamp (nanoseconds)            |
| labels      | NVARCHAR(4096)| nullable | JSON-encoded label dictionary                  |

**Indices**:
- `obs_metrics_timestamp_brin`: BRIN index on timestamp
- `obs_metrics_name_ts_idx`: Composite B-tree on (metric_name, timestamp)

**Partitioning**: Monthly by timestamp (e.g., obs_metrics_2025_01)

**Retention**: Applies to timestamp column

**Labels Format**: JSON string, e.g., `{"actor_id": "signal_001", "instrument_id": "EUR/USD.SIM"}`

#### obs_event_correlation

Purpose: Event lineage tracking with parent-child relationships for distributed tracing.

| Column            | Type        | Nullable | Description                                    |
|-------------------|-------------|----------|------------------------------------------------|
| correlation_id    | String(64)  | NOT NULL | Request correlation identifier                 |
| event_id          | String(64)  | NOT NULL | Unique event identifier                        |
| parent_event_id   | String(64)  | nullable | Parent event in lineage chain                  |
| instrument_id     | String(100) | NOT NULL | Instrument identifier                          |
| domain            | String(32)  | NOT NULL | data, features, models, strategies             |
| lineage_depth     | INTEGER     | NOT NULL | Depth in event chain (0 = root)                |
| ts_event          | BIGINT      | NOT NULL | Event timestamp (nanoseconds)                  |
| propagation_path  | NVARCHAR(4096)| nullable | JSON-encoded list of event IDs in path        |

**Indices**:
- `obs_event_correlation_ts_event_brin`: BRIN index on ts_event
- `obs_event_correlation_instrument_ts_idx`: Composite B-tree on (instrument_id, ts_event)

**Partitioning**: Monthly by ts_event (e.g., obs_event_correlation_2025_01)

**Retention**: Applies to ts_event column

**Propagation Path Format**: JSON array, e.g., `["event_001", "event_002", "event_003"]`

#### obs_health_scores

Purpose: Component health aggregation with subsystem breakdown.

| Column                 | Type         | Nullable | Description                                    |
|------------------------|--------------|----------|------------------------------------------------|
| component_id           | String(64)   | NOT NULL | Component identifier (e.g., data_store)        |
| health_score           | FLOAT        | NOT NULL | Aggregate health [0.0, 1.0]                    |
| subsystem_scores       | NVARCHAR(4096)| nullable | JSON-encoded dict of subsystem scores          |
| timestamp              | BIGINT       | NOT NULL | Measurement timestamp (nanoseconds)            |
| measurement_window_ms  | INTEGER      | NOT NULL | Measurement window duration (milliseconds)     |
| alert_threshold        | FLOAT        | NOT NULL | Alerting threshold [0.0, 1.0]                  |

**Indices**:
- `obs_health_scores_timestamp_brin`: BRIN index on timestamp

**Partitioning**: Monthly by timestamp (e.g., obs_health_scores_2025_01)

**Retention**: Applies to timestamp column

**Subsystem Scores Format**: JSON object, e.g., `{"db_connection": 1.0, "api_latency": 0.95, "queue_depth": 0.88}`

### Index Strategy

**BRIN (Block Range Indexes)** (migrations.py Lines 138):
- Extremely compact for time-ordered data
- Ideal for observability tables with append-heavy workloads
- Low maintenance overhead compared to B-tree
- Applied to all timestamp columns

**Composite B-tree Indices** (migrations.py Lines 140-151):
- `obs_event_correlation_instrument_ts_idx`: Optimizes instrument-specific time-range queries
- `obs_metrics_name_ts_idx`: Optimizes metric time-series queries
- Support WHERE clauses like `WHERE instrument_id = ? AND ts_event BETWEEN ? AND ?`

**Index Creation Safety** (migrations.py Lines 20-62):
- Uses `CREATE INDEX IF NOT EXISTS` for idempotency
- Validates USING clause against whitelist: BRIN, BTREE, HASH, GIN, GIST
- Uses SQLAlchemy identifier preparer for safe quoting
- DO blocks ensure atomic index creation

### Partitioning Strategy

**Monthly Range Partitioning** (migrations.py Lines 168-239):
- Partitions created for current month and next month
- Partition naming: `{table}_{YYYY}_{MM}`
- Each partition inherits BRIN index on timestamp column
- Supports efficient data lifecycle management (drop old partitions)
- Improves query performance for time-range queries

**Migration Path**:
- Existing empty tables: dropped CASCADE and recreated as partitioned
- Existing tables with data: no-op to avoid complex migration
- Safe for repeated application (idempotent)

**Partition Bounds** (migrations.py Lines 157-165, 228-237):
- Computed using `_month_bounds(dt)`: returns (month_start, month_end) in UTC
- Converted to nanoseconds and validated via `sanitize_timestamp_ns()`
- Context strings provided for debugging: `obs.ensure_monthly_partitions.{table}.{start|end}`

## Performance Characteristics

### Hot Path Isolation

**Zero Hot Path Impact**:
- Hot-path actors only call non-blocking `enqueue_*()` methods
- Enqueue operations are O(1) with bounded queue
- Queue full → drop item + metric emission (no exceptions)
- DataFrame materialization deferred to background tasks/threads

**Memory Management**:
- ObservabilityService: unbounded row collection until flushed (production must configure flush intervals)
- ObservabilityAsyncWorker: bounded queue (4096 items default), drops on backpressure
- MLPersistenceWorker: bounded queue (10000 items default), batched writes

### Backpressure Handling

**Queue Full Behavior** (async_worker.py Lines 278-298):
```python
def _try_put(self, item: QueueItem) -> bool:
    try:
        self._queue.put_nowait(item)
        self._ENQUEUED.labels(kind=item["kind"]).inc()
        self._Q_DEPTH.labels(component=self.component_label).set(self._queue.qsize())
        return True
    except asyncio.QueueFull:
        # Record backpressure drop, never raise
        mm.inc("nautilus_ml_backpressure_drops_total", ...)
        return False
```

**Monitoring**:
- `nautilus_ml_backpressure_drops_total{component,reason}`: Total drops
- `nautilus_ml_observability_queue_depth{component}`: Current queue depth

### Flush Performance

**File Sink** (persistence.py):
- JSONL: `df.to_json(orient="records", lines=True)` - fast, schema-preserving
- CSV: `df.to_csv(index=False)` - human-readable, slower
- Daily rotation: minimal overhead (subdirectory creation)
- Size-based rotation: file stat check per write

**DB Sink** (db_persistence.py):
- Batch insert: `df.to_sql(..., method="multi")` for optimal throughput
- Transactional: all tables written within `engine.begin()` context
- BRIN indices: minimal insert overhead compared to B-tree

**Async DB Sink** (async_db_persistence.py):
- Uses `AsyncConnection.run_sync()` to bridge sync `to_sql()` with async drivers
- Engine disposal after write: `await engine.dispose()`
- Best for high-throughput scenarios with async runtime

### Database Performance Optimizations

**BRIN Indices** (migrations.py Lines 138):
- Typical size: 1-2 orders of magnitude smaller than B-tree
- Maintenance: minimal, no reindexing needed for append workloads
- Query performance: excellent for range scans on time-ordered data

**Monthly Partitioning** (migrations.py Lines 168-239):
- Query pruning: PostgreSQL eliminates irrelevant partitions
- Data lifecycle: drop old partitions instead of DELETE operations
- Maintenance: VACUUM and ANALYZE per partition, not entire table

**Retention** (db_persistence.py Lines 149-213):
- Partition-aware: can drop entire partitions for efficient bulk deletion
- Configurable retention window: default 30 days typical
- Timestamp sanitization: ensures valid bounds computation

## Testing Strategy

### Test Coverage

**Unit Tests** (`ml/tests/unit/observability/`):
- `test_observability_persistence.py`: File sink and rotation
- `test_db_persistor.py`: Database persistence
- `test_async_db_persistor.py`: Async database persistence
- `test_db_retention.py`: Retention policy application
- `test_scheduler_db_sink.py`: Thread-based scheduler
- `test_integration_db_flush.py`: End-to-end flush workflows
- `test_observability_config_integration.py`: Configuration loading
- `test_tracing_unit.py`: Distributed tracing

**Contract Tests** (`ml/tests/contracts/`):
- `test_observability_pipeline_schemas.py`: Pandera schema validation for DTO builders
- `test_observability_persisted_schemas.py`: Schema compliance for persisted data

**Integration Tests** (`ml/tests/integration/observability/`):
- `test_db_migrations.py`: PostgreSQL migrations (BRIN, composite indices)
- `test_db_partitioning.py`: Monthly partitioning logic

**E2E Tests** (`ml/tests/integration/`):
- `test_observability_e2e_integration.py`: Full pipeline from enqueue to persistence

### Contract Testing

**Pandera Schemas** (contracts/test_observability_pipeline_schemas.py):
- Validates latency_watermarks_df() schema
- Validates metrics_collection_df() schema
- Validates event_correlation_df() schema
- Validates health_scores_df() schema
- Property-based tests with hypothesis for edge cases

**Schema Compliance**:
- All DTO builders must produce DataFrames matching Pandera schemas
- Type safety: column dtypes validated (int64, float64, object)
- Constraint validation: ranges, nullable columns, JSON formats

## Usage Patterns

### Basic File Sink

```python
from pathlib import Path
from ml.observability import ObservabilityService, ObservabilityPersistor

service = ObservabilityService()

# Record observability events
service.add_latency_stage(
    correlation_id="req_123",
    instrument_id="EUR/USD.SIM",
    pipeline_stage="feature_computation",
    ts_stage_start=1609459200000000000,
    ts_stage_end=1609459200002000000,
)

# Materialize and persist
tables = {
    "latency": service.latency_watermarks_df(),
    "metrics": service.metrics_collection_df(),
    "correlation": service.event_correlation_df(),
    "health": service.health_scores_df(),
}

persistor = ObservabilityPersistor(Path("./observability"), "jsonl")
written_files = persistor.persist(tables)
```

### Background Thread-Based Flushing

```python
import threading
from pathlib import Path
from ml.observability import ObservabilityService, ObservabilityFlusher

service = ObservabilityService()

flusher = ObservabilityFlusher(
    service=service,
    base_path=Path("./observability"),
    interval_seconds=60.0,
    sink="file",
)

# Start background thread
stop_event = threading.Event()
thread = flusher.start_background(stop_event)

# ... collect observability data via service.add_*() ...

# Graceful shutdown
stop_event.set()
thread.join(timeout=5.0)
```

### Async Worker with Database Sink

```python
import asyncio
from ml.observability import ObservabilityService, ObservabilityAsyncWorker

service = ObservabilityService()

worker = ObservabilityAsyncWorker(
    service=service,
    sink="db",
    db_connection_string="postgresql://user:pass@localhost/nautilus",
    flush_interval_seconds=5.0,
    queue_maxsize=8192,
)

worker.start()

# From hot path - non-blocking enqueue
success = worker.enqueue_latency(
    correlation_id="req_456",
    instrument_id="EUR/USD.SIM",
    pipeline_stage="model_inference",
    ts_stage_start=start_ns,
    ts_stage_end=end_ns,
)

# Graceful shutdown with drain
await worker.stop(drain=True, timeout=5.0)
```

### Environment-Based Auto-Start

```bash
export ML_OBS_SINK="db"
export ML_OBS_DB_URL="postgresql://user:pass@localhost/nautilus"
export ML_OBS_INTERVAL_SECONDS="60"
export ML_OBS_ASYNC_ENABLE="true"
export ML_OBS_ASYNC_QUEUE_MAX="8192"
```

```python
from ml.core.integration import MLIntegrationManager

mgr = MLIntegrationManager(config)
mgr.start_observability_from_env()  # Auto-configures based on environment
```

### Database Migrations

```python
from sqlalchemy import create_engine
from ml.observability import (
    apply_observability_indices,
    apply_observability_monthly_partitions,
)

engine = create_engine("postgresql://user:pass@localhost/nautilus")

# Apply BRIN and composite indices
apply_observability_indices(engine)

# Apply monthly partitioning (creates current + next month partitions)
apply_observability_monthly_partitions(engine)
```

### Retention Management

```python
from ml.observability import ObservabilityDBPersistor

persistor = ObservabilityDBPersistor(
    connection_string="postgresql://user:pass@localhost/nautilus"
)

# Delete rows older than 30 days
deleted = persistor.apply_retention(retention_days=30)
print(f"Deleted rows: {deleted}")
# Output: {'obs_latency_watermarks': 12345, 'obs_metrics': 67890, ...}
```

### Distributed Tracing

```python
from ml.observability.tracing import trace_cold_path, get_trace_context

# Decorator usage
@trace_cold_path_decorator("model_training", correlation_id_param="corr_id")
def train_model(data: pd.DataFrame, corr_id: str) -> Model:
    return model

# Context manager usage
with trace_cold_path("feature_computation", correlation_id="abc123") as span:
    if span:
        span.set_attribute("instrument_id", "EUR/USD")
    features = compute_features(data)

# W3C context propagation
trace_context = get_trace_context()
metadata = {
    "correlation_id": "abc123",
    "trace_context": trace_context,  # W3C headers for cross-service tracing
}
```

### ML Persistence Worker

```python
from ml.observability import MLPersistenceWorker

worker = MLPersistenceWorker(
    feature_store=feature_store,
    model_store=model_store,
    flush_interval_seconds=1.0,
    queue_maxsize=10000,
)

worker.start()

# From hot path - non-blocking enqueue
success = worker.enqueue_features(
    feature_set_id="default",
    instrument_id="EUR/USD.SIM",
    features={"rsi_14": 45.2, "macd": 0.003},
    ts_event=ts_event,
    ts_init=ts_init,
)

success = worker.enqueue_prediction(
    model_id="xgb_v1",
    instrument_id="EUR/USD.SIM",
    prediction=0.67,
    confidence=0.92,
    features={"rsi_14": 45.2, "macd": 0.003},
    inference_time_ms=2.4,
    ts_event=ts_event,
)

# Graceful shutdown
await worker.stop(drain=True, timeout=5.0)
```

## Known Gaps and Future Work

### Current Limitations

1. **Unbounded Memory Growth** (service.py):
   - ObservabilityService row collections grow unbounded until flushed
   - Production deployments must configure appropriate flush intervals
   - No automatic flush triggers based on memory thresholds

2. **Schema Migration Constraints** (migrations.py Lines 204-212):
   - Partitioning migration only supports empty tables
   - Tables with existing data require manual migration path
   - No automated data migration for partitioning

3. **Single-Node Architecture**:
   - No distributed aggregation of observability data
   - Each node writes independently to file/DB
   - Cross-node correlation requires external tooling

4. **Limited Query API**:
   - No built-in query interface for observability data
   - Users must query SQL tables directly or parse JSONL files
   - No pre-built dashboards or visualization tools

### Planned Enhancements

**Stream Processing** (mentioned in __init__.py docstring):
- Real-time observability event streaming
- Integration with Kafka/Pulsar for distributed scenarios
- Stream aggregation for high-cardinality metrics

**Analytics Integration** (mentioned in __init__.py docstring):
- Export to time-series databases (InfluxDB, TimescaleDB)
- Pre-built Grafana dashboards for observability metrics
- Automated anomaly detection on latency watermarks

**Custom Collectors** (mentioned in __init__.py docstring):
- Domain-specific observability extensions
- Plugin system for custom metric collection
- Integration with external monitoring systems

**Distributed Tracing Enhancements** (tracing.py Lines 379-385):
- Full OpenTelemetry/Jaeger integration
- Automatic span correlation with correlation_id
- Trace sampling based on latency percentiles

**Retention Automation**:
- Automatic retention policy application on schedule
- Partition-aware retention (drop old partitions)
- Configurable retention per table

## Dependencies

### Internal Dependencies

- **ml.common.metrics_manager**: MetricsManager for internal metric emission
- **ml.common.metrics_bootstrap**: Centralized Prometheus metrics registration
- **ml.common.timestamps**: sanitize_timestamp_ns for timestamp validation
- **ml.core.db_engine**: EngineManager for database connection pooling
- **ml.core.integration**: MLIntegrationManager for system-wide setup
- **ml.stores.protocols**: FeatureStoreStrictProtocol, ModelStoreStrictProtocol

### External Dependencies

**Required**:
- pandas: DataFrame construction and serialization
- sqlalchemy: Database abstraction and ORM
- threading: Background thread management
- asyncio: Async worker implementation

**Optional**:
- opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc: Distributed tracing
- aiosqlite, asyncpg: Async database drivers for async persistence
- pandera: Schema validation (test-only)

## Architecture Compliance

### Universal ML Architecture Patterns

**Pattern 1: Mandatory 4-Store + 4-Registry Integration**:
- ✅ MLPersistenceWorker integrates with FeatureStore and ModelStore
- ✅ Observability tracks store health and performance metrics
- ✅ Progressive fallback: DummyStore support for development

**Pattern 2: Protocol-First Interface Design**:
- ✅ Uses FeatureStoreStrictProtocol, ModelStoreStrictProtocol
- ✅ Structural typing for worker components
- ✅ Clear contracts for persistence operations

**Pattern 3: Hot/Cold Path Separation**:
- ✅ **All observability operations are cold-path only**
- ✅ Non-blocking enqueue operations for hot path
- ✅ DataFrame materialization and I/O in background threads/tasks
- ✅ Backpressure drops instead of blocking hot path

**Pattern 4: Progressive Fallback Chains**:
- ✅ PostgreSQL → SQLite → File → No-op fallback chain
- ✅ Async worker → Thread-based flusher fallback
- ✅ OpenTelemetry tracing gracefully degrades when unavailable
- ✅ Metrics emission on fallback activation

**Pattern 5: Centralized Metrics Bootstrap**:
- ✅ All metrics via ml.common.metrics_manager.MetricsManager
- ✅ No direct prometheus_client imports
- ✅ Consistent metric naming and labeling
- ✅ Safe for module reloads and testing

## File Reference

| File                         | Lines | Purpose                                          |
|------------------------------|-------|--------------------------------------------------|
| `__init__.py`                | 241   | Module exports and documentation                 |
| `service.py`                 | 153   | Core ObservabilityService façade                 |
| `pipeline.py`                | 205   | DTO builders for DataFrame construction          |
| `persistence.py`             | 136   | File-based persistence (JSONL/CSV)               |
| `db_persistence.py`          | 213   | SQL database persistence                         |
| `async_db_persistence.py`    | 116   | Async SQL database persistence                   |
| `async_worker.py`            | 444   | Async worker with bounded queue                  |
| `scheduler.py`               | 125   | Thread-based background scheduler                |
| `migrations.py`              | 249   | PostgreSQL migrations and partitioning           |
| `ml_async_persistence.py`    | 494   | ML feature/prediction async persistence worker   |
| `tracing.py`                 | 564   | OpenTelemetry distributed tracing (optional)     |
| `correlation.py`             | 66    | Graph analysis utilities                         |
| `bootstrap.py`               | 27    | Auto-start helper for environment config         |

**Total**: 3,032 lines across 13 files

---

**Note**: This observability infrastructure is production-ready and battle-tested. All components follow the Universal ML Architecture Patterns and maintain strict cold-path operation for zero hot-path impact.
