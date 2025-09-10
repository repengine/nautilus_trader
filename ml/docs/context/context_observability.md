# Context: Observability Module

## Overview

The ML observability module provides lightweight, off-hot-path infrastructure for collecting and persisting comprehensive system metrics, latency watermarks, event correlation data, and health scores. This module is designed to complement the existing monitoring system (ml/monitoring) by focusing on fine-grained observability data collection and structured persistence rather than real-time metrics export.

The module implements a "collect-and-defer" pattern where hot paths record minimal observability events, while background processes handle the expensive operations of materializing DataFrames and persisting to disk. This ensures zero performance impact on critical inference paths while maintaining comprehensive system visibility.

## Architecture

The observability module follows a layered architecture optimized for separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Hot Path (ML Actors)                     │
│           ┌─────────────────────────────────────┐          │
│           │ Minimal event recording only        │          │
│           │ (correlation IDs, timestamps)       │          │
│           └─────────────────────────────────────┘          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ObservabilityService (Cold Path)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Row Collection│  │DTO Builders │  │ DataFrame    │     │
│  │ (Lightweight)│  │ (Transform)  │  │ Generation   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Persistence Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Scheduled    │  │ File Writers │  │ Format       │     │
│  │ Flushing     │  │ (JSONL/CSV)  │  │ Serializers  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### ObservabilityService

**Location**: `ml/observability/service.py`

Central façade for collecting observability rows and materializing contract-compliant DataFrames. The service maintains four types of observability data:

- **Latency Watermarks**: End-to-end pipeline latency tracking with stage-by-stage breakdown
- **Metrics Collection**: Structured metrics data with Prometheus-compatible naming and labeling
- **Event Correlation**: Event lineage tracking with parent-child relationships for distributed tracing
- **Health Scores**: Component health aggregation with subsystem breakdown

Key methods:

- `add_latency_stage()`: Record pipeline stage execution timing
- `add_metric()`: Record metric observation with labels and timestamp
- `add_correlation()`: Record event correlation with lineage depth tracking
- `add_health()`: Record component health scores with measurement windows

DataFrame materialization methods:

- `latency_watermarks_df()`: Generate DataFrame with cumulative latency calculations
- `metrics_collection_df()`: Generate DataFrame with type-normalized metrics
- `event_correlation_df()`: Generate DataFrame with JSON-serialized propagation paths
- `health_scores_df()`: Generate DataFrame with JSON-encoded subsystem scores

### Pipeline DTO Builders

**Location**: `ml/observability/pipeline.py`

Provides typed builders that transform raw observability rows into pandas DataFrames with schema compliance and data normalization. Each builder handles specific data transformation concerns:

**`build_latency_watermarks()`**:

- Calculates stage latency from timestamp differences
- Computes cumulative latency across pipeline stages
- Validates timestamp ordering and consistency
- Handles empty input gracefully with proper dtype preservation

**`build_metrics_collection()`**:

- Normalizes metric values to float64
- Ensures timestamps are int64 nanoseconds since epoch
- JSON-encodes label dictionaries consistently
- Supports both dict and pre-serialized string labels

**`build_event_correlation()`**:

- Converts propagation paths to JSON string format
- Validates lineage depth consistency
- Handles nullable parent event relationships
- Preserves event ordering for lineage reconstruction

**`build_health_scores()`**:

- Clamps health scores to [0, 1] range
- JSON-encodes subsystem score dictionaries
- Adds default alert thresholds for schema compliance
- Validates measurement window constraints

### ObservabilityPersistor

**Location**: `ml/observability/persistence.py`

Handles persistence of observability DataFrames to disk in structured formats. Supports both JSONL and CSV output formats with automatic directory creation and empty DataFrame filtering.

Key features:

- **Format Support**: JSONL (default) for schema preservation, CSV for human readability
- **Path Management**: Automatic directory creation with `parents=True, exist_ok=True`
- **Empty Filtering**: Skips persistence of None or empty DataFrames
- **Return Mapping**: Provides dict of table name to written file path for verification

### ObservabilityFlusher

**Location**: `ml/observability/scheduler.py`

Background scheduler for periodic persistence of observability tables. Supports both tick-based operation (for deterministic testing) and threaded background operation (for production use).

**Operating Modes**:

- **Tick Mode**: Deterministic flush based on elapsed time intervals, suitable for tests
- **Background Mode**: Separate thread with configurable interval and graceful shutdown
- **Immediate Mode**: Single flush operation with interval_seconds <= 0

**Configuration**:

- `interval_seconds`: Flush frequency (60.0 seconds default)
- `now`: Time function (mockable for testing, defaults to `time.time()`)
- `file_format`: Output format ("jsonl" or "csv")

**Thread Safety**: Background thread handles exceptions gracefully and uses `threading.Event` for clean shutdown.

## Dependencies

### Internal Dependencies

- **ml.common.protocols**: MLComponentProtocol for health aggregation
- **ml.core.integration**: MLIntegrationManager integration and lifecycle management
- **pandas**: DataFrame construction and manipulation
- **pathlib**: Type-safe path handling for file operations

### External Dependencies

- **pandas**: Core DataFrame operations and serialization
- **threading**: Background scheduler thread management
- **json**: Label and metadata serialization
- **time**: Default timestamp generation

### Testing Dependencies

The module is extensively tested with:

- **pandera**: Schema validation and contract enforcement
- **hypothesis**: Property-based testing for edge cases
- **pytest**: Unit and integration testing framework

## Usage Patterns

### Integration with MLIntegrationManager

The observability module is automatically initialized by `MLIntegrationManager` when observability features are requested:

```python
from ml.core.integration import MLIntegrationManager

# Automatic initialization
integration = MLIntegrationManager(config)
integration.initialize_observability_pipeline()

# Background persistence
integration.start_observability_flush(
    base_path=Path("/data/observability"),
    interval_seconds=60.0,
    file_format="jsonl"
)
```

### Programmatic Usage

For custom applications requiring direct observability control:

```python
from ml.observability.service import ObservabilityService
from ml.observability.persistence import ObservabilityPersistor
from pathlib import Path

# Create service
service = ObservabilityService()

# Record observability events
service.add_latency_stage(
    correlation_id="uuid-here",
    instrument_id="EURUSD.SIM",
    pipeline_stage="feature_computation",
    ts_stage_start=1609459200000000000,
    ts_stage_end=1609459200002000000
)

# Materialize and persist
tables = {
    "latency": service.latency_watermarks_df(),
    "metrics": service.metrics_collection_df(),
    "correlation": service.event_correlation_df(),
    "health": service.health_scores_df(),
}

persistor = ObservabilityPersistor(Path("/output"), "jsonl")
written_files = persistor.persist(tables)
```

### Background Scheduling

For production environments requiring periodic persistence:

```python
from ml.observability.scheduler import ObservabilityFlusher
import threading

# Create background flusher with database sink
flusher = ObservabilityFlusher(
    service=service,
    base_path=Path("/data/observability"),
    interval_seconds=60.0,
    file_format="jsonl",
    sink="db",
    db_connection_string="postgresql://user:pass@localhost/nautilus"
)

# Start background thread
stop_event = threading.Event()
thread = flusher.start_background(stop_event)

# Graceful shutdown
stop_event.set()
thread.join(timeout=5.0)
```

### Async Observability Worker

For deployments requiring higher throughput or tighter hot-path budgets, use the async worker to enqueue observability rows without blocking and persist them off-path:

```python
from pathlib import Path
from ml.observability.service import ObservabilityService
from ml.observability.async_worker import ObservabilityAsyncWorker

svc = ObservabilityService()
worker = ObservabilityAsyncWorker(
    service=svc,
    sink="file",  # or "db"
    base_path=Path("./observability"),
    db_connection_string=None,
    flush_interval_seconds=5.0,
    queue_maxsize=4096,
)
worker.start()

# Hot path: enqueue is non-blocking (drops on backpressure and increments central counter)
worker.enqueue_metric(
    metric_name="ml_predictions_total",
    metric_type="counter",
    value=1.0,
    timestamp=1,
    labels={"actor_id": "a1"},
)

# Shutdown (off hot path)
import asyncio
asyncio.run(worker.stop(drain=True))
```

Environment (integration manager will start async worker when enabled):

```bash
export ML_OBS_ASYNC_ENABLE="true"
export ML_OBS_ASYNC_QUEUE_MAX="8192"
export ML_OBS_ASYNC_COMPONENT="obs_async_worker"
```

When `ML_OBS_ASYNC_ENABLE` is not set, the thread-based `ObservabilityFlusher` is used by default.

## Integration Points

### ML Monitoring System

The observability module complements but does not replace the existing monitoring system:

- **ml/monitoring**: Real-time Prometheus metrics export, hot-path instrumentation, alerting
- **ml/observability**: Structured data collection, batch persistence, schema validation

### Core Integration

Deeply integrated with `MLIntegrationManager` for automatic lifecycle management:

- **Lazy Initialization**: Service created only when observability features are requested
- **Graceful Degradation**: Missing dependencies don't break core functionality
- **Thread Management**: Background threads managed by integration manager lifecycle

### Data Stores and Registries

Observability data provides audit trails and performance insights for:

- **FeatureStore**: Feature computation latency and quality metrics
- **ModelStore**: Model inference performance and prediction tracking
- **StrategyStore**: Strategy execution timing and decision lineage
- **DataStore**: Data ingestion pipeline health and validation metrics

### Event System Integration

Observability correlation tracking integrates with Nautilus event propagation:

- **Correlation IDs**: UUID4 identifiers for end-to-end request tracing
- **Event Lineage**: Parent-child relationships for distributed request tracking
- **Domain Boundaries**: Tracks event propagation across data/features/models/strategies domains

## Implementation Notes

### Performance Considerations

**Hot Path Isolation**: The module is explicitly designed to avoid hot path impact:

- Row collection operations are O(1) list appends
- DataFrame materialization deferred to background processing
- File I/O operations isolated to background threads

**Memory Management**:

- Service maintains in-memory row collections that grow unbounded until flushed
- Production deployments should configure appropriate flush intervals
- DataFrame materialization creates temporary copies for transformation

### Schema Compliance

The module implements strict schema compliance through:

**Contract Tests**: Comprehensive Pandera-based schema validation in `ml/tests/contracts/test_observability_pipeline_schemas.py`

**Type Safety**: All public APIs use complete type annotations with pandas DataFrame generics

**Data Normalization**: DTO builders ensure consistent data types and formats regardless of input variations

### Error Handling

**Graceful Degradation**: Missing optional dependencies don't break functionality:

```python
try:
    from ml.observability.service import ObservabilityService
    self.observability_service = ObservabilityService()
except Exception:
    self.observability_service = None  # Graceful fallback
```

**Background Resilience**: Background threads handle exceptions without terminating:

```python
def _run():
    while not stop_event.is_set():
        try:
            self.tick()
        except Exception:
            pass  # Keep background resilient
```

### Testing Strategy

The module follows comprehensive testing patterns:

- **Unit Tests**: Individual component behavior validation
- **Contract Tests**: Schema compliance and data integrity
- **Integration Tests**: End-to-end workflow validation
- **Property Tests**: Hypothesis-based edge case coverage
- **Metamorphic Tests**: Cross-validation of observability relationships

### Future Extensions

The module architecture supports planned enhancements:

- **Distributed Tracing**: Integration with OpenTelemetry/Jaeger
- **Stream Processing**: Real-time observability event streaming
- **Analytics Integration**: Export to time-series databases
- **Custom Collectors**: Domain-specific observability extensions

This observability infrastructure provides the foundation for comprehensive ML system visibility while maintaining strict performance isolation from critical inference paths.
