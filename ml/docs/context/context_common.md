# Context: Common Module

## Overview

The `ml/common/` module provides foundational utilities, protocols, and infrastructure components shared across all ML components within Nautilus Trader. This module implements core patterns for component standardization, metrics collection, message passing, timestamp handling, and cross-domain event coordination. It serves as the foundational layer that ensures consistency, type safety, and performance across the entire ML system.

The module follows the Universal ML Architecture Patterns mandated by CLAUDE.md, particularly Pattern 5 (Centralized Metrics Bootstrap) and Pattern 2 (Protocol-First Interface Design). All components are designed for both hot-path performance and cold-path reliability, with careful separation of concerns.

## Architecture

The common module is organized into focused, single-responsibility components:

```
ml/common/
├── __init__.py              # Minimal exports (empty)
├── protocols.py             # Universal ML component protocol and mixin
├── metrics_bootstrap.py     # Safe, idempotent metrics creation utilities
├── metrics.py               # Centralized Prometheus metrics definitions
├── metrics_export.py        # Safe Prometheus metrics export wrapper
├── timestamps.py            # Timestamp normalization and sanitization
├── precision.py             # Nautilus Price/Quantity precision helpers
├── correlation.py           # Event tracing correlation ID utilities
├── cascade.py               # Cross-domain event cascade helpers
├── message_bus.py           # Message bus publisher protocol and implementation
├── message_topics.py        # Message bus topic builder and normalization
├── in_memory_bus.py         # In-memory pub/sub for testing and examples
├── throttler.py             # Token-bucket rate limiting for event publishing
└── topic_filters.py         # MQTT-style wildcard topic matching utilities
```

### Design Principles

1. **Protocol-First Design**: All interfaces defined via `typing.Protocol` for structural typing
2. **Zero Hot-Path Dependencies**: Utilities designed to avoid blocking operations in inference
3. **Centralized Metrics**: Single point of truth for all Prometheus metrics
4. **Type Safety**: Complete type annotations with strict mypy compliance
5. **Idempotent Operations**: Safe to call multiple times without side effects

## Key Components

### 1. MLComponentProtocol and MLComponentMixin (`protocols.py`)

**Purpose**: Standardizes health reporting, performance metrics, and configuration validation across all ML components.

**Key Features**:

- Runtime-checkable protocol with `@runtime_checkable` decorator
- Standard interface for health status, performance metrics, and configuration validation
- Safe default implementations via `MLComponentMixin`
- Designed to keep methods out of hot path

**Protocol Definition**:

```python
@runtime_checkable
class MLComponentProtocol(Protocol):
    def get_health_status(self) -> dict[str, Any]
    def get_performance_metrics(self) -> dict[str, float]
    def validate_configuration(self) -> list[str]
```

**Integration**: Used by all stores, registries, and actors via inheritance from `MLComponentMixin`.

### 2. Metrics Bootstrap (`metrics_bootstrap.py`)

**Purpose**: Provides safe, idempotent metric creation to avoid duplicate registration and prometheus-client conflicts.

**Key Functions**:

- `get_counter()`: Creates or retrieves Counter metrics
- `get_histogram()`: Creates or retrieves Histogram metrics with optional buckets
- `get_gauge()`: Creates or retrieves Gauge metrics

**Design Pattern**:

```python
from ml.common.metrics_bootstrap import get_counter
counter = get_counter("ml_predictions_total", "Total predictions made")
```

**Implementation**: Uses internal `_METRICS` dict with composite keys to ensure idempotency.

### 3. Centralized Metrics (`metrics.py`)

**Purpose**: Defines all Prometheus metrics once to avoid duplication and registration conflicts.

**Metric Categories**:

- **Data Pipeline**: Event tracking, watermark lag, coverage, contract violations
- **Data Collection**: Duration, errors, catalog operations
- **Feature Store**: Operations, computation duration, drift scores
- **Model Store**: Operations, inference duration, accuracy, confidence
- **Strategy Store**: Operations, signal generation, P&L tracking
- **Validation**: Contract violations, schema mismatches, quality scores
- **System Health**: Pipeline health, readiness status

**Helper Functions**:

- `record_pipeline_event()`: Consistent event recording with proper labeling
- `update_pipeline_health()`: Health score updates

**Legacy Compatibility**: Includes backwards-compatible aliases (`MODEL_INFERENCE_TIMER`, etc.) for existing tests.

### 4. Timestamp Utilities (`timestamps.py`)

**Purpose**: Normalizes UNIX timestamps to Nautilus-standard nanoseconds with configurable policies.

**Core Functions**:

- `normalize_timestamp_ns()`: Heuristic-based conversion (seconds/ms/μs → ns)
- `sanitize_timestamp_ns()`: Policy-driven sanitization with logging

**Policies**:

- `warn` (default): Normalize and log warnings
- `normalize`: Normalize silently
- `reject`: Raise ValueError for non-nanosecond timestamps

**Integration**: Used throughout stores for consistent timestamp handling.

### 5. Precision Helpers (`precision.py`)

**Purpose**: Ensures safe construction of Nautilus Price/Quantity objects by clamping float precision.

**Key Function**:

- `clamp_price_str()`: Clamps floats to safe decimal precision (≤16 decimals)

**Usage Pattern**:

```python
price_str = clamp_price_str(123.456789012345678901, decimals=9)  # "123.456789012"
```

### 6. Correlation Utilities (`correlation.py`)

**Purpose**: Generates deterministic correlation IDs for tracing events across the Data → Features → Predictions → Signals pipeline.

**Key Function**:

- `make_correlation_id()`: Creates SHA256-based correlation ID from run metadata

**Parameters**:

- `run_id`: Pipeline run identifier
- `dataset_id`: Dataset type (features, predictions, signals)
- `instrument_id`: Instrument identifier
- `ts_min/ts_max`: Timestamp range
- `count`: Record count

### 7. Event Cascade Utilities (`cascade.py`)

**Purpose**: Supports cross-domain event cascades with correlation preservation for testing and integration.

**Key Components**:

- `EventDict`: TypedDict for event structure
- `emit_cascade()`: Creates cascaded events preserving correlation

**Usage**: Primarily used by `MLIntegrationManager.emit_cascade()` for domain bookkeeping.

### 8. Metrics Export (`metrics_export.py`)

**Purpose**: Provides safe wrapper for Prometheus metrics export without direct prometheus_client imports.

**Key Features**:

- **Safe Import Pattern**: Uses dynamic imports to avoid violating centralized metrics bootstrap rules
- **Graceful Fallback**: Returns empty payload when prometheus_client is unavailable
- **Content Type Detection**: Automatically determines proper content type for metrics exposition

**Core Functions**:

- `generate_latest()`: Returns metrics exposition payload with fallback to empty bytes
- `CONTENT_TYPE_LATEST`: Proper content type constant for Prometheus metrics

**Usage Pattern**:

```python
from ml.common.metrics_export import generate_latest, CONTENT_TYPE_LATEST

# Safe metrics export for HTTP endpoints
metrics_data = generate_latest()  # bytes
```

### 9. Message Bus Abstraction (`message_bus.py`)

**Purpose**: Provides minimal, typed interface for message bus publishing with safe default implementation.

**Components**:

- `MessagePublisherProtocol`: Protocol for message publishers
- `NoopPublisher`: Safe default implementation (returns False)

**Integration Pattern**: Allows dependency injection while maintaining type safety.

### 10. In-Memory Bus (`in_memory_bus.py`)

**Purpose**: Lightweight pub/sub implementation for testing and examples with wildcard pattern matching.

**Key Features**:

- **Pattern Subscriptions**: Supports wildcard patterns using `topic_filters.match_topic`
- **Handler Registration**: Subscribe handlers to topic patterns
- **Testing Focus**: Designed for unit tests and local examples, not production hot-path usage

**Core Interface**:

```python
class InMemoryPublisher(MessagePublisherProtocol):
    def subscribe(self, pattern: str, handler: Handler) -> None
    def publish(self, topic: str, payload: dict[str, Any]) -> bool
```

**Usage Pattern**:

```python
from ml.common.in_memory_bus import InMemoryPublisher

publisher = InMemoryPublisher()
publisher.subscribe("ml.features.*", lambda topic, payload: print(f"Got: {topic}"))
publisher.publish("ml.features.updated.EURUSD", {"status": "computed"})
```

### 11. Throttler (`throttler.py`)

**Purpose**: Non-blocking token-bucket rate limiting for event publishing to prevent external bus flooding.

**Key Features**:

- **Per-Key Limiting**: Independent rate limits per topic/key
- **Token Bucket Algorithm**: Allows bursts up to configured limit with steady refill rate
- **Nanosecond Precision**: Uses nanosecond timestamps for accurate rate calculation
- **Non-Blocking**: Returns immediately without waiting for tokens

**Configuration**:

```python
from ml.common.throttler import Throttler

throttler = Throttler(rate_per_sec=10.0, burst=5)
if throttler.should_publish("ml.features.EURUSD", time_ns()):
    publisher.publish(topic, payload)
```

### 12. Topic Filters (`topic_filters.py`)

**Purpose**: MQTT-style wildcard pattern matching for topic-based message routing.

**Pattern Semantics**:

- `*`: Matches exactly one token (dot-separated)
- `#`: Matches zero or more tokens (only valid as complete token)
- Literal tokens must match exactly (case-sensitive)

**Examples**:

```python
from ml.common.topic_filters import match_topic

match_topic("ml.features.*", "ml.features.EURUSD")         # True
match_topic("events.ml.#", "events.ml.FEATURE_COMPUTED")   # True
match_topic("ml.*.updated", "ml.models.updated")           # True
```

**Integration**: Used by `InMemoryPublisher` and `events_consumer` CLI for pattern-based subscription.

### 13. Message Topics (`message_topics.py`)

**Purpose**: Centralizes ML message bus topic construction with consistent routing and safe character normalization.

**Topic Format**: `ml.{domain}.{operation}.{instrument_id}`

**Key Functions**:

- `build_topic()`: Validates and constructs canonical topic strings
- `map_stage_to_topic_segments()`: Maps pipeline stages to topic segments
- `_normalize_instrument_id()`: Sanitizes instrument IDs for topic safety

**Validation Rules**:

- Domain: lowercase letters only `[a-z]+`
- Operation: lowercase letters and underscore `[a-z_]+`
- Instrument: alphanumeric with `_.-` after normalization

## Dependencies

### Internal Dependencies

- `ml.config.events`: For Stage enum (message_topics.py only)
- `prometheus_client`: Direct import for base metric types in metrics.py only
- Self-referential: `metrics_bootstrap.py` imports from `metrics.py`
- Cross-references: `in_memory_bus.py` imports from `message_bus.py` and `topic_filters.py`

### External Dependencies

- `typing`: Protocol, runtime_checkable, TypedDict
- `hashlib`: SHA256 for correlation IDs
- `logging`: For timestamp sanitization warnings
- `re`: Pattern matching for topic validation
- `time`: Timestamp generation for health status

### Integration Points
The common module is widely imported across the ML system:

- **Actors**: Import `metrics_bootstrap` for metric creation, `protocols` for component interface
- **Stores**: Import `protocols`, `metrics`, `timestamps`, `correlation`, `message_bus`, `message_topics`
- **Registries**: Import `protocols` for component interface
- **Data Pipeline**: Import `metrics` for event recording
- **Monitoring**: Import `metrics_bootstrap` for collector metrics

## Usage Patterns

### 1. Component Standardization Pattern

```python
from ml.common.protocols import MLComponentMixin

class MyMLComponent(MLComponentMixin):
    def get_performance_metrics(self) -> dict[str, float]:
        return {"inference_count": self._inference_count}
```

### 2. Safe Metrics Creation Pattern

```python
from ml.common.metrics_bootstrap import get_counter, get_histogram

class MyActor:
    def __init__(self):
        self._prediction_counter = get_counter(
            "ml_predictions_total", "Total predictions made"
        )
        self._latency_hist = get_histogram(
            "ml_inference_duration_seconds", "Inference latency"
        )
```

### 3. Event Pipeline Pattern

```python
from ml.common.metrics import record_pipeline_event

record_pipeline_event(
    dataset_type="features",
    component="technical_indicators",
    stage="FEATURE_COMPUTED",
    status="success",
    count=batch_size
)
```

### 4. Timestamp Normalization Pattern

```python
from ml.common.timestamps import sanitize_timestamp_ns

normalized_ts = sanitize_timestamp_ns(
    timestamp_value,
    mode="warn",
    logger=logger,
    context="feature_ingestion"
)
```

### 5. Event Publishing with Rate Limiting Pattern

```python
from ml.common.throttler import Throttler
from ml.common.message_bus import MessagePublisherProtocol

class RateLimitedPublisher:
    def __init__(self, publisher: MessagePublisherProtocol):
        self._publisher = publisher
        self._throttler = Throttler(rate_per_sec=10.0, burst=5)

    def publish_if_allowed(self, topic: str, payload: dict) -> bool:
        if self._throttler.should_publish(topic, time_ns()):
            return self._publisher.publish(topic, payload)
        return False
```

### 6. Topic Pattern Subscription Pattern

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.topic_filters import match_topic

# Testing pattern with wildcard subscriptions
publisher = InMemoryPublisher()

# Subscribe to all feature events
publisher.subscribe("ml.features.*", handle_feature_event)

# Subscribe to all ML events
publisher.subscribe("ml.#", handle_all_ml_events)

# Subscribe to specific instrument predictions
publisher.subscribe("ml.predictions.*.EURUSD", handle_eurusd_predictions)
```

### 7. Safe Metrics Export Pattern

```python
from ml.common.metrics_export import generate_latest, CONTENT_TYPE_LATEST

# HTTP endpoint for metrics scraping
def metrics_endpoint():
    return {
        'content': generate_latest(),
        'content_type': CONTENT_TYPE_LATEST
    }
```

## Integration Points

### With Nautilus Trader Core

- **Precision Integration**: `precision.py` ensures compatibility with Nautilus Price/Quantity constraints
- **Component Protocol**: Provides standardized interface that aligns with Nautilus component patterns

### With ML Stores

- **Base Integration**: All stores inherit from `MLComponentMixin`
- **Timestamp Handling**: Consistent nanosecond timestamps across all stores
- **Event Correlation**: Tracking data lineage through correlation IDs
- **Message Bus**: Event publication for external systems

### With ML Actors

- **Metrics Bootstrap**: Centralized metric creation preventing registry conflicts
- **Component Health**: Standardized health reporting for monitoring
- **Performance Tracking**: Consistent latency and throughput metrics

### With ML Monitoring

- **Centralized Metrics**: Single source of truth for all system metrics
- **Health Reporting**: Standardized component health interface
- **Event Tracing**: Correlation-based observability

## Implementation Notes

### Performance Considerations

- **Hot Path Safety**: All utilities designed to avoid heavy computation in inference paths
- **Pre-allocated Patterns**: Metrics are created once and reused
- **Lazy Imports**: Some modules use local imports to avoid circular dependencies

### Error Handling

- **Graceful Degradation**: `NoopPublisher` provides safe defaults
- **Validation with Recovery**: Timestamp sanitization can normalize or reject based on policy
- **Type Safety**: Protocol-based design ensures compile-time type checking

### Testing Strategy

- **Protocol Compliance**: Tests verify protocol implementation across components
- **Idempotency**: Metrics bootstrap tested for repeated calls
- **Edge Cases**: Timestamp normalization covers all magnitude ranges
- **Topic Safety**: Message topic validation tested with various instrument ID formats

### Configuration Management

- **Environment Variables**: Timestamp normalization mode configurable via `ML_TS_NORMALIZATION_MODE`
- **Policy-Driven**: Components can specify behavior via parameters rather than hard-coding

### Memory Management

- **Singleton Pattern**: Metrics registry prevents duplicate allocations
- **Minimal State**: Most components are stateless utilities
- **Efficient Hashing**: Correlation IDs use SHA256 for deterministic results

This common module serves as the backbone of the ML system's consistency and observability, providing the foundational utilities that enable all other components to integrate seamlessly with Nautilus Trader's architecture while maintaining high performance and type safety standards.
