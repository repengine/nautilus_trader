# Context: Events & Message Bus System

## Overview

The ML events and message bus system provides a comprehensive event-driven architecture for the ML pipeline, enabling real-time communication between components, actors, stores, and external systems. The system supports multiple topic schemes, configurable backends (Redis streams, in-memory), throttling, and actor-side non-blocking event publishing with production-ready reliability patterns.

**Key Features:**

- **Dual Topic Schemes**: Canonical domain-operation and stage-first routing
- **Multiple Backends**: Redis streams for production, in-memory for testing
- **Actor Integration**: Non-blocking bridge with background flushing and backpressure handling
- **Throttling**: Token-bucket rate limiting to prevent external system flooding
- **Idempotent Processing**: Consumer templates with correlation ID deduplication
- **Production Ready**: Environment-driven configuration, metrics integration, graceful degradation

## Architecture

### System Components

```
ml/config/
├── events.py          # Event constants (Stage, Source, EventStatus enums)
├── bus.py            # Message bus configuration and environment parsing
└── actor_bus.py      # Actor-side bus configuration and throttling

ml/common/
├── message_bus.py    # Publisher protocols and factory functions
├── message_topics.py # Topic building and normalization utilities
├── in_memory_bus.py  # In-memory pub/sub for testing
└── throttler.py      # Token-bucket rate limiting

ml/actors/
└── ml_domain_events.py  # Non-blocking actor-side event bridge

ml/core/
└── bus_integration.py   # Integration manager bus attachment helpers

ml/consumers/
├── idempotent.py           # Idempotent consumer with watermark gating
└── redis_streams_consumer.py  # Production Redis streams consumer
```

### Event Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Actors     │───▶│ DomainEventBridge│───▶│ Message Bus     │
│   (Hot Path)    │    │ (Non-blocking)   │    │ (Redis/Memory)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Throttling &     │    │    Consumers    │
                       │ Backpressure     │    │ (Idempotent)    │
                       └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Event Constants (`ml.config.events`)

**Purpose**: Standardized enums for pipeline stages, sources, and status values.

**Core Enums**:

```python
class Stage(str, Enum):
    DATA_INGESTED = "INGESTED"
    CATALOG_WRITTEN = "CATALOG_WRITTEN"
    FEATURE_COMPUTED = "FEATURE_COMPUTED"
    PREDICTION_EMITTED = "PREDICTION_EMITTED"
    SIGNAL_EMITTED = "SIGNAL_EMITTED"

class Source(str, Enum):
    LIVE = "live"
    HISTORICAL = "historical"
    BACKFILL = "backfill"

class EventStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
```

**Usage**: Events emitters serialize `.value` to payloads and databases. Registry operations use `status=success` and place lifecycle hints in metadata (e.g., `{"deprecated": true}`).

### 2. Message Bus Configuration (`ml.config.bus`)

**Purpose**: Environment-driven configuration for message bus backends, schemes, and Redis integration.

**Core Configuration**:

```python
@dataclass(frozen=True)
class MessageBusConfig:
    enabled: bool = False                    # Disabled by default for hot-path safety
    backend: BusBackend = "noop"             # "noop" | "redis"
    scheme: TopicScheme = "domain_op"        # "domain_op" | "stage_first"
    topic_prefix: str = "events.ml"          # Stage-first prefix
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ml-events"
    redis_maxlen: int | None = None          # Optional stream size limit
```

**Environment Variables**:

- `ML_BUS_ENABLE`: `1|true|yes` to enable (default: disabled)
- `ML_BUS_BACKEND`: `noop|redis` (default: `noop`)
- `ML_BUS_SCHEME`: `domain_op|stage_first` (default: `domain_op`)
- `ML_BUS_TOPIC_PREFIX`: stage-first prefix (default: `events.ml`)
- `ML_BUS_REDIS_URL`: Redis URL (default: `redis://localhost:6379/0`)
- `ML_BUS_REDIS_STREAM`: Stream name (default: `ml-events`)
- `ML_BUS_REDIS_MAXLEN`: Approximate stream max length (optional)

### 3. Actor Bus Configuration (`ml.config.actor_bus`)

**Purpose**: Actor-side publishing configuration with throttling and path selection.

**Core Configuration**:

```python
@dataclass(frozen=True)
class ActorBusConfig:
    from_actor: bool                    # Publish from actor thread via bridge
    from_store: bool                    # Publish from store operations
    scheme: TopicScheme                 # Topic naming scheme
    prefix: str                         # Topic prefix for stage-first
    throttle_enabled: bool              # Enable token-bucket throttling
    throttle_rate_per_sec: float        # Rate limit per topic
    throttle_burst: int                 # Burst allowance per topic
```

**Additional Environment Variables**:

- `ML_BUS_FROM_ACTOR`: Publish from actor thread using DomainEventBridge (default: off)
- `ML_BUS_FROM_STORE`: Publish from store path (default: off)
- `ML_BUS_THROTTLE_ENABLE`: Enable publish throttling (default: off)
- `ML_BUS_THROTTLE_RATE`: Tokens per second per topic (default: 100.0)
- `ML_BUS_THROTTLE_BURST`: Burst tokens per topic (default: 100)

**Path Resolution**: When both `from_actor` and `from_store` are enabled, actor path is preferred to avoid store hot-path I/O.

### 4. Topic Schemes

**Canonical Domain-Operation Scheme**: `ml.{domain}.{operation}.{instrument_id}`

- Built with `ml.common.message_topics.build_topic(domain, operation, instrument_id)`
- Stage → (domain, operation) mapping via `map_stage_to_topic_segments(Stage)`
- Examples:
  - `ml.features.computed.EURUSD.SIM`
  - `ml.predictions.emitted.BTCUSDT.BINANCE`

**Stage-First Scheme**: `{prefix}.{STAGE}[.{instrument_id}]`

- Built with `ml.common.message_topics.build_stage_topic(Stage, instrument_id=None, prefix='events.ml')`
- Examples:
  - `events.ml.FEATURE_COMPUTED.EURUSD.SIM`
  - `events.ml.PREDICTION_EMITTED`

**Dynamic Topic Selection**:

```python
topic = build_topic_for_stage(
    stage="FEATURE_COMPUTED",
    instrument_id="EURUSD.SIM",
    scheme="domain_op",  # or "stage_first"
    prefix="events.ml"
)
```

**Instrument ID Normalization**: All builders normalize `instrument_id` to safe characters: `A-Za-z0-9_.-` (reserved characters `/*#+$` are replaced with `.`).

### 5. Domain Event Bridge (`ml.actors.ml_domain_events`)

**Purpose**: Non-blocking actor-side event publishing with background flushing and backpressure handling.

**Core Features**:

- **O(1) Enqueue**: Actor thread performs only queue insertion
- **Background Flusher**: Separate worker thread drains queue and publishes
- **Backpressure Handling**: Drops events when queue full, records metrics
- **Optional Throttling**: Token-bucket rate limiting per topic
- **Graceful Shutdown**: Configurable drain-on-stop behavior

**Usage Pattern**:

```python
from ml.actors.ml_domain_events import DomainEventBridge
from ml.common.throttler import Throttler

# Setup with throttling
throttler = Throttler(rate_per_sec=100.0, burst=100)
bridge = DomainEventBridge(
    publisher=publisher,
    max_queue=4096,
    throttler=throttler,
    component_id="ml_signal_actor"
)

# Lifecycle
bridge.start()
success = bridge.publish(topic, payload)  # Returns bool for backpressure detection
bridge.stop(drain=True)  # Drain remaining events before shutdown
```

**Metrics Integration**:

- `backpressure_drops_total`: Counter for dropped events
- `backpressure_queue_depth`: Gauge for current queue size

### 6. Bus Integration (`ml.core.bus_integration`)

**Purpose**: Helper functions to attach message publishers to MLIntegrationManager.

**Key Function**:

```python
def attach_publisher_from_env(manager: MLIntegrationManager) -> None:
    """Attach publisher based on environment configuration."""
    cfg = MessageBusConfig.from_env()
    manager.set_message_publisher(publisher_from_config(cfg))
```

**Integration**: Safe to call regardless of whether publishing is enabled; attaches NoopPublisher when disabled.

## Usage Patterns

### 1. Basic Event Publishing Setup

```python
from ml.core.bus_integration import attach_publisher_from_env
from ml.core.integration import MLIntegrationManager

# Setup integration manager with message publishing
manager = MLIntegrationManager()
attach_publisher_from_env(manager)  # Configures based on environment

# Manager now has publisher attached for store-level events
```

### 2. Actor-Side Event Publishing

```python
from ml.actors.ml_domain_events import DomainEventBridge
from ml.common.message_bus import publisher_from_config
from ml.config.bus import MessageBusConfig
from ml.config.actor_bus import ActorBusConfig

# Setup actor-side configuration
actor_config = ActorBusConfig.from_env()
bus_config = MessageBusConfig.from_env()

if actor_config.from_actor:
    # Create publisher and bridge
    publisher = publisher_from_config(bus_config)

    # Optional throttling
    throttler = None
    if actor_config.throttle_enabled:
        from ml.common.throttler import Throttler
        throttler = Throttler(
            rate_per_sec=actor_config.throttle_rate_per_sec,
            burst=actor_config.throttle_burst
        )

    # Setup non-blocking bridge
    bridge = DomainEventBridge(
        publisher=publisher,
        max_queue=4096,
        throttler=throttler,
        component_id="my_ml_actor"
    )

    # Actor lifecycle
    bridge.start()

    # Publish events (non-blocking)
    success = bridge.publish("ml.predictions.emitted.EURUSD", {
        "dataset_id": "predictions",
        "instrument_id": "EURUSD",
        "prediction": 0.65,
        "confidence": 0.82,
        "metadata": {"correlation_id": "pred-123"}
    })

    # Graceful shutdown
    bridge.stop(drain=True)
```

### 3. Topic Building Patterns

```python
from ml.common.message_topics import (
    build_topic, build_stage_topic, build_topic_for_stage,
    map_stage_to_topic_segments
)
from ml.config.events import Stage

# Domain-operation scheme
domain, operation = map_stage_to_topic_segments(Stage.FEATURE_COMPUTED)
topic1 = build_topic(domain, operation, "EURUSD.SIM")
# Result: "ml.features.computed.EURUSD.SIM"

# Stage-first scheme
topic2 = build_stage_topic(Stage.PREDICTION_EMITTED, "BTCUSDT.BINANCE")
# Result: "events.ml.PREDICTION_EMITTED.BTCUSDT.BINANCE"

# Dynamic scheme selection
topic3 = build_topic_for_stage(
    stage=Stage.SIGNAL_EMITTED,
    instrument_id="GBPUSD.SIM",
    scheme="domain_op"
)
# Result: "ml.signals.emitted.GBPUSD.SIM"
```

### 3. End-to-End: Ingestion → Aggregation → Lineage (Fixture-Based)

For provider-agnostic testing and local development, you can simulate an end-to-end flow using deterministic fixtures and in-memory components:

1) Generate envelopes from a fixture (e.g., TBBO):

```python
import pandas as pd
from ml.data.fixtures import make_tbbo_fixture

df, man = make_tbbo_fixture(instrument_id="EURUSD.SIM", rows=10)
```

2) Build envelopes and send through aggregator:

```python
from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.lineage_writer import LineageWriter
from ml.common.in_memory_bus import InMemoryPublisher
from ml.observability.service import ObservabilityService

svc = ObservabilityService()
writer = LineageWriter(service=svc)
bus = InMemoryPublisher()
bus.subscribe("aggregated.#", lambda t, p: writer.handle(t, p))
agg = AggregatingConsumer(downstream=bus, topic_mapper=lambda _stage: "aggregated.lineage")

for i, row in enumerate(df.itertuples(index=False), start=1):
    env = {
        "id": f"e{i}",
        "parent_id": None,
        "instrument_id": row.instrument_id,
        "ts_event": int(row.ts_event),
        "stage": "FEATURE_COMPUTED",
        "correlation_id": "c1",
        "payload": {"bid_px": float(row.bid_px), "ask_px": float(row.ask_px)},
    }
    agg.handle("events.ml.FEATURE_COMPUTED", env)

wm = int(df["ts_event"].max())
agg.advance_watermark("EURUSD.SIM", wm)

# Lineage DataFrame contains event_id and ts_event in order
lineage_df = svc.event_correlation_df()
```

3) Record ingestion metrics for dashboards/alerts:

```python
from ml.data.ingest.metrics import record_ingest_batch
record_ingest_batch(
    dataset="tbbo", instrument="EURUSD.SIM", source="historical",
    duration_seconds=0.005, ts_min=int(df.ts_event.min()), ts_max=int(df.ts_event.max())
)
```

Related tests:

- `ml/tests/integration/test_ingest_aggregate_lineage_integration.py`
- `ml/tests/property/test_ingestion_watermark_properties.py`
- `ml/tests/contracts/test_databento_fixtures_contracts.py`

### 4. Consumer Pattern with Wildcard Matching

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.topic_filters import match_topic
from ml.consumers.idempotent import IdempotentConsumer

# Setup
bus = InMemoryPublisher()
consumer = IdempotentConsumer()

def event_handler(topic: str, payload: dict) -> None:
    # Apply idempotent gating
    if consumer.process(payload):
        # Process accepted event
        print(f"Processing: {topic} -> {payload['dataset_id']}")

# Subscribe with wildcard patterns
bus.subscribe("events.ml.FEATURE_COMPUTED.#", event_handler)
bus.subscribe("ml.predictions.*", event_handler)
bus.subscribe("ml.signals.emitted.*", event_handler)

# Publish events
events = [
    ("events.ml.FEATURE_COMPUTED.EURUSD", {"dataset_id": "features", "instrument_id": "EURUSD", "source": "historical", "ts_max": 100, "metadata": {"correlation_id": "f1"}}),
    ("ml.predictions.emitted.BTCUSDT", {"dataset_id": "predictions", "instrument_id": "BTCUSDT", "source": "live", "ts_max": 200, "metadata": {"correlation_id": "p1"}}),
    ("ml.signals.emitted.GBPUSD", {"dataset_id": "signals", "instrument_id": "GBPUSD", "source": "live", "ts_max": 300, "metadata": {"correlation_id": "s1"}})
]

for topic, payload in events:
    bus.publish(topic, payload)
```

### 5. Production Redis Streams Integration

```python
import os
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.common.topic_filters import match_topic

# Environment setup
os.environ["ML_BUS_ENABLE"] = "true"
os.environ["ML_BUS_BACKEND"] = "redis"
os.environ["ML_BUS_REDIS_URL"] = "redis://localhost:6379/0"
os.environ["ML_BUS_REDIS_STREAM"] = "ml-events"

def production_handler(topic: str, payload: dict) -> None:
    """Production event handler with pattern routing."""
    if match_topic("events.ml.FEATURE_COMPUTED.*", topic):
        handle_feature_event(payload)
    elif match_topic("events.ml.PREDICTION_EMITTED.*", topic):
        handle_prediction_event(payload)
    elif match_topic("events.ml.SIGNAL_EMITTED.*", topic):
        handle_signal_event(payload)

def handle_feature_event(payload: dict) -> None:
    # Process feature computation event
    pass

def handle_prediction_event(payload: dict) -> None:
    # Process prediction event
    pass

def handle_signal_event(payload: dict) -> None:
    # Process trading signal event
    pass

# Setup Redis consumer
consumer = RedisStreamsConsumer(
    url="redis://localhost:6379/0",
    stream="ml-events",
    handler=production_handler
)

# Production consumption loop
while True:
    processed = consumer.poll_once(count=100, block_ms=5000)
    if processed == 0:
        print("No events processed")
```

## Integration Points

### With ML Pipeline Events

The event system integrates seamlessly with the standard ML pipeline stages:

```python
# Standard event payload structure
event = {
    "dataset_id": "features|predictions|signals",
    "instrument_id": "EURUSD.SIM",
    "source": "live|historical|backfill",
    "ts_max": 1640995200000000000,  # Nanosecond watermark
    "metadata": {
        "correlation_id": "unique-event-id",
        "run_id": "optional-batch-id",
        "status": "success|failed|partial"
    },
    "data": {
        # Stage-specific payload
    }
}
```

### With ML Stores and Registries

- **DataStore**: Automatic event emission when `set_message_publisher()` configured
- **FeatureStore**: Events for feature computation completion
- **ModelStore**: Events for prediction and model operations
- **StrategyStore**: Events for signal generation and trading decisions
- **Registry Operations**: Lifecycle events (register, promote, deprecate)

### With Observability System

- **Metrics Integration**: Bridge backpressure metrics recorded automatically
- **Event Correlation**: Correlation IDs link events across pipeline stages
- **Health Monitoring**: Publisher health included in system health checks

### With External Systems

- **Redis Streams**: Production event streaming for microservices
- **CLI Tools**: `events_consumer` CLI for debugging and monitoring
- **Monitoring Dashboards**: Topic-based routing for alerting and metrics

## Dependencies

### Internal Dependencies

```python
# Configuration
from ml.config.events import Stage, Source, EventStatus
from ml.config.bus import MessageBusConfig, BusBackend, TopicScheme
from ml.config.actor_bus import ActorBusConfig

# Core publishing
from ml.common.message_bus import MessagePublisherProtocol, publisher_from_config
from ml.common.message_topics import build_topic, build_stage_topic
from ml.common.throttler import Throttler
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.topic_filters import match_topic

# Actor integration
from ml.actors.ml_domain_events import DomainEventBridge

# Consumers
from ml.consumers.idempotent import IdempotentConsumer
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer

# Integration
from ml.core.bus_integration import attach_publisher_from_env
```

### External Dependencies

```python
# Standard library
import os, queue, threading, json
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, Protocol, Literal

# Optional dependencies
import redis  # For Redis Streams publisher/consumer (graceful fallback)
```

## Implementation Notes

### Performance Considerations

- **Hot Path Safety**: Publishing disabled by default to preserve inference performance
- **Non-blocking Actor Path**: DomainEventBridge ensures O(1) enqueue on actor threads
- **Backpressure Protection**: Queue limits prevent memory exhaustion under load
- **Throttling**: Token-bucket prevents external system flooding

### Error Handling

- **Graceful Degradation**: NoopPublisher when Redis unavailable
- **Backpressure Metrics**: Dropped events tracked for operational visibility
- **Consumer Resilience**: Continues processing despite individual handler failures
- **Connection Recovery**: Redis clients handle reconnection automatically

### Testing Strategy

- **In-Memory Testing**: `InMemoryPublisher` for unit tests and examples
- **Property-Based Testing**: Topic normalization and filter matching
- **Integration Tests**: End-to-end pub/sub flows with idempotent consumers
- **Contract Tests**: Event schema validation across producers/consumers

### Deployment Considerations

- **Environment Configuration**: All settings configurable via environment variables
- **Redis Streams**: Consider retention policies and consumer groups for scale
- **Monitoring**: Integrate with Prometheus for operational metrics
- **Security**: Redis URL should use secure connections in production

This event system provides the foundation for real-time ML pipeline coordination, enabling loose coupling between components while maintaining strong consistency guarantees through idempotent processing and correlation-based event tracking.
