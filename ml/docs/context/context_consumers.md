# Context: Consumers Module

## Overview

The ML consumers module provides robust, production-ready consumer patterns for the event-driven ML pipeline. It implements idempotent processing, watermark-based ordering guarantees, and safe Redis streams integration. The module ensures duplicate detection, monotonic processing order, and reliable event handling across distributed components in the ML system.

**Key Components:**

- **IdempotentConsumer**: In-memory duplicate detection and watermark management
- **RedisStreamsConsumer**: Production Redis streams integration with gating
- **Integration Patterns**: Safe topic filtering and handler management

## Architecture

### Module Structure

```
ml/consumers/
├── idempotent.py              # Idempotent consumer with correlation and watermark gating
└── redis_streams_consumer.py  # Redis streams consumer with built-in idempotency
```

### Design Principles

1. **Idempotent Processing**: Every event processed exactly once via correlation ID tracking
2. **Watermark Ordering**: Monotonic timestamp progression per consumer key
3. **Graceful Degradation**: Safe fallbacks when dependencies unavailable
4. **Production Ready**: Error handling, logging, and resilient processing loops

## Key Components

### 1. IdempotentConsumer (`idempotent.py`)

**Purpose**: Template implementation for duplicate detection and watermark-based ordering guarantees.

**Core Features**:

- **Duplicate Detection**: Tracks processed correlation IDs to prevent reprocessing
- **Watermark Management**: Ensures non-decreasing timestamp progression per key
- **Consumer Key**: Groups events by `(dataset_id, instrument_id, source)` tuple
- **Exception Safety**: Robust error handling with safe defaults

**Consumer Key Structure**:

```python
ConsumerKey = tuple[str, str, str]  # (dataset_id, instrument_id, source)
```

**State Management**:

```python
@dataclass
class IdempotentConsumer:
    seen: set[str] = field(default_factory=set)           # correlation IDs
    watermarks: MutableMapping[ConsumerKey, int] = field(default_factory=dict)  # timestamps
```

**Processing Logic**:

```python
def should_process(self, payload: Mapping[str, Any]) -> bool:
    # 1. Extract correlation_id from metadata
    # 2. Check for duplicate processing
    # 3. Apply watermark ordering per consumer key
    # 4. Return True only if both checks pass

def process(self, payload: Mapping[str, Any]) -> bool:
    # 1. Validate event should be processed
    # 2. Update seen correlation IDs
    # 3. Update watermark for consumer key
    # 4. Return True if event was accepted
```

**Required Event Structure**:

Events must contain these fields for proper gating:

- `metadata.correlation_id: str` — Unique event correlation for deduplication
- `dataset_id: str` — Dataset type (features, predictions, signals)
- `instrument_id: str` — Instrument identifier
- `source: str` — Data source (historical, live, backfill)
- `ts_max: int` — Watermark timestamp in nanoseconds (must be non-decreasing)

### 2. RedisStreamsConsumer (`redis_streams_consumer.py`)

**Purpose**: Production Redis streams integration with built-in idempotent gating and resilient processing.

**Core Features**:

- **Redis Streams Integration**: Reads from Redis streams with configurable batch sizes and blocking
- **Built-in Idempotency**: Integrates IdempotentConsumer for automatic duplicate detection
- **Resilient Processing**: Continues processing despite handler exceptions
- **Safe Fallbacks**: Graceful handling when Redis client unavailable
- **JSON Payload Handling**: Automatic JSON parsing with fallback to empty dict

**Constructor Parameters**:

```python
def __init__(
    self,
    *,
    url: str,                              # Redis connection URL
    stream: str,                           # Stream name to consume from
    handler: OnEvent,                      # Event handler callback
    gate: IdempotentConsumer | None = None # Optional custom gating logic
) -> None
```

**Event Handler Interface**:

```python
OnEvent = Callable[[str, dict[str, Any]], None]

def handler(topic: str, payload: dict[str, Any]) -> None:
    # Process accepted event
    # topic: Message topic string
    # payload: Event payload dictionary
```

**Processing Method**:

```python
def poll_once(
    self,
    *,
    count: int = 100,      # Max messages per batch
    block_ms: int = 0,     # XREAD blocking timeout (0 = non-blocking)
    last_id: str = "$"     # Stream position ("$" = latest)
) -> int:                  # Returns number of processed events
```

**Redis Stream Message Format**:

The consumer expects Redis stream entries with these fields:

- `topic`: Message topic string for routing
- `payload`: JSON-encoded event payload

**Error Handling Strategy**:

- **Connection Errors**: Returns 0 processed events when Redis unavailable
- **JSON Parse Errors**: Uses empty dict as fallback payload
- **Handler Exceptions**: Logs errors and continues processing remaining events
- **Stream Read Errors**: Returns 0 and continues (suitable for retry loops)

## Wildcard Topic Filters

Consumers often subscribe with wildcards. The helper `ml.common.topic_filters.match_topic(pattern, topic)` uses dot-separated tokens:

- `*` matches exactly one token
- `#` matches zero or more tokens

Examples:

- `ml.features.updated.*.*` matches `ml.features.updated.EURUSD.SIM`
- `events.ml.FEATURE_COMPUTED.#` matches `events.ml.FEATURE_COMPUTED` and `events.ml.FEATURE_COMPUTED.EURUSD.SIM`

## In-Memory Pub/Sub (Testing)

`ml.common.in_memory_bus.InMemoryPublisher` provides a test-friendly pub/sub:

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_stage_topic
from ml.consumers.idempotent import IdempotentConsumer

bus = InMemoryPublisher()
consumer = IdempotentConsumer()

def handler(topic: str, payload: dict[str, object]) -> None:
    if consumer.process(payload):
        ...  # process

bus.subscribe("events.ml.FEATURE_COMPUTED.#", handler)
topic = build_stage_topic("FEATURE_COMPUTED", "EURUSD.SIM")
bus.publish(topic, {"dataset_id": "features", "instrument_id": "EURUSD.SIM", "source": "historical", "ts_max": 100, "metadata": {"correlation_id": "CID-1"}})
```

## Redis Streams Consumer (Example)

`ml.consumers.redis_streams_consumer.RedisStreamsConsumer` demonstrates consuming from Redis Streams with built-in idempotent gating. Publisher fields are `topic` and `payload` JSON (as emitted by the Redis publisher adapter).

```python
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer

def handler(topic: str, payload: dict[str, object]) -> None:
    # Process accepted event
    pass

consumer = RedisStreamsConsumer(url="redis://localhost:6379/0", stream="ml-events", handler=handler)
consumer.poll_once(count=100, block_ms=0)
```

Notes:

- This example logs handler exceptions and continues.
- For production use, consider batching, retries, DLQ, and metrics.

## Usage Patterns

### 1. Basic Idempotent Consumer Pattern

```python
from ml.consumers.idempotent import IdempotentConsumer

# Create consumer with in-memory state
consumer = IdempotentConsumer()

def process_event(payload: dict[str, Any]) -> None:
    if consumer.process(payload):
        # Event accepted - process it
        dataset_id = payload["dataset_id"]
        instrument_id = payload["instrument_id"]
        print(f"Processing {dataset_id} for {instrument_id}")
    else:
        # Event rejected (duplicate or out-of-order)
        print("Event rejected by idempotent consumer")

# Example event
event = {
    "dataset_id": "features",
    "instrument_id": "EURUSD.SIM",
    "source": "historical",
    "ts_max": 1640995200000000000,  # nanoseconds
    "metadata": {"correlation_id": "unique-event-123"},
    "data": {"rsi": 65.4, "macd": 0.002}
}

process_event(event)  # First time: processed
process_event(event)  # Second time: rejected (duplicate)
```

### 2. Redis Streams Consumer Pattern

```python
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.common.topic_filters import match_topic

def handle_feature_event(topic: str, payload: dict[str, Any]) -> None:
    """Handler for feature computation events."""
    if match_topic("events.ml.FEATURE_COMPUTED.*", topic):
        instrument = payload.get("instrument_id", "unknown")
        print(f"Features computed for {instrument}")

def handle_prediction_event(topic: str, payload: dict[str, Any]) -> None:
    """Handler for prediction events."""
    if match_topic("events.ml.PREDICTION_EMITTED.*", topic):
        confidence = payload.get("confidence", 0.0)
        print(f"Prediction emitted with confidence {confidence}")

# Create consumer with custom handler
consumer = RedisStreamsConsumer(
    url="redis://localhost:6379/0",
    stream="ml-events",
    handler=handle_feature_event
)

# Poll for events (non-blocking)
processed = consumer.poll_once(count=50, block_ms=0)
print(f"Processed {processed} events")

### 3. Aggregation → Lineage → DLQ (Templates)

The following templates are provided for common cross-domain flows:

- `ml.consumers.aggregator.AggregatingConsumer`: buffers envelopes per instrument, enforces monotonic order under a watermark, and forwards flushed envelopes downstream. Idempotent by event id.
- `ml.consumers.lineage_writer.LineageWriter`: writes correlation/lineage rows from envelopes to `ObservabilityService` (idempotent by event id).
- `ml.consumers.retry.RetriableConsumer`: wraps a handler with bounded synchronous retries and publishes to DLQ on final failure (`dlq.{stage}`).

Example (in-memory):

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.lineage_writer import LineageWriter

svc = ObservabilityService()
writer = LineageWriter(service=svc)
bus = InMemoryPublisher()
bus.subscribe("aggregated.#", lambda t, p: writer.handle(t, p))

agg = AggregatingConsumer(downstream=bus, topic_mapper=lambda _stage: "aggregated.lineage")

envelope = {
  "id": "e1",
  "parent_id": None,
  "instrument_id": "EURUSD.SIM",
  "ts_event": 100,
  "stage": "FEATURE_COMPUTED",
  "correlation_id": "c1",
  "payload": {"x": 1}
}

agg.handle("events.ml.FEATURE_COMPUTED", envelope)
agg.advance_watermark("EURUSD.SIM", watermark_ns=200)  # flushes to writer
```

For DLQ/retry:

```python
from ml.consumers.retry import RetriableConsumer

dlq_bus = InMemoryPublisher()
rc = RetriableConsumer(handler=my_handler, dlq=dlq_bus)
rc.handle(topic, envelope)
```

Metrics

- `nautilus_ml_aggregator_buffer_size{instrument}` – buffered envelopes per instrument
- `nautilus_ml_aggregator_flushed_total{instrument}` – count of envelopes flushed
- `nautilus_ml_aggregator_duplicates_total` – duplicates dropped
- `nautilus_ml_aggregator_watermark_lag_seconds{instrument}` – (watermark_ns - last_flushed_ts) in seconds

Monitoring

- Dashboard (Consumers / Aggregator row):
  - Buffer Size: `nautilus_ml_aggregator_buffer_size`
  - Flushed Rate: `sum by (instrument)(rate(nautilus_ml_aggregator_flushed_total[$interval]))`
  - Duplicates Rate: `rate(nautilus_ml_aggregator_duplicates_total[$interval])`
  - Watermark Lag (max): `max(nautilus_ml_aggregator_watermark_lag_seconds)`
- Alerts:
  - `MLAggregatorDuplicatesHigh`: sustained duplicates > 1/s
  - `MLAggregatorBufferHigh`: max buffer > 5k for 10m
  - `MLAggregatorWatermarkLagHigh`: max lag > 5m for 10m

# Poll with blocking (wait up to 5 seconds)
processed = consumer.poll_once(count=100, block_ms=5000)

```

### 3. Multi-Pattern Consumer with Routing

```python
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.common.topic_filters import match_topic

class EventRouter:
    def __init__(self):
        self.feature_count = 0
        self.prediction_count = 0
        self.signal_count = 0

    def route_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Route events to appropriate handlers based on topic patterns."""
        if match_topic("events.ml.FEATURE_COMPUTED.*", topic):
            self._handle_feature(topic, payload)
        elif match_topic("events.ml.PREDICTION_EMITTED.*", topic):
            self._handle_prediction(topic, payload)
        elif match_topic("events.ml.SIGNAL_EMITTED.*", topic):
            self._handle_signal(topic, payload)
        else:
            print(f"Unhandled topic: {topic}")

    def _handle_feature(self, topic: str, payload: dict[str, Any]) -> None:
        self.feature_count += 1
        instrument = payload.get("instrument_id", "unknown")
        feature_count = len(payload.get("features", {}))
        print(f"Features: {feature_count} computed for {instrument}")

    def _handle_prediction(self, topic: str, payload: dict[str, Any]) -> None:
        self.prediction_count += 1
        prediction = payload.get("prediction", 0.0)
        confidence = payload.get("confidence", 0.0)
        print(f"Prediction: {prediction} (confidence: {confidence})")

    def _handle_signal(self, topic: str, payload: dict[str, Any]) -> None:
        self.signal_count += 1
        signal_type = payload.get("signal_type", "unknown")
        strength = payload.get("strength", 0.0)
        print(f"Signal: {signal_type} with strength {strength}")

# Setup router-based consumer
router = EventRouter()
consumer = RedisStreamsConsumer(
    url="redis://localhost:6379/0",
    stream="ml-events",
    handler=router.route_event
)

# Process events in loop
while True:
    processed = consumer.poll_once(count=100, block_ms=1000)
    if processed == 0:
        print("No new events")
        break
    print(f"Processed {processed} events")
    print(f"Counts - Features: {router.feature_count}, "
          f"Predictions: {router.prediction_count}, "
          f"Signals: {router.signal_count}")
```

### 4. Testing Pattern with In-Memory Bus

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_topic
from ml.consumers.idempotent import IdempotentConsumer

# Setup test environment
bus = InMemoryPublisher()
consumer = IdempotentConsumer()
processed_events = []

def test_handler(topic: str, payload: dict[str, Any]) -> None:
    """Handler that only processes accepted events."""
    if consumer.process(payload):
        processed_events.append((topic, payload))
        print(f"Accepted event: {topic}")

# Subscribe to ML events with wildcard
bus.subscribe("events.ml.#", test_handler)

# Publish test events
event_base = {
    "dataset_id": "features",
    "instrument_id": "EURUSD.SIM",
    "source": "test",
    "ts_max": 1640995200000000000,
    "data": {"rsi": 50.0}
}

# First event (should be accepted)
event1 = {**event_base, "metadata": {"correlation_id": "test-1"}}
topic1 = build_topic("FEATURE_COMPUTED", "EURUSD.SIM")
bus.publish(topic1, event1)

# Duplicate event (should be rejected)
bus.publish(topic1, event1)

# Out-of-order event (should be rejected)
event2 = {**event_base, "ts_max": event_base["ts_max"] - 1000,
          "metadata": {"correlation_id": "test-2"}}
bus.publish(topic1, event2)

# Valid newer event (should be accepted)
event3 = {**event_base, "ts_max": event_base["ts_max"] + 1000,
          "metadata": {"correlation_id": "test-3"}}
bus.publish(topic1, event3)

print(f"Total processed: {len(processed_events)}")  # Should be 2
```

### 5. Production Consumer Loop Pattern

```python
import time
import logging
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer

def production_consumer_loop():
    """Production-ready consumer loop with error handling and metrics."""
    logger = logging.getLogger(__name__)

    def robust_handler(topic: str, payload: dict[str, Any]) -> None:
        try:
            # Your business logic here
            process_business_event(topic, payload)
        except Exception as e:
            logger.error(f"Handler error for topic {topic}: {e}")
            # Could emit metrics here

    consumer = RedisStreamsConsumer(
        url="redis://localhost:6379/0",
        stream="ml-events",
        handler=robust_handler
    )

    consecutive_empty = 0
    max_empty = 5

    while True:
        try:
            processed = consumer.poll_once(count=100, block_ms=1000)

            if processed > 0:
                consecutive_empty = 0
                logger.info(f"Processed {processed} events")
            else:
                consecutive_empty += 1
                if consecutive_empty >= max_empty:
                    logger.info("No events for extended period, continuing...")
                    consecutive_empty = 0

        except KeyboardInterrupt:
            logger.info("Consumer shutdown requested")
            break
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
            time.sleep(5)  # Backoff on error

def process_business_event(topic: str, payload: dict[str, Any]) -> None:
    """Your business logic implementation."""
    pass
```

## Dependencies

### Internal Dependencies

```python
# Core consumer functionality
from ml.consumers.idempotent import IdempotentConsumer, ConsumerKey
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer, OnEvent

# Topic handling and filtering
from ml.common.topic_filters import match_topic
from ml.common.message_topics import build_topic, build_stage_topic
from ml.common.in_memory_bus import InMemoryPublisher
```

### External Dependencies

```python
# Standard library
import json, logging
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Final

# Optional dependencies
import redis  # For RedisStreamsConsumer (graceful fallback when unavailable)
```

## Integration Points

### With ML Pipeline Events

All consumer patterns integrate with the standard ML pipeline event structure:

```python
# Standard event structure expected by consumers
{
    "dataset_id": "features|predictions|signals",
    "instrument_id": "EURUSD.SIM",
    "source": "historical|live|backfill",
    "ts_max": 1640995200000000000,  # nanosecond watermark
    "metadata": {
        "correlation_id": "unique-event-id",
        "run_id": "optional-batch-id"
    },
    "data": {
        # Event-specific payload
    }
}
```

### With Message Bus Systems

- **Redis Streams**: Production event streaming via `RedisStreamsConsumer`
- **In-Memory Bus**: Testing and development via `InMemoryPublisher`
- **Topic Routing**: Wildcard pattern matching for flexible subscription
- **CLI Integration**: Used by `events_consumer` CLI for debugging and monitoring

### With ML Stores and Registries

- **Event Correlation**: Correlation IDs link events across stores and pipeline stages
- **Watermark Synchronization**: Ensures monotonic processing aligned with store watermarks
- **Idempotency**: Prevents duplicate processing during store recomputation or backfill

## Implementation Notes

### Performance Considerations

- **Memory Management**: IdempotentConsumer uses in-memory sets/dicts (consider persistence for large-scale)
- **Batch Processing**: Redis consumer processes events in configurable batches
- **Non-blocking Design**: Suitable for event loops and async integration

### Error Handling

- **Graceful Degradation**: Continues processing when individual events fail
- **Connection Resilience**: Safe fallbacks when Redis unavailable
- **Logging Strategy**: Detailed error logging without breaking consumer loops

### Testing Strategy

- **In-Memory Testing**: Use `InMemoryPublisher` for unit tests
- **Mock Integration**: Consumer patterns support dependency injection
- **Event Simulation**: Easy to create test events with required structure

### Production Deployment

- **Redis Configuration**: Configure Redis streams with appropriate retention
- **Consumer Groups**: Consider Redis consumer groups for distributed processing
- **Monitoring**: Integrate with metrics collection for operational visibility
- **Scaling**: Multiple consumer instances can process different topic patterns

This consumer module provides the foundation for reliable, ordered event processing in the ML pipeline, ensuring data consistency and preventing duplicate processing across all pipeline stages.
