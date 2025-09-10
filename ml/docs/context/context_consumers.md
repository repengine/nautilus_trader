# Consumer Examples & Patterns

This guide shows practical consumer patterns for the event-driven ML pipeline, with an emphasis on idempotency, watermarks, and safe topic filtering.

## Idempotent Consumer Template

Use `ml.consumers.idempotent.IdempotentConsumer` to gate events by correlation ID and watermark per `(dataset_id, instrument_id, source)`:

```python
from ml.consumers.idempotent import IdempotentConsumer

consumer = IdempotentConsumer()

def on_event(payload: dict[str, object]) -> bool:
    return consumer.process(payload)
```

Accepted events must carry:

- `metadata.correlation_id: str` — unique event correlation, used for deduplication
- `dataset_id, instrument_id, source: str`
- `ts_max: int` — watermark (ns); must be non-decreasing for a given key

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

## References

- Topic builders: `ml.common.message_topics`
- Wildcards: `ml.common.topic_filters`
- Idempotent gating: `ml.consumers.idempotent`
- Redis consumer: `ml.consumers.redis_streams_consumer`
- In-memory pub/sub: `ml.common.in_memory_bus`
